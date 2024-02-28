import os
from copy import deepcopy
from typing import Optional, List, Union, Tuple
import copy
import numpy as np
import wandb
from peft import PeftModel, LoraModel, load_peft_weights, set_peft_model_state_dict
from .BaseTrainerCL import BaseTrainerCL
import torch
from transformers import LlamaForCausalLM, PreTrainedModel, TrainerCallback
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
import torch.nn.functional as F


# from method.BaseTrainerCL import BaseTrainerCL


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['input_ids', 'labels', 'logits', 'task_labels', 'activations']
        self.init_buffer()

    def init_buffer(self) -> None:
        for attr_str in self.attributes:
            setattr(self, attr_str, [None for _ in range(self.buffer_size)])

    def add_data(self, input_ids, labels=None, logits=None, task_labels=None, activations=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param input_ids: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param activations: tensor containing the activations of the network
        :return:
        """
        n = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
        for i in range(n):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.input_ids[index] = input_ids[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if activations is not None:
                    self.activations[index] = activations[i].to(self.device)

    def get_data(self, size: int) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """
        n = self.input_ids.shape[0] if hasattr(self.input_ids, 'shape') else len(self.input_ids)
        if size > min(self.num_seen_examples, n):
            size = min(self.num_seen_examples, n)

        choice = np.random.choice(min(self.num_seen_examples, n), size=size, replace=False)
        if len(choice) == 0:
            return None, None
        max_input_id_len = max([self.input_ids[c].shape[0] for c in choice])
        max_label_len = max([self.labels[c].shape[0] for c in choice])
        # for left padding
        input_ids = torch.stack(
            [torch.cat([torch.zeros(max_input_id_len - ee.shape[0]).long().to(ee.device), ee]) for ee in
             [self.input_ids[c] for c in choice]]).reshape(size, max_input_id_len)
        labels = torch.stack([torch.cat([torch.zeros(max_label_len - ee.shape[0]).long().to(ee.device), ee]) for ee in
                              [self.labels[c] for c in choice]]).reshape(size, max_label_len)
        return input_ids, labels

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :return: a tuple with all the items in the memory buffer
        """
        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.input_ids]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


class ILoRAModel(LlamaForCausalLM):
    def __init__(self, model: PeftModel, reg_decay: bool = False):
        super().__init__(model.config)
        self.model = model
        self.current_task_name = "C-STANCE"
        # regularization settings
        peft_cfg = model.peft_config['default']
        self.model.add_adapter('ema', peft_cfg)
        self.model.to(self.model.device)

        self.ema_alpha: float = 0.25
        self.reg_weight: float = 1.0
        self.ema_update_freq: float = 0.1
        self.consistency_loss = nn.MSELoss(reduction='none')

        self.buffer = Buffer(500, 'cuda')  # same as ER
        self.l_cons = 0
        self.total = 0
        self.ori_loss = 0

    def update_ema_weights(self, step):
        alpha = min(1 - 1 / (step + 1), self.ema_alpha)

        self.model.set_adapter('default')
        model_state_dict = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.set_adapter('ema')
        for name, param in self.model.named_parameters():
            if name in model_state_dict.keys():
                param.data.mul_(alpha).add_(torch.mul(model_state_dict[name].data, 1 - alpha))
        self.model.set_adapter('default')

    def update_reg_weight(self, step, decay_steps):
        if self.reg_decay:
            self.alpha = self.fix_reg_weight * self.decay_rate ** (step / decay_steps)
        else:
            self.alpha = self.fix_reg_weight

    def update_ema_model(self, path):
        adapter_weights = load_peft_weights(path)
        set_peft_model_state_dict(self.model, adapter_weights, adapter_name='ema')

    def concat_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor, buffer_inputs_ids: torch.Tensor,
                      buffer_labels: torch.Tensor) -> Tuple:
        if buffer_inputs_ids is None or buffer_labels is None:
            return input_ids, labels

        max_input_id_len = max(input_ids.shape[1], buffer_inputs_ids.shape[1])
        max_labels_len = max(labels.shape[1], buffer_labels.shape[1])

        extended_input_ids = torch.cat([input_ids, torch.zeros(input_ids.shape[0],
                                                               max_input_id_len - input_ids.shape[1]).long().to(
            input_ids.device)], dim=1)
        extended_labels = torch.cat(
            [labels, torch.zeros(labels.shape[0], max_labels_len - labels.shape[1]).long().to(labels.device)], dim=1)

        extended_buffer_inputs_ids = torch.cat([buffer_inputs_ids, torch.zeros(buffer_inputs_ids.shape[0],
                                                                               max_input_id_len -
                                                                               buffer_inputs_ids.shape[1]).long().to(
            buffer_inputs_ids.device)], dim=1)
        extended_buffer_labels = torch.cat([buffer_labels, torch.zeros(buffer_labels.shape[0],
                                                                       max_labels_len - buffer_labels.shape[
                                                                           1]).long().to(buffer_labels.device)], dim=1)

        input_ids = torch.cat([extended_input_ids, extended_buffer_inputs_ids], dim=0)
        labels = torch.cat([extended_labels, extended_buffer_labels], dim=0)
        return input_ids, labels

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        buffer_inputs, buffer_labels = None, None
        if self.current_task_name != "C-STANCE":
            buffer_inputs, buffer_labels = self.buffer.get_data(input_ids.shape[0])
        l_cons = 0
        if labels is not None and buffer_inputs is not None and buffer_labels is not None:
            self.model.set_adapter('default')
            plastic_hiddn = self.model(
                buffer_inputs,
                labels=buffer_labels,
                output_hidden_states=True,
                return_dict=True)

            self.model.set_adapter('ema')
            with torch.no_grad():
                stable_hiddn = self.model(
                    buffer_inputs,
                    labels=buffer_labels,
                    output_hidden_states=True,
                    return_dict=True).hidden_states
            indexs = [inner_plastic > inner_stable for inner_plastic, inner_stable in
                      zip(plastic_hiddn.hidden_states, stable_hiddn)]
            reg_hiddn = [torch.where(idx, inner_plastic, inner_stable) for idx, inner_plastic, inner_stable in
                         zip(indexs, plastic_hiddn.hidden_states, stable_hiddn)]

            l_cons = torch.mean(
                torch.cat([self.consistency_loss(plastic, ema) for plastic, ema in
                           zip(plastic_hiddn.hidden_states, reg_hiddn)], dim=0))

            self.l_cons = l_cons  # for logging use

        self.model.set_adapter('default')
        ori_out = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=False)
        if labels is not None and buffer_inputs is not None and buffer_labels is not None:
            self.total_loss = (ori_out.loss + plastic_hiddn.loss) / 2 + self.reg_weight * l_cons
        else:
            self.total_loss = ori_out.loss + self.reg_weight * l_cons
        self.total = self.total_loss.item()

        return CausalLMOutputWithPast(
            loss=self.total_loss,
            past_key_values=ori_out.past_key_values,
            logits=ori_out.logits,
            hidden_states=ori_out.hidden_states
        )


class ILoRATrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = ILoRAModel(self.model)
        self.add_callback(CLSCallback(self.model))

    def compute_loss(self, model, inputs, return_outputs=False):
        self.model.buffer.add_data(inputs['input_ids'], inputs['labels'])
        outputs = self.model(**inputs)
        loss = outputs.loss
        print("loss:", loss.item())
        return (loss, outputs) if return_outputs else loss

    def continual_learning(self):
        resume_from_checkpoint = "False"
        for task_name, dataset in self.continual_training_dataset.items():
            self.model.current_task_name = task_name
            self.current_task_name = task_name
            self.train_dataset = dataset
            self.train()
            resume_from_checkpoint = self.save_model(task_name)
            self.model.load_ema_model(resume_from_checkpoint)
        wandb.finish()

    def save_model(self, name) -> str:
        if self.args.output_dir is not None:
            output_dir = os.path.join(self.args.output_dir, f"{self.cl_method}_{self.adapter}_checkpoint_{name}")
            self.model.model.set_adapter('default')
            self.model.model.save_pretrained(output_dir)
            return output_dir


class CLSCallback(TrainerCallback):
    def __init__(self, model: ILoRAModel):
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        self.model.update_ema_weights(state.global_step)
        if wandb.run:
            wandb.log({
                "reg_weight": self.model.reg_weight,
                "consist_loss": self.model.l_cons,
                "total_loss": self.model.total,
                "ori_loss": self.model.ori_loss,
            })