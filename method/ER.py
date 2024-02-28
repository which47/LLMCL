import random
import torch
import datasets
import numpy as np
import wandb
from datasets import Dataset, concatenate_datasets
from typing import Tuple
from .BaseTrainerCL import BaseTrainerCL


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
        input_ids = torch.stack([torch.cat([torch.zeros(max_input_id_len - ee.shape[0]).long().to(ee.device), ee]) for ee in [self.input_ids[c] for c in choice]]).reshape(size, max_input_id_len)
        labels = torch.stack([torch.cat([torch.zeros(max_label_len - ee.shape[0]).long().to(ee.device), ee]) for ee in [self.labels[c] for c in choice]]).reshape(size, max_label_len)
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


class ERTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        buffer_size: int = kwargs.get("buffer_size", None)
        buffer_rate: float = kwargs.get("buffer_rate", 0.1)
        buffer_method: str = kwargs.get("buffer_method", "random")
        if buffer_size is None and (0.0 < buffer_rate < 1.0):
            real_buffer_size = int(buffer_rate * self.ave_train_samples_per_task)
        else:
            real_buffer_size = buffer_size
            Warning("buffer_size is not None, buffer_rate will be ignored")
        self.buffer = Buffer(real_buffer_size, 'cpu')
    
    def concat_inputs(self, input_ids:torch.Tensor, labels:torch.Tensor, buffer_inputs_ids:torch.Tensor, buffer_labels:torch.Tensor) -> Tuple:
        max_input_id_len = max(input_ids.shape[1], buffer_inputs_ids.shape[1])
        max_labels_len = max(labels.shape[1], buffer_labels.shape[1])
        
        extended_input_ids = torch.cat([input_ids, torch.zeros(input_ids.shape[0], max_input_id_len - input_ids.shape[1]).long().to(input_ids.device)], dim=1)
        extended_labels = torch.cat([labels, torch.zeros(labels.shape[0], max_labels_len - labels.shape[1]).long().to(labels.device)], dim=1)
        
        extended_buffer_inputs_ids = torch.cat([buffer_inputs_ids, torch.zeros(buffer_inputs_ids.shape[0], max_input_id_len - buffer_inputs_ids.shape[1]).long().to(buffer_inputs_ids.device)], dim=1)
        extended_buffer_labels = torch.cat([buffer_labels, torch.zeros(buffer_labels.shape[0], max_labels_len - buffer_labels.shape[1]).long().to(buffer_labels.device)], dim=1)        
        
        input_ids = torch.cat([extended_input_ids, extended_buffer_inputs_ids], dim=0)
        labels = torch.cat([extended_labels, extended_buffer_labels], dim=0)
        return input_ids, labels
    
    def compute_loss(self, model, inputs, return_outputs=False):
        buffer_inputs, buffer_labels = self.buffer.get_data(inputs.input_ids.shape[0])
        
        if self.current_task_name != self.task_names[0] and buffer_inputs is not None and buffer_labels is not None:
            buffer_inputs, buffer_labels = buffer_inputs.to(inputs.input_ids.device), buffer_labels.to(inputs.labels.device)
            self.buffer.add_data(inputs.input_ids, inputs.labels)
            inputs.input_ids, inputs.labels = self.concat_inputs(inputs.input_ids, inputs.labels, buffer_inputs, buffer_labels)
            outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        
        return (outputs.loss,) if return_outputs else outputs.loss
            
            
    def continual_learning(self):
        resume_from_checkpoint = "False"
        for idx, (name, train_set) in enumerate(self.continual_training_dataset.items()):
            self.current_task_name = name
            self.update_adapter_and_train_set(resume_from_checkpoint, train_set)
            self.train()
            resume_from_checkpoint = self.save_model(name)
        wandb.finish()


