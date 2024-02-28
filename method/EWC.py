from transformers import TrainerCallback, TrainingArguments, PreTrainedModel
from transformers.trainer_callback import TrainerState, TrainerControl
from copy import deepcopy
from typing import Tuple, Dict
import os
import wandb
from .BaseTrainerCL import BaseTrainerCL
from peft import PeftModel

def format_name(name):
    index = name.rfind("model")
    return name[index:]
    
class GradientCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fisher = {}
        self.model: PeftModel = kwargs.get("model")
        self.previous_weights = {}
        self.trainable_params = {}
        self.init_fisher_and_weights()
        self.ewc_loss = 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        assert len(self.fisher) > 0 and len(self.trainable_params) > 0, "fisher and previous_weights should not be empty"
    
    def update_ewc_loss(self, ewc_loss):
        self.ewc_loss = ewc_loss
    
    def init_fisher_and_weights(self):
        print("init fisher and weights")
        for n, p in self.model.named_parameters():
            n = format_name(n)
            if p.requires_grad:
                self.fisher[n] = p.detach().clone().data.zero_()
                self.trainable_params[n] = p.detach().clone().data

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            save trainable parameters' weights
        """
        print("update trainable parameters' weights")
        self.previous_weights = {format_name(n): p.detach().clone() for n, p in self.model.named_parameters() if
                                    format_name(n) in self.trainable_params.keys()}

    def get_fisher_and_prior(self) -> Tuple[dict, dict]:
        return self.fisher, self.previous_weights

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            update fisher matrix
        """
        for n, p in self.model.named_parameters():
            n = format_name(n)
            if n in self.trainable_params.keys() and p.grad is not None:
                self.fisher[n] += p.grad.detach().clone().data.pow(2) / state.global_step
            elif p.grad is None:
                Warning(f"parameter {n} has no gradient")
        if self.local_rank <= 0:
            print(f"ewc loss: {self.ewc_loss}")

class EWCTrainer(BaseTrainerCL):
    """
        https://arxiv.org/abs/1612.00796
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GradientCallback(model=self.model))
        self.ewc_lambda = kwargs.get("ewc_lambda", 0.5)
        self.cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GradientCallback):
                self.cb = cb
                break
        
    def continual_learning(self):
        self.model.print_trainable_parameters()
        resume_from_checkpoint = "False"
        for name, train_set in self.continual_training_dataset.items():
            self.current_task_name = name
            self.update_adapter_and_train_set(resume_from_checkpoint, train_set)
            self.train()
            resume_from_checkpoint = self.save_model(name)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs.pop("labels")
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        ewc_loss = self.compute_ewc_loss(model)
        self.cb.update_ewc_loss(ewc_loss.item())
        loss += ewc_loss
        return (loss, outputs) if return_outputs else loss

    def compute_ewc_loss(self, model):
        """
            compute ewc loss
        """
        ewc_loss = 0
        fisher, previous_weights = self.cb.get_fisher_and_prior()
        assert len(fisher) > 0 and len(previous_weights) > 0, "fisher and previous_weights should not be empty"
        
        for n, p in model.named_parameters():
            n = format_name(n)
            if n in fisher.keys():
                ewc_loss += (fisher[n] * (p - previous_weights[n]).pow(2)).sum() * self.ewc_lambda / 2
                
        if ewc_loss < 1e-5:
            Warning("EWC regularization loss is too small, please check the hyper-parameters")
        return ewc_loss