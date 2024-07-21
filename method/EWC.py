from transformers import TrainerCallback, TrainingArguments, PreTrainedModel
from transformers.trainer_callback import TrainerState, TrainerControl
from typing import Tuple, Dict
import os
from .BaseTrainerCL import BaseTrainerCL
from peft import PeftModel

class GradientCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fisher = {}
        self.model: PeftModel = kwargs.get("model")
        self.previous_weights = {}
        self.trainable_params = {}
        self.init_fisher_and_weights()
        assert len(self.fisher) > 0 and len(self.trainable_params) > 0, "fisher and previous_weights should not be empty"
    
    def init_fisher_and_weights(self):
        print("init fisher and weights")
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = p.detach().clone().data.zero_()
                self.trainable_params[n] = p.detach().clone().data

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.previous_weights = {n: p.detach().clone() for n, p in self.model.named_parameters() if
                                    n in self.trainable_params.keys()}

    def get_fisher_and_prior(self) -> Tuple[dict, dict]:
        return self.fisher, self.previous_weights

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for n, p in self.model.named_parameters():
            if n in self.trainable_params.keys() and p.grad is not None:
                self.fisher[n] += p.grad.detach().clone().data.pow(2) / state.global_step
            elif p.grad is None:
                Warning(f"parameter {n} has no gradient")

class EWCTrainer(BaseTrainerCL):
    """
        https://arxiv.org/abs/1612.00796
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(GradientCallback(model=self.model))
        self.ewc_lambda = self.cl_args.ewc_lambda 
        self.cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, GradientCallback):
                self.cb = cb
                break
        
    def continual_learning(self):
        for i, name in enumerate(self.task_names):
            self.before_task_start(name)
            self.train()
            self.after_task_end(name)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        ewc_loss = self.compute_ewc_loss(model)
        loss += ewc_loss
        return (loss, outputs) if return_outputs else loss

    def compute_ewc_loss(self, model):
        fisher, previous_weights = self.cb.get_fisher_and_prior()
        assert len(fisher) > 0 and len(previous_weights) > 0, "fisher and previous_weights should not be empty"
        
        ewc_loss = 0
        for n, p in model.named_parameters():
            if n in fisher:
                ewc_loss += (fisher[n] * (p - previous_weights[n]).pow(2)).sum() * self.ewc_lambda / 2
                
        if ewc_loss < 1e-5:
            Warning("EWC regularization loss is too small, please check the hyper-parameters")
        return ewc_loss