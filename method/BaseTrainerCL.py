import os

import peft
import torch
from typing import Callable, Tuple, Union, Optional, Dict, List, overload
import wandb
import torch.nn as nn
from datasets import Dataset
import sys
from statistics import mean

from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig,
    PromptEncoder,
    PromptEncoderConfig,
    get_peft_model,
    PeftModel,
    set_peft_model_state_dict,
    load_peft_weights, PeftConfig,
    prepare_model_for_int8_training,
)
from transformers import Trainer, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, \
    TrainerCallback, TrainingArguments


class BaseTrainerCL(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dict[str, Dataset]] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            # customized parameters
            cl_method: str = None,
            adapter: str = None,
            # lora config
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_target_modules: List[str] = None,
            lora_dropout: float = 0.05,
            lora_bias: str = 'none',
            # prompt tuning config
            num_virtual_tokens: int = None,
    ):
        peft_cfg = get_adapter_cfg(adapter, lora_r, lora_alpha, lora_target_modules, lora_dropout,
                                   lora_bias, num_virtual_tokens, model.config)
        
        assert peft_cfg is not None

        if adapter is not None and cl_method != "adapter_cl" and cl_method != "hat":
            model = get_peft_model(model, peft_cfg)
            
        elif cl_method == "hat":
            from method.HAT import HAT
            model = HAT(model, peft_cfg)
        
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        
        self.cl_method: str = cl_method
        self.adapter: str = adapter
        self.continual_training_dataset = train_dataset
        self.current_task_name: str = list(train_dataset.keys())[0] # current task name: C-STANCE for the first
        self.ave_train_samples_per_task: int = mean([len(dataset) for dataset in train_dataset.values()]) # average training samples per task: almost 5000
        self.task_names = list(train_dataset.keys()) # task names: 8
        self.n_tasks: int = len(self.task_names) # number of tasks 8

    def continual_learning(self):
        resume_from_checkpoint = "False"
        for name, train_set in self.continual_training_dataset.items():
            self.current_task_name = name
            self.update_adapter_and_train_set(resume_from_checkpoint, train_set)
            self.train()
            resume_from_checkpoint = self.save_model(name)
        wandb.finish()

    def update_adapter_and_train_set(self, resume_from_checkpoint, train_set):
        """
            update adapter and train set before a trainer.train() start
        """
        print(f"\ncurrent task: {self.current_task_name}\n")
        if self.current_task_name != self.task_names[0] and self.cl_method != "cls":
            try:
                adapter_weights = load_peft_weights(resume_from_checkpoint)
                set_peft_model_state_dict(self.model, adapter_weights)
            except:
                print("load adapter weights failed, using default start")
            print(f"Restarting from checkpoint{resume_from_checkpoint}")
        self.train_dataset = train_set  # update train set

    def save_model(self, name) -> str:
        if self.args.output_dir is not None:
            save_dir = os.path.join(self.args.output_dir, f"{self.cl_method}_{self.adapter}_checkpoint_{name}")
            self.model.save_pretrained(save_dir)
            print(f"save task: {name} adapter to {self.args.output_dir}")
            return save_dir


def get_adapter_cfg(adapter: str,
                    lora_r: int = 8,
                    lora_alpha: int = 16,
                    lora_target_modules: List[str] = None,
                    lora_dropout: float = 0.05,
                    lora_bias: str = 'none',
                    # prompt tuning config
                    num_virtual_tokens: int = 30,
                    model_cfg = None,
                    ) -> PeftConfig:
    """
    Get the adapter model based on the provided adapter type and configuration.
    """
    if not isinstance(adapter, str):
        raise ValueError(f"adapter type: {adapter} not support")
    config = None
    if adapter.lower() == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type="CAUSAL_LM",
        )
    elif adapter.lower() == "prompt":
        config = PromptTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
            prompt_tuning_init_text="give a appropriate response to the following context",
            tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
        )
    elif adapter.lower() == "prefix":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    elif adapter.lower() == "p_tuning":
        config = PromptEncoderConfig(
            peft_type='P_TUNING',
            task_type="CAUSAL_LM",
            num_virtual_tokens=num_virtual_tokens,
            token_dim=model_cfg['hidden_size'],
            num_attention_heads=model_cfg['num_attention_heads'],
            num_layers=model_cfg['num_hidden_layers'],
            encoder_reparameterization_type="MLP",
            encoder_hidden_size=768,
        )
    else:
        ValueError(f"adapter method {adapter} not implement yet !")
    return config
