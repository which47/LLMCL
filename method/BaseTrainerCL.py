import torch
import torch.nn as nn
from pathlib import Path
from typing import Callable, Tuple, Union, Optional, Dict, List, overload
from torch.utils.data import Dataset
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from utils.arg_configs import CLArguments, TuningArguments, DataArguments
from get_dataset import get_joint_datasets
from peft import get_peft_model, PeftModel, LoraModel, PeftModelForCausalLM, get_peft_model_state_dict
from transformers import Trainer, PreTrainedModel, TrainerCallback


class BaseTrainerCL(Trainer):
    def __init__(self, **kwargs):
        self.cl_args = kwargs.pop("cl_args", None)
        self.tuning_args = kwargs.pop("tuning_args", None)
        self.data_args = kwargs.pop("data_args", None)
        kwargs['model'] = self.prepare_model_for_cl_traning(kwargs['model'], self.cl_args, self.tuning_args)
        self.continual_training_dataset, self.continual_evaluating_dataset = \
            self.prepare_dataset_for_cl_traininig(
            kwargs.get("train_dataset", None),
            kwargs.get("eval_dataset", None))
        self.task_names = list(self.continual_training_dataset.keys())
        self.num_tasks: int = len(self.continual_training_dataset)
        self.current_task_name: str = None
        super().__init__(**kwargs)
        
    def prepare_model_for_cl_traning(self, model: Union[PreTrainedModel, nn.Module], cl_args: CLArguments=None, tuning_args: TuningArguments=None) -> Union[PreTrainedModel, PeftModel]:
        peft_model = get_peft_model(
            model=model,
            peft_config=tuning_args.lora_config,
        )
        return peft_model
    
    def prepare_dataset_for_cl_traininig(self, train_dataset: Dict[str, Dataset], eval_dataset: Dict[str, Dataset]) -> Dict[str, Dataset]:
        
        if self.cl_args.cl_method == 'mtl':
            train_dataset = get_joint_datasets(train_dataset)
            eval_dataset = get_joint_datasets(eval_dataset)
            return train_dataset, eval_dataset
        return train_dataset, eval_dataset
    
    def continual_learning(self):        
        for i, name in enumerate(self.task_names):
            self.before_task_start(name)
            self.train()
            self.after_task_end(name)

    def before_task_start(self, task_name: str):
        """ update training and evaluation dataset for the current task """
        if self.cl_args.cl_method == 'mtl':
            self.train_dataset = self.continual_evaluating_dataset
            self.eval_dataset = self.continual_evaluating_dataset
            self.current_task_name = "joint"
            return
                
        if task_name not in self.continual_training_dataset:
            raise ValueError(f"task name {task_name} not found in the dataset")
        self.current_task_name = task_name
        self.train_dataset, self.eval_dataset = self.continual_training_dataset[task_name], self.continual_evaluating_dataset[task_name]
        
        # update model for the current task
        if self.cl_args.cl_method == 'one':
            if isinstance(self.model, LoraModel):
                self.model = get_peft_model(
                    model=self.model.model,
                    peft_config=self.tuning_args.lora_config,
                )
            
    def after_task_end(self, *args, **kwargs):
        """ save the model after training the current task """
        assert args[0] == self.current_task_name, f"task name mismatch: {args[0]} != {self.current_task_name}"
        wrappered_model_class = kwargs.get("wrappered_model_class", None)
        
        if isinstance(self.model, PeftModelForCausalLM):
            lora_state_dict = get_peft_model_state_dict(self.model)
            lora_config = self.model.peft_config
            if self.args.local_rank == 0:
                print(f"*** Saving lora adapter for task: {self.current_task_name} ***")
                torch.save(lora_state_dict, Path(self.args.output_dir).joinpath(f"lora_{self.current_task_name}.pt"))
                for adapter_name, adapter_config in lora_config.items():
                    if adapter_name == 'default':
                        adapter_config.save_pretrained(Path(self.args.output_dir).joinpath(f"lora_{self.current_task_name}"))
                    else:
                        adapter_config.save_pretrained(Path(self.args.output_dir).joinpath(f"lora_{self.current_task_name}_{adapter_name}"))
        elif not wrappered_model_class and isinstance(self.model, wrappered_model_class):
            raise NotImplementedError("not implemented yet") # TODO: implement for PP model
        
        self.tokenizer.save_pretrained(Path(self.args.output_dir).joinpath(f"tokenizer_{self.current_task_name}"))