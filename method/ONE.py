import os
from copy import deepcopy
from typing import Optional, List, Union, Tuple, Dict
import copy
import numpy as np
import wandb
from peft import PeftModel, get_peft_model
from .BaseTrainerCL import BaseTrainerCL
import torch
from transformers import LlamaForCausalLM, PreTrainedModel, TrainerCallback, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
import sys
sys.path.append("/root/autodl-tmp/LLMCL/")
from dataset import create_datasets

class ONETrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def continual_learning(self):
        "single-task learning"
        for task_name, task_dataset in self.continual_training_dataset.items():
            self.train_dataset = task_dataset
            self.train()
            _ = self.save_model(task_name)
    
    def save_model(self, name:str):
        """save model"""
        if self.args.output_dir is not None:
            save_path = os.path.join(self.args.output_dir, name)
            self.model.save_pretrained(save_path)
            # renew model
            cfg = self.model.peft_config['default']
            self.model = get_peft_model(self.model.base_model, cfg).to('cuda')
        return save_path
        