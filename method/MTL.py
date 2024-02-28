import os
from copy import deepcopy
from typing import Optional, List, Union, Tuple, Dict
import copy
import numpy as np
import wandb
from peft import PeftModel
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

def turn_cl_data2mlt_data(cl_datasets: Optional[Dict[str, Dataset]]):
    dataset = concatenate_datasets([data for data in cl_datasets.values()])
    return dataset
        

class MTLTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_dataset = turn_cl_data2mlt_data(self.train_dataset)
    
    def continual_learning(self):
        """multi-task learning"""
        self.train()
        _ = self.save_model(name="multi")
    
    def save_model(self, name:str):
        """save model"""
        if self.args.output_dir is not None:
            save_path = os.path.join(self.args.output_dir, name)
            self.model.save_pretrained(save_path)
        return save_path
        