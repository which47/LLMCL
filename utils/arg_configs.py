import os
import json
from typing import List, Dict, Tuple, Union, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, TaskType
import warnings
from utils.functions import list_strings
__all__ = ["get_args", "CLArguments", "TuningArguments", "DataArguments"]
import bitsandbytes

@dataclass
class DataArguments:
    data_path: Union[str, Path] = "./TRACE-Benchmark/LLM-CL-Benchmark_5000"
    dataset_names: str = "20Minuten,FOMC,MeetingBank,NumGLUE-ds,ScienceQA,C-STANCE,NumGLUE-cm"
    max_length: int = 1024
    train_on_inputs: bool = False
    # truncation: bool = True
    # padding: bool = True
    
    def __post_init__(self):
        self.dataset_names = list_strings(self.dataset_names)

@dataclass
class CLArguments:
    cl_method: str = 'seq'
    ewc_lambda: float = 0.1
    er_buffer_size: int = 1000
    er_buffer_method: str = "random"
    gem_memory_strength: float = 0.5
    l2p_pool_size: int = 10
    l2p_prompt_length: int = 5
    l2p_prompt_init: str = "random"
    pp_prefix_length: int = 20
    
    cl_config: Union[Dict[str, Any], Any] = field(init=False)    
    def __post_init__(self):
        assert self.cl_method in ["seq","ewc", "er", "gem", "agem", "l2p", "pp", "mtl", "one", "ilora"], f"cl_method '{self.cl_method}' not supported"
        if self.cl_method == 'seq':
            warnings.warn("Using 'seq' as cl_method, other cl configs are ignored")
            self.cl_config = {}
        if self.cl_method == 'ewc':
            warnings.warn("Using 'ewc' as cl_method, other cl configs are ignored")
            self.cl_config = {
                "lambda": self.ewc_lambda
            }
        if self.cl_method == 'er':
            warnings.warn("Using 'er' as cl_method, other cl configs are ignored")
            self.cl_config = {
                "buffer_size": self.er_buffer_size,
                "buffer_method": self.er_buffer_method
            }
        if self.cl_method == 'gem':
            warnings.warn("Using 'gem' as cl_method, other cl configs are ignored")
            self.cl_config = {
                "memory_strength": self.gem_memory_strength
            }
        if self.cl_method == 'l2p':
            warnings.warn("Using 'l2p' as cl_method, other cl configs are ignored")
            self.cl_config = {
                "pool_size": self.l2p_pool_size,
                "prompt_length": self.l2p_prompt_length,
                "prompt_init": self.l2p_prompt_init
            }
        if self.cl_method == 'pp':
            warnings.warn("Using 'pp' as cl_method, other cl configs are ignored")
            self.cl_config = {
                "prefix_length": self.pp_prefix_length
            }
        if self.cl_method == 'ilora':
            warnings.warn("Using 'ilora' as cl_method, other cl configs are ignored")
            self.cl_config = {}
        if self.cl_method == 'mtl':
            warnings.warn("Using 'mtl' as cl_method, other cl configs are ignored")
            self.cl_config = {}
        if self.cl_method == 'one':
            warnings.warn("Using 'one' as cl_method, other cl configs are ignored")
            self.cl_config = {}
        
        print(f"*** cl_config ***:\n\t{self.cl_config}")
        
@dataclass
class TuningArguments:
    # basic config
    model_name_or_path: Union[str, Path] = "meta-llama/Llama2-7b-hf"
    load_in_8bit: bool = False
    # lora config
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    load_8bit: bool = True
    lora_config: Union[LoraConfig, Any] = field(init=False)      
    manual_seed: int = 37

    # redundant args
    config_file: Optional[str] = None
    def __post_init__(self):
        self.target_modules = list_strings(self.target_modules)
        self.lora_config = LoraConfig(
            r=self.lora_r, 
            lora_alpha=self.lora_alpha, 
            lora_dropout=self.lora_dropout, 
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM
        )
@dataclass
class InferArguments:
    cl_method: str = 'seq'
    model_name_or_path: Union[str, Path] = "meta-llama/Llama2-7b-hf"
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    tokenizer_name_or_path: Union[str, Path] = "meta-llama/Llama2-7b-hf"
    peft_cfg_path: Optional[str] = None
    peft_weights_path: Optional[str] = None
    infer_batch_size: int = 4
    # generation config
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.75
    repetition_penalty: float = 1.15
    do_sample: bool = True
    generation_config: Union[GenerationConfig, Any] = field(init=False)
    bnb_config: BitsAndBytesConfig = field(init=False)
    save_path: Union[str, Path] = "./generated_texts.json"
     
    def __post_init__(self):
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample
        )
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit
        )
@dataclass
class EvalArguments:
    cl_method: str = 'seq'
    json_dirs: str = "outputs/seq/20Minuten.json"
    increment_order: str = '20Minuten'
    save_path: str = "./eval_table.csv"
    
    def __post_init__(self):
        self.json_dirs = list_strings(self.json_dirs)
        self.increment_order = list_strings(self.increment_order)
    
def get_args() -> Tuple[TrainingArguments, CLArguments, TuningArguments, DataArguments]:
    parser = HfArgumentParser((TrainingArguments, CLArguments, TuningArguments, DataArguments))
    train_args, cl_args, tuning_args, data_args = parser.parse_args_into_dataclasses()
    return train_args, cl_args, tuning_args, data_args

if __name__ == "__main__":
    train_args, cl_args, tuning_args = get_args()
    print(train_args)
    print(cl_args)
    print(tuning_args)