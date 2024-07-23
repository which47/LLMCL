import os
import time
import torch
import logging, json
import numpy as np
from typing import List, Dict, Union
from dataclasses import asdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    GenerationConfig,
    LlamaTokenizer
    )
from peft import load_peft_weights, set_peft_model_state_dict, get_peft_model, PeftConfig, LoraConfig
from get_dataset import get_datasets, DataCollector
from utils.arg_configs import DataArguments, InferArguments
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def prepare_model_for_inference(model_name_or_path:str, bnb_config:BitsAndBytesConfig, peft_cfg_path:str=None, peft_weights_path:str=None, device:str='cuda'):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="balanced_low_0",
    )
    
    if peft_cfg_path is not None and peft_weights_path is not None:
        peft_config = PeftConfig.from_pretrained(peft_cfg_path)
        model = get_peft_model(model, peft_config=peft_config)
        peft_state_dict = torch.load(peft_weights_path, map_location=device)
        set_peft_model_state_dict(model, peft_state_dict)
    return model

def prepare_dataloader(data_args:DataArguments, tokenizer:PreTrainedTokenizerBase, batch_size:int=4, max_length:int=1024)->Dict[str, DataLoader]:
    test_datasets = get_datasets(**asdict(data_args), tokenizer=tokenizer, split="test")
    dataloaders = {}
    for name, dataset in test_datasets.items():
        test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=DataCollector(tokenizer, padding="longest", max_length=max_length), num_workers=4, prefetch_factor=2)    
        dataloaders[name] = test_dataloader
    return dataloaders

@torch.no_grad()
def run_generation(model:torch.nn.Module, tokenizer:PreTrainedTokenizerBase, name:str, test_dataloader:DataLoader, generation_config:GenerationConfig) -> List[str]:
    model.eval()
    generated_texts = []
    for inputs in tqdm(test_dataloader, desc=f"Generating texts for {name}"):
        if 'labels' in inputs:
            inputs.pop('labels')
        input_ids_shape = inputs['input_ids'].shape
        generated_token_batch = model.generate(**inputs, generation_config=generation_config)
        # generated_token_batch = generated_token_batch.cpu().numpy().tolist()
        
        # mask input_ids to get only the generated text
        mask = torch.cat(
            (torch.zeros(input_ids_shape), torch.ones(input_ids_shape[0], generated_token_batch.shape[1] - input_ids_shape[1])),
            dim=-1
        ).to(torch.int64).to(generated_token_batch.device)
        generated_token_batch = (generated_token_batch * mask).cpu().numpy().tolist()
        generated_texts.extend(tokenizer.batch_decode(generated_token_batch, skip_special_tokens=True))
    return generated_texts

def get_meta_data(data_args:DataArguments, split="test")->Dict[str, List[Dict[str, str]]]:
    meta_datas = {}
    for name in data_args.dataset_names:
        full_path = os.path.join(data_args.data_path, name, f"{split}.json")
        assert os.path.exists(full_path), f"File {full_path} does not exist"
        
        with open(full_path, 'r') as f:
            data = json.load(f)
        meta_datas[name] = data
    return meta_datas


def main():
    parser = HfArgumentParser((InferArguments, DataArguments))
    infer_args, data_args = parser.parse_args_into_dataclasses()
    
    # prepare model, tokenizer and dataloaders
    model = prepare_model_for_inference(
        model_name_or_path=infer_args.model_name_or_path,
        bnb_config=infer_args.bnb_config,
        peft_cfg_path=infer_args.peft_cfg_path,
        peft_weights_path=infer_args.peft_weights_path,
    )
    
    # tokenizer_config = 
    tokenizer = AutoTokenizer.from_pretrained(infer_args.tokenizer_name_or_path)
    dataloaders = prepare_dataloader(data_args, tokenizer, batch_size=infer_args.infer_batch_size)
    logger.info(f"Model and data loaders prepared for {data_args.dataset_names}, starting generation")
    
    start = time.time()
    generated_texts_datasets = {}
    for name, dataloader in dataloaders.items():
        generated_texts = run_generation(
            model=model,
            tokenizer=tokenizer, 
            name=name,
            test_dataloader=dataloader,
            generation_config=infer_args.generation_config
        )
        generated_texts_datasets[name] = generated_texts
    end = time.time()
    
    # run generation
    logger.info(f"Generation completed, using {((end-start)/60):.2f} seconds")
    meta_datas = get_meta_data(data_args, split="test")
    results = {}
    for i, (name, gen_texts) in enumerate(generated_texts_datasets.items()):
        results[name] = []
        assert len(gen_texts) == len(meta_datas[name]), f"Number of generated texts ({len(gen_texts)}) does not match the number of meta datas ({len(meta_datas[name])})"
        gen_texts: List[str]
        meta_datas: Dict[str, List[Dict[str, str]]]
        for j, text in enumerate(gen_texts):
            results[name].append(dict(
                prompt=meta_datas[name][j]['prompt'],
                answer=meta_datas[name][j]['answer'],
                generated_text=text,
            ))
            
    # save results
    base_path = os.path.split(infer_args.save_path)[0]
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    with open(f"{infer_args.save_path}", 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to {infer_args.save_path}")
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    main()