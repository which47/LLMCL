import os
import logging
from accelerate import Accelerator
import fire
import time
import torch
from peft import PeftModel
from transformers import (GenerationConfig, BitsAndBytesConfig,DataCollatorWithPadding,
                          AutoModelForCausalLM, AutoTokenizer)
from tqdm import tqdm
import json
from dataset import create_test_datasets
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_model(model_name, adapter_checkpoint_dir, model_cfg, bnb_config, accelerator):
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = "auto",
            # load_in_8bit=True,
            # torch_dtype=torch.float16,
            quantization_config=bnb_config
        )
        if adapter_checkpoint_dir != "":
            model = PeftModel.from_pretrained(model, adapter_checkpoint_dir)
            if model_cfg['cl_method'].lower() == "l2p":
                from method.L2P import L2PModel
                model = L2PModel(model)
                model.load_prompt_weights(path=model_cfg['l2p_prompt_weight_path'],
                                          task_name=model_cfg['trained_adapter'])
                model.freeze_prompt()
            elif model_cfg['cl_method'].lower() == "pp":
                from method.PP import PPModel
                model = PPModel(model, prefix_len=20,
                                task_names=model_cfg['continual_datasets'])
                model.load_mlps_prompts(path=model_cfg['pp_mlps_promts_path'],
                                        task_name=model_cfg['trained_adapter'])
                model.freeze_all()
        else:
            logger.info("No adapter checkpoint directory provided, using base model")
        if model_cfg['cl_method'].lower() == "l2p" or model_cfg['cl_method'].lower() == "pp":
            model.model.resize_token_embeddings(model.config.vocab_size + 1)
            
            model.model.config.pad_token_id = 0  # unk
            model.model.config.bos_token_id = 1
            model.model.config.eos_token_id = 2
            model.model = model.model.to_bettertransformer() # make sure model supports bettertransformer
        else:
            model.resize_token_embeddings(model.config.vocab_size + 1)
            model.config.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
            model = model.to_bettertransformer() # make sure model supports bettertransformer
        # model.bfloat16()
        return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side='left',
                                              truncation=True)

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    return tokenizer

def get_configs(adapter_checkpoint_dir):
    generation_cfg = {
        'max_new_tokens': 128,
        'temperature': 0.1,
        'top_p': 0.75,
        'repetition_penalty': 1.15,
        'do_sample': False,
    }
    inference_cfg = {
        'max_length': 1024,
        'tok_batch_size': 16,
        'inference_batch_size': 12,
    }
    model_cfg = {
        'pad_to_multiple_of': 8,
        'cl_method': os.path.basename(adapter_checkpoint_dir).split('_')[0] if adapter_checkpoint_dir is not None else "base",
        'trained_adapter': adapter_checkpoint_dir.split("_")[-1] if adapter_checkpoint_dir is not None else "base",
        'l2p_prompt_weight_path': os.path.dirname(adapter_checkpoint_dir) if adapter_checkpoint_dir is not None else None,
        'pp_mlps_promts_path': os.path.dirname(adapter_checkpoint_dir) if adapter_checkpoint_dir is not None else None,
        'continual_datasets': ["C-STANCE","FOMC","MeetingBank","ScienceQA","NumGLUE-cm","20Minuten","medmcqa","jecqa"],
    }
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # load a insruction prompt
    from utils.prompter import Prompter
    prompt_config = Prompter("alpaca")
    return generation_cfg, inference_cfg, model_cfg, bnb_config, prompt_config

def get_input_prompt(instruction, prompt_config):
    res = prompt_config.generate_prompt(instruction, None)
    return res

def get_dataloader(dataset, tokenizer, inference_cfg, prompt_config, accelerator):
    with accelerator.main_process_first():
        dataset = dataset.map(lambda e: {'full_prompt': get_input_prompt(e['prompt'], prompt_config)})
        columns = dataset.column_names
        tokenized = dataset.map(
            lambda e: tokenizer(e['full_prompt'], truncation=True, return_tensors='pt',
                                padding='max_length', max_length=inference_cfg['max_length']),
            batched=True,
            batch_size=inference_cfg['tok_batch_size'])
        tokenized = tokenized.remove_columns(columns)
        data_collator = DataCollatorWithPadding(tokenizer)
        dataloader = DataLoader(tokenized, batch_size=inference_cfg['inference_batch_size'],
                                collate_fn=data_collator)
        return dataloader

def run_generation(generation_cfg, dataloader, tokenizer, model, accelerator,cl_method,trained_adapter,dataset_name):
    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.wait_for_everyone()

    output_sequences = []
    start_time = time.time()

    for batch in tqdm(dataloader,desc=f"{cl_method} adapter {trained_adapter} inference on {dataset_name}"):
        unwrapped_model = accelerator.unwrap_model(model)

        with torch.inference_mode():
            generated_tokens = unwrapped_model.generate(**batch, **generation_cfg)

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().tolist()
        
        outputs = [tokenizer.decode(ids[batch['input_ids'].shape[1]:], skip_special_tokens=True) for ids in generated_tokens]
        tqdm.write(f"Output:\n{outputs}")
        output_sequences.extend(outputs)

    generation_end_time = time.time()
    logger.info(f"Generation time: {generation_end_time - start_time} sec")
    return output_sequences



def list_of_strings(arg):
    return arg.split(",")

def main(
    model_name:str,
    output_dir:str,
    dataset_names:list_of_strings,
    adapter_checkpoint_dir:str = "",
    ):

    if not os.path.exists(adapter_checkpoint_dir):
        logger.error("Could not find adapter checkpoint directory")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    generation_cfg, inference_cfg, model_cfg, bnb_config, prompt_config = get_configs(adapter_checkpoint_dir)
    
    accelerator = Accelerator()
    
    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(model_name)
    
    logger.info("Loading datasets")
    test_datasets = create_test_datasets(list_of_strings(dataset_names))
    
    logger.info("Loading model")
    model = get_model(model_name, adapter_checkpoint_dir, model_cfg, bnb_config, accelerator)
    
    logger.info("starting inference")
    for name, test_set in test_datasets.items():
        if model_cfg['cl_method'].lower() == "pp":
            model.current_task_name = name
        logger.info(f"Adapter {model_cfg['trained_adapter']} inference on {name}")
        dataloader = get_dataloader(test_set, tokenizer, inference_cfg, prompt_config, accelerator)
        output_sequences = run_generation(generation_cfg, dataloader, tokenizer, model, accelerator,model_cfg['cl_method'],model_cfg['trained_adapter'],name)
    
        if accelerator.is_local_main_process:
            logger.info("Saving results")
            prompts = [p["prompt"] for p in test_set]
            answers = [p['answer'] for p in test_set]
            outputs = output_sequences
            with open(os.path.join(output_dir,f"{model_cfg['trained_adapter']}_{name}.json"),"w",encoding="utf-8") as f:
                json.dump({"prompts":prompts,
                           "answers":answers,
                           "outputs":outputs}, f, ensure_ascii=False,indent=4)
            
    
            
if __name__ == "__main__":
    fire.Fire(main)

