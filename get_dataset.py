import os
import torch
import json
import torch.utils
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer

class BaseDataset(Dataset):
    def __init__(self, tokenizer:PreTrainedTokenizerBase, json_dir:Union[str, Path], max_length:int=1024, train_on_inputs:bool=True, test:bool=False):
        super(BaseDataset).__init__()
        self.tokenizer = tokenizer
        assert self.tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id"
        self.json_dir = json_dir
        self.meta_data = json.load(open(json_dir))
        self.max_length = max_length
        self.train_on_inputs = train_on_inputs 
        self.keys_to_data = list(self.meta_data[0].keys())
        self.test = test
        self.data = self._tokenize_dataset()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _tokenize_dataset(self, ingnore_idx:int=-100) -> List[Dict[str, torch.Tensor]]:
        tokenized_samples = []
        for sample in self.meta_data:
            try:
                q_tokenized = self.tokenizer(sample[self.keys_to_data[0]], add_special_tokens=False)
                a_tokenized = self.tokenizer(sample[self.keys_to_data[1]], add_special_tokens=False)

                if not self.test:
                    input_ids = q_tokenized['input_ids'] + a_tokenized['input_ids']
                else:
                    input_ids = q_tokenized['input_ids']
                    
                if len(input_ids) > self.max_length - 2:
                    input_ids = input_ids[:self.max_length - 2]
                
                full_input_ids = [self.tokenizer.bos_token_id] + input_ids
                if not self.test:
                    full_input_ids += [self.tokenizer.eos_token_id]
                input_ids = torch.tensor(full_input_ids)
                attention_mask = torch.ones_like(input_ids)
                
                if (not self.train_on_inputs) and (not self.test):
                    labels = torch.full_like(input_ids, fill_value=ingnore_idx)
                    ans_start_idx = len(q_tokenized['input_ids']) + 1
                    labels[ans_start_idx:] = input_ids[ans_start_idx:]
                else:
                    labels = input_ids.clone()
                
                tokenized_samples.append(dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                ))
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        return tokenized_samples

class DataCollector(object):
    """ For a stable traning, we need to pad the input_ids to the `max_length` """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding: Union[str, bool], max_length: int=1024, ignore_idx: int=-100):
        self.tokenizer = tokenizer
        self.padding = padding
        assert self.padding in ['longest', True], "Padding must be either 'longest', 'max_length' or False"
        self.max_length = max_length
        self.ignore_idx = ignore_idx

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        input_ids = [sample['input_ids'] for sample in batch]
        attention_mask = [sample['attention_mask'] for sample in batch]
        labels = [sample['labels'] for sample in batch]
        
        len_pad_to = max([len(ids) for ids in input_ids]) if self.padding == 'longest' else self.max_length 
        for i in range(len(batch)):
            input_ids[i] = torch.cat([
                torch.full((len_pad_to - input_ids[i].shape[0],), fill_value=self.tokenizer.pad_token_id), # left padding
                input_ids[i]
            ])
            attention_mask[i] = torch.cat([
                torch.zeros((len_pad_to - attention_mask[i].shape[0],)),
                attention_mask[i]
            ])
            labels[i] = torch.cat([
                torch.full((len_pad_to- labels[i].shape[0],), fill_value=self.ignore_idx),
                labels[i]
            ])

        return dict(
            input_ids=torch.stack(input_ids),
            attention_mask=torch.stack(attention_mask),
            labels=torch.stack(labels)
        )
        
def get_datasets(dataset_names: List[str], data_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase, max_length: int=1024, split='train', train_on_inputs=False) -> Dict[str, Dataset]:
    datasets = {}
    for name in dataset_names:
        if os.path.exists(name) and os.path.isfile(name) and name.endswith(".json"):
            full_json_path = name
        else:
            full_json_path = os.path.join(data_path, name, f"{split}.json")
            assert os.path.exists(full_json_path), f"Path {full_json_path} does not exist"

        dataset = BaseDataset(tokenizer, full_json_path, max_length, train_on_inputs, test='test' in full_json_path)
        datasets[name] = dataset
    return datasets

def get_joint_datasets(datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
    return {'joint': torch.utils.data.ConcatDataset(list(datasets.values()))}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("llama2-7b-hf")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    dataset = get_datasets(["FOMC"], "./TRACE-Benchmark/LLM-CL-Benchmark_500", tokenizer)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset["FOMC"], batch_size=2, collate_fn=DataCollector(tokenizer, padding=True))
    for batch in data_loader:
        print(batch)
        break
    print(tokenizer("Hello World!, I am a sentence."))
