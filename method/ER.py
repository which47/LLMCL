import torch
import numpy as np
from typing import Tuple
from .BaseTrainerCL import BaseTrainerCL


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

def concat_inputs(input_ids:torch.Tensor, attention_mask:torch.Tensor, labels:torch.Tensor, buffer_input_ids:torch.Tensor, buffer_attention_mask:torch.Tensor, buffer_labels:torch.Tensor) -> Tuple:
    device = input_ids.device
    input_ids = torch.cat((input_ids, buffer_input_ids.to(device)), dim=0)
    attention_mask = torch.cat((attention_mask, buffer_attention_mask.to(device)), dim=0)
    labels = torch.cat((labels, buffer_labels.to(device)), dim=0)
    return input_ids, attention_mask, labels

class Buffer:
    def __init__(self, buffer_size:int, device:str, pad_id:int=2, ignore_index:int=-100):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['input_ids', 'attention_mask', 'labels', 'logits', 'task_labels', 'activations']
        self.init_buffer()
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        
    def init_buffer(self) -> None:
        for attr_str in self.attributes:
            setattr(self, attr_str, [None for _ in range(self.buffer_size)])

    def add_data(self, input_ids, attention_mask=None, labels=None, logits=None, task_labels=None, activations=None):
        n = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
        for i in range(n):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.input_ids[index] = input_ids[i].detach().clone().to(self.device)
                if attention_mask is not None:
                   self.attention_mask[index] = attention_mask[i].detach().clone().to(self.device) 
                if labels is not None:
                    self.labels[index] = labels[i].detach().clone().to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].detach().clone().to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].detach().clone().to(self.device)
                if activations is not None:
                    self.activations[index] = activations[i].detach().clone().to(self.device)

    def get_data(self, size: int, pad_to:int) -> Tuple:
        n = len(self.input_ids)
        if size > min(self.num_seen_examples, n):
            size = min(self.num_seen_examples, n)

        choice = np.random.choice(min(self.num_seen_examples, n), size=size, replace=False)
        if len(choice) == 0:
            return None, None
        # for left padding
        input_ids = []
        attention_mask = []
        labels = []
        
        for i in choice:

            input_ids.append(torch.cat(
                (torch.full((pad_to - self.input_ids[i].shape[-1],), self.pad_id, dtype=torch.long).to(self.device),
                self.input_ids[i]), dim=-1)
            )
            if self.attention_mask[i] is not None:
                attention_mask.append(torch.cat(
                    (torch.full((pad_to - self.attention_mask[i].shape[-1],), 0, dtype=torch.long).to(self.device),
                    self.attention_mask[i]), dim=-1)
                )
            if self.labels[i] is not None:
                labels.append(torch.cat(
                    (torch.full((pad_to - self.labels[i].shape[-1],), self.ignore_index, dtype=torch.long).to(self.device),
                    self.labels[i]), dim=-1)
                )
        
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels

    def is_empty(self) -> bool:
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.input_ids]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


class ERTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = self.cl_args.cl_config.get('buffer_size', None)
        self.buffer = Buffer(self.buffer_size, 'cpu', pad_id=self.tokenizer.pad_token_id, ignore_index=-100)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        if self.current_task_name == self.task_names[0]:
            self.buffer.add_data(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
            outputs = model(**inputs)
        else:
            buffer_inputs, buffer_attention_mask, buffer_labels = self.buffer.get_data(inputs["input_ids"].shape[0], inputs["input_ids"].shape[1])
            if buffer_inputs is not None and buffer_attention_mask is not None and buffer_labels is not None:
                inputs["input_ids"], inputs["attention_mask"], inputs["labels"] = concat_inputs(inputs["input_ids"], inputs["attention_mask"], inputs["labels"], buffer_inputs, buffer_attention_mask, buffer_labels) 
            outputs = model(**inputs)
            self.buffer.add_data(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss
            
            
    def continual_learning(self):
        for i, name in enumerate(self.task_names):
            self.before_task_start(name)
            self.train()
            self.after_task_end(name)


