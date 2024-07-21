from typing import Optional, List, Union, Tuple

import torch, os 
import torch.utils.data
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaForCausalLM, PreTrainedModel, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel
from .BaseTrainerCL import BaseTrainerCL
import numpy as np
from copy import deepcopy


def l2_normalize(x, dim=None, epsilon=1e-12):
    square_norm = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_norm, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm

class L2PModel(LlamaForCausalLM):
    def __init__(self, model:PreTrainedModel, pool_size:int=10, prompt_length:int=5, promt_init:str='random'):
        super().__init__(model.config)
        self.model = model
        self.embed_tokens = self.model.get_input_embeddings()
        self.embed_tk_shapes = self.embed_tokens.weight.shape
        self.prompt = None
        self.top_k = 3
        self.diversity_loss_weight = 0.5
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.init_prompt(promt_init)
        self.embeding_key = 'mean'
        self.batchwise_prompt: bool = False
        self.current_task_name:str = None
        
    def init_prompt(self,promt_init):
        self.prompt = nn.Parameter(
            torch.tensor(
                self.create_prompt(self.pool_size, self.prompt_length, promt_init), requires_grad=True
            )
        ).to(self.device)

    def create_prompt(self, pool_size, prompt_length, promt_init='random'):
        N = self.embed_tk_shapes[0]
        p_weights = []
        
        for p in range(self.pool_size):
            p_w = []
            for i in range(self.prompt_length):
                with torch.no_grad():
                    j = np.random.randint(N)
                    w = deepcopy(self.embed_tokens.weight[j].detach().cpu().numpy())
                    p_w.append(w)
            p_weights.append(p_w)
            
        return np.array(p_weights)

    def save_prompt_weights(self, path):
        state_dict = {"prompt_pool": self.prompt}
        torch.save(state_dict, os.path.join(path, f"prompt_weights_{self.current_task_name}.pt"))
    
    def load_prompt_weights(self, path, task_name="jecqa"):
        state_dict = torch.load(os.path.join(path, f"prompt_weights_{task_name}.pt"), map_location=self.device)
        self.prompt.data = state_dict["prompt_pool"].data
        print(f"Loaded prompt weights from {path}")
        
    def freeze_prompt(self):
        for n, p in self.named_parameters():
            p.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        i_input_embeds = self.embed_tokens(input_ids)
        out = dict()
        if self.embeding_key == 'mean':
            i_input_embeds_mean = torch.mean(i_input_embeds, dim=1)
        elif self.embeding_key == 'max':
            i_input_embeds_mean = torch.max(i_input_embeds, dim=1)[0]
        elif self.embeding_key == 'mean_max':
            i_input_embeds_mean = torch.max(i_input_embeds, dim=1)[0] + 2 * torch.mean(i_input_embeds, dim=1)
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        
        prompt_key = torch.mean(self.prompt, dim=1) # Pool_size, C
        prompt_norm = l2_normalize(prompt_key, dim=1).to("cuda")
        inputs_embeds_norm = l2_normalize(i_input_embeds_mean, dim=1)
        prompt_norm = prompt_norm.to(dtype=inputs_embeds_norm.dtype)
        similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t()) # B, Pool_size
        
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            major_prompt_id = prompt_id[major_idx]
            idx = major_prompt_id.expand(inputs_embeds.shape[0], -1)
        
        batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)
        inputs_embeds = torch.cat([batched_prompt, i_input_embeds],axis=1)
        
        prefix_length = batched_prompt.shape[1]
        attn_masks = torch.concat((torch.tensor(1).to("cuda").repeat(batch_size,prefix_length),attention_mask), axis=1)
        
        if labels is None: # inference
            return self.model(inputs_embeds=inputs_embeds.half(), attention_mask=attn_masks, use_cache=False, return_dict=True)
        
        labels = torch.concat((labels[0][0].repeat(batch_size,inputs_embeds.shape[1]-labels.shape[1]),labels),axis=1)
        outs = self.model(inputs_embeds=inputs_embeds,labels=labels,attention_mask=attn_masks,use_cache=False)
        loss = outs[0]
        batched_key_norm = prompt_norm[idx]
        inputs_embeds_norm = inputs_embeds_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * inputs_embeds_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

        loss -= reduce_sim * self.diversity_loss_weight
        return loss
    
class L2PTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = L2PModel(self.model)
        
    def compute_loss(self, model:L2PModel, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attn_masks = inputs['attention_mask']
        labels = inputs.pop('labels')
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attn_masks)
        return outputs # since we only calculate loss in model.forward(), we return outputs here
        
    def continual_learning(self):
        for i, name in enumerate(self.task_names):
            self.model.current_task_name = name
            self.before_task_start(name)
            self.train()
            self.after_task_end(name)

    def save_model(self, name) -> str:
        if self.args.output_dir is not None:
            output_dir = os.path.join(self.args.output_dir, f"{self.cl_method}_{self.adapter}_checkpoint_{name}")
            assert isinstance(self.model.model, PeftModel), "self.model.model is not a PeftModel"
            self.model.model.save_pretrained(output_dir)
            print(f"save task: {name} adapter to {self.args.output_dir}")
            return output_dir