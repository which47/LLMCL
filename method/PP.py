from typing import Optional, List, Union, Tuple

import torch, os, wandb
import torch.utils.data
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaForCausalLM, PreTrainedModel, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel
from .BaseTrainerCL import BaseTrainerCL
import numpy as np
from copy import deepcopy

class ResMLP(torch.nn.Module):
    def __init__(self, hidden_dim, bottleneck_size, module_type='MLP1', residual=True):
        super().__init__()
        self.residual = residual
        if module_type=='MLP1':
            self.module = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, hidden_dim),
            )

        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size),
                nn.Tanh(),
                nn.Linear(bottleneck_size, hidden_dim),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)
            
class PPModel(LlamaForCausalLM):
    def __init__(self,model:PreTrainedModel, prefix_len, task_names, prefix_path=None):
        super().__init__(model.config)
        self.model = model
        self.prefix_len:int = prefix_len
        self.prefix_pth:str = prefix_path
        self.task_names:List = task_names
        self.num_tasks:int = len(task_names)
        self.current_task_name:str = None
        self.embed_tokens = self.model.get_input_embeddings() # [vocab_size, embed_dim]
        self.embed_tokens_len = self.embed_tokens.weight.shape[0] # vocab_size
        
        self.prompt = nn.Parameter(
            torch.tensor(self.init_prompt(), requires_grad=True) # [prefix_len, embed_dim]
        ).to(self.device)
        
        self.previous_prompts = torch.zeros([0, self.prompt.shape[1]], requires_grad=False, dtype=torch.bfloat16).to(self.device) # [0, embed_dim] 
        self.mlps = nn.ModuleList([ResMLP(model.config.hidden_size, 128, module_type='MLP1', residual=True) for _ in range(self.num_tasks)]).to(self.device)
        
        assert self.prefix_len > 0
        
        self.pp = True
        self.is_train = True
    
    def init_prompt(self):
        prompt_weights = []
        for i in range(self.prefix_len):
            with torch.no_grad():
                j = np.random.randint(self.embed_tokens_len)
                w = deepcopy(self.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weights.append(w / 100)
        return np.array(prompt_weights)
    
    def progressive_previous_prompt(self, task_name):
        """
            update previous prompt at end of each task
        """
        if task_name != None and self.mlps != None:
            with torch.no_grad():
                new_prompt = self.mlps[self.task_names.index(task_name)](self.prompt)
                self.previous_prompts = torch.cat([self.previous_prompts, new_prompt], axis=0)
                print(f'updated previous prompt to: {self.previous_prompts.shape}')
    
    def freeze_mlps(self, name:str, requires_grad=False):
        """
        Freeze or unfreeze all the MLPs except the one for the current task
        """
        for i, mlp in enumerate(self.mlps):
            if i != self.task_names.index(name):
                for param in mlp.parameters():
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad
            else:
                for param in mlp.parameters():
                    if param.requires_grad == requires_grad:
                        param.requires_grad = not requires_grad
                        
    def save_mlps_prompts(self, path):
        """
            Save all the MLPs and prompts and previous learned prompts
        """
        mlp_state_dict = {"mlps": self.mlps.state_dict()}
        prompt_state_dict = {"prompt": self.prompt}
        previous_prompt_state_dict = {"previous_prompt": self.previous_prompts}
        torch.save(mlp_state_dict, os.path.join(path, f"mlps_{self.current_task_name}.pt"))
        torch.save(prompt_state_dict, os.path.join(path, f"prompt_{self.current_task_name}.pt"))
        torch.save(previous_prompt_state_dict, os.path.join(path, f"previous_prompt_{self.current_task_name}.pt"))
        
    def load_mlps_prompts(self, path, task_name = None):
        """
            Load all the MLPs and prompts
        """
        mlp_path = os.path.join(path, f"mlps_{task_name}.pt")
        prompt_path = os.path.join(path, f"prompt_{task_name}.pt")
        previous_prompt_path = os.path.join(path, f"previous_prompt_{task_name}.pt")
        assert mlp_path and prompt_path and previous_prompt_path, "mlp_path or prompt_path is None"
        
        mlp_state_dict = torch.load(mlp_path, map_location=self.device)
        prompt_state_dict = torch.load(prompt_path, map_location=self.device)
        previous_prompt_state_dict = torch.load(previous_prompt_path, map_location=self.device)
        self.mlps.load_state_dict(mlp_state_dict["mlps"])
        self.prompt.data = prompt_state_dict["prompt"].data
        self.previous_prompts.data = previous_prompt_state_dict["previous_prompt"].data
        print(f"Loaded mlps and prompt from {mlp_path} and {prompt_path}")
            
    def freeze_all(self):
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
        
        inputs_embeds = self.embed_tokens(input_ids) # [batch_size, seq_len, embed_dim]
        k = inputs_embeds.shape[0] # batch_size
    
        mlp = self.mlps[self.task_names.index(self.current_task_name)]
        prompt = mlp(self.prompt)  #[prefix_len, embed_dim]
        
        if self.pp:
            inputs_embeds = torch.cat([prompt.unsqueeze(0).repeat(k, 1, 1), # [batch_size, prefix_len, embed_dim]
                                          self.previous_prompts.unsqueeze(0).repeat(k, 1, 1),# [batch_size, len_of_learned_tasks, embed_dim]
                                          inputs_embeds], axis=1)# [batch_size, seq_len, embed_dim]
            full_prefix_len = prompt.shape[0] + self.previous_prompts.shape[0] # prefix_len + len_of_learned_tasks
        
        source_mask = torch.cat((torch.tensor(1).to('cuda').repeat(k, full_prefix_len),
                                 attention_mask), axis=1) # [batch_size, prefix_len + learned_tasks_len]
        if labels is not None:
            labels = torch.concat((labels[0][0].repeat(k, inputs_embeds.shape[1] - labels.shape[1]), labels),axis=1).detach()#[batch_size, prefix_len + learned_tasks_len, embed_dim]
            return self.model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=source_mask
            )
        else:
            inputs_embeds = inputs_embeds.half()
            return self.model(inputs_embeds=inputs_embeds, attention_mask=source_mask, use_cache=False, return_dict=True)
        
class PPTrainer(BaseTrainerCL):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model:PPModel = PPModel(model=self.model,
                             prefix_len=20,
                             task_names=list(self.continual_training_dataset.keys()))
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = self.model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def continual_learning(self):
        resume_from_checkpoint = "False"
        for name, train_set in self.continual_training_dataset.items():
            self.current_task_name = name
            self.model.current_task_name = name
            self.update_adapter_and_train_set(resume_from_checkpoint, train_set)
            self.model.freeze_mlps(name)
            self.train()
            self.model.progressive_previous_prompt(name)
            resume_from_checkpoint = self.save_model(name)
            self.model.save_mlps_prompts(self.args.output_dir)
        wandb.finish()
    
    def save_model(self, name) -> str:
        if self.args.output_dir is not None:
            output_dir = os.path.join(self.args.output_dir, f"{self.cl_method}_{self.adapter}_checkpoint_{name}")
            assert isinstance(self.model.model, PeftModel), "self.model.model is not a PeftModel"
            self.model.model.save_pretrained(output_dir)
            print(f"save task: {name} adapter to {self.args.output_dir}")
            return output_dir