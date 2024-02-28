import random
from copy import deepcopy
from typing import Dict, List
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datasets import Dataset
from qpth.qp import QPFunction
from .BaseTrainerCL import BaseTrainerCL
from peft import PeftModel
import re
from memory_profiler import profile
import time

def format_name(name):
    index = name.rfind("model")
    return name[index:]

class TaskGradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.model:PeftModel = kwargs.get('model')
        self.current_task_name = kwargs.get('current_task_name')  # need update during training
        self.n_tasks = kwargs.get('n_tasks')
        self.grads = {}
        self.task_names = kwargs.get('task_names')
        self.init_grads()
    
    def init_grads(self):
        """
            {param_name: torch.zeros([param_size, n_tasks], dtype=torch.bfloat16)}
        """
        self.model.print_trainable_parameters()
        print(self.n_tasks)
        for n, p in self.model.named_parameters():
            if p.requires_grad: # reduce memory usage
                self.grads[format_name(n)] = torch.ones([p.data.numel(), self.n_tasks], dtype=torch.bfloat16).to('cpu')
            
    def store_grads(self): 
        for n, p in self.model.named_parameters():
            if format(n) in self.grads.keys():
                self.grads[format_name(n)][:, self.task_names.index(self.current_task_name)] = p.grad.detach().clone().view(-1).to(
                    'cpu')
                print(f"store {format_name(n)} grad")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            update grads with projected grads if the dot product of current grads and previous grads is negative
        """
        for n, p in self.model.named_parameters():
            if format_name(n) in self.grads.keys() and p.requires_grad:
                p.grad = self.get_updated_grads(format_name(n), p.grad, self.task_names.index(self.current_task_name))
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
            store current task grads
        """
        self.store_grads()

    def get_updated_grads(self, name, grad, idx, margin=0.1, eps=1.0):
        """
            name: param name
            grad: current param grad
            idx: current task index
        """
        ori_shape = grad.shape
        grad = grad.view(-1)
        pre_grad = self.grads[name].cuda()[:, :idx+1].to(torch.float32)
        dot_product = torch.mm(grad.unsqueeze(0), pre_grad)
        if (dot_product < 0).sum() != 0:
            pre_grad_cuda = pre_grad.t()
            grad_cuda = grad.contiguous().view(-1)
            t = pre_grad_cuda.shape[0]
            P = torch.matmul(pre_grad_cuda, pre_grad_cuda.t())
            P = 0.5 * (P + P.t())
            
            P[torch.isnan(P)] = 0.0
            eigenvalues = torch.linalg.eigvals(P)
            if not (eigenvalues.real > 0).all(): # due to the grad clip happens after the projection, the grad could be huge, refactor eps=1.0 is reasonable
                P += torch.eye(t).cuda() * eps
            
            q = torch.matmul(pre_grad_cuda, grad_cuda).t() * -1

            P = P.to(torch.float32)
            q = q.to(torch.float32)
            G = torch.eye(t).cuda() * -1
            h = torch.zeros(t).cuda() - margin
            e = torch.Tensor().cuda()
            v = QPFunction(verbose=False)(P, q, G, h, e, e)[0]
            v = v.to(torch.float32)
            x = torch.matmul(v, pre_grad_cuda) + grad_cuda
            grad.copy_(x.view(-1))
        return grad.view(ori_shape)

    def update_current_task_name(self, name:str):
        self.current_task_name = name



class GEMTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_callback(TaskGradientCallback(
            model = self.model,
            current_task_name=self.current_task_name,
            n_tasks=self.n_tasks,
            task_names=self.task_names))
        
        self.gem_cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, TaskGradientCallback):
                self.gem_cb = cb
                break
            
    def continual_learning(self):
        self.model.print_trainable_parameters()
        resume_from_checkpoint = "False"
        for name, train_set in self.continual_training_dataset.items():
            self.current_task_name = name
            self.gem_cb.update_current_task_name(name)
            self.update_adapter_and_train_set(resume_from_checkpoint, train_set)
            self.train()
            resume_from_checkpoint = self.save_model(name)
