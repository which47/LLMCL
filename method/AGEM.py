import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .BaseTrainerCL import BaseTrainerCL
from peft import PeftModel


class AveTaskGradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.model:PeftModel = kwargs.get('model')
        self.current_task_name = kwargs.get('current_task_name')  # need update during training
        self.n_tasks = kwargs.get('n_tasks')
        self.grads = {}
        self.task_names = kwargs.get('task_names')
        self.init_grads()
    
    def init_grads(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.grads[n] = torch.ones([p.data.numel()], dtype=p.dtype, device=p.device)
            
    def store_grads(self): 
        for n, p in self.model.named_parameters():
            if n in self.grads:
                self.ave_grads(n, p.grad.detach().clone().view(-1))
                
    def ave_grads(self, name, new_grads):
        self.grads[name] = (self.grads[name] * (self.task_names.index(self.current_task_name) + 1) + new_grads) / (self.task_names.index(self.current_task_name) + 2)
            
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for n, p in self.model.named_parameters():
            if n in self.grads and p.requires_grad:
                p.grad = self.get_updated_grads(n, p.grad)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.store_grads()

    def get_updated_grads(self, name, grad, eps=1e-4):
        ori_shape = grad.shape
        grad = grad.view(-1)
        pre_grad = self.grads[name].cuda().to(torch.float32)
        grad, pre_grad = grad.unsqueeze(1), pre_grad.unsqueeze(1)
        dot_product = torch.mm(grad.t(), pre_grad)
        
        if (dot_product < 0) != 0:
            new_grad = grad - (torch.mm(grad.t(), pre_grad) + eps) / (torch.mm(pre_grad.t(), pre_grad) + eps) * pre_grad
            grad.copy_(new_grad)
            
        return grad.view(ori_shape)

    def update_current_task_name(self, name:str):
        self.current_task_name = name



class AveGEMTrainer(BaseTrainerCL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_callback(AveTaskGradientCallback(
            model = self.model,
            current_task_name=self.current_task_name,
            n_tasks=self.num_tasks,
            task_names=self.task_names))
        
        self.gem_cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, AveTaskGradientCallback):
                self.gem_cb = cb
                break
            
    def continual_learning(self):
        for i, name in enumerate(self.task_names):
            self.gem_cb.update_current_task_name(name)
            self.before_task_start(name)
            self.train()
            self.after_task_end(name)