import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from qpth.qp import QPFunction
from .BaseTrainerCL import BaseTrainerCL
from peft import PeftModel
from deepspeed.utils import safe_get_full_grad
import torch.distributed as dist
class TaskGradientCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.model:PeftModel = kwargs.get('model')
        self.current_task_name = kwargs.get('current_task_name')  # need update during training
        self.n_tasks = kwargs.get('n_tasks')
        self.grads = {}
        self.task_names = kwargs.get('task_names')
        self.device = kwargs.get('device', 'cpu')
        self.init_grads()
    
    def init_grads(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: # reduce memory usage
                self.grads[n] = torch.ones([p.data.numel(), self.n_tasks], dtype=p.dtype, device=self.device)
            
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for n, p in self.model.named_parameters():
            if n in self.grads:
                # p_grad = safe_get_full_grad(p)
                # print('rank', dist.get_rank(), n, '->', p.device)
                # assert p.grad is not None, f"parameter {n} has no gradient"
                if p.grad is None:
                    print(f"rank {dist.get_rank()} parameter {n} has no gradient in device {p.device}")
                p.grad = self.get_updated_grads(n, p.grad, self.task_names.index(self.current_task_name))
                grad_old = self.grads[n][:, self.task_names.index(self.current_task_name)].detach().clone()
                grad_new = (grad_old * state.step + p.grad.detach().clone().view(-1)) / (state.step + 1)
                self.grads[n][:, self.task_names.index(self.current_task_name)] = grad_new
    
    def get_updated_grads(self, name, grad, idx, margin=0.1, eps=1.0):
        ori_shape = grad.shape # None
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
            n_tasks=self.num_tasks,
            task_names=self.task_names))
        
        self.gem_cb = None
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, TaskGradientCallback):
                self.gem_cb = cb
                break
            
    def continual_learning(self):
        for i, name in enumerate(self.task_names):
            self.gem_cb.update_current_task_name(name)
            self.before_task_start(name)
            self.train()
            self.after_task_end(name)
