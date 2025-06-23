from typing import List, Optional
import numpy as np
from tensor import Tensor

ALLOWED_LR_SCHEDULES = [
    'constant',
    'exponential_falloff',
    'cosine',
    'cosine_with_warmup',
]

class SGDOptimizer:
    def __init__(self, params: List[Tensor], lr=1e-5, lr_schedule='constant', 
                 exponential_falloff_constant: Optional[float] = None, 
                 min_lr: Optional[float] = None,
                 total_steps: Optional[int] = None,
                 warmup_steps: int = 0):
        self.params = params
        self.base_lr = lr  # Store initial learning rate
        self.lr = lr
        if lr_schedule not in ALLOWED_LR_SCHEDULES:
            raise ValueError("LR schedule must be one of: " + ",".join(ALLOWED_LR_SCHEDULES))
        self.lr_schedule = lr_schedule
        self.exponential_falloff_constant = exponential_falloff_constant
        self.global_step = 0
        self.min_lr = min_lr if min_lr is not None else lr * 0.01  # Default to 1% of initial LR
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
        # Validate parameters for cosine schedules
        if 'cosine' in lr_schedule and total_steps is None:
            raise ValueError("total_steps must be provided for cosine schedules")

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.grad)

    def get_lr(self):
        """Calculate learning rate based on schedule"""
        if self.lr_schedule == 'constant':
            return self.base_lr
            
        elif self.lr_schedule == 'exponential_falloff':
            lr = self.base_lr * (self.exponential_falloff_constant ** self.global_step)
            return max(lr, self.min_lr)
            
        elif self.lr_schedule == 'cosine':
            # Cosine annealing without warmup
            progress = min(self.global_step / self.total_steps, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            return lr
            
        elif self.lr_schedule == 'cosine_with_warmup':
            if self.global_step < self.warmup_steps:
                # Linear warmup - start from 10% of base_lr instead of 0
                warmup_progress = self.global_step / self.warmup_steps
                lr = self.min_lr + (self.base_lr - self.min_lr) * warmup_progress
            else:
                # Cosine annealing after warmup
                progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            return lr
            
        return self.base_lr

    def step(self):
        # Update learning rate based on schedule
        self.lr = self.get_lr()
        
        # Gradient clipping
        total_norm = 0
        for param in self.params:
            param_norm = np.linalg.norm(param.grad)
            total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)

        clip_value = 40.0
        clip_coef = clip_value / max(clip_value, total_norm)

        # Gradient descent with clipping
        for param in self.params:
            param.data -= self.lr * param.grad * clip_coef
            
        self.global_step += 1