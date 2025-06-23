from typing import List, Optional
import numpy as np
from tensor import Tensor

ALLOWED_LR_SCHEDULES = [
    'constant',
    'exponential_falloff',
]

class SGDOptimizer:
    def __init__(self, params: List[Tensor], lr=1e-5, lr_schedule='constant', exponential_falloff_constant: Optional[int] = None, min_lr: Optional[float] = None):
        self.params = params
        self.lr = lr
        if lr_schedule not in ALLOWED_LR_SCHEDULES:
            raise ValueError("LR schedule must be one of: " + ",".join(ALLOWED_LR_SCHEDULES))
        self.lr_schedule = lr_schedule
        self.exponential_falloff_constant = exponential_falloff_constant
        self.global_step = 0
        self.min_lr = min_lr

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.grad)

    def step(self):
        # gradient descent
        for param in self.params:
            param.data -= self.lr * param.grad
        self.global_step += 1
        if self.lr_schedule == 'exponential_falloff':
            self.lr *= self.exponential_falloff_constant
            if self.min_lr:
                self.lr = max(self.lr, self.min_lr)