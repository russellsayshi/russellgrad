from typing import List, Optional, Tuple
import numpy as np
from tensor import Tensor

ALLOWED_LR_SCHEDULES = [
    'constant',
    'exponential_falloff',
    'cosine',
    'cosine_with_warmup',
]

class AdamOptimizer:
    def __init__(self,
                 params: List[Tensor],
                 lr=1e-3,
                 lr_schedule='cosine_with_warmup',
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 gradient_clip_value: Optional[float]=5,
                 exponential_falloff_constant: Optional[float] = None, 
                 min_lr: Optional[float] = None,
                 total_steps: Optional[int] = None,
                 warmup_steps: int = 0):
        self.params = params
        self.base_lr = lr  # Store initial learning rate
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gradient_clip_value = gradient_clip_value
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
        if self.lr_schedule == "exponential_falloff" and self.exponential_falloff_constant is None:
            raise ValueError("exponential_falloff_constant must be set")

    
        self.momentum_estimates = {id(p): np.zeros_like(p.grad) for p in self.params}
        self.variance_estimates = {id(p): np.zeros_like(p.grad) for p in self.params}

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

        updates: List[Tuple[Tensor, np.ndarray]] = [] # tuple of (param, update)
        for param in self.params:
            g = param.grad
            pid = id(param)
            if pid not in self.momentum_estimates or pid not in self.variance_estimates:
                raise ValueError("adam got a parameter we haven't seen before")

            # update the values
            moment_estimate = self.beta1 * self.momentum_estimates[pid] + (1 - self.beta1) * param.grad
            variance_estimate = self.beta2 * self.variance_estimates[pid] + (1 - self.beta2) * (param.grad ** 2)
            self.momentum_estimates[pid] = moment_estimate
            self.variance_estimates[pid] = variance_estimate

            moment_estimate_bias_corrected = moment_estimate/(1-(self.beta1 ** (self.global_step+1)))
            variance_estimate_bias_corrected = variance_estimate/(1-(self.beta2 ** (self.global_step+1)))

            update = moment_estimate_bias_corrected/(np.sqrt(variance_estimate_bias_corrected) + self.epsilon)

            updates.append((param, update))

        # NOTE: here we are implementing **update clipping** instead of strictly gradient clipping
        # this is because we first compute all the adam values on the raw, unclipped data
        # THEN do the gradient clipping
        # this differs from how pytorch does it
        # but matches openai's pretraining config
        # neither option is wrong

        if self.gradient_clip_value is None:
            clip_coef = 1 # no update clipping
        else:
            # update clipping
            total_norm = 0
            for _, update in updates:
                param_norm = np.linalg.norm(update)
                total_norm += param_norm ** 2
            total_norm = np.sqrt(total_norm)

            clip_value = self.gradient_clip_value
            clip_coef = clip_value / max(clip_value, total_norm)

        # Gradient descent with clipping
        for param, update in updates:
            param.data -= self.lr * update * clip_coef
            
        self.global_step += 1

################################################################################
###################################### TEST ####################################
################################################################################


if __name__ == "__main__":
    import torch, numpy as np
    from copy import deepcopy

    print("Testing Adam optimizer against PyTorch...")
    
    w_torch = torch.nn.Parameter(torch.randn(3, 3))
    w_np    = Tensor(w_torch.detach().numpy().copy())   # your tensor class

    opt_torch = torch.optim.Adam([w_torch], lr=1e-3)
    opt_np    = AdamOptimizer([w_np], lr=1e-3, gradient_clip_value=5, lr_schedule='constant')

    max_diff = 0.0
    for step in range(200):
        loss_t = (w_torch ** 2).sum()
        loss_t.backward()
        opt_torch.step()
        opt_torch.zero_grad()
        
        w_np.grad = 2 * w_np.data            # same analytic grad
        opt_np.step()
        opt_np.zero_grad()

        diff = np.abs(w_np.data - w_torch.detach().numpy()).max()
        max_diff = max(max_diff, diff)
        
        if step % 50 == 49:
            print(f"Step {step}: max difference = {diff:.2e}")
        
        assert np.allclose(w_np.data, w_torch.detach().numpy(), atol=1e-5), f"Failed at step {step}"
    
    print(f"\nTest PASSED! Maximum difference over all steps: {max_diff:.2e}")
    print("Adam optimizer implementation matches PyTorch within tolerance.")
