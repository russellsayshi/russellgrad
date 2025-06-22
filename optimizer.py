from typing import List
import numpy as np
from tensor import Tensor

class SGDOptimizer:
    def __init__(self, params: List[Tensor], lr=1e-5):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.grad)

    def step(self):
        # gradient descent
        for param in self.params:
            param.data -= self.lr * param.grad