import numpy as np
from tensor import Tensor

class Linear:
    def __init__(self, in_dim, out_dim):
        self.weights = Tensor(np.random.normal(0, 0.02, (out_dim, in_dim)))
        self.biases = Tensor(np.zeros((out_dim,), dtype=np.float64)[:, None])

    def forward(self, x):
        result = self.weights @ x
        result += self.biases
        return result

if __name__ == "__main__":
    # Test 1: Basic forward/backward
    layer1 = Linear(3, 2)
    x1 = Tensor([[1.0], [2.0], [3.0]])
    out1 = layer1.forward(x1)
    loss1 = out1.sum()
    loss1.backward()
    
    # PyTorch implementation of test 1
    import torch
    layer1_torch = torch.nn.Linear(3, 2)
    with torch.no_grad():
        layer1_torch.weight.copy_(torch.tensor(layer1.weights.data))
        layer1_torch.bias.copy_(torch.tensor(layer1.biases.data.flatten()))
    x1_torch = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    out1_torch = layer1_torch(x1_torch.T).T
    loss1_torch = out1_torch.sum()
    loss1_torch.backward()

    print("Test 1 - Basic forward/backward")
    print("RussellGrad vs PyTorch gradients:")
    print("weights grad diff:", np.abs(layer1.weights.grad - layer1_torch.weight.grad.numpy()).max())
    print("biases grad diff:", np.abs(layer1.biases.grad.flatten() - layer1_torch.bias.grad.numpy()).max())
    print("x grad diff:", np.abs(x1.grad - x1_torch.grad.numpy()).max())
    print("grad norm:", np.linalg.norm(layer1.weights.grad), np.linalg.norm(layer1_torch.weight.grad.numpy()))
    print()

    # Test 2: Multiple layers with batch input
    layer2a = Linear(4, 3)
    layer2b = Linear(3, 2) 
    x2 = Tensor(np.random.randn(4, 5))  # batch of 5 inputs
    
    out2 = layer2a.forward(x2)
    out2 = layer2b.forward(out2)
    loss2 = out2.sum()
    loss2.backward()

    # PyTorch implementation of test 2
    layer2a_torch = torch.nn.Linear(4, 3, dtype=torch.float64)
    layer2b_torch = torch.nn.Linear(3, 2, dtype=torch.float64)
    with torch.no_grad():
        layer2a_torch.weight.copy_(torch.tensor(layer2a.weights.data))
        layer2a_torch.bias.copy_(torch.tensor(layer2a.biases.data.flatten()))
        layer2b_torch.weight.copy_(torch.tensor(layer2b.weights.data))
        layer2b_torch.bias.copy_(torch.tensor(layer2b.biases.data.flatten()))
    x2_torch = torch.tensor(x2.data, dtype=torch.float64, requires_grad=True)
    out2_torch = layer2a_torch(x2_torch.T).T
    out2_torch = layer2b_torch(out2_torch.T).T
    loss2_torch = out2_torch.sum()
    loss2_torch.backward()

    print("Test 2 - Multiple layers with batch input")
    print("RussellGrad vs PyTorch gradients:")
    print("layer2a weights grad diff:", np.abs(layer2a.weights.grad - layer2a_torch.weight.grad.numpy()).max())
    print("layer2b weights grad diff:", np.abs(layer2b.weights.grad - layer2b_torch.weight.grad.numpy()).max())
    print("x grad diff:", np.abs(x2.grad - x2_torch.grad.numpy()).max())