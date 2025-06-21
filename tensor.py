from typing import Any, Callable, Tuple, Union
import numpy as np

def is_num(x: Any):
    return isinstance(x, int) or isinstance(x, float)

def is_num_or_numpy(x: Any):
    return is_num(x) or isinstance(x, np.ndarray)

def is_num_or_numpy_or_tensor(x: Any):
    return is_num(x) or isinstance(x, np.ndarray) or isinstance(x, Tensor)

def require_num_or_numpy_or_tensor(x: Any):
    if not is_num_or_numpy_or_tensor(x):
        raise ValueError(f"Expected input of type float, int, np.ndarray, or Tensor. Instead got: {x.__class__}")

def _reduce_grad(gradient: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    "Un-broadcast"s a gradient

    Basically just sum-reduces the `gradient` tensor where any of original shape is 1
    aka if gradient is of shape (5, 4, 4) and target_shape is (1, 4, 1) we will sum reduce
    over axes 0 and 2
    """
    offset = gradient.ndim - len(target_shape)
    if offset < 0:
        raise ValueError("gradient has a lower rank than target shape - should never happen")

    if gradient.shape == target_shape:
        # nothing was broadcast, we can return original gradient
        return gradient
    new_dims = tuple([1] * offset + list(target_shape)) # left pad target shape with 1s if it doesn't match
    axes_to_sum = tuple([i for i in range(len(new_dims)) if new_dims[i] == 1])
    summed = np.sum(gradient, axes_to_sum, dtype=np.float64, keepdims=True)
    assert summed.shape == new_dims
    assert summed.shape[:offset] == tuple([1] * offset)
    return summed.reshape(target_shape)

def can_broadcast(x1, x2):
    # x1 and x2 are shapes
    for i in range(1, min(len(x1), len(x2))+1):
        if x1[-i] == x2[-i]:
            continue
        if x1[-i] == 1 or x2[-i] == 1:
            continue
        return False
    return True

def broadcast_tensors(t1: np.array, t2: np.array):
    """
    Note: equivalent to np.broadcast_arrays(t1, t2) but I'm writing it for myself ✨
    """
    if t1.shape == t2.shape:
        # no broadcast needed
        return t1, t2
    if not can_broadcast(t1.shape, t2.shape):
        raise ValueError(f"Cannot broadcast between {t1.shape} and {t2.shape}")
    result_shape = [1] * max(len(t1.shape), len(t2.shape))
    for i in range(1, max(len(t1.shape), len(t2.shape))+1):
        if i > len(t1.shape):
            result_shape[-i] = t2.shape[-i]
        elif i > len(t2.shape):
            result_shape[-i] = t1.shape[-i]
        else:
            result_shape[-i] = max(t1.shape[-i], t2.shape[-i])
    return np.broadcast_to(t1, result_shape), np.broadcast_to(t2, result_shape)

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        if not (isinstance(data, float) or isinstance(data, int) or isinstance(data, list) or isinstance(data, np.ndarray)):
            raise ValueError(f"Value's data must be a float, int, list, or np array, not {data.__class__}")
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.grad = np.zeros_like(data, dtype=np.float64)
        self.label = label

    def __repr__(self):
        return f"Tensor(\n{self.data}\n)"

    # def _expand_dims_for_broadcasting(self, amt_to_repeat: Tuple[int]):
    #     # amt_to_repeat should be a tuple of length len(self.shape)
    #     # says how much to repeat each tensor
    #     # any item in amt_to_repeat > 1 should be a dim of size 1
    #     if len(amt_to_repeat) != len(self.shape):
    #         raise ValueError("Expand dims called incorrectly")
    #     if amt_to_repeat == tuple([1] * len(amt_to_repeat)):
    #         return self # nothing to change
    #     for i, repeat_amt in enumerate(amt_to_repeat):
    #         if repeat_amt != 1:
    #             if self.shape[i] != 1:
    #                 raise ValueError("cannot broadcast a dim of length != 1")
    #     result = self.data.repeat(amt_to_repeat)
    #     out = Tensor(result, (self,), 'expand_dims_for_broadcasting')
    #     def _expand_dims_for_broadcasting_backward():
    #         self.grad += _reduce_grad(out.grad, self.shape)
    #     out._backward = _expand_dims_for_broadcasting_backward()
    #     return out

    @classmethod
    def broadcast_tensors_if_necessary(cls, t1: "Tensor", t2: "Tensor") -> Tuple["Tensor", "Tensor"]:
        if t1.shape == t2.shape:
            return t1, t2
        t1result, t2result = np.broadcast_arrays(t1.data, t2.data)
        out1 = Tensor(t1result.copy(), (t1,), 'broadcast')
        out2 = Tensor(t2result.copy(), (t2,), 'broadcast')
        def _broadcast_backward_generic(in_tensor, out_tensor):
            def _broadcast_backward():
                print('t1', t1.shape, 't2', t2.shape, 'out', out_tensor.shape, 'in', in_tensor.shape, 'reduced', _reduce_grad(out_tensor.grad, in_tensor.shape).shape)
                in_tensor.grad += _reduce_grad(out_tensor.grad, in_tensor.shape)
            return _broadcast_backward
        out1._backward = _broadcast_backward_generic(t1, out1)
        out2._backward = _broadcast_backward_generic(t2, out2)
        return out1, out2

    def __add__(self, other):
        if is_num(other):
            out = Tensor(self.data + other, (self,), '+scalar')
            def _add_scalar_backward():
                self.grad += out.grad
            out._backward = _add_scalar_backward
            return out
        elif not isinstance(other, Tensor):
            raise ValueError(f"Right operand to add must be number or tensor, not {other}")
        b1, b2 = Tensor.broadcast_tensors_if_necessary(self, other)
        out = Tensor(b1.data + b2.data, (b1, b2), '+')
        def _add_backward():
            b1.grad += out.grad
            b2.grad += out.grad
        out._backward = _add_backward
        return out
    
    def __sub__(self, other):
        if is_num(other):
            out = Tensor(self.data - other, (self,), '-scalar')
            def _sub_scalar_backward():
                self.grad += out.grad
            out._backward = _sub_scalar_backward
            return out
        elif not isinstance(other, Tensor):
            raise ValueError(f"Right operand to add must be number or tensor, not {other}")
        b1, b2 = Tensor.broadcast_tensors_if_necessary(self, other)
        out = Tensor(b1.data - b2.data, (b1, b2), '-')
        def _sub_backward():
            b1.grad += 1.0 * out.grad
            b2.grad += -1.0 * out.grad
        out._backward = _sub_backward
        return out

    def __rsub__(self, other):
        if is_num(other):
            out = Tensor(other - self.data, (self,), 'r-scalar')
            def _rsub_scalar_backward():
                self.grad -= out.grad
            out._backward = _rsub_scalar_backward
            return out
        elif not isinstance(other, Tensor):
            raise ValueError(f"Right operand to add must be number or tensor, not {other}")
        b1, b2 = Tensor.broadcast_tensors_if_necessary(self, other)
        out = Tensor(b2.data - b1.data, (b1, b2), 'r-')
        def _rsub_backward():
            b1.grad -= out.grad
            b2.grad += out.grad
        out._backward = _rsub_backward
        return out

    def __mul__(self, other):
        if is_num(other):
            out = Tensor(self.data * other, (self,), '*scalar')
            def _mul_scalar_backward():
                self.grad += other * out.grad
            out._backward = _mul_scalar_backward
            return out
        elif not isinstance(other, Tensor):
            raise ValueError(f"Right operand to add must be number or tensor, not {other}")
        b1, b2 = Tensor.broadcast_tensors_if_necessary(self, other)
        out = Tensor(b1.data * b2.data, (b1, b2), '*')
        def _mul_backward():
            b1.grad += b2.data * out.grad
            b2.grad += b1.data * out.grad
        out._backward = _mul_backward
        return out
    
    def __truediv__(self, other):
        if is_num(other):
            out = Tensor(self.data / other, (self,), '/scalar')
            def _div_scalar_backward():
                self.grad += out.grad / other
            out._backward = _div_scalar_backward
            return out
        elif not isinstance(other, Tensor):
            raise ValueError(f"Right operand to add must be number or tensor, not {other}")
        b1, b2 = Tensor.broadcast_tensors_if_necessary(self, other)
        out = Tensor(b1.data / b2.data, (b1, b2), '/')
        def _div_backward():
            b1.grad += out.grad / b2.data
            b2.grad += -b1.data/(b2.data ** 2) * out.grad
        out._backward = _div_backward
        return out

    def __pow__(self, num):
        if not (isinstance(num, float) or isinstance(num, int)):
            raise ValueError(f"Can only raise Value class to a float or integer power, not {num}")
        result = self.data ** num
        out = Tensor(result, (self,), 'pow')
        def _pow_backward():
            self.grad += (num) * (self.data ** (num-1)) * out.grad
        out._backward = _pow_backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def tanh(self):
        x = self.data
        exp2x = np.exp(2*x)
        t = (exp2x-1)/(exp2x+1)
        out = Tensor(t, (self,), 'tanh')
        def _tanh_backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _tanh_backward
        return out

    def exp(self):
        x = self.data
        out = Tensor(np.exp(x), (self,), 'exp')
        def _exp_backward():
            self.grad += out.data * out.grad
        out._backward = _exp_backward
        return out

    def __matmul__(self, other):
        if is_num(other):
            raise ValueError("Cannot matmul by a scalar")
        elif not isinstance(other, Tensor):
            raise ValueError(f"Right operand to add must be number or tensor, not {other}")

        # ensure each matrix has at least 2 dims
        if len(self.data.shape) < 2:
            raise ValueError(f"Cannot do a matmul where left side operand has < 2 dims, dims: {self.data.shape}")
        if len(other.data.shape) < 2:
            raise ValueError(f"Cannot do a matmul where right side operand has < 2 dims, dims: {other.data.shape}")

        # check the last two dims. notation: n x m matrix @ k x l matrix
        n, m = self.data.shape[-2:]
        k, l = other.data.shape[-2:]
        if m != k:
            raise ValueError(f"Shape mismatch: cannot multiply matrices with shapes {n}x{m} @ {k}x{l}")
        
        # ensure that the batch dims all match
        batch_dims_1 = self.data.shape[:-2]
        batch_dims_2 = other.data.shape[:-2]

        if batch_dims_1 != batch_dims_2:
            raise ValueError("Cannot do matmul if batch dims don't match")

        # G is dim nxl
        # A^T @ G -> mxn @ nxl -> mxl -> B
        # G @ B^T -> nxl @ lxk -> nxk -> A

        out = Tensor(self.data @ other.data, (self, other), '@')
        def _matmul_backward():
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad
        out._backward = _matmul_backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def sum(self):
        summed_result = np.sum(self.data)
        out = Tensor(summed_result, (self,), 'sum')
        def _sum_backward():
            self.grad += out.grad
        out._backward = _sum_backward
        return out
    
    def backward(self):
        if self.shape != (1,) and self.shape != ():
            raise ValueError("Can only call .backward on scalars")
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if self.shape == (1,):
            self.grad = np.array([1.0])
        else:
            self.grad = np.array(1.0)
        for node in reversed(topo):
            node._backward()

    @property
    def shape(self):
        return self.data.shape

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