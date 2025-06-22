from tensor import Tensor

def softmax(x: Tensor, axis=-1): # TODO: remember this
    e_x = (x - x.max(axis=axis, keepdims=True)).exp()
    return e_x / e_x.sum(axis=axis, keepdims=True)