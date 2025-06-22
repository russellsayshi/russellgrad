import numpy as np
from tensor import Tensor
from softmax import softmax

def cross_entropy_loss(predicted: Tensor, correct_indices: np.ndarray, pad_token_idx: int) -> Tensor:
    # predicted: (batch x seq_len x vocab_size)
    # correct_indices: (batch x seq_len)
    assert len(predicted.shape) == 3
    assert len(correct_indices.shape) == 2
    assert predicted.shape[0:2] == correct_indices.shape[0:2]

    batch_indices = np.arange(correct_indices.shape[0])[:, None]
    seq_indices = np.arange(correct_indices.shape[1])[None, :]

    mask = correct_indices != pad_token_idx

    # set predictions for pad token to -1e9
    # pad token is the last one in the transformer
    # this is so it doesnt contribute to the softmax calc
    pad_predictions = np.zeros_like(predicted.data)
    pad_predictions[:,:,-1] = 1
    predicted = predicted.masked_fill(pad_predictions == 1, -1e9)

    # now we can do softmax
    softmax_output = softmax(predicted, axis=-1)

    predicted_softmax_values = softmax_output[batch_indices, seq_indices, correct_indices] # (batch x seq_len)

    # only select ones that arent predicting pad token
    good_to_log = predicted_softmax_values[mask]

    return (-good_to_log.log()).mean()