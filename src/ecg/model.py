# this file does the forward pass and the backward pass for a network with any depth
# it is a loop over the layers, so [254, 1] and [254, 64, 32, 1] use the same code
# X has shape (n_features, m), one column is one example, like in the course

import numpy as np

from .activations import activation, activation_backward
from .losses import loss_backward, l2_gradient


def number_of_layers(parameters):
    # parameters holds W1, b1, W2, b2, ... so we count the W keys
    count = 0
    for key in parameters:
        if key.startswith("W"):
            count = count + 1
    return count


def forward(X, parameters, hidden_activation="relu", output_activation="sigmoid",
            keep_prob=1.0, batch_norm=False, training=True, seed=None):
    """Run the network from the input to the output.

    returns: AL, cache
      AL is the output, shape (n_out, m)
      cache holds A, Z and the dropout masks, backward needs them

    Dropout only happens on the hidden layers, and only while training.
    We must keep the masks, because backward has to use the same ones.
    """
    if batch_norm:
        raise NotImplementedError("batch norm is Phase 4")

    L = number_of_layers(parameters)
    rng = np.random.default_rng(seed)

    A = X
    cache = {"A0": X, "masks": {}}

    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A) + b

        if l == L:
            name = output_activation
        else:
            name = hidden_activation
        A = activation(Z, name)

        # dropout: switch off some units, then scale up the rest so the
        # average stays the same (this is called inverted dropout)
        if l < L and keep_prob < 1.0 and training:
            mask = (rng.random(A.shape) < keep_prob).astype(A.dtype)
            A = A * mask / keep_prob
            cache["masks"][l] = mask

        cache["Z" + str(l)] = Z
        cache["A" + str(l)] = A

    AL = A
    return AL, cache


def backward(AL, Y, cache, parameters, hidden_activation="relu",
             output_activation="sigmoid", l2=0.0, keep_prob=1.0,
             batch_norm=False, class_weights=None, loss_name=None):
    """Go backwards and compute the gradient for every W and b.

    returns: dict like {"dW1": ..., "db1": ..., "dW2": ...}
    """
    if batch_norm:
        raise NotImplementedError("batch norm is Phase 4")

    L = number_of_layers(parameters)
    m = Y.shape[1]
    grads = {}

    if loss_name is None:
        if output_activation == "softmax":
            loss_name = "categorical_crossentropy"
        else:
            loss_name = "binary_crossentropy"

    # the last layer is special: sigmoid (or softmax) with cross entropy
    # gives dZL = (AL - Y) / m, so we start from there
    dZ = loss_backward(AL, Y, loss_name, class_weights)

    for l in range(L, 0, -1):
        A_prev = cache["A" + str(l - 1)]
        W = parameters["W" + str(l)]

        grads["dW" + str(l)] = np.dot(dZ, A_prev.T) + l2_gradient(W, l2, m)
        grads["db" + str(l)] = np.sum(dZ, axis=1, keepdims=True)

        if l > 1:
            dA_prev = np.dot(W.T, dZ)
            # dropout used the same mask in forward, so we reuse it here
            if keep_prob < 1.0 and (l - 1) in cache["masks"]:
                dA_prev = dA_prev * cache["masks"][l - 1] / keep_prob
            dZ = activation_backward(dA_prev, cache["Z" + str(l - 1)], hidden_activation)

    return grads


def predict_proba(X, parameters, hidden_activation="relu", output_activation="sigmoid"):
    # forward with training off, so dropout does nothing
    AL, _ = forward(X, parameters, hidden_activation, output_activation,
                    keep_prob=1.0, training=False)
    return AL


def predict(X, parameters, hidden_activation="relu", output_activation="sigmoid",
            threshold=0.5):
    """Give the predicted class for every example, as a flat array of ints."""
    AL = predict_proba(X, parameters, hidden_activation, output_activation)
    if AL.shape[0] == 1:
        return (AL > threshold).astype(np.int64).ravel()
    return np.argmax(AL, axis=0).astype(np.int64)
