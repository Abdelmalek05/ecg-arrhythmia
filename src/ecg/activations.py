# this file has the activation functions and their derivatives
# an activation is what makes the network able to learn curves, not only straight lines

import numpy as np

HIDDEN = ("relu", "tanh")
OUTPUT = ("sigmoid", "softmax")


def sigmoid(Z):
    # squeezes any number into the range 0 to 1
    # we do not write 1/(1+exp(-Z)) directly because exp(-Z) blows up when Z is very negative
    out = np.empty_like(Z, dtype=np.float64)
    pos = Z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    exp_z = np.exp(Z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def relu(Z):
    # keeps positive numbers, turns negative numbers into 0
    return np.maximum(0.0, Z)


def tanh(Z):
    # like sigmoid but the range is -1 to 1
    return np.tanh(Z)


def softmax(Z, axis=0):
    # turns a column of scores into probabilities that add up to 1
    # we take away the max first, otherwise exp() can overflow
    z_max = np.max(Z, axis=axis, keepdims=True)
    exp_z = np.exp(Z - z_max)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def log_softmax(Z, axis=0):
    # log of softmax, done in a safe way
    # never write np.log(softmax(Z)), it gives -inf when a probability is 0
    z_max = np.max(Z, axis=axis, keepdims=True)
    shifted = Z - z_max
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def activation(Z, name):
    """Apply one activation to Z.

    Z: array of shape (n_layer, m)
    name: "relu", "tanh", "sigmoid" or "softmax"
    """
    if name == "relu":
        return relu(Z)
    if name == "tanh":
        return tanh(Z)
    if name == "sigmoid":
        return sigmoid(Z)
    if name == "softmax":
        return softmax(Z, axis=0)
    raise ValueError("unknown activation " + str(name))


def activation_backward(dA, Z, name):
    """Turn dA into dZ, that is dZ = dA * g'(Z).

    This is only for the hidden layers. For the last layer we do not use this,
    because softmax with cross entropy (and sigmoid with cross entropy) give a
    much simpler result. Look at losses.loss_backward for that.
    """
    if name == "relu":
        # the slope is 1 where Z > 0 and 0 where Z <= 0
        return dA * (Z > 0)
    if name == "tanh":
        a = np.tanh(Z)
        return dA * (1.0 - a * a)
    if name == "sigmoid":
        a = sigmoid(Z)
        return dA * a * (1.0 - a)
    raise ValueError("no backward for activation " + str(name))
