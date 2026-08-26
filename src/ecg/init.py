# this file makes the first weights of the network
# you can pick how the weights start: zeros, random, xavier or he

import numpy as np

METHODS = ("zeros", "random", "xavier", "he")


def scale_for(method, n_prev, n_curr):
    # the number we multiply the random weights by
    # a big scale makes the signal explode, a small one makes it vanish
    if method == "zeros":
        return 0.0
    if method == "random":
        return 0.01
    if method == "xavier":
        return np.sqrt(1.0 / n_prev)
    if method == "he":
        return np.sqrt(2.0 / n_prev)
    raise ValueError("unknown init " + str(method))


def initialize_parameters(layer_dims, method="he", seed=None):
    """Make the W and b for every layer.

    layer_dims: list of layer sizes, for example [254, 64, 32, 1]
    method: one of METHODS
    returns: dict like {"W1": ..., "b1": ..., "W2": ..., ...}

    W1 has shape (layer_dims[1], layer_dims[0]) and b1 has shape (layer_dims[1], 1).
    b is always zeros. Only W needs to be different, that is enough to break symmetry.
    """
    if method not in METHODS:
        raise ValueError("unknown init " + str(method) + ", expected one of " + str(METHODS))

    rng = np.random.default_rng(seed)
    parameters = {}
    n_layers = len(layer_dims) - 1

    for l in range(1, n_layers + 1):
        n_prev = layer_dims[l - 1]
        n_curr = layer_dims[l]
        scale = scale_for(method, n_prev, n_curr)
        parameters["W" + str(l)] = rng.standard_normal((n_curr, n_prev)) * scale
        parameters["b" + str(l)] = np.zeros((n_curr, 1))

    return parameters


def count_parameters(parameters):
    # how many numbers the model has to learn
    total = 0
    for key in parameters:
        total = total + parameters[key].size
    return total
