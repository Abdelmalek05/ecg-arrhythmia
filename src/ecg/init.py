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


def number_of_layers(parameters):
    # parameters holds W1, b1, W2, b2, ... so we count the W keys
    count = 0
    for key in parameters:
        if key.startswith("W"):
            count = count + 1
    return count


def parameter_keys(parameters):
    """Every learnable key, in one fixed order: W1, b1, gamma1, beta1, W2, ...

    Every module that walks the parameters must use this, so that adding a new
    kind of parameter (like batch norm's gamma and beta) only changes one place.
    """
    keys = []
    for l in range(1, number_of_layers(parameters) + 1):
        for letter in ["W", "b", "gamma", "beta"]:
            key = letter + str(l)
            if key in parameters:
                keys.append(key)
    return keys


def initialize_parameters(layer_dims, method="he", seed=None, batch_norm=False):
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

        # batch norm learns a scale and a shift for every hidden layer.
        # gamma starts at 1 and beta at 0, so at the start batch norm only
        # normalises and does not change anything else.
        if batch_norm and l < n_layers:
            parameters["gamma" + str(l)] = np.ones((n_curr, 1))
            parameters["beta" + str(l)] = np.zeros((n_curr, 1))

    return parameters


def count_parameters(parameters):
    # how many numbers the model has to learn
    total = 0
    for key in parameters:
        total = total + parameters[key].size
    return total
