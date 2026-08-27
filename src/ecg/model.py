# this file does the forward pass and the backward pass for a network with any depth
# it is a loop over the layers, so [254, 1] and [254, 64, 32, 1] use the same code
# X has shape (n_features, m), one column is one example, like in the course

import numpy as np

from .init import number_of_layers, parameter_keys
from .activations import activation, activation_backward
from .losses import loss_backward, l2_gradient

BN_EPSILON = 1e-8
BN_MOMENTUM = 0.9      # how much of the old running average we keep


def new_bn_state(parameters):
    """The running mean and variance that batch norm uses at prediction time.

    While training we normalise with the mean and variance OF THE BATCH. At
    prediction time there may be only one example, and a single example has no
    variance, so we keep a slow running average during training and use that.
    """
    state = {}
    for l in range(1, number_of_layers(parameters)):
        if "gamma" + str(l) in parameters:
            n = parameters["gamma" + str(l)].shape[0]
            state["mean" + str(l)] = np.zeros((n, 1))
            state["var" + str(l)] = np.ones((n, 1))
    return state


def forward(X, parameters, hidden_activation="relu", output_activation="sigmoid",
            keep_prob=1.0, batch_norm=False, training=True, seed=None, bn_state=None):
    """Run the network from the input to the output.

    returns: AL, cache
      AL is the output, shape (n_out, m)
      cache holds A, Z and the dropout masks, backward needs them

    Dropout only happens on the hidden layers, and only while training.
    We must keep the masks, because backward has to use the same ones.
    """
    L = number_of_layers(parameters)
    rng = np.random.default_rng(seed)

    A = X
    cache = {"A0": X, "masks": {}, "bn": {}}

    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A) + b

        # batch norm on the hidden layers: make every unit mean 0 and std 1
        # across the batch, then let the network scale and shift it back with
        # gamma and beta if it wants to.
        if batch_norm and l < L and ("gamma" + str(l)) in parameters:
            gamma = parameters["gamma" + str(l)]
            beta = parameters["beta" + str(l)]
            if training:
                mu = np.mean(Z, axis=1, keepdims=True)
                var = np.var(Z, axis=1, keepdims=True)
                if bn_state is not None:
                    key_m, key_v = "mean" + str(l), "var" + str(l)
                    bn_state[key_m] = BN_MOMENTUM * bn_state[key_m] + (1 - BN_MOMENTUM) * mu
                    bn_state[key_v] = BN_MOMENTUM * bn_state[key_v] + (1 - BN_MOMENTUM) * var
            else:
                if bn_state is None:
                    raise ValueError("batch norm needs bn_state when training is False")
                mu = bn_state["mean" + str(l)]
                var = bn_state["var" + str(l)]

            std = np.sqrt(var + BN_EPSILON)
            Z_hat = (Z - mu) / std
            Z_bn = gamma * Z_hat + beta

            cache["bn"][l] = {"Z": Z, "Z_hat": Z_hat, "mu": mu, "var": var,
                              "std": std, "gamma": gamma}
            Z_for_activation = Z_bn
        else:
            Z_for_activation = Z

        if l == L:
            name = output_activation
        else:
            name = hidden_activation
        A = activation(Z_for_activation, name)

        # dropout: switch off some units, then scale up the rest so the
        # average stays the same (this is called inverted dropout)
        if l < L and keep_prob < 1.0 and training:
            mask = (rng.random(A.shape) < keep_prob).astype(A.dtype)
            A = A * mask / keep_prob
            cache["masks"][l] = mask

        cache["Z" + str(l)] = Z_for_activation
        cache["Zpre" + str(l)] = Z
        cache["A" + str(l)] = A

    AL = A
    return AL, cache


def batch_norm_backward(dZ_bn, bn_cache):
    """Send the gradient back through the batch norm step.

    This is the hardest derivative in the project, because mu and var are both
    made FROM the batch. So changing one Z changes the mean, which changes every
    other normalised value too. That is why there are three terms below and not
    one.

    dZ_bn : gradient with respect to the OUTPUT of batch norm
    returns: gradient with respect to the INPUT of batch norm, plus dgamma, dbeta
    """
    Z = bn_cache["Z"]
    Z_hat = bn_cache["Z_hat"]
    mu = bn_cache["mu"]
    std = bn_cache["std"]
    gamma = bn_cache["gamma"]
    m = Z.shape[1]

    # the easy two: gamma and beta are just a scale and a shift
    dgamma = np.sum(dZ_bn * Z_hat, axis=1, keepdims=True)
    dbeta = np.sum(dZ_bn, axis=1, keepdims=True)

    # gradient with respect to the normalised value
    dZ_hat = dZ_bn * gamma

    centered = Z - mu

    # term 1: Z affects Z_hat directly
    # term 2: Z affects the variance, which affects every Z_hat
    # term 3: Z affects the mean, which affects every Z_hat
    dvar = np.sum(dZ_hat * centered, axis=1, keepdims=True) * (-0.5) / (std ** 3)
    dmu = (np.sum(dZ_hat, axis=1, keepdims=True) * (-1.0) / std
           + dvar * np.sum(-2.0 * centered, axis=1, keepdims=True) / m)

    dZ = dZ_hat / std + dvar * 2.0 * centered / m + dmu / m

    return dZ, dgamma, dbeta


def backward(AL, Y, cache, parameters, hidden_activation="relu",
             output_activation="sigmoid", l2=0.0, keep_prob=1.0,
             batch_norm=False, class_weights=None, loss_name=None):
    """Go backwards and compute the gradient for every W and b.

    returns: dict like {"dW1": ..., "db1": ..., "dW2": ...}
    """
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
        # right now dZ is the gradient at the OUTPUT of batch norm.
        # push it back through batch norm first, so that dW and db use the
        # gradient at W*A + b, which is what they actually multiply.
        if batch_norm and l in cache.get("bn", {}):
            dZ, dgamma, dbeta = batch_norm_backward(dZ, cache["bn"][l])
            grads["dgamma" + str(l)] = dgamma
            grads["dbeta" + str(l)] = dbeta

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


def predict_proba(X, parameters, hidden_activation="relu", output_activation="sigmoid",
                  batch_norm=False, bn_state=None):
    # forward with training off, so dropout does nothing and batch norm uses
    # the running mean and variance
    AL, _ = forward(X, parameters, hidden_activation, output_activation,
                    keep_prob=1.0, batch_norm=batch_norm, training=False,
                    bn_state=bn_state)
    return AL


def predict(X, parameters, hidden_activation="relu", output_activation="sigmoid",
            threshold=0.5, batch_norm=False, bn_state=None):
    """Give the predicted class for every example, as a flat array of ints."""
    AL = predict_proba(X, parameters, hidden_activation, output_activation,
                       batch_norm, bn_state)
    if AL.shape[0] == 1:
        return (AL > threshold).astype(np.int64).ravel()
    return np.argmax(AL, axis=0).astype(np.int64)
