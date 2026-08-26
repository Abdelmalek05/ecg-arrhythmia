# this file has the loss functions
# the loss is one number that says how wrong the model is, we want it small

import numpy as np

from .activations import log_softmax

NAMES = ("binary_crossentropy", "categorical_crossentropy")

# we never let a probability be exactly 0 or 1, because log(0) is -inf
EPS = 1e-12


def l2_penalty(parameters, l2, m):
    # extra cost for having big weights, it pushes the model to stay simple
    # we only do this for W, never for b
    if l2 == 0.0:
        return 0.0
    total = 0.0
    for key in parameters:
        if key.startswith("W"):
            total = total + np.sum(parameters[key] * parameters[key])
    return (l2 / (2.0 * m)) * total


def l2_gradient(W, l2, m):
    # the part we add to dW when l2 is on
    if l2 == 0.0:
        return 0.0
    return (l2 / m) * W


def binary_crossentropy(AL, Y):
    # AL: predicted probability of class 1, shape (1, m)
    # Y: true label 0 or 1, shape (1, m)
    m = Y.shape[1]
    A = np.clip(AL, EPS, 1.0 - EPS)
    losses = -(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
    return float(np.sum(losses) / m)


def categorical_crossentropy(ZL, Y):
    """Cross entropy for many classes.

    ZL: the scores BEFORE softmax, shape (n_classes, m)
    Y: one hot labels, shape (n_classes, m)

    We pass the scores and not the probabilities on purpose, so we can use
    log_softmax and stay safe.
    """
    m = Y.shape[1]
    log_probs = log_softmax(ZL, axis=0)
    return float(-np.sum(Y * log_probs) / m)


def compute_loss(AL, Y, name, class_weights=None, parameters=None, l2=0.0, ZL=None):
    """The average loss over the batch, plus the l2 part if l2 > 0.

    AL: probabilities, shape (1, m) for binary or (n_classes, m) for many classes
    Y: labels, shape (1, m) for binary or one hot (n_classes, m) for many classes
    class_weights: array of shape (n_classes,) or None, used for rare classes
    parameters: only needed when l2 > 0
    ZL: scores before softmax, only for categorical, it is safer than AL
    """
    if name not in NAMES:
        raise ValueError("unknown loss " + str(name) + ", expected one of " + str(NAMES))

    m = Y.shape[1]

    if name == "binary_crossentropy":
        A = np.clip(AL, EPS, 1.0 - EPS)
        per_example = -(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
        if class_weights is not None:
            # weight 0 goes to label 0, weight 1 goes to label 1
            w = class_weights[1] * Y + class_weights[0] * (1.0 - Y)
            per_example = per_example * w
        loss = float(np.sum(per_example) / m)
    else:
        if ZL is None:
            raise ValueError("categorical_crossentropy needs ZL, the scores before softmax")
        log_probs = log_softmax(ZL, axis=0)
        per_example = -np.sum(Y * log_probs, axis=0, keepdims=True)
        if class_weights is not None:
            w = np.sum(class_weights.reshape(-1, 1) * Y, axis=0, keepdims=True)
            per_example = per_example * w
        loss = float(np.sum(per_example) / m)

    if l2 > 0.0:
        if parameters is None:
            raise ValueError("l2 > 0 needs the parameters")
        loss = loss + l2_penalty(parameters, l2, m)

    return loss


def loss_backward(AL, Y, name, class_weights=None):
    """The gradient of the loss with respect to ZL, the scores of the last layer.

    We return dZL and not dAL. For sigmoid with binary cross entropy, and for
    softmax with cross entropy, the maths gives the same simple answer:

        dZL = (AL - Y) / m

    This is nice because we never divide by AL, and dividing by a very small AL
    is exactly where numbers go wrong.
    """
    if name not in NAMES:
        raise ValueError("unknown loss " + str(name))

    m = Y.shape[1]
    dZL = AL - Y

    if class_weights is not None:
        if name == "binary_crossentropy":
            w = class_weights[1] * Y + class_weights[0] * (1.0 - Y)
        else:
            w = np.sum(class_weights.reshape(-1, 1) * Y, axis=0, keepdims=True)
        dZL = dZL * w

    return dZL / m
