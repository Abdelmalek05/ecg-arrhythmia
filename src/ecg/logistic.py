# logistic regression written from zero, like course 1 week 2
# it is one neuron: multiply by w, add b, then sigmoid
# we keep it separate from model.py on purpose, so the two can check each other

import numpy as np

from .activations import sigmoid
from .losses import compute_loss


def initialize(n_features):
    # for one neuron we can start at zero, there is no symmetry problem here
    w = np.zeros((n_features, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    """One forward pass and one backward pass.

    X: shape (n_features, m)
    Y: shape (1, m), values 0 or 1
    returns: loss, dw, db
    """
    m = X.shape[1]

    # forward
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    loss = compute_loss(A, Y, "binary_crossentropy")

    # backward
    dZ = (A - Y) / m
    dw = np.dot(X, dZ.T)
    db = float(np.sum(dZ))

    return loss, dw, db


def optimize(w, b, X, Y, num_iterations=1000, learning_rate=0.1, print_every=0):
    """Plain gradient descent on the whole training set."""
    costs = []
    for i in range(num_iterations):
        loss, dw, db = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs.append(loss)
        if print_every and i % print_every == 0:
            print("iteration " + str(i) + "  loss " + str(round(loss, 6)))
    return w, b, costs


def predict_proba(w, b, X):
    return sigmoid(np.dot(w.T, X) + b)


def predict(w, b, X, threshold=0.5):
    # returns a flat array of 0 and 1
    return (predict_proba(w, b, X) > threshold).astype(np.int64).ravel()


def fit(X, Y, num_iterations=1000, learning_rate=0.1, print_every=0):
    """Train and give back everything in one dict."""
    w, b = initialize(X.shape[0])
    w, b, costs = optimize(w, b, X, Y, num_iterations, learning_rate, print_every)
    return {"w": w, "b": b, "costs": costs}
