# this file checks that our backward pass is correct
# idea: move one weight a tiny bit up and down, see how the loss changes,
# and compare that with the gradient we computed
# run it as soon as backward exists, a wrong gradient still gives a nice looking curve

import numpy as np

from .model import forward, backward
from .losses import compute_loss

PASS_THRESHOLD = 1e-7


def dictionary_to_vector(parameters):
    """Put all the W and b into one long column vector.

    We also return the keys and shapes, so we can put everything back later.
    The order is always the same: W1, b1, W2, b2, ...
    """
    keys = []
    shapes = []
    pieces = []

    n_layers = 0
    for key in parameters:
        if key.startswith("W"):
            n_layers = n_layers + 1

    for l in range(1, n_layers + 1):
        for letter in ["W", "b"]:
            key = letter + str(l)
            keys.append(key)
            shapes.append(parameters[key].shape)
            pieces.append(parameters[key].reshape(-1, 1))

    theta = np.concatenate(pieces, axis=0)
    return theta, keys, shapes


def vector_to_dictionary(theta, keys, shapes):
    # the opposite of dictionary_to_vector
    parameters = {}
    start = 0
    for i in range(len(keys)):
        size = int(np.prod(shapes[i]))
        parameters[keys[i]] = theta[start:start + size].reshape(shapes[i])
        start = start + size
    return parameters


def gradients_to_vector(grads, keys, shapes):
    # same order as dictionary_to_vector, this is important
    pieces = []
    for i in range(len(keys)):
        pieces.append(grads["d" + keys[i]].reshape(-1, 1))
    return np.concatenate(pieces, axis=0)


def gradient_check(parameters, grads, X, Y, hidden_activation="tanh",
                   output_activation="sigmoid", l2=0.0, epsilon=1e-7, verbose=True):
    """Compare our gradients with numerical ones. Small answer means we are correct.

    relative error = ||numeric - ours|| / (||numeric|| + ||ours||)

    Use a tiny network and few examples. The cost is two forward passes for every
    single weight, so a big network takes forever.

    Careful:
      - turn dropout off (keep_prob = 1.0), it makes the loss random
      - use tanh, not relu: relu has a corner at 0 and gives false alarms
    """
    theta, keys, shapes = dictionary_to_vector(parameters)
    ours = gradients_to_vector(grads, keys, shapes)
    n_params = theta.shape[0]

    if output_activation == "softmax":
        loss_name = "categorical_crossentropy"
    else:
        loss_name = "binary_crossentropy"

    numeric = np.zeros((n_params, 1))

    for i in range(n_params):
        theta_plus = np.copy(theta)
        theta_plus[i, 0] = theta_plus[i, 0] + epsilon
        p_plus = vector_to_dictionary(theta_plus, keys, shapes)
        AL_plus, _ = forward(X, p_plus, hidden_activation, output_activation,
                             keep_prob=1.0, training=False)
        loss_plus = compute_loss(AL_plus, Y, loss_name, parameters=p_plus, l2=l2)

        theta_minus = np.copy(theta)
        theta_minus[i, 0] = theta_minus[i, 0] - epsilon
        p_minus = vector_to_dictionary(theta_minus, keys, shapes)
        AL_minus, _ = forward(X, p_minus, hidden_activation, output_activation,
                              keep_prob=1.0, training=False)
        loss_minus = compute_loss(AL_minus, Y, loss_name, parameters=p_minus, l2=l2)

        numeric[i, 0] = (loss_plus - loss_minus) / (2.0 * epsilon)

    top = np.linalg.norm(numeric - ours)
    bottom = np.linalg.norm(numeric) + np.linalg.norm(ours)
    relative_error = top / bottom

    if verbose:
        if relative_error < PASS_THRESHOLD:
            print("gradient check PASS, relative error " + str(relative_error))
        else:
            print("gradient check FAIL, relative error " + str(relative_error))
            worst = int(np.argmax(np.abs(numeric - ours)))
            print("   worst weight is number " + str(worst))
            print("   ours " + str(ours[worst, 0]) + "  numeric " + str(numeric[worst, 0]))

    return float(relative_error)
