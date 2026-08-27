# this file holds the update rules: gd, momentum, rmsprop, adam
# they all answer the same question: we have the gradient, how do we change the weights
# the state dict keeps the memory that momentum and rmsprop need between steps

import numpy as np

from .init import parameter_keys

NAMES = ("gd", "momentum", "rmsprop", "adam")
SCHEDULES = ("none", "inverse", "exponential")

DEFAULTS = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}


def init_optimizer_state(parameters, name):
    """Make the empty memory the optimizer needs.

    gd needs nothing.
    momentum keeps v, a running average of the gradient.
    rmsprop keeps s, a running average of the gradient squared.
    adam keeps both, and t, the number of steps done so far.
    """
    if name not in NAMES:
        raise ValueError("unknown optimizer " + str(name) + ", expected one of " + str(NAMES))

    state = {"t": 0}
    if name in ("momentum", "adam"):
        v = {}
        for key in parameter_keys(parameters):
            v[key] = np.zeros_like(parameters[key])
        state["v"] = v
    if name in ("rmsprop", "adam"):
        s = {}
        for key in parameter_keys(parameters):
            s[key] = np.zeros_like(parameters[key])
        state["s"] = s
    return state


def update_parameters(parameters, grads, state, name, learning_rate, **hp):
    """Do one update of every W and b. Returns (parameters, state)."""
    beta1 = hp.get("beta1", DEFAULTS["beta1"])
    beta2 = hp.get("beta2", DEFAULTS["beta2"])
    epsilon = hp.get("epsilon", DEFAULTS["epsilon"])

    state["t"] = state["t"] + 1
    t = state["t"]

    for key in parameter_keys(parameters):
        g = grads["d" + key]

        if name == "gd":
            step = g

        elif name == "momentum":
            # keep going in the direction we were already going
            state["v"][key] = beta1 * state["v"][key] + (1.0 - beta1) * g
            step = state["v"][key]

        elif name == "rmsprop":
            # divide by how big this weight's gradient usually is
            # so a weight with small gradients still gets a real step
            state["s"][key] = beta2 * state["s"][key] + (1.0 - beta2) * (g * g)
            step = g / (np.sqrt(state["s"][key]) + epsilon)

        elif name == "adam":
            # momentum and rmsprop together
            state["v"][key] = beta1 * state["v"][key] + (1.0 - beta1) * g
            state["s"][key] = beta2 * state["s"][key] + (1.0 - beta2) * (g * g)
            # at the start v and s are near zero, so we grow them back up
            v_fixed = state["v"][key] / (1.0 - beta1 ** t)
            s_fixed = state["s"][key] / (1.0 - beta2 ** t)
            step = v_fixed / (np.sqrt(s_fixed) + epsilon)

        else:
            raise ValueError("unknown optimizer " + str(name))

        parameters[key] = parameters[key] - learning_rate * step

    return parameters, state


def learning_rate_at(initial_lr, epoch, schedule="none", decay_rate=0.01):
    """How big the steps should be at this epoch.

    Big steps at the start to move fast, small steps later to settle down.
    """
    if schedule == "none":
        return initial_lr
    if schedule == "inverse":
        return initial_lr / (1.0 + decay_rate * epoch)
    if schedule == "exponential":
        return initial_lr * ((1.0 - decay_rate) ** epoch)
    raise ValueError("unknown schedule " + str(schedule) + ", expected one of " + str(SCHEDULES))
