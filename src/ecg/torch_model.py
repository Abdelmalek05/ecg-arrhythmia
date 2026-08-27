# the same network again, but written in pytorch
# we do this for two reasons:
#   1. to check our numpy code against a library that many people trust
#   2. to try a convolutional network, which numpy would make painful
#
# careful: our numpy code puts one example in a COLUMN, shape (features, batch).
# pytorch puts one example in a ROW, shape (batch, features). so we transpose.

import time

import numpy as np
import torch
import torch.nn as nn


def to_torch(X_columns, y=None):
    """Our (features, batch) arrays -> pytorch (batch, features) tensors.

    Our data is float32 on disk, but the model may be float64 (we switch to
    float64 for the agreement test). So we always follow whatever dtype torch
    is currently using, instead of whatever numpy happens to hand us.
    """
    Xt = torch.tensor(np.ascontiguousarray(X_columns.T),
                      dtype=torch.get_default_dtype())
    if y is None:
        return Xt
    return Xt, torch.tensor(np.asarray(y).ravel(), dtype=torch.long)


class MLP(nn.Module):
    """The same shape as our numpy model: linear, tanh, linear.

    There is no softmax at the end. nn.CrossEntropyLoss does the softmax itself,
    and it does it in the safe log-sum-exp way, the same as our log_softmax.
    """

    def __init__(self, n_input, hidden=(16,), n_output=4, activation="tanh"):
        super().__init__()
        sizes = [n_input] + list(hidden)
        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
        layers.append(nn.Linear(sizes[-1], n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def copy_parameters_in(model, parameters):
    """Put our numpy W and b into the pytorch model.

    nn.Linear stores weight with shape (out, in), which is exactly our W shape.
    Its bias has shape (out,) while ours is (out, 1), so we flatten.
    """
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, layer in enumerate(linears, start=1):
            W = parameters["W" + str(i)]
            b = parameters["b" + str(i)]
            assert layer.weight.shape == W.shape, (layer.weight.shape, W.shape)
            layer.weight.copy_(torch.tensor(W))
            layer.bias.copy_(torch.tensor(b.ravel()))
    return model


def gradients_out(model):
    """Read the pytorch gradients back into our naming, so we can compare."""
    grads = {}
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears, start=1):
        grads["dW" + str(i)] = layer.weight.grad.detach().numpy().copy()
        grads["db" + str(i)] = layer.bias.grad.detach().numpy().reshape(-1, 1).copy()
    return grads


class SmallCNN(nn.Module):
    """A small 1D convolutional network over the raw beat.

    Why this exists: in Phase 5 we found that the answer for class S is in the
    SHAPE of the beat, and that a flat network could not read it. A flat network
    sees 250 separate numbers and has to learn what each position means on its
    own. A convolution slides one small filter along the beat, so it learns a
    shape once and can find it anywhere.

    The 4 timing numbers are added at the end, next to the shape features, so the
    model gets both kinds of information.
    """

    def __init__(self, n_rr=4, n_output=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 31 + n_rr, 32), nn.ReLU(),
            nn.Linear(32, n_output),
        )
        self.n_rr = n_rr

    def forward(self, x):
        # x is (batch, 250 + n_rr): the beat first, the timing numbers after
        wave = x[:, :-self.n_rr].unsqueeze(1) if self.n_rr else x.unsqueeze(1)
        rr = x[:, -self.n_rr:] if self.n_rr else None
        h = self.features(wave)
        h = h.flatten(1)
        if rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)


def train_torch(model, Xtr, ytr, Xdv, ydv, epochs=30, batch_size=64,
                learning_rate=0.1, optimizer="sgd", seed=1, verbose=False):
    """Train a pytorch model. Xtr is (features, batch), like the rest of our code."""
    torch.manual_seed(seed)
    Xt, yt = to_torch(Xtr, ytr)
    Xd, yd = to_torch(Xdv, ydv)

    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    m = Xt.shape[0]
    rng = np.random.default_rng(seed)
    start = time.time()
    history = []

    for epoch in range(epochs):
        order = rng.permutation(m)
        model.train()
        for k in range(0, m, batch_size):
            idx = order[k:k + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            dev_pred = model(Xd).argmax(dim=1).numpy()
        history.append(float(loss.item()))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print("epoch " + str(epoch + 1) + " last batch loss " + str(round(history[-1], 5)))

    model.eval()
    with torch.no_grad():
        pred_dev = model(Xd).argmax(dim=1).numpy()
        pred_tr = model(Xt).argmax(dim=1).numpy()
    return {"model": model, "history": history, "wall_clock": time.time() - start,
            "pred_dev": pred_dev, "pred_train": pred_tr}
