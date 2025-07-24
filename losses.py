import numpy as np
from defs import Array
from native import updateGpuMemory


def one_hot(classes: int, truth: Array) -> Array:
    m = truth.shape[0]
    res = np.zeros((m, classes))
    res[np.arange(m), truth] = 1
    return res


def softmax(logits) -> Array:
    h = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - h)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs


class MSELoss:
    def __init__(self):
        self.probs = np.zeros((0, 0))
        self.targets = np.zeros((0, 0))
        self._onGPU = False

    def loss(self, logits, truth):
        probs = softmax(logits)
        targets = one_hot(logits.shape[1], truth)

        # cache it for backprop
        self.probs = probs
        self.targets = targets

        return -np.sum((probs - targets) ** 2)

    def backwards(self):
        # (y_i - t_i)
        dZ = 2 * (self.probs - self.targets)
        return dZ

    def toGPU(self):
        self._onGPU = True

    def __call__(self, *args):
        return self.loss(*args)

    def __repr__(self) -> str:
        return f"<MSELoss>"


class CrossEntropyLoss:
    def __init__(self):
        self.probs = np.zeros((0, 0))
        self.targets = np.zeros((0, 0))
        self._onGPU = False

    def loss(self, logits, truth, logitsPtr=None):
        m, n = logits.shape  # m, n
        probs = softmax(logits)
        targets = one_hot(n, truth)

        if self._onGPU and logitsPtr:
            self.logitsPtr = logitsPtr

        self.probs = probs
        self.targets = targets

        # sum(y * ln(p)) for p_i in logits
        # basically, take ln(prediction) for singular correct class b/c y is != 0
        return -np.sum(targets * np.log(probs + 1e-9)) / m

    def backwards(self) -> Array | None:
        # (y_i - t_i)
        dZ = self.probs - self.targets
        # store dZ in the logits buffer (output buff for layer L-1) & let the backwards of the linear layer handle it
        if self._onGPU:
            updateGpuMemory(dZ.reshape(-1), self.logitsPtr, *dZ.shape)
            return self.logitsPtr
        return dZ

    def toGPU(self):
        self._onGPU = True

    def __call__(self, *args):
        return self.loss(*args)

    def __repr__(self) -> str:
        return f"<CrossEntropyLoss>"
