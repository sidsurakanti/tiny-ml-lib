import numpy as np
from defs import Array
from native import updateGpuMemory


def one_hot(classes: int, truth: Array):
    m = truth.shape[0]
    arr = np.zeros((m, classes))
    arr[np.arange(m), truth] = 1
    return arr


def softmax(logits):
    h = np.max(logits, axis=1, keepdims=1)
    exp = np.exp(logits - h)
    probs = exp / np.sum(exp, axis=1, keepdims=1)
    return probs


class MSELoss:
    def __init__(self):
        self.probs = None
        self.targets = None

    def loss(self, logits, truth):
        probs = softmax(logits)
        targets = one_hot(logits.shape[1], truth)

        # cache it for backprop
        self.probs = probs
        self.targets = targets

        # sum(y * ln(p)) for p_i in logits
        # basically, take ln(prediction) for singular correct class b/c y is != 0
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
        self.probs = None
        self.targets = None

    def loss(self, logits, truth, logitsPtr=None):
        m, n = logits.shape  # m, n
        probs = softmax(logits)
        targets = one_hot(n, truth)

        if self._onGPU and logitsPtr:
            self.logitsPtr = logitsPtr

        # cache it for backprop
        self.probs = probs
        self.targets = targets

        # print(logits.shape, probs.shape, targets.shape)

        # sum(y * ln(p)) for p_i in logits
        # basically, take ln(prediction) for singular correct class b/c y is != 0
        return -np.sum(targets * np.log(probs + 1e-9)) / m

    def backwards(self) -> Array | None:
        # (y_i - t_i)
        dZ = self.probs - self.targets
        # store dZ in the logits buffer & let the backwards of the linear layer handle it
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


if __name__ == "__main__":
    truth = np.array([3, 5, 6, 9])
    logits = np.random.rand(10, 4)
    loss_fn = CrossEntropyLoss()

    # print(softmax(logits))
    # print(logits)
    print(loss_fn(logits, truth))
