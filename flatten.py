from layer import Layer
from defs import Array


class Flatten(Layer):
    def __init__(self):
        self._onGPU = False

    def forward(self, X) -> Array:
        self.old_shape = X.shape
        res = X.reshape(X.shape[0], -1)
        return res

    def backwards(self, out) -> Array:
        return out.reshape(self.old_shape)

    def toGPU(self):
        self._onGPU = True
        # assert False, "No GPU implementation for Convolutional Layer yet"

    def __repr__(self):
        return f"<Flatten>"
