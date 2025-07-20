import numpy as np
from defs import Array
from layer import Layer


class ReLU(Layer):
    def __init__(self) -> None:
        self.out = None
        self.X = None
        self._onGPU = False

    def forward(self, X: Array) -> Array:
        self.X = X
        self.out = np.maximum(0, X)  # (m, n)
        return self.out

    def backwards(self, dZ: Array) -> Array:
        # print(dZ.shape, self.X.shape)
        return dZ * (self.X > 0).astype(float)

    def toGPU(self):
        self._onGPU = True

    def __repr__(self) -> str:
        return f"<ReLU>"
