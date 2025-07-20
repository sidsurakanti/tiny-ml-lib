import numpy as np
from defs import Array
from layer import Layer
from native import toCPU, initBuff, relu


class ReLU(Layer):
    def __init__(self, output_size: int) -> None:
        self.out = None
        self.output_size = output_size
        self.X = None
        self._onGPU = False

    def forward(self, X: Array) -> Array:
        self.X = X
        if self._onGPU:
            relu(self.gpuPtr, self.batch_size, self.output_size)
            self.out = self.gpuPtr
        else:
            self.out = np.maximum(0, X)  # (m, n)
        return self.out

    def backwards(self, dZ: Array) -> Array:
        # print(dZ.shape, self.X.shape)
        return dZ * (self.X > 0).astype(float)

    def toGPU(self, batch_size):
        self._onGPU = True
        # init output buff
        self.gpuPtr = initBuff(batch_size, self.output_size)
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return f"<ReLU>"
