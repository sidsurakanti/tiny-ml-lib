import numpy as np
from defs import Array
from layer import Layer
from native import initBuff, relu, reluBack, toCPU


class ReLU(Layer):
    def __init__(self, output_shape: int = 0) -> None:
        self.X = np.zeros((0, 0))
        self.out = None
        self.output_shape = output_shape
        self.batch_size = 0
        self._onGPU = False

    def forward(self, X: Array) -> Array:
        self.X = X  # ptr to output buff (C) of prev layer if we're on gpu

        if self._onGPU:
            m, n = self.batch_size, self.output_shape
            relu(self.X, self.outBuf, m, n)  # write result to output Buf
            self.out = self.outBuf
        else:
            self.out = np.maximum(0, self.X)  # (m, n)

        return self.out

    def backwards(self, dZ: Array) -> None | Array:
        # print(dZ, self.ptrC)
        if self._onGPU:
            m, n = self.batch_size, self.output_shape
            reluBack(
                self.outBuf, self.X, m, n
            )  # write result to X (prev layer's C buf)
            return self.X
        else:
            return dZ * (self.X > 0).astype(float)

    def toGPU(self, batch_size) -> None:
        self._onGPU = True
        # init output buff
        self.batch_size = batch_size
        self.outBuf = initBuff(self.batch_size, self.output_shape)  # typeof PyCapsule

    def debug(self):
        # NOTE: do with these what you wish
        out = toCPU(self.out, self.batch_size, self.output_shape)
        input = toCPU(self.X, self.batch_size, self.output_shape)

    def __repr__(self) -> str:
        return f"<ReLU>"
