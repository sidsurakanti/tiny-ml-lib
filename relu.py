import numpy as np
from defs import Array
from layer import Layer
from native import initBuff, relu, reluBack, toCPU


class ReLU(Layer):
    def __init__(self, output_shape: int) -> None:
        self.out = None
        self.output_shape = output_shape
        self.X = np.zeros((1, 1))  # useless init to get rid of lsp
        self._onGPU = False

    def forward(self, X: Array) -> Array:
        self.X = X  # ptr to output buff (C) of prev layer if we're on gpu

        if self._onGPU:
            m, n = self.batch_size, self.output_shape
            relu(self.X, self.ptrC, m, n)
            self.out = self.ptrC
        else:
            self.out = np.maximum(0, self.X)  # (m, n)

        return self.out

    def backwards(self, dZ: Array) -> None | Array:
        # print(dZ, self.ptrC)
        if self._onGPU:
            m, n = self.batch_size, self.output_shape
            reluBack(self.ptrC, self.X, m, n)
            return self.X
        else:
            return dZ * (self.X > 0).astype(float)

    def toGPU(self, batch_size):
        self._onGPU = True
        # init output buff
        self.batch_size = batch_size
        self.ptrC = initBuff(self.batch_size, self.output_shape)

    def debug(self):
        print(self)
        print("input", self.X)
        print("out", self.ptrC)

        out = toCPU(self.out, self.batch_size, self.output_shape)
        input = toCPU(self.X, self.batch_size, self.output_shape)
        print("input", input[0, :5])
        print("output", out[0, :5])
        print()

    def __repr__(self) -> str:
        return f"<ReLU>"
