import numpy as np
from defs import Array
from layer import ParamLayer
from native import toCPU, toGPU, linear, initBuffers, linearBack, matMatSub


class Linear(ParamLayer):
    def __init__(self, inputs: int, outputs: int, init: str = "he") -> None:
        INITS = {
            "he": np.sqrt(2 / inputs),
            "xavier": np.sqrt(2 / (inputs + outputs)),
            "none": 1,
        }

        self.input_shape = inputs
        self.output_shape = outputs
        self.size = (inputs, outputs)

        self._onGPU = False
        self.batch_size = 0

        # init weights and biases
        self.W = np.random.randn(self.input_shape, self.output_shape) * INITS[init]
        self.b = np.zeros((1, self.output_shape))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.velocity = np.zeros_like(self.W)
        self.velocity_b = np.zeros_like(self.b)

        self.X = np.empty(
            (0,), dtype=np.float64
        )  # could init this as None but my lsp is buns

    def forward(self, X: Array) -> Array:
        self.X = X  # X === gpu ptr if self._onGPU && if this layer isn't the first
        return [self.cpuForward, self.gpuForward][self._onGPU]()

    def gpuForward(self):
        assert self._onGPU, "Tried to access GPU buffers but not on GPU"

        m, n, k = self.batch_size, self.input_shape, self.output_shape
        W, b, C = self.gpuPtrs[0]

        # send input to device if not already
        # this happens on the first linear layer on every batch
        if not isinstance(self.X, type(W)) and (self.X is not None):
            self.X = toGPU(self.X.reshape(-1), np.prod(self.X.shape))

        # X -> (m, inputs); W -> (inputs, outputs)
        # XW + b -> (m, outputs) => Z
        linear(self.X, W, b, C, m, n, k)
        return C  # ptr to output buff

    def cpuForward(self):
        return self.X @ self.W + self.b

    def backwards(self, dZ: Array):
        return [self.cpuBackwards, self.gpuBackwards][self._onGPU](dZ)

    def gpuBackwards(self, dZ):  # don't need dZ but keeping it for logging
        assert self._onGPU, "Tried to access GPU buffers but not on GPU"

        m, n, k = self.batch_size, self.input_shape, self.output_shape
        # we reuse this layer's output buffer to store dZ values from layer L+1
        W, _, C = self.gpuPtrs[0]
        dW, dB = self.gpuPtrs[1]

        # LOG: check if incoming dZ and stored C ptr are the same
        # print("incoming dZ:", dZ)
        # print("stored C ptr", self.gpuPtrs[0][2])

        # store dX (the dZ for prev layer) in self.X ptr (the C buf from prev layer)
        linearBack(self.X, W, dW, dB, C, m, n, k)
        return self.X

    def cpuBackwards(self, dZ: Array):
        m, _ = self.X.shape

        # (m, inputs).T * (m, outputs) -> (inputs, outputs)
        self.dW = self.X.T @ dZ / m
        # (m, outputs) -> (1, outputs)
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        # (m, outputs) * (inputs, outputs).T -> (m, inputs)
        dX = dZ @ self.W.T

        return dX

    def step(self, learning_rate: float = 0.03, momentum: float = 0.9) -> None:
        if self._onGPU:
            W, b, _ = self.gpuPtrs[0]
            dW, dB = self.gpuPtrs[1]
            matMatSub(W, dW, learning_rate, self.input_shape, self.output_shape)
            matMatSub(b, dB, learning_rate, 1, self.output_shape)
            return

        self.velocity = momentum * self.velocity + (1 - momentum) * self.dW
        self.velocity_b = momentum * self.velocity_b + (1 - momentum) * self.db

        self.W -= learning_rate * self.velocity
        self.b -= learning_rate * self.velocity_b

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def toGPU(self, batch_size: int):
        self._onGPU = True
        self.batch_size = batch_size
        # init all the buffers we're gonna use on gpu memory
        w, b, c, dw, db = initBuffers(
            self.W.reshape(-1), self.input_shape, self.output_shape, batch_size
        )
        self.gpuPtrs = [(w, b, c), (dw, db)]

    def gpuBufDump(self):
        w, b, c = self.gpuPtrs[0]
        dw, db = self.gpuPtrs[1]

        x = toCPU(self.X, self.batch_size, self.input_shape)
        w = toCPU(w, self.input_shape, self.output_shape)
        b = toCPU(b, 1, self.output_shape)
        c = toCPU(c, self.batch_size, self.output_shape)
        dw = toCPU(dw, self.input_shape, self.output_shape)
        db = toCPU(db, 1, self.output_shape)
        return x, w, b, c, dw, db

    def truthForward(self):
        X, W, b, _, _, _ = self.gpuBufDump()
        return X @ W + b

    def truthBackwards(self):
        X, W, _, dZ, _, _ = self.gpuBufDump()
        m = X.shape[0]
        dW = X.T @ dZ / m
        dB = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ W.T
        return dW, dB, dX

    def debug(self):
        # NOTE: i'm leaving this here to speed up logging internals; print these out as you wish
        w, b, c = self.gpuPtrs[0]
        dw, db = self.gpuPtrs[1]

        x, w, b, c, dw, db = self.gpuBufDump()
        tdw, tdb, tdx = self.truthBackwards()

    def get_weights(self):
        return (self.W, self.b)

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def __repr__(self) -> str:
        return f"<Linear: {self.input_shape} -> {self.output_shape}>"
