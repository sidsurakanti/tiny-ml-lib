import numpy as np
from defs import Array
from layer import Layer
from native import matmul, toCPU, toGPU, linear, initBuffers, linearBack, matMatSub


class Linear(Layer):
    # TODO: He init, Xavier init, etc
    def __init__(self, inputs: int, outputs: int, init: str = "he") -> None:
        INITS = {
            "he": np.sqrt(2 / inputs),
            "xavier": np.sqrt(6 / (inputs + outputs)),
            "none": 1,
        }

        self.input_shape = inputs
        self.output_shape = outputs
        self.size = (inputs, outputs)

        # init weights and biases
        self.W = np.random.randn(self.input_shape, self.output_shape) * INITS[init]
        # self.b = np.random.randn(1, self.output_shape)
        self.b = np.zeros((1, self.output_shape))

        # forward pass
        self.X = np.zeros((1, 1))  # could init this as None but my lsp is buns
        # backward pass
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._onGPU = False

    def forward(self, X: Array) -> Array:
        self.X = X  # gonna be a gpu ptr if self._onGpu
        if self._onGPU:
            return self.gpuForward()
        return self.cpuForward()

    def gpuForward(self):
        m, n, k = self.batch_size, self.input_shape, self.output_shape

        # send input to device if not already
        # this happens on the first linear layer
        if not isinstance(self.X, type(self.gpuPtrs[0][0])) and self.X is not None:
            self.X = toGPU(self.X.reshape(-1), np.prod(self.X.shape))

        # X -> (m, inputs); W -> (inputs, outputs)
        # XW + b -> (m, outputs) => Z
        W, b, C = self.gpuPtrs[0]
        linear(self.X, W, b, C, m, n, k)
        return C  # ptr to output buff

    def cpuForward(self):
        return self.X @ self.W + self.b

    def backwards(self, dZ: Array):
        if self._onGPU:
            return self.gpuBackwards(dZ)
        return self.cpuBackwards(dZ)

    def gpuBackwards(self, dZ):
        m, n, k = self.batch_size, self.input_shape, self.output_shape

        # dZ is just the curr output buff (C)
        # we reuse the output buffer to store dZ values from layer L+1
        w, _, c = self.gpuPtrs[0]
        dW, dB = self.gpuPtrs[1]
        # print("incoming dZ:", dZ)
        # print("stored C ptr", self.gpuPtrs[0][2])

        # store dX (the dZ for prev layer) in self.X ptr (the C buff for prev layer)
        linearBack(self.X, w, dW, dB, c, m, n, k)
        return self.X

    def getData(self):
        w, b, c = self.gpuPtrs[0]
        dw, db = self.gpuPtrs[1]

        x = toCPU(self.X, self.batch_size, self.input_shape)
        w = toCPU(w, self.input_shape, self.output_shape)
        b = toCPU(b, 1, self.output_shape)
        dw = toCPU(dw, self.input_shape, self.output_shape)
        db = toCPU(db, 1, self.output_shape)
        c = toCPU(c, self.batch_size, self.output_shape)
        return x, w, b, c, dw, db

    def truthForward(self):
        x, w, b, c, dw, db = self.getData()
        return x @ w + b

    def truthBackwards(self):
        x, w, b, dZ, _, _ = self.getData()
        m = x.shape[0]
        dW = x.T @ dZ / m
        # (m, outputs) -> (1, outputs)
        dB = np.sum(dZ, axis=0, keepdims=True) / m
        # (m, outputs) * (inputs, outputs).T -> (m, inputs)
        dX = dZ @ w.T
        return dW, dB, dX

    def debug(self):
        print(self)
        w, b, c = self.gpuPtrs[0]
        dw, db = self.gpuPtrs[1]

        # print("input", self.X)
        # print("w, b, c", w, b, c)
        # print("dw, db", dw, db)
        # print()

        x, w, b, c, dw, db = self.getData()

        tdw, tdb, tdx = self.truthBackwards()

        print("========")
        print("x/dZ\n", x[:5])
        print("w\n", w[0, :5])
        print()
        print("c\n", c[0, :5])
        print("cT\n", self.truthForward()[0, :5])
        print()
        print("dWT\n", tdw[:5])
        print("dw\n", dw[:5])
        print("=========", end="\n\n")

    def cpuBackwards(self, dZ: Array):
        m, _ = self.X.shape

        # (m, inputs).T * (m, outputs) -> (inputs, outputs)
        self.dW = self.X.T @ dZ / m
        # (m, outputs) -> (1, outputs)
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        # (m, outputs) * (inputs, outputs).T -> (m, inputs)
        dX = dZ @ self.W.T

        return dX

    def step(self, learning_rate: float = 0.03) -> None:
        if self._onGPU:
            W, b, _ = self.gpuPtrs[0]
            dW, dB = self.gpuPtrs[1]
            matMatSub(W, dW, learning_rate, self.input_shape, self.output_shape)
            matMatSub(b, dB, learning_rate, 1, self.output_shape)
            return

        # update weights and biases
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        # print(np.mean(self.W), np.std(self.W))
        # print(np.mean(self.dW), np.std(self.dW))
        # reset gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        return

    def get_weights(self):
        return (self.W, self.b)

    def set_weights(self, W, b):
        self.W = W
        self.b = b
        return

    def toGPU(self, batch_size: int):
        self._onGPU = True
        # having batch_size as an argument for now since we're gonna be calling it from the model class but might need to rework this for standalone usage
        self.batch_size = batch_size
        # init W, b, dW, dB, & Y on gpu memory
        w, b, c, dw, db = initBuffers(
            self.W.reshape(-1), self.input_shape, self.output_shape, batch_size
        )
        self.gpuPtrs = [(w, b, c), (dw, db)]

    def __repr__(self) -> str:
        return f"<Linear: {self.input_shape} -> {self.output_shape}>"
