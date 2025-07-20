import numpy as np
from defs import Array
from layer import Layer
from native import matmul, toGPU, linear, initBuffers


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
        self.X = np.zeros((1, 1))
        # backward pass
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._onGPU = False

    def forward(self, X: Array) -> Array:
        self.X = X
        if self._onGPU:
            return self.gpuLinear()
        return self.cpuLinear()

    def gpuLinear(self):
        m, n, k = self.batch_size, self.input_shape, self.output_shape

        # send input to device if not already
        if not isinstance(self.X, type(self.gpuPtrs[0][0])) and self.X is not None:
            self.X = toGPU(self.X.reshape(-1), np.prod(self.X.shape))

        # X -> (m, inputs); W -> (inputs, outputs)
        # XW + b -> (m, outputs) => Z
        linear(self.X, *self.gpuPtrs[0], m, n, k)

        # output = matmul(self.X.reshape(-1), self.W.reshape(-1), m, n, k) + self.b

        # print(self.gpuPtrs[0][2])
        return self.gpuPtrs[0][2]  # ptr to C

    def cpuLinear(self):
        return self.X @ self.W + self.b

    def backwards(self, dZ: Array) -> Array:
        m, n = self.X.shape[0], self.X.shape[1]

        dZ1 = dZ.reshape(-1)
        self.db = np.sum(dZ, axis=0, keepdims=True) / m  # (m, outputs) -> (1, outputs)
        # self.dW = self.X.T @ dZ / m  # (m, inputs).T * (m, outputs) -> (inputs, outputs)
        self.dW = matmul(self.X.T.reshape(-1), dZ1, n, m, dZ.shape[1]) / m
        # dX = dZ @ self.W.T  # (m, outputs) * (inputs, outputs).T -> (m, inputs)
        dX = matmul(
            dZ1,
            self.W.T.reshape(-1),
            dZ.shape[0],
            dZ.shape[1],
            self.W.T.shape[1],
        )

        return dX

    def step(self, learning_rate: float = 0.03) -> None:
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
        # init W, b, dW, dB, & Y on gpu memory
        # having batch_size as an argument for now since we're gonna be calling it from the model class but might need to rework this for standalone usage
        w, b, c, dw, db = initBuffers(
            self.W.reshape(-1), self.input_shape, self.output_shape, batch_size
        )
        self.gpuPtrs = [(w, b, c), (dw, db)]
        self.batch_size = batch_size
        self._onGPU = True

    def __repr__(self) -> str:
        return f"<Linear: {self.input_shape} -> {self.output_shape}>"
