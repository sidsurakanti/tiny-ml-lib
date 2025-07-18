import numpy as np
from defs import Array
from layer import Layer
from native import matmul


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
        self.b = np.random.randn(1, self.output_shape)

        # forward pass
        self.X = None
        # backward pass
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X: Array) -> Array:
        self.X = X
        m, n, k = self.X.shape[0], self.X.shape[1], self.W.shape[1]

        # X -> (m, inputs); W -> (inputs, outputs)
        # XW + b -> (m, outputs) => Z
        # output = self.X @ self.W + self.b
        output = matmul(self.X.reshape(-1), self.W.reshape(-1), m, n, k) + self.b
        return output

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

    def __repr__(self) -> str:
        return f"<Linear: {self.input_shape} -> {self.output_shape}>"
