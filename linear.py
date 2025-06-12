import numpy as np
from defs import Array
from layer import Layer

class Linear(Layer):
  # TODO: He init, Xavier init, etc
  def __init__(self, inputs: int, outputs: int) -> None:
    self.inputs = inputs
    self.outputs = outputs
    self.size = (inputs, outputs)
    # init weights and biases
    self.weights = np.random.randn(self.inputs, self.outputs)
    self.biases = np.random.randn(1, self.outputs)

    # forward pass
    self.X = None
    # backward pass
    self.dW = np.zeros_like(self.weights)
    self.db = np.zeros_like(self.biases)

  def forward(self, X: Array) -> Array:
    self.X = X 
    # print(self.weights.shape, self.X.shape)

    # X -> (m, inputs)
    # W -> (inputs, outputs) 
    # XW + b -> (m, outputs) Z
    output =  self.X @ self.weights + self.biases
    return output 

  def backwards(self, dZ: Array) -> tuple[Array, Array]:
    m = self.X.shape[0]
    # print(dZ.shape, self.X.shape)

    # dW -> (inputs, outputs)
    self.dW = self.X.T @ dZ / m # (m, inputs).T * (m, outputs) -> (inputs, outputs) 
    self.db = np.sum(dZ, axis=0, keepdims=True) / m # (m, outputs) -> (1, outputs)
    dX = dZ @ self.weights.T  # (m, outputs) * (inputs, outputs).T -> (m, inputs)

    return dX
  
  def step(self, learning_rate: float = 0.1) -> None:
    # update weights and biases
    self.weights -= learning_rate * self.dW
    self.biases -= learning_rate * self.db

    # print(np.mean(self.weights), np.std(self.weights))
    # print(np.mean(self.dW), np.std(self.dW))
    # reset gradients
    self.dW = np.zeros_like(self.weights)
    self.db = np.zeros_like(self.biases)
    return

  def __repr__(self) -> str:
    return f"<Linear: {self.inputs} -> {self.outputs}>"

