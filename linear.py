import numpy as np
from defs import Array


class Linear:
  # TODO: He init, Xavier init, etc
  def __init__(self, inputs: int, outputs: int) -> None:
    self.inputs = inputs
    self.outputs = outputs
    self.size = (inputs, outputs)
    # init weights and biases
    self.weights = np.random.randn(self.outputs, self.inputs)
    self.biases = np.zeros((self.outputs, 1))

    # forward pass
    self.X = None
    # backward pass
    self.dW = None
    self.db = None

  def forward(self, X: Array) -> Array:
    self.X = X
    # X -> (self.inputs, m)
    # WX + b -> (self.outputs, m) Z
    output =  self.weights @ self.X + self.biases
    return output

  def backwards(self, dZ: Array) -> tuple[Array, Array]:
    m = self.X.shape[1]
    # dOut/dIn
    self.dW = dZ @ self.X.T / m # (outputs, m) * (m, inputs) -> (outputs, inputs)
    self.db = np.sum(dZ, axis=1, keepdims=True) / m # (outputs, m) -> (outputs, 1)
    dX = self.weights.T @ dZ 
    return dX

  def __repr__(self) -> str:
    return f"<Linear: {self.inputs} -> {self.outputs}>"