import numpy as np
from defs import Array


class Linear:
  # TODO: He init, Xavier init, etc
  def __init__(self, inputs: int, outputs: int) -> None:
    self.inputs = inputs
    self.outputs = outputs
    self.size = (inputs, outputs)
    self.weights = np.random.randn(self.outputs, self.inputs)
    self.biases = np.zeros((self.outputs, 1))
    self.X = None

  def forward(self, X: Array) -> Array:
    self.X = X
    # X -> (self.inputs, m)
    # WX + b -> (self.outputs, m) Z
    output =  self.weights @ self.X + self.biases
    return output

  def backwards(self, dZ: Array) -> Array:
    m = self.X.shape[1]
    # dOut/dIn
    dW = dZ @ self.X.T / m 
    return dW

  def __repr__(self) -> str:
    return f"<Linear: {self.inputs} -> {self.outputs}>"