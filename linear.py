import numpy as np
from defs import Array
from layer import Layer

class Linear(Layer):
  # TODO: He init, Xavier init, etc
  def __init__(self, inputs: int, outputs: int, init: str = "he") -> None:
    INITS = { "he": np.sqrt(2  / inputs), "xavier": np.sqrt(6 / (inputs + outputs)), "none": 1}

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
    # print(self.W.shape, self.X.shape)

    # X -> (m, inputs)
    # W -> (inputs, outputs) 
    # XW + b -> (m, outputs) Z
    output =  self.X @ self.W + self.b
    return output 

  def backwards(self, dZ: Array) -> tuple[Array, Array]:
    m = self.X.shape[0]
    # print(dZ.shape, self.X.shape)

    # dW -> (inputs, outputs)
    self.dW = self.X.T @ dZ / m # (m, inputs).T * (m, outputs) -> (inputs, outputs) 
    self.db = np.sum(dZ, axis=0, keepdims=True) / m # (m, outputs) -> (1, outputs)
    dX = dZ @ self.W.T  # (m, outputs) * (inputs, outputs).T -> (m, inputs)

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

