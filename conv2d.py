import numpy as np
from scipy import signal
from layer import Layer
from typing import Tuple

class Conv2d(Layer):
  def __init__(self, input_shape: Tuple[int], filters: int, filter_size: int, stride: int = 1, padding: str = 'valid'):
    input_channels, input_height, input_width = input_shape
    self.input_shape = input_shape
    self.input_channels = input_channels
    self.filters = filters
    self.filter_size = filter_size
    self.stride = stride
    self.padding = padding

    self.filter_shape = (filters, input_channels, filter_size, filter_size)
    self.output_shape = (filters, 
                         (input_height - filter_size // stride + 1),
                         (input_width - filter_size // stride + 1))

    self.X = None
    self.out = None

    self.W = np.random.randn(*self.filter_shape)
    self.b = np.zeros(self.output_shape)
    self.dW = np.zeros(self.filter_shape)
    self.db = np.zeros(self.output_shape)

  def forward(self, X):
    self.X = X
    m = self.X.shape[0]
    self.out = np.zeros((m, *self.output_shape))

    # print(self.W.shape)
    # print(self.X.shape)

    for i in range(m):
      temp = np.copy(self.b)

      for f in range(self.filters):
        for c in range(self.input_channels):
          # print(X[i, c].shape, "\n", self.W[f, c].shape)
          temp[f] += signal.correlate(X[i, c], self.W[f, c], mode="valid")

      self.out[i] = temp

    return self.out

  def backwards(self, out_grad):
    m = out_grad.shape[0]
    dX = np.zeros((m, *self.input_shape))
    # print(out_grad[1])

    for i in range(m):
      temp = np.zeros(self.input_shape)

      for f in range(self.filters):
        for c in range(self.input_channels):
          # print(self.X[i, c].shape, out_grad[])
          self.dW[f, c] += signal.correlate(self.X[i, c], out_grad[i, f], mode="valid")
          temp[c] += signal.convolve(out_grad[i, f], self.W[f, c], mode="full")

      dX[i] = temp

    self.dW /= m
    self.db = np.sum(out_grad, axis=0) / m
    return dX


  def step(self, learning_rate: float = 0.1) -> None:
    # update weights and biases
    self.W -= learning_rate * self.dW
    self.b -= learning_rate * self.db
    # print(np.mean(self.W), np.std(self.W))
    # print(np.mean(self.dW), np.std(self.dW))
    # reset gradients
    self.dW = np.zeros_like(self.W)
    self.db = np.zeros_like(self.b) 
    return


  def __repr__(self):
        return f"<Conv2D: {self.input_shape} -> {self.output_shape}, Filters: {self.filters}, {self.filter_size}x{self.filters}>"


