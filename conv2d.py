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
    self.W = np.random.randn(*self.filter_shape) * 0.01
    self.b = np.zeros(self.output_shape)
    self.out = None
    self.dW = np.zeros(self.filter_shape)
    self.db = np.zeros(self.output_shape)

  def forward(self, X):
    self.X = X
    m = self.X.shape[0]
    self.out = np.zeros((m, *self.output_shape))
    print(self.W.shape)
    print(self.X.shape)

    for x in range(m):
      temp = np.copy(self.b)
      for i in range(self.filters):
        for input_channel in range(self.input_channels):
          print(X[x, input_channel].shape, "\n", self.W[i, input_channel].shape)
          temp[i] += signal.convolve(X[x, input_channel], self.W[i, input_channel], mode="valid")
      self.out[x] = temp

    return self.out

  def backwards(self, out_grad):
    m = out_grad.shape[0]
    dX = np.zeros((m, *self.input_shape))
    
    for x in range(m):
      temp = np.zeros(self.input_shape)
      for i in range(self.filters):
        for channel in range(self.input_channels):
          # print(self.X[x, channel].shape, out_grad[])
          self.dW[i, channel] += signal.correlate(self.X[x, channel], out_grad[x, i], mode="valid")
          temp[channel] += signal.convolve(out_grad[x, i], self.W[i, channel], mode="full")
      dX[x] = temp

    self.dW /= m
    self.db = np.sum(out_grad, axis=0) / m
    return dX


  def step(self, learning_rate: float = 0.1) -> None:
    # update weights and biases
    self.W -= learning_rate * self.dW
    self.b -= learning_rate * self.db
    # reset gradients
    self.dW = None
    self.db = None
    return


  def __repr__(self):
    return f"<Conv2D: {self.input_shape} -> {self.output_shape}>"


