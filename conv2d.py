import numpy as np
from scipy import signal
from layer import Layer
from typing import Tuple

class Conv2D(Layer):
  def __init__(self, input_shape: Tuple[int], filters: int, filter_size: int, stride: int = 1, padding: str = 'valid' | 'same'):
    input_depth, input_height, input_width = input_shape
    self.input_shape = input_shape
    self.input_depth = input_depth
    self.filters = filters
    self.filter_size = filter_size
    self.stride = stride
    self.padding = padding
    self.X = None
    self.filter_shape = (filters, input_depth, filter_size, filter_size)
    self.W = np.random.randn(*self.filter_shape) * 0.01
    self.output_shape = (filters, 
                         (input_height - filter_size // stride + 1),
                         (input_width - filter_size // stride + 1))
    self.b = np.zeros(self.output_shape)

  def forward(self, X):
    self.X = X
    out = np.copy(self.b)
    for i in range(self.filters):
      for input_channel in range(self.input_depth):
        out[i] += signal.convolve(X[input_channel], self.W[i, input_channel], mode='valid')
