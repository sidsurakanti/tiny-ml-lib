import numpy as np
import numpy.typing as npt
from defs import Array
from layer import Layer


class ReLU(Layer):
  def __init__(self) -> None:
    self.out = None
    self.X = None

  def forward(self, X: Array) -> Array:
    self.X = X
    self.out = np.maximum(0, X) # (n, m)
    return self.out 
    
  def backwards(self, dZ: Array) -> Array:
    return dZ * (self.X > 0).astype(float)
  
  def __repr__(self) -> str:
    return f"<ReLU>"
