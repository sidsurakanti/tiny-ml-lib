import numpy as np
import numpy.typing as npt
from defs import Array

class ReLU:
  def __init__(self, inputs: int, outputs: int) -> None:
    self.inputs = inputs
    self.outputs = outputs
    self.out = None

  def __call__(self, X: Array) -> Array:
    self.out = np.maximum(0, X)
    return self.out 

  def backwards(self, dZ: Array) -> Array:
    return dZ * (self.out > 0).astype(float)
  
  def __repr__(self) -> str:
    return f"<ReLU: {self.inputs} -> {self.outputs}>"