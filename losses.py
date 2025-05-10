import numpy as np
from defs import Array

def one_hot(c: int, mat: Array):
  m = mat.shape[0]
  arr = np.zeros((c, m))
  arr[mat, np.arange(m)] = 1 
  return arr

def softmax(logits):
  h = np.max(logits, axis=0, keepdims=1)
  exp = np.exp(logits - h)
  probs = exp / np.sum(exp, axis=0, keepdims=1)
  return probs
   
class CrossEntropyLoss:
  def __init__(self):
    self.probs = None

  def loss(self, logits, truth):
    probs = softmax(logits)
    targets = one_hot(logits.shape[0], truth)

    # cache it for backprop
    self.probs = probs 
    self.targets = targets 

    # sum(y * ln(p)) for p_i in logits 
    # basically, take ln(prediction) for singular correct class b/c target is =/= 0
    return -np.mean(np.sum(targets * np.log(probs + 1e-9))) / logits.shape[1]

  def backwards(self):
    # (y_i - t_i) / m
    dZ = (self.probs - self.targets)  
    return dZ

  def __call__(self, *args, **kwds):
    return self.loss(*args)       
  
  def __repr__(self) -> str:
    return f"<CrossEntropyLoss >"

if __name__ == "__main__":
  truth = np.array([3, 5, 6, 9])
  logits = np.random.rand(10, 4)
  loss_fn = CrossEntropyLoss()

  # print(softmax(logits))
  # print(logits)
  print(loss_fn(logits, truth))