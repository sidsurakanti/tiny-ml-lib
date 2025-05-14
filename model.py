from batcher import Batcher
from defs import Sequence, Array
import numpy as np
from math import ceil

class Model:
  def __init__(self, sequence: Sequence, loss_fn) -> None:
    self.sequence = sequence
    self.loss = loss_fn

  def forward(self, X: Array, y: Array, *args, **kwargs):
    out = X
    for layer in self.sequence:
      out = layer.forward(out)
    loss = self.loss(out, y) 
    return (out, loss)

  def backwards(self, *args, **kwargs):
    dZ = self.loss.backwards()
    for layer in reversed(self.sequence):
      dZ = layer.backwards(dZ)
    return dZ
  
  def step(self):
    for layer in self.sequence:
      if hasattr(layer, "step"):
        layer.step()

  def fit(self, epochs: int = 5, *args, **kwargs):
    for epoch in range(epochs):
      loss = self.train(*args, **kwargs)
      print(f"EPOCH {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    return 

  def train(self, X: Array, y: Array, batch_size: int = 0):  
    batches = Batcher((X, y), batch_size)
    total_batches = len(batches)

    for i, (x, y) in enumerate(batches, start=1):
      _, loss = self.forward(x, y)
      self.backwards()
      if i % ceil(total_batches / 4) == 0 or i == total_batches:
        print(f"Batch {i}/{total_batches}, Loss: {loss:.4f}", end="\r")

      # for i in [0, 2]:
        # print("Layer", i)
        # print("W:", np.min(self.sequence[i].weights), np.max(self.sequence[i].weights))
        # print("dW:", np.min(self.sequence[i].dW), np.max(self.sequence[i].dW))

      self.step()
    return loss

  def evaluate(self, X_test: Array, y_test: Array):
    out, _ = self.forward(X_test, y_test)
    preds = np.argmax(out, axis=0)
    correct = np.sum(preds == y_test)
    print("Sample preds:", preds[:10])
    print("Sample labels:", y_test[:10])
    total = y_test.shape[0]
    return correct / total

  def predict(self, *args, **kwargs):
    raise NotImplementedError("Model predict method not implemented.")

  def summary(self): 
    raise NotImplementedError("Model summary method not implemented.")

  def save(self, path: str):
    raise NotImplementedError("Model save method not implemented.")

  def load(self, path: str):
    raise NotImplementedError("Model load method not implemented.")

  def __repr__(self):
    return f"Model()"

  def __call__(self, *args, **kwargs):
    return self.fit(*args, **kwargs)
