from batcher import Batcher
from defs import Sequence, Array
import numpy as np
from math import ceil
from datetime import datetime
import pickle

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
      # print(layer.__repr__(), dZ.shape)
      dZ = layer.backwards(dZ)
    return dZ
  
  def step(self):
    for layer in self.sequence:
      if hasattr(layer, "step"):
        layer.step()

  def fit(self, epochs: int = 5, *args, timed: bool = True, **kwargs):
    print("\nARCHITECTURE:")
    for layer in self.sequence:
        print(layer)
    print(self.loss)

    print("\nTRAINING...")
    start_time = datetime.now()    
    for epoch in range(epochs):
      loss = self.train(*args, **kwargs)
      print(f"EPOCH {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    print(f"Time spent training: {(datetime.now() - start_time).total_seconds():.2f}s")

    return 

  def train(self, X: Array, y: Array, batch_size: int = 0):  
    batches = Batcher((X, y), batch_size)
    total_batches = len(batches)

    for i, (x, y) in enumerate(batches, start=1):
      _, loss = self.forward(x, y)
      self.backwards()
      if i % ceil(total_batches / 4) == 0 or i == total_batches:
        print(f"Batch {i}/{total_batches}, Loss: {loss:.4f}", end="\r")

      self.step()
    return loss

  def evaluate(self, X_test: Array, y_test: Array):
    print("\nEVALUATING...")
    out, _ = self.forward(X_test, y_test)
    preds = np.argmax(out, axis=1)
    correct = np.sum(preds == y_test)
    r = np.random.randint(0, preds.shape[0])

    print("Sample labels:", y_test[r:r+10])
    print("Sample preds:", preds[r:r+10])

    total = y_test.shape[0]
    return correct / total

  def predict(self, *args, **kwargs):
    raise NotImplementedError("Model predict method not implemented.")

  def save(self, path: str = "model_weights.pkl"):
    model_weights = [] 
    for layer in self.sequence:
      if hasattr(layer, "W") and hasattr(layer, "b"):
        model_weights.append((layer.W, layer.b))

    with open(path, "wb") as f:
      pickle.dump(model_weights, f)
      print("Saved model weights.")
          
    return

  def load(self, path: str):
    raise NotImplementedError("Model load method not implemented.")

  def __repr__(self):
    return f"Model()"

  def __call__(self, *args, **kwargs):
    return self.fit(*args, **kwargs)

if __name__ == "__main__":
  m = Model(1, 1)
  m.save()
