from defs import Sequence, Array
import numpy as np

class Model:
    def __init__(self, sequence: Sequence, loss_fn) -> None:
      self.sequence = sequence
      self.out = None
      self.loss = loss_fn

    def __repr__(self):
        return f"Model()"

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def forward(self, X: Array, y: Array, *args, **kwargs):
      out = X
      for layer in self.sequence:
        out = layer.forward(out)
      self.out = out
      loss = self.loss(out, y) 
      return loss

    def backwards(self, *args, **kwargs):
      dZ = self.loss.backwards()
      for layer in reversed(self.sequence):
        dZ = layer.backwards(dZ)
      return dZ
    
    def step(self):
      for layer in self.sequence:
        if hasattr(layer, "step"):
          layer.step()

    def save(self, path: str):
      raise NotImplementedError("Model save method not implemented.")

    def load(self, path: str):
      raise NotImplementedError("Model load method not implemented.")

    def fit(self, epochs: int = 5, *args, **kwargs):
      for epoch in range(epochs):
        loss = self.train(*args, **kwargs)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
      return 

    def train(self, *args, **kwargs):
      loss = self.forward(*args, **kwargs)
      self.backwards()
      # for i in [0, 2]:
        # print("Layer", i)
        # print("W:", np.min(self.sequence[i].weights), np.max(self.sequence[i].weights))
        # print("dW:", np.min(self.sequence[i].dW), np.max(self.sequence[i].dW))

      self.step()
      return loss

    def evaluate(self, *args, **kwargs):
      raise NotImplementedError("Model evaluate method not implemented.")

    def predict(self, *args, **kwargs):
      raise NotImplementedError("Model predict method not implemented.")

    def summary(self): 
      raise NotImplementedError("Model summary method not implemented.")

