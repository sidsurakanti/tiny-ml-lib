import math
class Batcher:
  def __init__(self, data: tuple, batch_size: int = 0):
    self.data = data
    self.x, self.y = data
    self.batch_size = batch_size if batch_size != 0 else self.y.shape[0]
    self.index = 0

  def __len__(self):
    return math.ceil(self.y.shape[0] / self.batch_size)

  def __iter__(self):
    return self

  def __next__(self):
    if self.index >= self.y.shape[0]:
      raise StopIteration

    x = self.x[:,self.index:self.index + self.batch_size + 1]
    y = self.y[self.index:self.index + self.batch_size + 1]
    self.index += min(self.batch_size, self.y.shape[0] - self.index)
    return (x, y)