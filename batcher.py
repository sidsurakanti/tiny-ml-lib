class Batcher:
  def __init__(self, data, batch_size: int = 0):
    self.data = data
    self.batch_size = batch_size if batch_size != 0 else data.shape[1]
    self.index = 0

  def __len__(self):
    return self.data.shape[1] // self.batch_size

  def __iter__(self):
    return self

  def __next__(self):
    if self.index >= len(self.data.shape[1]):
        raise StopIteration
    batch = self.data[:,self.index:self.index + self.batch_size + 1]
    self.index += self.batch_size
    return batch