from abc import ABC, abstractmethod

class Layer(ABC):
  @abstractmethod
  def forward(self, *args, **kwargs):
    pass

  @abstractmethod
  def backwards(self, *args, **kwargs):
    pass

