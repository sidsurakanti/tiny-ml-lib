from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
from typing import Any, Union


class Layer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backwards(self, *args, **kwargs) -> Union[None, NDArray[np.float64]]:
        pass

    @abstractmethod
    def toGPU(self, *args, **kwargs) -> Any:
        pass


class ParamLayer(Layer):
    def __init__(self):
        self.input_shape: Any
        self.output_shape: Any

    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def set_weights(self, *args, **kwargs) -> Any:
        pass
