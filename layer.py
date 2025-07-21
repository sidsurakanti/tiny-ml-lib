from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> NDArray[np.float64]:
        pass

    @abstractmethod
    #
    def backwards(self, *args, **kwargs) -> None | NDArray[np.float64]:
        pass
