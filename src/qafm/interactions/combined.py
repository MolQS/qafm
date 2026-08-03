# combined.py

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable


ForceFunction = Callable[[ArrayLike], NDArray[np.float64]]


@dataclass(frozen=True)
class CombinedForce:
    ''' TODO: Documentation 
    '''
    forces: tuple[ForceFunction, ...]
    gradients: tuple[ForceFunction, ...]
    zero_for_negative_z: bool = False

    def force(self, z_axis: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(z_axis, dtype=float)
        result = sum(force(z) for force in self.forces)

        if self.zero_for_negative_z:
            result = np.where(z < 0, 0.0, result)

        return result

    def gradient(self, z_axis: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(z_axis, dtype=float)
        result = sum(gradient(z) for gradient in self.gradients)

        if self.zero_for_negative_z:
            result = np.where(z < 0, 0.0, result)

        return result