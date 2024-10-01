from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class CalibrationSaveData:
    size: tuple[int, int]
    matrix: np.ndarray[tuple[Literal[3], Literal[3]]]
    dist: np.ndarray[tuple[Literal[1], Literal[5]]]
