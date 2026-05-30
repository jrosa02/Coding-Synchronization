from dataclasses import dataclass

import numpy as np


@dataclass
class ModulationParams:
    ppm_rank: int
    slot_time: np.float64
