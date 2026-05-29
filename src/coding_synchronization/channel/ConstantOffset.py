import numpy as np


class ConstantOffset:
    def apply_offset(self, frames: np.ndarray):
        return frames + 1024 * np.random.random()
