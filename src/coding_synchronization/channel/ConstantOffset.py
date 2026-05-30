import numpy as np

from coding_synchronization.StageABC import StageABC


class ConstantOffset(StageABC):
    def __init__(self, max_offset: int = 0, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.max_offset = max_offset

    def process(self, signal: np.ndarray) -> np.ndarray:
        return signal + self.max_offset * self.rng.random()

    def reset(self) -> None:
        pass
