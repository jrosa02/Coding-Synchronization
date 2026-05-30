import numpy as np

from coding_synchronization.StageABC import StageABC


class VanishPulses(StageABC):
    def __init__(self, rate: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.rate = rate

    def process(self, signal: np.ndarray) -> np.ndarray:
        mask = self.rng.random(len(signal)) >= self.rate
        return signal[mask]

    def reset(self) -> None:
        pass
