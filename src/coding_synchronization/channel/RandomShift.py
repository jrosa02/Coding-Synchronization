import numpy as np

from coding_synchronization.StageABC import StageABC


class RandomShift(StageABC):
    def __init__(self, sigma: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.sigma = sigma

    def process(self, signal: np.ndarray) -> np.ndarray:
        signal = signal.astype(np.float64)
        return signal + self.rng.normal(0, self.sigma, len(signal))

    def reset(self) -> None:
        pass
