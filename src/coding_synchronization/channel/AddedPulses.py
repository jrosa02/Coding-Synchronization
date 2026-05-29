import numpy as np


class AddedPulses:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def add(self, offsets: np.ndarray) -> np.ndarray:
        assert offsets.dtype == np.float64
        n_added = np.random.poisson(self.rate * len(offsets))
        spurious = np.random.uniform(offsets[0], offsets[-1], n_added)
        return np.sort(np.concatenate([offsets, spurious]))
