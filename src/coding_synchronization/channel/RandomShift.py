import numpy as np


class RandomShift:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def shift(self, offsets: np.ndarray) -> np.ndarray:
        assert offsets.dtype == np.float64
        return offsets + np.random.normal(0, self.sigma, len(offsets))
