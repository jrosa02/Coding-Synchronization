import logging

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class RandomShift(StageABC):
    def __init__(self, sigma: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.sigma = sigma
        logger.info("RandomShift initialized: sigma=%.4f", sigma)

    def process(self, signal: np.ndarray) -> np.ndarray:
        signal = signal.astype(np.float64)
        noise = self.rng.normal(0, self.sigma, len(signal))
        logger.debug("RandomShift: %d pulses, sigma=%.4f", len(signal), self.sigma)
        return signal + noise

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"RandomShift(sigma={self.sigma})"
