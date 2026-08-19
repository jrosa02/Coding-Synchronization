import logging

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class VanishPulses(StageABC):
    def __init__(self, rate: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.rate = rate
        logger.info("VanishPulses initialized: rate=%.4f", rate)

    def process(self, signal: np.ndarray) -> np.ndarray:
        mask = self.rng.random(len(signal)) >= self.rate
        result = signal[mask]
        logger.debug("VanishPulses: %d → %d pulses (%d removed)", len(signal), len(result), len(signal) - len(result))
        return result

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"VanishPulses(rate={self.rate})"
