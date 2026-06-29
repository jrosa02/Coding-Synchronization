import logging

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class AddedPulses(StageABC):
    def __init__(self, rate: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.rate = rate
        logger.info("AddedPulses initialized: rate=%.4f", rate)

    def process(self, signal: np.ndarray) -> np.ndarray:
        signal = signal.astype(np.float64)
        n_added = self.rng.poisson(self.rate * len(signal))
        spurious = self.rng.uniform(signal[0], signal[-1], n_added)
        result = np.sort(np.concatenate([signal, spurious]))
        logger.debug("AddedPulses: %d → %d pulses (+%d spurious)", len(signal), len(result), n_added)
        return result

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"AddedPulses(rate={self.rate})"
