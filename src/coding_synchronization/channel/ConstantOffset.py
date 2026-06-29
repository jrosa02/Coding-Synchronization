import logging

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class ConstantOffset(StageABC):
    def __init__(self, max_offset: int = 0, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.max_offset = max_offset
        logger.info("ConstantOffset initialized: max_offset=%d", max_offset)

    def process(self, signal: np.ndarray) -> np.ndarray:
        offset = self.max_offset * self.rng.random()
        logger.debug("ConstantOffset: offset=%.2f chirps applied to %d pulses", offset, len(signal))
        return signal + offset

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"ConstantOffset(max_offset={self.max_offset})"
