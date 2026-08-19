import logging

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class MeasurementGen(StageABC):
    """Source stage that injects measured pulse offsets (slot units) into the decoder pipeline."""

    def __init__(self, offsets: np.ndarray | None = None, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._offsets: np.ndarray | None = None
        self._generated = False
        if offsets is not None:
            self.load(offsets)
        logger.info("MeasurementGen initialized: n_pulses=%d", self.n_pulses)

    @property
    def n_pulses(self) -> int:
        return 0 if self._offsets is None else len(self._offsets)

    def load(self, offsets: np.ndarray) -> None:
        self._offsets = np.sort(np.asarray(offsets, dtype=np.float64))
        self._generated = False

    def generate(self) -> np.ndarray | None:
        if self._generated or self._offsets is None:
            return None
        self._generated = True
        if len(self._offsets) == 0:
            logger.warning("MeasurementGen: no offsets loaded — emitting nothing")
            return None
        span = float(self._offsets[-1] - self._offsets[0])
        logger.debug(
            "MeasurementGen: emitting %d pulses spanning %.1f slots", len(self._offsets), span
        )
        return self._offsets

    def reset(self) -> None:
        self._generated = False

    def __repr__(self) -> str:
        return f"MeasurementGen(n_pulses={self.n_pulses})"
