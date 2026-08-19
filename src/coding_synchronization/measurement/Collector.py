import logging
from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class Collector(StageABC):
    """Keeps everything it sees, so a Model can save the decoded symbols.

    Works as a terminal sink (`consume`) or as a mid-pipeline tap (`process` passes the signal
    straight through), so the same stage can capture the Syncer output *and* the final output.
    """

    def __init__(self, name: str = "Collector", seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.name = name
        self.items: list[np.ndarray] = []

    def _capture(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> None:
        self.items.append(signal)
        logger.debug("%s: captured chunk with %d frames", self.name, len(signal))

    def consume(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> None:
        self._capture(signal)

    def process(
        self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        self._capture(signal)
        return signal

    @property
    def frames(self) -> list[np.ndarray]:
        return [np.asarray(frame) for chunk in self.items for frame in chunk]

    def reset(self) -> None:
        self.items = []

    def __repr__(self) -> str:
        return f"Collector({self.name})"
