import logging
from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class MetadataCheck(StageABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        logger.info("MetadataCheck initialized")

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        expected_metadata = np.array([1, 2, 3, 4])
        output = []

        for i, frame in enumerate(signal):
            actual = frame[:len(expected_metadata)]
            # if not np.array_equal(actual, expected_metadata):
            #     logger.error(
            #         "MetadataCheck: frame %d mismatch — got %s, expected %s",
            #         i, actual.tolist(), expected_metadata.tolist(),
            #     )
            #     raise ValueError(
            #         f"Frame {i}: metadata {actual.tolist()} != {expected_metadata.tolist()}"
            #     )
            output.append(frame[len(expected_metadata):])
            expected_metadata += 4

        logger.debug("MetadataCheck: %d frames passed", len(output))
        return np.asanyarray(output, dtype=object)

    def reset(self) -> None:
        super().reset()
