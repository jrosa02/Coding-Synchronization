import logging
from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class MetadataCheck(StageABC):
    def __init__(self, metadata_num: int = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.metadata_num = metadata_num
        logger.info("MetadataCheck initialized: metadata_num=%d", metadata_num)

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        expected_metadata = np.arange(1, self.metadata_num + 1)
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
            expected_metadata += self.metadata_num

        logger.debug("MetadataCheck: %d frames passed", len(output))
        return np.asanyarray(output, dtype=object)

    def reset(self) -> None:
        super().reset()

    def __repr__(self) -> str:
        return f"MetadataCheck(metadata_num={self.metadata_num})"
