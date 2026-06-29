from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC


class MetadataCheck(StageABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed) 

    def process(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        expected_metadata = np.array([1, 2, 3, 4])

        output = []

        for i, frame in enumerate(signal):
            assert all(frame[:len(expected_metadata)] == expected_metadata), f"{frame[:len(expected_metadata)]} != {expected_metadata}"
            # print(f"{frame[:len(expected_metadata)]} != {expected_metadata}")
            output.append(frame[len(expected_metadata):])
            expected_metadata += 4

        return np.asanyarray(output, dtype=object)

    def reset(self) -> None:
        super().reset()
