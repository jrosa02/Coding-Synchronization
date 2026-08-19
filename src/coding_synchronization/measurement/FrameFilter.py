import logging
from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


def as_frame_array(frames: list[np.ndarray]) -> np.ndarray:
    """1-D object array of frames.

    np.array(list_of_arrays, dtype=object) collapses to 2-D when every frame happens to have the
    same length, which is exactly what happens once the partial frames are gone.
    """
    out = np.empty(len(frames), dtype=object)
    for i, frame in enumerate(frames):
        out[i] = frame
    return out


class FrameFilter(StageABC):
    """Drop frames the decoder cannot trust: the capture-start fragment and wrong-length ones."""

    def __init__(
        self,
        expected_pulses: int | None = None,
        drop_first: bool = False,
        drop_wrong_length: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.expected_pulses = expected_pulses
        self.drop_first = drop_first
        self.drop_wrong_length = drop_wrong_length
        logger.info(
            "FrameFilter initialized: expected_pulses=%s, drop_first=%s, drop_wrong_length=%s",
            expected_pulses, drop_first, drop_wrong_length,
        )

    def process(
        self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        kept: list[np.ndarray] = []
        for i, frame in enumerate(signal):
            n = len(frame)
            if self.drop_first and i == 0:
                logger.info(
                    "FrameFilter: dropping frame 0 (%d pulses) — the capture starts mid-frame", n
                )
                continue
            if (
                self.drop_wrong_length
                and self.expected_pulses is not None
                and n != self.expected_pulses
            ):
                logger.warning(
                    "FrameFilter: dropping frame %d — %d pulses, expected %d",
                    i, n, self.expected_pulses,
                )
                continue
            kept.append(np.asarray(frame))
        logger.debug("FrameFilter: %d frames in, %d kept", len(signal), len(kept))
        return as_frame_array(kept)

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"FrameFilter(expected_pulses={self.expected_pulses}, "
            f"drop_first={self.drop_first}, drop_wrong_length={self.drop_wrong_length})"
        )
