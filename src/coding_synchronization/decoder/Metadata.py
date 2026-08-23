import logging
from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class MetadataCheck(StageABC):
    """Remove the metadata words of each frame, and verify them when the caller asks.

    `FrameGen._fill_metadata` writes one counter that increases by one for every metadata word and
    continues across frames. Two rules follow from that, and they are not equally safe:

    - Inside a frame the words must be consecutive. This holds whatever happened earlier in the
      capture, so the stage always checks it when `verify` is set.
    - Across frames the counter must continue from the previous frame. A dropped frame breaks that
      chain, and `FrameFilter` drops frames by design, so the stage checks this in `strict` mode
      only.

    Without `strict` a mismatch increments `mismatches` and writes a warning. With `strict` it
    raises `ValueError`. The stage runs after `EccDecode`, so it tests corrected words.
    """

    def __init__(
        self, metadata_num: int = 4, verify: bool = False, strict: bool = False, seed: int = 42
    ) -> None:
        super().__init__(seed)
        self.metadata_num = metadata_num
        self.verify = verify or strict
        self.strict = strict
        self.frames_checked = 0
        self.mismatches = 0
        self.metadata_frames: list[np.ndarray] = []
        self._next_expected: int | None = None
        logger.info(
            "MetadataCheck initialized: metadata_num=%d, verify=%s, strict=%s",
            metadata_num, self.verify, strict,
        )

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / self.frames_checked if self.frames_checked else 0.0

    def _fail(self, index: int, got: list[int], reason: str, expected: list[int] | None) -> None:
        self.mismatches += 1
        message = f"Frame {index}: metadata {got} {reason}"
        if expected is not None:
            message += f", expected {expected}"
        if self.strict:
            logger.error("MetadataCheck: %s", message)
            raise ValueError(message)
        logger.warning("MetadataCheck: %s", message)

    def _check(self, index: int, actual: np.ndarray) -> None:
        got = [int(v) for v in actual]
        self.frames_checked += 1

        if len(got) < self.metadata_num:
            self._fail(index, got, f"holds fewer than {self.metadata_num} words", None)
            return

        consecutive = list(range(got[0], got[0] + self.metadata_num))
        if got != consecutive:
            self._fail(index, got, "is not a consecutive counter", consecutive)
        elif self.strict and self._next_expected is not None and got[0] != self._next_expected:
            expected = list(range(self._next_expected, self._next_expected + self.metadata_num))
            self._fail(index, got, "does not continue the counter", expected)

        self._next_expected = got[0] + self.metadata_num

    def process(
        self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        output = []
        for i, frame in enumerate(signal):
            actual = np.asarray(frame[: self.metadata_num])
            self.metadata_frames.append(actual)
            if self.verify:
                self._check(i, actual)
            output.append(frame[self.metadata_num:])

        if self.verify:
            logger.debug(
                "MetadataCheck: %d frames checked, %d mismatches", self.frames_checked,
                self.mismatches,
            )
        else:
            logger.debug("MetadataCheck: %d frames passed", len(output))
        return np.asanyarray(output, dtype=object)

    def reset(self) -> None:
        super().reset()
        self.frames_checked = 0
        self.mismatches = 0
        self.metadata_frames = []
        self._next_expected = None

    def __repr__(self) -> str:
        return (
            f"MetadataCheck(metadata_num={self.metadata_num}, verify={self.verify}, "
            f"strict={self.strict})"
        )
