import logging

import numpy as np

from coding_synchronization.encoder.Modulation import ModulationParams
from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class Syncer(StageABC):
    def __init__(self, modparam: ModulationParams, sync_num: int = 4, seed: int = 42) -> None:
        super().__init__(seed)
        self.sync_num = sync_num
        self.max_value = (2**modparam.ppm_rank) - 1
        self.word_period = self.max_value + 1 + modparam.dead_slots
        self.margin = self.word_period // 8
        logger.info(
            "Syncer initialized: sync_num=%d, word_period=%d, margin=%d",
            sync_num, self.word_period, self.margin,
        )

    def _sync_frame(self, positions: np.ndarray) -> np.ndarray | None:
        pos = np.sort(positions.astype(np.float64))
        if len(pos) < self.sync_num:
            return None

        anchor = pos[0]
        sync_pos = np.empty(self.sync_num, dtype=np.float64)
        for k in range(self.sync_num):
            expected = anchor + k * self.word_period
            candidates = pos[np.abs(pos - expected) <= self.margin]
            if len(candidates) > 0:
                sync_pos[k] = candidates[np.argmin(np.abs(candidates - expected))]
            else:
                sync_pos[k] = expected

        # sync word k is at frame_start + k*word_period + max_value
        frame_start = (
            float(np.mean(sync_pos - np.arange(self.sync_num) * self.word_period)) - self.max_value
        )

        sync_end = frame_start + self.sync_num * self.word_period
        remaining = pos[pos >= sync_end]
        if len(remaining) == 0:
            return np.array([], dtype=np.uint16)

        relative = remaining - frame_start
        word_indices = np.round(relative / self.word_period).astype(np.int64)
        values = (
            np.round(relative - word_indices * self.word_period)
            .clip(0, self.max_value)
            .astype(np.uint16)
        )

        valid = word_indices >= self.sync_num
        word_indices = word_indices[valid]
        values = values[valid]

        order = np.argsort(word_indices, stable=True)
        word_indices = word_indices[order]
        values = values[order]

        _, first_occ = np.unique(word_indices, return_index=True)
        return values[first_occ]

    def process(self, signal: np.ndarray) -> np.ndarray:
        results = []
        failed = 0
        for frame in signal:
            decoded = self._sync_frame(np.asarray(frame))
            if decoded is not None:
                results.append(decoded)
            else:
                failed += 1
        logger.debug("Syncer: %d frames in, %d decoded, %d failed sync", len(signal), len(results), failed)
        if failed > 0:
            logger.warning("Syncer: %d/%d frames failed synchronization", failed, len(signal))
        return np.array(results, dtype=object)

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Syncer(sync_num={self.sync_num}, word_period={self.word_period})"
