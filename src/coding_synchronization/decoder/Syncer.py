import logging

import numpy as np

from coding_synchronization.encoder.Modulation import ModulationParams
from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class Syncer(StageABC):
    def __init__(
        self,
        modparam: ModulationParams,
        sync_num: int = 4,
        sync_value: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed)
        self.sync_num = sync_num
        self.max_value = (2**modparam.ppm_rank) - 1
        self.word_period = self.max_value + 1 + modparam.dead_slots
        # The slot_time_s the caller used to turn raw seconds into the "slot" units this frame
        # arrives in. Only needed to report the auto-calibrated value in real units — the
        # calibration itself (see _sync_frame) works purely on ratios, so it self-corrects even
        # if this is wrong.
        self.nominal_slot_time_s = float(modparam.slot_time)
        # wrong offsets every decoded word by the same amount (0 - true sync value). Default is
        # 0: the sync pulse sits right at its word boundary (slot 0), which is also frame_start
        # itself for word 0 — the simplest, most robust convention.
        self.sync_value = 0 if sync_value is None else int(sync_value)
        self.margin = self.word_period // 8
        # Diagnostic side-channel: decoded sync-word values per frame, how many of the sync
        # pulses were actually found rather than assumed, the fitted frame_start (same slot
        # units as the input positions), the raw positions that were synced, and the per-frame
        # slot-time calibration. Not part of the data stream — lets a plot re-derive exactly
        # which pulse fell in which section.
        self.sync_values: list[np.ndarray] = []
        self.sync_found: list[int] = []
        self.frame_starts: list[float] = []
        self.raw_frames: list[np.ndarray] = []
        self.slot_scales: list[float] = []
        self.inferred_slot_time_s: list[float] = []
        self.sync_residuals: list[np.ndarray] = []
        self.sync_gap_devs: list[np.ndarray] = []
        self._last_sync: np.ndarray | None = None
        self._last_found: int = 0
        self._last_frame_start: float | None = None
        self._last_scale: float = 1.0
        self._last_calibrated_positions: np.ndarray | None = None
        self._last_sync_residual: np.ndarray | None = None
        self._last_sync_gap_dev: np.ndarray | None = None
        logger.info(
            "Syncer initialized: sync_num=%d, word_period=%d, margin=%d, sync_value=%d",
            sync_num, self.word_period, self.margin, self.sync_value,
        )

    def _locate_sync(self, pos: np.ndarray) -> tuple[np.ndarray, int, float]:
        anchor = pos[0]
        sync_pos = np.empty(self.sync_num, dtype=np.float64)
        found = 0
        for k in range(self.sync_num):
            expected = anchor + k * self.word_period
            candidates = pos[np.abs(pos - expected) <= self.margin]
            if len(candidates) > 0:
                sync_pos[k] = candidates[np.argmin(np.abs(candidates - expected))]
                found += 1
            else:
                sync_pos[k] = expected

        # sync word k is at frame_start + k*word_period + sync_value
        frame_start = (
            float(np.mean(sync_pos - np.arange(self.sync_num) * self.word_period)) - self.sync_value
        )
        return sync_pos, found, frame_start

    def _decode_positions(
        self, pos: np.ndarray, frame_start: float
    ) -> tuple[np.ndarray, np.ndarray]:
        relative = pos - frame_start
        # floor, not round: the pulse sits `value` slots *after* its word boundary, and any
        # value past half a word period would otherwise be attributed to the next word.
        word_indices = np.floor(relative / self.word_period).astype(np.int64)
        raw_values = relative - word_indices * self.word_period
        # A pulse landing in the dead zone (max_value, word_period) is ambiguous: either a late
        # word-k pulse near max_value, or an early word-(k+1) pulse near 0. Split the difference
        # at the middle of the dead zone. Testing against max_value itself is wrong — it wraps
        # every sync pulse whose measured value jitters a fraction above max_value.
        dead_zone_mid = 0.5 * (self.max_value + self.word_period)
        wrapped = raw_values > dead_zone_mid
        word_indices = word_indices.copy()
        word_indices[wrapped] += 1
        raw_values = raw_values.copy()
        raw_values[wrapped] -= self.word_period
        values = np.round(raw_values).clip(0, self.max_value).astype(np.uint16)
        return word_indices, values

    def _sync_frame(self, positions: np.ndarray) -> np.ndarray | None:
        pos_raw = np.sort(positions.astype(np.float64))
        if len(pos_raw) < self.sync_num:
            return None

        logger.debug(
            "Syncer: pulse gaps in frame, delta(t)=%s ns",
            np.round(np.diff(pos_raw) * self.nominal_slot_time_s * 1e9, 3).tolist(),
        )

        # Slot-time calibration from the sync section alone. Every sync word carries (almost)
        # the same PPM value, so consecutive sync pulses are spaced one whole word_period apart,
        # give or take the residual timing error — averaging sync_num-1 gaps is enough to locate
        # every pulse in the frame via the generous `margin`.
        head_gaps = np.diff(pos_raw[: self.sync_num])
        if len(head_gaps) > 0:
            # Fractional jitter of the sync-to-sync gaps (mean~1.0 for clean data, since gaps
            # should be ~word_period), scaled into an equivalent 3-sigma slot error accumulated
            # over the full PPM dynamic range — how bad the worst-case timing error would be for
            # a word way out at the far end of the range, given this level of gap jitter.
            gap_ratio = head_gaps / self.word_period
            with np.errstate(invalid="ignore", divide="ignore"):
                error_metric = np.log(gap_ratio.std() * 3 * (self.max_value + 1) / gap_ratio.mean())
            logger.debug("Syncer: gap-jitter error metric = %.6g", error_metric)
        scale = float(np.median(head_gaps)) / self.word_period if len(head_gaps) > 0 else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        pos = pos_raw / scale
        sync_pos, found, frame_start = self._locate_sync(pos)

        self._last_scale = scale
        self._last_calibrated_positions = pos
        if abs(scale - 1.0) > 1e-9:
            logger.debug(
                "Syncer: slot-time calibration scale=%.6g -> inferred slot_time=%.6g s "
                "(nominal was %.6g s)",
                scale, self.nominal_slot_time_s * scale, self.nominal_slot_time_s,
            )

        # Decode the sync words with the same rule as the data words, so they can be printed.
        # They are expected to read sync_value; deviation shows the residual timing error.
        sync_rel = sync_pos - frame_start - np.arange(self.sync_num) * self.word_period
        self._last_sync = np.round(sync_rel).clip(0, self.max_value).astype(np.uint16)
        self._last_found = found
        self._last_frame_start = frame_start
        # Diagnostics: sub-slot residual of each sync pulse from its expected slot (un-rounded,
        # can be negative), and how each sync-to-sync gap deviates from the nominal word_period.
        self._last_sync_residual = sync_rel - self.sync_value
        self._last_sync_gap_dev = np.diff(sync_pos) - self.word_period

        # Do NOT pre-cut at frame_start + sync_num*word_period: a word carrying PPM value 0
        # sits exactly on that boundary, so a fraction of a slot of error in frame_start would
        # delete it. Sync words are removed below by `word_indices >= sync_num`, which has the
        # wrap handling to place a boundary pulse in the right word.
        if len(pos) == 0:
            return np.array([], dtype=np.uint16)

        word_indices, values = self._decode_positions(pos, frame_start)

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
            self._last_sync = None
            self._last_frame_start = None
            self._last_scale = 1.0
            self._last_calibrated_positions = None
            self._last_sync_residual = None
            self._last_sync_gap_dev = None
            raw = np.asarray(frame)
            decoded = self._sync_frame(raw)
            if decoded is not None:
                results.append(decoded)
                self.sync_values.append(
                    self._last_sync
                    if self._last_sync is not None
                    else np.array([], dtype=np.uint16)
                )
                self.sync_found.append(self._last_found)
                self.frame_starts.append(
                    self._last_frame_start if self._last_frame_start is not None else 0.0
                )
                # Store the slot-time-calibrated positions, not the raw ones: frame_start and
                # word_period are both in the calibrated scale, so a plot overlaying raw pulses
                # on section bands needs pulses in that same scale to line up.
                self.raw_frames.append(
                    self._last_calibrated_positions
                    if self._last_calibrated_positions is not None
                    else raw
                )
                self.slot_scales.append(self._last_scale)
                self.inferred_slot_time_s.append(self.nominal_slot_time_s * self._last_scale)
                self.sync_residuals.append(
                    self._last_sync_residual
                    if self._last_sync_residual is not None
                    else np.array([])
                )
                self.sync_gap_devs.append(
                    self._last_sync_gap_dev if self._last_sync_gap_dev is not None else np.array([])
                )
            else:
                failed += 1
        logger.debug("Syncer: %d frames in, %d decoded, %d failed sync", len(signal), len(results), failed)
        if failed > 0:
            logger.warning("Syncer: %d/%d frames failed synchronization", failed, len(signal))
        return np.array(results, dtype=object)

    def reset(self) -> None:
        self.sync_values = []
        self.sync_found = []
        self.frame_starts = []
        self.raw_frames = []
        self.slot_scales = []
        self.inferred_slot_time_s = []
        self.sync_residuals = []
        self.sync_gap_devs = []
        self._last_sync = None
        self._last_frame_start = None
        self._last_scale = 1.0
        self._last_calibrated_positions = None
        self._last_sync_residual = None
        self._last_sync_gap_dev = None

    def __repr__(self) -> str:
        return (
            f"Syncer(sync_num={self.sync_num}, word_period={self.word_period}, "
            f"sync_value={self.sync_value})"
        )
