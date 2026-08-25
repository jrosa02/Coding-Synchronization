import logging

import numpy as np

from coding_synchronization.decoder.SimpleSyncer import SimpleSyncer
from coding_synchronization.encoder.Modulation import ModulationParams

logger = logging.getLogger(__name__)


class TwoPointSync(SimpleSyncer):
    """A `SimpleSyncer` that also tracks how the scale drifts frame to frame.

    Sync-pulse acquisition (`_acquire`, inherited unchanged) is identical to `SimpleSyncer`:
    `frame_start` and this frame's own single-frame scale `spread_i` come from exactly the same
    fit. What differs is the decode step: `SimpleSyncer` applies `spread_i` as one flat scale
    across the whole frame, which is why it can only absorb a *constant* Doppler rate (see
    `docs/math.md` section 7). `TwoPointSync` instead fits the one line through
    `(tstart_{i-1}, spread_{i-1})` and `(tstart_i, spread_i)` — the previous and current frame's
    own fitted points — and applies its slope as a piecewise-linear scale correction across the
    current frame, absorbing the *curvature* term a flat fit leaves behind.
    """

    def __init__(
        self,
        modparam: ModulationParams,
        sync_num: int = 4,
        sync_value: int | None = None,
        frame_duration_slots: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__(modparam, sync_num, sync_value, seed)
        # The nominal number of slots between one frame's start and the next, in the same
        # calibrated units as `frame_start` — needed because Splitter zeroes each frame chunk
        # locally, so `frame_start` alone cannot tell how far apart two frames actually are.
        self.frame_duration_slots = float(frame_duration_slots)
        self._prev_tstart_abs: float | None = None
        self._prev_spread: float | None = None
        # The two-point slope applied to each decoded frame — one entry per successfully
        # synced frame, same indexing as `slot_scales`. Diagnostic only: a plot needs this to
        # draw the piecewise-linear correction as the sloped segment it actually is, rather than
        # `slot_scales[i]` alone, which is only the flat scale at the frame's own start.
        self.slopes: list[float] = []
        logger.info(
            "TwoPointSync initialized: sync_num=%d, word_period=%d, frame_duration_slots=%.6g",
            sync_num, self.word_period, self.frame_duration_slots,
        )

    def _sync_frame(self, positions: np.ndarray) -> np.ndarray | None:
        acquired = self._acquire(positions)
        if acquired is None:
            return None
        pos, spread, frame_start = acquired

        # Put this frame's frame_start on the same continuous axis as the previous one, by
        # adding back the nominal gap Splitter's per-chunk zeroing removed.
        tstart_abs = (
            self._prev_tstart_abs + self.frame_duration_slots
            if self._prev_tstart_abs is not None
            else frame_start
        )

        if self._prev_tstart_abs is not None and self._prev_spread is not None:
            dt = tstart_abs - self._prev_tstart_abs
            slope = (spread - self._prev_spread) / dt if dt != 0.0 else 0.0
        else:
            # First frame of a run, or the frame right after a sync failure: no previous point,
            # so fall back to SimpleSyncer's flat scale for this one frame.
            slope = 0.0

        self._prev_tstart_abs = tstart_abs
        self._prev_spread = spread
        self.slopes.append(slope)

        if len(pos) == 0:
            return np.array([], dtype=np.uint16)

        # Piecewise-linear scale correction: `pos` is already calibrated by the flat `spread`,
        # so the leftover position-dependent error, relative to frame_start, is
        # slope*(pos - frame_start) in calibrated slot units (slope is a fractional
        # scale-per-slot rate, so this term stays tiny by construction — parts per billion of a
        # few hundred slots).
        pos_corrected = pos - slope * (pos - frame_start)

        return self._finish_decode(pos_corrected, frame_start)

    def reset(self) -> None:
        super().reset()
        self._prev_tstart_abs = None
        self._prev_spread = None
        self.slopes = []

    def __repr__(self) -> str:
        return (
            f"TwoPointSync(sync_num={self.sync_num}, word_period={self.word_period}, "
            f"sync_value={self.sync_value}, frame_duration_slots={self.frame_duration_slots:.6g})"
        )
