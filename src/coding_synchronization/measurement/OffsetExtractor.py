import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from coding_synchronization.measurement.WaveformLoader import Waveform

logger = logging.getLogger(__name__)

_BASELINE_SAMPLES = 2_000_000  # cap for the median/mean estimate — strided, not truncated


@dataclass
class ExtractionParams:
    threshold: float | None = None  # in signal units (volts); None -> auto
    polarity: Literal["auto", "pos", "neg"] = "auto"
    baseline: Literal["median", "mean", "none"] = "median"
    # The measuring device already inverts one leg of the pair, so the two zero-centred
    # channels are summed. Use "sub" for a pair that was captured without that inversion.
    combine: Literal["add", "sub"] = "add"
    hysteresis: float = 0.5  # a run ends only below hysteresis * threshold
    min_width_samples: int = 1
    # Two detections closer than this are the same physical pulse (reflection / re-trigger):
    # PPM guarantees consecutive pulses are at least dead_slots + 1 slots apart, so anything
    # closer is an artefact. 0 disables the check.
    min_separation_samples: float = 0.0
    # centroid: amplitude-weighted centroid; peak: plain argmax; edge: mean of the
    # interpolated rising-edge (crosses threshold) / falling-edge (crosses re-arm) positions
    # — most precise for clean, flat-topped pulses. rising: interpolated rising-edge crossing
    # only — ignores the falling edge entirely, so a variable pulse width/trailing reflection
    # doesn't pull the position around; the leading edge alone carries the timing.
    method: Literal["centroid", "peak", "edge", "rising"] = "centroid"


@dataclass
class Differential:
    values: np.ndarray
    baseline_a: float
    baseline_b: float
    polarity: Literal["pos", "neg"]
    combine: Literal["add", "sub"] = "add"

    @property
    def baseline(self) -> float:
        """Combined pedestal that was removed, for reporting."""
        if self.combine == "add":
            return self.baseline_a + self.baseline_b
        return self.baseline_a - self.baseline_b


@dataclass
class Offsets:
    samples: np.ndarray  # sub-sample pulse positions, float
    amplitudes: np.ndarray
    widths: np.ndarray
    dt_s: float
    threshold: float
    polarity: str
    baseline: float
    combine: str = "add"

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def seconds(self) -> np.ndarray:
        return self.samples * self.dt_s

    def to_slots(self, slot_time_s: float) -> np.ndarray:
        return self.seconds / float(slot_time_s)

    def __repr__(self) -> str:
        return f"Offsets(n_pulses={len(self)}, threshold={self.threshold:.6g})"


def _baseline(values: np.ndarray, mode: str) -> float:
    if mode == "none":
        return 0.0
    stride = max(1, len(values) // _BASELINE_SAMPLES)
    sub = values[::stride]
    return float(np.median(sub) if mode == "median" else np.mean(sub))


def differential(wf: Waveform, params: ExtractionParams) -> Differential:
    """Zero-centre each channel, then combine the pair, sign-normalised so pulses point up.

    The scope inverts one leg of the differential pair, so the default combination is A + B once
    both channels have had their own DC pedestal removed. `combine="sub"` gives the classic A - B.
    """
    base_a = _baseline(wf.ch_a, params.baseline)
    base_b = _baseline(wf.ch_b, params.baseline)
    a = wf.ch_a.astype(np.float32) - np.float32(base_a)
    b = wf.ch_b.astype(np.float32) - np.float32(base_b)
    values = a + b if params.combine == "add" else a - b
    del a, b

    polarity: Literal["pos", "neg"] = "pos"
    if params.polarity == "auto":
        stride = max(1, len(values) // _BASELINE_SAMPLES)
        sub = values[::stride]
        pos_ex = float(np.percentile(sub, 99.9))
        neg_ex = -float(np.percentile(sub, 0.1))
        polarity = "neg" if neg_ex > pos_ex else "pos"
        logger.info(
            "Polarity auto-detect: +excursion=%.6g, -excursion=%.6g -> %s",
            pos_ex, neg_ex, polarity,
        )
    else:
        polarity = params.polarity

    if polarity == "neg":
        logger.info("Inverting combined signal (pulses point down in this capture)")
        values = -values

    op = "+" if params.combine == "add" else "-"
    logger.info(
        "Combined A %s B: baseline_a=%.6g, baseline_b=%.6g, polarity=%s, range=[%.6g, %.6g]",
        op, base_a, base_b, polarity, float(values.min()), float(values.max()),
    )
    return Differential(
        values=values,
        baseline_a=base_a,
        baseline_b=base_b,
        polarity=polarity,
        combine=params.combine,
    )


def auto_threshold(values: np.ndarray) -> float:
    """Half-amplitude point between the (already removed) baseline and the typical pulse peak.

    Two passes, because a percentile of the whole record is useless when pulses are sparse — the
    99.9th percentile of a record that is 0.5% pulses still sits in the noise. Pass 1 sets a coarse
    gate from the noise sigma (MAD-based, so the pulses themselves don't inflate it); pass 2 takes
    the median of the peaks found above that gate as the pulse amplitude.
    """
    stride = max(1, len(values) // _BASELINE_SAMPLES)
    sub = values[::stride]
    peak_max = float(np.max(sub))
    if peak_max <= 0.0:
        raise ValueError(
            "Auto-threshold failed: no positive excursion found after baseline removal. "
            "Check --polarity / --combine / --pos-col / --neg-col."
        )
    sigma = 1.4826 * float(np.median(np.abs(sub - np.median(sub))))
    coarse = max(6.0 * sigma, 0.05 * peak_max)

    starts, _ = _runs(values > np.float32(coarse))
    if len(starts) == 0:
        thr = 0.5 * peak_max
        logger.warning(
            "Auto-threshold: nothing above the coarse gate %.6g — falling back to %.6g",
            coarse, thr,
        )
        return thr

    amplitude = float(np.median(np.maximum.reduceat(values, starts)))
    thr = 0.5 * amplitude
    logger.info(
        "Auto-threshold: noise sigma=%.6g, coarse gate=%.6g, %d candidate pulses, "
        "median amplitude=%.6g -> threshold=%.6g",
        sigma, coarse, len(starts), amplitude, thr,
    )
    return thr


def _runs(above: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of contiguous True runs."""
    if len(above) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    edges = np.diff(above.view(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.concatenate(([0], starts))
    if above[-1]:
        ends = np.concatenate((ends, [len(above)]))
    return starts.astype(np.int64), ends.astype(np.int64)


def _crossing(values: np.ndarray, i0: int, i1: int, level: float) -> float:
    """Sub-sample index in [i0, i1] where the segment linearly crosses `level`."""
    v0, v1 = float(values[i0]), float(values[i1])
    if v1 == v0:
        return float(i0)
    return i0 + (level - v0) / (v1 - v0)


def rising_edge_crossing(
    values: np.ndarray, approx_sample: float, thr: float, search_radius: int = 50
) -> float:
    """Re-derive the precise rising-edge (threshold) crossing near `approx_sample`, regardless
    of which `ExtractionParams.method` was used to originally locate the pulse — used to anchor
    a pulse on its own leading edge (e.g. an eye-diagram overlay) independent of extraction mode.
    """
    lo = max(0, int(approx_sample) - search_radius)
    hi = min(len(values), int(approx_sample) + search_radius)
    above = np.flatnonzero(values[lo:hi] > thr)
    if len(above) == 0:
        return float(approx_sample)
    i_rise = lo + int(above[0])
    return _crossing(values, i_rise - 1, i_rise, thr) if i_rise > 0 else float(i_rise)


def _edge_positions(
    values: np.ndarray, starts: np.ndarray, ends: np.ndarray, thr: float, thr_low: float
) -> np.ndarray:
    """Pulse centre = mean of the interpolated rising-edge (thr) / falling-edge (thr_low)
    crossings. Assumes a clean, monotonic edge on either side of the pulse top.
    """
    n = len(values)
    positions = np.empty(len(starts), dtype=np.float64)
    for k, (s, e) in enumerate(zip(starts, ends, strict=True)):
        above = np.flatnonzero(values[s:e] > thr)
        i_rise = s + int(above[0])
        rise_pos = _crossing(values, i_rise - 1, i_rise, thr) if i_rise > 0 else float(i_rise)
        if e < n:
            fall_pos = _crossing(values, e - 1, e, thr_low)
        else:
            fall_pos = float(e - 1)
        positions[k] = 0.5 * (rise_pos + fall_pos)
    return positions


def _rising_edge_positions(
    values: np.ndarray, starts: np.ndarray, ends: np.ndarray, thr: float
) -> np.ndarray:
    """Pulse position = interpolated rising-edge (threshold) crossing only — the falling edge is
    never consulted, so trailing-edge jitter (reflections, variable pulse width) can't move it.
    """
    positions = np.empty(len(starts), dtype=np.float64)
    for k, (s, e) in enumerate(zip(starts, ends, strict=True)):
        above = np.flatnonzero(values[s:e] > thr)
        i_rise = s + int(above[0])
        positions[k] = _crossing(values, i_rise - 1, i_rise, thr) if i_rise > 0 else float(i_rise)
    return positions


def _enforce_min_separation(
    positions: np.ndarray, amplitudes: np.ndarray, min_sep: float
) -> np.ndarray:
    """Keep the strongest detection out of any cluster spaced closer than `min_sep`."""
    keep = np.ones(len(positions), dtype=bool)
    last = 0
    for i in range(1, len(positions)):
        if positions[i] - positions[last] < min_sep:
            # same physical pulse — keep whichever detection is taller
            if amplitudes[i] > amplitudes[last]:
                keep[last] = False
                last = i
            else:
                keep[i] = False
        else:
            last = i
    return keep


def extract_offsets(
    diff: Differential,
    dt_s: float,
    params: ExtractionParams,
    threshold: float | None = None,
) -> Offsets:
    """Group above-threshold samples into pulses and take each pulse's centroid (or peak)."""
    values = diff.values
    thr = threshold if threshold is not None else params.threshold
    if thr is None:
        thr = auto_threshold(values)
    thr_low = float(thr) * float(params.hysteresis)

    starts, ends = _runs(values > np.float32(thr_low))
    empty = Offsets(
        samples=np.empty(0),
        amplitudes=np.empty(0),
        widths=np.empty(0, dtype=np.int64),
        dt_s=dt_s,
        threshold=float(thr),
        polarity=diff.polarity,
        baseline=diff.baseline,
        combine=diff.combine,
    )
    if len(starts) == 0:
        logger.warning("extract_offsets: no samples above %.6g — threshold too high?", thr_low)
        return empty

    widths = ends - starts
    run_max = np.maximum.reduceat(values, starts).astype(np.float64)
    keep = (run_max > thr) & (widths >= params.min_width_samples)
    n_drop_amp = int(np.sum(run_max <= thr))
    n_drop_width = int(np.sum((run_max > thr) & (widths < params.min_width_samples)))
    if not np.any(keep):
        logger.warning("extract_offsets: %d runs found, none exceeded %.6g", len(starts), thr)
        return empty

    if params.method == "centroid":
        # Outside a run the weight is exactly 0, so reduceat over `starts` alone is correct.
        w = np.clip(values.astype(np.float64) - thr_low, 0.0, None)
        idx = np.arange(len(values), dtype=np.float64)
        sum_w = np.add.reduceat(w, starts)
        sum_iw = np.add.reduceat(w * idx, starts)
        with np.errstate(invalid="ignore", divide="ignore"):
            positions = sum_iw / sum_w
        positions = positions[keep]
    elif params.method == "edge":
        positions = _edge_positions(values, starts[keep], ends[keep], thr, thr_low)
    elif params.method == "rising":
        positions = _rising_edge_positions(values, starts[keep], ends[keep], thr)
    else:
        runs = zip(starts[keep], ends[keep], strict=True)
        positions = np.array(
            [s + int(np.argmax(values[s:e])) for s, e in runs], dtype=np.float64
        )

    amps = run_max[keep]
    kept_widths = widths[keep]
    if params.min_separation_samples > 0.0 and len(positions) > 1:
        survivors = _enforce_min_separation(positions, amps, params.min_separation_samples)
        n_merged = len(positions) - int(survivors.sum())
        if n_merged:
            logger.info(
                "extract_offsets: merged %d detection(s) closer than %.2f samples "
                "(same physical pulse)",
                n_merged, params.min_separation_samples,
            )
        positions, amps, kept_widths = positions[survivors], amps[survivors], kept_widths[survivors]

    offsets = Offsets(
        samples=positions,
        amplitudes=amps,
        widths=kept_widths,
        dt_s=dt_s,
        threshold=float(thr),
        polarity=diff.polarity,
        baseline=diff.baseline,
        combine=diff.combine,
    )
    logger.info(
        "extract_offsets: %d pulses (threshold=%.6g, re-arm=%.6g, dropped %d low / %d narrow)",
        len(offsets), thr, thr_low, n_drop_amp, n_drop_width,
    )
    if len(positions) > 1:
        logger.debug(
            "extract_offsets: pulse gaps, delta(t)=%s ns",
            np.round(np.diff(positions) * dt_s * 1e9, 3).tolist(),
        )
    return offsets


def save_offsets(
    path: Path, offsets: Offsets, slot_time_s: float, meta: dict[str, Any] | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    slots = offsets.to_slots(slot_time_s)
    np.savez(
        path,
        offsets_slots=slots,
        offsets_s=offsets.seconds,
        offsets_samples=offsets.samples,
        amplitudes=offsets.amplitudes,
        widths=offsets.widths,
    )
    info: dict[str, Any] = {
        "n_pulses": len(offsets),
        "dt_s": offsets.dt_s,
        "slot_time_s": float(slot_time_s),
        "threshold": offsets.threshold,
        "polarity": offsets.polarity,
        "combine": offsets.combine,
        "baseline": offsets.baseline,
    }
    if meta:
        info.update(meta)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str))
    logger.info("Saved %d offsets to %s (+ .json)", len(offsets), path)


def load_offsets(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (offsets in slot units, metadata dict) — the script2 -> script3 hand-off."""
    path = Path(path)
    with np.load(path) as npz:
        slots = np.asarray(npz["offsets_slots"], dtype=np.float64)
    meta_path = path.with_suffix(".json")
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    logger.info("Loaded %d offsets (slot units) from %s", len(slots), path)
    return slots, meta


def params_to_dict(params: ExtractionParams) -> dict[str, Any]:
    return asdict(params)
