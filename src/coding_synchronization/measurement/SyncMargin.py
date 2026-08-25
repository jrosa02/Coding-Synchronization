"""Shared math for propagating sync-word calibration uncertainty across a whole frame.

`Syncer` fits `frame_start`/`word_period` from a frame's sync section only (see
`Syncer._decode_positions`), then reuses that fit, unmodified, to place every other word in
the frame. This module holds the one OLS-fit implementation shared by
`Plotting.plot_offset_regression` (the existing sync-section-only diagnostic) and the
frame-wide margin/risk scripts (`plot_sync_margin.py`, `plot_decode_risk.py`,
`plot_margin_validation.py`), which extrapolate that same fit's prediction interval out to
word indices the fit never saw.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SyncFit:
    """An OLS fit of residual (slots) vs. word index, over some frame's sync words."""

    slope: float  # residual drift, slots/word — leftover scale error after calibration
    intercept: float  # fitted residual at word index 0
    intercept_se: float
    slope_se: float
    s2: float  # pooled residual variance (intrinsic per-pulse jitter^2)
    xbar: float
    sxx: float
    n: int


def fit_sync_residuals(x: np.ndarray, y: np.ndarray) -> SyncFit:
    """OLS fit of `y` (residual, slots) against `x` (word index). `n` must be >= 3.

    This is the same fit `Plotting.plot_offset_regression` draws — factored out here so the
    frame-wide margin/risk scripts extrapolate the identical model instead of a second,
    possibly-drifting reimplementation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        raise ValueError(f"fit_sync_residuals needs at least 3 points, got {n}")

    xbar, ybar = float(x.mean()), float(y.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    slope = float(np.sum((x - xbar) * (y - ybar)) / sxx)
    intercept = float(ybar - slope * xbar)
    resid = y - (slope * x + intercept)
    dof = n - 2
    s2 = float(np.sum(resid**2) / dof) if dof > 0 else 0.0
    intercept_se = math.sqrt(s2 * (1.0 / n + xbar**2 / sxx))
    slope_se = math.sqrt(s2 / sxx)

    return SyncFit(
        slope=slope, intercept=intercept, intercept_se=intercept_se, slope_se=slope_se,
        s2=s2, xbar=xbar, sxx=sxx, n=n,
    )


def predicted_sigma(fit: SyncFit, k: np.ndarray) -> np.ndarray:
    """sqrt(OLS prediction-interval variance) at word index `k`, extrapolated beyond the
    sync section `fit` came from: `s2 * (1 + 1/n + (k - xbar)^2 / sxx)`.

    Assumes any leftover miscalibration is a single linear term across the whole frame — it
    will not capture non-linear (e.g. curvature-shaped clock drift) timing error.
    """
    k = np.asarray(k, dtype=np.float64)
    return np.sqrt(fit.s2 * (1.0 + 1.0 / fit.n + (k - fit.xbar) ** 2 / fit.sxx))


def exceed_probability(sigma: np.ndarray, boundary: float = 0.5) -> np.ndarray:
    """P(|timing error| > boundary slots), assuming Gaussian jitter of std `sigma`."""
    sigma = np.asarray(sigma, dtype=np.float64)
    z = boundary / (sigma * math.sqrt(2.0))
    return np.vectorize(math.erfc)(z)


def frame_sync_residual(
    chunk: np.ndarray, sync_num: int, word_period: float, sync_value: int,
    calibration: str = "ls",
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Calibrate one chunk's sync section the same way Syncer's Pass 1 does.

    Returns (x, y_raw, scale, frame_start, decoded, residual). Identical algorithm to
    `plot_sync_regression.py`'s own `_frame_residual` — shared here so the frame-wide
    margin/risk scripts fit sync sections exactly the way that script's own independent
    (Model2-free) diagnostic does.
    """
    y_raw = chunk[:sync_num]
    x = np.arange(sync_num, dtype=np.float64)
    head_gaps = np.diff(y_raw) if sync_num > 1 else np.array([])
    scale = float(np.median(head_gaps)) / word_period if len(head_gaps) > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if calibration == "ls" and sync_num >= 3:
        refined = float(np.polyfit(x, y_raw, 1)[0]) / word_period
        if np.isfinite(refined) and refined > 0.0 and abs(refined / scale - 1.0) <= 0.01:
            scale = refined
    pos = y_raw / scale
    frame_start = float(np.mean(pos - x * word_period)) - sync_value
    decoded = frame_start + x * word_period + sync_value
    residual = pos - decoded
    return x, y_raw, scale, frame_start, decoded, residual


def per_frame_fits(
    chunks: list[np.ndarray], sync_num: int, word_period: float, sync_value: int,
    calibration: str = "ls",
) -> list[tuple[int, SyncFit]]:
    """One independent `SyncFit` per usable chunk — `(original chunk index, fit)`.

    Each fit uses only that chunk's own `sync_num` points (`n=sync_num`, `dof=sync_num-2`).
    No chunk's fit depends on any other chunk's data: an earlier pooled version of this
    function concatenated every chunk's sync residuals into one combined fit, which made the
    calibration-uncertainty term of `predicted_sigma` shrink as more frames were pooled —
    backwards, since real frame-to-frame slot-time disagreement doesn't average away. Each
    frame's own noisy, small-n fit is the honest independent result.
    """
    if sync_num < 3:
        raise ValueError(
            f"per_frame_fits needs sync_num >= 3 to fit a frame's own sync section "
            f"(slope + residual variance need at least 3 points); got sync_num={sync_num}"
        )
    fits: list[tuple[int, SyncFit]] = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < sync_num:
            continue
        x, _, _, _, _, residual = frame_sync_residual(
            chunk, sync_num, word_period, sync_value, calibration
        )
        fits.append((i, fit_sync_residuals(x, residual)))
    return fits
