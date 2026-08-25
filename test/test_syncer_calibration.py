"""Pass-1 slot-time calibration: unbiased, and still robust to a missing sync pulse.

SimpleSyncer calibrates the slot time from the sync section alone. The median sync-to-sync gap survives
a dropped pulse, but it is not the least-squares slope, and the difference is a residual tilt that
accumulates over the frame — parts per million of scale error become a fraction of a slot by the
last word, which is what pushed real captures past the +-0.5-slot decision boundary. These tests
pin both properties of the two-step estimator: the median locates, the least-squares refit sets
the scale.
"""

import numpy as np

from coding_synchronization.decoder.SimpleSyncer import SimpleSyncer
from coding_synchronization.encoder import ModulationParams

PPM_RANK = 10
DEAD_SLOTS = 16
WORD_PERIOD = (1 << PPM_RANK) + DEAD_SLOTS
SYNC_NUM = 8


def _syncer(sync_value: int = 0) -> SimpleSyncer:
    mod_params = ModulationParams(
        ppm_rank=PPM_RANK, slot_time=np.float64(1e-6), dead_slots=DEAD_SLOTS
    )
    return SimpleSyncer(mod_params, sync_num=SYNC_NUM, sync_value=sync_value)


def _sync_section(rng: np.random.Generator, sigma: float, sync_value: int = 0) -> np.ndarray:
    """One frame's sync pulses on a perfect grid, plus independent per-pulse timing noise."""
    k = np.arange(SYNC_NUM, dtype=np.float64)
    return k * WORD_PERIOD + sync_value + rng.normal(0.0, sigma, SYNC_NUM)


def _residual_slope(syncer: SimpleSyncer, positions: np.ndarray) -> float:
    """Slope of the sync residual against word index — the diagonal the regression plot draws."""
    syncer._sync_frame(positions)
    residual = syncer._last_sync_residual
    assert residual is not None
    return float(np.polyfit(np.arange(SYNC_NUM, dtype=np.float64), residual, 1)[0])


def test_refined_scale_tilts_far_less_than_the_median_estimator():
    """The refined calibration must tilt far less than the median-gap estimator on the same data.

    Some tilt is always present, so this asserts an improvement rather than its absence. The same
    frames go through both estimators, which shows the tilt belongs to the estimator and not to
    the data.
    """
    rng = np.random.default_rng(20260819)
    syncer = _syncer()
    frames = [_sync_section(rng, sigma=0.02) for _ in range(400)]

    refined = np.array([_residual_slope(syncer, f) for f in frames])

    # Same frames, median-gap scale only — replicated here rather than reached for in SimpleSyncer,
    # which no longer offers it.
    x = np.arange(SYNC_NUM, dtype=np.float64)
    median_only = []
    for f in frames:
        scale = float(np.median(np.diff(f))) / WORD_PERIOD
        pos = f / scale
        median_only.append(float(np.polyfit(x, pos - (np.mean(pos - x * WORD_PERIOD) + x * WORD_PERIOD), 1)[0]))
    median_only = np.array(median_only)

    # The median-only path must show the bias this calibration was changed to remove, otherwise
    # the comparison below would pass for the wrong reason.
    assert abs(median_only.mean()) > 1e-4, "the median estimator should tilt on this data"
    assert abs(refined.mean()) < 0.01 * abs(median_only.mean()), (
        f"refined tilt {refined.mean():+.3g} is not far below the median tilt "
        f"{median_only.mean():+.3g} slots/word"
    )


def test_scale_survives_a_missing_sync_pulse():
    """A dropped sync pulse must not drag the scale off.

    This is why the coarse pass stays median-based and why the refit uses each located pulse's own
    word index: fitting against consecutive indices instead would read the 2-word gap as one word
    and inflate the scale by ~20%.
    """
    rng = np.random.default_rng(7)
    syncer = _syncer()
    # A real frame carries data words after the sync section. Without them the frame holds fewer
    # than sync_num pulses, and _sync_frame rejects it before it ever calibrates.
    data_k = np.arange(SYNC_NUM, SYNC_NUM + 20, dtype=np.float64)
    data = data_k * WORD_PERIOD + rng.integers(0, 1 << PPM_RANK, len(data_k))
    frame = np.concatenate([_sync_section(rng, sigma=0.02), data])
    dropped = np.delete(frame, 4)

    syncer._sync_frame(dropped)
    scale = syncer._last_scale
    assert syncer._last_found == SYNC_NUM - 1, "the missing pulse should be the only one not found"
    assert abs(scale - 1.0) < 1e-4, f"scale drifted to {scale} with one sync pulse missing"

    # The failure mode being guarded against, for contrast: a fit against consecutive indices
    # reads the two-word gap as one word. How far that lands depends on the frame length, so this
    # compares the two errors rather than a fixed figure.
    naive = float(np.polyfit(np.arange(len(dropped), dtype=np.float64), dropped, 1)[0])
    naive_error = abs(naive / WORD_PERIOD - 1.0)
    assert naive_error > 100 * abs(scale - 1.0), (
        f"consecutive-index fit is off by {naive_error:.3%}, the word-index fit by "
        f"{abs(scale - 1.0):.3%}"
    )


def test_residual_is_independent_of_sync_value():
    """sync_value cancels: frame_start subtracts it and the decode adds it back.

    A tilted or offset residual therefore never indicates a wrong --sync-value.
    """
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.02, SYNC_NUM)
    k = np.arange(SYNC_NUM, dtype=np.float64)

    residuals = []
    for sync_value in (0, 512, 1023):
        syncer = _syncer(sync_value=sync_value)
        syncer._sync_frame(k * WORD_PERIOD + sync_value + noise)
        residuals.append(syncer._last_sync_residual.copy())

    for other in residuals[1:]:
        np.testing.assert_allclose(residuals[0], other, atol=1e-9)
