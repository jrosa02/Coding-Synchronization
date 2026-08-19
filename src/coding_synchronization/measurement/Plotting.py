import logging
import math

import numpy as np
from matplotlib.axes import Axes

from coding_synchronization.encoder import FrameParams

logger = logging.getLogger(__name__)

# Section order matches transmission order within one frame period: sync words, metadata
# words, data words, ecc words, then the eof guard interval (dead time, no pulses at all).
_SECTION_COLORS: dict[str, str] = {
    "sync": "tab:blue",
    "metadata": "tab:orange",
    "data": "tab:green",
    "ecc": "tab:red",
    "eof": "0.5",
}


def frame_sections(frame_params: FrameParams) -> list[tuple[str, int]]:
    """(name, word count) for each section of one frame period, in transmission order."""
    return [
        ("sync", frame_params.sync_num),
        ("metadata", frame_params.metadata_num),
        ("data", frame_params.data_num),
        ("ecc", frame_params.ecc_num),
        ("eof", frame_params.eof_num),
    ]

def plot_trace(ax: Axes, values: np.ndarray, dt_s: float, label: str, **kw) -> None:
    """Plot the full-resolution trace — every sample, no decimation."""
    t = np.arange(len(values), dtype=np.float64) * dt_s
    ax.plot(t, values, linewidth=0.5, label=label, **kw)


def plot_gap_histogram(
    ax: Axes, offsets_slots: np.ndarray, split_threshold: float | None = None, bins: int = 200
) -> None:
    """Inter-pulse gap distribution in slot units — says at a glance if splitting will work."""
    if len(offsets_slots) < 2:
        ax.text(0.5, 0.5, "not enough pulses", ha="center", va="center", transform=ax.transAxes)
        return
    gaps = np.diff(np.sort(offsets_slots))
    ax.hist(gaps, bins=bins, log=True)
    if split_threshold is not None:
        ax.axvline(
            split_threshold,
            color="red",
            linestyle="--",
            label=f"split threshold={split_threshold:.0f}",
        )
        n_frames = int(np.sum(gaps > split_threshold)) + 1
        ax.set_title(f"Inter-pulse gaps (slots) — {n_frames} frames at this threshold")
        ax.legend()
    else:
        ax.set_title("Inter-pulse gaps (slots)")
    ax.set_xlabel("Gap (slots)")
    ax.set_ylabel("Count (log)")
    ax.grid(True)


def plot_offset_regression(
    ax: Axes, x: np.ndarray, y: np.ndarray, title: str | None = None, ylabel: str = "Offset (slots)"
) -> None:
    """`y` (raw offsets, or a residual against some assumed decode) vs. its natural index `x`,
    with an OLS line fit and a shaded standard-error band — visualizes exactly what Pass-1
    calibration does. The fitted x=0 intercept is marked with its own uncertainty, not pinned to
    0: it's expected to hover near zero, not be forced there.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    ax.scatter(x, y, color="tab:blue", s=24, zorder=3, label="data")

    if n < 3:
        ax.set_xlabel("Index")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if title:
            ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        return

    xbar, ybar = x.mean(), y.mean()
    sxx = float(np.sum((x - xbar) ** 2))
    slope = float(np.sum((x - xbar) * (y - ybar)) / sxx)
    intercept = float(ybar - slope * xbar)
    resid = y - (slope * x + intercept)
    dof = n - 2
    s2 = float(np.sum(resid**2) / dof) if dof > 0 else 0.0
    intercept_se = math.sqrt(s2 * (1.0 / n + xbar**2 / sxx))

    x_fit = np.linspace(min(0.0, x.min()), x.max(), 200)
    y_fit = slope * x_fit + intercept
    se_fit = np.sqrt(s2 * (1.0 / n + (x_fit - xbar) ** 2 / sxx))

    ax.plot(x_fit, y_fit, color="tab:red", linewidth=1.2, zorder=2, label="OLS fit")
    ax.fill_between(
        x_fit, y_fit - se_fit, y_fit + se_fit, color="tab:red", alpha=0.25, zorder=1,
        label="±1 SE",
    )
    ax.fill_between(
        x_fit, y_fit - 2 * se_fit, y_fit + 2 * se_fit, color="tab:red", alpha=0.12, zorder=0,
        label="±2 SE",
    )

    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6, label="y=0")
    ax.errorbar(
        [0.0], [intercept], yerr=[intercept_se], fmt="D", color="black", markersize=6,
        capsize=4, zorder=4, label="fitted intercept",
    )
    # Fixed corner, not anchored to the data point: the intercept can land anywhere in the
    # y-range (including right under the title), so tying the label's position to it directly
    # risks exactly that overlap.
    ax.text(
        0.02, 0.02, f"intercept = {intercept:.4g} ± {intercept_se:.4g} slots\n(not fixed to 0)",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )

    ax.set_xlabel("Index")
    ax.set_ylabel("Offset (slots)")
    ax.grid(True)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def ideal_pulse_train(
    t: np.ndarray, centers: np.ndarray, width: float, amplitude: float
) -> np.ndarray:
    """Sum of triangular kernels at `centers`, evaluated at `t` — a synthetic "ideal PPM" trace
    directly comparable (same time/amplitude units) to a real detected pulse train.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) == 0:
        return np.zeros_like(t, dtype=np.float64)
    half_width = max(width, 1e-12) / 2.0
    dist = np.abs(t[:, None] - centers[None, :])
    kernels = amplitude * np.clip(1.0 - dist / half_width, 0.0, None)
    return kernels.sum(axis=1)


def plot_frame_detail(
    ax: Axes,
    t: np.ndarray,
    trace: np.ndarray,
    detected_t: np.ndarray,
    detected_amp: np.ndarray,
    ideal_t: np.ndarray,
    ideal_width: float,
    ideal_amp: float,
    section_bounds: list[tuple[str, float, float]],
    slot_positions: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Raw trace + detected-pulse ticks + a synthetic ideal-PPM overlay, over one frame's
    sync/metadata window — so a real bump with no ideal marker (or vice versa) is obvious.

    `section_bounds` is `[(name, start_s, end_s), ...]` (only sync/metadata expected here),
    reusing `_SECTION_COLORS` for shading. `slot_positions`, if given, draws a very faint
    full-height line at every individual PPM slot boundary — a fine ruler to count slots against
    once zoomed in, well below the section/pulse/ideal layers. `t`/`detected_t`/`ideal_t` are all
    in seconds. Meant to be viewed interactively (`plt.show()`) — a real pulse is only 1-3 samples
    wide, so zoom/pan into individual words to actually see the ideal-PPM shape at native
    resolution.
    """
    if slot_positions is not None and len(slot_positions):
        ax.vlines(
            slot_positions, 0, 1, transform=ax.get_xaxis_transform(),
            color="0.3", linewidth=0.4, alpha=0.5, zorder=0,
        )
        ax.grid(False)

    for name, start, end in section_bounds:
        if end > start:
            ax.axvspan(start, end, color=_SECTION_COLORS[name], alpha=0.3, label=name)

    ax.plot(t, trace, linewidth=0.6, color="tab:green", label="raw signal", zorder=2)

    if len(detected_t):
        ax.vlines(
            detected_t, 0, detected_amp, color="black", linewidth=1.2, alpha=0.8,
            label="detected pulse", zorder=3,
        )

    ideal_trace = ideal_pulse_train(t, ideal_t, ideal_width, ideal_amp)
    ax.plot(
        t, ideal_trace, linewidth=1.0, linestyle="--", color="tab:purple", alpha=0.8,
        label="ideal PPM", zorder=1,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles, strict=True))
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8, ncol=len(seen))


def plot_sync_eye(
    ax: Axes,
    t_rel_list: list[np.ndarray],
    trace_list: list[np.ndarray],
    decoded_values: np.ndarray,
    sync_value: int,
    title: str | None = None,
) -> None:
    """Eye-diagram overlay: every sync pulse in the capture, each re-anchored on its own
    rising-edge crossing (t=0), stacked on one axes — pulse-shape/jitter/amplitude consistency
    is directly visible, and each trace is colored by its decoded PPM value (should all read
    `sync_value`; any other color is a sync word that decoded wrong).
    """
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    spread = float(np.max(np.abs(decoded_values.astype(np.float64) - sync_value))) if len(decoded_values) else 0.0
    spread = max(spread, 1.0)
    norm = Normalize(vmin=sync_value - spread, vmax=sync_value + spread)
    cmap = colormaps["coolwarm"]

    for t_rel, trace, value in zip(t_rel_list, trace_list, decoded_values, strict=True):
        ax.plot(t_rel, trace, linewidth=0.6, alpha=0.25, color=cmap(norm(float(value))))

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="rising edge")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax)
    cbar.set_label(f"decoded PPM value (expected {sync_value})")

    ax.set_xlabel("Time relative to rising-edge crossing (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def plot_frame_sections(
    ax: Axes,
    positions: np.ndarray,
    frame_start: float,
    word_period: float,
    frame_params: FrameParams,
    title: str | None = None,
) -> None:
    """Semi-transparent bands over one frame's raw pulses, marking sync/metadata/data/ecc/eof.

    `positions` are the frame-local pulse positions (slot units) that fed the Syncer for this
    frame; `frame_start` is the fitted word-0 boundary in the same units (Syncer.frame_starts).
    Pulses are drawn relative to `frame_start`, so a well-decoded frame lines up its pulses
    exactly with the section band they were assigned to.
    """
    rel = np.asarray(positions, dtype=np.float64) - frame_start
    cursor = 0.0
    for name, n in frame_sections(frame_params):
        width = n * word_period
        if width > 0:
            ax.axvspan(cursor, cursor + width, color=_SECTION_COLORS[name], alpha=0.25, label=name)
        cursor += width

    margin = word_period
    in_view = rel[(rel >= -margin) & (rel <= cursor + margin)]
    ax.vlines(in_view, 0, 1, color="black", linewidth=1.0, alpha=0.8, label="detected pulse")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(-margin, cursor + margin)
    ax.set_yticks([])
    ax.set_xlabel("Slot (relative to fitted frame start)")
    ax.grid(True, axis="x")
    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles, strict=True))
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7, ncol=len(seen))
