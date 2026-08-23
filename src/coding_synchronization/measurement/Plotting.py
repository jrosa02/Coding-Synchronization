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
    ax: Axes, x: np.ndarray, y: np.ndarray, title: str | None = None, ylabel: str = "Offset (slots)",
    frame_words: int | None = None,
) -> None:
    """`y` (raw offsets, or a residual against some assumed decode) vs. its natural index `x`,
    with an OLS line fit and a shaded standard-error band — visualizes exactly what Pass-1
    calibration does. The fitted x=0 intercept is marked with its own uncertainty, not pinned to
    0: it's expected to hover near zero, not be forced there.

    The fitted slope is reported with its own uncertainty and, when `frame_words` is given,
    extrapolated over a whole frame. A residual tilt only matters by how much it accumulates
    before the last word of the frame — and the axes autoscale to a residual spread of a few
    hundredths of a slot, so any slope at all draws as a dramatic diagonal without that number
    next to it.
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
    slope_se = math.sqrt(s2 / sxx)

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
    lines = [
        f"intercept = {intercept:.4g} ± {intercept_se:.4g} slots (not fixed to 0)",
        f"slope = {slope:.3g} ± {slope_se:.2g} slots/word",
    ]
    if frame_words:
        lines.append(
            f"  → {slope * frame_words:+.3g} slots accumulated over {frame_words} words "
            f"(decision boundary ±0.5)"
        )
    ax.text(
        0.02, 0.02, "\n".join(lines),
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

    `t` must be sorted ascending. Each kernel is only evaluated over the samples it actually
    touches (a pulse is a few samples wide, a whole-frame window is millions), so this stays
    linear in `len(t)` instead of building the full len(t) x len(centers) outer product.
    """
    t = np.asarray(t, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    out = np.zeros_like(t, dtype=np.float64)
    if len(centers) == 0:
        return out
    half_width = max(width, 1e-12) / 2.0
    lo = np.searchsorted(t, centers - half_width, side="left")
    hi = np.searchsorted(t, centers + half_width, side="right")
    for center, a, b in zip(centers, lo, hi, strict=True):
        if b > a:
            out[a:b] += amplitude * np.clip(
                1.0 - np.abs(t[a:b] - center) / half_width, 0.0, None
            )
    return out


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
    """Raw trace + detected-pulse ticks + a synthetic ideal-PPM overlay, over one frame (or a
    chosen span of its sections) — so a real bump with no ideal marker (or vice versa) is obvious.

    `section_bounds` is `[(name, start_s, end_s), ...]`, any of sync/metadata/data/ecc/eof,
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

    ax.plot(t, trace, linewidth=0.12, color="tab:green", label="raw signal", zorder=2)

    if len(detected_t):
        ax.vlines(
            detected_t, 0, detected_amp, color="black", linewidth=0.5, alpha=0.8,
            linestyle=":", label="detected pulse", zorder=3,
        )

    if len(ideal_t):
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
    title: str | None = None,
    xlabel: str = "Time relative to rising-edge crossing (s)",
    anchor_label: str = "rising edge",
    bins: tuple[int, int] = (800, 400),
    cmap: str = "turbo",
) -> None:
    """Persistence eye diagram: every sync pulse in the capture, re-anchored on a common
    reference (t=0), accumulated into a 2D density map the way a scope's persistence display
    builds one — color is how many traces pass through that (time, amplitude) bin, so the
    common path glows and a single stray trace stays dim.

    Traces are resampled onto a shared time grid and each column-to-column segment is rasterized
    over every amplitude bin it crosses, so a steep edge is a continuous line rather than the
    dotted trail of hits that point-sampling would leave. Bins no trace touches stay transparent.
    """
    from matplotlib.colors import LogNorm

    n_x, n_y = bins
    t_min = min(float(t[0]) for t in t_rel_list)
    t_max = max(float(t[-1]) for t in t_rel_list)
    t_edges = np.linspace(t_min, t_max, n_x + 1)
    # Off the ends of a given trace: NaN, so a short trace contributes nothing there rather
    # than a flat line at its own edge value.
    resampled = np.vstack([
        np.interp(t_edges, t_rel, trace, left=np.nan, right=np.nan)
        for t_rel, trace in zip(t_rel_list, trace_list, strict=True)
    ])

    y_min = float(np.nanmin(resampled))
    y_max = float(np.nanmax(resampled))
    if y_max <= y_min:
        y_max = y_min + 1e-12
    y_edges = np.linspace(y_min, y_max, n_y + 1)
    scaled = (resampled - y_min) / (y_max - y_min) * n_y
    # NaN (off the end of a trace) would warn on the int cast; the `valid` mask below drops
    # those columns anyway, so park them in bin 0.
    idx = np.clip(np.floor(np.nan_to_num(scaled, nan=0.0)).astype(np.int64), 0, n_y - 1)

    # One vertical span per (trace, column): mark its first and last amplitude bin, then a
    # cumulative sum down the amplitude axis fills everything in between — the same trick a
    # scan-line rasterizer uses, and far cheaper than sampling each segment densely.
    valid = np.isfinite(resampled[:, :-1]) & np.isfinite(resampled[:, 1:])
    lo = np.minimum(idx[:, :-1], idx[:, 1:])[valid]
    hi = np.maximum(idx[:, :-1], idx[:, 1:])[valid]
    cols = np.broadcast_to(np.arange(n_x), valid.shape)[valid]
    acc = np.zeros((n_y + 1, n_x), dtype=np.float64)
    np.add.at(acc, (lo, cols), 1.0)
    np.add.at(acc, (hi + 1, cols), -1.0)
    density = np.cumsum(acc, axis=0)[:n_y]

    im = ax.imshow(
        np.ma.masked_less_equal(density, 0.0), origin="lower", aspect="auto",
        extent=(t_edges[0], t_edges[-1], y_edges[0], y_edges[-1]), cmap=cmap,
        norm=LogNorm(vmin=1.0, vmax=max(float(density.max()), 2.0)), zorder=2,
    )
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("traces through this bin (log)")

    ax.axvline(
        0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=3,
        label=f"{anchor_label} ({len(t_rel_list)} pulses)",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
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
