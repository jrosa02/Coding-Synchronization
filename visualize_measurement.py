"""Visualize a two-column differential scope capture.

    python visualize_measurement.py measurments/RefCurve_2026-08-18_0_124555.Wfm.csv

Prints the sniffed CSV format, then plots the raw pair, the zero-centred pair, the combined signal
(added by default — the measuring device inverts one leg) with the detection threshold and the
extracted pulse centroids, and the inter-pulse gap histogram in slot units (the plot that says
whether frame splitting will work).
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization._logging import setup_logging
from coding_synchronization.measurement.Cli import (
    add_extraction_args,
    add_modulation_args,
    add_verbose_arg,
    add_waveform_args,
    extraction_params,
    log_level,
    output_dir,
    slot_time_s,
    split_threshold,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import (
    auto_threshold,
    differential,
    extract_offsets,
)
from coding_synchronization.measurement.Plotting import plot_gap_histogram, plot_trace
from coding_synchronization.measurement.WaveformLoader import load_waveform

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")

_MAX_ZOOM_SAMPLES = 200_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_waveform_args(parser)
    add_extraction_args(parser)
    add_modulation_args(parser)
    add_verbose_arg(parser)
    parser.add_argument("--zoom-start", type=float, default=0.0, help="zoom window start, seconds")
    parser.add_argument(
        "--zoom-span", type=float, default=None,
        help="zoom window width in seconds; default: 20 word periods",
    )
    parser.add_argument("--no-show", action="store_true", help="save the figure without plt.show()")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    wf = load_waveform(waveform_params(args))
    ex = extraction_params(args)
    diff = differential(wf, ex)
    thr = ex.threshold if ex.threshold is not None else auto_threshold(diff.values)
    offsets = extract_offsets(diff, wf.dt_s, ex, threshold=thr)
    slot_s = slot_time_s(args, wf.dt_s)
    slots = offsets.to_slots(slot_s)

    word_period = (1 << args.ppm_rank) + args.dead_slots
    col_a = args.pos_col if args.pos_col is not None else wf.fmt.value_cols[0]
    col_b = args.neg_col if args.neg_col is not None else wf.fmt.value_cols[1]
    op = "+" if ex.combine == "add" else "−"
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(12, 17))

    plot_trace(ax0, wf.ch_a, wf.dt_s, label=f"col {col_a} (A)")
    plot_trace(ax0, wf.ch_b, wf.dt_s, label=f"col {col_b} (B)")
    ax0.set_title(f"Raw channels — {wf.source.name} ({wf.n} samples, {wf.duration_s:.6g} s)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.legend(loc="upper right")
    ax0.grid(True)

    plot_trace(
        ax1, wf.ch_a - np.float32(diff.baseline_a), wf.dt_s, label=f"A − {diff.baseline_a:.6g}"
    )
    plot_trace(
        ax1, wf.ch_b - np.float32(diff.baseline_b), wf.dt_s, label=f"B − {diff.baseline_b:.6g}"
    )
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_title("Zero-centred pair (per-channel DC pedestal removed)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    plot_trace(ax2, diff.values, wf.dt_s, label=f"A {op} B", color="tab:green")
    ax2.axhline(thr, color="red", linestyle="--", label=f"threshold={thr:.6g}")
    ax2.axhline(
        thr * ex.hysteresis, color="orange", linestyle=":",
        label=f"re-arm={thr * ex.hysteresis:.6g}",
    )
    ax2.set_title(
        f"Combined A {op} B (polarity={diff.polarity}) — {len(offsets)} pulses detected"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    span_s = args.zoom_span if args.zoom_span else 20 * word_period * slot_s
    start = int(np.clip(args.zoom_start / wf.dt_s, 0, max(0, wf.n - 2)))
    stop = int(min(wf.n, start + min(_MAX_ZOOM_SAMPLES, max(2, round(span_s / wf.dt_s)))))
    t_zoom = wf.time_s(start, stop)
    ax3.plot(t_zoom, diff.values[start:stop], linewidth=0.8, color="tab:green", label=f"A {op} B")
    ax3.axhline(thr, color="red", linestyle="--", label=f"threshold={thr:.6g}")
    in_zoom = offsets.samples[(offsets.samples >= start) & (offsets.samples < stop)]
    if len(in_zoom):
        ax3.vlines(
            in_zoom * wf.dt_s, 0, thr, color="black", linewidth=1.0, alpha=0.7,
            label=f"{len(in_zoom)} centroids",
        )
    ax3.set_title(f"Zoom: samples {start}–{stop} ({(stop - start) * wf.dt_s:.6g} s)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    plot_gap_histogram(ax4, slots, split_threshold=split_threshold(args))
    ax4.axvline(word_period, color="green", linestyle="-.", label=f"word_period={word_period}")
    ax4.legend()

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "measurement_overview.png", dpi=150)
    logger.info("Saved %s", out / "measurement_overview.png")

    if len(slots) > 1:
        gaps = np.diff(slots)
        logger.info(
            "Gaps (slots): median=%.2f, mean=%.2f, max=%.2f, word_period=%d, "
            "gaps>split_threshold=%d",
            float(np.median(gaps)), float(np.mean(gaps)), float(np.max(gaps)), word_period,
            int(np.sum(gaps > split_threshold(args))),
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
