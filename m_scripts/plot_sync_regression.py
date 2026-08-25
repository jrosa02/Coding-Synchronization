"""Fit a line through the sync residuals, and show how far the residuals drift.

    python m_scripts/plot_sync_regression.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...
    python m_scripts/plot_sync_regression.py ... --all-frames

The script plots the raw sync pulse offsets in slots against their word index. It calibrates them
the way pass 1 of Syncer._sync_frame does, so the residual stays near zero instead of showing the
raw scale mismatch as a slope. The fit marks its own intercept at x=0 with an uncertainty, because
the intercept is expected to lie near zero and is not forced to zero.

By default the script uses the sync section of one frame. --all-frames fits one line through the
sync sections of every frame, which reveals a bias that repeats across frames.

The script does not run Model2 or the Syncer. It repeats the same chunking that the Splitter and
the FrameFilter perform, so it needs the extracted offsets only.

A tilted fit means the slot time scale is wrong. It does not mean the sync words decode wrongly.
sync_value cancels in this calculation, because frame_start subtracts it and the decode adds it
back, so the residual is the same for every --sync-value. --calibration median reproduces the tilt
of the earlier median-only scale. The default value ls matches the Syncer.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization._logging import setup_logging
from coding_synchronization.measurement.Cli import (
    add_extraction_args,
    add_modulation_args,
    add_slot_calibration_arg,
    add_title_arg,
    add_verbose_arg,
    add_waveform_args,
    extract_calibrated,
    extraction_params,
    figure_title,
    log_level,
    output_dir,
    parse_args_with_sidecar,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import differential
from coding_synchronization.measurement.Plotting import plot_offset_regression
from coding_synchronization.measurement.WaveformLoader import load_waveform

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_waveform_args(parser)
    add_extraction_args(parser)
    add_modulation_args(parser)
    add_slot_calibration_arg(parser)
    add_title_arg(parser)
    add_verbose_arg(parser)
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help="Which chunk to inspect, counted from 0 after --keep-first-frame. The default is "
             "the first chunk with the expected pulse count. If no chunk matches, the script "
             "uses chunk 0. --all-frames ignores this option.",
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help="Fit one line through the sync sections of every frame, and not through one frame."
    )
    parser.add_argument(
        "--calibration", choices=("ls", "median"), default="ls",
        help="How to fit the slot time scale of the sync section. This matches pass 1 of the "
             "Syncer. ls is the default, and it refines the median gap with a least-squares fit "
             "over the sync pulses. median uses the median gap alone, which tilts the "
             "residual.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    return parse_args_with_sidecar(parser)


def _frame_residual(
    chunk: np.ndarray, sync_num: int, word_period: int, sync_value: int,
    calibration: str = "ls",
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Calibrate one chunk's sync section the same way Syncer's Pass 1 does.

    Returns (x, y_raw, scale, frame_start, decoded, residual). `residual` reproduces
    `Syncer.sync_residuals` for this chunk, computed independently of Model2/Syncer.

    `calibration="median"` is the old median-gap-only scale, kept so the bias it introduces can be
    rendered side by side with the fixed one: the median of a handful of noisy gaps is not the
    least-squares slope, and the difference tilts the whole residual. Every pulse in the sync
    section here is a real detection (unlike Syncer, which may have to assume a missing one), so
    the refinement is a plain fit over all `sync_num` points.
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


def _gap_jitter_metric(y_raw: np.ndarray, word_period: int, ppm_rank: int) -> float | None:
    if len(y_raw) < 2:
        return None
    gap_ratio = np.diff(y_raw) / word_period
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.log(gap_ratio.std() * 3 * (1 << ppm_rank) / gap_ratio.mean()))


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    wp = waveform_params(args)
    wf = load_waveform(wp)
    ex = extraction_params(args)
    diff = differential(wf, ex)
    offsets, slot_s, _thr = extract_calibrated(diff, wf, ex, args)
    if len(offsets) == 0:
        raise SystemExit("No pulses extracted — nothing to plot")
    slots = offsets.to_slots(slot_s)

    word_period = (1 << args.ppm_rank) + args.dead_slots
    expected_total = args.sync_num + args.metadata_num + args.data_num + args.ecc_num
    split_threshold = args.eof_num * word_period
    sync_value = 0 if args.sync_value is None else int(args.sync_value)

    # Replicate Splitter + FrameFilter directly on `slots`, keeping the native chunk-zeroed
    # slot-unit view Syncer itself receives (unlike plot_frame_detail.py, which needs to undo
    # this zeroing to recover absolute time).
    boundaries = np.where(np.diff(slots) > split_threshold)[0] + 1
    raw_chunks = [c - c[0] for c in np.split(slots, boundaries) if len(c) > 0]
    chunks = raw_chunks if args.keep_first_frame else raw_chunks[1:]
    if not chunks:
        raise SystemExit("No frame chunks found — nothing to plot")

    out = output_dir()
    root = ET.Element(
        "sync_regression", source=wf.source.name, word_period=f"{word_period:.6f}",
        sync_value=str(sync_value), calibration=args.calibration,
    )

    if args.all_frames:
        candidates = [(i, c) for i, c in enumerate(chunks) if len(c) >= args.sync_num]
        if not candidates:
            raise SystemExit(f"No chunk has at least sync_num={args.sync_num} pulses")
        xs, ys, metrics = [], [], []
        for chunk_idx, chunk in candidates:
            x, y_raw, scale, frame_start, decoded, residual = _frame_residual(
                chunk, args.sync_num, word_period, sync_value, args.calibration
            )
            xs.append(x)
            ys.append(residual)
            metric = _gap_jitter_metric(y_raw, word_period, args.ppm_rank)
            if metric is not None:
                metrics.append(metric)
            frame_el = ET.SubElement(
                root, "frame", chunk=str(chunk_idx), scale=f"{scale:.10f}",
                frame_start=f"{frame_start:.6f}",
                gap_jitter_error_metric="" if metric is None else f"{metric:.6f}",
            )
            for i in range(args.sync_num):
                ET.SubElement(
                    frame_el, "word", index=str(i), raw_offset=f"{y_raw[i]:.6f}",
                    decoded=f"{decoded[i]:.6f}", residual=f"{residual[i]:.6f}",
                )
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        if metrics:
            logger.info(
                "gap-jitter error metric across %d frames: mean=%.6g, std=%.6g",
                len(metrics), float(np.mean(metrics)), float(np.std(metrics)),
            )
        logger.info("Pooled %d sync pulses from %d frames", len(y_all), len(candidates))
        title = f"Sync residual (calibrated actual − decoded) — {wf.source.name} (all {len(candidates)} frames)"
        x, y = x_all, y_all
    else:
        if args.frame_index is not None:
            idx = args.frame_index
            if not (0 <= idx < len(chunks)):
                raise SystemExit(f"--frame-index {idx} out of range (0..{len(chunks) - 1})")
        else:
            idx = next((i for i, c in enumerate(chunks) if len(c) == expected_total), None)
            if idx is None:
                logger.warning(
                    "No chunk has the exact expected pulse count (%d) — falling back to chunk 0",
                    expected_total,
                )
                idx = 0

        chunk = chunks[idx]
        if len(chunk) < args.sync_num:
            raise SystemExit(
                f"Chunk {idx} has only {len(chunk)} pulses, fewer than sync_num={args.sync_num}"
            )
        x, y_raw, scale, frame_start, decoded, y = _frame_residual(
            chunk, args.sync_num, word_period, sync_value, args.calibration
        )
        metric = _gap_jitter_metric(y_raw, word_period, args.ppm_rank)
        if metric is not None:
            logger.info("gap-jitter error metric (same formula as Syncer's debug log) = %.6g", metric)

        root.set("chunk", str(idx))
        root.set("scale", f"{scale:.10f}")
        root.set("frame_start", f"{frame_start:.6f}")
        root.set("gap_jitter_error_metric", "" if metric is None else f"{metric:.6f}")
        for i in range(args.sync_num):
            ET.SubElement(
                root, "word", index=str(i), raw_offset=f"{y_raw[i]:.6f}",
                decoded=f"{decoded[i]:.6f}", residual=f"{y[i]:.6f}",
            )
        title = f"Sync residual (calibrated actual − decoded) — {wf.source.name} (chunk {idx})"

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plot_offset_regression(
        ax, x, y, title=figure_title(args, title), ylabel="Actual − decoded (slots)",
        frame_words=expected_total,
    )

    fig.tight_layout()
    fig.savefig(out / "sync_regression.png", dpi=150)
    logger.info("Saved %s", out / "sync_regression.png")

    xml_path = out / "sync_regression.xml"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    logger.info("Saved %s", xml_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
