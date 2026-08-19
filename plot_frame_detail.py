"""Diagnose one frame's sync+metadata detection against the raw waveform.

    python plot_frame_detail.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...

Clips the raw combined-differential trace around one frame's sync+metadata section and overlays
(a) the pulses OffsetExtractor actually detected and (b) a synthetic "ideal PPM" pulse train at
every word slot's expected position (ground truth for sync words, the value Syncer already
decoded for metadata words) — a real bump with no ideal marker, or vice versa, means detection
missed something.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization._logging import setup_logging
from coding_synchronization.encoder import FrameParams, ModulationParams
from coding_synchronization.measurement.Cli import (
    add_extraction_args,
    add_modulation_args,
    add_verbose_arg,
    add_waveform_args,
    extraction_params,
    log_level,
    min_separation_samples,
    slot_time_s,
    split_replica,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import (
    auto_threshold,
    differential,
    extract_offsets,
)
from coding_synchronization.measurement.Plotting import plot_frame_detail
from coding_synchronization.measurement.WaveformLoader import load_waveform
from coding_synchronization.Model import Model2

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_waveform_args(parser)
    add_extraction_args(parser)
    add_modulation_args(parser)
    add_verbose_arg(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help="which synced frame to inspect (0-based); default: first with the exact expected "
             "pulse count, falling back to frame 0 if none match",
    )
    parser.add_argument(
        "--margin-words", type=float, default=1.0,
        help="extra word_periods of context shown before/after the sync+metadata window",
    )
    parser.add_argument("--no-show", action="store_true", help="save the figure without plt.show()")
    return parser.parse_args()


def _locate_frame_start(
    args: argparse.Namespace,
    model: Model2,
    frame_params: FrameParams,
    offsets,
    slots: np.ndarray,
    idx: int,
) -> tuple[float, float]:
    """Absolute (frame_start_abs_seconds, calibrated_seconds_per_slot) for synced frame `idx`.

    Splitter zeroes each frame chunk (`chunk - chunk[0]`) before Syncer ever sees it, so
    `model.frame_starts`/`raw_synced_frames` are frame-local, not absolute against the file. This
    replicates the exact same split (same threshold, same input order) directly on the absolute
    sample-unit offsets, then re-derives FrameFilter's exact keep/drop decisions, to recover which
    real sample the chosen frame actually starts at.
    """
    expected_total = (
        frame_params.sync_num + frame_params.metadata_num
        + frame_params.data_num + frame_params.ecc_num
    )
    kept_for_sync = split_replica(
        offsets.samples, slots, expected_total, frame_params.sync_num, frame_params.eof_num,
        model.word_period, args.keep_first_frame, args.drop_partial_frames,
    )

    raw_frames = model.raw_synced_frames
    if idx >= len(kept_for_sync):
        raise SystemExit(
            f"Internal split replica disagrees with Model2 ({len(kept_for_sync)} candidate "
            f"chunks vs. {len(raw_frames)} synced frames) — cannot locate frame {idx}"
        )
    abs_chunk = kept_for_sync[idx]
    if len(abs_chunk) != len(raw_frames[idx]):
        logger.warning(
            "Split replica mismatch for frame %d: %d pulses locally vs. %d in Model2 — absolute "
            "positions below may be off", idx, len(abs_chunk), len(raw_frames[idx]),
        )

    frame_start_rel = model.frame_starts[idx]  # calibrated, chunk-relative (slot units)
    slot_dur_s = model.inferred_slot_time_s[idx]  # calibrated seconds-per-slot for this frame
    first_pulse_abs_s = float(abs_chunk[0]) * offsets.dt_s
    frame_start_abs_s = first_pulse_abs_s + frame_start_rel * slot_dur_s
    return frame_start_abs_s, slot_dur_s


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    wp = waveform_params(args)
    wf = load_waveform(wp)
    ex = extraction_params(args)
    diff = differential(wf, ex)
    slot_s = slot_time_s(args, wf.dt_s)
    ex.min_separation_samples = min_separation_samples(args, wf.dt_s, slot_s)
    thr = ex.threshold if ex.threshold is not None else auto_threshold(diff.values)
    offsets = extract_offsets(diff, wf.dt_s, ex, threshold=thr)
    if len(offsets) == 0:
        raise SystemExit("No pulses extracted — nothing to plot")
    slots = offsets.to_slots(slot_s)

    mod_params = ModulationParams(
        ppm_rank=args.ppm_rank, slot_time=np.float64(slot_s), dead_slots=args.dead_slots
    )
    frame_params = FrameParams(
        sync_num=args.sync_num, metadata_num=args.metadata_num, data_num=args.data_num,
        ecc_num=args.ecc_num, eof_num=args.eof_num,
    )
    expected_total = (
        frame_params.sync_num + frame_params.metadata_num
        + frame_params.data_num + frame_params.ecc_num
    )

    model = Model2(
        offsets=slots,
        frame_params=frame_params,
        mod_params=mod_params,
        extraction_params=ex,
        waveform_params=wp,
        sync_value=args.sync_value,
        drop_first_frame=not args.keep_first_frame,
        drop_wrong_length=args.drop_partial_frames,
        plot=False,
        seed=args.seed,
    )
    model.construct_pipeline()
    model.run()

    raw_frames = model.raw_synced_frames
    if not raw_frames:
        raise SystemExit("No synced frames — nothing to plot")

    if args.frame_index is not None:
        idx = args.frame_index
        if not (0 <= idx < len(raw_frames)):
            raise SystemExit(f"--frame-index {idx} out of range (0..{len(raw_frames) - 1})")
    else:
        idx = next((i for i, f in enumerate(raw_frames) if len(f) == expected_total), None)
        if idx is None:
            logger.warning(
                "No synced frame has the exact expected pulse count (%d) — falling back to "
                "frame 0", expected_total,
            )
            idx = 0

    frame_start_abs_s, slot_dur_s = _locate_frame_start(
        args, model, frame_params, offsets, slots, idx
    )
    slot_dur_samples = slot_dur_s / wf.dt_s
    word_period_samples = model.word_period * slot_dur_samples
    frame_start_abs_sample = frame_start_abs_s / wf.dt_s

    sync_value = 0 if args.sync_value is None else int(args.sync_value)
    metadata_values = np.asarray(
        model.decoded_frames_with_metadata[idx][: frame_params.metadata_num], dtype=np.float64
    )

    sync_ideal = frame_start_abs_sample + (
        np.arange(frame_params.sync_num) * word_period_samples + sync_value * slot_dur_samples
    )
    metadata_ideal = frame_start_abs_sample + (
        (frame_params.sync_num + np.arange(frame_params.metadata_num)) * word_period_samples
        + metadata_values * slot_dur_samples
    )
    ideal_samples = np.concatenate([sync_ideal, metadata_ideal])

    margin_samples = args.margin_words * word_period_samples
    window_start = frame_start_abs_sample - margin_samples
    window_end = (
        frame_start_abs_sample
        + (frame_params.sync_num + frame_params.metadata_num) * word_period_samples
        + margin_samples
    )
    start_sample = max(0, int(np.floor(window_start)))
    end_sample = min(wf.n, int(np.ceil(window_end)))
    if end_sample <= start_sample:
        raise SystemExit("Computed window is empty — check --margin-words / frame params")

    t = wf.time_s(start_sample, end_sample)
    trace = diff.values[start_sample:end_sample]

    in_window = (offsets.samples >= start_sample) & (offsets.samples < end_sample)
    detected_t = offsets.samples[in_window] * wf.dt_s
    detected_amp = offsets.amplitudes[in_window]

    if np.any(in_window):
        ideal_width_samples = float(np.median(offsets.widths[in_window]))
        ideal_amp = float(np.median(offsets.amplitudes[in_window]))
    else:
        ideal_width_samples = float(np.median(offsets.widths))
        ideal_amp = float(np.median(offsets.amplitudes))
    ideal_width_s = ideal_width_samples * wf.dt_s

    word_period_s = word_period_samples * wf.dt_s
    sync_end_s = frame_start_abs_s + frame_params.sync_num * word_period_s
    metadata_end_s = sync_end_s + frame_params.metadata_num * word_period_s
    section_bounds = [
        ("sync", frame_start_abs_s, sync_end_s),
        ("metadata", sync_end_s, metadata_end_s),
    ]

    # Very faint per-slot ruler across the whole visible window, so you can count slots against
    # a detected/ideal pulse once zoomed in.
    slot_idx_start = int(np.floor((t[0] - frame_start_abs_s) / slot_dur_s))
    slot_idx_end = int(np.ceil((t[-1] - frame_start_abs_s) / slot_dur_s))
    slot_positions = frame_start_abs_s + np.arange(slot_idx_start, slot_idx_end + 1) * slot_dur_s

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_frame_detail(
        ax, t, trace, detected_t, detected_amp, ideal_samples * wf.dt_s,
        ideal_width_s, ideal_amp, section_bounds, slot_positions=slot_positions,
        title=(
            f"Frame {idx} sync+metadata — {wf.source.name} "
            f"(sync_found={model.sync_found[idx]}/{frame_params.sync_num}, "
            f"metadata={metadata_values.astype(int).tolist()})"
        ),
    )

    out = model.output_dir or Path("output")
    fig.tight_layout()
    fig.savefig(out / "frame_detail.png", dpi=150)
    logger.info("Saved %s", out / "frame_detail.png")
    logger.info(
        "Frame %d: sync_found=%d/%d, metadata=%s, window=[%d, %d) samples, %d pulses detected "
        "in window. Zoom/pan the interactive window into individual words to see the ideal-PPM "
        "shape at native (1-3 sample) resolution — it's invisible at whole-frame zoom.",
        idx, model.sync_found[idx], frame_params.sync_num,
        metadata_values.astype(int).tolist(), start_sample, end_sample, int(np.sum(in_window)),
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
