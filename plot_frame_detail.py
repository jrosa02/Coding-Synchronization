"""Diagnose one whole frame's pulse detection against the raw waveform.

    python plot_frame_detail.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...
    python plot_frame_detail.py ... --sections sync metadata     # just the frame head

Clips the raw combined-differential trace around one frame — by default the whole frame,
sync+metadata+data+ecc — and overlays (a) the pulses OffsetExtractor actually detected and (b) a
synthetic "ideal PPM" pulse train at every word slot's expected position (ground truth for sync
words, the value Syncer already decoded for every other word) — a real bump with no ideal marker,
or vice versa, means detection missed something.

--sections narrows the window to a span of sections (they are contiguous, so a set is shown as
the range from the earliest to the latest), which is what makes individual pulses resolvable
again: a full frame is millions of samples wide and a pulse is 1-3 samples.
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
    add_title_arg,
    add_verbose_arg,
    add_waveform_args,
    extraction_params,
    figure_title,
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
from coding_synchronization.measurement.Plotting import frame_sections, plot_frame_detail
from coding_synchronization.measurement.WaveformLoader import load_waveform
from coding_synchronization.Model import Model2

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")

_SECTIONS = ("sync", "metadata", "data", "ecc")
# Above this many slots the per-slot ruler is a solid grey wall that costs seconds to draw and
# tells you nothing — a whole 240-word frame is ~300k slots.
_MAX_SLOT_RULER = 4000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_waveform_args(parser)
    add_extraction_args(parser)
    add_modulation_args(parser)
    add_title_arg(parser)
    add_verbose_arg(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help="which synced frame to inspect (0-based); default: first with the exact expected "
             "pulse count, falling back to frame 0 if none match",
    )
    parser.add_argument(
        "--sections", nargs="+", choices=_SECTIONS, default=None,
        help="which frame sections to show; default: the whole frame. Sections are contiguous, "
             "so several are shown as one window spanning the earliest to the latest",
    )
    parser.add_argument(
        "--margin-words", type=float, default=1.0,
        help="extra word_periods of context shown before/after the plotted window",
    )
    parser.add_argument(
        "--trace-only", action="store_true",
        help="plot only the measured trace — no section bands, detected-pulse ticks, ideal-PPM "
             "overlay or slot ruler",
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
    # Sync words carry a known value; every other word is whatever Syncer decoded for it
    # (decoded_frames_with_metadata is metadata + data + ecc, in transmission order).
    decoded = np.asarray(model.decoded_frames_with_metadata[idx], dtype=np.float64)
    n_nonsync = frame_params.metadata_num + frame_params.data_num + frame_params.ecc_num
    if len(decoded) < n_nonsync:
        logger.warning(
            "Frame %d decoded %d of the expected %d non-sync words — plotting the ideal train "
            "only as far as the decoded words reach", idx, len(decoded), n_nonsync,
        )
    word_values = np.concatenate(
        [np.full(frame_params.sync_num, float(sync_value)), decoded[:n_nonsync]]
    )
    metadata_values = decoded[: frame_params.metadata_num]

    # Word index range of every section except eof (the guard interval carries no words).
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, n in frame_sections(frame_params):
        if name in _SECTIONS:
            spans[name] = (cursor, cursor + n)
            cursor += n
    selected = [name for name in _SECTIONS if args.sections is None or name in args.sections]
    selected = [name for name in selected if spans[name][1] > spans[name][0]]
    if not selected:
        raise SystemExit("No non-empty sections selected — nothing to plot")

    # Sections are contiguous, so a selection is just the range from the earliest to the latest.
    first_word = min(spans[name][0] for name in selected)
    last_word = min(max(spans[name][1] for name in selected), len(word_values))
    if last_word <= first_word:
        raise SystemExit(
            f"Frame {idx} has no decoded words in {'+'.join(selected)} — nothing to plot"
        )

    word_idx = np.arange(first_word, last_word)
    ideal_samples = frame_start_abs_sample + (
        word_idx * word_period_samples + word_values[first_word:last_word] * slot_dur_samples
    )

    margin_samples = args.margin_words * word_period_samples
    window_start = frame_start_abs_sample + first_word * word_period_samples - margin_samples
    window_end = frame_start_abs_sample + last_word * word_period_samples + margin_samples
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
    section_bounds = [
        (
            name,
            frame_start_abs_s + spans[name][0] * word_period_s,
            frame_start_abs_s + min(spans[name][1], last_word) * word_period_s,
        )
        for name in selected
    ]

    # Very faint per-slot ruler across the whole visible window, so you can count slots against
    # a detected/ideal pulse once zoomed in — skipped over a wide window, where it would be a
    # solid grey wall rather than a ruler.
    slot_idx_start = int(np.floor((t[0] - frame_start_abs_s) / slot_dur_s))
    slot_idx_end = int(np.ceil((t[-1] - frame_start_abs_s) / slot_dur_s))
    n_slots = slot_idx_end - slot_idx_start + 1
    if args.trace_only:
        slot_positions = None
    elif n_slots <= _MAX_SLOT_RULER:
        slot_positions = (
            frame_start_abs_s + np.arange(slot_idx_start, slot_idx_end + 1) * slot_dur_s
        )
    else:
        slot_positions = None
        logger.info(
            "Per-slot ruler skipped: the window spans %d slots (> %d) — narrow it with "
            "--sections to get it back", n_slots, _MAX_SLOT_RULER,
        )

    if args.trace_only:
        # Everything except the trace is something *we* inferred; --trace-only shows the
        # measurement alone, so you can judge the raw signal without our overlays on top of it.
        detected_t = detected_t[:0]
        detected_amp = detected_amp[:0]
        ideal_samples = ideal_samples[:0]
        section_bounds = []

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_frame_detail(
        ax, t, trace, detected_t, detected_amp, ideal_samples * wf.dt_s,
        ideal_width_s, ideal_amp, section_bounds, slot_positions=slot_positions,
        title=figure_title(
            args,
            f"Frame {idx} {'+'.join(selected)} — {wf.source.name} "
            f"(sync_found={model.sync_found[idx]}/{frame_params.sync_num}, "
            f"metadata={metadata_values.astype(int).tolist()})",
        ),
    )

    out = model.output_dir or Path("output")
    fig.tight_layout()
    fig.savefig(out / "frame_detail.png", dpi=150)
    logger.info("Saved %s", out / "frame_detail.png")
    logger.info(
        "Frame %d %s: words %d-%d, sync_found=%d/%d, metadata=%s, window=[%d, %d) samples, "
        "%d pulses detected in window. Zoom/pan the interactive window into individual words to "
        "see the ideal-PPM shape at native (1-3 sample) resolution — it's invisible at "
        "whole-frame zoom.",
        idx, "+".join(selected), first_word, last_word - 1, model.sync_found[idx],
        frame_params.sync_num, metadata_values.astype(int).tolist(), start_sample, end_sample,
        int(np.sum(in_window)),
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
