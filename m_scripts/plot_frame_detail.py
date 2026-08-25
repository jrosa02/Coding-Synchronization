"""Compare the detected pulses of one frame against the raw waveform.

    python m_scripts/plot_frame_detail.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...
    python m_scripts/plot_frame_detail.py ... --sections sync metadata     # the head of the frame only

The script cuts the raw signal around one frame and draws two sets of marks on it. The first set
holds the pulses that OffsetExtractor detected. The second set holds an ideal PPM pulse at the
expected position of every word. A real pulse without an ideal mark means the detection missed
something. An ideal mark without a real pulse means the decode placed a word wrongly.

The ideal value of a sync word is known. For every other word the script uses the value that the
Syncer decoded.

--sections narrows the window to a range of sections. Use it to see individual pulses, because a
whole frame is millions of samples wide and one pulse is a few samples wide.
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
    add_slot_calibration_arg,
    add_title_arg,
    add_verbose_arg,
    add_waveform_args,
    extract_calibrated,
    extraction_params,
    figure_title,
    log_level,
    parse_args_with_sidecar,
    split_replica,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import differential
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
    add_slot_calibration_arg(parser)
    add_title_arg(parser)
    add_verbose_arg(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help="Which frame to inspect, counted from 0. The default is the first frame with the "
             "expected pulse count. If no frame matches, the script uses frame 0.",
    )
    parser.add_argument(
        "--sections", nargs="+", choices=_SECTIONS, default=None,
        help="Which frame sections to show. The default is the whole frame. The sections "
             "follow each other, so the window reaches from the earliest section to the "
             "latest.",
    )
    parser.add_argument(
        "--margin-words", type=float, default=1.0,
        help="How many extra word periods to show on each side of the window."
    )
    parser.add_argument(
        "--trace-only", action="store_true",
        help="Plot the measured trace alone. The figure then holds no section bands, no pulse "
             "marks, no ideal PPM overlay and no slot ruler.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    return parse_args_with_sidecar(parser)


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
    offsets, slot_s, _thr = extract_calibrated(diff, wf, ex, args)
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
