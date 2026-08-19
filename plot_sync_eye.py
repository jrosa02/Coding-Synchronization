"""Eye-diagram overlay of every sync pulse in a capture, anchored on the rising edge and on the decoded PPM slot.

    python plot_sync_eye.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...

Every sync-word pulse across every frame is overlaid on the same axes, in one style — pulse
shape, amplitude, and jitter consistency across the whole capture are all directly visible at
once. How many sync words decoded to something other than `sync_value` is logged, not drawn.

Two panels, same pulses, different t=0:

* anchored on each pulse's own rising-edge crossing — pulse shape and width consistency, with
  every pulse's own timing error divided out;
* anchored on the decoded PPM slot position (fitted frame start + word index * word_period +
  decoded value) — the same traces against the grid the decoder actually placed them on, so the
  horizontal spread *is* the timing error budget: sub-slot residual plus frame-start fit error.
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
    rising_edge_crossing,
)
from coding_synchronization.measurement.Plotting import plot_sync_eye
from coding_synchronization.measurement.WaveformLoader import load_waveform
from coding_synchronization.Model import Model2

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_waveform_args(parser)
    add_extraction_args(parser)
    add_modulation_args(parser)
    add_title_arg(parser)
    add_verbose_arg(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window-factor", type=float, default=6.0,
        help="half-window shown around each pulse's rising edge, as a multiple of the median "
             "detected pulse width",
    )
    parser.add_argument("--no-show", action="store_true", help="save the figure without plt.show()")
    return parser.parse_args()


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

    if not model.raw_synced_frames:
        raise SystemExit("No synced frames — nothing to plot")

    kept_for_sync = split_replica(
        offsets.samples, slots, expected_total, frame_params.sync_num, frame_params.eof_num,
        model.word_period, args.keep_first_frame, args.drop_partial_frames,
    )
    n = min(len(kept_for_sync), len(model.raw_synced_frames), len(model.sync_frames))
    if len(kept_for_sync) != len(model.raw_synced_frames):
        logger.warning(
            "Split replica disagrees with Model2 (%d chunks vs. %d synced frames) — using the "
            "first %d frames only", len(kept_for_sync), len(model.raw_synced_frames), n,
        )

    half_window = max(3, int(round(args.window_factor * float(np.median(offsets.widths)))))
    sync_value = 0 if args.sync_value is None else int(args.sync_value)

    t_rel_list: list[np.ndarray] = []
    t_ppm_list: list[np.ndarray] = []
    trace_list: list[np.ndarray] = []
    decoded_values: list[int] = []
    for idx in range(n):
        abs_chunk = kept_for_sync[idx]
        decoded = model.sync_frames[idx]
        n_sync = min(frame_params.sync_num, len(abs_chunk), len(decoded))
        # Splitter zeroed this chunk before Syncer saw it, so frame_starts[idx] is frame-local:
        # re-anchor it on the chunk's first pulse to get the decode grid in absolute time.
        slot_dur_s = model.inferred_slot_time_s[idx]
        frame_start_abs_s = float(abs_chunk[0]) * wf.dt_s + model.frame_starts[idx] * slot_dur_s
        for k in range(n_sync):
            approx = float(abs_chunk[k])
            rise = rising_edge_crossing(diff.values, approx, thr, search_radius=2 * half_window)
            lo = max(0, int(np.floor(rise)) - half_window)
            hi = min(wf.n, int(np.ceil(rise)) + half_window)
            if hi <= lo:
                continue
            samples = np.arange(lo, hi, dtype=np.float64)
            ideal_s = frame_start_abs_s + (
                k * model.word_period + float(decoded[k])
            ) * slot_dur_s
            t_rel_list.append((samples - rise) * wf.dt_s)
            t_ppm_list.append(samples * wf.dt_s - ideal_s)
            trace_list.append(diff.values[lo:hi])
            decoded_values.append(int(decoded[k]))

    if not t_rel_list:
        raise SystemExit("No sync pulses collected — nothing to plot")

    fig, (ax_rise, ax_ppm) = plt.subplots(2, 1, figsize=(11, 12))
    plot_sync_eye(
        ax_rise, t_rel_list, trace_list,
        title="Anchored on each pulse's own rising edge (shape and width consistency)",
    )
    plot_sync_eye(
        ax_ppm, t_ppm_list, trace_list,
        title="Anchored on the decoded PPM slot position (spread = timing error)",
        xlabel="Time relative to decoded PPM slot position (s)",
        anchor_label="decoded slot position",
    )
    fig.suptitle(
        figure_title(
            args,
            f"Sync pulse eye diagram — {wf.source.name} "
            f"({len(t_rel_list)} pulses across {n} frames)",
        )
    )

    out = model.output_dir or Path("output")
    fig.tight_layout()
    fig.savefig(out / "sync_eye.png", dpi=150)
    logger.info("Saved %s", out / "sync_eye.png")

    wrong = int(np.sum(np.array(decoded_values) != sync_value))
    logger.info(
        "%d/%d sync pulses decoded to a value other than sync_value=%d",
        wrong, len(decoded_values), sync_value,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
