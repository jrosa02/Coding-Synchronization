"""Find the pulses in a capture at a given threshold, and save their positions.

    python m_scripts/extract_offsets.py measurments/RefCurve_2026-08-18_0_124555.Wfm.csv --threshold 0.05

The script removes the DC level of each channel, merges the two channels, and applies the
threshold. It computes one position for every run above the threshold. It writes the positions in
slots to offsets.npz, offsets.json and offsets.csv, which decode_measurement.py reads.

The measuring device inverts one leg of the pair, so the default --combine add restores the pulse.
Use --sweep LOW HIGH N first when you do not know a good threshold.
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
    add_slot_calibration_arg,
    add_title_arg,
    add_verbose_arg,
    add_waveform_args,
    apply_suptitle,
    extract_calibrated,
    extraction_params,
    log_level,
    min_separation_samples,
    output_dir,
    parse_args_with_sidecar,
    slot_time_s,
    split_threshold,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import (
    differential,
    extract_offsets,
    params_to_dict,
    save_offsets,
)
from coding_synchronization.measurement.Plotting import plot_gap_histogram
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
    parser.add_argument("--out", type=str, default=None, help="The path of the output .npz file.")
    parser.add_argument(
        "--sweep", nargs=3, type=float, metavar=("LOW", "HIGH", "N"), default=None,
        help="Try N thresholds between LOW and HIGH. The script prints a table and saves "
             "nothing.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not draw the diagnostic figure.")
    return parse_args_with_sidecar(parser)


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    wp = waveform_params(args)
    wf = load_waveform(wp)
    ex = extraction_params(args)
    diff = differential(wf, ex)
    split_thr = split_threshold(args)
    slot_s = slot_time_s(args, wf.dt_s)
    ex.min_separation_samples = min_separation_samples(args, wf.dt_s, slot_s)

    if args.sweep is not None:
        low, high, n = args.sweep
        logger.info("Sweeping %d thresholds in [%.6g, %.6g]", int(n), low, high)
        print(f"{'threshold':>14} {'pulses':>10} {'frames':>8} {'median gap':>12}")
        for thr in np.linspace(low, high, int(n)):
            off = extract_offsets(diff, wf.dt_s, ex, threshold=float(thr))
            slots = off.to_slots(slot_s)
            gaps = np.diff(slots) if len(slots) > 1 else np.zeros(1)
            frames = int(np.sum(gaps > split_thr)) + 1 if len(slots) > 1 else 0
            print(f"{thr:>14.6g} {len(off):>10d} {frames:>8d} {float(np.median(gaps)):>12.2f}")
        return

    offsets, slot_s, thr = extract_calibrated(diff, wf, ex, args)
    slots = offsets.to_slots(slot_s)

    out_dir = output_dir()
    out_path = args.out if args.out else out_dir / "offsets.npz"
    save_offsets(
        out_path, offsets, slot_s,
        meta={
            "source": str(wf.source),
            "n_samples": wf.n,
            "csv_format": repr(wf.fmt),
            "extraction_params": params_to_dict(ex),
            "split_threshold_slots": split_thr,
        },
    )
    np.savetxt(out_dir / "offsets.csv", slots, fmt="%.6f", header="offset_slots", comments="")

    word_period = (1 << args.ppm_rank) + args.dead_slots
    if len(slots) > 1:
        gaps = np.diff(slots)
        n_frames = int(np.sum(gaps > split_thr)) + 1
        logger.info(
            "Summary: %d pulses | gaps(slots) median=%.2f mean=%.2f max=%.2f | "
            "word_period=%d | %d frames at split threshold %.0f",
            len(offsets), float(np.median(gaps)), float(np.mean(gaps)), float(np.max(gaps)),
            word_period, n_frames, split_thr,
        )
    else:
        logger.warning("Only %d pulse(s) extracted — threshold is probably wrong", len(offsets))

    if not args.no_plot:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))
        ax0.plot(offsets.amplitudes, ".", markersize=2)
        ax0.axhline(thr, color="red", linestyle="--", label=f"threshold={thr:.6g}")
        ax0.set_title(f"Pulse amplitude vs index — {len(offsets)} pulses")
        ax0.set_xlabel("Pulse index")
        ax0.set_ylabel("Peak amplitude")
        ax0.legend()
        ax0.grid(True)
        plot_gap_histogram(ax1, slots, split_threshold=split_thr)
        apply_suptitle(fig, args)
        fig.tight_layout()
        fig.savefig(out_dir / "offsets.png", dpi=150)
        logger.info("Saved %s", out_dir / "offsets.png")


if __name__ == "__main__":
    main()
