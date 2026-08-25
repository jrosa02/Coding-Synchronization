"""Extrapolate the sync-section calibration fit across the whole frame.

    python m_scripts/plot_sync_margin.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...

`plot_sync_regression.py` fits a line through a frame's sync-word residuals to check the
Pass-1 calibration (`frame_start`/slot-time scale). But `Syncer._decode_positions` reuses
that exact same fit, unmodified, to place every other word in the frame — metadata, data
and ECC words get no further correction. So the fit's own uncertainty doesn't stay
contained in the sync section: it grows the further a word is from the sync words the fit
was built on, via the standard OLS prediction interval

    Var_pred(k) = s2 * (1 + 1/n + (k - xbar)^2 / sxx)

Each frame fits its own slope/intercept/σ from only its own sync words — no frame's numbers
are combined, averaged, or otherwise influence any other frame's fit (an earlier pooled
version of this script concatenated every frame's sync residuals into one fit, which made
the calibration-uncertainty term shrink as more frames were added — backwards, since real
frame-to-frame slot-time disagreement doesn't average away). Every frame's own extrapolated
±1σ/±3σ band is drawn on the same axes at low alpha, so the visual spread of independently
drawn bands — not any computed pooled quantity — is what shows inter-frame disagreement.

Like plot_sync_regression.py, this does not run Model2 or the Syncer — it repeats the same
sync-section chunking and fit independently. See plot_margin_validation.py to check this
prediction against real decoded pulses.
"""

import argparse
import json
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
    extract_calibrated,
    extraction_params,
    figure_title,
    log_level,
    output_dir,
    parse_args_with_sidecar,
    waveform_params,
)
from coding_synchronization.encoder import FrameParams
from coding_synchronization.measurement.OffsetExtractor import differential
from coding_synchronization.measurement.Plotting import _SECTION_COLORS, frame_sections
from coding_synchronization.measurement.SyncMargin import per_frame_fits, predicted_sigma
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
        "--calibration", choices=("ls", "median"), default="ls",
        help="How to fit the slot time scale of each sync section — same option as "
             "plot_sync_regression.py.",
    )
    parser.add_argument(
        "--boundary", type=float, default=0.5,
        help="The decode decision boundary, in slots. The default (0.5) is the actual "
             "Syncer rounding boundary.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    return parse_args_with_sidecar(parser)


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
    sync_value = 0 if args.sync_value is None else int(args.sync_value)
    frame_params = FrameParams(
        sync_num=args.sync_num, metadata_num=args.metadata_num, data_num=args.data_num,
        ecc_num=args.ecc_num, eof_num=args.eof_num,
    )
    sections = frame_sections(frame_params)
    total_words = sum(count for _, count in sections)

    split_threshold = args.eof_num * word_period
    boundaries = np.where(np.diff(slots) > split_threshold)[0] + 1
    raw_chunks = [c - c[0] for c in np.split(slots, boundaries) if len(c) > 0]
    chunks = raw_chunks if args.keep_first_frame else raw_chunks[1:]
    if not chunks:
        raise SystemExit("No frame chunks found — nothing to plot")

    fits = per_frame_fits(chunks, args.sync_num, word_period, sync_value, args.calibration)
    if not fits:
        raise SystemExit(f"No chunk has at least sync_num={args.sync_num} pulses")
    logger.info("Fitted %d/%d frames independently (no pooling)", len(fits), len(chunks))

    k = np.arange(total_words, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    cursor = 0.0
    for name, count in sections:
        width = float(count)
        if width > 0:
            ax.axvspan(cursor, cursor + width, color=_SECTION_COLORS[name], alpha=0.15, label=name)
        cursor += width

    frame_rows: list[dict] = []
    for j, (idx, fit) in enumerate(fits):
        y_line = fit.slope * k + fit.intercept
        sigma_1 = predicted_sigma(fit, k)
        sigma_3 = 3.0 * sigma_1

        ax.plot(k, y_line, color="tab:red", linewidth=0.8, alpha=0.25, zorder=3,
                label="frame fit" if j == 0 else None)
        ax.fill_between(
            k, y_line - sigma_1, y_line + sigma_1, color="tab:red", alpha=0.06, zorder=2,
            label="±1σ (per frame)" if j == 0 else None,
        )
        ax.fill_between(
            k, y_line - sigma_3, y_line + sigma_3, color="tab:red", alpha=0.03, zorder=1,
            label="±3σ (per frame)" if j == 0 else None,
        )

        crossing = np.where(
            (y_line + sigma_3 >= args.boundary) | (y_line - sigma_3 <= -args.boundary)
        )[0]
        margin_word: int | None = int(crossing[0]) if len(crossing) else None
        frame_rows.append({
            "frame": idx, "slope": fit.slope, "slope_se": fit.slope_se,
            "intercept": fit.intercept, "intercept_se": fit.intercept_se,
            "s2": fit.s2, "n": fit.n, "first_word_over_boundary": margin_word,
        })

    n_crossing = sum(1 for row in frame_rows if row["first_word_over_boundary"] is not None)
    logger.info(
        "%d/%d frames' own ±3σ band crosses the ±%.3g slot decode boundary within %d words",
        n_crossing, len(frame_rows), args.boundary, total_words,
    )

    ax.axhline(args.boundary, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"decode boundary ±{args.boundary:g}")
    ax.axhline(-args.boundary, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.6, alpha=0.5)

    title = f"Sync-fit margin across the frame — {wf.source.name} ({len(fits)} frames, independent)"
    ax.set_xlabel("Word index")
    ax.set_ylabel("Predicted residual (slots)")
    ax.grid(True, alpha=0.4)
    ax.set_title(figure_title(args, title))
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "sync_margin.png", dpi=150)
    logger.info("Saved %s", out / "sync_margin.png")

    summary = {
        "source": wf.source.name,
        "sync_value": sync_value,
        "calibration": args.calibration,
        "total_words": total_words,
        "boundary_slots": args.boundary,
        "frames": frame_rows,
    }
    json_path = out / "sync_margin.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", json_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
