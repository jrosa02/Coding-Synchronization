"""Convert the sync-fit's extrapolated uncertainty into a per-word decode-error risk curve.

    python m_scripts/plot_decode_risk.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...

Same model as plot_sync_margin.py: each frame fits its own slope/intercept/σ from only its
own sync words — no frame's numbers are combined, averaged, or influence any other frame's
fit — extrapolated with the OLS prediction interval to word indices the fit never saw.
Instead of plotting the ±σ band directly, this converts each frame's own band into
P(|timing error| > boundary), assuming Gaussian jitter — a risk-vs-word-index curve that
answers "how likely is THIS word to decode wrong" rather than "how wide is the uncertainty
band here". Every frame's own risk curve is drawn on the same axes at low alpha. The risk
spans many orders of magnitude across a frame, so the y-axis is log-scaled.

Like plot_sync_regression.py and plot_sync_margin.py, this does not run Model2 or the
Syncer — it repeats the same sync-section chunking and fit independently.
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
from coding_synchronization.measurement.SyncMargin import (
    exceed_probability,
    per_frame_fits,
    predicted_sigma,
)
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
    section_bounds: list[tuple[str, int, int]] = []
    for name, count in sections:
        width = float(count)
        if width > 0:
            ax.axvspan(cursor, cursor + width, color=_SECTION_COLORS[name], alpha=0.15, label=name)
            section_bounds.append((name, int(cursor), int(cursor + width)))
        cursor += width

    frame_rows: list[dict] = []
    for j, (idx, fit) in enumerate(fits):
        sigma_k = predicted_sigma(fit, k)
        risk_k = exceed_probability(sigma_k, boundary=args.boundary)

        ax.semilogy(
            k, np.clip(risk_k, 1e-300, 1.0), color="tab:red", linewidth=0.8, alpha=0.15,
            zorder=3, label=f"P(|error| > {args.boundary:g} slots)" if j == 0 else None,
        )

        section_risk = {
            name: float(np.mean(risk_k[lo:hi])) if hi > lo else 0.0
            for name, lo, hi in section_bounds
        }
        frame_rows.append({
            "frame": idx, "slope": fit.slope, "s2": fit.s2,
            "section_mean_risk": section_risk,
            "max_risk": float(np.max(risk_k)),
            "max_risk_word_index": int(np.argmax(risk_k)),
        })

    title = f"Decode-error risk across the frame — {wf.source.name} ({len(fits)} frames, independent)"
    ax.set_xlabel("Word index")
    ax.set_ylabel("P(decode error)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(figure_title(args, title))
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    # Display only: the median across frames' own independent section-risk means, purely to
    # make the plot readable at a glance — every frame's own numbers stay in decode_risk.json.
    lines = []
    for name, _, _ in section_bounds:
        vals = [row["section_mean_risk"][name] for row in frame_rows]
        lines.append(f"{name}: median across frames = {float(np.median(vals)):.3e}")
    ax.text(
        0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=8, va="bottom",
        ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "decode_risk.png", dpi=150)
    logger.info("Saved %s", out / "decode_risk.png")

    summary = {
        "source": wf.source.name,
        "boundary_slots": args.boundary,
        "total_words": total_words,
        "frames": frame_rows,
    }
    json_path = out / "decode_risk.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", json_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
