"""Check the sync-fit uncertainty model against real decoded pulses.

    python m_scripts/plot_margin_validation.py measurments/RefCurve_....csv --eof-num 2 --threshold 0.3 ...

plot_sync_margin.py and plot_decode_risk.py extrapolate a sync-section fit's OLS
prediction interval across the whole frame, on the assumption that the same jitter/scale
error the sync words show also applies to metadata/data/ECC words. This script checks that
assumption against reality: it runs the actual Model2/Syncer pipeline, then — for every
real pulse in every synced frame, not just the sync ones — replicates
`Syncer._decode_positions`'s floor+round decode to get that pulse's own sub-slot residual
(how far it sits from the center of its own decided slot).

Each frame fits its own band from only its own real `Syncer.sync_residuals` — frame N's real
metadata/data/ECC residuals are only ever compared against frame N's own independently-fit
band, never a pooled one. Every frame's own band and its own real scatter are drawn on the
same axes at low alpha, so it's visible whether the propagation model over- or
under-estimates real jitter away from the sync section, without any cross-frame averaging
hiding the answer.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
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
    output_dir,
    parse_args_with_sidecar,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import differential
from coding_synchronization.measurement.Plotting import _SECTION_COLORS, frame_sections
from coding_synchronization.measurement.SyncMargin import fit_sync_residuals, predicted_sigma
from coding_synchronization.measurement.WaveformLoader import load_waveform
from coding_synchronization.Model import Model2

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")


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
        "--boundary", type=float, default=0.5,
        help="The decode decision boundary, in slots. The default (0.5) is the actual "
             "Syncer rounding boundary.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    return parse_args_with_sidecar(parser)


def _decode_word_residuals(
    pos: np.ndarray, frame_start: float, word_period: float, max_value: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reimplements Syncer._decode_positions, generalized to every word (not just sync).

    Returns (word_indices, residual), where `residual = raw_values - values` is the same
    sub-slot quantity Syncer keeps as `sync_residuals`, but for every pulse in the frame:
    how far it sits from the center of its own decided slot.
    """
    relative = pos - frame_start
    word_indices = np.floor(relative / word_period).astype(np.int64)
    raw_values = relative - word_indices * word_period
    dead_zone_mid = 0.5 * (max_value + word_period)
    wrapped = raw_values > dead_zone_mid
    word_indices = word_indices.copy()
    word_indices[wrapped] += 1
    raw_values = raw_values.copy()
    raw_values[wrapped] -= word_period
    values = np.round(raw_values).clip(0, max_value)
    residual = raw_values - values
    return word_indices, residual


def _section_for_word(word_index: int, section_ranges: list[tuple[str, int, int]]) -> str:
    for name, start, end in section_ranges:
        if start <= word_index < end:
            return name
    return "out-of-range"


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
    sections = frame_sections(frame_params)
    total_words = sum(count for _, count in sections)
    section_ranges: list[tuple[str, int, int]] = []
    cursor = 0
    for name, count in sections:
        section_ranges.append((name, cursor, cursor + count))
        cursor += count

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

    max_value = (1 << args.ppm_rank) - 1
    word_period = model.word_period
    k = np.arange(total_words, dtype=np.float64)

    root = ET.Element(
        "margin_validation", source=wf.source.name, word_period=f"{word_period:.6f}",
        total_words=str(total_words),
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))

    cursor_x = 0.0
    for name, count in sections:
        width = float(count)
        if width > 0:
            ax.axvspan(cursor_x, cursor_x + width, color=_SECTION_COLORS[name], alpha=0.15, label=name)
        cursor_x += width

    n_pulses_total = 0
    n_frames_fit = 0
    for i, (pos, frame_start, sync_residual) in enumerate(
        zip(model.raw_synced_frames, model.frame_starts, model.sync_residuals, strict=True)
    ):
        if len(sync_residual) < 3:
            logger.warning("Frame %d has only %d sync words — skipping (need >= 3 to fit)", i, len(sync_residual))
            continue
        # This frame's own fit, from this frame's own real Syncer.sync_residuals only — no
        # other frame's data enters this calculation.
        fit = fit_sync_residuals(np.arange(len(sync_residual), dtype=np.float64), sync_residual)
        n_frames_fit += 1

        word_indices, residual = _decode_word_residuals(pos, frame_start, word_period, max_value)
        keep = (word_indices >= 0) & (word_indices < total_words)
        word_indices, residual = word_indices[keep], residual[keep]
        n_pulses_total += len(word_indices)

        y_line = fit.slope * k + fit.intercept
        sigma_1 = predicted_sigma(fit, k)
        sigma_3 = 3.0 * sigma_1

        first = n_frames_fit == 1
        ax.plot(k, y_line, color="tab:red", linewidth=0.8, alpha=0.2, zorder=3,
                label="frame fit (from real sync residuals)" if first else None)
        ax.fill_between(k, y_line - sigma_1, y_line + sigma_1, color="tab:red", alpha=0.05, zorder=2,
                         label="±1σ (per frame)" if first else None)
        ax.fill_between(k, y_line - sigma_3, y_line + sigma_3, color="tab:red", alpha=0.03, zorder=1,
                         label="±3σ (per frame)" if first else None)
        if len(word_indices):
            ax.scatter(word_indices, residual, s=6, alpha=0.15, color="black", zorder=4,
                       label="real pulses" if first else None)

        within_1 = within_3 = None
        if len(word_indices):
            predicted_at_real = predicted_sigma(fit, word_indices.astype(np.float64))
            within_1 = float(np.mean(np.abs(residual) <= predicted_at_real))
            within_3 = float(np.mean(np.abs(residual) <= 3.0 * predicted_at_real))

        frame_el = ET.SubElement(
            root, "frame", index=str(i), frame_start=f"{frame_start:.6f}",
            within_1_sigma="" if within_1 is None else f"{within_1:.4f}",
            within_3_sigma="" if within_3 is None else f"{within_3:.4f}",
        )
        ET.SubElement(
            frame_el, "fit", slope=f"{fit.slope:.6g}", intercept=f"{fit.intercept:.6g}",
            s2=f"{fit.s2:.6g}", n=str(fit.n),
        )
        for wi, r in zip(word_indices.tolist(), residual.tolist(), strict=True):
            ET.SubElement(
                frame_el, "pulse", word_index=str(wi), section=_section_for_word(wi, section_ranges),
                residual=f"{r:.6f}",
            )

    logger.info(
        "Decoded %d real pulses across %d/%d synced frames (each fit independently)",
        n_pulses_total, n_frames_fit, len(model.raw_synced_frames),
    )

    ax.axhline(args.boundary, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"decode boundary ±{args.boundary:g}")
    ax.axhline(-args.boundary, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    title = f"Sync-fit margin vs. real pulses — {wf.source.name} ({n_frames_fit} frames, independent)"
    ax.set_xlabel("Word index")
    ax.set_ylabel("Residual (slots)")
    ax.grid(True, alpha=0.4)
    ax.set_title(figure_title(args, title))
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    out = output_dir()
    fig.tight_layout()
    fig.savefig(out / "margin_validation.png", dpi=150)
    logger.info("Saved %s", out / "margin_validation.png")

    xml_path = out / "margin_validation.xml"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    logger.info("Saved %s", xml_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
