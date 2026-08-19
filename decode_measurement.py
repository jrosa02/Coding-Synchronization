"""Feed measured pulse offsets into the existing decoding pipeline.

    python decode_measurement.py output/<ts>/offsets.npz --ppm-rank 10 --dead-slots 8
    python decode_measurement.py measurments/RefCurve_....csv --threshold 0.05   # one-shot

Builds MeasurementGen -> Splitter -> Syncer -> MetadataCheck -> Collector via Model2 and writes the
usual output/<timestamp>/ artifacts (run.log, pipeline.txt, params.json, decoded.npz).
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
    add_verbose_arg,
    add_waveform_args,
    extraction_params,
    log_level,
    min_separation_samples,
    slot_time_s,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import (
    ExtractionParams,
    auto_threshold,
    differential,
    extract_offsets,
    load_offsets,
)
from coding_synchronization.measurement.Plotting import frame_sections, plot_frame_sections
from coding_synchronization.measurement.WaveformLoader import WaveformParams, load_waveform
from coding_synchronization.Model import Model2

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_waveform_args(parser)
    add_extraction_args(parser)
    add_modulation_args(parser)
    add_verbose_arg(parser)
    parser.add_argument("--plot", action="store_true", help="render per-stage tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="only print the first N frames (all frames are still saved to decoded.txt)",
    )
    return parser.parse_args()


def _offsets_from(args: argparse.Namespace) -> tuple[
    np.ndarray, float, ExtractionParams | None, WaveformParams | None
]:
    """Return (offsets in slot units, seconds per slot, extraction params, waveform params)."""
    path = Path(args.path)
    if path.suffix == ".npz":
        slots, meta = load_offsets(path)
        # The .npz is already in slot units; keep the slot time it was extracted with.
        return slots, float(meta.get("slot_time_s", args.slot_time)), None, None

    wp = waveform_params(args)
    wf = load_waveform(wp)
    ex = extraction_params(args)
    diff = differential(wf, ex)
    slot_s = slot_time_s(args, wf.dt_s)
    ex.min_separation_samples = min_separation_samples(args, wf.dt_s, slot_s)
    thr = ex.threshold if ex.threshold is not None else auto_threshold(diff.values)
    offsets = extract_offsets(diff, wf.dt_s, ex, threshold=thr)
    return offsets.to_slots(slot_s), slot_s, ex, wp


def _frame_sections_xml(model: Model2, frame_params: FrameParams) -> ET.Element:
    """<frame_sections word_period=...><frame index=... frame_start=...>
    <section name=sync|metadata|data|ecc|eof start=... end=... words=.../>*
    <pulse position=.../>* (position is relative to frame_start, same units as start/end)
    </frame>*</frame_sections>
    """
    root = ET.Element("frame_sections", word_period=f"{model.word_period:.6f}")
    for i, (frame_start, positions) in enumerate(
        zip(model.frame_starts, model.raw_synced_frames, strict=True)
    ):
        frame_el = ET.SubElement(root, "frame", index=str(i), frame_start=f"{frame_start:.6f}")
        cursor = 0.0
        for name, n in frame_sections(frame_params):
            width = n * model.word_period
            ET.SubElement(
                frame_el, "section",
                name=name, start=f"{cursor:.6f}", end=f"{cursor + width:.6f}", words=str(n),
            )
            cursor += width
        for pos in np.asarray(positions, dtype=np.float64):
            ET.SubElement(frame_el, "pulse", position=f"{pos - frame_start:.6f}")
    return root


def _save_frame_sections(model: Model2, frame_params: FrameParams) -> None:
    """One row per synced frame: raw pulses over semi-transparent sync/metadata/data/ecc/eof bands.

    Saves both frame_sections.png (for a quick look) and frame_sections.xml (the same section
    boundaries and pulse positions, relative to each frame's fitted start, for downstream tooling).
    """
    starts = model.frame_starts
    raw = model.raw_synced_frames
    out_dir = model.output_dir or Path("output")
    if not starts:
        logger.warning("No synced frames to plot frame sections for")
        return

    fig, axes = plt.subplots(len(starts), 1, figsize=(12, 2.2 * len(starts)), squeeze=False)
    for i, (frame_start, positions) in enumerate(zip(starts, raw, strict=True)):
        plot_frame_sections(
            axes[i, 0], positions, frame_start, model.word_period, frame_params,
            title=f"Frame {i}",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "frame_sections.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_dir / "frame_sections.png")

    tree = ET.ElementTree(_frame_sections_xml(model, frame_params))
    ET.indent(tree, space="  ")
    tree.write(out_dir / "frame_sections.xml", encoding="utf-8", xml_declaration=True)
    logger.info("Saved %s", out_dir / "frame_sections.xml")


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    slots, slot_s, ex, wp = _offsets_from(args)
    if len(slots) == 0:
        logger.error("No offsets to decode — nothing to do")
        return

    mod_params = ModulationParams(
        ppm_rank=args.ppm_rank,
        slot_time=np.float64(slot_s),
        dead_slots=args.dead_slots,
    )
    frame_params = FrameParams(
        sync_num=args.sync_num,
        metadata_num=args.metadata_num,
        data_num=args.data_num,
        ecc_num=args.ecc_num,
        eof_num=args.eof_num,
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
        plot=args.plot,
        seed=args.seed,
    )
    model.construct_pipeline()
    model.run()

    if args.plot:
        _save_frame_sections(model, frame_params)

    frames = model.decoded_frames_with_metadata
    meta_num = args.metadata_num
    logger.info(
        "Decoded %d frames — printing every PPM value (first %d per frame are metadata)",
        len(frames), meta_num,
    )
    syncs = model.sync_frames
    found = model.sync_found
    slot_times = model.inferred_slot_time_s
    residuals = model.sync_residuals
    gap_devs = model.sync_gap_devs
    shown = frames if args.max_frames is None else frames[: args.max_frames]
    for i, frame in enumerate(shown):
        values = np.asarray(frame).tolist()
        meta = values[:meta_num]
        rest = values[meta_num:]
        print(f"frame {i:3d} ({len(values)} values)")
        if i < len(syncs):
            sync = syncs[i].tolist()
            n_found = found[i] if i < len(found) else len(sync)
            print(f"  sync[{len(sync)}]: {sync}   ({n_found}/{len(sync)} pulses located)")
        if i < len(residuals) and len(residuals[i]):
            r = residuals[i]
            print(
                f"  sync residual (slots, vs sync_value): {np.round(r, 3).tolist()}   "
                f"rms={np.sqrt(np.mean(r**2)):.3f} max={np.max(np.abs(r)):.3f}"
            )
        if i < len(gap_devs) and len(gap_devs[i]):
            g = gap_devs[i]
            print(
                f"  sync gap - word_period (slots): {np.round(g, 3).tolist()}   "
                f"rms={np.sqrt(np.mean(g**2)):.3f} max={np.max(np.abs(g)):.3f}"
            )
        if i < len(slot_times):
            print(f"  inferred slot_time: {slot_times[i] * 1e9:.4f} ns  (--slot-time was {slot_s * 1e9:.4f} ns)")
        print(f"  metadata[{len(meta)}]: {meta}")
        print(f"  payload+ecc[{len(rest)}]: {rest}")


if __name__ == "__main__":
    main()
