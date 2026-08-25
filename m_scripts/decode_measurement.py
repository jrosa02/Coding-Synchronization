"""Decode measured pulse offsets into frames, and check the frames against their ECC.

    python m_scripts/decode_measurement.py output/<ts>/offsets.npz --ppm-rank 10 --dead-slots 8
    python m_scripts/decode_measurement.py measurments/RefCurve_....csv --threshold 0.05   # one command

The script builds the pipeline MeasurementGen -> Splitter -> Syncer -> MetadataCheck -> Collector
through Model2. It prints every decoded word of each frame, together with the sync diagnostics.
It writes the usual output/<timestamp>/ files: run.log, pipeline.txt, params.json and decoded.npz.

--check-ecc decodes each frame as a Reed-Solomon codeword and reports the error rates before and
after correction. docs/measurement.md describes that check.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization._logging import setup_logging
from coding_synchronization.decoder.Ecc import EccParams, EccReport
from coding_synchronization.encoder import FrameParams, ModulationParams
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
    parse_args_with_sidecar,
    waveform_params,
)
from coding_synchronization.measurement.OffsetExtractor import (
    ExtractionParams,
    differential,
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
    add_slot_calibration_arg(parser)
    add_title_arg(parser)
    add_verbose_arg(parser)
    g = parser.add_argument_group("error correction")
    g.add_argument(
        "--check-ecc", action="store_true",
        help="Decode every frame as a Reed-Solomon codeword. The report gives the word error "
             "rate and the bit error rate before and after correction. The metadata words and "
             "the data words carry the information. The ECC words carry the parity. One PPM word "
             "is one GF(2^ppm_rank) symbol.",
    )
    g.add_argument(
        "--rs-fcr", type=int, default=0,
        help="The first consecutive root of the RS code. The default is 0.",
    )
    g.add_argument(
        "--rs-generator", type=int, default=2,
        help="The generator element of the RS code. The default is 2.",
    )
    g.add_argument(
        "--check-metadata", action="store_true",
        help="Verify that the metadata words of each frame form a consecutive counter. The stage "
             "counts the mismatches and logs them. It runs after the ECC decoding, so it reads "
             "corrected words.",
    )
    g.add_argument(
        "--strict-metadata", action="store_true",
        help="Stop the run at the first metadata mismatch. This also verifies that the counter "
             "continues from one frame to the next.",
    )
    g.add_argument(
        "--rs-prim", type=lambda v: int(v, 0), default=None,
        help="The primitive polynomial of the RS field, for example 0x409 for GF(2^10). The "
             "default comes from --ppm-rank.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Draw one table for each pipeline stage, and one figure for each frame.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Print only the first N frames. The script still saves every frame to "
             "decoded.txt.",
    )
    return parse_args_with_sidecar(parser)


def _offsets_from(args: argparse.Namespace) -> tuple[
    np.ndarray, float, ExtractionParams | None, WaveformParams | None
]:
    """Return (offsets in slot units, seconds per slot, extraction params, waveform params)."""
    path = Path(args.path)
    if path.suffix == ".npz":
        with np.load(path) as npz:
            is_offsets_file = "offsets_slots" in npz
        if is_offsets_file:
            slots, meta = load_offsets(path)
            # The .npz is already in slot units; keep the slot time it was extracted with.
            return slots, float(meta.get("slot_time_s", args.slot_time)), None, None
        # else: a raw waveform .npz (ch_a/ch_b/dt_s) — fall through to the waveform path below.

    wp = waveform_params(args)
    wf = load_waveform(wp)
    ex = extraction_params(args)
    diff = differential(wf, ex)
    offsets, slot_s, _thr = extract_calibrated(diff, wf, ex, args)
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


def _save_frame_sections(
    model: Model2, frame_params: FrameParams, args: argparse.Namespace
) -> None:
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
    apply_suptitle(fig, args)
    fig.tight_layout()
    fig.savefig(out_dir / "frame_sections.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_dir / "frame_sections.png")

    tree = ET.ElementTree(_frame_sections_xml(model, frame_params))
    ET.indent(tree, space="  ")
    tree.write(out_dir / "frame_sections.xml", encoding="utf-8", xml_declaration=True)
    logger.info("Saved %s", out_dir / "frame_sections.xml")


def _print_ecc_report(report: EccReport) -> None:
    """Frame-by-frame RS verdict, then the rates either side of correction."""
    p = report.params
    rates = report.rates()
    print(f"\nECC check — RS({p.n},{p.info_num}) over GF(2^{p.ppm_rank}), "
          f"corrects up to {p.correctable} symbols/frame")
    for f in report.frames:
        if not f.ok:
            print(f"  frame {f.index:3d}: UNCORRECTABLE")
        elif f.symbol_errors:
            print(f"  frame {f.index:3d}: corrected {f.symbol_errors} symbols "
                  f"({f.bit_errors} bits) at {f.positions}")
    clean = sum(1 for f in report.frames if f.ok and f.symbol_errors == 0)
    print(f"  {report.n_frames} frames: {clean} clean, "
          f"{report.n_frames - clean - report.n_uncorrectable} corrected, "
          f"{report.n_uncorrectable} uncorrectable")
    print(f"  {'':24} {'WER (symbol)':>14} {'BER (bit)':>14}")
    print(f"  {f'before ECC (of {report.n_decoded})':24} "
          f"{rates['wer_pre']:>14.3e} {rates['ber_pre']:>14.3e}")
    print(f"  frame error rate (uncorrectable frames): {report.frame_error_rate:.3e}")
    if report.n_uncorrectable:
        # The pre-ECC rates cover the decoded frames only. An uncorrectable frame has no
        # reference, so its error count cannot be measured, and it is counted here instead.
        print(f"  the {report.n_uncorrectable} uncorrectable frames carry no reference, so they "
              f"are outside the rates above")


def _save_ecc_report(report: EccReport, out_dir: Path) -> None:
    p = report.params
    rates = report.rates()
    root = ET.Element(
        "ecc_report", n=str(p.n), k=str(p.info_num), ppm_rank=str(p.ppm_rank),
        ecc_num=str(p.ecc_num), fcr=str(p.fcr), generator=str(p.generator),
        correctable=str(p.correctable), frames=str(report.n_frames),
        uncorrectable=str(report.n_uncorrectable),
        frame_error_rate=f"{report.frame_error_rate:.6e}",
        **{k: f"{v:.6e}" for k, v in rates.items()},
    )
    for f in report.frames:
        ET.SubElement(
            root, "frame", index=str(f.index), ok=str(f.ok).lower(),
            symbol_errors=str(f.symbol_errors), bit_errors=str(f.bit_errors),
            positions=" ".join(str(j) for j in f.positions),
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_dir / "ecc_report.xml", encoding="utf-8", xml_declaration=True)
    logger.info("Saved %s", out_dir / "ecc_report.xml")


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

    ecc_params = None
    if args.check_ecc:
        # The codeword is the frame minus its sync words: metadata + data carry the information,
        # the ecc words are the parity over both.
        ecc_params = EccParams(
            ppm_rank=args.ppm_rank,
            ecc_num=args.ecc_num,
            info_num=args.metadata_num + args.data_num,
            fcr=args.rs_fcr,
            generator=args.rs_generator,
            prim=args.rs_prim,
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
        ecc_params=ecc_params,
        verify_metadata=args.check_metadata,
        strict_metadata=args.strict_metadata,
    )
    model.construct_pipeline()
    model.run()

    if args.plot:
        _save_frame_sections(model, frame_params, args)

    if model.ecc_report is not None:
        _print_ecc_report(model.ecc_report)
        _save_ecc_report(model.ecc_report, model.output_dir or Path("output"))

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
