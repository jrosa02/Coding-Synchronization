import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from coding_synchronization.measurement.OffsetExtractor import ExtractionParams
from coding_synchronization.measurement.WaveformLoader import WaveformParams
from coding_synchronization.physical_units import Hz, Quantity

logger = logging.getLogger(__name__)


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="enable DEBUG-level logging (pulse gaps, per-frame calibration detail, ...)",
    )


def log_level(args: argparse.Namespace) -> int:
    return logging.DEBUG if args.verbose else logging.INFO


def add_waveform_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("waveform")
    g.add_argument("path", type=Path, help="two-column differential CSV (R&S RefCurve export)")
    g.add_argument(
        "--sample-rate", type=float, default=None,
        help="samples per second; required only if the CSV has no time column",
    )
    g.add_argument("--pos-col", type=int, default=None, help="0-based column index of P")
    g.add_argument("--neg-col", type=int, default=None, help="0-based column index of N")
    g.add_argument(
        "--max-samples", type=int, default=None, help="load only the first N samples"
    )


def add_extraction_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("extraction")
    g.add_argument(
        "--threshold", type=float, default=None,
        help="detection threshold on the combined pair; default: auto (half amplitude)",
    )
    g.add_argument(
        "--combine", choices=("add", "sub"), default="add",
        help="how to merge the zero-centred pair: add (scope inverts one leg, default) or sub",
    )
    g.add_argument(
        "--polarity", choices=("auto", "pos", "neg"), default="auto",
        help="auto flips the combined signal if its pulses point down",
    )
    g.add_argument(
        "--baseline", choices=("median", "mean", "none"), default="median",
        help="per-channel DC pedestal estimator, removed before combining",
    )
    g.add_argument(
        "--hysteresis", type=float, default=0.5,
        help="a pulse ends only once it falls below hysteresis*threshold (merges ringing)",
    )
    g.add_argument("--min-width", type=int, default=1, help="drop runs narrower than N samples")
    g.add_argument(
        "--min-gap-slots", type=float, default=None,
        help="merge detections closer than this many slots (same physical pulse); "
             "must be <= eof_num, which is also the default",
    )
    g.add_argument(
        "--edge", choices=("centroid", "peak", "edge", "rising"), default="centroid",
        help="pulse position: amplitude-weighted centroid (default), argmax, mean of the "
             "interpolated rising/falling edge crossings (best for clean, flat-topped pulses), "
             "or rising-edge crossing only (ignores trailing-edge/width jitter entirely)",
    )


def add_modulation_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("modulation / framing")
    g.add_argument(
        "--slot-time", type=float, default=1.0 / 32.1429e6,
        help="PPM slot duration in seconds; default is 1/(serializing clock = 32.1429 MHz)",
    )
    g.add_argument(
        "--samples-per-slot", type=float, default=None,
        help="samples per PPM slot; overrides --slot-time and makes --sample-rate unnecessary "
             "(measure it as median_inter_pulse_gap_samples / word_period)",
    )
    g.add_argument("--ppm-rank", type=int, default=10)
    g.add_argument("--dead-slots", type=int, default=8)
    g.add_argument("--sync-num", type=int, default=8)
    g.add_argument(
        "--sync-value", type=int, default=None,
        help="PPM value carried by each sync word (default: 0). "
             "Getting this wrong offsets every decoded word by the same amount",
    )
    g.add_argument("--metadata-num", type=int, default=4)
    g.add_argument("--data-num", type=int, default=240)
    g.add_argument("--ecc-num", type=int, default=4)
    g.add_argument("--eof-num", type=int, default=64)
    g.add_argument(
        "--keep-first-frame", action="store_true",
        help="keep the first frame; by default it is dropped because the scope trigger "
             "lands mid-frame, so the capture starts with a fragment",
    )
    g.add_argument(
        "--drop-partial-frames", action="store_true",
        help="also drop any frame whose pulse count != sync+metadata+data+ecc",
    )


def waveform_params(args: argparse.Namespace) -> WaveformParams:
    rate: Quantity | None = args.sample_rate * Hz if args.sample_rate else None
    if rate is None and getattr(args, "samples_per_slot", None):
        # No real rate needed: with dt = 1 s a "second" is one sample, and --samples-per-slot
        # carries the only conversion the decoder actually depends on.
        rate = 1.0 * Hz
    return WaveformParams(
        path=args.path,
        sample_rate=rate,
        pos_col=args.pos_col,
        neg_col=args.neg_col,
        max_samples=args.max_samples,
    )


def extraction_params(args: argparse.Namespace) -> ExtractionParams:
    return ExtractionParams(
        threshold=args.threshold,
        polarity=args.polarity,
        baseline=args.baseline,
        combine=args.combine,
        hysteresis=args.hysteresis,
        min_width_samples=args.min_width,
        method=args.edge,
    )


def slot_time_s(args: argparse.Namespace, dt_s: float) -> float:
    """Seconds per PPM slot — from --samples-per-slot when given, else --slot-time."""
    if getattr(args, "samples_per_slot", None):
        return float(args.samples_per_slot) * dt_s
    return float(args.slot_time)


def min_gap_slots(args: argparse.Namespace) -> float:
    """Detections closer than this many slots are one physical pulse. Bounded by eof_num."""
    # Never merge across more than the closest two PPM pulses can physically be
    # (word_period - max_value == dead_slots + 1), and never more than eof_num.
    physical_min = args.dead_slots + 1
    default = min(args.eof_num, physical_min)
    if args.eof_num > physical_min:
        logger.info(
            "--eof-num %d exceeds the physical minimum pulse spacing of %d slots "
            "(dead_slots + 1); using %d as the merge distance",
            args.eof_num, physical_min, default,
        )
    gap_slots = args.min_gap_slots
    if gap_slots is None:
        return float(default)
    if gap_slots > args.eof_num:
        logger.warning(
            "--min-gap-slots %.3f exceeds --eof-num %d; clamping to %d",
            gap_slots, args.eof_num, args.eof_num,
        )
        return float(args.eof_num)
    return float(gap_slots)


def min_separation_samples(args: argparse.Namespace, dt_s: float, slot_s: float) -> float:
    """Same bound as min_gap_slots, expressed in samples."""
    return min_gap_slots(args) * slot_s / dt_s


def split_threshold(args: argparse.Namespace) -> float:
    """Splitter gap threshold in slot units — same derivation as Model1."""
    word_period = (1 << args.ppm_rank) + args.dead_slots
    return float(args.eof_num * word_period)


def split_replica(
    offsets_samples: np.ndarray,
    slots: np.ndarray,
    expected_total: int,
    sync_num: int,
    eof_num: int,
    word_period: float,
    keep_first_frame: bool,
    drop_partial_frames: bool,
) -> list[np.ndarray]:
    """Replicate Splitter + FrameFilter's frame grouping directly on absolute sample-unit
    offsets, in the same order Syncer receives frames.

    Splitter zeroes each chunk (`chunk - chunk[0]`) before Syncer ever sees it, so
    `Model2.frame_starts`/`raw_synced_frames` are frame-local, not absolute against the file.
    This reproduces the exact same split (same threshold, same input order) so the caller can
    recover which real samples belong to a given synced-frame index. The returned list aligns
    1:1, in order, with `Model2.raw_synced_frames` — every element here satisfies the same
    `len >= sync_num` condition that decides whether Syncer keeps a frame.
    """
    threshold = eof_num * word_period
    boundaries = np.where(np.diff(slots) > threshold)[0] + 1
    raw_chunks = [c for c in np.split(offsets_samples, boundaries) if len(c) > 0]

    kept = []
    for i, chunk in enumerate(raw_chunks):
        if not keep_first_frame and i == 0:
            continue
        if drop_partial_frames and len(chunk) != expected_total:
            continue
        kept.append(chunk)
    return [c for c in kept if len(c) >= sync_num]


def output_dir(name: str | None = None) -> Path:
    """output/<YYYY-MM-DD_HH-MM-SS>/ — the project-wide artifact convention."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path("output") / (stamp if name is None else f"{stamp}_{name}")
    path.mkdir(parents=True, exist_ok=True)
    return path
