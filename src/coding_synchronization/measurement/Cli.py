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
        help="Write DEBUG messages to the log. These include the pulse gaps and the calibration "
             "of each frame.",
    )


def log_level(args: argparse.Namespace) -> int:
    return logging.DEBUG if args.verbose else logging.INFO


def add_title_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--title", type=str, default=None,
        help="Set the figure title and replace the generated one. On a figure with more than one "
             "panel, this title goes above all the panels.",
    )


def figure_title(args: argparse.Namespace, default: str) -> str:
    """The single-axes title to use: --title when given, else the generated one."""
    title = getattr(args, "title", None)
    return default if title is None else title


def apply_suptitle(fig, args: argparse.Namespace) -> None:
    """Put --title above a multi-panel figure, leaving the per-panel titles alone.

    Call before `fig.tight_layout()` so the suptitle is accounted for in the layout.
    """
    title = getattr(args, "title", None)
    if title:
        fig.suptitle(title)


def add_waveform_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("waveform")
    g.add_argument(
        "path", type=Path,
        help="The waveform CSV file. The file holds two differential columns, as the R&S RefCurve "
             "export does. A single-channel export with a tInc= header line also works.",
    )
    g.add_argument(
        "--sample-rate", type=float, default=None,
        help="The number of samples per second. Give this option only when the CSV has no time "
             "column.",
    )
    g.add_argument(
        "--pos-col", type=int, default=None,
        help="The column index of channel P. The first column is 0.",
    )
    g.add_argument(
        "--neg-col", type=int, default=None,
        help="The column index of channel N. The first column is 0.",
    )
    g.add_argument(
        "--max-samples", type=int, default=None, help="Load only the first N samples of the file."
    )


def add_extraction_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("extraction")
    g.add_argument(
        "--threshold", type=float, default=None,
        help="The detection threshold on the combined signal. The default is half of the pulse "
             "amplitude, measured from the capture.",
    )
    g.add_argument(
        "--combine", choices=("add", "sub"), default="add",
        help="How to merge the two zero-centred channels. Use add when the measuring device "
             "inverts one leg. This is the default.",
    )
    g.add_argument(
        "--polarity", choices=("auto", "pos", "neg"), default="auto",
        help="The direction of the pulses. auto inverts the combined signal when the pulses point "
             "down.",
    )
    g.add_argument(
        "--baseline", choices=("median", "mean", "none"), default="median",
        help="How to estimate the DC level of each channel. The script removes this level before "
             "it merges the channels.",
    )
    g.add_argument(
        "--hysteresis", type=float, default=0.5,
        help="A pulse ends when the signal falls below hysteresis * threshold. A value below 1 "
             "merges the ringing into the pulse.",
    )
    g.add_argument(
        "--min-width", type=int, default=1,
        help="Discard every detection narrower than N samples.",
    )
    g.add_argument(
        "--min-gap-slots", type=float, default=None,
        help="Merge two detections closer than this many slots into one pulse. The value must be "
             "less than or equal to eof_num, which is also the default.",
    )
    g.add_argument(
        "--edge", choices=("centroid", "peak", "edge", "rising"), default="centroid",
        help="How to compute the position of a pulse. centroid is the amplitude-weighted centre "
             "and the default. peak is the highest sample. edge is the mean of the two "
             "interpolated crossings, which suits clean flat-topped pulses. rising uses the "
             "rising crossing only, and ignores the width of the pulse.",
    )


def add_modulation_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("modulation / framing")
    g.add_argument(
        "--slot-time", type=float, default=1.0 / 32.1429e6,
        help="The duration of one PPM slot, in seconds. The default is one period of the "
             "32.1429 MHz serializing clock.",
    )
    g.add_argument(
        "--samples-per-slot", type=float, default=None,
        help="The number of samples in one PPM slot. This option replaces --slot-time, and it "
             "makes --sample-rate unnecessary. To measure it, divide the median pulse gap in "
             "samples by the word period.",
    )
    g.add_argument(
        "--ppm-rank", type=int, default=10,
        help="One PPM word carries 2^ppm_rank possible values.",
    )
    g.add_argument(
        "--dead-slots", type=int, default=8,
        help="The number of slots that follow the PPM range of each word.",
    )
    g.add_argument(
        "--sync-num", type=int, default=8, help="The number of sync words in each frame.",
    )
    g.add_argument(
        "--sync-value", type=int, default=None,
        help="The PPM value that each sync word carries. The default is 0. A wrong value moves "
             "every decoded word by the same amount.",
    )
    g.add_argument(
        "--metadata-num", type=int, default=4,
        help="The number of metadata words in each frame.",
    )
    g.add_argument(
        "--data-num", type=int, default=240, help="The number of data words in each frame.",
    )
    g.add_argument(
        "--ecc-num", type=int, default=4, help="The number of ECC words in each frame.",
    )
    g.add_argument(
        "--eof-num", type=int, default=64,
        help="The length of the gap between two frames, in words. The splitter cuts the capture "
             "at every gap longer than eof_num word periods.",
    )
    g.add_argument(
        "--keep-first-frame", action="store_true",
        help="Keep the first frame. The script drops it by default, because the trigger of the "
             "measuring device lands inside a frame and the capture starts with a fragment.",
    )
    g.add_argument(
        "--drop-partial-frames", action="store_true",
        help="Also drop every frame that does not hold exactly sync + metadata + data + ecc "
             "pulses.",
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


def add_slot_calibration_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-auto-slot", dest="auto_slot", action="store_false",
        help="Use --slot-time or --samples-per-slot as given. By default the script measures the "
             "slot time from the median pulse gap. That measurement keeps the results correct "
             "when --sample-rate is wrong.",
    )
    parser.set_defaults(auto_slot=True)


def measured_slot_time_s(
    offsets_samples: np.ndarray, word_period: float, dt_s: float, assumed_slot_s: float
) -> float | None:
    """Slot duration implied by the data: median inter-pulse gap / word_period.

    Every PPM word carries exactly one pulse, so the *typical* gap between consecutive pulses is
    one word period whatever values are being sent — the PPM value only jitters it either side.
    The median of that distribution is therefore a direct measurement of the slot duration, in
    the same sample domain the pulses were found in.

    Returns None when there are too few pulses to measure.
    """
    if len(offsets_samples) < 3 or word_period <= 0:
        return None
    median_gap = float(np.median(np.diff(np.sort(offsets_samples))))
    if not np.isfinite(median_gap) or median_gap <= 0.0:
        return None
    slot_s = median_gap * dt_s / word_period
    ratio = slot_s / assumed_slot_s if assumed_slot_s > 0 else float("inf")
    log = logger.warning if abs(ratio - 1.0) > 0.02 else logger.info
    log(
        "Slot time from the data: %.6g s (%.4f samples/slot) vs %.6g s as given — ratio %.4f",
        slot_s, median_gap / word_period, assumed_slot_s, ratio,
    )
    return slot_s


def extract_calibrated(
    diff, wf, ex: ExtractionParams, args: argparse.Namespace
) -> tuple[object, float, float]:
    """Extract pulses, then calibrate the slot time against them. Returns (offsets, slot_s, thr).

    Everything downstream is expressed in slot units, and the frame split threshold in particular
    is `eof_num * word_period` *slots*. So a wrong --slot-time (or a wrong --sample-rate feeding
    it) moves that threshold in real samples: too small and it cuts inside frames, too large and
    it never fires — either way the chunks handed to the Syncer are not frames, and every result
    downstream is meaningless while looking perfectly well-formed. Measuring the slot time from
    the pulses themselves makes all of it independent of what --sample-rate was guessed at.

    The first extraction still needs *some* slot time for its merge distance, so this runs a
    second pass whenever the measured value moves that distance materially.
    """
    from coding_synchronization.measurement.OffsetExtractor import auto_threshold, extract_offsets

    word_period = float((1 << args.ppm_rank) + args.dead_slots)
    slot_s = slot_time_s(args, wf.dt_s)
    ex.min_separation_samples = min_separation_samples(args, wf.dt_s, slot_s)
    thr = ex.threshold if ex.threshold is not None else auto_threshold(diff.values)
    offsets = extract_offsets(diff, wf.dt_s, ex, threshold=thr)

    if not getattr(args, "auto_slot", True) or len(offsets.samples) == 0:
        return offsets, slot_s, thr

    measured = measured_slot_time_s(offsets.samples, word_period, wf.dt_s, slot_s)
    if measured is None:
        return offsets, slot_s, thr

    if abs(measured / slot_s - 1.0) > 0.01:
        # The merge distance was derived from the wrong slot time, so redo the extraction with
        # the measured one before anything reads the pulse positions.
        ex.min_separation_samples = min_separation_samples(args, wf.dt_s, measured)
        offsets = extract_offsets(diff, wf.dt_s, ex, threshold=thr)
        logger.info(
            "Re-extracted with the measured slot time: %d pulses", len(offsets.samples)
        )
    return offsets, measured, thr


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
