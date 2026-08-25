"""Map word error rate before ECC against sync pulse count and clock jitter, with Model1.

    python s_scripts/simulate_wer_map_sync_jitter.py --no-show
    python s_scripts/simulate_wer_map_sync_jitter.py --sync-values 2,3,4,5,6,8 --sigma-count 20

This answers a design question, not a coding-gain question: given a clock of some stability, how
many sync pulses does a frame need to keep timing errors from corrupting its words? The frame
still carries the 16 ECC parity words of the 24082026 capture layout, so its duration and the
timing error accumulated over it match that layout exactly, but the ECC decoder never runs
(`Model1(run_ecc=False)`) — reporting a corrected rate would answer a different question, and
Reed-Solomon decode is the most expensive stage per frame, so skipping it is what makes a 2D grid
of this size affordable.

The only impairment is `RandomShift`, which adds one Gaussian draw to each pulse offset and
returns an array of the same length. No pulse is inserted, and none is removed. `sigma` is the
grid's second axis, swept log-spaced between `--sigma-min` and `--sigma-max`; the Hardware Test 2
value 0.00465 slot (RMS residual, 09_tests.tex) falls inside the default range as a reference
point.

Model1 knows every word it sent, so this compares the decoded words against the sent words
directly (`model._synced.frames` vs `frames_sent`), rather than reading an `EccReport`. The word
rate and the bit rate are counted separately and both are exact.

A frame that comes back with the wrong word count counts as a total loss. `Syncer` drops a word
when timing error pushes two pulses into one word index, and that failure has to stay visible
here rather than being silently excluded.

Each `(sync_num, sigma)` grid cell is independent and is computed in its own process
(`ProcessPoolExecutor`), averaged over `--seeds` noise realizations.
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from coding_synchronization._logging import setup_logging
from coding_synchronization.channel.Channel import ChannelParams
from coding_synchronization.encoder import FrameParams, ModulationParams
from coding_synchronization.encoder.PassageGen import PassageParams
from coding_synchronization.measurement.Cli import (
    add_title_arg,
    add_verbose_arg,
    figure_title,
    log_level,
    output_dir,
)
from coding_synchronization.Model import Model1

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")

# The 24082026 capture layout, so the simulated grid overlays the measured 8/4/2 points.
PPM_RANK = 10
DEAD_SLOTS = 16
SLOT_TIME_S = 1.666632354617692e-05
METADATA_NUM = 5
DATA_NUM = 200
ECC_NUM = 16

# Hardware Test 2, 09_tests.tex: RMS residual 0.00465 slot (0.388 ns), max 0.0118 slot.
DEFAULT_SIGMA = 0.00465


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sync-values", type=str, default="2,4,8,16,32",
        help="Comma-separated sync_num values to sweep. The default gives the powers of two. "
             "Add intermediate values to resolve the knee more finely.",
    )
    parser.add_argument(
        "--sigma-min", type=float, default=0.0005,
        help="Smallest per-pulse timing jitter to sweep, in slots.",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=0.05,
        help="Largest per-pulse timing jitter to sweep, in slots.",
    )
    parser.add_argument(
        "--sigma-count", type=int, default=12,
        help=f"Number of log-spaced sigma points between --sigma-min and --sigma-max. The "
             f"default range brackets the measured Hardware Test 2 value ({DEFAULT_SIGMA} slot).",
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Independent noise realizations per grid cell. The seed drives both the payload and "
             "the noise, because StageRunner re-seeds every stage.",
    )
    parser.add_argument(
        "--frames", type=int, default=200,
        help="Frames per run, the same for every grid cell. This sets the resolution of the "
             "measurement: the smallest non-zero word error rate visible is 1/(frames x seeds x "
             "221).",
    )
    parser.add_argument(
        "--eof-num", type=int, default=64,
        help="Gap between transmitted frames, in words.",
    )
    parser.add_argument(
        "--split-eof-num", type=int, default=8,
        help="Splitter threshold, in words. This is a receiver setting, and it does not have to "
             "equal --eof-num. It only has to sit between the largest gap inside a frame and the "
             "smallest gap between frames.",
    )
    parser.add_argument(
        "--altitude-km", type=float, default=500.0,
        help="Pass altitude. Doppler is off for this sweep and the frame count is set directly, "
             "so this value does not affect any result. It is recorded for the run parameters.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for the grid. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    add_title_arg(parser)
    add_verbose_arg(parser)
    return parser.parse_args()


def _check_split_window(sync_num: int, eof_num: int, split_eof_num: int) -> None:
    """Warn when the Splitter threshold sits outside the window that can split frames correctly.

    Two words inside a frame sit `word_period + v_next - v_prev` slots apart, so the largest gap
    inside a frame is `word_period + max_value`. The gap between frames is
    `(eof_num + 1) * word_period - v_last`, smallest when the last word carries max_value.
    """
    word_period = (1 << PPM_RANK) + DEAD_SLOTS
    max_value = (1 << PPM_RANK) - 1
    floor_slots = word_period + max_value
    ceiling_slots = (eof_num + 1) * word_period - max_value
    threshold = split_eof_num * word_period
    if not floor_slots < threshold < ceiling_slots:
        logger.warning(
            "sync_num=%d: Splitter threshold %d slots is outside the safe window (%d, %d) — "
            "frames may split in the wrong place",
            sync_num, threshold, floor_slots, ceiling_slots,
        )


def _frame_errors(sent: np.ndarray, received: np.ndarray, n: int) -> tuple[int, int]:
    """Word errors and bit errors of one frame, against what was sent.

    A frame of the wrong length is a total loss: `Syncer` collapsed two pulses into one word
    index, or lost one, so the words no longer line up with what was sent and no per-word
    comparison is meaningful. Charge every word and every bit.
    """
    if len(received) != n:
        return n, n * PPM_RANK
    diff = np.asarray(received, dtype=np.int64) ^ np.asarray(sent, dtype=np.int64)
    bad = np.nonzero(diff)[0]
    if len(bad) == 0:
        return 0, 0
    bit_errors = int(sum(int(d).bit_count() for d in diff[bad]))
    return len(bad), bit_errors


def _run_seed(sync_num: int, sigma: float, seed: int, frames: int, eof_num: int,
              split_eof_num: int, altitude_km: float) -> dict:
    """One Model1 run, ECC decode skipped. Returns pre-ECC word/bit error counts."""
    mod_params = ModulationParams(
        ppm_rank=PPM_RANK, slot_time=np.float64(SLOT_TIME_S), dead_slots=DEAD_SLOTS
    )
    frame_params = FrameParams(
        sync_num=sync_num, metadata_num=METADATA_NUM, data_num=DATA_NUM,
        ecc_num=ECC_NUM, eof_num=eof_num,
    )

    # Ask for the frame count directly. Deriving it from a pass duration would make it depend on
    # sync_num, because a larger sync_num makes the frame longer, and it would also cap the count
    # at whatever the orbit geometry allows. Doppler is off here, so the orbit changes nothing.
    passage_params = PassageParams(
        altitude_km=altitude_km, max_elevation_deg=90.0, n_frames=frames
    )
    channel_params = ChannelParams(
        sigma=sigma, vanish_rate=None, max_const_offset=None, added_rate=None, doppler=False,
    )

    model = Model1(
        data=None, frame_params=frame_params, mod_params=mod_params,
        overflight_params=passage_params, channel_params=channel_params,
        plot=False, seed=seed, split_eof_num=split_eof_num, run_ecc=False,
    )
    model.construct_pipeline()

    if model.n_frames != frames:
        logger.warning(
            "sync_num=%d sigma=%g seed=%d: got %d frames, asked for %d — the pass window was "
            "capped", sync_num, sigma, seed, model.n_frames, frames,
        )

    # PassageGen fixes n_frames at construction, so size the payload only now.
    rng = np.random.default_rng(seed)
    total_words = model.n_frames * DATA_NUM
    model.data = rng.integers(0, 1 << PPM_RANK, total_words, dtype=np.uint16)
    model.run(save_artifacts=False)

    # The sent codeword, without the sync words. Syncer already strips those on the received side.
    frames_sent = model._gen._frame_gen.frames_sent[:, sync_num:]
    received = model._synced.frames
    n = METADATA_NUM + DATA_NUM + ECC_NUM
    n_frames = len(frames_sent)

    if len(received) != n_frames:
        # Only offset noise is applied, so no frame should appear or disappear. If one does, the
        # per-frame comparison below would compare mismatched frames and report nonsense.
        raise RuntimeError(
            f"sync_num={sync_num} sigma={sigma:g} seed={seed}: sent {n_frames} frames but "
            f"received {len(received)} — frame alignment is broken"
        )

    words = bits = dirty_frames = within_ecc_limit = 0
    burst_lengths: list[int] = []
    correctable = ECC_NUM // 2
    for sent, got in zip(frames_sent, received, strict=True):
        w, b = _frame_errors(sent, got, n)
        words += w
        bits += b
        if w:
            dirty_frames += 1
            burst_lengths.append(w)
            # RS(221,205) corrects up to 8 symbols. Count how often a damaged frame would have
            # been inside that limit, for reference — ECC itself never runs here.
            if w <= correctable:
                within_ecc_limit += 1

    total_words_cmp = n_frames * n
    total_bits_cmp = total_words_cmp * PPM_RANK

    return {
        "sync_num": sync_num,
        "sigma": sigma,
        "seed": seed,
        "frames": n_frames,
        "wer_pre": words / total_words_cmp,
        "ber_pre": bits / total_bits_cmp,
        "damaged_frames": dirty_frames,
        "damaged_within_ecc_limit": within_ecc_limit,
        "mean_bad_words_when_damaged": float(np.mean(burst_lengths)) if burst_lengths else 0.0,
    }


def _run_grid_point(sync_num: int, sigma: float, seeds: int, frames: int, eof_num: int,
                     split_eof_num: int, altitude_km: float) -> dict:
    """One grid cell: every seed for one (sync_num, sigma) pair, run in this worker process."""
    rows = [
        _run_seed(sync_num, sigma, seed, frames, eof_num, split_eof_num, altitude_km)
        for seed in range(seeds)
    ]
    entry: dict = {"sync_num": sync_num, "sigma": sigma, "seeds": len(rows)}
    for key in ("wer_pre", "ber_pre"):
        values = [r[key] for r in rows]
        entry[key] = float(np.mean(values))
        entry[f"{key}_min"] = float(np.min(values))
        entry[f"{key}_max"] = float(np.max(values))
    entry["damaged_frames"] = sum(r["damaged_frames"] for r in rows)
    entry["damaged_within_ecc_limit"] = sum(r["damaged_within_ecc_limit"] for r in rows)
    bad = [r["mean_bad_words_when_damaged"] for r in rows if r["damaged_frames"]]
    entry["mean_bad_words_when_damaged"] = float(np.mean(bad)) if bad else 0.0
    entry["rows"] = rows
    return entry


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    sync_values = [int(v) for v in args.sync_values.split(",") if v.strip()]
    if not sync_values:
        raise SystemExit("--sync-values is empty")
    if min(sync_values) < 2:
        raise SystemExit("sync_num must be at least 2 — Syncer needs one sync-to-sync gap")
    sigma_values = np.geomspace(args.sigma_min, args.sigma_max, args.sigma_count).tolist()

    for sync_num in sync_values:
        _check_split_window(sync_num, args.eof_num, args.split_eof_num)

    grid: dict[tuple[int, float], dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_grid_point, sync_num, sigma, args.seeds, args.frames,
                args.eof_num, args.split_eof_num, args.altitude_km,
            ): (sync_num, sigma)
            for sync_num in sync_values for sigma in sigma_values
        }
        for i, future in enumerate(as_completed(futures), start=1):
            sync_num, sigma = futures[future]
            entry = future.result()
            grid[(sync_num, sigma)] = entry
            logger.info(
                "[%d/%d] sync_num=%2d sigma=%.4g: wer_pre=%.3e ber_pre=%.3e damaged=%d/%d",
                i, len(futures), sync_num, sigma, entry["wer_pre"], entry["ber_pre"],
                entry["damaged_frames"], entry["seeds"] * args.frames,
            )

    wer_grid = np.array(
        [[grid[(sync_num, sigma)]["wer_pre"] for sigma in sigma_values] for sync_num in sync_values]
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    # A rate of exactly zero has no place on a log color scale. Floor it so a true zero draws as
    # the bottom of the color range rather than breaking the norm.
    positive = wer_grid[wer_grid > 0]
    floor = positive.min() / 10.0 if positive.size else 1e-6
    drawn = np.where(wer_grid > 0, wer_grid, floor)

    mesh = ax.pcolormesh(
        sigma_values, sync_values, drawn,
        norm=LogNorm(vmin=floor, vmax=drawn.max()), shading="nearest", cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="Pre-ECC word error rate")

    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_yticks(sync_values)
    ax.set_yticklabels([str(v) for v in sync_values])
    ax.set_xlabel("Per-pulse timing jitter, sigma (slots)")
    ax.set_ylabel("Sync pulses per frame (sync_num)")

    default_title = (
        f"Pre-ECC WER map: sync_num vs jitter ({args.frames} frames, {args.seeds} seeds/cell)"
    )
    fig.suptitle(figure_title(args, default_title))
    fig.tight_layout()

    out = output_dir()
    fig.savefig(out / "wer_map_sync_jitter.png", dpi=150)
    logger.info("Saved %s", out / "wer_map_sync_jitter.png")

    payload = {
        "sync_values": sync_values,
        "sigma_values": sigma_values,
        "frames": args.frames,
        "seeds": args.seeds,
        "eof_num": args.eof_num,
        "split_eof_num": args.split_eof_num,
        "layout": {
            "ppm_rank": PPM_RANK, "dead_slots": DEAD_SLOTS, "slot_time_s": SLOT_TIME_S,
            "metadata_num": METADATA_NUM, "data_num": DATA_NUM, "ecc_num": ECC_NUM,
        },
        "grid": [grid[(sync_num, sigma)] for sync_num in sync_values for sigma in sigma_values],
    }
    json_path = out / "wer_map_sync_jitter.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %s", json_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
