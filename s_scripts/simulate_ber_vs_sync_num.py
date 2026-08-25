"""Simulate WER/BER against the number of sync pulses, with Model1.

    python m_scripts/simulate_ber_vs_sync_num.py --no-show
    python m_scripts/simulate_ber_vs_sync_num.py --sync-values 2,3,4,5,6,8 --seeds 20

The bench measurement gave three points only: sync_num 8, 4 and 2. This sweeps the same frame
layout in simulation, so the curve between and beyond those points becomes visible.

The only impairment is `RandomShift`, which adds one Gaussian draw to each pulse offset and
returns an array of the same length. No pulse is inserted, and none is removed. `sigma` defaults
to 0.00465 slot, the RMS residual measured in Hardware Test 2 (see 09_tests.tex in the thesis
repository).

Model1 knows every word it sent, so this compares the decoded words against the sent words
instead of reading `EccReport.rates()`. On a real capture there is no reference copy, so
`rates()` has to charge every uncorrectable frame in full, which makes `wer_post`, `ber_post` and
`frame_error_rate` one number under three names. Here the word rate and the bit rate are counted
separately and both are exact.

The plot shows post-ECC only. The pre-ECC rates are computed and kept in the JSON, but they are
not drawn, because they sit almost exactly on the post-ECC curve. That overlap is itself the
result: a timing error is progressive, so the words it corrupts run in one block from the word
where the accumulated error passes half a slot to the end of the frame. That block is almost
always far longer than the 8 symbols RS(221,205) can correct, so the parity changes nothing and
the frame is lost whole. Reed-Solomon parity answers sprinkled symbol errors, not drift.

A frame that comes back with the wrong word count counts as a total loss. `Syncer` drops a word
when timing error pushes two pulses into one word index, and `EccDecode` then skips the short
frame and leaves it out of its report entirely. Charging it in full keeps that failure visible.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# The 24082026 capture layout, so the simulated curve overlays the measured 8/4/2 points.
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
        "--seeds", type=int, default=10,
        help="Independent noise realizations per sync_num value. The seed drives both the "
             "payload and the noise, because StageRunner re-seeds every stage.",
    )
    parser.add_argument(
        "--sigma", type=float, default=DEFAULT_SIGMA,
        help=f"Per-pulse timing jitter, in slots. The default {DEFAULT_SIGMA} is the measured "
             "Hardware Test 2 value.",
    )
    parser.add_argument(
        "--frames", type=int, default=200,
        help="Frames per run, the same for every sync_num. This sets the resolution of the "
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


def _frame_errors(sent: np.ndarray, received: np.ndarray, n: int) -> tuple[int, int, int]:
    """Word errors, bit errors and first bad word index of one frame, against what was sent.

    A frame of the wrong length is a total loss: `Syncer` collapsed two pulses into one word
    index, or lost one, so the words no longer line up with what was sent and no per-word
    comparison is meaningful. Charge every word and every bit.

    The first bad index matters here. A scale error is progressive, so the errors it makes are
    not sprinkled through the frame. They start at the word where the accumulated error passes
    half a slot, and they continue to the end of the frame.
    """
    if len(received) != n:
        return n, n * PPM_RANK, 0
    diff = np.asarray(received, dtype=np.int64) ^ np.asarray(sent, dtype=np.int64)
    bad = np.nonzero(diff)[0]
    if len(bad) == 0:
        return 0, 0, -1
    bit_errors = int(sum(int(d).bit_count() for d in diff[bad]))
    return len(bad), bit_errors, int(bad[0])


def _run_point(sync_num: int, seed: int, args: argparse.Namespace) -> dict:
    """One Model1 run. Returns the five rates, compared against the words that were sent."""
    mod_params = ModulationParams(
        ppm_rank=PPM_RANK, slot_time=np.float64(SLOT_TIME_S), dead_slots=DEAD_SLOTS
    )
    frame_params = FrameParams(
        sync_num=sync_num, metadata_num=METADATA_NUM, data_num=DATA_NUM,
        ecc_num=ECC_NUM, eof_num=args.eof_num,
    )

    # Ask for the frame count directly. Deriving it from a pass duration would make it depend on
    # sync_num, because a larger sync_num makes the frame longer, and it would also cap the count
    # at whatever the orbit geometry allows. Doppler is off here, so the orbit changes nothing.
    passage_params = PassageParams(
        altitude_km=args.altitude_km, max_elevation_deg=90.0, n_frames=args.frames
    )
    channel_params = ChannelParams(
        sigma=args.sigma, vanish_rate=None, max_const_offset=None, added_rate=None,
        doppler=False,
    )

    model = Model1(
        data=None, frame_params=frame_params, mod_params=mod_params,
        overflight_params=passage_params, channel_params=channel_params,
        plot=False, seed=seed, split_eof_num=args.split_eof_num,
    )
    model.construct_pipeline()

    if model.n_frames != args.frames:
        logger.warning(
            "sync_num=%d seed=%d: got %d frames, asked for %d — the pass window was capped",
            sync_num, seed, model.n_frames, args.frames,
        )

    # PassageGen fixes n_frames at construction, so size the payload only now.
    rng = np.random.default_rng(seed)
    total_words = model.n_frames * DATA_NUM
    model.data = rng.integers(0, 1 << PPM_RANK, total_words, dtype=np.uint16)
    model.run(save_artifacts=False)

    # The sent codeword, without the sync words. Syncer already strips those on the received side.
    frames_sent = model._gen._frame_gen.frames_sent[:, sync_num:]
    pre = model._synced.frames
    post = model._corrected.frames
    n = METADATA_NUM + DATA_NUM + ECC_NUM
    n_frames = len(frames_sent)

    if len(pre) != n_frames or len(post) != n_frames:
        # Only offset noise is applied, so no frame should appear or disappear. If one does, the
        # per-frame comparison below would compare mismatched frames and report nonsense.
        raise RuntimeError(
            f"sync_num={sync_num} seed={seed}: sent {n_frames} frames but received "
            f"{len(pre)} pre-ECC and {len(post)} post-ECC — frame alignment is broken"
        )

    pre_words = pre_bits = post_words = post_bits = failed_frames = 0
    dirty_frames = 0
    within_ecc_limit = 0
    burst_lengths: list[int] = []
    correctable = ECC_NUM // 2
    for sent, got_pre, got_post in zip(frames_sent, pre, post, strict=True):
        w, b, first_bad = _frame_errors(sent, got_pre, n)
        pre_words += w
        pre_bits += b
        if w:
            dirty_frames += 1
            burst_lengths.append(w)
            # RS(221,205) corrects up to 8 symbols. Count how often a damaged frame is inside
            # that limit, because that is the only case where the parity can do anything.
            if w <= correctable:
                within_ecc_limit += 1
        w, b, _ = _frame_errors(sent, got_post, n)
        post_words += w
        post_bits += b
        failed_frames += 1 if w else 0

    total_words_cmp = n_frames * n
    total_bits_cmp = total_words_cmp * PPM_RANK

    # The same quantity a real capture can measure: raw words against the RS-corrected codeword,
    # over the frames RS decoded. A capture has no reference copy, so this is the only true word
    # and bit rate it can report. Recording it here lets the measured curve be compared against a
    # simulated curve of the same definition and the same population.
    ecc_rates = {} if model.ecc_report is None else model.ecc_report.rates()

    return {
        "sync_num": sync_num,
        "seed": seed,
        "frames": n_frames,
        "ecc_frames_reported": 0 if model.ecc_report is None else model.ecc_report.n_frames,
        "ecc_n_decoded": 0 if model.ecc_report is None else model.ecc_report.n_decoded,
        "ecc_wer_pre": ecc_rates.get("wer_pre", 0.0),
        "ecc_ber_pre": ecc_rates.get("ber_pre", 0.0),
        "wer_pre": pre_words / total_words_cmp,
        "ber_pre": pre_bits / total_bits_cmp,
        "wer_post": post_words / total_words_cmp,
        "ber_post": post_bits / total_bits_cmp,
        "fer_post": failed_frames / n_frames,
        "damaged_frames": dirty_frames,
        "damaged_within_ecc_limit": within_ecc_limit,
        "mean_bad_words_when_damaged": float(np.mean(burst_lengths)) if burst_lengths else 0.0,
    }


def _predicted_knee(sigma: float, n_words: int) -> float | None:
    """The sync_num where the fitted scale error stops accumulating past the 0.5 slot boundary.

    The per-frame least-squares scale fit has a slope error of `sigma / sqrt(Sxx)`, with
    `Sxx = n(n^2 - 1) / 12`. That error accumulates over the words of the frame. Returns the
    smallest sync_num that keeps the accumulated error under half a slot, or None if even 64
    pulses are not enough.
    """
    for n in range(3, 65):
        sxx = n * (n * n - 1) / 12.0
        if n_words * sigma / math.sqrt(sxx) < 0.5:
            return n
    return None


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    sync_values = [int(v) for v in args.sync_values.split(",") if v.strip()]
    if not sync_values:
        raise SystemExit("--sync-values is empty")
    if min(sync_values) < 2:
        raise SystemExit("sync_num must be at least 2 — Syncer needs one sync-to-sync gap")

    rows: list[dict] = []
    for sync_num in sync_values:
        _check_split_window(sync_num, args.eof_num, args.split_eof_num)
        for seed in range(args.seeds):
            row = _run_point(sync_num, seed, args)
            rows.append(row)
            logger.info(
                "sync_num=%2d seed=%2d: wer_post=%.3e ber_post=%.3e fer_post=%.3f "
                "(ECC reported %d/%d frames)",
                sync_num, seed, row["wer_post"], row["ber_post"], row["fer_post"],
                row["ecc_frames_reported"], row["frames"],
            )

    keys = (
        "wer_pre", "ber_pre", "wer_post", "ber_post", "fer_post",
        "ecc_wer_pre", "ecc_ber_pre",
    )
    summary = []
    for sync_num in sync_values:
        group = [r for r in rows if r["sync_num"] == sync_num]
        entry: dict = {"sync_num": sync_num, "seeds": len(group), "frames": group[0]["frames"]}
        for key in keys:
            values = [r[key] for r in group]
            entry[key] = float(np.mean(values))
            entry[f"{key}_min"] = float(np.min(values))
            entry[f"{key}_max"] = float(np.max(values))
        damaged = sum(r["damaged_frames"] for r in group)
        entry["damaged_frames"] = damaged
        entry["damaged_within_ecc_limit"] = sum(r["damaged_within_ecc_limit"] for r in group)
        bad = [r["mean_bad_words_when_damaged"] for r in group if r["damaged_frames"]]
        entry["mean_bad_words_when_damaged"] = float(np.mean(bad)) if bad else 0.0
        summary.append(entry)
        print(
            f"sync_num={sync_num:2d}  wer_post={entry['wer_post']:.3e}  "
            f"ber_post={entry['ber_post']:.3e}  fer_post={entry['fer_post']:.3f}  "
            f"damaged={damaged:3d} (ECC could fix {entry['damaged_within_ecc_limit']:3d}, "
            f"mean {entry['mean_bad_words_when_damaged']:.0f} bad words)"
        )

    x = np.array(sync_values, dtype=float)
    # A rate of zero has no place on a log axis. Put it on the floor of the drawn range instead,
    # and mark those points as open circles so a true zero never reads as a small positive rate.
    positive = [e[k] for e in summary for k in ("wer_post", "ber_post") if e[k] > 0]
    floor = min(positive) / 10.0 if positive else 1e-6

    fig, (ax_wer, ax_ber) = plt.subplots(1, 2, figsize=(12, 5.5))
    n_words = METADATA_NUM + DATA_NUM + ECC_NUM
    knee = _predicted_knee(args.sigma, n_words)

    for ax, post_key, ylabel in (
        (ax_wer, "wer_post", "Post-ECC word error rate (symbols)"),
        (ax_ber, "ber_post", "Post-ECC bit error rate"),
    ):
        mean = np.array([e[post_key] for e in summary])
        lo = np.array([e[f"{post_key}_min"] for e in summary])
        hi = np.array([e[f"{post_key}_max"] for e in summary])
        drawn = np.where(mean > 0, mean, floor)
        ax.plot(x, drawn, "-", color="tab:red", marker="o", label="post-ECC",
                linewidth=1.8, markerfacecolor="tab:red", markersize=6)
        zero = mean == 0
        if zero.any():
            ax.plot(x[zero], np.full(int(zero.sum()), floor), "o", color="tab:red",
                    markerfacecolor="white", markersize=8, zorder=5, label="0 errors (exact)")
        ax.fill_between(x, np.where(lo > 0, lo, floor), np.where(hi > 0, hi, floor),
                        color="tab:red", alpha=0.15, label=f"spread over {args.seeds} seeds")

        ax.set_xscale("log", base=2)
        ax.set_xticks(sync_values)
        ax.set_xticklabels([str(v) for v in sync_values])
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / 2)
        ax.set_xlabel("Sync pulses per frame (sync_num)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    default_title = (
        f"Simulated WER/BER against sync pulse count "
        f"(sigma={args.sigma:g} slot, {args.frames} frames, {args.seeds} seeds)"
    )
    fig.suptitle(figure_title(args, default_title))
    fig.tight_layout()

    out = output_dir()
    fig.savefig(out / "ber_vs_sync_num.png", dpi=150)
    logger.info("Saved %s", out / "ber_vs_sync_num.png")

    payload = {
        "sigma": args.sigma,
        "frames": args.frames,
        "seeds": args.seeds,
        "eof_num": args.eof_num,
        "split_eof_num": args.split_eof_num,
        "layout": {
            "ppm_rank": PPM_RANK, "dead_slots": DEAD_SLOTS, "slot_time_s": SLOT_TIME_S,
            "metadata_num": METADATA_NUM, "data_num": DATA_NUM, "ecc_num": ECC_NUM,
        },
        "predicted_knee_sync_num": knee,
        "summary": summary,
        "runs": rows,
    }
    json_path = out / "ber_vs_sync_num.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %s", json_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
