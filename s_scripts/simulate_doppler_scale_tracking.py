"""Show how Syncer's per-frame scale fit tracks the Doppler-induced slot-clock drift.

    python s_scripts/simulate_doppler_scale_tracking.py --no-show

Compares two sync-stage variants on the identical channel and data:

- `SimpleSyncer` fits one flat scale per frame from that frame's own sync pulses. Per
  docs/math.md section 7 ("Tolerance to Doppler shift"), this already absorbs a *constant*
  Doppler rate for free — a constant rate is exactly a scale error — but leaves an open,
  undetermined residual from the *curvature* (second derivative) of the Doppler profile, because
  a flat, single-frame fit cannot follow a rate that is itself changing.
- `TwoPointSync` fits the same frame_start/scale per frame, then also fits the one line through
  the previous and current frame's own fitted points, and applies that slope as a
  piecewise-linear scale correction across the current frame — directly targeting the curvature
  term `SimpleSyncer` leaves behind.

The channel carries only two things: Doppler shift, and a per-frame ConstantOffset. No jitter
(RandomShift), no VanishPulses, no AddedPulses. docs/math.md section 3 states that ConstantOffset
models "an unknown time of arrival", and Syncer's least-squares scale fit absorbs a constant term
exactly — it is the standard no-op impairment this codebase uses to exercise the fit without
introducing anything the fit cannot already remove. What is left over, after that removal, is the
curvature-only tracking error this script measures.

DopplerShift turns a pulse's raw offset (slots since pass start) into a time relative to closest
approach, computes the slant range, and converts the extra propagation delay back into slots
(docs/math.md section 3):

    t = (x - x_tca) * T_c
    rho(t) = sqrt(h^2 + (v*t)^2)
    rho_dot(t) = v^2 * t / rho(t)          # range rate
    frac_error(t) = rho_dot(t) / c         # fractional slot-clock error this induces

Both syncers fit a straight line per frame through that frame's sync pulses (their shared
`_acquire`/`_locate_sync`/`_refine_scale` machinery) and report the fitted scale in
`slot_scales`. `slot_scales[i] - 1` is each syncer's own estimate of frac_error at frame i — the
empirical counterpart to the analytic curve above.

A straight line cannot follow a curve exactly. Differentiating frac_error(t) gives the curvature
term the fit leaves behind:

    rho_ddot(t) = v^2 * h^2 / rho(t)^3
    eps_res(t) ~= rho_ddot(t) * T_frame / (2*c)

This is exactly the open bound docs/math.md section 7 ("Tolerance to Doppler shift") names but
never derives a number for. This script measures it directly: true frac_error, each syncer's
recovered frac_error, their difference, and the analytic eps_res prediction, all against
frame/time across one full pass.

Frame timing is deterministic (PassageGen.py): frame i starts at slot offset i * frame_duration in
the unperturbed layout. That nominal offset, not either syncer's own frame_starts (which carries
the per-frame ConstantOffset draw), is what this script uses for the x-axis and for evaluating the
analytic curve — the constant offset is there to exercise robustness, not something to plot.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization._logging import setup_logging
from coding_synchronization.channel.Channel import ChannelParams
from coding_synchronization.channel.DopplerShift import C, GM, R_EARTH
from coding_synchronization.decoder.SimpleSyncer import SimpleSyncer
from coding_synchronization.decoder.TwoPointSync import TwoPointSync
from coding_synchronization.encoder import FrameParams, ModulationParams
from coding_synchronization.encoder.PassageGen import PassageGen, PassageParams
from coding_synchronization.measurement.Cli import (
    add_title_arg,
    add_verbose_arg,
    figure_title,
    log_level,
    output_dir,
)
from coding_synchronization.Model import Model1

logger = logging.getLogger(f"coding_synchronization.{Path(__file__).stem}")

# The 24082026 capture layout.
PPM_RANK = 10
DEAD_SLOTS = 16
SLOT_TIME_S = 1.666632354617692e-05
METADATA_NUM = 5
DATA_NUM = 200
ECC_NUM = 16

VARIANTS = {"SimpleSyncer": SimpleSyncer, "TwoPointSync": TwoPointSync}
COLORS = {"SimpleSyncer": "tab:red", "TwoPointSync": "tab:orange"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--altitude-km", type=float, default=500.0, help="Satellite altitude.",
    )
    parser.add_argument(
        "--max-elevation-deg", type=float, default=90.0,
        help="Peak elevation of the pass. 90 (zenith) is the worst case for Doppler curvature — "
             "see docs/math.md section 3.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=None,
        help="Override the frame count instead of letting the orbital geometry decide the full "
             "pass length.",
    )
    parser.add_argument(
        "--sync-num", type=int, default=8, help="Sync pulses per frame.",
    )
    parser.add_argument(
        "--max-const-offset", type=float, default=20.0,
        help="Per-frame ConstantOffset upper bound, in slots. Small compared to the sync margin "
             "(word_period // 8) — the fit absorbs it regardless of size, so this only has to "
             "stay inside Syncer's pulse-location margin.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    add_title_arg(parser)
    add_verbose_arg(parser)
    return parser.parse_args()


def _run_variant(
    syncer_cls, mod_params, frame_params, passage_params, channel_params, seed, ppm_rank
) -> tuple[int, list, list | None]:
    """Run one Model1 with the given syncer class.

    Returns (n_decoded, slot_scales, slopes). `slopes` is `TwoPointSync.slopes` (one two-point
    slope per decoded frame) when the syncer has it, `None` for `SimpleSyncer`, which has no
    such thing — its scale is flat across the whole frame.
    """
    model = Model1(
        data=None, frame_params=frame_params, mod_params=mod_params,
        overflight_params=passage_params, channel_params=channel_params,
        plot=False, seed=seed, run_ecc=False, syncer_cls=syncer_cls,
    )
    model.construct_pipeline()

    rng = np.random.default_rng(seed)
    total_words = model.n_frames * DATA_NUM
    model.data = rng.integers(0, 1 << ppm_rank, total_words, dtype=np.uint16)
    model.run(save_artifacts=False)

    syncer = model._syncer
    n_decoded = len(syncer.slot_scales)
    if n_decoded != model.n_frames:
        logger.warning(
            "%s: %d/%d frames decoded — %d frame(s) failed synchronization",
            syncer_cls.__name__, n_decoded, model.n_frames, model.n_frames - n_decoded,
        )
    slopes = list(syncer.slopes) if hasattr(syncer, "slopes") else None
    return n_decoded, list(syncer.slot_scales), slopes


def main() -> None:
    args = _parse_args()
    setup_logging(level=log_level(args))

    mod_params = ModulationParams(
        ppm_rank=PPM_RANK, slot_time=np.float64(SLOT_TIME_S), dead_slots=DEAD_SLOTS
    )
    frame_params = FrameParams(
        sync_num=args.sync_num, metadata_num=METADATA_NUM, data_num=DATA_NUM,
        ecc_num=ECC_NUM, eof_num=64,
    )
    passage_params = PassageParams(
        altitude_km=args.altitude_km, max_elevation_deg=args.max_elevation_deg,
        n_frames=args.n_frames,
    )

    # A throwaway PassageGen, just to learn n_frames from the same orbital-elevation math the
    # real run will use — this avoids re-deriving that geometry here.
    scratch_gen = PassageGen(frame_params, mod_params, passage_params, seed=args.seed)
    n_frames = scratch_gen.n_frames
    word_period = (1 << PPM_RANK) + DEAD_SLOTS
    frame_len = frame_params.sync_num + METADATA_NUM + DATA_NUM + ECC_NUM
    frame_duration_slots = (frame_len + frame_params.eof_num) * word_period
    total_slots = n_frames * frame_duration_slots
    tca_slots = total_slots / 2.0

    channel_params = ChannelParams(
        sigma=None, vanish_rate=None, added_rate=None,
        max_const_offset=args.max_const_offset,
        chirp_duration_s=float(mod_params.slot_time), tca_chirp=tca_slots, doppler=True,
    )

    results = {}
    for name, syncer_cls in VARIANTS.items():
        n_decoded, slot_scales, slopes = _run_variant(
            syncer_cls, mod_params, frame_params, passage_params, channel_params,
            args.seed, PPM_RANK,
        )
        results[name] = (n_decoded, slot_scales, slopes)

    n_decoded = min(n for n, _, _ in results.values())

    # Ground truth, evaluated at each decoded frame's nominal (unperturbed) start offset —
    # the per-frame ConstantOffset draw is deliberately not used here, see module docstring.
    velocity_m_s = np.sqrt(GM / (R_EARTH + args.altitude_km * 1e3))
    altitude_m = args.altitude_km * 1e3
    frame_idx = np.arange(n_decoded, dtype=np.float64)
    x_slots = frame_idx * frame_duration_slots
    t_s = (x_slots - tca_slots) * SLOT_TIME_S

    rho = np.sqrt(altitude_m**2 + (velocity_m_s * t_s) ** 2)
    rho_dot = velocity_m_s**2 * t_s / rho
    true_frac_error = rho_dot / C
    rho_ddot = velocity_m_s**2 * altitude_m**2 / rho**3
    frame_duration_s = frame_duration_slots * SLOT_TIME_S
    predicted_residual = rho_ddot * frame_duration_s / (2 * C)

    recovered = {
        name: np.array(slot_scales[:n_decoded]) - 1.0
        for name, (_, slot_scales, _) in results.items()
    }
    # The two-point slope applied within each frame (dimensionless per slot), only present for
    # TwoPointSync — SimpleSyncer's fit is flat, so it has no such quantity.
    slopes = {
        name: (np.array(slope_list[:n_decoded]) if slope_list is not None else None)
        for name, (_, _, slope_list) in results.items()
    }
    residual = {name: true_frac_error - rec for name, rec in recovered.items()}

    # Dense, oversampled curve for the smooth ground-truth trace.
    t_dense = np.linspace(t_s.min(), t_s.max(), 2000) if n_decoded > 1 else t_s
    rho_dense = np.sqrt(altitude_m**2 + (velocity_m_s * t_dense) ** 2)
    true_frac_error_dense = velocity_m_s**2 * t_dense / rho_dense / C

    # Which frame each dense sample falls in, and how far across that frame (0 at frame_start,
    # 1 at the next frame_start) — needed to evaluate each syncer's own piecewise approximation
    # (flat for SimpleSyncer, piecewise-linear for TwoPointSync) at the same resolution as the
    # true curve, rather than only at each frame's own single sample point.
    frame_of_dense = np.clip(
        np.floor((t_dense - t_s[0]) / frame_duration_s).astype(int), 0, n_decoded - 1
    )
    pos_in_slots_dense = (t_dense - t_s[frame_of_dense]) / frame_duration_s * frame_duration_slots

    # The curvature-only error against what each syncer actually applied, sampled continuously
    # across every frame — not just at each frame's own anchor point — so the jagged edges where
    # a flat (or piecewise-linear) approximation meets the true curve show up: largest where the
    # curve bends the most within a frame, resetting at each frame boundary.
    residual_dense = {}
    for name in VARIANTS:
        rec = recovered[name]
        slope = slopes[name]
        syncer_value_dense = rec[frame_of_dense]
        if slope is not None:
            syncer_value_dense = syncer_value_dense + slope[frame_of_dense] * pos_in_slots_dense
        arr = true_frac_error_dense - syncer_value_dense
        # Each frame's own fit starts fresh, so the residual genuinely jumps at the frame
        # boundary — but that jump is a reset, not something happening over time, so a line
        # drawn straight across it would show a vertical edge that never actually occurred.
        # Breaking the line there leaves the one-sided ramp within each frame on its own.
        boundary = np.flatnonzero(np.diff(frame_of_dense) != 0) + 1
        arr[boundary] = np.nan
        residual_dense[name] = arr

    # A fractional scale error only means something physically once it's scaled by how many
    # slots it acts over. Multiplying by frame_duration_slots turns each dimensionless error into
    # the timing drift it actually accumulates across one frame, in slots — fractional, since the
    # scale error itself is a continuous quantity.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(t_dense, true_frac_error_dense * frame_duration_slots, color="tab:blue",
             linewidth=1.5, label="True Doppler drift per frame")
    for name in VARIANTS:
        rec = recovered[name]
        slope = slopes[name]
        color = COLORS[name]
        # TwoPointSync is dashed in both panels, so the two variants stay visually distinct even
        # where their curves nearly overlap.
        linestyle = "-"
        for i in range(n_decoded):
            t0 = t_s[i]
            t1 = t0 + frame_duration_s
            drift0 = rec[i] * frame_duration_slots
            if slope is not None:
                # TwoPointSync applies a piecewise-linear scale across the frame (see
                # docs/math.md section 7, "Two-point tracking"): the fitted value rises or falls
                # across the frame instead of staying flat, so the segment must be a sloped line,
                # not a horizontal one, or the plot misrepresents what the syncer actually did.
                drift1 = (rec[i] + slope[i] * frame_duration_slots) * frame_duration_slots
            else:
                # SimpleSyncer fits one flat scale per frame, so its estimate is constant across
                # the frame's own span — a horizontal segment, not a rising line. The staircase
                # these segments form is the piecewise-constant approximation it computed for
                # the smooth true curve.
                drift1 = drift0
            ax1.plot(
                [t0, t1], [drift0, drift1],
                color=color, linewidth=1.2, linestyle=linestyle,
                label=f"{name}'s per-frame fitted scale" if i == 0 else None,
            )
    ax1.set_ylabel("Timing drift per frame (slots)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Its own y-scale: the residual is orders of magnitude smaller than the true/recovered curves
    # of panel 1, and would be invisible plotted on that same scale.
    for name in VARIANTS:
        linestyle = "-"
        ax2.plot(t_dense, residual_dense[name] * frame_duration_slots, color=COLORS[name],
                 linewidth=1.2, linestyle=linestyle, label=f"{name} residual")
    ax2.set_xlabel("Time since closest approach (s)")
    ax2.set_ylabel("Curvature-only tracking error (slots per frame)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    default_title = (
        f"SimpleSyncer vs. TwoPointSync tracking Doppler drift (altitude={args.altitude_km:g} km, "
        f"sync_num={args.sync_num}, {n_decoded} frames)"
    )
    fig.suptitle(figure_title(args, default_title))
    fig.tight_layout()

    out = output_dir()
    fig.savefig(out / "doppler_scale_tracking.png", dpi=150)
    logger.info("Saved %s", out / "doppler_scale_tracking.png")

    payload = {
        "altitude_km": args.altitude_km,
        "max_elevation_deg": args.max_elevation_deg,
        "sync_num": args.sync_num,
        "max_const_offset": args.max_const_offset,
        "n_decoded": n_decoded,
        "frame_duration_s": frame_duration_s,
        "frame_duration_slots": frame_duration_slots,
        "variants": {
            name: {
                "n_decoded": results[name][0],
                "frames": [
                    {
                        "frame": i,
                        "t_s": float(t_s[i]),
                        "true_frac_error": float(true_frac_error[i]),
                        "recovered_frac_error": float(recovered[name][i]),
                        "slope": None if slopes[name] is None else float(slopes[name][i]),
                        "residual": float(residual[name][i]),
                        "predicted_residual_bound": float(predicted_residual[i]),
                    }
                    for i in range(n_decoded)
                ],
            }
            for name in VARIANTS
        },
    }
    json_path = out / "doppler_scale_tracking.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %s", json_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
