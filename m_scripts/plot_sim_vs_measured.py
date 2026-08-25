"""Plot the simulated error rates against the measured ones, over sync pulse count.

    python m_scripts/plot_sim_vs_measured.py --no-show
    python m_scripts/plot_sim_vs_measured.py output/<ts>/ber_vs_sync_num.json --reports out/a/ecc_report.xml

The simulated side comes from `simulate_ber_vs_sync_num.py`, which runs Model1. The measured side
comes from `ecc_report.xml`, written by `decode_measurement.py --check-ecc` over a real capture.

The three panels are the word error rate, the bit error rate, and the lost frames. Every one of
them counts over **all** frames sent, which is what makes the two sides comparable and makes each
curve fall as sync_num rises.

**Why all frames, and not the decoded ones.** An earlier version of this script rated only the
frames Reed-Solomon decoded. That population is selected by the very thing being measured: as
sync_num rises, marginal frames start surviving into the denominator and bring their errors with
them, so the rate can climb while the link improves. That is survivorship, not a channel
measurement, and it is why the measured curve appeared to run backwards.

**What a real capture can measure.** It holds no reference copy of the payload, so it cannot count
errors by comparing against what was sent. For a frame RS decodes, though, the corrected codeword
is an exact reference, and every symbol RS changed is a symbol that arrived wrong. Each `<frame>`
element records that as `symbol_errors` and `bit_errors`, and this script sums them from the
per-frame elements rather than the summary attributes, so a report written by an older version of
`EccReport.rates()` still reads correctly.

For a frame RS cannot decode there is no reference, so its error count is bounded rather than
known: at least `correctable + 1` symbols, or RS would have decoded it, and at most the whole
codeword. The measured word and bit rates are therefore drawn as a band between those bounds, and
the true value lies inside it. The simulated rate is a single exact line, because Model1 knows
every word it sent.

The simulation applies Gaussian timing jitter and nothing else. A real capture also carries
dropouts, noise bursts and threshold effects. Expect the simulation to be optimistic, and read the
gap as the part of the measured failure that timing jitter alone does not explain.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization.measurement.Cli import add_title_arg, figure_title, output_dir
from plot_ber_wer_vs_sync_num import _sync_num_from_run_log

DEFAULT_REPORTS = [
    Path("output/2026-08-25_12-16-45/ecc_report.xml"),   # sync_num 8
    Path("output/2026-08-25_12-16-51/ecc_report.xml"),   # sync_num 4
    Path("output/2026-08-25_12-26-17/ecc_report.xml"),   # sync_num 2
]

# (simulated key, measured key, axis label, panel title). Every rate here is counted over ALL
# frames, so all three fall as sync_num rises and the two sides are directly comparable.
PANELS = (
    ("wer_post", "wer", "Word error rate (symbols)", "Word error rate"),
    ("ber_post", "ber", "Bit error rate", "Bit error rate"),
    ("fer_post", "fer", "Lost frames (fraction of all frames)", "Lost frames"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "simulation", type=Path, nargs="?",
        default=Path("output/2026-08-25_13-25-10/ber_vs_sync_num.json"),
        help="ber_vs_sync_num.json, written by simulate_ber_vs_sync_num.py.",
    )
    parser.add_argument(
        "--reports", type=Path, nargs="+", default=DEFAULT_REPORTS,
        help="ecc_report.xml files from real captures. The sync count of each is read from its "
             "sibling run.log.",
    )
    add_title_arg(parser)
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    return parser.parse_args()


def _read_measured(paths: list[Path]) -> list[dict]:
    """Measured word and bit rates over ALL frames, as a bounded range, plus the frame error rate.

    Counting over all frames is what makes the rate comparable with the simulation and monotonic
    in sync_num. Counting over the decoded frames only does neither: that population is selected
    by the very thing being measured, so as sync_num rises, marginal frames start surviving into
    the denominator and bring their errors with them. The rate can then climb while the link
    improves, which is survivorship, not a channel measurement.

    A frame RS decoded has an exact count, because the corrected codeword is a reference, and the
    per-frame elements carry it. A frame RS could not decode has no reference, so its count is
    bounded rather than known:

    - at least `correctable + 1` symbols, or RS would have decoded it;
    - at most `n` symbols, the whole codeword.

    Each bad symbol differs in at least one bit and at most `ppm_rank` bits, which bounds the bit
    count the same way. The result is a band, and the true value lies inside it.

    The summary attributes on the root element are not used, so a report written before
    `EccReport.rates()` was corrected still reads correctly.
    """
    rows = []
    for path in paths:
        root = ET.parse(path).getroot()
        n = int(root.get("n"))
        ppm_rank = int(root.get("ppm_rank"))
        correctable = int(root.get("correctable"))
        frames = root.findall("frame")
        decoded = [f for f in frames if f.get("ok") == "true"]
        n_decoded = len(decoded)
        n_lost = len(frames) - n_decoded
        symbol_errors = sum(int(f.get("symbol_errors")) for f in decoded)
        bit_errors = sum(int(f.get("bit_errors")) for f in decoded)

        total_words = len(frames) * n
        total_bits = total_words * ppm_rank
        rows.append({
            "sync_num": _sync_num_from_run_log(path),
            "frames": len(frames),
            "n_decoded": n_decoded,
            "uncorrectable": n_lost,
            "symbol_errors_decoded": symbol_errors,
            "bit_errors_decoded": bit_errors,
            "wer_lo": (symbol_errors + n_lost * (correctable + 1)) / total_words,
            "wer_hi": (symbol_errors + n_lost * n) / total_words,
            "ber_lo": (bit_errors + n_lost * (correctable + 1)) / total_bits,
            "ber_hi": (bit_errors + n_lost * n * ppm_rank) / total_bits,
            "fer": n_lost / len(frames) if frames else 0.0,
            "source": str(path),
        })
    rows.sort(key=lambda r: r["sync_num"])
    return rows


def main() -> None:
    args = _parse_args()

    sim = json.loads(args.simulation.read_text())["summary"]
    sim.sort(key=lambda e: e["sync_num"])
    if "wer_post" not in sim[0]:
        raise SystemExit(
            f"{args.simulation} has no wer_post — it does not look like a "
            f"simulate_ber_vs_sync_num.py output."
        )
    measured = _read_measured(list(args.reports))

    sim_x = np.array([e["sync_num"] for e in sim], dtype=float)
    mea_x = np.array([r["sync_num"] for r in measured], dtype=float)

    # A rate of zero has no place on a log axis. Draw it on the floor as an open marker, so an
    # exact zero never reads as a small positive rate.
    measured_values = [
        r[key] for r in measured
        for key in ("wer_lo", "wer_hi", "ber_lo", "ber_hi", "fer") if r[key] > 0
    ]
    positive = [e[k] for e in sim for k, _, _, _ in PANELS if e[k] > 0] + measured_values
    floor = min(positive) / 10.0 if positive else 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (sim_key, mea_key, ylabel, panel_title) in zip(axes, PANELS, strict=True):
        mean = np.array([e[sim_key] for e in sim])
        lo = np.array([e[f"{sim_key}_min"] for e in sim])
        hi = np.array([e[f"{sim_key}_max"] for e in sim])
        ax.plot(sim_x, np.where(mean > 0, mean, floor), "-", color="tab:red", marker="o",
                markersize=6, linewidth=1.8, label="simulated (jitter only)")
        zero = mean == 0
        if zero.any():
            ax.plot(sim_x[zero], np.full(int(zero.sum()), floor), "o", color="tab:red",
                    markerfacecolor="white", markersize=8, zorder=5, label="simulated = 0 (exact)")
        ax.fill_between(sim_x, np.where(lo > 0, lo, floor), np.where(hi > 0, hi, floor),
                        color="tab:red", alpha=0.15, label="spread over seeds")

        if mea_key == "fer":
            mea_y = np.array([r["fer"] for r in measured])
            ax.plot(mea_x, np.where(mea_y > 0, mea_y, floor), "--", color="tab:blue", marker="s",
                    markersize=8, linewidth=1.8, label="measured (real capture)")
            # The fraction alone hides how few frames each capture holds. Show the raw counts.
            for xi, r in zip(mea_x, measured, strict=True):
                ax.annotate(f"{r['uncorrectable']}/{r['frames']}",
                            (xi, max(r["fer"], floor)), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color="tab:blue")
            for xi, e in zip(sim_x, sim, strict=True):
                lost = int(round(e["fer_post"] * e["frames"] * e["seeds"]))
                ax.annotate(f"{lost}/{e['frames'] * e['seeds']}",
                            (xi, max(e["fer_post"], floor)), textcoords="offset points",
                            xytext=(6, -12), fontsize=8, color="tab:red")
        else:
            # A frame RS could not decode has no reference, so its error count is bounded, not
            # known. Draw the band those bounds allow; the true measured rate is inside it.
            lo_m = np.array([r[f"{mea_key}_lo"] for r in measured])
            hi_m = np.array([r[f"{mea_key}_hi"] for r in measured])
            ax.fill_between(mea_x, np.where(lo_m > 0, lo_m, floor), np.where(hi_m > 0, hi_m, floor),
                            color="tab:blue", alpha=0.18, label="measured (bounded range)")
            for arr, style in ((hi_m, "--"), (lo_m, ":")):
                ax.plot(mea_x, np.where(arr > 0, arr, floor), style, color="tab:blue",
                        marker="s", markersize=6, linewidth=1.4)
            ax.plot([], [], "--s", color="tab:blue", markersize=6,
                    label="measured upper bound (frame lost whole)")
            ax.plot([], [], ":s", color="tab:blue", markersize=6,
                    label="measured lower bound (RS limit + 1)")

        ax.set_xscale("log", base=2)
        ticks = sorted({int(v) for v in list(sim_x) + list(mea_x)})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / 2)
        ax.set_xlabel("Sync pulses per frame (sync_num)")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    default_title = "Simulated against measured error rates, over sync pulse count"
    fig.suptitle(figure_title(args, default_title))
    fig.tight_layout()

    sim_by_sync = {e["sync_num"]: e for e in sim}
    header = (f"{'sync':>5}  {'quantity':>5}  {'measured range':>25}  {'simulated':>11}  "
              f"{'verdict':>16}")
    print(header)
    print("-" * len(header))
    comparison = []
    for r in measured:
        e = sim_by_sync.get(r["sync_num"])
        if e is None:
            continue
        entry = {
            "sync_num": r["sync_num"],
            "measured_frames": r["frames"],
            "measured_decoded": r["n_decoded"],
            "measured_uncorrectable": r["uncorrectable"],
        }
        for sim_key, mea_key, _, _ in PANELS:
            s = e[sim_key]
            if mea_key == "fer":
                lo = hi = r["fer"]
                shown = f"{lo:>25.3e}"
            else:
                lo, hi = r[f"{mea_key}_lo"], r[f"{mea_key}_hi"]
                shown = f"{lo:>11.3e} .. {hi:<11.3e}"
            if lo <= s <= hi:
                verdict = "sim inside"
            elif s < lo:
                verdict = f"sim {lo / s:.0f}x low" if s > 0 else "sim = 0"
            else:
                verdict = f"sim {s / hi:.0f}x high"
            print(f"{r['sync_num']:>5}  {mea_key:>5}  {shown}  {s:>11.3e}  {verdict:>16}")
            entry[f"measured_{mea_key}_lo"] = lo
            entry[f"measured_{mea_key}_hi"] = hi
            entry[f"simulated_{mea_key}"] = s
        comparison.append(entry)

    out = output_dir()
    fig.savefig(out / "sim_vs_measured.png", dpi=150)
    print(f"\nSaved {out / 'sim_vs_measured.png'}")

    payload = {
        "simulation_source": str(args.simulation),
        "measured_sources": [r["source"] for r in measured],
        "measured_detail": measured,
        "comparison": comparison,
        "note": (
            "Pre-ECC word and bit rates cover the frames RS decoded, on both sides. For those "
            "frames the RS-corrected codeword is an exact reference. A frame RS could not decode "
            "carries no reference, so its error count is unmeasurable and it appears only in the "
            "frame error rate."
        ),
    }
    (out / "sim_vs_measured.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved {out / 'sim_vs_measured.json'}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
