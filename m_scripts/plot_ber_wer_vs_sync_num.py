"""Plot post-ECC WER/BER against the number of sync pulses used, from saved ecc_report.xml files.

    python m_scripts/plot_ber_wer_vs_sync_num.py output/<ts1>/ecc_report.xml output/<ts2>/ecc_report.xml ...

Each `ecc_report.xml` (written by decode_measurement.py --check-ecc) already carries
wer_post/ber_post/frame_error_rate on the root element; the sync word count it was decoded with
is read from that report's sibling run.log pipeline line. No re-decoding happens here, this only
plots numbers already computed.

Only post-ECC is plotted, and the two panels carry the same curve. An uncorrectable frame gives
nothing usable, so `EccReport.rates()` charges it in full. Charging whole frames makes the symbol
rate and the bit rate scale together, so `wer_post`, `ber_post` and `frame_error_rate` are one
number under three names. Without a reference copy of the payload no finer post-ECC measurement
exists on a real capture.

`wer_pre`/`ber_pre` are not plotted here. They cover the frames RS decoded and leave the
uncorrectable ones out, because those carry no reference and their error count is unmeasurable.
That makes them exact over a smaller population, so they do not share a denominator with the
post-ECC curve and the two cannot be read off one axis.

For a curve where the pre-ECC and post-ECC rates are both true and comparable, use
`simulate_ber_vs_sync_num.py`. It runs Model1, which knows every word it sent.

The x-axis is the number of sync pulses (read from each report's sibling run.log pipeline line),
not the position on the command line, so the default three-file example — captured chronologically
as 8, then 4, then 2 sync pulses — still plots in that same 8/4/2 order because that's decreasing
sync_num, not because of argument order.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

from coding_synchronization.measurement.Cli import add_title_arg, figure_title, output_dir

_SYNC_NUM_RE = re.compile(r"Syncer\(sync_num=(\d+)")


def _sync_num_from_run_log(xml_path: Path) -> int:
    """Read sync_num from the pipeline line in the report's sibling run.log."""
    run_log = xml_path.parent / "run.log"
    text = run_log.read_text()
    m = _SYNC_NUM_RE.search(text)
    if m is None:
        raise ValueError(f"{run_log}: no 'Syncer(sync_num=N' pipeline line found")
    return int(m.group(1))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "reports", type=Path, nargs="*",
        default=[
            Path("output/2026-08-25_15-17-34/ecc_report.xml"),
            Path("output/2026-08-25_15-15-18/ecc_report.xml"),
            Path("output/2026-08-25_15-15-08/ecc_report.xml"),
        ],
        help="ecc_report.xml files to plot. Defaults to the sync_num=8/4/2 sweep from "
             "measurments/testy_24082026/npz/60kHz_{8,4,2}sync200payload_10kOhm_long.npz.",
    )
    add_title_arg(parser)
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figure, and do not open a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rows = []
    for path in args.reports:
        root = ET.parse(path).getroot()
        sync_num = _sync_num_from_run_log(path)
        rows.append({
            "path": str(path),
            "sync_num": sync_num,
            "frames": int(root.get("frames")),
            "uncorrectable": int(root.get("uncorrectable")),
            "wer_post": float(root.get("wer_post")),
            "ber_post": float(root.get("ber_post")),
            "frame_error_rate": float(root.get("frame_error_rate")),
        })
    # Decreasing sync_num, e.g. 8, 4, 2 — matches the chronological capture order for the
    # default files without depending on argv order.
    rows.sort(key=lambda r: r["sync_num"])

    sync_nums = [r["sync_num"] for r in rows]
    x = range(len(rows))

    fig, ax_wer = plt.subplots(1, 1, figsize=(11, 5))

    for ax, post_key, ylabel in (
        (ax_wer, "wer_post", "Post-ECC Frame error rate (symbols)"),
    ):
        ax.semilogy(x, [r[post_key] for r in rows], "o-", color="tab:red")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(n) for n in sync_nums])
        ax.set_xlabel("Sync pulses per frame (sync_num)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    for r, xi in zip(rows, x, strict=True):
        note = f"{r['frames'] - r['uncorrectable']}/{r['frames']} ok"
        ax_wer.annotate(
            note, (xi, r["wer_post"]), textcoords="offset points", xytext=(0, 8),
            fontsize=7, ha="center",
        )

    default_title = "Post-ECC Frame Error Rate vs. sync pulse count"
    fig.suptitle(figure_title(args, default_title))
    fig.tight_layout()

    out = output_dir()
    fig.savefig(out / "ber_wer_vs_sync_num.png", dpi=150)
    print(f"Saved {out / 'ber_wer_vs_sync_num.png'}")

    summary = {"reports": rows}
    json_path = out / "ber_wer_vs_sync_num.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {json_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
