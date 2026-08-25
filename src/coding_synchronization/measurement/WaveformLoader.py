import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from coding_synchronization.physical_units import Quantity

logger = logging.getLogger(__name__)

_SNIFF_LINES = 40
_DELIMITERS: tuple[str | None, ...] = (",", ";", "\t", None)


@dataclass
class WaveformParams:
    path: Path
    sample_rate: Quantity | None = None  # required only if the CSV has no time column
    pos_col: int | None = None  # None -> auto (first of the two value columns)
    neg_col: int | None = None  # None -> auto (second of the two value columns)
    max_samples: int | None = None  # head-truncate for quick looks


@dataclass
class CsvFormat:
    delimiter: str | None
    header_lines: int
    n_cols: int
    time_col: int | None
    value_cols: tuple[int, ...]
    dt_s: float | None
    header_text: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        delim = "whitespace" if self.delimiter is None else repr(self.delimiter)
        dt = "n/a" if self.dt_s is None else f"{self.dt_s * 1e12:.3f} ps"
        return (
            f"CsvFormat(delimiter={delim}, header_lines={self.header_lines}, "
            f"n_cols={self.n_cols}, time_col={self.time_col}, "
            f"value_cols={self.value_cols}, dt={dt})"
        )


@dataclass
class Waveform:
    ch_a: np.ndarray
    ch_b: np.ndarray
    dt_s: float
    fmt: CsvFormat
    source: Path

    @property
    def n(self) -> int:
        return len(self.ch_a)

    @property
    def duration_s(self) -> float:
        return self.n * self.dt_s

    def time_s(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Materialize a time axis for a slice only — never for the full record."""
        stop = self.n if stop is None else stop
        return np.arange(start, stop, dtype=np.float64) * self.dt_s

    def __repr__(self) -> str:
        return f"Waveform(n={self.n}, dt_s={self.dt_s:.6g}, duration_s={self.duration_s:.6g})"


def _split(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return line.split()
    return line.split(delimiter)


def _strip_trailing_empty(fields: list[str]) -> list[str]:
    """Drop trailing empty fields left by a trailing delimiter (e.g. 'value,,' -> ['value'])."""
    while fields and fields[-1].strip() == "":
        fields = fields[:-1]
    return fields


def _all_float(fields: list[str]) -> bool:
    fields = _strip_trailing_empty(fields)
    if not fields:
        return False
    for f in fields:
        try:
            float(f)
        except ValueError:
            return False
    return True


_T_INC_RE = re.compile(r"tInc\s*=\s*([-+0-9.eE]+)")


def _header_dt_s(header_text: list[str]) -> float | None:
    """Sample period from a single-channel scope header line (e.g. 'tInc = 2.000000e-06')."""
    for ln in header_text:
        m = _T_INC_RE.search(ln)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _is_time_axis(col: np.ndarray) -> bool:
    """Monotonically increasing with a near-constant step."""
    if len(col) < 3:
        return False
    steps = np.diff(col)
    if np.any(steps <= 0):
        return False
    return bool(np.std(steps) <= 1e-3 * abs(np.mean(steps)))


def sniff_format(path: Path) -> CsvFormat:
    """Read only the first few lines to work out delimiter, header size and column roles."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = [fh.readline() for _ in range(_SNIFF_LINES)]
    lines = [ln.strip() for ln in raw if ln.strip()]
    if not lines:
        raise ValueError(f"{path} appears to be empty")

    best: tuple[int, str | None, int] = (0, ",", 0)  # (n_cols, delimiter, n_consistent)
    for delim in _DELIMITERS:
        stripped = [_strip_trailing_empty(_split(ln, delim)) for ln in lines]
        counts = [len(f) for f in stripped if _all_float(f)]
        if not counts:
            continue
        n_cols = max(set(counts), key=counts.count)
        n_consistent = counts.count(n_cols)
        if n_cols >= 1 and (n_cols, n_consistent) > (best[0], best[2]):
            best = (n_cols, delim, n_consistent)

    n_cols, delimiter, _ = best
    if n_cols < 1:
        raise ValueError(
            f"Could not find at least 1 numeric column in {path}. "
            f"First line was: {lines[0][:120]!r}"
        )

    header_lines = 0
    for ln in raw:
        if ln.strip() and _all_float(_split(ln.strip(), delimiter)):
            break
        header_lines += 1
    header_text = [ln.strip() for ln in raw[:header_lines] if ln.strip()]

    rows = [_strip_trailing_empty(_split(ln, delimiter)) for ln in lines]
    data = np.array(
        [[float(x) for x in r] for r in rows if len(r) == n_cols and _all_float(r)],
        dtype=np.float64,
    )

    time_col: int | None = None
    dt_s: float | None = None
    if len(data) >= 3 and _is_time_axis(data[:, 0]):
        time_col = 0
        dt_s = float(np.median(np.diff(data[:, 0])))

    value_cols_all = [c for c in range(n_cols) if c != time_col]
    if len(value_cols_all) < 1:
        raise ValueError(
            f"{path} has {n_cols} columns of which column 0 looks like a time axis, leaving no "
            f"value columns. Pass explicit --pos-col/--neg-col if the auto-detection is wrong."
        )
    value_cols = tuple(value_cols_all[:2])

    if dt_s is None:
        dt_s = _header_dt_s(header_text)

    fmt = CsvFormat(
        delimiter=delimiter,
        header_lines=header_lines,
        n_cols=n_cols,
        time_col=time_col,
        value_cols=value_cols,
        dt_s=dt_s,
        header_text=header_text,
    )
    logger.info("Sniffed %s: %r", path.name, fmt)
    for ln in header_text:
        logger.info("  header: %s", ln[:200])
    return fmt


def _load_npz(path: Path, params: WaveformParams) -> Waveform:
    with np.load(path) as npz:
        ch_a = np.asarray(npz["ch_a"], dtype=np.float32)
        ch_b = (
            np.asarray(npz["ch_b"], dtype=np.float32) if "ch_b" in npz else np.zeros_like(ch_a)
        )
        file_dt_s = float(npz["dt_s"])

    dt_s = file_dt_s
    if params.sample_rate is not None:
        dt_s = params.sample_rate.to_s()
        if abs(dt_s - file_dt_s) > 1e-3 * file_dt_s:
            logger.warning(
                "Overriding npz dt=%.6g s with --sample-rate dt=%.6g s", file_dt_s, dt_s,
            )
    if params.max_samples is not None:
        ch_a = ch_a[: params.max_samples]
        ch_b = ch_b[: params.max_samples]

    fmt = CsvFormat(
        delimiter=None, header_lines=0, n_cols=2, time_col=None,
        value_cols=(0, 1), dt_s=file_dt_s, header_text=[],
    )
    wf = Waveform(ch_a=ch_a, ch_b=ch_b, dt_s=float(dt_s), fmt=fmt, source=path)
    logger.info(
        "Loaded %d samples from %s, dt=%.6g s (%.6g Sa/s), duration=%.6g s",
        wf.n, path.name, wf.dt_s, 1.0 / wf.dt_s, wf.duration_s,
    )
    return wf


def load_waveform(params: WaveformParams) -> Waveform:
    path = Path(params.path)
    if path.suffix.lower() == ".npz":
        if params.pos_col is not None or params.neg_col is not None:
            logger.warning("--pos-col/--neg-col are ignored for .npz input (fixed ch_a/ch_b)")
        return _load_npz(path, params)
    fmt = sniff_format(path)

    pos_col = params.pos_col if params.pos_col is not None else fmt.value_cols[0]
    single_channel = params.neg_col is None and len(fmt.value_cols) < 2
    neg_col = params.neg_col if params.neg_col is not None else (
        None if single_channel else fmt.value_cols[1]
    )
    if not single_channel and pos_col == neg_col:
        raise ValueError(f"pos_col and neg_col must differ (both are {pos_col})")

    dt_s = fmt.dt_s
    if params.sample_rate is not None:
        dt_s = params.sample_rate.to_s()
        if fmt.dt_s is not None and abs(dt_s - fmt.dt_s) > 1e-3 * fmt.dt_s:
            logger.warning(
                "Overriding CSV time column dt=%.6g s with --sample-rate dt=%.6g s",
                fmt.dt_s, dt_s,
            )
    if dt_s is None:
        raise ValueError(
            f"{path.name} has no time column, so the sample period is unknown. "
            f"Pass --sample-rate (e.g. --sample-rate 5e9 for 5 GSa/s)."
        )

    logger.info(
        "Loading %s (cols %s, skiprows=%d, max_rows=%s) ...",
        path.name, pos_col if single_channel else f"{pos_col},{neg_col}",
        fmt.header_lines, params.max_samples,
    )
    usecols = (pos_col,) if single_channel else (pos_col, neg_col)
    arr = np.loadtxt(
        path,
        delimiter=fmt.delimiter,
        skiprows=fmt.header_lines,
        usecols=usecols,
        max_rows=params.max_samples,
        dtype=np.float32,
        ndmin=2,
    )
    ch_a = np.ascontiguousarray(arr[:, 0])
    ch_b = np.zeros_like(ch_a) if single_channel else np.ascontiguousarray(arr[:, 1])
    if single_channel:
        logger.info("Single-channel capture (no second column) — ch_b synthesized as zeros")
    wf = Waveform(
        ch_a=ch_a,
        ch_b=ch_b,
        dt_s=float(dt_s),
        fmt=fmt,
        source=path,
    )
    logger.info(
        "Loaded %d samples, dt=%.6g s (%.6g Sa/s), duration=%.6g s",
        wf.n, wf.dt_s, 1.0 / wf.dt_s, wf.duration_s,
    )
    return wf
