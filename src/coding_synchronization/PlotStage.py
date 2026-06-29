from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from coding_synchronization.physical_units import Quantity
from coding_synchronization.StageABC import StageABC


@dataclass
class PlotInput:
    ax: Axes
    indxs: tuple[int, int]  # (chunk_index, signal_index)


@dataclass
class PlotInputFactory:
    axs: list[Axes]
    indxs: tuple[int, int]
    ax_nr: int = field(default=0, init=False)

    def reset(self) -> None:
        self.ax_nr = 0

    def __call__(self) -> PlotInput:
        result = PlotInput(self.axs[self.ax_nr], self.indxs)
        self.ax_nr += 1
        return result


class PlotStage(StageABC):
    _MAX_DISPLAY = 70
    _N_COLS = 8

    def __init__(
        self,
        plt_in: PlotInput,
        plot_type: Literal['plot', 'bar', 'vlines', 'table'] = 'plot',
        title: str | None = None,
        sample_rate: Quantity | None = None,
        plot_kwargs: dict | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed)
        self._chunk_index = 0
        self._plot_indexes = plt_in.indxs
        self.ax = plt_in.ax
        self.type = plot_type
        self.plot_kwargs = plot_kwargs or {}
        self.sample_rate = sample_rate
        if title:
            self.ax.set_title(title)

    def _get_time_axis(self, n_samples: int) -> tuple[np.ndarray, str]:
        if self.sample_rate is None:
            return np.arange(n_samples), "sample index"

        time_per_sample = self.sample_rate.to_s()
        time_values = np.arange(n_samples) * time_per_sample

        max_time = time_values[-1] if n_samples > 1 else time_per_sample
        if max_time > 1.0:
            scale, unit = 1.0, "s"
        elif max_time > 1e-3:
            scale, unit = 1e3, "ms"
        elif max_time > 1e-6:
            scale, unit = 1e6, "μs"
        else:
            scale, unit = 1e9, "ns"

        return time_values * scale, f"time ({unit})"

    # --- plot type implementations ---

    def _plot_line_or_bar(self, signals: np.ndarray) -> None:
        signal_index = self._plot_indexes[1]
        signal_values = signals[signal_index] if signals.dtype == object else signals
        x_axis, x_label = self._get_time_axis(len(signal_values))

        match self.type:
            case 'plot':
                self.ax.plot(
                    x_axis, signal_values,
                    **{'label': f"Signal at {self._plot_indexes}", **self.plot_kwargs},
                )
            case 'bar':
                width = (x_axis[1] - x_axis[0]) * 0.8 if len(x_axis) > 1 else 0.8
                self.ax.bar(
                    x_axis, signal_values,
                    **{'width': width, 'label': f"Signal at {self._plot_indexes}", **self.plot_kwargs},
                )
        self.ax.set_xlabel(x_label)
        self.ax.grid(True)

    def _plot_vlines(self, signal: np.ndarray) -> None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if signal.dtype == object:
            offset = 0.0
            for i, row in enumerate(signal):
                row = np.asarray(row, dtype=float)
                if self.sample_rate is not None and len(row) > 0:
                    _, x_label = self._get_time_axis(2)
                    time_per_sample = self.sample_rate.to_s()
                    max_time = (row[-1] - row[0]) * time_per_sample if len(row) > 1 else time_per_sample
                    if max_time > 1.0:
                        t_scale, x_label = 1.0, "s"
                    elif max_time > 1e-3:
                        t_scale, x_label = 1e3, "ms"
                    elif max_time > 1e-6:
                        t_scale, x_label = 1e6, "μs"
                    else:
                        t_scale, x_label = 1e9, "ns"
                    converted = (row - row[0]) * time_per_sample * t_scale
                    span = converted[-1] if len(converted) > 1 else 1.0
                    self.ax.vlines(
                        converted + offset, 0, 1, lw=0.5, alpha=0.7,
                        color=colors[i % len(colors)], **self.plot_kwargs,
                    )
                    self.ax.set_xlabel(x_label)
                else:
                    span = (row[-1] - row[0]) if len(row) > 1 else 1.0
                    self.ax.vlines(
                        row - row[0] + offset, 0, 1, lw=0.5, alpha=0.7,
                        color=colors[i % len(colors)], **self.plot_kwargs,
                    )
                offset += span * 1.3
        else:
            signal = np.asarray(signal, dtype=float)
            if self.sample_rate is not None and len(signal) > 0:
                time_per_sample = self.sample_rate.to_s()
                converted = signal * time_per_sample
                max_time = converted[-1] if len(converted) > 1 else time_per_sample
                if max_time > 1.0:
                    t_scale, x_label = 1.0, "s"
                elif max_time > 1e-3:
                    t_scale, x_label = 1e3, "ms"
                elif max_time > 1e-6:
                    t_scale, x_label = 1e6, "μs"
                else:
                    t_scale, x_label = 1e9, "ns"
                self.ax.vlines(
                    converted * t_scale, 0, 1, linewidth=0.5, alpha=0.7,
                    color=colors[0], **self.plot_kwargs,
                )
                self.ax.set_xlabel(x_label)
            else:
                self.ax.vlines(signal, 0, 1, linewidth=0.5, alpha=0.7, color=colors[0], **self.plot_kwargs)
        self.ax.set_yticks([])
        self.ax.grid(True, axis="x")

    def _fmt_arr(self, arr: np.ndarray) -> list[str]:
        arr = np.asarray(arr).flatten()
        if len(arr) > self._MAX_DISPLAY:
            return [f"({len(arr)} values)"]
        if np.issubdtype(arr.dtype, np.floating) or arr.dtype == object:
            return [f"{float(v):.2f}" for v in arr]
        return [f"{int(v):d}" for v in arr]

    def _make_table(self, rows: list[list[str]]) -> None:
        self.ax.axis("off")
        if not rows:
            return
        max_cols = max(len(r) for r in rows)
        padded = [r + [""] * (max_cols - len(r)) for r in rows]
        table = self.ax.table(cellText=padded, loc="center", cellLoc="right")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for cell in table.get_celld().values():
            cell.set_edgecolor("lightgrey")

    def _plot_table(self, signal: np.ndarray) -> None:
        self.ax.axis("off")
        if signal.dtype == object:
            rows: list[list[str]] = []
            for i, sub in enumerate(signal):
                vals = self._fmt_arr(np.asarray(sub))
                chunks = [vals[j: j + self._N_COLS] for j in range(0, len(vals), self._N_COLS)]
                for k, chunk in enumerate(chunks):
                    label = f"F{i}" if k == 0 else ""
                    rows.append([label, *chunk])
            self._make_table(rows)
        else:
            vals = self._fmt_arr(signal)
            if len(vals) == 1 and vals[0].startswith("("):
                self.ax.text(
                    0.5, 0.5, vals[0],
                    ha="center", va="center", transform=self.ax.transAxes,
                    fontsize=8, color="grey",
                )
                return
            chunks = [vals[j: j + self._N_COLS] for j in range(0, len(vals), self._N_COLS)]
            self._make_table(chunks)

    # --- pipeline interface ---

    def plot(self, signal: np.ndarray) -> None:
        match self.type:
            case 'plot' | 'bar':
                self._plot_line_or_bar(signal)
            case 'vlines':
                self._plot_vlines(signal)
            case 'table':
                self._plot_table(signal)

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self._chunk_index == self._plot_indexes[0]:
            self.plot(signal)
        self._chunk_index += 1
        return signal

    def reset(self) -> None:
        self._chunk_index = 0
        self.ax.clear()

    def __repr__(self) -> str:
        return f"PlotStage({self.type})"
