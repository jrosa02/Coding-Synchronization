import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from coding_synchronization.StageABC import StageABC


class BarPlotStage(StageABC):
    def __init__(
        self, chunk_index: int = 0, ax: Axes | None = None, title: str | None = None, seed: int = 42
    ) -> None:
        super().__init__(seed)
        self.chunk_index = chunk_index
        self._current_chunk = 0
        if ax is not None:
            self.ax: Axes = ax
        else:
            _, self.ax = plt.subplots(figsize=(8, 4))
        if title:
            self.ax.set_title(title)

    def plot(self, signal: np.ndarray) -> None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if signal.dtype == object:
            offset = 0
            for i, row in enumerate(signal):
                color = colors[i % len(colors)]
                xs = list(range(offset, offset + len(row)))
                self.ax.bar(xs, row, color=color, alpha=0.7, linewidth=0)
                offset += len(row)
        else:
            self.ax.bar(list(range(len(signal))), signal, color=colors[0], alpha=0.7, linewidth=0)
        self.ax.grid(True, axis="y")

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self._current_chunk == self.chunk_index:
            self.plot(signal)
        self._current_chunk += 1
        return signal

    def reset(self) -> None:
        self._current_chunk = 0
        self.ax.clear()

    def __repr__(self) -> str:
        return "BarPlot"


class PulsePlotStage(StageABC):
    def __init__(
        self, chunk_index: int = 0, ax: Axes | None = None, title: str | None = None, seed: int = 42
    ) -> None:
        super().__init__(seed)
        self.chunk_index = chunk_index
        self._current_chunk = 0
        if ax is not None:
            self.ax: Axes = ax
        else:
            _, self.ax = plt.subplots(figsize=(8, 4))
        if title:
            self.ax.set_title(title)

    def plot(self, signal: np.ndarray) -> None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if signal.dtype == object:
            offset = 0.0
            for i, row in enumerate(signal):
                row = np.asarray(row, dtype=float)
                span = row[-1] - row[0] if len(row) > 1 else 1.0
                self.ax.vlines(
                    row - row[0] + offset, 0, 1, lw=0.5, alpha=0.7, color=colors[i % len(colors)]
                )
                offset += span * 1.3
        else:
            self.ax.vlines(signal, 0, 1, linewidth=0.5, alpha=0.7, color=colors[0])
        self.ax.set_yticks([])
        self.ax.grid(True, axis="x")

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self._current_chunk == self.chunk_index:
            self.plot(signal)
        self._current_chunk += 1
        return signal

    def reset(self) -> None:
        self._current_chunk = 0
        self.ax.clear()

    def __repr__(self) -> str:
        return "PulsePlot"


class ListDisplayStage(StageABC):
    def __init__(
        self, chunk_index: int = 0, ax: Axes | None = None, title: str | None = None, seed: int = 42
    ) -> None:
        super().__init__(seed)
        self.chunk_index = chunk_index
        self._current_chunk = 0
        if ax is not None:
            self.ax: Axes = ax
        else:
            _, self.ax = plt.subplots(figsize=(8, 4))
        if title:
            self.ax.set_title(title)

    _MAX_DISPLAY = 70
    _N_COLS = 8

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

    def display(self, signal: np.ndarray) -> None:
        self.ax.axis("off")
        if signal.dtype == object:
            rows = []
            for i, sub in enumerate(signal):
                vals = self._fmt_arr(np.asarray(sub))
                chunks = [vals[j : j + self._N_COLS] for j in range(0, len(vals), self._N_COLS)]
                for k, chunk in enumerate(chunks):
                    label = f"F{i}" if k == 0 else ""
                    rows.append([label, *chunk])
            self._make_table(rows)
        else:
            vals = self._fmt_arr(signal)
            if len(vals) == 1 and vals[0].startswith("("):
                self.ax.text(
                    0.5,
                    0.5,
                    vals[0],
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    fontsize=8,
                    color="grey",
                )
                return
            chunks = [vals[j : j + self._N_COLS] for j in range(0, len(vals), self._N_COLS)]
            self._make_table(chunks)

    def process(self, signal: np.ndarray) -> np.ndarray:
        if self._current_chunk == self.chunk_index:
            self.display(signal)
        self._current_chunk += 1
        return signal

    def reset(self) -> None:
        self._current_chunk = 0
        self.ax.clear()

    def __repr__(self) -> str:
        return "List"
