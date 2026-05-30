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
            for i, row in enumerate(signal):
                self.ax.vlines(row, 0, 1, lw=0.5, alpha=0.7, color=colors[i % len(colors)])
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

    _MAX_DISPLAY = 50

    def display(self, signal: np.ndarray) -> None:
        total = sum(len(r) for r in signal) if signal.dtype == object else len(signal)
        if total > self._MAX_DISPLAY:
            self.ax.axis("off")
            self.ax.text(
                0.5,
                0.5,
                f"({total} values)",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=8,
                color="grey",
            )
            return
        self.ax.axis("off")
        flat = np.concatenate(list(signal)) if signal.dtype == object else signal
        fmt = ".2f" if flat.dtype.kind == "f" else "d"
        vals = [f"{v:{fmt}}" for v in flat]
        n_cols = 5
        n_rows = max(1, (len(vals) + n_cols - 1) // n_cols)
        vals += [""] * (n_rows * n_cols - len(vals))
        cell_text = [vals[r::n_rows] for r in range(n_rows)]
        table = self.ax.table(cellText=cell_text, loc="center", cellLoc="right")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for cell in table.get_celld().values():
            cell.set_edgecolor("lightgrey")

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
