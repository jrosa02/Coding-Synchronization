import logging
from typing import Any

import numpy as np

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


class Splitter(StageABC):
    def __init__(self, threshold: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.threshold = threshold
        logger.info("Splitter initialized: threshold=%.1f", threshold)

    def split(self, offsets: np.ndarray) -> list[np.ndarray]:
        gaps = np.diff(offsets)
        boundaries = np.where(gaps > self.threshold)[0] + 1
        return [chunk - chunk[0] for chunk in np.split(offsets, boundaries) if len(chunk) > 0]

    def process(
        self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        result = np.array(self.split(signal), dtype=object)
        logger.debug("Splitter: %d pulses → %d frames (threshold=%.1f)", len(signal), len(result), self.threshold)
        return result

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Splitter(threshold={self.threshold})"


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    # Synthetic stream: 4 frames, each 10 pulses, separated by large gaps
    rng = np.random.default_rng(42)
    frames_in = [np.sort(rng.uniform(i * 1000, i * 1000 + 200, 10)) for i in range(4)]
    positions = np.concatenate(frames_in)

    splitter = Splitter(threshold=500.0)
    frames_out = splitter.split(positions)
    gaps = np.diff(positions)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 8))

    ax0.vlines(positions, 0, 1, linewidth=0.8)
    ax0.set_title("Raw pulse stream (unsplit)")
    ax0.set_xlabel("Position")
    ax0.set_yticks([])
    ax0.grid(True, axis="x")

    ax1.plot(gaps, linewidth=0.8)
    ax1.axhline(
        splitter.threshold,
        color="red",
        linestyle="--",
        label=f"threshold={splitter.threshold:.0f}",
    )
    ax1.set_title("Inter-pulse gaps")
    ax1.set_xlabel("Pulse index")
    ax1.set_ylabel("Gap (chirps)")
    ax1.legend()
    ax1.grid(True)

    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(frames_out)))
    for i, (frame, color) in enumerate(zip(frames_out, colors, strict=False)):
        ax2.vlines(frame, i, i + 0.8, linewidth=1.0, color=color)
    ax2.set_title(f"Split result: {len(frames_out)} frames")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Frame index")
    ax2.grid(True, axis="x")

    os.makedirs("output", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output/splitter.png", dpi=150)
    logger.debug("saved splitter.png — %d frames", len(frames_out))
    plt.show()
