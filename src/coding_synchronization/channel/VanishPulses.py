import numpy as np


class VanishPulses:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def vanish(self, frames: np.ndarray) -> np.ndarray:
        mask = np.random.random(len(frames)) >= self.rate
        return frames[mask]
