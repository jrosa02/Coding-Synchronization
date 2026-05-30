import numpy as np

from coding_synchronization.StageABC import StageABC


class AddedPulses(StageABC):
    def __init__(self, rate: float, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.rate = rate

    def process(self, signal: np.ndarray) -> np.ndarray:
        signal = signal.astype(np.float64)
        n_added = self.rng.poisson(self.rate * len(signal))
        spurious = self.rng.uniform(signal[0], signal[-1], n_added)
        return np.sort(np.concatenate([signal, spurious]))

    def reset(self) -> None:
        pass
