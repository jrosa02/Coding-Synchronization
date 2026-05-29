import numpy as np


class Syncer:
    def __init__(self, sync_num: int, sigma: float, vanish_rate: float, word_size: int) -> None:
        self.sync_num = sync_num
        self.sigma = sigma
        self.vanish_rate = vanish_rate
        self.slot_width = 2**word_size
        self.sync_symbols = np.array([0, 1023, 256, 512])
        self.sync_offsets = np.arange(len(self.sync_symbols)) * self.slot_width + self.sync_symbols

    def construct_bayesian_reference(self) -> np.ndarray:
        span = int(self.sync_offsets[-1] + 6 * self.sigma) + 1
        grid = np.arange(span, dtype=np.float64)
        norm = self.sigma * np.sqrt(2 * np.pi)
        reference = np.zeros(span)
        for offset in self.sync_offsets:
            reference += (
                (1 - self.vanish_rate) * np.exp(-0.5 * ((grid - offset) / self.sigma) ** 2) / norm
            )
        return reference

    def _log_likelihood(self, candidates: np.ndarray, sorted_positions: np.ndarray) -> np.ndarray:
        baseline = self.vanish_rate / float(self.sync_offsets[-1] + 6 * self.sigma)
        norm = self.sigma * np.sqrt(2 * np.pi)
        scores = np.zeros(len(candidates))
        for offset in self.sync_offsets:
            expected = candidates + offset
            idx = np.searchsorted(sorted_positions, expected).clip(1, len(sorted_positions) - 1)
            d_right = np.abs(sorted_positions[idx] - expected)
            d_left = np.abs(sorted_positions[idx - 1] - expected)
            nearest = np.minimum(d_right, d_left)
            p = (1 - self.vanish_rate) * np.exp(-0.5 * (nearest / self.sigma) ** 2) / norm
            scores += np.log(p + baseline)
        return scores

    def detect(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fpos = positions.astype(np.float64)
        sorted_pos = np.sort(fpos)
        # Only pulses near a slot boundary can be the first sync symbol (value=0)
        slot_phase = fpos % self.slot_width
        margin = 3 * self.sigma
        near_boundary = (slot_phase < margin) | (slot_phase > self.slot_width - margin)
        candidates = fpos[near_boundary]
        scores = self._log_likelihood(candidates, sorted_pos)
        threshold = np.percentile(scores, 90)
        is_peak = np.zeros(len(scores), dtype=bool)
        if len(scores) >= 3:
            is_peak[1:-1] = (scores[1:-1] > scores[:-2]) & (scores[1:-1] > scores[2:])
        if len(scores) >= 2:
            is_peak[0] = scores[0] > scores[1]
            is_peak[-1] = scores[-1] > scores[-2]
        elif len(scores) == 1:
            is_peak[0] = True
        is_peak &= scores > threshold
        return candidates[is_peak], scores[is_peak]


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    from src.coding_synchronization.channel.RandomShift import RandomShift
    from src.coding_synchronization.channel.VanishPulses import VanishPulses
    from src.coding_synchronization.FrameGen import FrameGen

    WORD_SIZE = 10
    SIGMA = 0.5
    VANISH_RATE = 0.05

    gen = FrameGen(
        sync_num=4, metadata_num=4, data_num=4, ecc_num=4, eof_num=4, word_size=WORD_SIZE
    )
    syncer = Syncer(sync_num=4, sigma=SIGMA, vanish_rate=VANISH_RATE, word_size=WORD_SIZE)

    data = np.random.randint(0, (1 << WORD_SIZE) - 1, 4, dtype=np.uint16)
    positions = gen.construct_frames(data)
    positions = VanishPulses(rate=VANISH_RATE).vanish(positions)
    positions = RandomShift(sigma=SIGMA).shift(positions.astype(np.float64))
    positions = positions[1:]

    reference = syncer.construct_bayesian_reference()
    sync_positions, sync_scores = syncer.detect(positions)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Reference KDE
    axes[0].plot(reference)
    axes[0].set_title("Bayesian reference — sync pattern KDE (chip space)")
    axes[0].set_xlabel("Chip offset from sync start")
    axes[0].set_ylabel("Probability density")
    axes[0].grid(True)

    # Pulse stream (first 5 frames worth of chips)
    frame_chips = (gen.frame_len + gen.eof_num) * syncer.slot_width
    view = positions[positions < 5 * frame_chips]
    axes[1].vlines(view, 0, 1, linewidth=0.5, alpha=0.6, label="pulses")
    axes[1].vlines(
        sync_positions[sync_positions < 5 * frame_chips],
        0,
        1,
        color="red",
        linewidth=1.2,
        label="detected sync",
    )
    axes[1].set_title("Pulse stream — first 5 frames (red = detected sync)")
    axes[1].set_xlabel("Chip position")
    axes[1].set_yticks([])
    axes[1].legend()
    axes[1].grid(True)

    # Log-likelihood scores
    axes[2].plot(
        positions,
        syncer._log_likelihood(positions.astype(np.float64), np.sort(positions.astype(np.float64))),
        linewidth=0.5,
        alpha=0.8,
    )
    axes[2].scatter(sync_positions, sync_scores, color="red", zorder=5, s=20, label="peaks")
    axes[2].set_title("Log-likelihood score per candidate pulse")
    axes[2].set_xlabel("Chip position")
    axes[2].set_ylabel("Log-likelihood")
    axes[2].legend()
    axes[2].grid(True)

    os.makedirs("output", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output/syncer_detection.png", dpi=150)
    print(f"saved output/syncer_detection.png — {len(sync_positions)} sync markers detected")
    plt.show()
