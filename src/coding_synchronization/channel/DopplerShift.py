import numpy as np

from coding_synchronization.StageABC import StageABC

C = 2.998e8  # m/s
GM = 3.986e14  # m³/s²
R_EARTH = 6.371e6  # m


class DopplerShift(StageABC):
    def __init__(
        self,
        altitude_km: float,
        slot_time_s: np.float64 = 20e-9,
        tca_slot: float | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.altitude_m = altitude_km * 1e3
        self.chip_duration_s = slot_time_s
        self.tca_chip = tca_slot
        self.velocity_m_s = np.sqrt(GM / (R_EARTH + self.altitude_m))

    def _resolve_tca(self, offsets: np.ndarray) -> float:
        if self.tca_chip is not None:
            return float(self.tca_chip)
        return float((offsets[0] + offsets[-1]) / 2.0)

    def _offsets_to_time(self, offsets: np.ndarray, tca: float) -> np.ndarray:
        return (offsets - tca) * self.chip_duration_s

    def _slant_range(self, t: np.ndarray) -> np.ndarray:
        return np.sqrt(self.altitude_m**2 + (self.velocity_m_s * t) ** 2)

    def _range_delay_offsets(self, slant_range: np.ndarray) -> np.ndarray:
        return (slant_range - self.altitude_m) / (C * self.chip_duration_s)

    def process(self, signal: np.ndarray) -> np.ndarray:
        signal = signal.astype(np.float64)
        tca = self._resolve_tca(signal)
        t = self._offsets_to_time(signal, tca)
        slant_range = self._slant_range(t)
        return signal + self._range_delay_offsets(slant_range)

    def reset(self) -> None:
        pass


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    doppler = DopplerShift(altitude_km=500)

    # Simulate a 600-second pass centered at TCA, 20ns chips
    pass_duration_s = 600.0
    n_chips = int(pass_duration_s / doppler.chip_duration_s)
    offsets = np.linspace(0, n_chips - 1, 10_000, dtype=np.float64)

    tca = doppler._resolve_tca(offsets)
    t = doppler._offsets_to_time(offsets, tca)
    slant_range = doppler._slant_range(t)
    delay_chips = doppler._range_delay_offsets(slant_range)
    delay_us = delay_chips * doppler.chip_duration_s * 1e6

    # Instantaneous Doppler frequency shift: d(delay)/dt * f_chip
    doppler_factor = np.gradient(delay_chips, offsets)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, delay_us)
    ax1.set_ylabel("Propagation delay (µs)")
    ax1.set_title(f"Doppler shift — LEO zenith pass, altitude={500} km")
    ax1.grid(True)

    ax2.plot(t, doppler_factor * 1e6)
    ax2.set_xlabel("Time relative to TCA (s)")
    ax2.set_ylabel("Instantaneous shift (chips/Mchip)")
    ax2.grid(True)

    os.makedirs("output", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output/doppler_shift.png", dpi=150)
    print("saved output/doppler_shift.png")
    plt.show()
