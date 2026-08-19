import logging
import math
from dataclasses import dataclass

import numpy as np

from coding_synchronization.encoder.FrameGen import FrameGen, FrameParams
from coding_synchronization.encoder.Modulation import ModulationParams
from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)

_R_EARTH_KM = 6371.0
_GM_KM3_S2 = 3.986e5  # km³/s²


def _elevation_to_time(altitude_km: float, max_elevation_deg: float) -> float:
    """Compute satellite pass duration from orbital altitude and peak elevation angle."""
    h = altitude_km
    r = _R_EARTH_KM + h
    theta = math.radians(max_elevation_deg)

    T_orb = 2 * math.pi * math.sqrt(r**3 / _GM_KM3_S2)
    lambda_0 = math.acos(_R_EARTH_KM / r)
    nadir = math.asin(_R_EARTH_KM * math.cos(theta) / r)
    lambda_min = math.pi / 2 - theta - nadir

    lambda_min = max(0.0, min(lambda_min, lambda_0))
    half_arc = math.sqrt(max(0.0, lambda_0**2 - lambda_min**2))
    return T_orb * half_arc / math.pi


@dataclass
class PassageParams:
    altitude_km: float
    max_elevation_deg: float
    time_s: float | None = None  # if set, caps the elevation-derived pass time


class PassageGen(StageABC):
    def __init__(
        self,
        frame_params: FrameParams,
        mod_params: ModulationParams,
        overflight_params: PassageParams,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self._frame_gen = FrameGen(frame_params, modulparams=mod_params, seed=seed)
        self._data_num = frame_params.data_num

        frame_duration_s = (
            (self._frame_gen.frame_len + frame_params.eof_num)
            * self._frame_gen.word_period
            * float(mod_params.slot_time)
        )

        computed_time_s = _elevation_to_time(
            overflight_params.altitude_km, overflight_params.max_elevation_deg
        )
        if overflight_params.time_s is not None:
            actual_time_s = min(overflight_params.time_s, computed_time_s)
        else:
            actual_time_s = computed_time_s

        self.n_frames = max(1, math.floor(actual_time_s / frame_duration_s))

        self._data: np.ndarray | None = None
        self._generated = False
        logger.info(
            "Frames2send: %d  (pass_time_s=%.3f, frame_duration_s=%.6f)",
            self.n_frames, actual_time_s, frame_duration_s,
        )

    def load(self, data: np.ndarray) -> None:
        self._data = data
        self._generated = False

    def generate(self) -> np.ndarray | None:
        if self._generated:
            return None
        self._generated = True

        total_words = self.n_frames * self._data_num
        if self._data is not None:
            if len(self._data) >= total_words:
                data = self._data[:total_words]
            else:
                repeats = math.ceil(total_words / len(self._data))
                data = np.tile(self._data, repeats)[:total_words]
        else:
            data = self.rng.integers(0, self._frame_gen.max_value + 1, total_words, dtype=np.uint16)

        result = self._frame_gen.encode(data)
        logger.debug("generate: emitting %d pulses across %d frames", len(result), self.n_frames)
        return result

    def reset(self) -> None:
        self._generated = False
        self._frame_gen.reset()

    def __repr__(self) -> str:
        return f"PassageGen(n_frames={self.n_frames})"
