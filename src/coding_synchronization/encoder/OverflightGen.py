import math
from dataclasses import dataclass

import numpy as np

from coding_synchronization.encoder.FrameGen import FrameGen, FrameParams
from coding_synchronization.encoder.Modulation import ModulationParams
from coding_synchronization.StageABC import StageABC


@dataclass
class OverflightParams:
    time_s: float


class OverflightGen(StageABC):
    def __init__(
        self,
        frame_params: FrameParams,
        mod_params: ModulationParams,
        overflight_params: OverflightParams,
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
        self.n_frames = max(1, math.floor(overflight_params.time_s / frame_duration_s))

        self._data: np.ndarray | None = None
        self._generated = False
        print(f"Frames2send: {self.n_frames}")

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

        return self._frame_gen.encode(data)

    def reset(self) -> None:
        self._generated = False
        self._frame_gen.reset()
