import logging
from dataclasses import dataclass

import numpy as np

from coding_synchronization.decoder.Ecc import EccParams, make_codec
from coding_synchronization.encoder.Modulation import ModulationParams

logger = logging.getLogger(__name__)


@dataclass
class FrameParams:
    sync_num: int
    metadata_num: int
    data_num: int
    ecc_num: int
    eof_num: int


class FrameGen:
    def __init__(self, params: FrameParams, modulparams: ModulationParams, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.sync_num = params.sync_num
        self.metadata_num = params.metadata_num
        self.data_num = params.data_num
        self.ecc_num = params.ecc_num
        self.eof_num = params.eof_num
        self.ppm_rank = modulparams.ppm_rank
        self.max_value = (2**self.ppm_rank) - 1
        self.dead_slots = modulparams.dead_slots
        self.frame_len = sum(
            (params.sync_num, params.metadata_num, params.data_num, params.ecc_num)
        )
        self._meta_counter = 0

        # The ECC words are real Reed-Solomon parity over the metadata and data words, so a
        # simulated frame is a valid codeword and the decoder can correct it. Building the codec
        # through Ecc.make_codec keeps the encoder and the checker on one field and one generator.
        self.ecc_params = EccParams(
            ppm_rank=self.ppm_rank,
            ecc_num=self.ecc_num,
            info_num=self.metadata_num + self.data_num,
        )
        if self.ecc_num > 0:
            if self.ecc_params.n > self.max_value:
                raise ValueError(
                    f"Frame layout does not fit the RS field: metadata+data+ecc="
                    f"{self.ecc_params.n} exceeds 2^{self.ppm_rank} - 1 = {self.max_value}"
                )
            self._codec, prim = make_codec(self.ecc_params)
            logger.info(
                "FrameGen ECC: RS(%d,%d) over GF(2^%d), prim=%#x",
                self.ecc_params.n, self.ecc_params.info_num, self.ppm_rank, prim,
            )
        else:
            self._codec = None

        logger.info(
            "FrameGen initialized: frame_len=%d, ppm_rank=%d, word_period=%d",
            self.frame_len, self.ppm_rank, self.word_period,
        )

    @property
    def word_period(self) -> int:
        return self.max_value + 1 + self.dead_slots

    def reset(self) -> None:
        self._meta_counter = 0

    @property
    def coarse_sync(self) -> np.ndarray:
        return np.zeros(self.sync_num, dtype=np.uint16)

    def _fill_sync(self, frames: np.ndarray, s0: int) -> None:
        frames[:, :s0] = self.coarse_sync

    def _fill_metadata(self, frames: np.ndarray, s0: int, s1: int, n_frames: int) -> None:
        start = self._meta_counter + 1
        counter = np.arange(start, start + n_frames * self.metadata_num, dtype=np.uint16)
        frames[:, s0:s1] = counter.reshape(n_frames, self.metadata_num)
        self._meta_counter += n_frames * self.metadata_num

    def _fill_data(self, frames: np.ndarray, s1: int, s2: int, split_data: np.ndarray) -> None:
        frames[:, s1:s2] = split_data

    def _fill_ecc(self, frames: np.ndarray, s0: int, s2: int, s3: int, n_frames: int) -> None:
        """Reed-Solomon parity over the metadata and data words of each frame.

        Runs after _fill_metadata and _fill_data, because it encodes what they wrote.
        """
        if self._codec is None:
            return
        for i in range(n_frames):
            info = [int(v) for v in frames[i, s0:s2]]
            frames[i, s2:s3] = np.asarray(self._codec.encode(info)[-self.ecc_num:], dtype=np.uint16)

    def _to_positions(self, frames: np.ndarray, n_frames: int) -> np.ndarray:
        eof_slots = np.zeros((n_frames, self.eof_num), dtype=np.uint16)
        total_words = self.frame_len + self.eof_num
        all_words = np.concatenate([frames, eof_slots], axis=1).flatten()

        slot_indices = np.arange(len(all_words), dtype=np.uint64)
        positions = slot_indices * np.uint64(self.word_period) + all_words.astype(np.uint64)

        frame_indices = np.arange(n_frames, dtype=np.uint64)
        eof_starts = frame_indices * total_words + self.frame_len
        eof_offsets = np.arange(self.eof_num, dtype=np.uint64)
        eof_slot_indices = (eof_starts[:, None] + eof_offsets).flatten()
        mask = np.ones(len(all_words), dtype=bool)
        mask[eof_slot_indices] = False

        return positions[mask]

    def encode(self, data: np.ndarray) -> np.ndarray:
        remainder = len(data) % self.data_num
        if remainder != 0:
            data = np.pad(data, (0, self.data_num - remainder), constant_values=0)

        split_data = data.reshape(-1, self.data_num).astype(np.uint16)
        n_frames = split_data.shape[0]
        frames = np.zeros((n_frames, self.frame_len), dtype=np.uint16)

        s0, s1, s2, s3 = np.cumsum([self.sync_num, self.metadata_num, self.data_num, self.ecc_num])

        self._fill_sync(frames, s0)
        self._fill_metadata(frames, s0, s1, n_frames)
        self._fill_data(frames, s1, s2, split_data)
        self._fill_ecc(frames, s0, s2, s3, n_frames)

        positions = self._to_positions(frames, n_frames)
        logger.debug("encode: %d data words → %d frames, %d pulses", len(data), n_frames, len(positions))
        return positions
