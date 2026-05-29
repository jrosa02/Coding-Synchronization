import numpy as np


class FrameGen:
    def __init__(
        self,
        sync_num: int,
        metadata_num: int,
        data_num: int,
        ecc_num: int,
        eof_num: int,
        word_size: int,
    ) -> None:
        self.sync_num = sync_num
        self.metadata_num = metadata_num
        self.data_num = data_num
        self.ecc_num = ecc_num
        self.eof_num = eof_num
        self.word_size = word_size
        self.max_value = (2**word_size) - 1
        self.frame_len = sum((sync_num, metadata_num, data_num, ecc_num))

    def construct_frames(self, data: np.ndarray) -> np.ndarray:
        remainder = len(data) % self.data_num
        if remainder != 0:
            data = np.pad(data, (0, self.data_num - remainder), constant_values=0)

        split_data = data.reshape(-1, self.data_num).astype(np.uint16)
        n_frames = split_data.shape[0]
        frames = np.zeros((n_frames, self.frame_len), dtype=np.uint16)

        s0, s1, s2, s3 = np.cumsum([self.sync_num, self.metadata_num, self.data_num, self.ecc_num])

        # SYNC: max_value repeated
        frames[:, :s0] = [0, 1023, 256, 512]

        # METADATA: continuous counter starting at 1
        meta_counter = np.arange(1, n_frames * self.metadata_num + 1, dtype=np.uint16)
        frames[:, s0:s1] = meta_counter.reshape(n_frames, self.metadata_num)

        # DATA
        frames[:, s1:s2] = split_data

        # ECC: random placeholder
        frames[:, s2:s3] = np.random.randint(
            0, self.max_value + 1, size=(n_frames, self.ecc_num), dtype=np.uint16
        )

        # Append eof_num vacant slots per frame (value=0, no pulse emitted)
        eof_slots = np.zeros((n_frames, self.eof_num), dtype=np.uint16)
        total_words_per_frame = self.frame_len + self.eof_num
        all_frames = np.concatenate([frames, eof_slots], axis=1).flatten()

        # PPM: pulse position = slot_index * (max_value + 1) + word_value
        slot_indices = np.arange(len(all_frames), dtype=np.uint64)
        positions = slot_indices * (self.max_value + 1) + all_frames.astype(np.uint64)

        # Mask out EOF slots by position — they occupy the last eof_num slots of each frame
        frame_indices = np.arange(n_frames, dtype=np.uint64)
        eof_starts = frame_indices * total_words_per_frame + self.frame_len
        eof_offsets = np.arange(self.eof_num, dtype=np.uint64)
        eof_slot_indices = (eof_starts[:, None] + eof_offsets).flatten()
        mask = np.ones(len(all_frames), dtype=bool)
        mask[eof_slot_indices] = False

        return positions[mask]

    def decode_frames(self, positions: np.ndarray) -> np.ndarray:
        slot_size = np.uint64(self.max_value + 1)
        total_words_per_frame = np.uint64(self.frame_len + self.eof_num)

        slot_indices = positions // slot_size
        values = (positions % slot_size).astype(np.uint16)

        frame_nums = slot_indices // total_words_per_frame
        word_in_frame = slot_indices % total_words_per_frame

        n_frames = int(frame_nums[-1]) + 1
        frames = np.zeros((n_frames, self.frame_len), dtype=np.uint16)
        frames[frame_nums, word_in_frame] = values

        _, s1, s2, _ = np.cumsum([self.sync_num, self.metadata_num, self.data_num, self.ecc_num])

        return frames[:, s1:s2].flatten()


if __name__ == "__main__":
    generator = FrameGen(4, 4, 4, 4, 4, 10)
    data = np.random.randint(0, 1 << 10 - 1, 8)
    print(data)
    frames = generator.construct_frames(data)
    print(frames)
    print(generator.decode_frames(frames))
