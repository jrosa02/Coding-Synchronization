import numpy as np
from coding_synchronization.channel.RandomShift import RandomShift
from coding_synchronization.channel.VanishPulses import VanishPulses

from src.coding_synchronization.FrameGen import FrameGen

WORD_SIZE = 10
CONSTANT_OFFSET = 1000

generator = FrameGen(
    sync_num=4, metadata_num=4, data_num=4, ecc_num=4, eof_num=4, word_size=WORD_SIZE
)
vanisher = VanishPulses(rate=0.05)
shifter = RandomShift(sigma=0.5)

data = np.random.randint(0, (1 << WORD_SIZE) - 1, 8, dtype=np.uint16)
print("original data: ", data)

positions = generator.construct_frames(data)
print("ppm positions (first 8):", positions[:8])

positions = vanisher.vanish(positions)
print(f"after vanish ({len(positions)} pulses remaining):", positions[:8])

positions = shifter.shift(positions.astype(np.float64))
print("after awgn shift (first 8):", positions[:8])

positions = positions + CONSTANT_OFFSET
print("after constant offset (first 8):", positions[:8])

positions = np.round(positions).astype(np.uint64)
decoded = generator.decode_frames(positions)
print("decoded data:", decoded)
