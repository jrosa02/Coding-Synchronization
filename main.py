import numpy as np

from coding_synchronization.channel.RandomShift import RandomShift
from coding_synchronization.channel.VanishPulses import VanishPulses
from coding_synchronization.encoder.FrameGen import FrameGen, FrameParams

WORD_SIZE = 10
CONSTANT_OFFSET = 1000

generator = FrameGen(
    FrameParams(sync_num=4, metadata_num=4, data_num=4, ecc_num=4, eof_num=4, word_size=WORD_SIZE)
)
vanisher = VanishPulses(rate=0.05)
shifter = RandomShift(sigma=0.5)

data = np.random.randint(0, (1 << WORD_SIZE) - 1, 8, dtype=np.uint16)
print("original data: ", data)

generator.load(data)
positions = generator.generate()
assert positions is not None
print("ppm positions (first 8):", positions[:8])

positions = vanisher.process(positions)
print(f"after vanish ({len(positions)} pulses remaining):", positions[:8])

positions = shifter.process(positions.astype(np.float64))
print("after awgn shift (first 8):", positions[:8])

positions = positions + CONSTANT_OFFSET
print("after constant offset (first 8):", positions[:8])

positions = np.round(positions).astype(np.uint64)
decoded = generator.decode(positions)
print("decoded data:", decoded)
