"""The encoder writes real parity, the ECC stage corrects, and the metadata check verifies.

These three pieces only work together: `FrameGen` must produce valid codewords, `EccDecode` must
hand corrected words to the next stage, and `MetadataCheck` must read those corrected words.
"""

import numpy as np
import pytest

from coding_synchronization.decoder.Ecc import EccDecode, EccParams, check_frames
from coding_synchronization.decoder.Metadata import MetadataCheck
from coding_synchronization.encoder import FrameParams, ModulationParams
from coding_synchronization.encoder.FrameGen import FrameGen

PPM_RANK = 10
DEAD_SLOTS = 16
WORD_PERIOD = (1 << PPM_RANK) + DEAD_SLOTS
FRAME = FrameParams(sync_num=8, metadata_num=5, data_num=16, ecc_num=16, eof_num=2)
MOD = ModulationParams(ppm_rank=PPM_RANK, slot_time=np.float64(1e-6), dead_slots=DEAD_SLOTS)
ECC = EccParams(ppm_rank=PPM_RANK, ecc_num=FRAME.ecc_num,
                info_num=FRAME.metadata_num + FRAME.data_num)


def _frames_from_encoder(n_frames: int = 3) -> list[np.ndarray]:
    """Encode n_frames and read the words back out of the pulse positions, sync words dropped."""
    gen = FrameGen(FRAME, MOD)
    data = np.random.default_rng(0).integers(0, 1 << PPM_RANK, FRAME.data_num * n_frames)
    positions = gen.encode(data.astype(np.uint16))

    per_frame = FRAME.sync_num + FRAME.metadata_num + FRAME.data_num + FRAME.ecc_num
    out = []
    for i in range(n_frames):
        chunk = positions[i * per_frame:(i + 1) * per_frame]
        rel = chunk - chunk[0]
        index = np.floor(rel / WORD_PERIOD).astype(int)
        out.append((rel - index * WORD_PERIOD).astype(np.uint16)[FRAME.sync_num:])
    return out


def test_frame_gen_writes_valid_parity():
    """Every encoded frame must decode with zero corrections."""
    report = check_frames(_frames_from_encoder(), ECC)
    assert report.n_frames == 3
    assert report.n_uncorrectable == 0
    assert [f.symbol_errors for f in report.frames] == [0, 0, 0]


def test_frame_gen_rejects_a_layout_that_does_not_fit_the_field():
    """metadata + data + ecc must fit in 2^ppm_rank - 1 symbols."""
    too_big = FrameParams(sync_num=8, metadata_num=5, data_num=2000, ecc_num=16, eof_num=2)
    with pytest.raises(ValueError, match="does not fit the RS field"):
        FrameGen(too_big, MOD)


def test_ecc_stage_hands_corrected_words_to_the_next_stage():
    """A corrupted word must reach MetadataCheck already corrected."""
    frames = _frames_from_encoder(2)
    original = [f.copy() for f in frames]
    frames[0][2] = (int(frames[0][2]) + 7) % (1 << PPM_RANK)   # break a metadata word

    stage = EccDecode(ECC)
    corrected = stage.process(np.asanyarray(frames, dtype=object))

    np.testing.assert_array_equal(corrected[0], original[0])
    report = stage.report()
    assert report.frames[0].symbol_errors == 1
    assert report.frames[0].positions == [2]
    assert report.n_uncorrectable == 0


def test_ecc_stage_passes_an_uncorrectable_frame_through_unchanged():
    frames = _frames_from_encoder(1)
    for j in range(ECC.correctable + 4):
        frames[0][j] = (int(frames[0][j]) + 1) % (1 << PPM_RANK)
    broken = frames[0].copy()

    stage = EccDecode(ECC)
    out = stage.process(np.asanyarray(frames, dtype=object))

    np.testing.assert_array_equal(out[0], broken)
    assert stage.report().n_uncorrectable == 1


def test_metadata_check_counts_a_mismatch_and_strict_raises():
    good = np.arange(1, FRAME.metadata_num + 1, dtype=np.uint16)
    bad = good.copy()
    bad[2] += 5
    payload = np.zeros(FRAME.data_num, dtype=np.uint16)
    frames = np.asanyarray(
        [np.concatenate([good, payload]), np.concatenate([bad, payload])], dtype=object
    )

    stage = MetadataCheck(FRAME.metadata_num, verify=True)
    output = stage.process(frames)
    assert stage.frames_checked == 2
    assert stage.mismatches == 1
    assert stage.mismatch_rate == pytest.approx(0.5)
    assert len(output[0]) == FRAME.data_num, "the metadata words must still be stripped"

    with pytest.raises(ValueError, match="is not a consecutive counter"):
        MetadataCheck(FRAME.metadata_num, strict=True).process(frames)


def test_metadata_check_stays_silent_when_verification_is_off():
    frames = np.asanyarray([np.array([9, 4, 7, 1, 3] + [0] * FRAME.data_num, dtype=np.uint16)],
                           dtype=object)
    stage = MetadataCheck(FRAME.metadata_num)
    stage.process(frames)
    assert stage.frames_checked == 0
    assert stage.mismatches == 0
