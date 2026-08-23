"""RS integrity check: the error counts and rates it reports must match errors deliberately made.

A clean capture reports all zeros, which proves nothing about the arithmetic — so these tests
corrupt known symbols in valid codewords and check the report against what was corrupted.
"""

import numpy as np
import pytest

from coding_synchronization.decoder.Ecc import EccParams, check_frames, make_codec

PPM_RANK = 10
INFO_NUM = 21  # metadata_num=5 + data_num=16
ECC_NUM = 16   # RS(37, 21), correcting up to 8 symbols


def _params() -> EccParams:
    return EccParams(ppm_rank=PPM_RANK, ecc_num=ECC_NUM, info_num=INFO_NUM)


def _codewords(n_frames: int, seed: int = 0) -> list[np.ndarray]:
    codec, _ = make_codec(_params())
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        info = rng.integers(0, 1 << PPM_RANK, INFO_NUM).tolist()
        frames.append(np.array(codec.encode(info), dtype=np.uint16))
    return frames


def test_valid_codewords_report_no_errors():
    report = check_frames(_codewords(5), _params())
    assert report.n_frames == 5
    assert all(f.ok and f.symbol_errors == 0 for f in report.frames)
    assert report.rates() == {"wer_pre": 0.0, "ber_pre": 0.0, "wer_post": 0.0, "ber_post": 0.0}


@pytest.mark.parametrize("n_bad", [1, 3, 8])
def test_correctable_errors_are_counted_and_cleared(n_bad):
    """Up to the correction limit: pre-ECC counts what was broken, post-ECC is clean."""
    params = _params()
    frames = _codewords(4)
    positions = list(range(n_bad))
    expected_bits = 0
    for frame in frames:
        for j in positions:
            corrupted = int(frame[j]) ^ 0b101  # 2 bits per corrupted symbol
            expected_bits += 2
            frame[j] = corrupted

    report = check_frames(frames, params)
    assert all(f.ok for f in report.frames), "within the limit RS must still decode"
    assert [f.symbol_errors for f in report.frames] == [n_bad] * 4
    assert [f.positions for f in report.frames] == [positions] * 4
    assert sum(f.bit_errors for f in report.frames) == expected_bits

    rates = report.rates()
    total_symbols = 4 * params.n
    assert rates["wer_pre"] == pytest.approx(4 * n_bad / total_symbols)
    assert rates["ber_pre"] == pytest.approx(expected_bits / (total_symbols * PPM_RANK))
    assert rates["wer_post"] == 0.0 and rates["ber_post"] == 0.0
    assert report.frame_error_rate == 0.0


def test_beyond_the_limit_the_frame_is_charged_in_full():
    """Past the correction limit RS gives nothing back, so the whole frame counts as lost."""
    params = _params()
    frames = _codewords(4)
    for frame in frames[:1]:
        for j in range(params.correctable + 4):  # 12 > 8 correctable
            frame[j] = (int(frame[j]) + 1) % (1 << PPM_RANK)

    report = check_frames(frames, params)
    assert report.n_uncorrectable == 1
    assert report.frame_error_rate == pytest.approx(1 / 4)

    rates = report.rates()
    total_symbols = 4 * params.n
    # Post-ECC: every symbol of the lost frame.
    assert rates["wer_post"] == pytest.approx(params.n / total_symbols)
    assert rates["ber_post"] == pytest.approx(params.n / total_symbols)
    # Pre-ECC: the lower bound, one symbol past what RS can correct.
    assert rates["wer_pre"] == pytest.approx((params.correctable + 1) / total_symbols)


def test_wrong_length_frames_are_skipped_not_counted():
    """A frame that lost a pulse is a framing failure, not a channel measurement."""
    frames = _codewords(3)
    frames[1] = frames[1][:-1]
    report = check_frames(frames, _params())
    assert report.n_frames == 2
    assert [f.index for f in report.frames] == [0, 2]
