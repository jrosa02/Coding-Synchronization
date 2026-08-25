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
    assert report.n_decoded == 3
    assert report.frame_error_rate == pytest.approx(1 / 4)

    rates = report.rates()
    total_symbols = 4 * params.n
    # Post-ECC: every symbol of the lost frame. Charging whole frames makes the symbol rate and
    # the bit rate scale together, so both equal the frame error rate.
    assert rates["wer_post"] == pytest.approx(params.n / total_symbols)
    assert rates["ber_post"] == pytest.approx(params.n / total_symbols)
    assert rates["wer_post"] == pytest.approx(report.frame_error_rate)
    # Pre-ECC covers the decoded frames only. The uncorrectable frame has no reference, so its
    # error count is unmeasurable and it is left out rather than charged a made-up floor. The
    # other three frames are clean, so the rate over them is exactly zero.
    assert rates["wer_pre"] == 0.0
    assert rates["ber_pre"] == 0.0


def test_pre_ecc_rates_ignore_the_uncorrectable_frame():
    """One damaged-but-decodable frame and one lost frame: only the first sets the pre-ECC rate."""
    params = _params()
    frames = _codewords(2)
    # Frame 0 stays decodable: 3 symbols, 2 bits each.
    for j in range(3):
        frames[0][j] = int(frames[0][j]) ^ 0b101
    # Frame 1 is past the limit and carries no reference.
    for j in range(params.correctable + 4):
        frames[1][j] = (int(frames[1][j]) + 1) % (1 << PPM_RANK)

    report = check_frames(frames, params)
    assert report.n_decoded == 1 and report.n_uncorrectable == 1

    rates = report.rates()
    # Exact over the one frame RS could read, not diluted by the frame it could not.
    assert rates["wer_pre"] == pytest.approx(3 / params.n)
    assert rates["ber_pre"] == pytest.approx(6 / (params.n * PPM_RANK))
    assert report.frame_error_rate == pytest.approx(1 / 2)


def test_wrong_length_frames_are_skipped_not_counted():
    """A frame that lost a pulse is a framing failure, not a channel measurement."""
    frames = _codewords(3)
    frames[1] = frames[1][:-1]
    report = check_frames(frames, _params())
    assert report.n_frames == 2
    assert [f.index for f in report.frames] == [0, 2]
