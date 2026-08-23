"""Reed-Solomon integrity check over decoded frames, with error rates either side of it.

A frame is one RS codeword: the metadata and data words are the information symbols, the ecc
words are the parity, and one PPM word is one GF(2^ppm_rank) symbol — so RS(metadata+data+ecc,
metadata+data) with ppm_rank-bit symbols, correcting up to ecc_num/2 symbol errors.

There is no transmitted copy of the payload to compare against, so the *corrected* codeword is
the reference: every symbol RS had to change is one that arrived wrong. That measures the channel
exactly as long as RS decodes, and RS decoding is itself the check — a frame it cannot correct is
counted as a frame error rather than folded into the symbol counts, because with more than
ecc_num/2 bad symbols the true error count is unknowable (and a mis-correction, which RS can also
produce beyond its limit, would understate it).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import reedsolo

from coding_synchronization.StageABC import StageABC

logger = logging.getLogger(__name__)


@dataclass
class EccParams:
    """RS parameters. `prim` defaults to the primitive polynomial reedsolo derives for the field."""

    ppm_rank: int
    ecc_num: int
    info_num: int  # metadata_num + data_num
    fcr: int = 0
    generator: int = 2
    prim: int | None = None

    @property
    def n(self) -> int:
        return self.info_num + self.ecc_num

    @property
    def correctable(self) -> int:
        return self.ecc_num // 2


@dataclass
class FrameResult:
    index: int
    ok: bool                       # RS decoded the frame (correctable or already clean)
    symbol_errors: int = 0         # symbols RS had to change; only meaningful when ok
    bit_errors: int = 0
    positions: list[int] = field(default_factory=list)


@dataclass
class EccReport:
    params: EccParams
    frames: list[FrameResult]
    symbols_per_frame: int
    bits_per_symbol: int

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_uncorrectable(self) -> int:
        return sum(1 for f in self.frames if not f.ok)

    @property
    def frame_error_rate(self) -> float:
        """Frames RS could not correct — the rate that survives ECC at frame level."""
        return self.n_uncorrectable / self.n_frames if self.n_frames else 0.0

    def rates(self) -> dict[str, float]:
        """Word (symbol) and bit error rates either side of ECC decoding.

        Post-ECC counts every symbol of an uncorrectable frame as an error: RS delivers nothing
        usable for such a frame, so charging it in full is the honest accounting.
        """
        total_symbols = self.n_frames * self.symbols_per_frame
        total_bits = total_symbols * self.bits_per_symbol
        if total_symbols == 0:
            return dict.fromkeys(("wer_pre", "ber_pre", "wer_post", "ber_post"), 0.0)

        pre_symbols = sum(f.symbol_errors for f in self.frames if f.ok)
        pre_bits = sum(f.bit_errors for f in self.frames if f.ok)
        # An uncorrectable frame's own error count is unknown, so the pre-ECC figures charge it
        # the least it can possibly hold: one more bad symbol than RS can correct, and — since a
        # wrong symbol differs in at least one bit — that same number of bad bits. Both pre-ECC
        # rates are therefore lower bounds whenever any frame is uncorrectable.
        pre_symbols += self.n_uncorrectable * (self.params.correctable + 1)
        pre_bits += self.n_uncorrectable * (self.params.correctable + 1)
        post_symbols = self.n_uncorrectable * self.symbols_per_frame
        return {
            "wer_pre": pre_symbols / total_symbols,
            "ber_pre": pre_bits / total_bits,
            "wer_post": post_symbols / total_symbols,
            "ber_post": post_symbols * self.bits_per_symbol / total_bits,
        }


def make_codec(params: EccParams) -> tuple["reedsolo.RSCodec", int]:
    """Build the reedsolo codec for these parameters, returning it with the primitive poly used."""
    prim = params.prim
    if prim is None:
        prim = int(reedsolo.find_prime_polys(
            generator=params.generator, c_exp=params.ppm_rank, fast_primes=False, single=True
        ))
    return reedsolo.RSCodec(
        params.ecc_num,
        nsize=(1 << params.ppm_rank) - 1,
        fcr=params.fcr,
        prim=prim,
        generator=params.generator,
        c_exp=params.ppm_rank,
    ), prim


def decode_frame(
    codec, params: EccParams, values: list[int], index: int
) -> tuple[list[int], FrameResult | None]:
    """RS-decode one frame. Returns the corrected words and the result, or None when skipped.

    A frame of the wrong length comes back unchanged with no result: it lost or gained a pulse
    before the ECC ever saw it, which is a framing failure and not a measurement of the channel.
    An uncorrectable frame also comes back unchanged, but it is recorded, because a lost frame is
    a channel outcome.
    """
    if len(values) != params.n:
        logger.warning(
            "Frame %d has %d words, not the %d an RS codeword needs — skipped",
            index, len(values), params.n,
        )
        return values, None
    try:
        corrected = list(codec.decode(values)[1])
    except reedsolo.ReedSolomonError:
        logger.warning("Frame %d is uncorrectable (> %d bad symbols)", index, params.correctable)
        return values, FrameResult(index=index, ok=False)

    pairs = enumerate(zip(values, corrected, strict=True))
    bad = [j for j, (got, want) in pairs if got != want]
    bits = sum(int(values[j] ^ corrected[j]).bit_count() for j in bad)
    if bad:
        logger.info("Frame %d: RS corrected %d symbols at %s", index, len(bad), bad)
    return corrected, FrameResult(
        index=index, ok=True, symbol_errors=len(bad), bit_errors=bits, positions=bad
    )


def log_code(params: EccParams, prim: int) -> None:
    logger.info(
        "RS(%d,%d) over GF(2^%d) (prim=%#x, fcr=%d, generator=%d), correcting up to %d symbols "
        "per frame",
        params.n, params.info_num, params.ppm_rank, prim, params.fcr, params.generator,
        params.correctable,
    )


def check_frames(frames: list[np.ndarray], params: EccParams) -> EccReport:
    """RS-decode every full-length frame and count what had to be corrected.

    This reports on frames that are already decoded. `EccDecode` does the same work inside a
    pipeline, and it hands the corrected words to the next stage.
    """
    codec, prim = make_codec(params)
    log_code(params, prim)

    results: list[FrameResult] = []
    for i, frame in enumerate(frames):
        _, result = decode_frame(codec, params, [int(v) for v in np.asarray(frame)], i)
        if result is not None:
            results.append(result)

    return EccReport(
        params=params, frames=results, symbols_per_frame=params.n, bits_per_symbol=params.ppm_rank
    )


class EccDecode(StageABC):
    """Correct each frame with its Reed-Solomon parity, before anything reads the words.

    This is the first step of the decode path. Every later stage therefore sees corrected words,
    so a metadata check tests the corrected stream and not the raw one.

    The stage keeps one FrameResult per frame, and `report` turns them into the error rates.
    """

    def __init__(self, params: EccParams, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.params = params
        self.codec, prim = make_codec(params)
        self.results: list[FrameResult] = []
        log_code(params, prim)

    def process(self, signal: Any) -> Any:
        out = []
        for i, frame in enumerate(signal):
            values = [int(v) for v in np.asarray(frame)]
            corrected, result = decode_frame(self.codec, self.params, values, i)
            if result is not None:
                self.results.append(result)
            out.append(np.asarray(corrected, dtype=np.uint16))
        logger.debug("EccDecode: %d frames corrected where needed", len(out))
        return np.asanyarray(out, dtype=object)

    def report(self) -> EccReport:
        return EccReport(
            params=self.params, frames=self.results, symbols_per_frame=self.params.n,
            bits_per_symbol=self.params.ppm_rank,
        )

    def reset(self) -> None:
        super().reset()
        self.results = []

    def __repr__(self) -> str:
        return f"EccDecode(n={self.params.n}, k={self.params.info_num})"
