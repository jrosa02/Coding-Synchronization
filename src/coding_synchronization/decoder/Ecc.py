"""Reed-Solomon integrity check over decoded frames, with error rates either side of it.

A frame is one RS codeword: the metadata and data words are the information symbols, the ecc
words are the parity, and one PPM word is one GF(2^ppm_rank) symbol — so RS(metadata+data+ecc,
metadata+data) with ppm_rank-bit symbols, correcting up to ecc_num/2 symbol errors.

There is no transmitted copy of the payload to compare against, so the *corrected* codeword is
the reference: every symbol RS had to change is one that arrived wrong. That is exact as long as
RS decodes. Beyond its correction limit RS itself has nothing to compare against either, but the
syndrome it already computes before attempting correction — specifically the Berlekamp-Massey
error-locator degree — still gives a real estimate of how many symbols are bad, so pre-ECC rates
are reported for every frame, not only the ones RS could correct. A frame it cannot correct is
still counted as a frame error at the post-ECC level, because post-ECC has nothing usable to
deliver regardless of how many symbols were actually wrong.
"""

import itertools
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
    syndrome_symbol_errors: int = 0  # BM error-locator degree; exact when ok, an estimate when not
    syndrome_bit_errors: int = 0     # bit_errors when ok, else syndrome_symbol_errors * ppm_rank


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

    @property
    def n_decoded(self) -> int:
        """Frames RS read successfully. Only these have a measurable error count."""
        return sum(1 for f in self.frames if f.ok)

    def rates(self) -> dict[str, float]:
        """Error rates either side of ECC decoding, over the whole frame population.

        **The pre-ECC rates use a syndrome-based estimate, uniformly.** For a frame RS decoded,
        the corrected codeword is an exact reference, so every symbol RS changed is a symbol that
        arrived wrong — `syndrome_symbol_errors` equals that exact count. For a frame RS could not
        decode there is no reference at all, but the Berlekamp-Massey error-locator degree
        computed from the syndrome (see `_syndrome_error_estimate`) still gives a real, always-
        available estimate of how many symbols are bad, not just a bound. It is exact up to
        `correctable` errors and an estimate beyond it — BM can alias to a different, sometimes
        smaller, degree once the true error count is large. Each bad symbol is charged the full
        `bits_per_symbol` bits for `ber_pre`, matching the convention used everywhere else in this
        report for a quantity that is not exactly known (a wrong-length frame, a lost frame's
        post-ECC contribution): charge the whole word rather than an expected-value fraction of it.

        **The post-ECC rates equal `frame_error_rate` by construction.** RS repairs every frame it
        decodes, so a decoded frame contributes no residual error. An uncorrectable frame delivers
        nothing usable, so it is charged in full. Charging whole frames makes the symbol rate and
        the bit rate scale together, and the `bits_per_symbol` factor cancels. The three names
        therefore carry one number. They are kept because callers read them, and because without a
        reference copy of the payload no finer post-ECC measurement exists.
        """
        total_symbols = self.n_frames * self.symbols_per_frame
        if total_symbols == 0:
            return dict.fromkeys(("wer_pre", "ber_pre", "wer_post", "ber_post"), 0.0)

        total_bits = total_symbols * self.bits_per_symbol
        pre_symbols = sum(f.syndrome_symbol_errors for f in self.frames)
        pre_bits = sum(f.syndrome_bit_errors for f in self.frames)
        post_symbols = self.n_uncorrectable * self.symbols_per_frame
        return {
            "wer_pre": pre_symbols / total_symbols,
            "ber_pre": pre_bits / total_bits,
            "wer_post": post_symbols / total_symbols,
            "ber_post": post_symbols / total_symbols,
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


def _syndrome_error_estimate(codec: "reedsolo.RSCodec", params: EccParams, values: list[int]) -> int:
    """Berlekamp-Massey error-locator degree, computed from the syndrome alone.

    This is `reedsolo.rs_find_error_locator`'s own algorithm, with its final Singleton-bound
    check (`raise ReedSolomonError("Too many errors to correct")`) removed, so a frame beyond
    `params.correctable` still returns a number instead of nothing. It is exact up to
    `correctable` errors — `codec.decode()` would succeed at that point and report the same
    count — and an estimate beyond it: BM can alias to a different, sometimes smaller, degree
    once the true error count is large. See `EccReport.rates()` for how the estimate is used.

    No erasures are ever passed to `codec.decode()` in this codebase, so the Forney syndrome
    reduces to the plain syndrome with its leading (always-zero) coefficient dropped.
    """
    reedsolo.gf_log, reedsolo.gf_exp, reedsolo.field_charac = (
        codec.gf_log, codec.gf_exp, codec.field_charac
    )
    synd = reedsolo.rs_calc_syndromes(values, params.ecc_num, params.fcr, params.generator)
    if max(synd) == 0:
        return 0
    fsynd = synd[1:]
    # reedsolo picks its internal array type (plain bytearray, or array('i', ...) for a field
    # wider than GF(2^8)) as a side effect of constructing the codec, and keeps it in the
    # module-global `_bytearray`. Use it here too, rather than plain lists, so it stays
    # interoperable with gf_poly_add/gf_poly_scale exactly like reedsolo's own BM loop does.
    err_loc = reedsolo._bytearray([1])
    old_loc = reedsolo._bytearray([1])
    for i in range(params.ecc_num):
        delta = fsynd[i]
        for j in range(1, len(err_loc)):
            delta ^= reedsolo.gf_mul(err_loc[-(j + 1)], fsynd[i - j])
        old_loc = old_loc + reedsolo._bytearray([0])
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = reedsolo.gf_poly_scale(old_loc, delta)
                old_loc = reedsolo.gf_poly_scale(err_loc, reedsolo.gf_inverse(delta))
                err_loc = new_loc
            err_loc = reedsolo.gf_poly_add(err_loc, reedsolo.gf_poly_scale(old_loc, delta))
    err_loc = list(itertools.dropwhile(lambda x: x == 0, err_loc))
    return min(len(err_loc) - 1, params.n)


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
        estimate = _syndrome_error_estimate(codec, params, values)
        return values, FrameResult(
            index=index, ok=False, syndrome_symbol_errors=estimate,
            syndrome_bit_errors=estimate * params.ppm_rank,
        )

    pairs = enumerate(zip(values, corrected, strict=True))
    bad = [j for j, (got, want) in pairs if got != want]
    bits = sum(int(values[j] ^ corrected[j]).bit_count() for j in bad)
    if bad:
        logger.info("Frame %d: RS corrected %d symbols at %s", index, len(bad), bad)
    return corrected, FrameResult(
        index=index, ok=True, symbol_errors=len(bad), bit_errors=bits, positions=bad,
        syndrome_symbol_errors=len(bad), syndrome_bit_errors=bits,
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
