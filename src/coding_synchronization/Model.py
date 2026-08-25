import abc
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from coding_synchronization._logging import add_file_handler
from coding_synchronization.channel import (
    AddedPulses,
    ChannelParams,
    ConstantOffset,
    DopplerShift,
    RandomShift,
    VanishPulses,
)
from coding_synchronization.decoder.Ecc import EccDecode, EccParams, EccReport
from coding_synchronization.decoder.Metadata import MetadataCheck
from coding_synchronization.decoder.Splitter import Splitter
from coding_synchronization.decoder.Syncer import Syncer
from coding_synchronization.encoder import (
    FrameParams,
    ModulationParams,
    PassageGen,
    PassageParams,
)
from coding_synchronization.measurement.Collector import Collector
from coding_synchronization.measurement.FrameFilter import FrameFilter
from coding_synchronization.measurement.MeasurementGen import MeasurementGen
from coding_synchronization.measurement.OffsetExtractor import ExtractionParams
from coding_synchronization.measurement.WaveformLoader import WaveformParams
from coding_synchronization.PlotStage import PlotInput, PlotStage
from coding_synchronization.StageABC import StageRunner

logger = logging.getLogger(__name__)


class ModelResult:  # TODO: define result fields
    pass


class ModelABC(abc.ABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed
        self.runner = StageRunner(seed)

    @abc.abstractmethod
    def construct_pipeline(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> None:
        pass

    def reset(self) -> None:
        self.runner.reset()

    def _report_ecc(self, report: EccReport) -> None:
        """Log the RS verdict and the error rates either side of it."""
        rates = report.rates()
        clean = sum(1 for f in report.frames if f.ok and f.symbol_errors == 0)
        corrected = sum(1 for f in report.frames if f.ok and f.symbol_errors > 0)
        logger.info(
            "ECC: %d frames checked — %d clean, %d corrected, %d uncorrectable",
            report.n_frames, clean, corrected, report.n_uncorrectable,
        )
        logger.info(
            "ECC: before decoding (over the %d decoded frames) WER=%.3e BER=%.3e | "
            "frame error rate=%.3e",
            report.n_decoded, rates["wer_pre"], rates["ber_pre"], report.frame_error_rate,
        )
        if report.n_uncorrectable:
            logger.warning(
                "ECC: %d/%d frames exceeded the %d-symbol correction limit — they carry no "
                "reference, so they are outside the pre-ECC rates above and are counted only in "
                "the frame error rate", report.n_uncorrectable, report.n_frames,
                report.params.correctable,
            )

    def _report_metadata(self, stage: MetadataCheck | None) -> None:
        if stage is None or not stage.verify or stage.frames_checked == 0:
            return
        logger.info(
            "Metadata: %d frames checked, %d mismatches (rate=%.3e)",
            stage.frames_checked, stage.mismatches, stage.mismatch_rate,
        )


class Model1(ModelABC):
    def __init__(
        self,
        data: np.ndarray,
        frame_params: FrameParams,
        mod_params: ModulationParams,
        overflight_params: PassageParams,
        channel_params: ChannelParams,
        plot: bool = False,
        seed: int = 42,
        split_eof_num: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.data = data
        self.frame_params = frame_params
        self.mod_params = mod_params
        self.overflight_params = overflight_params
        self.channel_params = channel_params
        self.plot = plot
        # The Splitter threshold, in EOF words. None means "use frame_params.eof_num", which is
        # the transmitted gap width. The receiver does not need that number: the threshold only
        # has to sit between the largest gap inside a frame (word_period + max_value) and the
        # smallest gap between frames. Setting it apart lets a wide transmitted gap coexist with
        # a threshold that keeps a margin on both sides.
        self.split_eof_num = split_eof_num
        self._gen: PassageGen | None = None
        self._collector: Collector | None = None
        self._synced: Collector | None = None
        self._corrected: Collector | None = None
        self._ecc: EccDecode | None = None
        self._metadata: MetadataCheck | None = None
        self.ecc_report: EccReport | None = None

    def construct_pipeline(self) -> None:
        stage_names = [
            "OverflightGen",
            "VanishPulses",
            "RandomShift",
            "DopplerShift",
            "ConstantOffset",
            "AddedPulses",
            "Splitter",
            "Syncer",
            "MetadataCheck"
        ]
        self._figures: list[Figure] = []
        plot_axes = None
        list_axes = None
        vline_axes = None
        if self.plot:
            # fig, plot_axes = plt.subplots(len(stage_names), 1, figsize=(12, 18))
            # self._figures.append(fig)
            fig, list_axes = plt.subplots(len(stage_names), 1, figsize=(12, 18))
            self._figures.append(fig)
            # fig, vline_axes = plt.subplots(len(stage_names), 1, figsize=(12, 18))
            # self._figures.append(fig)

        def maybe_plot(i: int) -> None:
            if plot_axes is not None:
                self.runner.append(PlotStage(
                    PlotInput(ax=plot_axes[i], indxs=(0, 0)),
                    plot_type='bar', title=f"After {stage_names[i]}",
                ))
            if list_axes is not None:
                self.runner.append(PlotStage(
                    PlotInput(ax=list_axes[i], indxs=(0, 0)),
                    plot_type='table', title=f"After {stage_names[i]}",
                ))
            if vline_axes is not None:
                self.runner.append(PlotStage(
                    PlotInput(ax=vline_axes[i], indxs=(0, 0)),
                    plot_type='vlines', title=f"After {stage_names[i]}",
                ))

        self._gen = PassageGen(
            self.frame_params, self.mod_params, self.overflight_params, seed=self.seed
        )
        self.runner.append(self._gen)
        maybe_plot(0)
        # Each impairment runs only when its ChannelParams field is set. The stages raise on a
        # rate of None, which is why these lines used to be commented out. channel/Channel.py
        # gates the same way.
        cp = self.channel_params
        if cp.vanish_rate is not None:
            self.runner.append(VanishPulses(rate=cp.vanish_rate))
            maybe_plot(1)
        if cp.sigma is not None:
            self.runner.append(RandomShift(sigma=cp.sigma))
            maybe_plot(2)
        if cp.doppler:
            self.runner.append(
                DopplerShift(
                    altitude_km=self.overflight_params.altitude_km,
                    slot_time_s=cp.chirp_duration_s,
                    tca_slot=cp.tca_chirp,
                )
            )
            maybe_plot(3)
        if cp.max_const_offset is not None:
            self.runner.append(ConstantOffset(cp.max_const_offset))
            maybe_plot(4)
        if cp.added_rate is not None:
            self.runner.append(AddedPulses(rate=cp.added_rate))
            maybe_plot(5)
        word_period = (1 << self.mod_params.ppm_rank) + self.mod_params.dead_slots
        eof_for_split = (
            self.frame_params.eof_num if self.split_eof_num is None else self.split_eof_num
        )
        threshold = eof_for_split * word_period
        self.runner.append(Splitter(threshold))
        maybe_plot(6)
        self.runner.append(Syncer(self.mod_params, self.frame_params.sync_num))
        maybe_plot(7)
        # The words as received, before the ECC corrects them.
        self._synced = Collector("synced")
        self.runner.append(self._synced)
        fp = self.frame_params
        if fp.ecc_num > 0:
            # FrameGen writes real parity, so the ECC can always run here. It runs first, so the
            # metadata check below tests corrected words.
            self._ecc = EccDecode(
                EccParams(
                    ppm_rank=self.mod_params.ppm_rank,
                    ecc_num=fp.ecc_num,
                    info_num=fp.metadata_num + fp.data_num,
                )
            )
            self.runner.append(self._ecc)
            self._corrected = Collector("corrected")
            self.runner.append(self._corrected)
        # FrameGen writes the metadata counter, so the simulation can verify it.
        self._metadata = MetadataCheck(
            fp.metadata_num, verify=True, value_range=1 << self.mod_params.ppm_rank
        )
        self.runner.append(self._metadata)
        maybe_plot(8)
        self._collector = Collector("payload")
        self.runner.append(self._collector)

    def run(self, save_artifacts: bool = True) -> None:
        """Run the pipeline. `save_artifacts=False` skips every file this method writes.

        A sweep calls this once per point. Each call otherwise makes its own output directory and
        adds another log file handler, and the handlers are never removed — run N then writes to
        all N log files. The directory name has one-second resolution as well, so two fast runs
        share one directory and overwrite each other. The ECC report and the metadata report are
        attributes, so they survive either way.
        """
        assert self._gen is not None, "call construct_pipeline() before run()"

        output_dir: Path | None = None
        if save_artifacts:
            output_dir = Path("output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir.mkdir(parents=True, exist_ok=True)
            add_file_handler(output_dir / "run.log")
            logger.info("Run started, output directory: %s", output_dir)
            (output_dir / "pipeline.txt").write_text(repr(self.runner))
        logger.info("Pipeline: %s", repr(self.runner))

        self._gen.load(self.data)
        self.runner.run()

        if self._ecc is not None:
            self.ecc_report = self._ecc.report()
            self._report_ecc(self.ecc_report)
        self._report_metadata(self._metadata)

        if output_dir is not None:
            for i, fig in enumerate(self._figures):
                fig.tight_layout()
                fig.savefig(output_dir / f"figure_{i}.png", dpi=150)
                logger.debug("Saved figure_%d.png", i)

            params = {
                "frame_params": dataclasses.asdict(self.frame_params),
                "mod_params": dataclasses.asdict(self.mod_params),
                "overflight_params": dataclasses.asdict(self.overflight_params),
                "channel_params": dataclasses.asdict(self.channel_params),
                "seed": self.seed,
            }
            (output_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
            logger.info("Results saved to %s", output_dir)

        for fig in self._figures:
            plt.close(fig)

    @property
    def decoded_frames(self) -> list[np.ndarray]:
        """Payload + ECC words, with the metadata and sync words stripped by MetadataCheck."""
        return [] if self._collector is None else self._collector.frames

    @property
    def n_frames(self) -> int:
        """Frames PassageGen decided to emit, from the overflight pass duration."""
        return 0 if self._gen is None else self._gen.n_frames


class Model2(ModelABC):
    """Decode measured pulse offsets (slot units) through the existing decoder stages."""

    def __init__(
        self,
        offsets: np.ndarray,
        frame_params: FrameParams,
        mod_params: ModulationParams,
        extraction_params: ExtractionParams | None = None,
        waveform_params: WaveformParams | None = None,
        sync_value: int | None = None,
        drop_first_frame: bool = True,
        drop_wrong_length: bool = False,
        plot: bool = False,
        seed: int = 42,
        ecc_params: EccParams | None = None,
        verify_metadata: bool = False,
        strict_metadata: bool = False,
    ) -> None:
        super().__init__(seed=seed)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        # When set, an EccDecode stage corrects every frame before the metadata check reads it,
        # and the result is kept here.
        self.ecc_params = ecc_params
        self.ecc_report: EccReport | None = None
        self.verify_metadata = verify_metadata or strict_metadata
        self.strict_metadata = strict_metadata
        self.sync_value = sync_value
        self.drop_first_frame = drop_first_frame
        self.drop_wrong_length = drop_wrong_length
        self.frame_params = frame_params
        self.mod_params = mod_params
        self.extraction_params = extraction_params
        self.waveform_params = waveform_params
        self.plot = plot
        self._gen: MeasurementGen | None = None
        self._collector: Collector | None = None
        self._synced: Collector | None = None
        self._corrected: Collector | None = None
        self._split: Collector | None = None
        self._syncer: Syncer | None = None
        self._ecc: EccDecode | None = None
        self._metadata: MetadataCheck | None = None
        self.output_dir: Path | None = None

    def construct_pipeline(self) -> None:
        stage_names = ["MeasurementGen", "Splitter", "FrameFilter", "Syncer", "MetadataCheck"]
        self._figures: list[Figure] = []
        list_axes = None
        if self.plot:
            fig, list_axes = plt.subplots(len(stage_names), 1, figsize=(12, 12))
            self._figures.append(fig)

        def maybe_plot(i: int) -> None:
            if list_axes is not None:
                self.runner.append(PlotStage(
                    PlotInput(ax=list_axes[i], indxs=(0, 0)),
                    plot_type="table", title=f"After {stage_names[i]}",
                ))

        self._gen = MeasurementGen(self.offsets, seed=self.seed)
        self.runner.append(self._gen)
        maybe_plot(0)

        word_period = (1 << self.mod_params.ppm_rank) + self.mod_params.dead_slots
        threshold = self.frame_params.eof_num * word_period
        self.runner.append(Splitter(threshold))
        # tap the raw frames so wrong-length ones can be reported before they are decoded
        self._split = Collector("split")
        self.runner.append(self._split)
        fp = self.frame_params
        self.runner.append(FrameFilter(
            expected_pulses=fp.sync_num + fp.metadata_num + fp.data_num + fp.ecc_num,
            drop_first=self.drop_first_frame,
            drop_wrong_length=self.drop_wrong_length,
        ))
        maybe_plot(2)
        self._syncer = Syncer(
            self.mod_params, self.frame_params.sync_num, sync_value=self.sync_value
        )
        self.runner.append(self._syncer)
        # tap before MetadataCheck strips the metadata words, so every decoded PPM value is kept
        self._synced = Collector("synced")
        self.runner.append(self._synced)
        maybe_plot(3)
        if self.ecc_params is not None:
            # RS decoding is the first step of the decode path, so the metadata check below reads
            # corrected words. It stays opt-in: without the transmitter's RS parameters every
            # frame would read as uncorrectable.
            self._ecc = EccDecode(self.ecc_params)
            self.runner.append(self._ecc)
            self._corrected = Collector("corrected")
            self.runner.append(self._corrected)
        self._metadata = MetadataCheck(
            self.frame_params.metadata_num,
            verify=self.verify_metadata,
            strict=self.strict_metadata,
            value_range=1 << self.mod_params.ppm_rank,
        )
        self.runner.append(self._metadata)
        maybe_plot(4)
        self._collector = Collector("payload")
        self.runner.append(self._collector)

    def run(self) -> None:
        assert self._gen is not None, "call construct_pipeline() before run()"
        assert self._collector is not None

        output_dir = Path("output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        add_file_handler(output_dir / "run.log")
        logger.info("Run started, output directory: %s", output_dir)

        (output_dir / "pipeline.txt").write_text(repr(self.runner))
        logger.info("Pipeline: %s", repr(self.runner))

        self._gen.load(self.offsets)
        self.runner.run()

        def _flatten(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            lengths = np.array([len(f) for f in frames], dtype=np.int64)
            if not frames:
                return np.empty(0, dtype=np.uint16), lengths
            return np.concatenate([np.asarray(f, dtype=np.uint16) for f in frames]), lengths

        self._report_frame_lengths()

        frames = self._collector.frames
        symbols, lengths = _flatten(frames)
        all_frames = self._synced.frames if self._synced is not None else frames
        all_symbols, all_lengths = _flatten(all_frames)
        np.savez(
            output_dir / "decoded.npz",
            symbols=symbols,
            frame_lengths=lengths,
            symbols_with_metadata=all_symbols,
            frame_lengths_with_metadata=all_lengths,
        )
        self._write_symbol_dump(output_dir / "decoded.txt", all_frames)
        logger.info(
            "Decoded %d frames, %d payload symbols (%d including metadata)",
            len(frames), len(symbols), len(all_symbols),
        )

        if self._ecc is not None:
            self.ecc_report = self._ecc.report()
            self._report_ecc(self.ecc_report)
        self._report_metadata(self._metadata)

        for i, fig in enumerate(self._figures):
            fig.tight_layout()
            fig.savefig(output_dir / f"figure_{i}.png", dpi=150)
            logger.debug("Saved figure_%d.png", i)

        params = {
            "frame_params": dataclasses.asdict(self.frame_params),
            "mod_params": dataclasses.asdict(self.mod_params),
            "n_offsets": len(self.offsets),
            "sync_value": None if self._syncer is None else self._syncer.sync_value,
            "seed": self.seed,
        }
        if self.extraction_params is not None:
            params["extraction_params"] = dataclasses.asdict(self.extraction_params)
        if self.waveform_params is not None:
            params["waveform_params"] = dataclasses.asdict(self.waveform_params)
        (output_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
        logger.info("Results saved to %s", output_dir)

        for fig in self._figures:
            plt.close(fig)

    def _report_frame_lengths(self) -> None:
        """Every frame should hold exactly frame_len pulses; say so loudly when one does not."""
        if self._split is None:
            return
        fp = self.frame_params
        expected = fp.sync_num + fp.metadata_num + fp.data_num + fp.ecc_num
        counts = [len(f) for f in self._split.frames]
        bad = [(i, n) for i, n in enumerate(counts) if n != expected]
        logger.info("Splitter produced %d frames, expected %d pulses each", len(counts), expected)
        logger.info("Frame pulse counts: %s", counts)
        for i, n in bad:
            logger.warning(
                "Frame %d has %d pulses, expected %d (%+d)", i, n, expected, n - expected
            )
        if not bad:
            logger.info("All %d frames have exactly %d pulses", len(counts), expected)

    def _write_symbol_dump(self, path: Path, frames: list[np.ndarray]) -> None:
        meta_num = self.frame_params.metadata_num
        syncs = self.sync_frames
        lines = [f"# {len(frames)} frames, metadata_num={meta_num} (first {meta_num} per frame)"]
        for i, frame in enumerate(frames):
            values = np.asarray(frame).tolist()
            if i < len(syncs):
                lines.append(f"frame {i} sync ({len(syncs[i])} values): {syncs[i].tolist()}")
            lines.append(f"frame {i} ({len(values)} values): {values}")
        path.write_text("\n".join(lines) + "\n")

    @property
    def decoded_frames(self) -> list[np.ndarray]:
        """Payload + ECC words, with the metadata stripped by MetadataCheck."""
        return [] if self._collector is None else self._collector.frames

    @property
    def sync_frames(self) -> list[np.ndarray]:
        """Decoded sync-word values per frame — expected to read sync_value (default 0)."""
        return [] if self._syncer is None else list(self._syncer.sync_values)

    @property
    def sync_found(self) -> list[int]:
        """How many sync pulses were actually located per frame (rest were assumed)."""
        return [] if self._syncer is None else list(self._syncer.sync_found)

    @property
    def decoded_frames_with_metadata(self) -> list[np.ndarray]:
        """Every PPM value the Syncer recovered — metadata words included."""
        return [] if self._synced is None else self._synced.frames

    @property
    def frame_starts(self) -> list[float]:
        """Fitted frame_start (slot units) per successfully-synced frame."""
        return [] if self._syncer is None else list(self._syncer.frame_starts)

    @property
    def raw_synced_frames(self) -> list[np.ndarray]:
        """Slot-time-calibrated pulse positions (frame-local) the Syncer decoded, one per frame."""
        return [] if self._syncer is None else list(self._syncer.raw_frames)

    @property
    def word_period(self) -> float:
        return 0.0 if self._syncer is None else float(self._syncer.word_period)

    @property
    def inferred_slot_time_s(self) -> list[float]:
        """Auto-calibrated slot_time_s per synced frame, from that frame's sync section."""
        return [] if self._syncer is None else list(self._syncer.inferred_slot_time_s)

    @property
    def sync_residuals(self) -> list[np.ndarray]:
        """Un-rounded (sync_decoded - sync_value) per sync pulse, per frame — sub-slot timing error."""
        return [] if self._syncer is None else list(self._syncer.sync_residuals)

    @property
    def sync_gap_devs(self) -> list[np.ndarray]:
        """(consecutive sync-pulse gap - word_period) per frame — should be ~0 if calibration is good."""
        return [] if self._syncer is None else list(self._syncer.sync_gap_devs)


if __name__ == "__main__":
    mod_params = ModulationParams(ppm_rank=10, slot_time=np.float64(20e-9), dead_slots=8)
    frame_params = FrameParams(sync_num=4, metadata_num=4, data_num=4, ecc_num=4, eof_num=4)
    overflight_params = PassageParams(altitude_km=1500.0, max_elevation_deg=60.0)
    channel_params = ChannelParams(
        sigma=0.1, vanish_rate=0.005, max_const_offset=1024, added_rate=0.005
    )
    data = np.random.randint(0, 1 << mod_params.ppm_rank, 8, dtype=np.uint16)

    model = Model1(
        data=None,
        frame_params=frame_params,
        mod_params=mod_params,
        overflight_params=overflight_params,
        channel_params=channel_params,
        plot=True,
    )
    model.construct_pipeline()
    model.run()
