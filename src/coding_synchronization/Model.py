import abc
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from coding_synchronization.channel import (
    AddedPulses,
    ChannelParams,
    ConstantOffset,
    DopplerShift,
    RandomShift,
    VanishPulses,
)
from coding_synchronization.decoder.Metadata import MetadataCheck
from coding_synchronization.decoder.Splitter import Splitter
from coding_synchronization.decoder.Syncer import Syncer
from coding_synchronization.encoder import (
    FrameParams,
    ModulationParams,
    PassageGen,
    PassageParams,
)
from coding_synchronization.PlotStage import PlotInput, PlotStage
from coding_synchronization.StageABC import StageRunner, Terminator


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
    ) -> None:
        super().__init__(seed=seed)
        self.data = data
        self.frame_params = frame_params
        self.mod_params = mod_params
        self.overflight_params = overflight_params
        self.channel_params = channel_params
        self.plot = plot
        self._gen: PassageGen | None = None

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
        cp = self.channel_params
        # self.runner.append(VanishPulses(rate=cp.vanish_rate))
        # maybe_plot(1)
        self.runner.append(RandomShift(sigma=cp.sigma))
        maybe_plot(2)
        self.runner.append(
            DopplerShift(
                altitude_km=cp.altitude_km, slot_time_s=cp.chip_duration_s, tca_slot=cp.tca_chip
            )
        )
        maybe_plot(3)
        self.runner.append(ConstantOffset())
        maybe_plot(4)
        # self.runner.append(AddedPulses(rate=cp.added_rate))
        # maybe_plot(5)
        self.runner.append(Splitter(threshold=3072))
        maybe_plot(6)
        self.runner.append(Syncer(self.mod_params, self.frame_params.sync_num))
        maybe_plot(7)
        self.runner.append(MetadataCheck())
        maybe_plot(8)
        self.runner.append(Terminator())

    def run(self) -> None:
        assert self._gen is not None, "call construct_pipeline() before run()"
        self._gen.load(self.data)
        self.runner.run()

        output_dir = Path("output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, fig in enumerate(self._figures):
            fig.tight_layout()
            fig.savefig(output_dir / f"figure_{i}.png", dpi=150)

        params = {
            "frame_params": dataclasses.asdict(self.frame_params),
            "mod_params": dataclasses.asdict(self.mod_params),
            "overflight_params": dataclasses.asdict(self.overflight_params),
            "channel_params": dataclasses.asdict(self.channel_params),
            "seed": self.seed,
        }
        (output_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
        print(f"Results saved → {output_dir}")

        if self.plot:
            plt.show()


if __name__ == "__main__":
    mod_params = ModulationParams(ppm_rank=10, slot_time=np.float64(20e-9), dead_slots=8)
    frame_params = FrameParams(sync_num=4, metadata_num=4, data_num=4, ecc_num=4, eof_num=4)
    overflight_params = PassageParams(altitude_km=1500.0, max_elevation_deg=60.0)
    channel_params = ChannelParams(
        sigma=0.1, vanish_rate=0.005, max_const_offset=1024, altitude_km=1500.0, added_rate=0.005
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
    print("pipeline:", model.runner)
    model.run()
