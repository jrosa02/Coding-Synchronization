import abc

import matplotlib.pyplot as plt
import numpy as np

from coding_synchronization.channel import (
    AddedPulses,
    ChannelParams,
    ConstantOffset,
    DopplerShift,
    RandomShift,
    VanishPulses,
)
from coding_synchronization.decoder.Splitter import Splitter
from coding_synchronization.encoder import (
    FrameParams,
    ModulationParams,
    OverflightGen,
    OverflightParams,
)
from coding_synchronization.PlotStage import BarPlotStage, ListDisplayStage, PulsePlotStage
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
        overflight_params: OverflightParams,
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
        self._gen: OverflightGen | None = None

    def construct_pipeline(self) -> None:
        stage_names = [
            "OverflightGen",
            "VanishPulses",
            "RandomShift",
            "DopplerShift",
            "ConstantOffset",
            "AddedPulses",
            "Splitter",
        ]
        plot_axes = None
        list_axes = None
        vline_axes = None
        if self.plot:
            _, plot_axes = plt.subplots(len(stage_names), 1, figsize=(12, 18))
            _, list_axes = plt.subplots(len(stage_names), 1, figsize=(12, 18))
            _, vline_axes = plt.subplots(len(stage_names), 1, figsize=(12, 18))

        def maybe_plot(i: int) -> None:
            if plot_axes is not None:
                self.runner.append(BarPlotStage(ax=plot_axes[i], title=f"After {stage_names[i]}"))
            if list_axes is not None:
                self.runner.append(
                    ListDisplayStage(ax=list_axes[i], title=f"After {stage_names[i]}")
                )
            if vline_axes is not None:
                self.runner.append(
                    PulsePlotStage(ax=vline_axes[i], title=f"After {stage_names[i]}")
                )

        self._gen = OverflightGen(
            self.frame_params, self.mod_params, self.overflight_params, seed=self.seed
        )
        self.runner.append(self._gen)
        maybe_plot(0)
        cp = self.channel_params
        self.runner.append(VanishPulses(rate=cp.vanish_rate))
        maybe_plot(1)
        self.runner.append(RandomShift(sigma=cp.sigma))
        maybe_plot(2)
        self.runner.append(
            DopplerShift(
                altitude_km=cp.altitude_km, chip_duration_s=cp.chip_duration_s, tca_chip=cp.tca_chip
            )
        )
        maybe_plot(3)
        self.runner.append(ConstantOffset())
        maybe_plot(4)
        self.runner.append(AddedPulses(rate=cp.added_rate))
        maybe_plot(5)
        self.runner.append(Splitter(threshold=4096))
        maybe_plot(6)
        self.runner.append(Terminator())

    def run(self) -> None:
        assert self._gen is not None, "call construct_pipeline() before run()"
        self._gen.load(self.data)
        self.runner.run()

        if self.plot:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    mod_params = ModulationParams(ppm_rank=10, slot_time=np.float64(20e-9))
    frame_params = FrameParams(sync_num=4, metadata_num=4, data_num=4, ecc_num=4, eof_num=4)
    overflight_params = OverflightParams(time_s=0.005)
    channel_params = ChannelParams(
        sigma=0.5, vanish_rate=0.05, max_const_offset=1024, altitude_km=500.0, added_rate=0.02
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
