from dataclasses import dataclass, field

from coding_synchronization.channel.AddedPulses import AddedPulses
from coding_synchronization.channel.ConstantOffset import ConstantOffset
from coding_synchronization.channel.DopplerShift import DopplerShift
from coding_synchronization.channel.RandomShift import RandomShift
from coding_synchronization.channel.VanishPulses import VanishPulses
from coding_synchronization.StageABC import CompoundStage, StageABC


@dataclass
class ChannelParams:
    sigma: float | None
    vanish_rate: float | None
    max_const_offset: int | None
    altitude_km: float | None
    added_rate: float | None
    chip_duration_s: float = field(default=20e-9)
    tca_chip: float | None = None


class Channel(CompoundStage):
    def __init__(self, params: ChannelParams, seed: int = 42) -> None:
        stages: list[StageABC] = []
        if params.vanish_rate is not None:
            stages.append(VanishPulses(rate=params.vanish_rate))
        if params.sigma is not None:
            stages.append(RandomShift(sigma=params.sigma))
        if params.altitude_km is not None:
            stages.append(
                DopplerShift(
                    altitude_km=params.altitude_km,
                    slot_time_s=params.chip_duration_s,
                    tca_slot=params.tca_chip,
                )
            )
        if params.max_const_offset is not None:
            stages.append(ConstantOffset(max_offset=params.max_const_offset))
        if params.added_rate is not None:
            stages.append(AddedPulses(rate=params.added_rate))
        super().__init__(stages=stages, seed=seed)
