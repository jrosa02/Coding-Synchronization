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
    added_rate: float | None
    chirp_duration_s: float = field(default=20e-9)
    tca_chirp: float | None = None
    # Every other impairment switches off through its own None field. DopplerShift takes its
    # altitude from the passage, not from here, so it needs its own flag. The default keeps the
    # drift on. A sweep that varies the pass length must set this False, or the drift varies
    # with the swept parameter and hides the effect being measured.
    doppler: bool = True


class Channel(CompoundStage):
    def __init__(self, params: ChannelParams, altitude_km: float, seed: int = 42) -> None:
        stages: list[StageABC] = []
        if params.vanish_rate is not None:
            stages.append(VanishPulses(rate=params.vanish_rate))
        if params.sigma is not None:
            stages.append(RandomShift(sigma=params.sigma))
        if params.doppler:
            stages.append(
                DopplerShift(
                    altitude_km=altitude_km,
                    slot_time_s=params.chirp_duration_s,
                    tca_slot=params.tca_chirp,
                )
            )
        if params.max_const_offset is not None:
            stages.append(ConstantOffset(max_offset=params.max_const_offset))
        if params.added_rate is not None:
            stages.append(AddedPulses(rate=params.added_rate))
        super().__init__(stages=stages, seed=seed)
