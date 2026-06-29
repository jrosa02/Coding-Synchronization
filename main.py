import numpy as np

from coding_synchronization.channel import ChannelParams
from coding_synchronization.encoder import FrameParams, ModulationParams, PassageParams
from coding_synchronization.Model import Model1

mod_params = ModulationParams(ppm_rank=10, slot_time=np.float64(64e-9), dead_slots=8)
frame_params = FrameParams(sync_num=8, metadata_num=4, data_num=240, ecc_num=4, eof_num=64)
overflight_params = PassageParams(altitude_km=1500.0, max_elevation_deg=60.0)
channel_params = ChannelParams(
    sigma=0.1, vanish_rate=0.005, max_const_offset=1024,
    altitude_km=1500.0, added_rate=0.005,
)
data = np.random.randint(0, 1 << mod_params.ppm_rank, 8, dtype=np.uint16)

model = Model1(
    data=data,
    frame_params=frame_params,
    mod_params=mod_params,
    overflight_params=overflight_params,
    channel_params=channel_params,
    plot=True,
)
model.construct_pipeline()
model.run()
