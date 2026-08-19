import logging
from pathlib import Path

import numpy as np

from coding_synchronization.channel import ChannelParams
from coding_synchronization.encoder import FrameParams, ModulationParams, PassageParams
from coding_synchronization.Model import Model1


def test_model1_decodes_cleanly_with_relaxed_parameters(caplog):
    """Strict correctness check: with negligible jitter/offset/doppler, every simulated frame
    should sync with zero warnings AND the recovered payload should exactly match what was fed
    in — not just "the pipeline didn't crash". Saves to the repo's ./output, same as every other
    real run of this pipeline.
    """
    mod_params = ModulationParams(ppm_rank=10, slot_time=np.float64(1e-6), dead_slots=16)
    frame_params = FrameParams(sync_num=8, metadata_num=5, data_num=16, ecc_num=16, eof_num=8)
    # time_s caps the pass to a handful of milliseconds -> a small, fast n_frames, and short
    # enough that the physical Doppler drift never approaches the sync margin.
    overflight_params = PassageParams(altitude_km=500.0, max_elevation_deg=90.0, time_s=0.005)
    channel_params = ChannelParams(
        sigma=1e-4,  # timing jitter in slots — far below Syncer's margin (word_period // 8)
        vanish_rate=None,
        max_const_offset=20,
        added_rate=None,
        chirp_duration_s=float(mod_params.slot_time),
    )

    model = Model1(
        data=None,
        frame_params=frame_params,
        mod_params=mod_params,
        overflight_params=overflight_params,
        channel_params=channel_params,
        plot=False,
    )
    model.construct_pipeline()

    # PassageGen only decides how many frames to send (from the pass duration) once the
    # pipeline is constructed, and it silently truncates/tiles whatever data it's given to
    # exactly n_frames * data_num words — so build the known payload to that exact size.
    max_value = (1 << mod_params.ppm_rank) - 1
    total_words = model.n_frames * frame_params.data_num
    expected_data = (np.arange(total_words, dtype=np.uint16) % (max_value + 1)).astype(np.uint16)
    model.data = expected_data

    output_root = Path("./output")
    before = set(output_root.iterdir()) if output_root.exists() else set()

    caplog.set_level(logging.WARNING)
    model.run()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, f"unexpected warnings from an easy-to-decode run: {warnings}"

    decoded = model.decoded_frames
    assert len(decoded) == model.n_frames, (
        f"expected {model.n_frames} decoded frames, got {len(decoded)}"
    )
    # Each decoded frame is data_num payload words followed by ecc_num random filler words —
    # only the payload half is checked against what was actually sent.
    payload = np.concatenate(
        [np.asarray(frame)[: frame_params.data_num] for frame in decoded]
    )
    np.testing.assert_array_equal(payload, expected_data)

    new_dirs = set(output_root.iterdir()) - before
    assert len(new_dirs) == 1, f"expected exactly one new run directory, got {new_dirs}"
    out = new_dirs.pop()
    assert (out / "pipeline.txt").exists()
    assert (out / "params.json").exists()
    assert (out / "run.log").exists()
