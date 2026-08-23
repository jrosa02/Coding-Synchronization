from pathlib import Path

import numpy as np

from coding_synchronization.measurement.WaveformLoader import (
    WaveformParams,
    load_waveform,
    sniff_format,
)


def test_sniff_format_two_column_differential(tmp_path):
    path = tmp_path / "differential.csv"
    path.write_text("0.0057525691;0.28458497\n0.0037762846;0.28853756\n0.0041111111;0.29123456\n")

    fmt = sniff_format(path)

    assert fmt.delimiter == ";"
    assert fmt.header_lines == 0
    assert fmt.n_cols == 2
    assert fmt.value_cols == (0, 1)
    assert fmt.dt_s is None


def test_sniff_format_single_channel_with_header(tmp_path):
    path = tmp_path / "single_channel.csv"
    path.write_text(
        "CH4V,t0 =-2.000000e+00, tInc = 2.000000e-06,\n"
        "-5.573333e-04,,\n"
        "-9.866666e-05,,\n"
        "-2.346667e-04,,\n"
    )

    fmt = sniff_format(path)

    assert fmt.header_lines == 1
    assert fmt.n_cols == 1
    assert fmt.value_cols == (0,)
    assert fmt.dt_s == 2.000000e-06


def test_load_waveform_single_channel_synthesizes_zero_second_channel(tmp_path):
    path = tmp_path / "single_channel.csv"
    values = [-5.573333e-04, -9.866666e-05, -2.346667e-04, 4.053333e-04]
    path.write_text(
        "CH4V,t0 =-2.000000e+00, tInc = 2.000000e-06,\n"
        + "".join(f"{v:e},,\n" for v in values)
    )

    wf = load_waveform(WaveformParams(path=Path(path)))

    assert wf.n == len(values)
    assert wf.dt_s == 2.000000e-06
    np.testing.assert_allclose(wf.ch_a, values, rtol=1e-6)
    assert np.all(wf.ch_b == 0)
