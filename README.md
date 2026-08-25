# coding-synchronization

Coding and synchronization layer of a pulse position modulation (PPM) laser downlink.

The transmitter sends frames of PPM words. Each word carries its value as the position of one pulse
inside the word. The receiver must find the frame boundaries, lock onto the sync section, recover
the slot time, and decode every word.

**This repository is a design tool.** It exists to choose the frame format and the timing
parameters before the hardware freezes them. It answers one question: which configuration carries
the most data, and still decodes after the impairments of a real low Earth orbit pass?
[`docs/why.md`](docs/why.md) states the decision metrics, and lists which parameters are frozen and
which are still open.

This repository holds two parts that share one decoder core.

## The two parts

**Modeling** simulates the link. It generates the frames of a satellite pass, adds timing noise,
Doppler shift and pulse loss, and decodes the result. Use it to predict the level of impairment
that a configuration survives. See [`docs/modeling.md`](docs/modeling.md).

**Measurement metrics** decodes a real oscilloscope capture. It finds the pulses in a CSV export,
calibrates the slot time, decodes the frames, and measures the error rates with the Reed-Solomon
parity. Use it to test the hardware against that prediction. See
[`docs/measurement.md`](docs/measurement.md).

Both parts build a pipeline from the same stages, and both use the same `Splitter`,
`SimpleSyncer`/`TwoPointSync`, `EccDecode` and `MetadataCheck`. See
[`docs/pipeline.md`](docs/pipeline.md).

## Install

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run the simulation

```bash
uv run main.py
```

`main.py` holds every parameter. Edit the file to change a run.

## Sync pulses vs. clock jitter

```bash
uv run s_scripts/simulate_wer_map_sync_jitter.py --no-show
```

`RandomShift` (`src/coding_synchronization/channel/RandomShift.py`) models clock jitter as one
independent Gaussian draw per pulse:

$$x' = x + n, \qquad n \sim \mathcal{N}(0, \sigma^{2})$$

`SimpleSyncer` (`docs/math.md#7-synchronization`) fits a scale $\hat{s}$ through the sync pulses by
least squares, and a residual scale error $\varepsilon$ in that fit drifts word $j$'s decoded
value by $\varepsilon W j$ slots, $W$ the word period. A word survives while that drift stays
under half a slot, so the last word of the frame is the first to fail:

$$|\varepsilon| < \varepsilon_{\max} = \frac{1}{2 W (L-1)}$$

with $L$ the frame length in words. `docs/math.md` leaves two bounds open at this point: the
margin a given $\sigma$ and $N_{\mathrm{sync}}$ actually buy against that limit (section 6), and
the probability that pass 1b of `SimpleSyncer` locates fewer than 3 sync pulses as a function of
$\sigma$, the pulse loss rate and $N_{\mathrm{sync}}$ (section 7) — neither has a closed form.
`s_scripts/simulate_wer_map_sync_jitter.py` answers both empirically instead of deriving them: it
sweeps $N_{\mathrm{sync}}$ and $\sigma$ on a 2D grid, runs `Model1` with the ECC decoder disabled
(`run_ecc=False`, since only the pre-ECC/sync question is asked and Reed-Solomon decode is the
most expensive stage per frame), and plots the resulting pre-ECC word error rate as a map. Each
grid cell is independent and runs in its own process.

## Tracking Doppler drift: SimpleSyncer vs. TwoPointSync

```bash
uv run s_scripts/simulate_doppler_scale_tracking.py --no-show
```

`SimpleSyncer` fits one flat scale per frame, which absorbs a *constant* Doppler rate but leaves
the curvature term $\varepsilon_{\mathrm{res}} \approx \ddot{\rho}T_f/(2c)$ open
(`docs/math.md#7-synchronization`, "Tolerance to Doppler shift"). `TwoPointSync` adds a second
step: it fits a line through the previous and current frame's own fitted points, and applies that
line's slope as a piecewise-linear scale correction across the frame instead of one flat value
(`docs/math.md#7-synchronization`, "Two-point tracking"). This script runs both variants on the
same simulated pass — Doppler shift plus a per-frame `ConstantOffset` only, no jitter — and plots
each one's fitted drift and residual against the analytic Doppler curve, so the curvature
correction `TwoPointSync` adds is visible directly.

## Decode a capture

```bash
uv run m_scripts/decode_measurement.py measurments/<capture>.Wfm.csv \
  --sample-rate 1e9 --combine sub --threshold 0.3 \
  --ppm-rank 10 --dead-slots 16 --sync-num 8 --sync-value 0 \
  --metadata-num 5 --data-num 16 --ecc-num 16 --eof-num 2 --check-ecc
```

The frame layout options must match the transmitter, and the command above matches the captures in
this repository. The current design candidate is a longer frame, with `--data-num 200` and
`--eof-num 64` at a 32 MHz slot clock. [`docs/why.md`](docs/why.md) lists both, and it names the
defaults that disagree with the candidate.

`measurments/legend.txt` records the options that work for each capture.
[`docs/workflow.md`](docs/workflow.md) gives the full procedure for a capture that is new to you.

Every run writes to `output/<timestamp>/`. That directory holds the log, the pipeline description,
the parameters, and the decoded words.

## Documentation

| File | Content |
|---|---|
| [`docs/why.md`](docs/why.md) | Why this repository exists, what a run tells you, and which parameters are still open |
| [`docs/math.md`](docs/math.md) | The equation that each stage computes, and the limits that follow |
| [`docs/pipeline.md`](docs/pipeline.md) | The stage framework and the shared decoder stages |
| [`docs/modeling.md`](docs/modeling.md) | The simulation, its channel impairments and its parameters |
| [`docs/measurement.md`](docs/measurement.md) | The measurement scripts, the slot time calibration and the ECC check |
| [`docs/workflow.md`](docs/workflow.md) | The procedure to decode a capture that is new to you |
| [`docs/glossary.md`](docs/glossary.md) | One term for one concept, and every symbol of `docs/math.md` |

## Tests

```bash
uv run pytest test
```

## Known gaps

- The metadata layout is provisional. One word counts the frames and the other four are spare.
- `DopplerShift` models a zenith pass past a flat Earth. That is the worst case for the receiver,
  and [`docs/math.md`](docs/math.md#3-channel) states the assumption. A pass at a lower elevation
  is easier, and the model does not describe it.
- Four bounds in [`docs/math.md`](docs/math.md) are still open. The largest one is the relation
  between the pre-ECC word error rate and the post-ECC frame error rate, which decides how many
  parity words a frame needs.
- The repository does not collect the results of past runs. Each run writes its own directory, and
  the comparison between configurations is manual.
