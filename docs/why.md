# Why this repository exists

This repository is a design tool. It exists to choose the frame format and the timing parameters of
a pulse position modulation (PPM) laser downlink, before the hardware design freezes them.

The question it answers is this: **which configuration carries the most data, and still decodes
after the impairments of a real low Earth orbit pass?**

The two parts answer that question from two directions. The simulation in
[`docs/modeling.md`](modeling.md) predicts how a configuration behaves under an impairment that you
choose and control. The measurement part in [`docs/measurement.md`](measurement.md) decodes what the
hardware actually sent, so a prediction meets a measurement. A configuration that works only in the
simulation is not a result.

## What is frozen, and what is not

| Parameter | State | Value today |
|---|---|---|
| `ppm_rank` | frozen | 10. One word therefore carries 10 bits and 1024 possible values. |
| `dead_slots` | frozen | 16 |
| `sync_num` | open | 8 |
| `metadata_num` | open | 5 |
| `data_num` | open | 200 |
| `ecc_num` | open | 16 |
| `eof_num` | open | 64, held near one quarter of the frame length |
| slot clock | open | 32 MHz is the target. The captures in `measurments/` used 12 MHz as a test setting. |

The open parameters are the subject of the work. Every one of them can move, and these tests decide
where it moves to. Read the last column as the current candidate, and not as the design.

### The code defaults are older than the candidate

`main.py` and the command line defaults in `src/coding_synchronization/measurement/Cli.py` still
carry an earlier set. They disagree with the candidate above in four places:

| Parameter | Candidate | `main.py` and `Cli.py` default |
|---|---|---|
| `dead_slots` | 16, frozen | 8 |
| `metadata_num` | 5 | 4 |
| `data_num` | 200 | 240 |
| `ecc_num` | 16 | 4 |

State every frame parameter on the command line, and do not rely on a default. `params.json` in
each output directory records what a run actually used.

## What decides that one configuration beats another

Two metrics decide it.

**Payload data rate, at a post-ECC frame error rate near zero.** A configuration passes when
Reed-Solomon corrects almost every frame. Above that limit the rate is what counts.
[`docs/math.md`](math.md) gives the rate as a formula of the frame parameters. `data_num`,
`ecc_num` and `eof_num` trade against each other directly, because every word that carries no
payload still costs a whole word period.

**Tolerance to Doppler shift and to slot clock error.** The receiver recovers the slot time from
the sync section of each frame. That recovery has a limit. Beyond the limit the words at the end of
a frame decode one slot out. [`docs/math.md`](math.md#tolerance-to-a-slot-clock-error) derives the
limit from the word period and the frame length.

## What one run tells you

| Run | Output | The decision it supports |
|---|---|---|
| `main.py` with a chosen `ChannelParams` | The ECC report and the metadata report of `Model1` | How much timing noise, pulse loss and false pulses this configuration survives |
| `main.py` with `sigma` raised step by step | The impairment level at which frames stop correcting | The margin that the configuration holds over the expected channel |
| `decode_measurement.py --check-ecc` | Word error rate, bit error rate, frame error rate | Whether the hardware meets the limit that the simulation predicted |
| `plot_sync_regression.py --all-frames` | The residual tilt, and the slots it accumulates over one frame | Whether the slot time recovery holds across the whole frame length |
| `plot_sync_eye.py` | The timing spread of every sync pulse | How much of the slot budget the transmitter jitter already spends |
| `extract_offsets.py --sweep` | The pulse count against the threshold | Whether the optical signal separates from the noise at all |

One run reports the metrics of one configuration. The comparison between configurations is yours to
make, because the repository does not collect past runs. Every run writes its own
`output/<timestamp>/` directory, and `params.json` in that directory records the configuration that
produced the numbers.

## The order of work

1. Choose a candidate configuration.
2. Run the simulation with the impairments that you expect. Read the error rates.
3. Program the hardware with the same configuration, and capture the output.
4. Decode the capture. Compare the measured rates against the predicted rates.
5. Change one parameter. Repeat from step 2.

The measurement part also finds what the simulation cannot: a wrong slot clock, a pulse that the
extractor never detected, and a pulse shape that no model describes.
[`docs/workflow.md`](workflow.md) gives the procedure for step 4.

## What this repository does not do

It does not model the optical channel. It holds no photon count, no receiver noise figure and no
link budget. The impairments act on pulse positions and not on optical power, so this repository
answers a timing question and not a power question. A link budget belongs beside it, and not inside
it.
