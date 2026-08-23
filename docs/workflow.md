# Decode a new capture

This procedure takes an unknown capture and ends with decoded frames. Follow the steps in order.
Each step tells you what to check before you continue.

`docs/measurement.md` describes the scripts. `docs/glossary.md` defines the terms.

## Before you start

Collect the frame layout from the transmitter: `--ppm-rank`, `--dead-slots`, `--sync-num`,
`--sync-value`, `--metadata-num`, `--data-num`, `--ecc-num` and `--eof-num`. The scripts cannot
measure these values. A wrong layout gives wrong results without an error.

`measurments/legend.txt` records the options that already work for each capture in the repository.
Read it first.

## Steps

1. **Look at the waveform.**

   ```bash
   uv run visualize_measurement.py <capture.csv> --sample-rate 1e9 --combine sub
   ```

   Check the raw channels. One channel must hold the pulses. Check the combined signal. The pulses
   must point up, and the threshold line must sit between the noise and the pulse tops.

2. **Find a threshold.** Use the sweep when step 1 leaves you unsure.

   ```bash
   uv run extract_offsets.py <capture.csv> --sweep 0.05 0.5 10 --sample-rate 1e9 --combine sub
   ```

   Pick a threshold in the range where the pulse count stays stable. A count that changes with
   every step means the threshold sits inside the noise.

3. **Decode the frames.**

   ```bash
   uv run decode_measurement.py <capture.csv> --sample-rate 1e9 --combine sub --threshold 0.3 \
     --ppm-rank 10 --dead-slots 16 --sync-num 8 --sync-value 0 \
     --metadata-num 5 --data-num 16 --ecc-num 16 --eof-num 2
   ```

   Read the log line "Slot time from the data". A large ratio means `--slot-time` or
   `--sample-rate` is wrong, and the script corrected it.

4. **Check the frame lengths.** The log prints one pulse count for each chunk. Almost every count
   must equal `sync_num + metadata_num + data_num + ecc_num`. Counts of 1 to 5 pulses mean the
   `Splitter` cut inside the frames. Go back to step 2 and check `--eof-num`.

5. **Check the sync lock.** Each frame must report `8/8 pulses located` for `--sync-num 8`. The
   sync residual must stay far below 0.5 slots.

6. **Check the sync value.** Every sync word must decode to `--sync-value`. If the frames hold one
   word too few, and one pulse falls inside the last sync word, try the largest PPM value as the
   sync value.

7. **Verify the data.** Add `--check-ecc` to the command in step 3.

   ```bash
   uv run decode_measurement.py <capture.csv> ... --check-ecc
   ```

   Frames that decode with zero corrections confirm the whole chain: the framing, the sync value,
   and the slot time. The report gives the error rates before and after correction.

## When a step fails

| Symptom | Likely cause | Action |
|---|---|---|
| Few pulses, or none | The threshold is too high, or `--combine` is wrong | Repeat step 1 and step 2 |
| Chunks of 1 to 5 pulses | The `Splitter` cuts inside frames | Check `--eof-num`, then read the "Slot time from the data" line |
| One chunk only | The threshold never fires | Check `--eof-num` and the capture length |
| Sync fails on every frame | The frame layout is wrong | Check `--ppm-rank` and `--dead-slots` against the transmitter |
| One word too few per frame | The sync value is wrong | Try the largest PPM value, as step 6 describes |
| Uncorrectable frames only | The ECC parameters are wrong | Check `--rs-fcr`, `--rs-generator` and `--rs-prim` |

## Look closer

These scripts answer questions that the decode log cannot:

- `plot_frame_detail.py` draws the raw signal of one frame with the detected pulses and the ideal
  pulse positions. Use it when a frame holds the wrong number of pulses. Use `--sections sync
  metadata` to see individual pulses, because a whole frame is too wide.
- `plot_sync_eye.py` stacks every sync pulse into one eye diagram. The first panel shows the pulse
  shape. The second panel shows the timing error against the decode grid.
- `plot_sync_regression.py --all-frames` fits a line through the sync residuals. A tilted line means
  the slot time calibration is wrong. The annotation gives the error that accumulates over a whole
  frame, which is the number that decides whether the tilt matters.

## Record what worked

Add one line to `measurments/legend.txt` for each new capture. Record the file name, the working
options, and any value that you had to find by experiment. The next reader starts from that line.
