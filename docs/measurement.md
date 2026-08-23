# Measurement metrics

The measurement part reads an oscilloscope capture, finds the pulses, decodes the frames, and
measures the quality of the link. `Model2` in `src/coding_synchronization/Model.py` builds the
pipeline. Six scripts in the repository root drive it.

[`docs/workflow.md`](workflow.md) gives the procedure for a new capture. This page describes the
parts. [`docs/math.md`](math.md) states what each of them computes, and [`docs/why.md`](why.md)
states which decision the numbers support.

## From a CSV file to decoded words

The steps below run inside every measurement script.

1. **`WaveformLoader.load_waveform`** reads the capture. `sniff_format` reads the first 40 lines
   and works out the delimiter, the number of header lines, and the role of each column. A column
   that increases by a constant step is the time axis. Give `--sample-rate` when the file holds no
   time column. `--max-samples` loads only the first N samples.

   The loader accepts two forms. A two-column file holds a differential pair. A single-column file
   holds one channel. The loader then reads the sample period from a `tInc=` header line, and it
   fills the second channel with zeros.
2. **`OffsetExtractor.differential`** removes the DC pedestal of each channel and merges the pair.
   The measuring device inverts one leg, so the default `--combine add` restores the pulse.
   `--polarity auto` flips the result when the pulses point down.
3. **`OffsetExtractor.extract_offsets`** finds every run above the threshold and computes one
   position for each run. `--edge` selects the method: `centroid`, `peak`, `edge` or `rising`.
   Hysteresis merges the ringing after a pulse into the pulse itself.
4. **`Cli.extract_calibrated`** measures the slot time from the pulses. The next section explains
   why.
5. **`Model2`** runs the decoder stages. [`docs/pipeline.md`](pipeline.md) describes them.

`Model2` adds two stages that the modeling part does not use. `MeasurementGen` feeds the measured
offsets into the pipeline as a source. `FrameFilter` drops the first chunk, because the
oscilloscope trigger lands inside a frame and the capture starts with a fragment.
`--drop-partial-frames` also drops every chunk with the wrong pulse count.

## Why the scripts measure the slot time

The `Splitter` threshold is `eof_num * word_period` **slots**. The pulse positions arrive in
samples. The slot time converts between them, and `--slot-time` or `--samples-per-slot` states it.

A wrong slot time moves the threshold in samples:

- A threshold that is too small cuts inside frames. Each chunk then holds a part of a frame, the
  first words of the chunk are not the sync section, and every later result is wrong.
- A threshold that is too large never fires. The whole capture becomes one chunk.

Neither failure raises an error. Both produce well-formed output, so a reader can trust a plot that
means nothing.

`Cli.extract_calibrated` removes the risk. Every word carries exactly one pulse, so the typical gap
between two pulses is one word period, whatever values the transmitter sends. The median gap
divided by the word period therefore measures the slot time directly. The function extracts the
pulses, measures the slot time, and extracts again when the merge distance changes. It logs the
measured value against the given value, and it raises a warning when the two differ by more than
two percent.

Use `--no-auto-slot` to keep the given value. The results then depend on `--sample-rate` again.

## The scripts

| Script | Question it answers | Files it writes |
|---|---|---|
| `visualize_measurement.py` | What does this capture hold, and is the threshold right? | `measurement_overview.png` |
| `extract_offsets.py` | Where is every pulse? | `offsets.npz`, `offsets.csv`, `offsets.png` |
| `decode_measurement.py` | What data did the frames carry, and did it arrive intact? | `decoded.npz`, `decoded.txt`, `ecc_report.xml`, `frame_sections.png` |
| `plot_frame_detail.py` | Does a detected pulse exist for every word of one frame? | `frame_detail.png` |
| `plot_sync_eye.py` | How consistent are the pulse shape and the pulse timing? | `sync_eye.png` |
| `plot_sync_regression.py` | Does the sync section show a timing error that grows across the frame? | `sync_regression.png`, `sync_regression.xml` |

Every script shares the waveform, extraction and modulation options. `Cli.py` defines them, so a
command that works for one script works for the others.

`extract_offsets.py --sweep LOW HIGH N` prints the pulse count and the frame count for N thresholds.
Use it when you do not know a good threshold.

## The Reed-Solomon check

`decode_measurement.py --check-ecc` decodes each frame as a Reed-Solomon codeword. The metadata
words and the data words carry the information. The ECC words carry the parity. One word is one
symbol of `GF(2^ppm_rank)`.

The transmitted data is not available, so the check uses the corrected codeword as the reference.
Every symbol that Reed-Solomon changes is a symbol that arrived wrong. The report gives the word
error rate and the bit error rate, before and after correction. It also gives the frame error rate,
which is the share of frames that Reed-Solomon could not correct.

Two rules keep the numbers honest:

- A frame that exceeds the correction limit counts as a lost frame. Every one of its words counts
  as an error after correction. Before correction it counts the smallest number of errors that can
  make a frame uncorrectable, so that figure is a lower bound.
- A chunk with the wrong number of words never reaches the check. It lost or gained a pulse before
  the ECC saw it, which is a framing failure and not a measurement of the channel.

`--rs-fcr`, `--rs-generator` and `--rs-prim` set the code parameters. The default primitive
polynomial comes from `--ppm-rank`. `measurments/legend.txt` records the parameters of each capture.

The transmitter follows the Reed-Solomon code of the CCSDS Orange Book that the mission adopted.
The captures confirm the parameters: every full frame decodes as a valid codeword with no
correction. [`docs/math.md`](math.md#8-error-correction) gives the code and the rate definitions.

`--check-metadata` verifies that the metadata words of each frame form a consecutive counter, and
`--strict-metadata` stops the run at the first mismatch. Both run after the ECC decoding, so they
read corrected words. The current transmitter sends one frame counter and four spare words. That
layout is provisional, so a mismatch against the counter rule is expected today.
