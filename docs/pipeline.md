# The shared pipeline

Both parts of this project build a pipeline from the same stages. This page describes the stage
framework and the decoder stages that both parts use.
[`docs/glossary.md`](glossary.md) defines the terms, and [`docs/math.md`](math.md) states what each
stage computes.

The decode path runs in this order in both models:

```
Splitter -> [FrameFilter] -> Syncer -> Collector("synced") -> EccDecode
    -> Collector("corrected") -> MetadataCheck -> Collector("payload")
```

`FrameFilter` belongs to the measurement part only. `EccDecode` and its collector appear only when
the model receives RS parameters.

## Stages and queues

`StageABC` in `src/coding_synchronization/StageABC.py` is the base class of every stage. A stage
holds an input queue, an output queue, or both. The queues are `asyncio.Queue` objects with a
maximum size of 2, so a slow stage stops the stage before it.

A stage takes one of three roles, and `StageABC.run` selects the role from the queues:

1. A source has an output queue only. It calls `generate` until `generate` returns `None`.
2. A pipe has both queues. It calls `process` on each item.
3. A sink has an input queue only. It calls `consume` on each item.

A stage passes `None` to its output queue to signal the end of the stream.

`StageRunner` builds the pipeline. `StageRunner.append` connects each new stage to the previous one
and gives it the shared seed. `StageRunner.run` runs every stage at the same time with
`asyncio.gather`.

`CompoundStage` holds a list of stages and presents them as one stage. The `Channel` class uses it.

## Decoder stages

The stages below live in `src/coding_synchronization/decoder/`. Both `Model1` and `Model2` use
them.

### Splitter

`Splitter` cuts the offset stream into chunks. It cuts at every gap larger than its threshold. The
threshold is `eof_num * word_period` slots, because the EOF gap is the only gap that large.

The `Splitter` zeroes each chunk. It subtracts the position of the first pulse from every pulse in
that chunk. Every position after this stage is frame-local, not absolute against the capture.

A wrong slot time changes the threshold in samples. The `Splitter` then cuts inside frames, and
every later stage reads chunks that are not frames. `docs/measurement.md` describes the guard
against this.

### Syncer

`Syncer` locates the sync section, calibrates the slot time, and decodes every word. It works in
two passes.

Pass 1 calibrates the scale:

1. The median gap between the first `sync_num` pulses gives a coarse scale. The median survives a
   missing sync pulse, because one missing pulse only doubles one gap.
2. `_locate_sync` finds each sync pulse within `margin` slots of its expected position. The margin
   is one eighth of the word period.
3. `_refine_scale` fits a least-squares line through the located pulses against their word indices.
   The slope gives the final scale. The `Syncer` keeps the coarse scale in two cases: fewer
   than three pulses were located, or the new scale differs from the coarse one by more than one
   percent.

The median alone is not enough for pass 1. The median of a few noisy gaps is not the least-squares
slope, and the difference tilts the residual. That tilt accumulates over the frame. A scale error of a
few parts per million therefore becomes a fraction of a slot at the last word.

Pass 2 decodes the words. `_decode_positions` divides each position by the word period. The integer
part gives the word index and the remainder gives the PPM value. A pulse that lands in the dead
zone is ambiguous, so the `Syncer` splits the difference at the middle of the dead zone.

The `Syncer` also keeps these diagnostics for each frame:

- the decoded sync words and the number of sync pulses located,
- the frame start and the scale,
- the residual of each sync pulse,
- the deviation of each sync gap from the word period.

### EccDecode

`EccDecode` corrects each frame with its Reed-Solomon parity, and it hands the corrected words to
the next stage. It is the first step of the decode path, so every later stage reads corrected
words. It keeps one result per frame, and `report` turns those results into the error rates.

A frame that holds more bad symbols than the code can correct passes through unchanged, and the
stage counts it as uncorrectable. A frame of the wrong length also passes through unchanged, and it
stays out of the statistics. Such a frame lost or gained a pulse before the ECC ever saw it.

`Model1` inserts this stage whenever `ecc_num` is above 0. `Model2` inserts it only under
`--check-ecc`. The RS parameters of an unknown transmitter are unknown, and a wrong parameter makes
every frame read as uncorrectable.
[`docs/math.md`](math.md#8-error-correction) gives the code and the rate definitions.

### MetadataCheck

`MetadataCheck` removes the first `metadata_num` words of each frame and passes the rest on. With
`verify` it also checks them, and it runs after `EccDecode`, so it tests corrected words.

Two rules follow from the counter that `FrameGen._fill_metadata` writes, and they are not equally
safe:

- Inside a frame the metadata words must be consecutive. This holds whatever happened earlier in
  the capture, so the stage checks it whenever `verify` is set.
- Across frames the counter must continue. `FrameFilter` drops frames by design, and a dropped
  frame breaks that chain, so the stage checks this in `strict` mode only.

Without `strict` a mismatch increments `mismatches` and writes a warning. With `strict` it raises
`ValueError`. `Model1` sets `verify`. `Model2` gains it from `--check-metadata` and
`--strict-metadata`.

### Collector

`Collector` keeps every frame it sees. It works as a sink at the end of a pipeline. It also works
as a tap in the middle, because `process` passes the frames through unchanged. `Model2` uses
up to four collectors: one after the `Splitter`, one after the `Syncer`, one after `EccDecode`, and
one at the end. The tap after the `Syncer` holds the words as received, so the diagnostic plots
still see what arrived and not what the ECC repaired.

### PlotStage

`PlotStage` draws the data that passes through it onto a Matplotlib axes. Both models add these
stages only when the caller asks for plots.

## Artifacts

`Cli.output_dir` creates `output/<YYYY-MM-DD_HH-MM-SS>/`. Every run writes its files there, so two
runs never overwrite each other. Both models save `run.log`, `pipeline.txt` and `params.json`.
`Model2` also saves the decoded words as `decoded.npz` and `decoded.txt`. Each script adds its own
files. `docs/measurement.md` lists them.
