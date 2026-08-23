# Glossary

This project uses one term for one concept. The code does not yet do this everywhere. The
"Also seen as" column names the other words that the code uses for the same thing. Use it to map
old text onto the term of this documentation.

| Term | Meaning | Also seen as |
|---|---|---|
| slot | The smallest time unit. One PPM pulse occupies one slot. | — |
| slot time | The duration of one slot, in seconds. | `slot_time`, `slot_s` |
| samples per slot | The number of waveform samples in one slot. | `--samples-per-slot` |
| word | One PPM symbol. A word carries a value from 0 to `2^ppm_rank - 1`. The value is the position of the pulse inside the word. | symbol, PPM value |
| word period | The length of one word in slots. It is `2^ppm_rank + dead_slots`. | `word_period` |
| dead slots | Slots added after the PPM range of a word. They give the transmitter time to recover. | `dead_slots` |
| offset | The position of one detected pulse. Offsets are in samples after extraction and in slots after conversion. | pulse position, centroid |
| frame | One full transmission unit: the sync words, the metadata words, the data words and the ECC words. | — |
| chunk | A group of pulses that the `Splitter` cut out of the capture. A chunk becomes a frame only after the `Syncer` accepts it. | frame (in `Splitter` and `Model2` logs) |
| sync section | The first `sync_num` words of a frame. Every sync word carries the same value. | sync words, sync head |
| sync value | The PPM value that every sync word carries. `--sync-value` sets it. | `sync_value` |
| frame start | The fitted position of word 0 in a frame, in slots. It can be negative. | `frame_start` |
| residual | The distance between where a pulse arrived and where the decode grid puts it, in slots. | sync residual, timing error |
| scale | The correction factor that the `Syncer` applies to the given slot time. | `slot_scale`, calibration |
| EOF gap | The dead time between two frames. `eof_num` gives its length in words. | `eof_num`, guard interval |
| codeword | One frame without its sync words. The metadata and data words carry the information. The ECC words carry the parity. | RS block |
| word error rate (WER) | The share of words that arrived wrong. | symbol error rate |
| bit error rate (BER) | The share of bits that arrived wrong. One word holds `ppm_rank` bits. | — |
| frame error rate | The share of frames that Reed-Solomon could not correct. | — |
| capture | One oscilloscope recording, saved as a two-column CSV file. | waveform, measurement |

## Two words that are easy to confuse

**Chunk against frame.** The `Splitter` cuts the pulse stream at every gap larger than its
threshold. Each cut gives one chunk. A chunk holds the right number of pulses only if the split was
correct. The `Syncer` then locates the sync section and turns the chunk into a frame. A log line
that reports 192 frames of 3 pulses reports chunks, not frames.

**Slot time against samples per slot.** The slot time is a property of the transmitter. The number
of samples per slot also depends on the sample rate of the oscilloscope. `Cli.extract_calibrated`
measures the slot time from the capture, so a wrong sample rate does not change the result.

## Symbols

[`docs/math.md`](math.md) uses these symbols and no others. Each symbol has one meaning.

### Frame and modulation

| Symbol | Meaning | Code |
|---|---|---|
| $M$ | PPM rank. One word carries $M$ bits. | `ppm_rank` |
| $D$ | Dead slots after the PPM range of a word. | `dead_slots` |
| $W$ | Word period in slots, $W = 2^{M} + D$. | `word_period` |
| $N_{\mathrm{sync}}$ | Sync words in a frame. | `sync_num` |
| $N_{\mathrm{meta}}$ | Metadata words in a frame. | `metadata_num` |
| $N_{\mathrm{data}}$ | Data words in a frame. | `data_num` |
| $N_{\mathrm{ecc}}$ | Parity words in a frame. | `ecc_num` |
| $N_{\mathrm{eof}}$ | Words in the gap between two frames. | `eof_num` |
| $L$ | Words in a frame, without the EOF gap. | `frame_len` |
| $P$ | Words in one frame period, $P = L + N_{\mathrm{eof}}$. | — |
| $v_{i,j}$ | PPM value of word $j$ in frame $i$. | — |
| $v_{\mathrm{sync}}$ | The value that every sync word carries. | `sync_value` |
| $x_{i,j}$ | Position of a pulse, in slots. | `offsets` |

### Time

| Symbol | Meaning | Code |
|---|---|---|
| $T_{s}$ | Slot time, in seconds. | `slot_time` |
| $T_{f}$ | Frame period, in seconds. | — |
| $\Delta t$ | Sample period of the capture, in seconds. | `dt_s` |
| $\tau_{p}$ | Optical pulse width, in seconds. | — |
| $T_{c}$ | Time unit that `DopplerShift` uses for one slot. | `chirp_duration_s` |
| $R_{b}$ | Payload bit rate. | — |
| $\eta$ | Duty cycle of the optical output. | — |

### Channel

| Symbol | Meaning | Code |
|---|---|---|
| $\sigma$ | Standard deviation of the per-pulse timing noise, in slots. | `sigma` |
| $u_{\max}$ | Largest constant offset of a frame, in slots. | `max_const_offset` |
| $p_{v}$ | Probability that a pulse disappears. | `vanish_rate` |
| $p_{a}$ | Mean number of false pulses per real pulse. | `added_rate` |
| $h$ | Orbital altitude. | `altitude_km` |
| $r$ | Orbital radius, $r = R_{E} + h$. | — |
| $v$ | Orbital speed. | `velocity_m_s` |
| $\theta$ | Peak elevation angle of the pass. | `max_elevation_deg` |
| $\rho$ | Slant range from the ground station to the satellite. | `slant_range` |
| $R_{E}$, $\mu$, $c$ | Earth radius, Earth gravitational parameter, speed of light. | `R_EARTH`, `GM`, `C` |

### Receiver

| Symbol | Meaning | Code |
|---|---|---|
| $\gamma$ | Detection threshold on the combined signal. | `threshold` |
| $\kappa$ | Hysteresis factor. A pulse ends below $\kappa\gamma$. | `hysteresis` |
| $\hat{x}$ | Estimated pulse position, in samples. | `offsets.samples` |
| $\tau$ | Splitter threshold, in slots, $\tau = N_{\mathrm{eof}}W$. | `split_threshold` |
| $\hat{s}$ | Fitted scale of the slot time. 1 means the given slot time was right. | `slot_scales` |
| $m$ | Search margin for a sync pulse, in slots. | `margin` |
| $\hat{f}$ | Fitted frame start, in slots. | `frame_starts` |
| $r_{k}$ | Residual of sync pulse $k$, in slots. | `sync_residuals` |
| $\varepsilon$ | Relative error that remains in $\hat{s}$. | — |
| $j$, $k$ | Word index in a frame. $k$ counts sync words only. | — |

### Error correction

| Symbol | Meaning | Code |
|---|---|---|
| $n$, $k_{\mathrm{info}}$ | Codeword length and information length, in words. | `EccParams.n`, `info_num` |
| $t$ | Symbols that the code can correct, $t = \lfloor N_{\mathrm{ecc}}/2 \rfloor$. | `correctable` |
| $e_{i}$ | Symbols that the decoder changed in frame $i$. | `symbol_errors` |
| $\mathcal{C}$, $\mathcal{U}$ | Correctable frames, and uncorrectable frames. | — |
