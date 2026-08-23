# Mathematical model of each stage

This page states what each stage computes. [`docs/glossary.md`](glossary.md#symbols) defines every
symbol once, and this page uses no other symbol. Each section names the file that implements the
equation.

Sections marked **Bound (open)** hold a requirement that this project has not yet derived. They are
placeholders, and they name the quantity that a future derivation must produce.

## 1. Frame and timing

`src/coding_synchronization/encoder/FrameGen.py`, `src/coding_synchronization/encoder/Modulation.py`

One word carries one pulse. The value of the word is the position of that pulse inside the word:

$$W = 2^{M} + D$$

$$L = N_{\mathrm{sync}} + N_{\mathrm{meta}} + N_{\mathrm{data}} + N_{\mathrm{ecc}},
\qquad P = L + N_{\mathrm{eof}}$$

The pulse of word $j$ of frame $i$ sits at slot

$$x_{i,j} = (iP + j)\,W + v_{i,j}, \qquad v_{i,j} \in \{0, \dots, 2^{M}-1\}$$

`FrameGen._to_positions` computes this, and then removes the EOF words, which carry no pulse.

The frame period, the payload rate and the duty cycle follow:

$$T_{f} = P\,W\,T_{s}, \qquad
R_{b} = \frac{N_{\mathrm{data}}\,M}{T_{f}}, \qquad
\eta = \frac{L\,\tau_{p}}{P\,W\,T_{s}}$$

$R_b$ is the metric of [`docs/why.md`](why.md). Every word that carries no payload appears in $P$
and not in $N_{\mathrm{data}}$. Sync words, metadata words, parity words and the EOF gap therefore
all cost data rate in the same way.

## 2. Pass geometry

`src/coding_synchronization/encoder/PassageGen.py`

The orbital radius, the orbital period and the circular orbit speed:

$$r = R_{E} + h, \qquad T_{\mathrm{orb}} = 2\pi\sqrt{\frac{r^{3}}{\mu}}, \qquad
v = \sqrt{\frac{\mu}{r}}$$

`_elevation_to_time` computes the pass duration from the Earth central angles. $\lambda_{0}$ is the
half angle of the visible cap. $\eta_{n}$ is the nadir angle at elevation $\theta$.
$\lambda_{\min}$ is the central angle at that elevation.

$$\lambda_{0} = \arccos\frac{R_{E}}{r}, \qquad
\eta_{n} = \arcsin\left(\frac{R_{E}\cos\theta}{r}\right), \qquad
\lambda_{\min} = \frac{\pi}{2} - \theta - \eta_{n}$$

$$T_{\mathrm{pass}} = \frac{T_{\mathrm{orb}}}{\pi}\sqrt{\lambda_{0}^{2} - \lambda_{\min}^{2}}$$

The number of frames in the pass is $\lfloor T_{\mathrm{pass}} / T_{f} \rfloor$, and at least 1.
`PassageParams.time_s` caps $T_{\mathrm{pass}}$ when it is set.

## 3. Channel

`src/coding_synchronization/channel/`

Each stage maps pulse positions in slots to pulse positions in slots.

**RandomShift** adds independent timing noise to every pulse:

$$x' = x + n, \qquad n \sim \mathcal{N}(0, \sigma^{2})$$

**ConstantOffset** adds one draw to every pulse of a frame, which models an unknown time of
arrival:

$$x' = x + u, \qquad u \sim \mathcal{U}(0, u_{\max})$$

**VanishPulses** keeps each pulse with probability $1 - p_{v}$, so a frame of $L$ pulses keeps
$\mathrm{Bin}(L, 1-p_{v})$ of them. **AddedPulses** inserts $\mathrm{Pois}(p_{a}L)$ false pulses,
drawn uniformly across the span of the frame.

**DopplerShift** converts each position to a time relative to the closest approach, computes the
slant range, and converts the extra propagation delay back into slots:

$$t = (x - x_{\mathrm{tca}})\,T_{c}, \qquad
\rho(t) = \sqrt{h^{2} + (v t)^{2}}, \qquad
x' = x + \frac{\rho(t) - h}{c\,T_{c}}$$

The range rate and the fractional clock error that it produces are

$$\dot{\rho}(t) = \frac{v^{2}t}{\rho(t)}, \qquad
\frac{\Delta T_{s}}{T_{s}} = \frac{\dot{\rho}}{c}, \qquad
|\dot{\rho}| < v$$

**Assumptions, and why they give an upper bound.** The model uses a straight line past a flat
Earth, a static ground station directly below the ground track, and a pass through the zenith. A
zenith pass has the largest range rate that the orbit can produce, so the model is the worst case
for the receiver. A real pass at a lower elevation gives a smaller $\dot{\rho}$ and a smaller
$\ddot{\rho}$. The model therefore over-states the impairment, which is the safe direction for a
design decision. `PassageParams.max_elevation_deg` shortens the pass, and it does not change this
geometry.

## 4. Pulse extraction

`src/coding_synchronization/measurement/OffsetExtractor.py`

The combined signal comes from the two channels after the removal of their own DC level $b_a$ and
$b_b$:

$$y[n] = (a[n] - b_{a}) \pm (b[n] - b_{b})$$

A run is a set of consecutive samples above the threshold $\gamma$. A run ends when the signal
falls below $\kappa\gamma$, which merges the ringing of a pulse into the pulse. `--edge` chooses
how one run becomes one position:

| Method | Position |
|---|---|
| `centroid` | $\hat{x} = \dfrac{\sum_{n} n\,y[n]}{\sum_{n} y[n]}$ over the run |
| `peak` | $\hat{x} = \arg\max_{n} y[n]$ |
| `edge` | the mean of the two interpolated crossings of $\gamma$ |
| `rising` | the interpolated rising crossing of $\gamma$ only |

A position in samples becomes a position in slots through $x = \hat{x}\,\Delta t / T_{s}$.

## 5. Slot time measurement

`src/coding_synchronization/measurement/Cli.py`, in `extract_calibrated`

Every word carries exactly one pulse, so the gap between two pulses is one word period plus the
difference of two PPM values. Those differences are symmetric around zero, so the median gap
measures the word period:

$$\hat{T}_{s} = \frac{\mathrm{median}_k\,(\hat{x}_{k+1} - \hat{x}_{k})\;\Delta t}{W}$$

This holds whatever the transmitter sends, and it does not depend on the sample rate that the
caller stated. [`docs/measurement.md`](measurement.md#why-the-scripts-measure-the-slot-time)
explains why a wrong slot time destroys the framing.

## 6. Frame splitting

`src/coding_synchronization/decoder/Splitter.py`

The stage cuts wherever a gap exceeds $\tau = N_{\mathrm{eof}} W$. Two conditions must hold.

No cut inside a frame. The largest gap between two pulses of one frame occurs when one word carries
0 and the next carries $2^{M}-1$:

$$g_{\max}^{\mathrm{in}} = W + 2^{M} - 1 \le \tau
\quad\Longrightarrow\quad
N_{\mathrm{eof}} \ge 1 + \frac{2^{M}-1}{W}$$

A cut at every frame boundary. The smallest gap across the EOF gap has the last word of a frame at
$2^{M}-1$, and the first word of the next frame at 0:

$$g_{\min}^{\mathrm{out}} = (N_{\mathrm{eof}} + 1)W - (2^{M} - 1) > \tau
\quad\Longrightarrow\quad
W > 2^{M} - 1$$

The second condition holds for every $D \ge 0$. The first gives $N_{\mathrm{eof}} \ge 2$ for the
frozen $M = 10$ and $D = 16$. The project sets $N_{\mathrm{eof}}$ near $L/4$, which is far above
that minimum, so the splitter keeps a large margin against timing noise. The cost of that margin is
the data rate term of section 1.

**Bound (open).** The minimum above assumes exact positions. The margin that a given $\sigma$ and a
given pulse loss rate need is not yet derived.

## 7. Synchronization

`src/coding_synchronization/decoder/Syncer.py`

Pass 1a takes a coarse scale from the sync section, because the median survives one missing pulse:

$$\hat{s}_{0} = \frac{\mathrm{median}\,(\Delta p_{0..N_{\mathrm{sync}}-1})}{W}$$

Pass 1b locates each sync pulse within the margin $m = \lfloor W/8 \rfloor$ of its expected
position. It then fits a least-squares line through the located pulses against their word index
$k$. The slope of that fit sets the scale:

$$\hat{s} = \frac{1}{W}\,
\frac{\sum_{k \in \mathcal{F}} (k - \bar{k})(p_{k} - \bar{p})}{\sum_{k \in \mathcal{F}} (k - \bar{k})^{2}}$$

$\mathcal{F}$ holds the located pulses only. A pulse that was never found contributes nothing, and
this is why the fit uses the word index and not the position in the list. The median alone is not
the least-squares slope, and the difference appears as a residual tilt.

The frame start and the residual of each sync pulse:

$$\hat{f} = \frac{1}{N_{\mathrm{sync}}}\sum_{k} \left(\frac{p_{k}}{\hat{s}} - kW\right) - v_{\mathrm{sync}},
\qquad
r_{k} = \frac{p_{k}}{\hat{s}} - \left(\hat{f} + kW + v_{\mathrm{sync}}\right)$$

$v_{\mathrm{sync}}$ cancels between the two equations, so the residual does not depend on it. The
sync value is therefore a free parameter, and a tilted residual never means that the sync value is
wrong.

Pass 2 decodes every pulse of the frame:

$$j = \left\lfloor \frac{p/\hat{s} - \hat{f}}{W} \right\rfloor, \qquad
v = p/\hat{s} - \hat{f} - jW$$

A pulse that lands in the dead zone is ambiguous, because it can be a late word $j$ or an early
word $j+1$. The stage splits the difference at the middle of the dead zone:

$$v > \frac{(2^{M}-1) + W}{2} \quad\Longrightarrow\quad j \leftarrow j+1,\; v \leftarrow v - W$$

### Tolerance to a slot clock error

Let $\varepsilon$ be the relative error that remains in $\hat{s}$ after the fit. The decoded value
of word $j$ then drifts by $\varepsilon W j$ slots. A word decodes correctly while that drift stays
inside half a slot, and the last word of the frame is the first to fail:

$$|\varepsilon|\,W\,(L-1) < \tfrac{1}{2}
\quad\Longrightarrow\quad
|\varepsilon| < \varepsilon_{\max} = \frac{1}{2W(L-1)}$$

Worked example, with $M = 10$, $D = 16$ and the frame of [`docs/why.md`](why.md), so $W = 1040$ and
$L = 229$: $\varepsilon_{\max} = 2.1\times10^{-6}$, which is 2.1 parts per million. A longer frame
tightens this limit in direct proportion, and this is the second metric of
[`docs/why.md`](why.md).

### Tolerance to Doppler shift

The range rate of section 3 is a fractional clock error of $\dot{\rho}/c$. The scale fit of pass 1b
absorbs a constant rate, because a constant rate is exactly a scale error. What survives the fit is
the change of the rate across one frame:

$$\varepsilon_{\mathrm{res}} \approx \frac{\ddot{\rho}\,T_{f}}{2c}$$

The requirement is $\varepsilon_{\mathrm{res}} < \varepsilon_{\max}$ from the section above.

**Bound (open).** The maximum $\ddot{\rho}$ of the zenith pass, and the frame length that follows
from it, are not yet derived here.

**Bound (open).** The probability that pass 1b locates fewer than 3 sync pulses, as a function of
$\sigma$, the pulse loss rate and $N_{\mathrm{sync}}$. That probability is the acquisition failure
rate of a frame.

## 8. Error correction

`src/coding_synchronization/decoder/Ecc.py`

One frame without its sync words is one Reed-Solomon codeword over $GF(2^{M})$:

$$n = N_{\mathrm{meta}} + N_{\mathrm{data}} + N_{\mathrm{ecc}}, \qquad
k_{\mathrm{info}} = N_{\mathrm{meta}} + N_{\mathrm{data}}, \qquad
t = \left\lfloor \frac{N_{\mathrm{ecc}}}{2} \right\rfloor$$

The decoder corrects up to $t$ symbol errors in a codeword. It reports a frame as uncorrectable
beyond that. The code follows the CCSDS Orange Book that the mission adopted, and the parameters
were confirmed by decoding the captures. [`docs/measurement.md`](measurement.md#the-reed-solomon-check)
names the source.

The measured rates use the corrected codeword as the reference, because no copy of the transmitted
data exists. With $\mathcal{C}$ the set of correctable frames and $\mathcal{U}$ the uncorrectable
ones:

$$\mathrm{WER}_{\mathrm{pre}} = \frac{\sum_{i \in \mathcal{C}} e_{i} + |\mathcal{U}|(t+1)}{n\,(|\mathcal{C}|+|\mathcal{U}|)},
\qquad
\mathrm{WER}_{\mathrm{post}} = \frac{|\mathcal{U}|\,n}{n\,(|\mathcal{C}|+|\mathcal{U}|)},
\qquad
\mathrm{FER} = \frac{|\mathcal{U}|}{|\mathcal{C}|+|\mathcal{U}|}$$

$e_{i}$ counts the symbols that the decoder changed in frame $i$. An uncorrectable frame holds at
least $t+1$ bad symbols, so the pre-ECC rates are lower bounds whenever $\mathcal{U}$ is not empty.
The bit rates use the same counts. They put the bit errors of the corrected symbols in place of
$e_i$, and $nM$ bits per frame in the denominator.

**Bound (open).** The relation between the pre-ECC word error rate and the post-ECC frame error
rate, which is the curve that decides how large $N_{\mathrm{ecc}}$ must be.
