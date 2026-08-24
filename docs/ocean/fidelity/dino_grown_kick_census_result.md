# The grown-perturbation census — the chain link does NOT close, and one half of it is REFUTED

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_grown_kick_census.md`,
committed before a single number existed.
Probe: `scripts/validate/ocean_fidelity/dino_1226/grown_kick_census.py`.
Offline, from ensemble states already on disk. No model run, no GPU.

## Verdict

**The convective switch is not the differential rectifier.** On the two horizons
where the measurement is possible at all, legoESM's trigger resolves differently
between a perturbed member and its control 4.8–8.6× more often than NEMO's —
but that difference cannot be separated from the fact that legoESM's
perturbation is ~90× larger there, and the two statistics that *can* be
separated both point the other way:

| statistic | what it asks | result |
|---|---|---|
| near-threshold occupancy (amplitude-free) | is legoESM's stratification intrinsically more tippable? | **≤ 1.045×** — COMPARABLE |
| flips per differing interface (normalised) | does legoESM's trigger convert a perturbed interface into a flip more often? | **0.57–0.61** — NEMO is 1.6–1.75× HIGHER |
| raw flip rate (horizon-matched) | does legoESM flip more, full stop? | 4.8–8.6× higher, but confounded |

The two models sit on statistically indistinguishable stratification, and per
unit of perturbation legoESM's switch fires *less* readily, not more. What
differs is the perturbation being fed to it. **The convective edge is a passive
counter reporting an amplification that arrives from somewhere else** — which
sharpens, rather than contradicts, the note already carried on this chain that
the difference is not the edge but the noise feeding it.

## What was asked, and what the data could actually answer

Day 5 does not exist. Neither model wrote a 3-D state at day 5; both wrote days
0, 10, …, 90. That column is ABSENT, not interpolated.

Day 10 is not measurable either, and neither are days 20–70. legoESM's snapshots
are float32; a perturbed member differs from its control in **2 cells out of
342,134 wet cells at day 10**. The pre-registered resolvability bar (0.1% of wet
cells, on **every** member and on **both** models) is cleared only at **days 80
and 90**.

## The numbers, lane `kick2` (the lane whose kick matches NEMO's), 332,214 wet interfaces

Three arms throughout: legoESM as stored (float32), NEMO native (float64), and
NEMO cast to float32 and back — the storage-matched control legoESM is compared
against.

**(a) The growth asymmetry is real and large.** Max |ΔT|, day 90: legoESM
1.05e-2 K, NEMO 1.24e-4 K — 85×. At the 90th percentile the gap is ~1400×. The
two perturbation *distributions* never overlap: legoESM's median non-zero |ΔT|
is 1.6e-6 K (one float32 ulp — i.e. censored), NEMO's is 3.4e-12 K.

**(c) Pair flips, days 80 and 90** (per member, control-vs-perturbed):

| day | legoESM | NEMO fp64 | NEMO fp32 (noise floor) |
|---|---|---|---|
| 80 | 1544, 97, 1252 | 10, 8, 10 | 145, 114, 147 |
| 90 | 1642, 192, 1500 | 11, 11, 8 | 234, 234, 232 |

NEMO's convective trigger produces **zero** pair flips for the first sixty days
and 8–11 out of 332,214 at days 80–90. Its members agree with each other;
legoESM's do not — **member 2 never clears the storage-noise floor at any
horizon**, while members 1 and 3 clear it by 6–10×.

**(f) The amplitude-free statistic — the one that settles it.** The density of
each model's own N² near its own trigger, control member only, no perturbation
involved. Which bands are trustworthy is computed, not assumed: a band counts
only where casting NEMO to float32 moves it by less than 10%.

| |N²−thr| < | 1e-12 | 1e-11 | 1e-10 | 1e-9 | 1e-8 |
|---|---|---|---|---|---|
| legoESM, day 90 | 0.0009% | 3.5868% | 6.3339% | 14.7351% | 16.9975% |
| NEMO fp64, day 90 | 0.1264% | 0.7438% | 5.5937% | 14.7802% | 16.7669% |
| NEMO fp32, day 90 | 0.0024% | 3.4526% | 6.0584% | 14.6144% | 16.7639% |
| fp32 faithful? | no | no | yes | yes | yes |

Across every faithful band and all sampled horizons, legoESM's near-threshold
occupancy is at most **1.045×** NEMO's. The two models have essentially the same
density of tippable interfaces.

## Why the horizon-matched ratio cannot close the link

At a matched horizon legoESM's perturbation is up to ~2300× larger, and a larger
perturbation flips more triggers for trivial reasons — so using the
horizon-matched ratio to explain why legoESM amplifies more is circular. The
amplitude-matched control was built to break that circularity, and it **cannot**:
once both arms are required to be resolvable, the surviving horizons (80, 90)
have legoESM's perturbation ~90× larger than *any* NEMO horizon. **No
amplitude-matched pair exists.** NEMO's perturbation never grows to legoESM's
size inside 90 days.

## Retractions

1. **"NEMO's convection fires more often than legoESM's" is RETRACTED.** The
   apparent gap (NEMO 12.06% of wet interfaces at day 10 rising to 14.62%,
   legoESM 9.65% rising to 10.91%) is a **storage artifact**: casting NEMO to
   legoESM's float32 reproduces **99.9%** of the gap at day 10 and **98.7%** at
   day 90. legoESM's true float64 firing rate was never saved and is UNKNOWN.

2. **The prior census's denominator counted land.** The shipped census in
   `switch_rectifier.py` takes its wet mask from `isfinite(N²)`, which is true
   everywhere: its 362,180 is exactly 199×52×35, the whole grid, and the kick
   document's 372,528 is 199×52×36 the same way. Neither applied any dry-cell
   masking; 8.3% of the interfaces are land. The load-bearing prior conclusion —
   **zero** interfaces within reach at 1e-14 — is a count of zero and survives.

3. **The threshold-neighbourhood census must not be cited across models.** It
   applies one global maximum reach to every interface, and the two models'
   tail-to-bulk ratios differ by orders of magnitude, so it over-counts them by
   *different* factors. The pre-registration's claim that keeping the shipped
   convention made the two numbers comparable is withdrawn.

4. **This lane's first pass returned CONFIRM, and that is withdrawn.** It rested
   on a mean over members reading 99, 0, 0; on a resolvability bar applied to
   that mean and only to legoESM; and on an amplitude-matched control that
   compared legoESM's censored rate against NEMO's uncensored one while
   asserting in prose that the bias ran the other way. Both adversarial reviews
   found it independently.

## A trigger difference, sized rather than assumed

legoESM fires on the single now-level `N² < −1e-12`
(`enhanced_diffusion.py:195`). NEMO fires on `MIN(rn2, rn2b) <= −1e-12`
(`zdfevd.F90:93`), a minimum over two time levels — a latch, not a damper: it
fires if *either* level is unstable, so it strictly *increases* firing. On
NEMO's own state it adds only **0.2–0.4%** more firing. `bn2(tb,sb)` stands in
for `rn2b` and is a proxy; legoESM's before-level 3-D state was never saved, so
this column cannot be produced for legoESM at all.

## What this data cannot do

* **It cannot test the 2Δt flicker.** The snapshots are 10 days — 320 timesteps —
  apart, so a two-timestep alternation is completely aliased. No statistic here
  supports or weakens that link in either direction. It remains untested.
* **It cannot carry small-count ratios.** NEMO's float64 flip counts run
  0,0,0,0,0,0,1,9,10 over three pairs; a ratio against a Poisson count of order
  1–10 carries ~±35% at one sigma from the denominator alone.
* **Three members.** Every rate is a coarse estimate and every ratio a coarser
  one.

## Controls, all green before any number above was read

Nine self-checks, each shown to fire in the direction it guards: identical
states give exactly zero; a gross instability planted in a **dry** cell moves no
statistic while the **same** violation in a wet cell moves one; removing the
mask makes the dry test fail; the planted flip lands at the **exact** interface
below the cooled cell (so a uniform off-by-one cannot pass); NaN inside the mask
is fatal while NaN on land is tolerated; the float32 control is not inert; and
the verdict rule reaches CONFIRM, REFUTE, INDETERMINATE and UNDEFINED.

A separate `--ladder-check` builds the twin once and confirms the depth ladders,
EOS form and threshold this probe reads from NEMO's `mesh_mask` are the shipped
card's own — exact agreement (0.0 relative) under the fp64 policy — and measures
directly that the ladder choice cannot change a flip count.

Both mandatory adversarial reviews were run and both returned BLOCK. Every
blocking finding is fixed in the probe; the four they overturned are recorded as
retractions above.

## The single cheapest thing that would make a cross-model flip statistic interpretable

**Dump one legoESM member pair in float64 at days 30 and 90.** One rerun of two
members. It removes the storage confound entirely, and it is the only way to
learn legoESM's true firing rate, which is currently unknown. Until it exists,
no cross-model flip statistic at this magnitude can be read.

## What it does to the chain

The link "the hard edge rectifies the grown noise **more in legoESM**" is
**refuted on the two statistics that can be measured cleanly**, and
**unseparated** on the one that cannot. The edge's ~300×-per-step gain, measured
earlier at a sampled state, is not in question — but it applies to *both* models
equally, on stratification that is statistically the same. The amplification
asymmetry therefore still needs an owner, and the growth asymmetry itself
(~1400× in bulk |ΔN²| by day 90) is the thing that needs explaining. The 2Δt
flicker remains the leading hypothesis for it and remains **untested** — this
dataset, at a 10-day cadence, cannot test it.
