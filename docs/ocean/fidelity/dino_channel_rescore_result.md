# Per-row rescore of the circumpolar-channel verdict

**Status: CANDIDATE pending the campaign's two independent adversarial
reviews.** The instrument's registered classifications are reported exactly;
no claim is promoted to a campaign finding until both reviews sign off.

**Instrument:** `scripts/validate/ocean_fidelity/dino_1226/channel_rescore.py`
at clean producer `bad21740475bd57e6b47f95ef19696ac6dd9aa61`.

**Pre-registration:** `PREREG_channel_rescore.md` at `e2fd09158`, committed
before any statistic in this lane was computed.

**Stamped artifact:** `dino_channel_rescore_artifact.json`; legoESM input
producer `a7b940f75c04d824b478de4e1728220e3a71989e`, all four members stamped fp64,
ladder `both`, and seasonal origin 180 days.  NEMO members 0--2 come from the
recorded `RUN_VERDICT360_M*` directories; member 3 is the upstream audit's
named relocated exception `/tmp/dino_v360_m3`.

Everything is offline from recorded verdict360 states.  No model was stepped;
CPU only.

## Verdict

> The channel's row-level disagreements are **ensemble-stable, smooth spatial
> structure** (**CONFIRMED**) whose persistence across the four seasonal
> horizons is **UNRESOLVED**, not decorrelating member noise.  Genuine row-level
> agreement is **REFUTED**.  Therefore the 17 ppm full-year band statistic is
> **PLAUSIBLY masking a real day-360 row-pattern error of magnitude
> X = 0.122876 Sv**, but the registered final verdict is **PLAUSIBLE, not
> CONFIRMED**, because temporal persistence and the all-horizon cancellation
> bar did not clear.

The competing conclusion -- *decorrelating noise, therefore genuine agreement
at row level* -- is **REFUTED** by the member-stability and row-agreement legs.
The stronger conclusion -- *one fixed physical row pattern persists all year*
-- is also not supported: the temporal leg is **UNRESOLVED** and several
horizon pairs reverse sign.

This is a statement about one year from a common NEMO state.  It is not a
shared-climate claim.  Every per-row floor has `n=4` per side and is a
factor-of-two estimate.  The floors remain unsaturated at day 360, so every
gap/floor multiple and every count beyond a floor bar is an **upper bound**.

## The registered bars and results

Each correlation uses the centered Pearson denominator.  Its uncertainty uses
the preregistered all-lag Bartlett effective row count, not 35 independent
rows; aggregate intervals use the same-row moving-block bootstrap.

| decider | CONFIRM bar | REFUTE/competing bar | measured | registered result |
|---|---|---|---|---|
| A: across horizons | median r >= 0.70 and block-CI lower > 0.30 | median r <= 0.30 and CI upper < 0.70 | **r = -0.0656**, CI **[-0.1645, 0.7426]**, median/min effective rows **8.31/6.34** | **UNRESOLVED**; the upper endpoint misses the decorrelation bar by 0.0426 |
| B: across members | median r >= 0.70 and CI lower > 0.30 | median r <= 0.30 and CI upper < 0.70 | **r = 0.9967**, CI **[0.9890, 0.9980]**, median/min effective rows **7.23/4.31** | **CONFIRMED member-stable**; all-16 cross-pairing r = **0.9975** |
| C: spatial structure | physical-predictor r >= 0.70 with corrected lower > 0.30, or roughness Z <= 0.25; Z < 0.60 | all predictor r <= 0.30 with corrected upper < 0.70, Z >= 0.60, and >=2/3 resolved adjacent pairs flip | **Z = 0.1806**; **1/30** resolved adjacent pairs flips | **CONFIRMED smooth physical structure**, not grid-scale alternation |
| D: row agreement | >=90% rows within 2 floors and <=10% beyond 5 floors | >=25% rows beyond 5 floors | **4/35** within 2; **17/35** beyond 5; X = **0.122876 Sv** | genuine row-level agreement **REFUTED** |
| E: band sensitivity | median C <= 0.10 and one-row J >=2 or mask-edge E >=2 | median C >=0.50 and J,E <=1 | median **C = 0.10230**; day-360 **C = 0.01985**, **J = 1.858**, **E = 3.680** band floors | **UNRESOLVED**; all-horizon median misses by 0.00230, while day 360 is edge-sensitive |

`C = |band net| / sum|row gap|`; its denominator is the gross row-level
disagreement.  `J` is the largest single-row deletion movement divided by the
original channel-band floor.  `E` is the largest movement from shifting each
mask edge by at most one row, divided by that same floor.

## What “persistence” actually measured

The six horizon-pair correlations below are medians over the four paired
member profiles.  The same sequence occurs in every member, which is why
member stability can be confirmed while persistence through time cannot.

| horizon pair | median row-pattern r | interpretation under registered bars |
|---|---:|---|
| day 90 vs 180 | +0.207 | neither persistent nor resolved decorrelation |
| day 90 vs 270 | -0.110 | neither |
| day 90 vs 360 | **-0.792** | pattern reverses |
| day 180 vs 270 | **+0.894** | locally persistent |
| day 180 vs 360 | -0.295 | neither |
| day 270 vs 360 | -0.011 | neither |

This is temporally evolving deterministic structure, not a single fixed
profile and not member-to-member noise.  One annual cycle cannot distinguish a
seasonally rotating structure from elapsed-time evolution, so no mechanism is
attached.

## Spatial structure

The physical-predictor correlations are descriptive because adjacent-row
dependence leaves only about 4--9 effective rows and every Bonferroni interval
crosses zero.  Smoothness, not an individual predictor, clears the registered
spatial bar.

| horizon | transport capacity r | NEMO row transport r | NEMO row-gradient r |
|---:|---:|---:|---:|
| 90 | -0.457 | -0.407 | -0.189 |
| 180 | +0.559 | +0.201 | +0.693 |
| 270 | +0.678 | +0.196 | +0.752 |
| 360 | +0.529 | +0.494 | +0.394 |

At day 360 the winning predictor is thickness-and-width-weighted row transport
capacity (`r=0.529`), but its 98.33% interval is `[-0.812, 0.981]`; calling
bathymetry the owner would therefore be unsupported.  What is confirmed is
negative space: `Z=0.181` against the smooth bar 0.25 and only one sign flip in
30 resolved adjacent pairs rule out a sign-alternating grid mode.

## The band sum through the four horizons

These are control-member values.  Counts use each row's own two-sided floor at
the same horizon; their multiples are upper bounds because the floors are
unsaturated.

| day | band net [Sv] | sum|row gap| X [Sv] | C = |net|/X | rows <=2 floors | rows >5 floors |
|---:|---:|---:|---:|---:|---:|
| 90 | +0.069672 | 0.080826 | 0.8620 | 0/35 | 35/35 |
| 180 | -0.005252 | 0.074864 | 0.0702 | 7/35 | 20/35 |
| 270 | -0.028208 | 0.139385 | 0.2024 | 3/35 | 26/35 |
| 360 | **+0.002440** | **0.122876** | **0.01985** | **4/35** | **17/35** |

The endpoint retains 1.985% of the row-level magnitude, i.e. 98.0% cancels.
Moving either channel boundary by at most one row changes the net by as much as
3.68 channel-band floors.  But the registered sensitivity decider used the
median across all 16 horizon/member cases; day 90 carries little cancellation,
leaving that median at 0.10230 just outside the 0.10 CONFIRM bar.  This near miss
is reported, not rounded into a pass.

The conservative floor-excess companion is 0.078098 Sv, 63.6% of X.  The
pre-registration called this a lower bound; that direction was wrong.  Because
the floors are still growing, subtracting today's `2*floor` makes this an
**upper bound on the eventual floor-excess magnitude**.  The probe and stamped
artifact carry the corrected label.  No verdict bar changed.

## Day-360 per-row ledger

Gap is legoESM minus NEMO.  The floor is that row's RSS of the two sides' own
four-member sample standard deviations.  Multiples are upper bounds.

| row | latitude | gap [Sv] | row floor [Sv] | |gap|/floor |
|---:|---:|---:|---:|---:|
| 14 | -64.44 | +0.014669 | 0.005989 | 2.45 |
| 15 | -64.00 | +0.004399 | 0.001418 | 3.10 |
| 16 | -63.56 | -0.017846 | 0.001883 | 9.48 |
| 17 | -63.11 | -0.004886 | 0.001132 | 4.31 |
| 18 | -62.66 | -0.005593 | 0.000958 | 5.84 |
| 19 | -62.20 | -0.004910 | 0.000696 | 7.05 |
| 20 | -61.73 | -0.005733 | 0.000781 | 7.34 |
| 21 | -61.25 | -0.005681 | 0.000767 | 7.41 |
| 22 | -60.76 | -0.006021 | 0.000708 | 8.50 |
| 23 | -60.27 | -0.003414 | 0.000743 | 4.59 |
| 24 | -59.77 | -0.002469 | 0.000566 | 4.36 |
| 25 | -59.26 | -0.001962 | 0.000446 | 4.40 |
| 26 | -58.75 | -0.001191 | 0.000356 | 3.34 |
| 27 | -58.23 | -0.000512 | 0.000255 | 2.01 |
| 28 | -57.70 | +0.000616 | 0.000231 | 2.67 |
| 29 | -57.16 | +0.000919 | 0.000194 | 4.73 |
| 30 | -56.61 | +0.001366 | 0.000211 | 6.46 |
| 31 | -56.06 | +0.001462 | 0.000257 | 5.69 |
| 32 | -55.49 | +0.001515 | 0.000276 | 5.50 |
| 33 | -54.92 | +0.000158 | 0.000257 | 0.62 |
| 34 | -54.34 | +0.000432 | 0.000264 | 1.63 |
| 35 | -53.76 | +0.000467 | 0.000299 | 1.56 |
| 36 | -53.16 | +0.000993 | 0.000316 | 3.15 |
| 37 | -52.56 | +0.001213 | 0.000322 | 3.77 |
| 38 | -51.95 | +0.000504 | 0.000301 | 1.67 |
| 39 | -51.33 | +0.000930 | 0.000303 | 3.07 |
| 40 | -50.70 | +0.001286 | 0.000291 | 4.42 |
| 41 | -50.06 | +0.002049 | 0.000275 | 7.44 |
| 42 | -49.41 | +0.002719 | 0.000274 | 9.91 |
| 43 | -48.76 | +0.003921 | 0.000292 | 13.42 |
| 44 | -48.09 | +0.004636 | 0.000307 | 15.08 |
| 45 | -47.42 | +0.004716 | 0.000343 | 13.77 |
| 46 | -46.74 | +0.004708 | 0.000351 | 13.40 |
| 47 | -46.05 | +0.004484 | 0.000329 | 13.61 |
| 48 | -45.35 | +0.004495 | 0.000335 | 13.41 |

The profile is not a checkerboard: it has a negative lobe in the southern half
of the channel and a positive lobe on the northern flank.  Those lobes cancel
in the band sum.  The profile's association with bathymetry or jet structure
is **PLAUSIBLE**, not confirmed; its smoothness is confirmed.

## Controls and tests

The inherited regional-audit self-test passes before every measurement.  The
new probe then prints all six plants firing:

| planted violation | observed control result |
|---|---|
| wrong row weight | row-to-band closure aborts |
| layer average substituted for thickness weighting | reducer comparison aborts |
| 35 independent rows asserted on a duplicated-block profile | effective-row check aborts |
| persistent profile rows independently permuted | persistence classifier crosses to the decorrelating side |
| alternating cancellation profile called robust | cancellation/roughness control aborts |
| one global floor substituted for two row-specific floors | classification equality aborts |

Direct test result: **10 passed**.  Static check: **ruff clean**.  The final run
records **6/6 planted violations fired** and a clean producer SHA.

## Repository-layout discrepancy

The task expected `AGENTS.md` and the regional audit on this branch.  This
checkout's `HEAD` contained neither.  `AGENTS.md` was read from committed source
`24be4d763`; the audit result/probe were read from
`fidelity/dino-regional-audit` at `102ef501a`.  The exact audit and tracer-atlas
sources were imported into this branch; their SHA-256 values match that branch.

Newer `main` had renamed the audit's clock helper from public to private without
changing its body.  A committed compatibility alias lets the original audit
stamp check run unchanged.  The run log shows all four phases verified at
180.00 days before any scored state was loaded.
