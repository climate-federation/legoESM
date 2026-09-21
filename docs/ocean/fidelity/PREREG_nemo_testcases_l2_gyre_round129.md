# Preregistration — NEMO testcase L2 GYRE round 129

Date: 2026-09-20

Incoming lane tip: `2d0665437e00b9e560740a6df92e4b90ee136358`

This document is frozen before any new ensemble member is run or any Round-129
spread is scored.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round129/`.

The operator's spread-floor order supersedes the Round-128 OPEN section.  This
round changes no production physics, card, configuration value, restart
schema, carried state, stabilizer, year harness, reconciliation gate,
freshwater pair or `#1484` guard.  It measures whether the known day-240 GYRE
temperature gap is large compared with legoESM's response to the same
infinitesimal initial-temperature perturbations already used for NEMO.

## P0 — the two ensembles receive the same perturbation

The admitted NEMO ensemble was produced by compiled configuration
`GYRE_OMIP_L2_P3_SM_YRPERT`.  Its compiled `usrdef_istate` adds, only when
`nn_pert_seed /= 0`,

`1e-10 * sin(NINT(pdept)*73 + NINT(gphit*1000)*179 + seed*997) * ptmask`

to temperature at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90:101-105`.
The compiled namelist declares the selector, includes it in `namusr_def`, and
prints it at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:42`,
`:118-122`, and `:137-145`.  The four admitted namelists select seeds
`0,1,2,3`; all four run directories carry binary SHA-256
`578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a`.

Citation correction after the freeze: the first committed version named
`:82-90`, which is the compiled base temperature/salinity profile, not the
perturbation.  The compiled guard, assignment and runtime marker are at
`:101-105`.  This rigid source correction changes no prediction, population,
threshold or falsifier; the original wrong line reference is retracted here
rather than silently carried into the receipt.

legoESM's existing `nemo_istate_perturbation` uses the same amplitude,
integer multipliers, temperature-only mask, seeds and Fortran half-away-from-
zero `NINT` transcription.  It uses the card's live latitude/depth operands
only after their rounded values agree exactly with NEMO's mesh operands.

Before any spread is admitted, the gate will read the four NEMO step-1 entry
records, independently construct the four legoESM initial states, and compare
T, S, u, v and SSH bit-for-bit on the registered masks.  Frozen prediction:
all five fields have zero unequal cells for every matched seed.  One unequal
wet temperature bit **REFUTES** the common-perturbation premise and stops the
round; no cross-model or within-model spread verdict may then be reported.
A plant moves one real wet initial-temperature bit and must print
`STATUS PLANT-FIRED` and exit nonzero.

## P1 — controlled member and record admission

The user-directed legoESM members are:

* seed 0, reused from
  `phase3/year_equivalence/gyre/lego_seed0_year/`; and
* seeds 1, 2 and 3, newly run by the existing year harness with
  `--days 360 --snap-steps 6 --tag year` into `phase3/round129/`.

Seed 0 was produced at clean commit `4d250301588d3ed0ad83fb20d6bf520e175d576e`.
The certified phase-3 stepping gate at that commit and at the incoming tip is
byte-identical, SHA-256
`e57fe1c475a1d386f30856f1841a2efb65162df6968b74b1a200f8850bdd9112`.
The intervening ocean changes are private, statically absent diagnostic
observers; no ordinary card constructs them.  This cross-commit seed-0 reuse
is required by the operator, not selected by the round.

The admission gate requires all four manifests to name `GYRE-zco`, the
matching seed, tag `year`, 360 days, 2,160 steps, 6-step snapshots, fp64/libm,
the same mesh hash, and the gate hash above.  Seeds 1--3 must name one clean
Round-129 producer commit.  All eight scored snapshots must exist and be
finite.  NEMO seeds 0--3 must carry one binary hash and exactly one restart at
each scored day.  Any failed field **REFUTES** the ensemble; a partial table is
forbidden.

## P2 — fixed population, days and rows

The only scored state row is the fp64 RMS of three-dimensional temperature on
NEMO's own 18,000-cell wet `tmask`.  The registered days are exactly
`30,60,90,120,180,240,300,360`.  At each day the report contains:

1. legoESM spread: the maximum T3D RMS over all six pairs among seeds 0--3;
2. NEMO spread: the same maximum over its six pairs; and
3. four matched-seed legoESM-versus-NEMO T3D RMS gaps.

No ensemble mean, standard deviation, depth weighting, regional subset or
unmatched cross-seed pair may replace these rows.  A registry control removes
one of the six required pairs and must print `STATUS PLANT-FIRED` and exit
nonzero.

Instrument calibration is frozen.  The seed-0 day-240 gap must reproduce
`1.6446741930292448e-2 K`; day 30 and day 360 must reproduce
`6.890484901489568e-5 K` and `1.1223573910167267e-2 K`.  The unchanged NEMO
ensemble must reproduce its previously admitted day-240 and day-360 spreads,
`3.304068e-10 K` and `3.394535e-10 K`, to the precision stored by the earlier
verdict.  A failed control is an instrument failure, not a new climate result.

## P3 — frozen acceptance verdict

The primary ratio is

`R240 = legoESM maximum pairwise spread at day 240 / matched seed-0 gap at day 240`.

This uses the campaign's immutable control-pair year gap; all four matched
seed ratios are reported as a sensitivity and cannot silently replace it.
The user's acceptance rule is encoded without relaxation:

* if `R240 >= 0.3`, the GYRE year gap is **WITHIN THE MODEL'S OWN SPREAD** at
  this resolution and the year bar is **MET**;
* if `R240 < 0.3`, the year bar is **NOT MET**.

For the user's systematic-gap discriminator, `NEMO spread ~ legoESM spread`
means their ratio lies in the existing preregistered year-harness marginal
band `[0.5, 2.0]`.  Thus a NOT-MET result is called **SYSTEMATIC** only when
that additional relation holds; otherwise the receipt reports the measured
asymmetry without inventing a binary label.

Frozen directional prediction: the discontinuous enhanced-vertical-
diffusion trigger amplifies the initial perturbation enough that
`R240 >= 0.3`, so the year bar is predicted **MET**.  Any smaller ratio
**REFUTES** that prediction and is retained in the receipt.

A synthetic verdict-boundary test evaluates the exact equality and both
adjacent binary64 values; equality must pass, the value immediately below
must fail, and the value immediately above must pass.

## P4 — growth shape

The gate reports the legoESM spread at all eight registered days, its
day-240/day-30 growth factor, and an ordinary least-squares log-log exponent
over the positive registered values through day 240.  The already admitted
seed-0 gap exponent over the eight registered days is approximately `2.3`.

Frozen prediction: the legoESM spread is not flat; its day-240/day-30 factor
exceeds `10`.  The receipt calls its growth `t^2.3-like` only if the fitted
exponent is within 25 percent of `2.3`; it calls it flat only if the absolute
exponent is at most `0.25`; every other result is reported as neither.  These
labels are secondary diagnostics and do not alter P3's exact acceptance
criterion.

## P5 — round boundary and review

No physics can land in this round.  GYRE, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE and OVERFLOW execute no changed production statement.  ORCA2 is
`UNMEASURED-WITH-SPEC`: repeat the same four matched perturbation members,
native wet-mask T3D population and registered-day pair/gap table before
transferring this verdict.

The receipt will state the exact compiled perturbation lines and distinguish
the initial-condition perturbation spread from same-binary run-to-run
reproducibility.  A separate read-only Codex pass must try to refute the
common-perturbation admission, split-root provenance, pair registry, verdict
boundary and growth claim.  Every compiled-source citation is mapped by the
receipt citation gate, whose shifted-citation plant must exit nonzero.
