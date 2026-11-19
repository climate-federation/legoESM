# NEMO testcase L2 GYRE high resolution — round 9 receipt

Date: 2026-10-15

Incoming lane tip: `387e7a14a3e8fb4b826db868f991c5339b6d3836`

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round9/`.
All oracle comparisons below use admitted scalar-math NEMO records and the
certified CPU fp64/libm policy.  The long GPU run remains deliberately
non-certified and statistical.

## Verdict

**The material step-2 tracer transcription defect is fixed; the exact rung
remains DEBT.**  Substituting NEMO's recorded `avt` reduces the solved
temperature residual from `1.81572090696136e-9 K` to
`1.177724584522366e-12 K` (a `1541.7x` reduction).  The sea-surface time
level used in the TKE shear divisor is the dominant owner of the material
`1.8e-9 K` component: NEMO calls the closure with the step-entry slot in both
formal positions, while legoESM had passed the half-step surface needed by
the later implicit tracer solve into the closure too.  Routing those users
separately leaves `1.1795009413617663e-12 K` in the production solve, at the
pre-solve floor.  That remaining residual is not assigned an owner.

The plain and Smagorinsky quarter-degree temperature rows improve by roughly
three orders at entry `kt=3`, but neither ladder is bit-exact.  The first remaining
zero-word miss is entry `kt=2` temperature (`1.0658141036401503e-14 K`);
the largest entry-`kt=3` residual is SSH (`1.8483e-11 m`).  No term owner is
claimed for either remaining boundary.

## F5 — measured vertical-coefficient owner

**Finding.**  Round 8's first-differing-list entry nominated TKE `avt`; it
did not prove ownership.  `step2_vertical_owner.json` adds a same-graph,
one-variable substitution at the executed `tra_zdf` seam.  The production
solve and a seam with no replacement both give `1.81572090696136e-9 K`.
NEMO `avt` gives `1.177724584522366e-12 K`; NEMO `avm` gives no improvement.
The active input plant moves the solve, and the recorded coefficient identity
passes.  The no-replacement seam is itself `DEBT` by `1.0658e-14 K`, but it
passes its preregistered passive limit and preserves the production maximum.

Above the registered `1e-12` relative threshold, the differing `avt/avm`
ratio occupies 227 wet cells, all at the first interface; none is on the
background floor or EVD arm.  Replacing only those
ratio cells leaves `1.9632295789051568e-11 K`, while replacing their
complement leaves the original maximum.  This ratio reconstruction is a
sensitivity split, not a claim that the writer recorded NEMO's private
Prandtl work array.

**Disposition.**  Fixed in the shared WS-RK3 path.  The implicit vertical
solve retains half-step `eta_now`; TKE shear separately receives step-entry
`state.eta`.  After the change, the direct production solve is
`1.1795009413617663e-12 K`, NEMO-`avt` substitution is
`1.177724584522366e-12 K`, no ratio/common-factor cells remain above the
registered `1e-12` relative threshold, `avt` differs by at most
`9.107298248878237e-18 m2/s` (42 ULP), and `avm` by at most 22 ULP.  The
probe labels this final tiny comparison `CONFOUNDED` because its
pre-registered passive-seam relative bar becomes stricter than the residual;
the quantitative substitution, not that label, supports the fix.

The defect begins in step 2 because the model starts from rest: step-1 shear
is zero, so the two surface time levels cannot distinguish the divisor.  It
is not quarter-degree-specific.

**Evidence.**  `step2_vertical_owner.json` (commit `45af3540c`) and
`step2_vertical_owner_after_fix.json` (commit `5f92ee93f`).

**Reversion.**  With only the routing restored to the half-step surface,
`test_gyre_ws_step_routes_step_entry_ssh_to_tke_shear` fails at the
production-observed face metrics (`tke_shear_reversion.log`); restored code
passes.

## F3 — stage-3 solar-stretch composition

**Finding.**  NEMO constructs Kmm stretch as the mean of the two already
rounded endpoint `r3t` ratios.  Round 8 used a mean live thickness divided by
reference thickness.  They are the same real number but differ at quarter
degree in 4,252/9,600 wet columns (maximum 2 ULP) and in 39,762/288,000 QSR
accumulator words (maximum absolute `3.3881317890172014e-21 K/s`).  At one
degree they differ in 242/600 columns and 2,510/18,000 accumulator words.

**Disposition.**  Fixed locally for stage-3 two-band solar forcing with
`nemo_r3t_rk3_stage2_stretch`, which rounds reciprocal depth, each endpoint
product, their sum, the half-product, and `1+` in source order.  Both the
production helper and an independent NumPy statement replay match NEMO at
the zero-word bar for both decks.  The existing ratio-of-mean-SSH form also
happens to match these records, but it is not used as a general identity.

**Evidence.**  `stage3_kmm_stretch.json` at commit `219db484b`, verdict
`CONFIRMED_COMPOSITION`; both one-word ULP plants fire.

**Reversion.**  Restoring the round-8 mean-thickness expression makes
`test_stage3_two_band_solar_uses_source_literal_kmm_stretch` fail in
2,288/18,000 active tendency words, maximum `8.470329472543003e-22 K/s`
(`qsr_stretch_reversion.log`).

## F8 — unsupported two-band integrators

**Finding.**  Only `rk3_ws` replaces the shared pipeline field with NEMO's
single stage-3 Kmm two-band call.  Another tracer integrator would retain
only the step-entry field.

**Disposition.**  Model construction now refuses `nemo_qsr_2bd` unless the
tracer integrator is `rk3_ws`; legal controls for ordinary RK3 without that
scheme and WS-RK3 with it both construct.

**Evidence and reversion.**  The focused test passes normally.  Removing only
the guard makes it fail with `DID NOT RAISE`
(`qsr_integrator_guard_reversion.log`).

## Quarter-degree scalar ladders

Maximum absolute entry-state residuals follow.  Every one-ULP field plant
fires.  `r4` is stamped clean at `c81071b3a`; `r4_smag` is stamped clean at
`709c5c235`.  The latter used `JAX_CPU_ENABLE_ASYNC_DISPATCH=false` after two
otherwise complete executions hit a Python-3.13/JAX GIL race during
interpreter finalization; the fp64/libm policy and calculated rows are
unchanged.

Plain `r4`:

| kt | T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0` | `0` | `0` | `0` | `0` |
| 2 | `1.0658e-14` | `2.1316e-14` | `1.3878e-17` | `1.3878e-17` | `0` |
| 3 | `1.2239e-12` | `1.5632e-13` | `8.9114e-13` | `8.5017e-13` | `1.8483e-11` |
| 4 | `2.6859e-12` | `3.3396e-13` | `1.4809e-12` | `1.3760e-12` | `1.7007e-11` |
| 5 | `8.4235e-12` | `1.3308e-11` | `2.0567e-12` | `1.5838e-12` | `2.4530e-11` |
| 6 | `2.3110e-11` | `3.0475e-11` | `3.1410e-12` | `2.0153e-12` | `2.6091e-11` |
| 7 | `2.7470e-11` | `4.3670e-11` | `3.5869e-12` | `3.2010e-12` | `1.7352e-11` |
| 8 | `3.1083e-11` | `4.9610e-11` | `5.4731e-12` | `4.3517e-12` | `2.3188e-11` |
| 9 | `3.6788e-11` | `5.9138e-11` | `6.4092e-12` | `8.0834e-12` | `3.4164e-11` |
| 10 | `4.3812e-11` | `6.9946e-11` | `1.0455e-11` | `1.7638e-11` | `3.5509e-11` |

Smagorinsky `r4_smag`:

| kt | T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0` | `0` | `0` | `0` | `0` |
| 2 | `1.0658e-14` | `2.1316e-14` | `1.3878e-17` | `1.3878e-17` | `0` |
| 3 | `1.4104e-12` | `1.6342e-13` | `8.7927e-13` | `9.5890e-13` | `1.8483e-11` |
| 4 | `2.8813e-12` | `3.4817e-13` | `1.5745e-12` | `1.5969e-12` | `1.7058e-11` |
| 5 | `8.4306e-12` | `1.3308e-11` | `2.3741e-12` | `1.9564e-12` | `2.4373e-11` |
| 6 | `1.7209e-11` | `2.7370e-11` | `3.0811e-12` | `2.3309e-12` | `2.6142e-11` |
| 7 | `2.4681e-11` | `4.0927e-11` | `4.2530e-12` | `3.5556e-12` | `1.7635e-11` |
| 8 | `3.2248e-11` | `5.3191e-11` | `5.4185e-12` | `5.3771e-12` | `2.4366e-11` |
| 9 | `4.7987e-11` | `6.3295e-11` | `7.0352e-12` | `1.7917e-11` | `3.4848e-11` |
| 10 | `7.0369e-11` | `6.2883e-11` | `9.9557e-12` | `1.6648e-11` | `3.6497e-11` |

Both verdicts remain `DEBT` under the zero-word bit bar.  Relative to round 8,
entry-`kt=3` T falls from `1.8157e-9` to `1.2239e-12 K` (`r4`) and
`1.8163e-9` to `1.4104e-12 K` (`r4_smag`); entry-`kt=10` T falls from
`8.384e-9` to `4.3812e-11 K` and `7.634e-9` to `7.0369e-11 K`.

## F1/F2 — shipped card and retractions

The digest remains `99b2dfce324b0f77`; it pins resolved configuration, not
distance to NEMO.  Direct scalar-math ten-step scoring is:

| row | round 7, before QSR fix | incoming round 8 | after round 9 |
|---|---:|---:|---:|
| kt3 T | `1.3765817e-8` | `1.1306994e-8` | `6.6108896e-11` |
| kt10 T | `1.2497120e-7` | `8.7309644e-9` | `3.0060576e-10` |
| kt10 S | `3.0449989e-8` | `4.5561350e-9` | `2.8397551e-10` |
| kt10 U | `1.7869446e-9` | `2.5533897e-10` | `2.5545262e-10` |
| kt10 V | `2.3732241e-9` | `8.7901158e-10` | `1.8078911e-10` |
| kt10 SSH | `1.2749727e-9` | `5.2343915e-10` | `5.2387667e-10` |

Round 8 moved 37/70 rows closer to NEMO, 15 farther, and left 18 unchanged.
Round 9 moves 42/70 closer than the incoming tip, 13 farther, and leaves 15
unchanged.  Round 8's regression was a cellwise oracle-relative control: it
correctly fired because some cells worsened against NEMO by more than its
two-row-scale-ULP bar, even though aggregate maxima moved closer in 37 rows
and farther in 15.  A new no-slack ceiling test covers all seventy current
row maxima and has a one-row worsening plant.

The round-8 receipt now carries that correction and retracts its unmeasured
`avt` localization.  The round-190 receipt now retracts the claim that its
10,200/18,000-word Kbb-QSR split was only an observer artifact: it was the
production false source removed in round 8.

## Long-run disposition

A new five-year GPU run is warranted.  The existing `post_fix_b401ec3f3`
run predates both the round-8 live-QSR correction and this round's TKE-shear
fix.  The registered `post_round9_219db484b` run pins the physics commit,
deck, native-GPU policy, output directory, and physical GPU 0.  Removing the
material step-2 seed should delay early legoESM–NEMO separation toward the
NEMO-scalar–NEMO-vector twin envelope; it cannot make chaotic five-year
trajectories identical.  The decision target remains the divergence envelope
through days 30–180 and yearly eddy fractions inside the spread of the two
NEMO builds, followed by the already preregistered year-5 climate bars.

## Inherited pins / OPEN

Inherited unchanged: the shipped one-degree deck and digest, `r4` and
`r4_smag` namelists, 31 levels, RK3, split-explicit free surface, Redi
tracers, TKE/EVD selection, 360-day forcing clock, scalar-libm CPU
certification, zero-word ladder bar, thickness-weighted daily-temperature
convention, and explicit physical-GPU selection.  No deck value or physics
knob changed.

OPEN, in order:

1. Localize the remaining scalar-oracle entry-`kt=2` T/S last-bit boundary;
   after that, the entry-`kt=3` SSH boundary is the largest early residual.
2. Score the registered post-round-9 five-year run against both scalar- and
   vector-math NEMO references.  Do not use the older post-Smagorinsky run as
   the controlling comparison.
3. The shared TKE route remains unmeasured on ORCA2-zps; this round establishes
   only both GYRE resolutions/decks.  The local two-band QSR composition
   change does not alter ORCA2's chlorophyll/RGB path.
4. The scalar 30-day rung and the exact bit bar remain inherited evidence;
   neither is promoted by this ten-step repair.

## Tests and evidence

The focused helper, production, admission, and instrument tests pass; each
landed behavior has a recorded failing reversion above.  The repository's
documented JAX compiler limit required the established lane battery to run by
module.  Its exact aggregate final line is:

`FULL_LANE_BATTERY_SPLIT_PASS: 128 passed, 2 deselected across 10 clean pytest processes`

The final receipt citation gate at clean commit `53c81b4b3` reports `PASS` for
all seven citations with zero citation and full-map audit failures.  The
shifted-`zdfsh2` plant at that same commit reports `FAIL`, as required.

## Compiled-source citations

The compiled step program calls `zdf_phy` once with both closure time-level
arguments bound to `Nbb` at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3.f90:168-216`.
The closure computes shear before selecting TKE and passes those formal slots
through at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfphy.f90:316-336`.
The executed no-wave shear stencil divides its before and middle velocity
differences by the corresponding two live face metrics at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfsh2.f90:98-108`.

Stage 2 forms the stage-3 T-point stretch as the mean of the two endpoint
ratios at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:215-221`.
Stage 3 calls the solar kernel once with Kmm, in tracer execution order, at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:691-725`.
That same stage passes Kmm geometry into the implicit vertical tracer solve,
whose content equation applies the Kmm live thickness at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90:284-289`.
The two-band kernel uses that same live stretch in attenuation depth and
layer thickness at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/traqsr.f90:615-645`.
