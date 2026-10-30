# NEMO testcase L2 GYRE high-resolution — round 7 receipt

Date: 2026-10-13

Incoming lane tip: `d279942e5e666bbbcf880cfaeffa1ad5eefd4770`

## Verdict

**HELD for one scalar-math acquisition; neither unrecorded quarter-degree
owner is guessed.**  The scalar-math NEMO records close the vector-libm wind
ambiguity and narrow the first-step boundaries, but they do not close the
resolution rung.  The large second owner first appears during step 2 in
temperature: the `kt=3` entry has a maximum
`3.3392346665550576e-9 K` residue, 939,911 ULP at that value, at native
legoESM index `[j=1,i=97,k=0]`, the surface southern wall.  The available
scalar writer serializes the three RK boundaries and operands only for step 1,
and it does not serialize the executed `tra_zdf` matrix/RHS even there.  It
therefore cannot name either the first-step tracer owner or the deciding
step-2 operand.  The committed scalar step-2 plus `tra_zdf` writer is the
stop-rule outcome.

Two source-fidelity corrections do land.  The seasonal clock now reproduces
NEMO's literal completed-360-day-year arithmetic, and the inert final T-point
mask is removed from the dynamic mode-32 Smagorinsky coefficient.  No deck
value changes.  The shipped one-degree card digest and ten-step legoESM
trajectory remain byte-identical.

## Oracle admissions and exact bar

The committed registry admits all four ten-step roots by absolute path,
compiler-math flavour, manifest digest, executable, namelist, all ten entry
records, and every auxiliary record the gate consumes.  For `r4` those
additional hashes cover stages 1/2/3; the tracer, ZDF, and barotropic records;
and both stage-2 source/composition streams.  A planted auxiliary-record digest
mutation refuses admission.

- `r4:scalar` and `r4_smag:scalar`: executable SHA-256
  `bbb992d9be39e6c4b57a1c7f8786241687d84dd8e6f59570107e2c2119ecdf2b`,
  `conda-scalarmath`, `-fno-tree-vectorize`, scalar libm;
- `r4:vector` and `r4_smag:vector`: executable SHA-256
  `974dc326b4f5237a0e8dfe50ba0ae8a4573507d57429fb82069ff3c20abfad41`,
  GNU vector `sin`/`cos` in the compiled loops plus scalar-libm remainders.

The scalar flavour is controlling.  The vector flavour is retained as a
labelled compiler-program comparison.  The bar is zero unequal active float64
words.  All five one-ULP plants fire for each of the four ten-step ladders.

## Ten-step ladders

Values are maximum absolute active-field residuals.  These are step-entry
records: `kt=1` is the common initial state, and `kt=N+1` is after step N.

### Plain scaled deck (`r4`)

| kt | scalar T | S | U | V | SSH | vector T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1.066e-14 | 2.132e-14 | 1.388e-17 | 1.388e-17 | 0 | 1.066e-14 | 2.132e-14 | 1.103e-15 | 1.103e-15 | 2.602e-18 |
| 3 | 3.339e-9 | 1.171e-10 | 1.260e-11 | 1.397e-11 | 1.848e-11 | 3.339e-9 | 1.171e-10 | 1.260e-11 | 1.397e-11 | 1.848e-11 |
| 4 | 7.115e-9 | 1.337e-10 | 7.995e-11 | 8.918e-11 | 3.497e-11 | 7.115e-9 | 1.337e-10 | 7.995e-11 | 8.918e-11 | 3.497e-11 |
| 5 | 1.093e-8 | 2.390e-10 | 1.775e-10 | 2.515e-10 | 6.192e-11 | 1.093e-8 | 2.390e-10 | 1.775e-10 | 2.515e-10 | 6.192e-11 |
| 6 | 1.289e-8 | 1.201e-9 | 3.598e-10 | 4.208e-10 | 8.852e-11 | 1.289e-8 | 1.201e-9 | 3.600e-10 | 4.207e-10 | 8.853e-11 |
| 7 | 1.389e-8 | 9.366e-10 | 4.283e-10 | 6.947e-10 | 1.006e-10 | 1.389e-8 | 9.367e-10 | 4.283e-10 | 6.947e-10 | 1.006e-10 |
| 8 | 1.507e-8 | 7.414e-10 | 5.362e-10 | 1.020e-9 | 1.202e-10 | 1.507e-8 | 7.414e-10 | 5.361e-10 | 1.020e-9 | 1.201e-10 |
| 9 | 1.918e-8 | 9.543e-10 | 7.221e-10 | 2.088e-9 | 1.517e-10 | 1.918e-8 | 9.543e-10 | 7.221e-10 | 2.088e-9 | 1.517e-10 |
| 10 | 2.206e-8 | 1.399e-9 | 7.612e-10 | 1.990e-9 | 1.926e-10 | 2.206e-8 | 1.399e-9 | 7.614e-10 | 1.990e-9 | 1.925e-10 |

### Smagorinsky deck (`r4_smag`)

| kt | scalar T | S | U | V | SSH | vector T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1.066e-14 | 2.132e-14 | 1.388e-17 | 1.388e-17 | 0 | 1.066e-14 | 2.132e-14 | 1.103e-15 | 1.103e-15 | 2.602e-18 |
| 3 | 3.340e-9 | 1.171e-10 | 1.256e-11 | 1.426e-11 | 1.848e-11 | 3.340e-9 | 1.171e-10 | 1.256e-11 | 1.426e-11 | 1.848e-11 |
| 4 | 7.115e-9 | 1.413e-10 | 7.963e-11 | 9.391e-11 | 3.504e-11 | 7.115e-9 | 1.413e-10 | 7.963e-11 | 9.391e-11 | 3.503e-11 |
| 5 | 1.089e-8 | 2.510e-10 | 1.851e-10 | 2.826e-10 | 6.191e-11 | 1.089e-8 | 2.510e-10 | 1.851e-10 | 2.826e-10 | 6.191e-11 |
| 6 | 1.227e-8 | 1.140e-9 | 3.647e-10 | 4.719e-10 | 8.878e-11 | 1.227e-8 | 1.140e-9 | 3.647e-10 | 4.719e-10 | 8.878e-11 |
| 7 | 1.404e-8 | 5.635e-10 | 4.613e-10 | 8.691e-10 | 1.016e-10 | 1.404e-8 | 5.635e-10 | 4.613e-10 | 8.691e-10 | 1.016e-10 |
| 8 | 1.536e-8 | 6.007e-10 | 7.734e-10 | 1.300e-9 | 1.229e-10 | 1.536e-8 | 5.991e-10 | 7.735e-10 | 1.300e-9 | 1.230e-10 |
| 9 | 1.963e-8 | 7.645e-10 | 8.308e-10 | 2.545e-9 | 1.560e-10 | 1.963e-8 | 7.685e-10 | 8.307e-10 | 2.545e-9 | 1.560e-10 |
| 10 | 2.272e-8 | 1.878e-9 | 9.729e-10 | 2.957e-9 | 1.988e-10 | 2.272e-8 | 1.878e-9 | 9.729e-10 | 2.957e-9 | 1.990e-10 |

The `r4` and `r4_smag` scalar rows remain at the same scale through the first
two completed steps.  The large temperature debt is therefore a resolution-
rung owner, not a Smagorinsky-port increment.  Calling it inherited rounding
debt is withdrawn: at `kt=3`, legoESM is about four orders of magnitude farther
from NEMO than NEMO's scalar and vector builds are from each other.

## Scalar boundary walk; both tracer owners open

The step-1 scalar RK walk is:

1. stage 1: T, S, U, V, and SSH are exact;
2. stage 2: T, S, and SSH are exact; U and V first differ, each with maximum
   absolute residue `2.710505431213761e-20 m/s` and 2 ULP at the maximum;
3. stage 3: the entry-`kt=2` residues first reach T
   `1.0658141036401503e-14 K` (3 ULP), S
   `2.1316282072803006e-14` (3 ULP), and U/V
   `1.3877787807814457e-17 m/s` (4 ULP at the maximum); SSH is exact.

The acquired source-order stream records accumulators, not isolated NEMO
addends.  Subtracting two rounded endpoints is not an inverse of NEMO's
executed addition, so the earlier reconstructed-increment attribution to
vorticity is withdrawn.  Replaying each isolated legoESM component from the
exact preceding NEMO accumulator gives; the compiled call/write order is
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:396-412`:

| accumulated boundary | U max | U unequal | V max | V unequal |
|---|---:|---:|---:|---:|
| after HPG | 0 | 0 | 0 | 0 |
| after vorticity | 0 | 0 | 0 | 0 |
| after advection | 0 | 0 | 0 | 0 |

All three accumulated boundaries are exact.  The production raw RK update is
not: U/V each reach `1.6940658945086007e-21 m/s`, with 63,830/64,270 unequal
words.  Replacing the complete stage-2 RHS with NEMO's record makes that raw
update bit-exact in both components.  Both one-ULP controls fire, and a
material `1e-6` RHS plant moves the consumed raw U by `1.8e-3 m/s`; the
substitution cannot pass vacuously.

After the same exact raw update, the ordinary barotropic-mean replacement
still leaves 20,603 unequal U words and 21,079 unequal V words, each with
maximum `2.710505431213761e-20 m/s`.  This localizes the first U/V state split
to the mean-replacement boundary, but not to its arithmetic: the current
reader does not substitute NEMO's recorded barotropic target or `zub/zvb`.
Input versus composition remains OPEN.  NEMO's raw vector update is at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:461-462`; its reference-
thickness column reduction and correction are at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:506-526`.

The stage-1 tracer-transport candidate is independently refuted.  Horizontal
transports `zFu` and `zFv` are exact and `zFw` differs by at most
`4.3520742565306136e-14`, but substituting NEMO's whole triplet changes none
of the `kt=2` T/S unequal counts or maxima.  A material `1 m3/s` active-`zFw`
plant moves 87 T and 74 S cells downstream (maxima `1.77e-9 K` and
`1.44e-10`), proving the seam is consumed.  The surface-boundary endpoint
difference contains a small reconstructed residue, but endpoint subtraction
cannot identify the executed addend; the earlier surface-forcing attribution
is withdrawn.

The already-acquired stage-3 stream was also exhausted without overclaiming.
T at the stage-2 boundary and all three QCO ratios are exact.  A post-hoc
ratio-only replay is exact, whereas legoESM's reference-thickness diagnostic
has 86,821 unequal T words, maximum `3.552713678800501e-15 K` (2 ULP), and is
reconstructed exactly by retaining `e3t_0`.  This is not an owner: NEMO's
executed `tra_zdf` RHS and tridiagonal solve also retain `e3t_0` at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90:284-289`.  The report therefore
says `UNMEASURED_EXECUTED_TRAZDF_OPERANDS`, not “reference-thickness owner.”

The admitted scalar executable calls scalar `sin` and `cos` only.  It has no
vector transcendental and no `exp` call in the executed analytic boundary
routine.  The vector executable has both vector and scalar `sin`/`cos`, also no
`exp`.  With the scalar root, legoESM's production slow forcing and NEMO's
written left-associated wind composite are bit-exact; the right-associated and
reciprocal-first planted alternatives differ.  This closes the round-6 vector-
sine owner without changing production grouping.

One vertical variable was tested before requesting more evidence: replacing
only legoESM's step-entry `avt/avm` profiles with NEMO's recorded profiles
leaves the `kt=2` T and S maxima unchanged.  This refutes those entry
coefficients, but not the executed `tra_zdf` RHS, matrix, recurrences, or clamp.

The next writer captures `oracle_trazdf_matrix_kt00000001.bin` and
`...00000002.bin` in addition to all three step-2 RK state boundaries, the
ZDF-entry bundle, stage-1/2 tracer operands, stage-3 transport and accumulator
boundaries, and the isolated penetrating-solar term.  One-variable replays can
then separate first-step `tra_zdf` arithmetic from the large step-2 owner and
test vertical physics, solar/non-solar forcing, slopes, and the FCT limiter in
registered order.  No candidate is promoted before acquisition.

## Review repairs

All auxiliary binaries consumed by the discriminators are now hash-admitted,
including the decisive source/composition streams; a planted mutation of that
auxiliary digest refuses.  The daily acquisition runner also writes and
count-checks a payload-hash manifest for all 31 sparse records.  The source-
order and surface-boundary claims were narrowed to what their recorded
accumulators can support.

The Smagorinsky momentum-LDF discriminator no longer hard-codes `kt=2`:

- at `kt=2`, substituting NEMO's own `ahmt/ahmf` leaves maximum U/V increment
  residues `4.235164736271502e-22 m/s2`; the reverted final-F-mask plant moves
  them by about `1.15e-8 m/s2`, confirming the coefficient owner;
- at `kt=3`, the same substitution leaves maxima `7.515723340987407e-18`
  (U) and `8.51162232872165e-18 m/s2` (V).  Nearly every word differs because
  the trajectory inputs already differ.  The report says INCONCLUSIVE: this
  record lacks the stage-3 inputs needed to separate operator composition from
  inherited input debt.

The five-year forcing clock now subtracts completed 360-day years before the
seasonal cosines.  The exact test covers all 43,800 quarter-degree steps and is
bit-equal to a literal scalar replay.  Its non-vacuity arm finds more than
30,000 seasonal words moved by the old unbounded expression.  The first ten
shipped one-degree steps precede a year boundary and do not move.

The redundant completed T-point mask is removed from the dynamic mode-32
coefficient.  As with the already-fixed F point, the compiled branch masks the
deformation inputs, applies independent velocity lower limits, converts both
completed coefficients to their square-root bilaplacian form without a final
T/F mask, then halo-fills them.  The removal is inert on the acquired GYRE T
coefficients but makes the transcription literal.

## Thirty-day vector-flavour report

The daily root is admitted by the executable, namelist, manifest, and all 31
entry hashes.  The daily gate schedules and reads exactly the sparse writer
sequence `kt=1,25,...,721`; its registration adds the required sparse Nbb
names beyond the shared registry's inherited per-step `kt=1..60` range and
continues to refuse planted non-emitted neighbors `kt=72` and `kt=722`.  All
five day-0 one-ULP plants fire.

This is deliberately REPORT_ONLY because NEMO used the vector-math executable
while legoESM used the certified scalar-libm policy.  Selected maximum active-
field residues are:

| day | kt | T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1 | 25 | 1.655e-7 | 1.126e-8 | 8.753e-9 | 1.471e-8 | 9.674e-10 |
| 5 | 121 | 1.295e-5 | 2.342e-6 | 2.245e-6 | 1.618e-6 | 1.028e-7 |
| 10 | 241 | 3.588e-4 | 1.071e-4 | 1.335e-4 | 5.252e-5 | 1.064e-7 |
| 20 | 481 | 6.742e-4 | 6.323e-5 | 5.184e-5 | 5.468e-5 | 8.680e-7 |
| 30 | 721 | 2.071e-2 | 1.973e-3 | 1.678e-2 | 3.254e-2 | 4.420e-5 |

The preregistered day-30 branch prediction is partly confirmed and partly
refuted on NEMO's own before-state:

| point | active | lower | deformation | upper |
|---|---:|---:|---:|---:|
| T | 288,000 | 23,746 (8.245%) | 264,254 (91.755%) | 0 |
| F | 294,030 | 32,806 (11.157%) | 261,224 (88.843%) | 0 |

Relative to `kt=10`, lower-limit ownership falls and deformation ownership is
nonzero, as predicted.  The predicted natural upper-limit activation is
refuted: it remains zero at day 30.

## Shipped-deck invariant

The default one-degree card digest remains `99b2dfce324b0f77`.  Its registered
ten-step trajectory regression compares 70 certified rows with zero moved
cells, zero worsened cells, and zero maximum oracle-residual worsening.  Both
the pre-change and current reports first exceed the oracle bar at `kt=3` in
all five fields; that existing status does not change.

The GPU driver is untouched.  Its required physical-GPU argument, native-
policy label, daily snapshot/mean provenance, tiny-grid direct test, and
two-day scan/per-step comparison remain inherited.  The first two-day
composition difference remains step 3 SSH, with maximum surface-T difference
`1.8106049992638873e-10`; it is not a transcription difference and is outside
the CPU ladder's zero-bit claim.

## Acquisition handoff

The scalar step-2 writer preflight and scalar daily-writer build preflight both
pass.  The daily runner correctly refuses before the writer exists.  The
operator should run these exact commands from the committed clone, single
rank, without changing NEMO engine sources:

```text
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-1954525847 && EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_scalar_step2 bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/acquire_scalar_step2.sh --run
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-1954525847 && bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/build_daily_writer.sh --math-flavour scalar --run
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-1954525847 && EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_smag_daily_stage_scalar RANKS=1 bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/run_daily.sh --math-flavour scalar --run
```

Expected evidence roots are exactly:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_scalar_step2/`;
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_smag_daily_stage_scalar/`.

## Tests and evidence

Round-7 artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/`:

- `r4_scalar_ladder.json`, `r4_vector_ladder.json`,
  `r4_smag_scalar_ladder.json`, and `r4_smag_vector_ladder.json` contain the
  four admitted ten-step scores;
- `r4_smag_vector_30d_ladder.json` contains all 31 day-boundary scores and the
  NEMO-state branch census;
- `r4_scalar_stage2_composition.json` contains the clean-stamped NEMO-RHS raw
  update discriminator, accumulated source-order walk, two firing one-ULP
  controls, and material RHS/`zFw` non-vacuity controls;
- `r4_scalar_stage3_geometry.json` records the bounded post-hoc ratio replay
  and explicitly withholds an executed-`tra_zdf` owner;
- `shipped_trajectory.json` and `shipped_trajectory_regression.json` prove the
  one-degree no-movement invariant;
- `acquisition_preflights.log` records the operator-tool preflights.

Verification from clean committed trees:

- ladder, daily-ladder, acquisition, citation-gate, and independent
  Smagorinsky tests: 63 passed, 1 slow test deselected;
- GYRE card, dispatch, and 43,800-step calendar tests: 10 passed, 25 unrelated
  tests deselected;
- GPU-selection, deck-cadence, climate-refusal, and movie-refusal tests:
  16 passed, 1 slow parity test deselected;
- compiled-source citation gate: PASS for all 13 receipt citations; the
  explicit shifted executed-`tra_zdf` citation plant: FIRED;
- scalar step-2 writer preflight and scalar daily-writer build preflight:
  PASS; daily run preflight: expected fail-closed refusal until the compiled
  scalar daily writer exists.  The runner writes and count-checks a manifest
  containing all 31 record payload hashes after acquisition.

The numerical artifacts above were all produced from clean commits and record
those commit IDs in their provenance.

## Inherited pins / OPEN

Inherited unchanged are the namelist-parameterised shipped, `r4`, and
`r4_smag` cards; 31 levels; RK3; split-explicit free surface; TKE/EVD; Redi
tracers; analytic seasonal forcing; closed free-slip walls; daily climate-file
conventions; the exact CPU fp64/scalar-libm ladder bar; and all one-ULP plants.
The shipped one-degree deck remains the default.  The registered pre-fix and
post-fix five-year GPU runs retain their immutable commit/policy labels; this
round launches neither run.

OPEN:

1. Acquire the scalar step-1/2 `tra_zdf` matrix/RHS plus step-2 RK/operand
   records.  First close the `kt=2` T/S owner from the executed step-1 solve,
   then perform one-variable step-2 replays until the term producing the
   `3.339e-9 K` owner is named.
2. Acquire and admit the scalar-math 30-day root before giving any bit-level
   certification beyond ten steps.  The existing vector report is not that
   certification.
3. The naturally active Smagorinsky upper-limit branch remains unmeasured at
   both ten steps and day 30; only its synthetic unit arm is exercised.
4. The `kt=3` momentum-LDF record cannot isolate operator arithmetic from its
   already-diverged stage inputs; this is documented debt, not an exactness
   claim.

## Compiled-source citations

- The scalar build subtracts completed calendar years and evaluates the two
  seasonal cosines at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90:92-106`.
- Its analytic heat, solar, freshwater, and restoring expressions contain the
  executed trigonometric calls and no exponential at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90:121-145`.
- Its seasonal wind clock and latitude sine are at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90:176-191`.
- The before-level vertical-physics call, slope call, external mode, and all
  three RK stage calls are ordered at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3.f90:168-216`.
- The stage tracer-transport operands and write boundary are at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:604-634`.
- Stage 2 accumulates HPG, vorticity, and advection in that order and writes
  each resulting RHS boundary at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:396-412`.
- The stage-2 raw U/V update statements are at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:461-462`; the subsequent
  reference-thickness barotropic reduction and correction are at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:506-526`.
- The surface, solar, lateral-diffusion, and vertical-diffusion tracer calls
  are ordered at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:691-725`.
- The executed stage-3 tracer solve retains reference thickness in its surface
  RHS and interior tridiagonal RHS at
  `GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90:284-289`.
- Dynamic mode 32 builds, clamps, and square-root converts the T/F coefficients
  without a completed-point mask at
  `cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:462-528`, followed by the halo
  exchange at `cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:512-534`.
- The daily writer's sparse condition and Nbb payload are compiled at
  `GYRE_OMIP_L2_P3_DAILY/BLD/ppsrc/nemo/stprk3.f90:90-99`.
