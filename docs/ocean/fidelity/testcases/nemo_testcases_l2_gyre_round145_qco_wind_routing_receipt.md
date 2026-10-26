# NEMO testcase L2 GYRE round 145 — QCO wind reciprocal routing

Date: 2026-09-21

Status: **HELD**.  The one-variable candidate is source-exact at the developed
wind boundary, but it worsens day-30 T RMS by `2.3052112600513083e-11 K` and
day-360 T RMS by `9.799620172799839e-8 K`.  Decision 43(a) and Decision 45(f)
therefore refuse it.  Production was restored and the exact candidate is
preserved at
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round145_qco_wind_held.patch`.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round145.md`, committed at
`683f58206d06a640e89c0a6e6b1786d00e102001` before candidate measurement.
The measured clean candidate is
`c08132745455b382c610a21d94c20eeabb4be576`.  All evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round145/`.

## Compiled statement and candidate

The record-producing compiled program materialises `r1_hu_0` and `r1_hv_0`
at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90:212-215`.
Its QCO program enters with the step-entry SSH at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:160-182` and constructs
the surface-area-weighted U/V ratios at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:188-218`.
The external step then adds wind as the recorded stress times
`r1_hu_0/(1+r3u(Kbb))` and `r1_hv_0/(1+r3v(Kbb))` at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:249-250`.

The candidate added no formula, state, stabiliser, or selector.  It requested
reciprocals from the already shared `_nemo_ws_qco_stage_faces` evaluation and
routed those two outputs to the wind statement.  The momentum-advection face
thickness continued to consume the same helper result.

## Developed-state local proof

The production-JIT walk starts from NEMO's admitted day-180 state.  The
round-145 mode deliberately drops round 143's obsolete pre-candidate residual
pins while retaining the record, callback, plain-call, and row-coverage
controls.

| production row | unequal / wet | max absolute difference | verdict |
|---|---:|---:|---|
| live incoming U | 0 / 580 | 0 | BIT |
| live incoming V | 0 / 570 | 0 | BIT |
| live final U | 268 / 580 | `6.882142696441190e-22` | downstream residue |
| live final V | 247 / 570 | `8.470329472543003e-22` | downstream residue |

Thus the routed live boundary, not a recorded substitution, is the first
closing family.  Callback versus plain external-call state is BIT over all
`198,956` cells in 23 leaves.  The production-JIT inverse-depth one-ULP plant
printed `ROUND145 WIND WIND-INVERSE-DEPTH-ULP STATUS PLANT-FIRED` and exited
1.  The ordinary run is stamped clean at the measured candidate commit.

Two preliminary attempts are retained but not cited as results.  First, the
round-144 mode refused the candidate because its old round-143 residual pins
describe the before arm; that instrument mismatch motivated the same-harness
round-145 mode.  Second, the first year scoring invocation pointed at the
30-day `year_owners` layout and refused; the certified from-rest NEMO root is
`phase3/year_fromrest`, used by every table below.

## Certified ladder and moved-row registry

The base clone at the preregistration commit reproduced every frozen headline.
The in-run 70-row comparison is stamped clean, registers all 58 moved rows,
has no missing or unexpected registry entries, loses no kt1 AT-BAR row, and
keeps first-over-bar at kt2 U/V.

| headline | base | candidate | result |
|---|---:|---:|---|
| kt2 T max abs | `1.4210854715202004e-14` | `1.4210854715202004e-14` | unchanged, AT-BAR |
| kt2 S max abs | `2.1316282072803006e-14` | `2.1316282072803006e-14` | unchanged, AT-BAR |
| kt2 U RMS | `2.7377110452773967e-12` | `2.7377110452773967e-12` | unchanged |
| kt2 V RMS | `3.2849219221489645e-12` | `3.2849219221489645e-12` | unchanged |
| kt3 T RMS | `8.659373840202989e-7` | `8.659374053365809e-7` | worsened |
| kt3 S RMS | `7.027291104577671e-8` | `7.027291815120407e-8` | worsened |

The complete exact-name registry is `round145/moved_rows.tsv`; it consists of
kt2 `after.uu_b/vv_b`, kt3 through kt10 `after.uu_b/vv_b`, and kt3 through
kt10 each of `before.T/S/u/v/ssh`.  This is 2 + 16 + 40 = 58 rows.  Per-row
cell counts, improved/worsened counts, maxima, and row-scale ULP values are in
`ladder_trajectory_comparison_certified.json`.  The largest worsening is
`2,351,188.4765625` row-scale ULPs at `GYRE-zco.kt10.before.ssh`.

## Month and year landing gate

| day | base T3D RMS (K) | candidate T3D RMS (K) | candidate - base |
|---:|---:|---:|---:|
| 30 | `6.890431487825909e-5` | `6.890433793037169e-5` | `+2.3052112600513083e-11` |
| 60 | `1.932998392040720e-4` | `1.932997894194814e-4` | `-4.978459064810571e-11` |
| 90 | `1.864501723318785e-3` | `1.864499073662812e-3` | `-2.649655973145995e-9` |
| 120 | `1.050124876579511e-3` | `1.050120120259947e-3` | `-4.756319563247249e-9` |
| 180 | `3.580550293289633e-3` | `3.580548221494673e-3` | `-2.0717949599825813e-9` |
| 240 | `1.644674023317539e-2` | `1.644674015195974e-2` | `-8.121564878948995e-11` |
| 300 | `1.359740243654708e-2` | `1.359740181010901e-2` | `-6.264380715548246e-10` |
| 360 | `1.122357124784366e-2` | `1.122366924404539e-2` | `+9.799620172799839e-8` |

The preregistered strict day-30 decrease and non-worse day-360 predictions are
**REFUTED**.  The day-240 non-worsening prediction is confirmed, but its
`8.12e-11 K` improvement is not enough to overcome either veto.  The
Decision-43/45 gate reports FAIL with every other criterion true: month/year
day 30 agree, all eight year rows and all 58 ladder moves are registered, all
executing cards are discharged, first-over-bar is not earlier, and kt1 stays
at the bar.

## Certified-card blast radius

The recipe-derived execution table, rather than a hard-coded list, is:

| card | executes candidate | measured result / exclusion |
|---|---|---|
| GYRE-zco | yes | 70-row ladder plus month/year above |
| NEMO-GYRE-recipe | yes | all 15 certified three-step state rows move; certification remains PASS |
| LOCK_EXCHANGE-zco | no | `surface_stress_implicit=False` |
| OVERFLOW-zps | no | `surface_stress_implicit=False` |
| DINO:nemo_dino_kamm | no | Euler momentum and `surface_stress_implicit=False` |
| DINO:nemo_dino_kamm_mlf | no | Euler momentum and `surface_stress_implicit=False` |

The generic recipe registers `T/S/u/v/eta` at each of steps 1, 2, and 3.  Its
largest candidate movement is `2.1891779567514286e-10` in step-3 V; the full
15-row before/after hash table is in `generic_card/comparison.json`.

The DINO route predicate is false on both resolved shipped cards, so the
candidate code is not executed there.  As a real-card control, the same
`tests/ocean/unit/test_dino_experiment.py` suite reported `128 passed, 9
warnings` on both clean base and clean candidate arms.  LOCK_EXCHANGE and
OVERFLOW are similarly outside the exact source condition, so no tank row can
execute this statement.  ORCA2 is **UNMEASURED-WITH-SPEC**: its ocean-only
card must resolve QCO plus implicit surface stress and run its own 10-step
ladder before this result can be generalized; its SI3 selector gap remains a
separate campaign decision.

## Evidence and controls

| artifact | SHA-256 |
|---|---|
| `local_production_jit_final.json` | `fceaec9e736b00c0fe32312658bd1e895088a98757c51c10a5eaa44a89b05a15` |
| `local_inverse_depth_ulp_plant.json` | `586e4ea20d9a40db2abeb70a0540163f8c378d42cae7fae610a4c13ee179a479` |
| `ladder_trajectory_comparison_certified.json` | `a09b264ff7bf6c2fd7780f4fd627c91313ba05adf05f62754d2194dd2f26d301` |
| `decision43_45.json` | `3bd216487d7470d2cae002f3633334ca728cf566b098dc82be796b98550783ba` |
| `generic_card/comparison.json` | `ab473c39e250d004d1441ad45a30ec9c89bb632ef2ea3b8321b28b07592130ea` |
| `moved_rows.tsv` | `de3b39aee07f8f84288bda763287e5efaa4eb4725dc5acc4eb7aa07459a409bb` |
| base / candidate month | `a7a2b0f3d5b6c6256370df04e16efb294086c32a01a2cd6f52d2dc6f71ac3fc` / `9ad052fa8c59c13a51844c90094a025bebeb22c8115436cb90326e7506fc0517` |
| base / candidate year | `56a83b6c3fd05ec1523670271fa2c21a3a52160f309a65ccb36cf8839d1a7566` / `8c9ce0763af343855dcfe4c19c9456f5e1bcfbb86ae70c607aefb40dfbae6bf2` |

The required separate read-only review was attempted against the final held
diff.  It was unavailable inside the sandbox and emitted verbatim:
`Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`.  It returned 1 before a `SHIP`/`DO NOT SHIP` verdict;
per the standing operator rule this is recorded as **independent review
unavailable in-sandbox**, not treated as approval.  No physics remains to ship.

The receipt citation gate reported `status: PASS`, 4 citations, 0 unmapped, 0
failures, and 0 failing map entries.  Shifting the exact wind-statement
citation by two lines reported `status: FAIL` and
exited 1.  The preregistration's four citations independently passed the same
gate.

Test summaries:

* candidate and base DINO real-card suites each reported `128 passed, 9
  warnings in 113 s`;
* the final candidate-sensitive five-file suite reported `63 passed in
  247.38s`;
* the mandated combined ocean fidelity/unit run collected 8,379 tests and
  reached 94%, but six xdist workers aborted inside unrelated JAX compilation
  and the replacement pool stopped emitting progress; it emitted **no summary
  line** before infrastructure termination.  A lower-concurrency split of the
  complete 1,533-test fidelity tree reached 98% with one documented known red
  and likewise stopped emitting progress; it also emitted **no summary line**.
  These are retained in `full_ocean_tests.log` and
  `full_fidelity_split.log`, not represented as passing suites;
* every test named by a worker-abort traceback was rerun in a fresh serial
  process.  The six summary lines were respectively `1 passed in 29.67s`, `1
  passed in 8.46s`, `1 passed in 22.29s`, `1 passed in 10.98s`, `1 passed in
  116.67s`, and `1 passed in 1.47s`.  The DINO aborting node is covered by the
  complete 128-test DINO pass above.

The aborted nodes were partial-cell grid construction, leapfrog dispatch,
NEMO MLF dispatch, bottom drag, ocean SCM, DINO wind-from-rest, and Sweeney
shortwave.  Their isolated passes prove the aborts are the documented
large-suite compiler-pressure failure, not round-145 failures.  The known-red
inventory remains the campaign state's worktree-stamp ratchet, RK3-WS/MXL3,
SI3 provenance, root ratchets, and two TKE coefficient tests.  The changed
candidate-sensitive set introduced no new red.

## OPEN — round 146

Do not revive this downstream wind candidate: it closes its source statement
but fails both trajectory horizons.  Return to the larger developed-state
owner already measured by round 142: the completed three-dimensional momentum
RHS removes about 69% of the U and 75% of the V incoming slow-forcing error,
whereas round 145 closes only the residual wind boundary.  Rank the compiled
RHS term families at the admitted day-180 state by a directed one-family
production substitution and carry the largest family through the day-240 year
gate.  If the existing round-140 record lacks term-level fields, preregister a
passive, no-observer acquisition before changing physics.  Do not chase the
remaining `1e-22` Coriolis rounding floor.

`ACQUISITION_NEEDED`: NONE for round 145.  `DECISION_NEEDED`: NONE.
