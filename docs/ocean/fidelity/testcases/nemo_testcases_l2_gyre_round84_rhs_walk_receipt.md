# NEMO testcase L2 GYRE phase 3 — round 84 cumulative momentum-RHS receipt

## Outcome

**HELD.** The admitted kt2 stage-1 record and the current production-JIT
CPU/fp64/libm path were walked through every compiled cumulative momentum-RHS
boundary. The frozen HPG prediction is **REFUTED**. HPG is bit-exact on all
17,400 U and 17,100 V wet face-levels. LDF is the first non-bit cumulative
boundary, at `2.5292467120726215e-14` U and `3.502735092670824e-14` V. The
large residual first appears when ZAD is added: its incremental residual
maxima are `1.9220297482797664e-09` U and `1.966061294804274e-09` V.

Exact operator ownership is deliberately withheld. The source-ordered
accumulator differs from the independently captured live total by
`8.470329472543003e-22` on both faces (6,887 U and 6,566 V cells), so the
preregistered closure requirement fails. This is a measured association or
unregistered-contribution boundary, not permission to label the ZAD formula
itself wrong. No production physics changed, and neither the Decision-37
absolute-history candidate nor the Round-82 drag/inverse pair is promoted.

## Preregistration, records, and evidence

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round84.md`, committed as
`c1ccf9766f09b5983616d4ee977074388edbde62` before any Round-84 scientific
comparison. The new gate and its unit controls were committed as
`861bc579ed5d98f3d68dcb2dc6cadee4513792cf` before the sealed runs. Evidence
is under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round84/`.

The inherited Round-64 admission again passed its 43 byte-identical records,
20 classified changed records, and 132 admitted differences. The kt2 stage-1
projection has zero owned-field differences; its seven raw differences are
the previously admitted undefined `ww` halo values. The producer commit is
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`.

The required issue-#1455 coordination post could not be made: the installed
GitHub CLI reports its configured token invalid, and no connected GitHub app
is available in this sandbox. No external issue state was mutated. Round 85
must post this receipt's exact HPG/LDF/ZAD/closure findings to #1455 once
authenticated access is available.

The sealed ordinary artifact is `rhs_walk.json`, SHA-256
`faeaff7279a8474f440b3f6d8f2c4ee4e310c27c7cbd5c63ff6c59a49af3f39f`.
Its status is **REFUTED** and its command exits 1. The complete boundary rows
are:

| boundary | U absolute max | V absolute max | classification |
|---|---:|---:|---|
| after HPG | 0 | 0 | bit-exact |
| after LDF | 2.5292467120726215e-14 | 3.502735092670824e-14 | first non-bit |
| after VOR | 2.5292467332484452e-14 | 3.502735092670824e-14 | non-bit |
| after KEG | 2.5292467332484452e-14 | 3.502735092670824e-14 | non-bit |
| after ZAD | 1.9220276136604423e-09 | 1.966059508480186e-09 | first magnitude boundary |
| after ADV snapshot | 1.9220276136604423e-09 | 1.966059508480186e-09 | identical to after ZAD in NEMO |

HPG's T, S, SSH, density anomaly, live and reference thicknesses, reciprocal
metrics, masks, and handed U/V arrays are all bit-exact. The only scored
non-bit HPG input is `r3t` (600 cells, maximum
`1.1031574463560756e-16`), yet the complete HPG result is bit-exact. This
directly kills the predicted density boundary: density differs at zero cells,
not merely below a tolerance. The private trace is non-interfering: T, S, U,
V, and SSH all match an independently compiled ordinary step bit-for-bit.

All three maximum-residual one-ULP controls printed `PLANT_FIRED` and exited
1:

- HPG result: `plant_hpg.json`, SHA-256
  `b13e4ed567cb284d22f151e327530a2fee0a45323b6222925abe5d15a10204e5`;
- density input: `plant_density.json`, SHA-256
  `68ecf65ff5a19fee6477d824bec8ffd12a7d02ed581fec66fef303ae33a22278`;
- live-total closure: `plant_closure.json`, SHA-256
  `3f9d2d41cc9edda7b1c5164f48a636f3bc52ef932a72a476d73c15c6a56c1b2b`.

Each plant changes the registered maximum and the gate status. The HPG and
density controls each create exactly one differing wet cell from an exact
baseline; the closure control moves the already non-exact maximum-residual
cell farther away. No dry, zero, overwritten, or aggregate-invisible plant is
accepted.

## Refutations and instrument limits

The following failed statements remain first-class output.

1. “HPG is the first non-bit and first magnitude-bearing cumulative boundary”
   is **REFUTED**: HPG is exact; LDF is first non-bit; ZAD is the magnitude
   boundary.
2. “Density is the first non-bit HPG input” is **REFUTED**: density is exact;
   `r3t` is non-bit, but it leaves the complete HPG output exact.
3. “The reconstructed source-order final accumulator is bit-exact with the
   live total” is **REFUTED** at `8.470329472543003e-22` on both faces.
   Therefore no exact LDF or ZAD formula ownership is claimed.
4. The first focused citation test exposed three ambiguous endpoint tokens in
   the new map. No citation was accepted from that run. The entries were
   repaired with occurrence-pinned endpoints before the clean-tree gate.

The gate sees already-computed production term arrays and their cumulative
association. It cannot by itself distinguish a wrong ZAD input from wrong ZAD
arithmetic, and the failed live-total closure prevents it from excluding a
later association contribution. Those are the next registered measurements.

## Compiled-source findings

The admitted record's own compiled branch overwrites Krhs with HPG, adds LDF
and VOR, then calls KEG and ZAD; its `after_adv` write-only snapshot immediately
follows ZAD, and the final vertical average consumes that accumulator
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`,
`:202-207`). These are the exact executing boundaries scored above.

The compiled active s-coordinate HPG forms the surface and interior pressure
gradients and overwrites U/V Krhs at each level
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynhpg.f90:378-435`). The measured
HPG boundary is bit-exact; no HPG change is eligible.

The producer's runtime record resolves dynamic lateral mixing to the div-rot,
iso-level Laplacian with constant viscosity
(`round64/oracle_krhs_split/ocean.output:682-708`). The compiled dispatcher
selects `dynldf_lev_lap` for the Laplacian case
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf.f90:81-90`), whose executing
rotational branch builds the `zwf`/`zwt` intermediates and adds their written
differences to U/V Krhs
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`). This
addition is the first measured non-bit statement boundary. Its small scale
does not explain the campaign's final RHS magnitude, and exact formula
ownership is withheld pending closure and operand scoring.

The compiled ZAD routine initializes its carried vertical-advection workspace,
forms the `ww` transports and Kmm vertical velocity differences, and updates U
and V Krhs at interior and bottom levels
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138`). The large
residual first appears across this cumulative addition. This establishes the
next measurement boundary, not whether `ww`, face thickness/r3, velocity,
metric association, or the update statement is its owner.

## Rule-12 table

| Lane | Registered disposition |
|---|---|
| GYRE kt2 cumulative RHS | **MEASURED / NO CANDIDATE.** HPG is exact, LDF is first non-bit, and the large gap appears at the ZAD addition. Exact operator ownership is withheld because live-total closure is non-bit. |
| GYRE kt=1..10 | **PRESERVED, NOT RERUN.** No production physics changed. The immutable before arm remains `decision36_nemo_face_shear/after_kt1_10.json`; first-over-bar remains kt2 U/V. |
| Every moved GYRE row | **NONE.** This round adds only a diagnostic, tests, citation metadata, preregistration, and receipt. |
| GYRE days 1..30 | **PRESERVED, NOT RERUN.** No eligible short-ladder candidate exists. The immutable before arm remains `decision36_nemo_face_shear/after_day_gap.json`. |
| LOCK_EXCHANGE-zco | **PRESERVED, NOT RERUN.** It shares the split-explicit program, but no shared production statement changed and no neutrality claim is made. Its recorded kt rows must be measured if an RHS fix becomes eligible. |
| OVERFLOW-zps | **PRESERVED, NOT RERUN.** No shared production statement changed and no neutrality claim is made. A future shared-statement change must show non-execution or score its recorded kt rows. |
| DINO | **SHARED-STATEMENT RISK.** DINO's leapfrog card has separate histories but executes the shared HPG/LDF/VOR/KEG/ZAD implementations. Its 96–98% regional cancellation forbids neutrality inference from GYRE. |
| ORCA2 | **UNMEASURED WITH SPEC.** Independently align every cumulative RHS boundary and every HPG/LDF/VOR/KEG/ZAD operand, T/S/U/V/SSH, masks, live/reference geometry, and six histories on native staggering. Require elementwise fp64 bit equality and normalized L-infinity through kt=1..10 at bar `1e-15`; reject any AT-BAR loss, earlier first-over-bar, or wet-point operand/history mismatch. |

No restart/checkpoint representation, configuration, selector, coefficient,
timestep, stabilizer, carried state, year harness, reconciliation gate,
freshwater pair, #1484 guard, held manifest, or NEMO source changed. The
Decision-37 loud-failure restart contract remains untouched.

## Review and focused verification

The required separate read-only review command was attempted against the
complete diagnostic diff and sealed evidence. It failed before a review model
started, so there is no `SHIP` or `DO NOT SHIP` verdict. Its terminal line is
quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**;
work continued. The complete output is `round84_codex_review.txt`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No review approval is claimed and no physics is shipped.

The final focused CPU/fp64 suite covers the Round-83 predecessor, all new
Round-84 helpers and controls, and the receipt citation gate: 22 tests pass.
The citation gate passes every mapped compiled-source citation from this
receipt on a clean tree, and its shifted-citation plant exits 1 with
`SYMBOL-NOT-AT-LINE`. Both changed Python scripts compile, and
`git diff --check` passes. The known unrelated
`test_rk3_ws_differs_from_rk3_and_is_finite` failure was not encountered.

## Choices and uncertainty

Choices made: none. Decision 37 remains YES, but its candidate remains held
under its existing trajectory condition. This round makes no configuration or
carried-state choice.

The measured first bit boundary is LDF and the measured magnitude boundary is
the ZAD addition. The unresolved uncertainty is explicit: exact ZAD ownership
requires operand/formula scoring, and exact downstream ownership requires the
source-order reconstruction to close bit-for-bit to the live total.

## OPEN — exact handoff to round 85

1. Reuse the admitted Round-64 stage-1 record; no NEMO acquisition is needed.
   Preregister a ZAD walk because that addition introduces the `~1.9e-9`
   residual. Score, in compiled statement order, the incoming cumulative Krhs,
   `ww`/`wsd` branch, Kmm U/V, live face thickness and r3u/r3v factors,
   e1e2 metrics/reciprocals, masks, the carried `zWdzU/zWdzV`, interior update,
   and bottom update. Use given-NEMO-input literal replay before proposing a
   shared implementation change.
2. Reconcile the `8.470329472543003e-22` source-order/live-total closure
   boundary. Capture the live total immediately at the same stage-1 routine
   boundary or enumerate every post-ZAD contribution; do not attribute ZAD
   until closure is bit-exact.
3. Keep the LDF `2.53e-14`/`3.50e-14` first-non-bit statement as explicit
   fidelity debt. After the magnitude ZAD owner is resolved, walk its Kbb
   operands and div-rot Laplacian statement; no tolerance may discharge it.
4. Keep the Round-79 history arm and Round-82 drag/inverse pair held. Revisit
   only after the upstream RHS owner is exact and only with the immutable
   Decision-36 kt1..10/day-30 before arms and full Rule-12 lanes.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
