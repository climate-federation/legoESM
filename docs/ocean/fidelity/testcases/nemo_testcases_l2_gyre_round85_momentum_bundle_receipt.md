# NEMO testcase L2 GYRE phase 3 — round 85 momentum-bundle receipt

## Outcome

**LANDED.** Decision 38's first momentum bundle combines three statements that
are independently exact when supplied NEMO's recorded inputs: the six absolute
barotropic histories already present at the incoming tip, the stored
tracer-cell bottom-drag coefficient plus external-window entry reciprocal, and
stage-specific ZAD operands.  The preregistered GYRE kt2 whole-state targets
both moved toward NEMO by the predicted amounts:

| target | immutable Decision-36 before | Round-85 candidate | disposition |
|---|---:|---:|---|
| kt2 U absolute maximum | 2.7478404751243857e-12 | 2.7377110452773967e-12 | improved |
| kt2 V absolute maximum | 3.305560306813421e-12 | 3.284922138989399e-12 | improved |

No AT-BAR row became DEBT and first-over-bar remains kt2 U/V.  The separate
Decision-38 gate passes and all three of its planted violations exit 1.
LOCK_EXCHANGE and OVERFLOW are each bit-identical to the recorded post-history
baseline on all 50 certified trajectory rows.  The day-30 temperature RMS gap
also decreases, from 1.2397568272314757e-02 K to
1.2397011295506804e-02 K.

This is not an identity claim.  The candidate ladder remains DEBT and the
compiled LDF addition remains the first measured non-bit stage-1 momentum
statement.  The bundle lands because the exact local statements and the
explicit Decision-38 acceptance rule pass, not because later compensation has
been eliminated.

## Frozen registration and implementation

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round85.md`, committed as
`8be5f40001eb09dc39f86acb0c7ee0b5da55b6cb` before any Round-85 scientific
measurement.  The production bundle and its mechanical Decision-38 gate were
committed as `928fdcab9ee829d922d51ca8d7372f19b2d431af`.  All measurements use
that clean commit, CPU, JAX fp64, and the repository precision policy.  Evidence
is under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round85/`.

The implementation preserves one shared NEMO-identity path.  It materializes
the standard drag law at tracer cells before face averaging, constructs entry
inverse depths in NEMO's written reciprocal/division association, preserves the
source multiplication order in explicit drag, and hands each WS-RK3 stage its
own W, live thickness, and r3 ZAD operands.  No selector, coefficient,
timestep, stabilizer, freshwater pair, reconciliation gate, year harness,
#1484 guard, held manifest, or NEMO source changed.  Decision 37 already
authorized the inherited absolute histories; their restart loading and loud
legacy-format failure contract remain intact.

The required issue-#1455 coordination post could not be made.  The installed
GitHub CLI reports the active account's token invalid, and no authenticated
GitHub connector is available in this sandbox.  No external issue state was
mutated.  Round 86 must post this receipt's bundle, Rule-12, tank, day-30, and
remaining LDF/ZAD-path findings once authenticated access is available.

## Local exactness proofs and controls

Each member was re-proved on the final clean candidate before bundling:

1. Absolute history replay reaches bit equality through every registered
   substep-1 history and midpoint row.  The first remaining raw boundary is
   substep-2 `un_e`, maximum 3.032539284029834e-09.  Ordinary artifact
   `history_exact.json`, SHA-256
   `54c9936f4f59a21188c964fca9c821887f8dc498e8bc371c5308b08be130b2d0`;
   one-ULP plant `history_plant.json`, SHA-256
   `88030db688d96d8e944b93d6e978a2d0c08b30cd811db4babd8a6b7699c97e6c`,
   exits 1 and prints `ROUND79 HISTORY PLANT FIRED`.
2. Drag, entry depth/reciprocal, and explicit drag trends are bit-exact on the
   recorded first external substep.  The first remaining boundary is imported
   `slow_u`, maximum 1.0529650291768787e-11.  Ordinary artifact
   `drag_inverse_exact.json`, SHA-256
   `49217b4de90d4b6b85dbc483132f94c7c1249cedd9cacf2afc9ecf006fd96507`;
   null-slow-U plant `drag_inverse_plant.json`, SHA-256
   `dbe8b94999a2751b7bcdc8db192cf24111f216d438e5893617feb497c7ce8831`,
   exits 1 and prints `ROUND82 NULL-SLOW-U PLANT FIRED`.
3. Literal WZV/KEG/ZAD replay is bit-exact at all 30 registered kt1/kt2,
   stage-1/2/3, U/V calibration cells.  This proof is deliberately scoped to
   NEMO's own operands: the broader model-path advection bucket remains non-bit
   at about 1.9e-9, so no upstream-input debt is retracted.  Ordinary artifact
   `zad_given_exact.json`, SHA-256
   `3e48633e2561c21cca6ddd1e04d852b48be7a8121db297c9547305d57458b04d`;
   given-input plant `zad_given_plant.json`, SHA-256
   `33490912a15e712658532e22c06e278d0daf374898c3792ce83d3c9c4271b4be`,
   exits 1.

## GYRE Rule-12 adjudication

The canonical kt1..10 artifact is `bundle_kt1_10.json`, SHA-256
`003f8f63fba7326fad98f1dc3064c4eaf0dc9d616f306c184816ba71d78f8dd1`;
its fail-closed residual sidecar is SHA-256
`d5ce82ed1fa06b210e7adb16bab27503aa176b51403921c35623853abef31893`.
As expected, the canonical fidelity gate returns DEBT.  The frozen
Decision-38 comparison is `bundle_rule12.json`, SHA-256
`1f971f4733ec7c69278dae988b94e8bfc2d9b7e1db505586fcaa8932328bd08a`,
and returns PASS.

The old two-ULP movement criterion returns FAIL for this deliberately bundled
candidate; its complete violations are retained in the Decision-38 artifact.
Decision 38 explicitly permits those downstream movements only because both
fields at the first-over-bar boundary move toward the bar.  There are 80 moved
rows, all registered here without elision:

- kt1: `bt.jn02.drag_u`, `bt.jn02.drag_v`, `bt.jn07.trd_u`,
  `bt.jn09.trd_v`, `bt.jn25.trd_v`, `bt.jn27.trd_v`, `bt.jn29.trd_v`,
  `bt.jn39.trd_v`, `bt.jn40.trd_u`, `bt.jn45.trd_u`, `stage3.u`, and
  `stage3.v`;
- kt2: `after.uu_b`, `after.vv_b`; U/V for each of
  `arm.omit_barotropic_substep_drag`, `arm.omit_momentum_transport_reconcile`,
  `arm.omit_stage_barotropic_correction`, and
  `arm.omit_surface_boundary_forcing`; and `before.u`, `before.v`;
- every kt3 through kt10: exactly `after.uu_b`, `after.vv_b`, `before.S`,
  `before.T`, `before.ssh`, `before.u`, and `before.v`.

The kt3 preregistered numerical prediction is **REFUTED and retained**.  T was
predicted to worsen to about 3.722344652268031e-04 K but instead improves from
1.6275114177588534e-04 to 1.627497246303733e-04 K.  S was predicted to worsen
to about 3.683245441046983e-05 but instead improves from
6.327755180279837e-06 to 6.327735185607253e-06.  This favorable sign was not
used to select the candidate.

All three Decision-38 controls are non-vacuous and exit 1:
`target-u-worse`, `at-bar-to-debt`, and `earlier-first-over-bar`.  They
respectively reverse the registered U target, demote a real AT-BAR row, and
advance the first-over-bar step.

The 30-day score is `bundle_day_gap.json`, SHA-256
`5212be0f260ac5b6b35fd4e51d4347d7b5d1921c3101b2e082c7c424e35a014c`.
The preregistration made no day-30 sign prediction.  T and S improve on all
30 days; U improves on 26 and worsens on 4; V improves on 28 and worsens on 2;
SSH improves on 29 and worsens on 1.  At day 30 all five RMS scores improve:
T by 5.569768079527471e-07 K, S by 1.0742593209520784e-08, U by
4.29349159923988e-08 m/s, V by 4.3288699797818485e-08 m/s, and SSH by
5.6799830044711955e-08 m.

| Rule-12 lane | Result |
|---|---|
| GYRE kt1..10 | **PASS under Decision 38.** Both kt2 U/V targets improve; no AT-BAR loss; first-over-bar remains kt2 U/V; all 80 moved rows are listed above and sealed cellwise. |
| GYRE days 1..30 | **MEASURED / FAVORABLE.** Full daily table sealed; day-30 all fields improve, while the six earlier U/V/SSH worsening rows remain reported. |
| LOCK_EXCHANGE kt1..10 | **PASS.** 50/50 rows bit-unchanged, zero worsening ULP, first-over-bar remains kt4 U. |
| OVERFLOW kt1..10 | **PASS.** 50/50 rows bit-unchanged, zero worsening ULP, first-over-bar remains kt2 T/U. |
| DINO | **SHARED-STATEMENT RISK.** Its leapfrog history and non-WS-RK3 ZAD paths do not execute the corresponding two bundle members, but its standard drag law shares the materialization statement.  No numerical neutrality is inferred, especially given DINO's 96--98% regional cancellation. |
| ORCA2 | **UNMEASURED WITH SPEC.** Independently align an ORCA2 NEMO restart and legoESM NEMO-identity state at the same Kbb frame.  Score T/S/U/V/SSH, six absolute histories, tracer/face drag coefficients, entry depths/reciprocals, and ZAD W/thickness/r3 operands on native tmask/umask/vmask staggering by elementwise fp64 equality and normalized L-infinity for kt1..10 at bar 1e-15.  Falsifier: any AT-BAR loss, earlier first-over-bar, or wet-point operand/history mismatch. |

LOCK_EXCHANGE comparison `lock_comparison.json`, SHA-256
`c84badf12f244c40f4cb309b7880892a3553b1522d2af2d8e01db931d2bdc08b`;
OVERFLOW comparison `overflow_comparison.json`, SHA-256
`b3eb2da2c15c617ad84bdfdfc94532e930932f7ba73906b75d78b1d5c37d91a7`.
The exact candidate trajectories and hashed residual sidecars are retained
beside those comparisons.

## Compiled-source basis

The history member follows the recorded GYRE branch that seeds current
external velocity from the instantaneous NEMO state while keeping the absolute
history arrays, forms the AB3-AM4 midpoint directly from current/b/bb values,
and rotates those absolute arrays after each substep
(`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378`,
`:481-509`, `:783-795`).  There is no deviation-history reconstruction in the
executing statements.

The drag member follows the later recorded branch.  Standard drag first stores
the square-root coefficient at the tracer point
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-235`).  At window
entry NEMO separately materializes face depth and computes its reciprocal as
the reference reciprocal divided by `1+r3`
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:363-375`).  The
explicit non-wetting branch then consumes coefficient, entry velocity, and
reciprocal in written multiplication order (`:703-706`).

The WS-RK3 branch computes the stage-specific W field from Kmm U/V for stages
after stage 1
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:329-335`).  The
compiled ZAD routine forms W transport, multiplies the Kmm vertical velocity
difference, divides by live face thickness times `1+r3*mask`, updates the
interior RHS, carries the product downward, and updates the bottom
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:105-137`).  Those are
the exact associations implemented by the private stage operands.

The first still non-bit compiled statement is not ZAD.  The active dispatcher
selects the iso-level Laplacian (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf.f90:81-90`),
whose rotational branch writes `zwf`/`zwt` and adds their differences to the
U/V RHS (`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`).
That cumulative boundary remains non-bit at 2.5292467120726215e-14 U and
3.502735092670824e-14 V.  Round 85 does not retract or hide it.

## Review and verification

The required separate command was run against the complete candidate diff and
the measured Rule-12 table.  It failed before a reviewer model started.  Its
terminal verdict is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**;
work continued.  No `SHIP` verdict is claimed.  Complete output is
`round85_codex_review.txt`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Focused CPU/fp64 verification passes 110 tests: the Round-85 bundle gate,
complete ZAD/dynamic-diffusion composition suite, AB3-AM4/history suite, and
ocean restart suite.  This includes the named loud failure for absent absolute
histories.  `git diff --check` and Python compilation pass.  The known
unrelated `test_rk3_ws_differs_from_rk3_and_is_finite` failure was not
encountered.

The receipt citation gate passes all 10/10 compiled-source citations with zero
unmapped entries and zero global map-audit failures.  Its shifted-citation
plant exits 1 and reports `SYMBOL-NOT-AT-LINE`.  The production additions
shifted older legoESM source lines; the affected cumulative receipt and 29 map
anchors were mechanically re-anchored, and the newly duplicated ZAD barrier
endpoint was occurrence-pinned rather than accepted ambiguously.

## Choices and uncertainty

Choices made: none.  Decision 37 and Decision 38 are applied exactly as given.
No configuration or new carried-state choice was introduced.

The bundle's local arithmetic is proven only for the captured NEMO inputs.
The first remaining non-bit statement is LDF; the first remaining
magnitude-bearing model-path boundary is the upstream composition entering
ZAD.  The candidate's small day-30 improvement does not identify either owner.

## OPEN — exact handoff to round 86

1. Start from landed commit `928fdcab9ee829d922d51ca8d7372f19b2d431af` plus
   this receipt commit.  Keep the Decision-36 GYRE ladder/year artifacts and
   the Round-79 tank artifacts as immutable comparison arms.
2. Rank by magnitude.  Reconcile the remaining approximately 1.9e-9 stage-1
   advection/ZAD-path gap using the already admitted Round-64 record: close the
   source-order reconstruction to the live total and walk W, Kmm velocity,
   live face thickness/r3, and accumulator association.  Do not change the
   now locally exact literal ZAD statements without a new first-non-bit walk.
3. Retain LDF as the first non-bit statement.  After the magnitude owner is
   resolved, score its Kbb operands, `zwf`/`zwt`, and the exact U/V additions;
   its 2.53e-14/3.50e-14 debt may not be rounded away.
4. Numerically score DINO before claiming neutrality for the shared standard
   drag materialization.  Preserve the regional/per-row cancellation audit.
5. ORCA2 remains UNMEASURED with the specification in the Rule-12 table.  No
   new NEMO acquisition is needed for the next GYRE owner walk.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
