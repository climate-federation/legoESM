# NEMO testcase L2 GYRE phase 3 — round 86 ZAD-operand receipt

## Outcome

**HELD.** The first magnitude-bearing non-bit input to the stage-1 ZAD
statement is the W field. Kmm U/V, live face thickness, T-cell area, and U/V
reciprocal face area are bit-exact. W differs at all 18,000 scored wet
point-levels, with maximum `7.946658315637966e-07 m/s`. Substituting only that
live W into the exact literal ZAD replay reproduces 100 percent of the final
ZAD residual on both faces: `1.9220297482797664e-09` U and
`1.966061294804274e-09` V. Velocity-, thickness-, and metric-only
substitutions remain bit-exact.

The frozen combined prediction is nevertheless **REFUTED and retained**.
Its W ownership and zero thickness contribution predictions pass, but its
required source-order/live-total closure does not: the independent association
row remains non-bit at `8.470329472543003e-22` on both faces. This round
therefore does not attribute the upstream W mismatch to one WZV input, does not
change production physics, and does not rerun trajectory gates.

## Frozen registration and evidence

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round86.md`, committed as
`7066f46209356a100612fafa789b51f157b1f6d6` before a new scientific
comparison. The fail-closed instrument and controls were committed before the
sealed arm, finally at `2905480052316a45158aa6df53808583dc3e5142`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round86/`; all model
executions were CPU/JAX fp64/libm with production JIT.

The ordinary artifact is `zad_operands_v3.json`, SHA-256
`0b784090063d13ca4559f01834192a1d99d19c5db093cf60ed8bcaa93b7c8502`.
It exits 1 with status `REFUTED`, as required for the failed combined
prediction. Round-64 admission retains 43 byte-identical records, 20
classified changed records, 132 admitted differences, producer
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`, and zero owned-and-defined
differences in the consumed kt2 stage-1 projection.

The installed GitHub CLI still reports an invalid token, and no authenticated
GitHub connector is available. The Round-85 and Round-86 findings could not be
posted to issue #1455; no external state was mutated.

## Ordered operand and replay result

The complete registered operand rows are:

| operand | unequal wet cells | absolute maximum | disposition |
|---|---:|---:|---|
| Kmm U | 0 / 17,400 | 0 | bit-exact |
| Kmm V | 0 / 17,100 | 0 | bit-exact |
| W | 18,000 / 18,000 | `7.946658315637966e-07` | first non-bit; magnitude owner at this boundary |
| r3u | 580 / 580 | `1.1010194822576465e-16` | non-bit input representation |
| r3v | 570 / 570 | `1.107889978251063e-16` | non-bit input representation |
| live U face thickness | 0 / 17,400 | 0 | bit-exact materialized divisor |
| live V face thickness | 0 / 17,100 | 0 | bit-exact materialized divisor |
| T-cell area | 0 / 600 | 0 | bit-exact |
| reciprocal U face area | 0 / 580 | 0 | bit-exact |
| reciprocal V face area | 0 / 570 | 0 | bit-exact |

The r3 rows are retained as real bit debt, but the compiled ZAD statement
consumes their already-materialized face thickness. Those thickness arrays are
exact, so the r3 representation does not enter the measured ZAD residual.

The inherited numpy transcription of the complete compiled ZAD loop remains
bit-exact against NEMO on both U and V, with zero unequal cells. Controlled
replays that replace only current production operands give:

| replacement | U absolute maximum | V absolute maximum | fraction of all-live maximum |
|---|---:|---:|---:|
| Kmm velocity | 0 | 0 | 0 |
| W | `1.9220297482797664e-09` | `1.966061294804274e-09` | 1.0 / 1.0 |
| face thickness | 0 | 0 | 0 |
| area/reciprocal area | 0 | 0 | 0 |
| all registered live operands | `1.9220297482797664e-09` | `1.966061294804274e-09` | 1.0 / 1.0 |

This is a controlled substitution on NEMO's post-KEG accumulator. It proves W
owns the measured ZAD-boundary magnitude; it does not prove which input or
association inside WZV owns W.

## Controls and instrument failures

All three maximum-residual one-ULP controls print `PLANT_FIRED` and exit 1:

- W: `zad_operands_ww_plant.json`, SHA-256
  `d3b1e3bde2c94031ef5cdd58f58a9d504a12c01422350333e287263117ebb511`;
- exact face thickness: `zad_operands_thickness_plant.json`, SHA-256
  `43b7cfa8a075f7b9cb86780d426748f9ea1c4424b3ac1cb4d1a65f31b752790e`;
- non-exact association closure: `zad_operands_association_plant.json`,
  SHA-256
  `c9c74e3c4f0eb3f6893b7b5e98f1c2163b27590b31ce18bbfc4f61c4a23dad8e`.

Two pre-result instrument failures are retained. The first attempt stopped on
a generic comparison-shape error before writing JSON. The committed diagnostic
then named the second failure as the singleton stage plane on live r3u
(`(22,32,1)` versus the record's `(22,32)`). Selecting that one stage plane is
the rank conversion already used by the stage trace; after the repair all
three shapes are checked before every row. Neither failed attempt is evidence.

## Compiled-source basis

The executing stage-1 branch calls WZV on Kbb velocity and the Kaa
free-surface slot, then calls KEG and ZAD before its write-only post-advection
snapshot
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`).
The active QCO WZV branch integrates the horizontal-divergence and
`r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb))` terms from the bottom upward
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`). The
compiled ZAD routine consumes W, Kmm U/V, T-cell area, face-area reciprocals,
live face thickness, and masks in its carried interior and bottom updates
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138`).

Those citations identify the compiled statements that run under the admitted
GYRE header. The measured first non-bit statement at this granularity is the
completed WZV output handed to ZAD. Because this record does not independently
score every WZV input/association against the current live construction, no
narrower WZV statement is named and no production edit is eligible.

## Rule-12 disposition

| lane | Round-86 result |
|---|---|
| GYRE kt2 ZAD chain | **MEASURED / NO CANDIDATE.** Given-input ZAD is bit-exact; W alone owns the `~1.9e-9` boundary; live-total association remains non-bit. |
| GYRE kt=1..10 | **PRESERVED, NOT RERUN.** No production change. Current before remains `round85/bundle_kt1_10.json`; first-over-bar remains kt2 U/V. |
| Every moved GYRE row | **NONE.** This round changes only a diagnostic, tests, citation metadata, preregistration, and receipt. |
| GYRE days 1..30 | **PRESERVED, NOT RERUN.** Current before remains `round85/bundle_day_gap.json`; day-30 T RMS remains `1.2397011295506804e-02 K`. |
| LOCK_EXCHANGE-zco | **PRESERVED, NOT RERUN.** No shared statement changed; the Round-85 50-row post-history arm remains the control. |
| OVERFLOW-zps | **PRESERVED, NOT RERUN.** No shared statement changed; the Round-85 50-row post-history arm remains the control. |
| DINO | **SHARED-STATEMENT RISK.** Its leapfrog route calls the same W/ZAD implementation; 96--98% regional cancellation forbids a neutrality inference. |
| ORCA2 | **UNMEASURED WITH SPEC.** Align Kbb/Kaa SSH, Kmm U/V, W divergence/stretching inputs, live thickness/r3, every ZAD carry/update, cumulative RHS, and T/S/U/V/SSH on native masks through kt1..10 in fp64. Reject any wet-input mismatch, AT-BAR loss, or earlier first-over-bar. |

No configuration/default, selector, coefficient, timestep, stabilizer,
carried-state representation, restart format, year harness, reconciliation
gate, freshwater pair, #1484 guard, held manifest, NEMO source, or NEMO
executable changed.

## Review and focused verification

The required separate read-only Codex command was run against the complete
Round-86 diff and Rule-12 table. It failed before a reviewer model started, so
there is no `SHIP` or `DO NOT SHIP` verdict. Its terminal verdict is quoted
verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**;
work continued. The complete output is `round86_codex_review.txt`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No reviewer approval is claimed and no physics is shipped.

Focused CPU/fp64 verification passes 69 tests covering the Round-41 literal
KEG/ZAD replay, Round-46 stage gate, Round-84 association walk, Round-86
helpers and controls, citation gate, ZAD bottom masks, QCO pairing, and
ZDF/dynZDF composition. The decisive line is `69 passed in 37.35s`; log
SHA-256 is
`f9b0b652db4e94581cce5ecf8cc5b260e24393cd7be866e6524b4998ddaac26c`.
Python compilation and `git diff --check` pass. The known unrelated
`test_rk3_ws_differs_from_rk3_and_is_finite` failure was not encountered.

The receipt citation gate passes all 3/3 compiled-source citations with zero
unmapped or map-audit failures; artifact SHA-256 is
`46f23527986b9623077b8e666e02f7fa163ee0061deaa9b2a9771ce7146f7caa`.
Shifting the WZV citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`; plant SHA-256 is
`9b80fd8e4f1564b7678a22ab2700907258738c3c491200f1e140f874f82a4277`.
The first citation attempt refused an ambiguous repeated QCO-arm anchor; no
citation was accepted from that attempt. Both endpoints were occurrence-pinned
before the clean pass.

## Choices and uncertainty

Choices made: none. Decisions 37--38 remain unchanged. No configuration or
carried-state choice was introduced.

The first magnitude-bearing boundary is now W, not ZAD arithmetic. The exact
upstream WZV owner and the independent `8.47e-22` association boundary remain
unresolved and are not conflated.

## OPEN — exact handoff to round 87

1. Reuse the admitted Round-64 stage record. Preregister a WZV input walk in
   its compiled order: Kbb U/V, live Kmm U/V thickness and metric products,
   face-flux differences, live T thickness divide/multiply, Kbb and Kaa r3,
   `r1_Dt`, per-level stretch, mask, and bottom-up carry. Score the stored Kaa
   free-surface forecast separately from any recomputed continuity forecast.
2. Require the existing literal WZV replay to remain bit-exact, add a planted
   maximum-residual Kaa/stretch value, and name the first non-bit WZV input or
   arithmetic statement. Only an all-input-exact/result-non-bit statement is
   eligible for a shared implementation change.
3. Reconcile the independent source-order/live-total association at
   `8.470329472543003e-22`; do not hide it behind W's larger gap. Retain LDF as
   the first cumulative non-bit statement at `2.53e-14` U / `3.50e-14` V.
4. Numerically score DINO before claiming neutrality for any shared W/ZAD
   change. ORCA2 remains UNMEASURED under the Rule-12 specification above.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
