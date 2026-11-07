# ORCA2 round 46 preregistration — stage-1 Kaa r3t interpolation

Date frozen: 2026-09-27  
Card: `orca2_vector_een_c2`  
Claim label: **given NEMO's entry** (Decision 52)

## Scope and compiled statement

This round changes one executed arithmetic statement and no configuration:
stage-1 Kaa `r3t` is formed by source-ordered interpolation of the already
rounded Kbb and after-level ratios.  The executing compiled ORCA2 branch is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.
The downstream tracer assignment at `stprk3_stg.f90:670-681` is explicitly
out of scope and remains the next statement.

No selector, default, carried-state rule, score domain, threshold, sea-ice
field, or stabiliser may change.  The six-item `unmeasured_features` tuple is
frozen.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R46-P1 | Round 45 reproduces before the edit. | Current Kaa stretch is 2 / 8,613 unequal, source-ordered replay is 0 / 8,613, and production stage 1 is 57,160 T / 57,180 S unequal. | Any count differs: STOP and reconcile the instrument before editing. |
| R46-P2 | The one-statement model edit closes the Kaa operand exactly. | Production Kaa stretch is 0 / 8,613 unequal against the admitted record. | Any unequal cell: revert the model edit and HELD. |
| R46-P3 | The downstream fused tracer assignment remains separate debt. | Production stage 1 becomes 57,141 T / 57,169 S unequal, matching round 45's recorded-operand fused replay; it does not become exact. | Exact stage 1 or different counts: do not attribute; walk the unexpected boundary. |
| R46-P4 | The ORCA2 kt=1..10 landing policy passes. | No AT-BAR row leaves the bar, first-over-bar is not earlier, every moved row is registered, and all 40 checkpoints complete. | Any bar loss, earlier first divergence, unregistered move, or refusal: do not land. |
| R46-P5 | GYRE is byte-identical because its linear-free-surface T ratio is exactly one at every stage. | Base versus tip: 0 differing certified rows, all 210 residual arrays `np.array_equal`, and all 30 daily snapshots byte-identical. | Any difference: this is a GYRE landing too; stop with `DECISION_NEEDED`. |
| R46-P6 | The edit preserves JIT and autodiff. | Focused eager/JIT equality and finite gradient checks pass on a synthetic non-degenerate endpoint pair. | Any mismatch, non-finite gradient, or zero sensitivity: do not land. |

## Required controls and gates

1. Reuse the committed round-45 record scorer before and after the edit.
2. Add a direct synthetic arithmetic test whose interpolated-SSH spelling is
   not bit-identical to the endpoint-ratio spelling; demonstrate that reverting
   the production call site turns the test red.
3. Run the complete ORCA2 kt=1..10 ladder at base and tip and compare under the
   established landing policy.
4. Run the GYRE ten-step trajectory at base and tip, compare with
   `nemo_testcase_offline_compare.py`, and require array-equal residuals; run
   the 30-day member at base and tip and require byte-identical snapshots.
5. Run citation gates, the focused tests, the 170-test card battery, the push
   gate, and one `tests/ocean/fidelity -n 12` battery, one pytest process at a
   time.
6. Request a separate read-only Codex review; record
   `independent review unavailable in-sandbox` if it cannot initialize.

## Landing rule

Land only if R46-P2, R46-P4, R46-P5, and R46-P6 confirm.  R46-P3 is expected
open debt and must be named, not folded into this round.
