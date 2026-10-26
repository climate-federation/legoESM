# ORCA2 round 48 preregistration — source-ordered QCO/RK tracer assignment

Date frozen: 2026-09-27  
Base: `7dd54b6f18d8f21307e5d2b2f5269b553dd6331f`  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
GYRE/tank claim label: **independent**

## One-statement scope

Round 47 made the stage-1 Kaa stretch exact.  This round changes only the
following QCO/RK tracer assignment, which NEMO evaluates as separately rounded
Kbb and Krhs products, their rounded sum, and the Kaa division at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.
The same statement executes at stages 1 and 2.  The committed round-45 scorer
already contains its source-ordered replay and non-vacuous one-ULP plant; no
second diagnostic spelling will be added.

No configuration, selector, default, carried state, score domain, threshold,
stabiliser, or sea-ice field may change.  The ORCA2 card's six-item
`unmeasured_features` tuple is frozen.

Repository search found the existing source-ordered implementation in
`nemo_testcase_l4_orca2_round45_qco_rk_gate.py::_jax_source_ordered` and the
production consumer in `_nemo_ws_rk3_tracer_pair_step::_stage`.  The measured
spelling will be moved into that existing production site rather than copied
to a new helper.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R48-P1 | The round-47 base reproduces before editing. | Kaa ratio is 0 / 8,613 unequal; production stage 1 is 57,141 T / 57,169 S unequal; recorded-operand source-order replay is 0 / 228,641 for both tracers. | Any value differs: stop and reconcile instrument drift. |
| R48-P2 | The one production statement closes ORCA2 stage 1 given NEMO's entry. | Production stage-1 T and S are each 0 / 228,641 unequal, while the fused-expression control retains the round-47 unequal counts. | Any production unequal cell or an inert control: do not land. |
| R48-P3 | ORCA2's ten-step policy passes. | 40 / 40 checkpoints complete; no AT-BAR row leaves; first-over-bar is not earlier; every moved row is registered. | Any refusal, bar loss, earlier first row, or unregistered move: hold. |
| R48-P4 | GYRE's full Decision 43/45/55/59 gate admits the shared statement. | Day-30 T RMS decreases from `6.572572612618985e-05 K`; day-240/360 do not worsen or stay within their authorized allowances; first-over-bar is not earlier; kt=1 AT-BAR rows stay; every moved row is registered. | Any unadmitted row: hold and name it. |
| R48-P5 | Tanks and the generic card retain their certified classifications; DINO does not execute the WS-RK3 statement. | LOCK_EXCHANGE/OVERFLOW have no earlier first-over-bar or kt=1 bar loss; generic-card certified snapshots are unchanged; both DINO cards resolve Euler. | Any violating row or execution-census mismatch: hold. |
| R48-P6 | The arithmetic preserves JIT and autodiff. | Direct eager/JIT source equality, a wrong fused-association control, finite nonzero gradients, and a revert-red production binding test pass. | Any mismatch, inert control, invalid gradient, or revert-green binding test: do not land. |

## Required measurements and landing rule

1. Run the round-45 internal scorer at base and candidate, including its plant.
2. Run the ORCA2 kt=1..10 ladder at base and candidate and compare every row.
3. Because GYRE executes this shared statement, run its complete Decision
   43/45/55/59 ten-step, day-30, day-240, and day-360 gate from this lane's
   round-47 baseline.  Run both tank and generic-card comparisons and the
   source-derived card census.
4. Land only if R48-P2 through R48-P6 pass.  Run the default and round receipt
   citation gates with a real plant, focused tests, the 170-test card battery,
   the push gate, and one `tests/ocean/fidelity -n 12` attempt.
5. Request the separate read-only Codex review; if it cannot initialize,
   record `independent review unavailable in-sandbox`.

## Frozen OPEN if the statement lands

Re-score the whole-card first non-bit statement and continue from its next
compiled producer.  Round 20's slow-forcing walk and the Decision-52
independent ORCA2 initial state/year remain open.  Sea ice remains out of scope.

## Choices

ASKED: source-exact shared statements are authorized under the full GYRE
Decision 43/45/55/59 gate by operator note B8.

UNASKED: none.
