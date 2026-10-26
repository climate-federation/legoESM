# ORCA2 round 58 preregistration — source-ordered UP3 T-face flux landing

Date frozen: 2026-09-27  
Base: `8c4f6a700`  
Cards: `orca2_vector_een_c2` plus every shared executing card  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
OVERFLOW/GYRE/tank claim label: **independent**, except the direct statement
replay, which is **given NEMO's recorded operands**

## Authorization and one-statement scope

Round 57 established that the first non-bit statement in the admitted OVERFLOW
kt=3 stage-2 UP3 walk is the U T-face flux: 282 / 17,000 unequal, split into
four wet/dry-boundary mask omissions and 278 wet-interior association
differences.  This round implements that one compiled statement and judges it
under the standing full shared landing gate.

The executing compiled branch forms the masked U curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, selects
the left or right curvature from the sign of the Kmm velocity pair at
`:182-192`, and multiplies the transport sum by that source-ordered face value
at `:194-195`.  The production NEMO-UP3 arm instead expands the algebraically
equivalent reconstruction before multiplying its averaged transport.

The candidate may change only the existing NEMO-UP3 same-direction T-point
flux spelling.  It adds no scheme, selector, configuration field, default,
carried state, score domain, threshold, stabiliser, or sea-ice field.  The
Oceananigans UP3 arm and first-order/centred arms retain their existing
arithmetic.  ORCA2's six sea-ice selectors and `unmeasured_features` tuple stay
unchanged at `STOP_SELECTOR_GAP`.  The held QCO/RK candidate remains absent.

Repository and test search found the existing production `_up3_reconstruct`,
its four production call sites, the round-57 header-derived statement gate,
and the existing UP3 unit/selector/differentiability tests.  The candidate will
extend those paths; it will not add a second advection operator or diagnostic
copy.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R58-P1 | The clean base reproduces round 57 before editing. | Direct statement gate reports 282 / 17,000 unequal, partitioned as four boundary and 278 wet-interior cells, with maximum `0.005124451203774175`; its one-ULP plant adds exactly one refusal. | Any count, partition, maximum, provenance, or plant differs: stop and reconcile before implementation. |
| R58-P2 | NEMO source order closes the direct statement given NEMO's recorded operands. | Earlier curvature/pair/selector rows remain 0 unequal and the production U T-face flux becomes 0 / 17,000 unequal, including 0 / 4 boundary and 0 / 278 prior-interior cells. | Any earlier row moves or any flux cell remains unequal: do not land. |
| R58-P3 | The implementation is one binding, differentiable production statement. | Direct eager/JIT values agree; gradients are finite and non-zero; independent association and omitted-mask controls each fail; reverting the production call makes the binding test fail. | Any eager/JIT mismatch, zero/non-finite gradient, dead control, or revert-green binding test: do not land. |
| R58-P4 | ORCA2 retains its admitted ten-step policy given NEMO's entry. | All 40 checkpoints complete; no AT-BAR row leaves; first non-bit remains kt=1 stage-1 T; every moved row is registered. | Any refusal, AT-BAR loss, earlier first row, or unregistered move: hold. |
| R58-P5 | The full GYRE Decision 43/45/55/59 gate passes on this shared statement. | Ten-step first-over-bar is not earlier, kt=1 AT-BAR rows stay, every moved row is registered against the `2e-10 K` floor, day-30 T RMS decreases, and day-240/360 do not worsen beyond the standing Decision-59/AT/AW allowances. | Any binding row fails: hold and name it. |
| R58-P6 | The source-ordered statement does not regress the other executing cards. | OVERFLOW has no earlier first-over-bar, no kt=1 AT-BAR loss, and no row worsens beyond its strict 2-ULP bar; LOCK_EXCHANGE and generic-card gates pass; DINO/tank/card census and focused gates pass. | Any card refusal or unregistered movement: hold and name the first failing row. |
| R58-P7 | The one statement lands. | R58-P2 through P6 pass, default and round citation gates have zero unmapped citations with real plants firing, the required test batteries have no new red, and independent review has no unresolved blocker. | Any prerequisite fails: revert the model candidate and finish `HELD` with the failing row registered. |

Failed predictions remain in the receipt as **REFUTED**.  No post-hoc pair is
eligible to land in this round; if the held QCO/RK statement is retested, that
requires a later preregistration after this statement's disposition is known.

## Required measurements and landing rule

1. Re-run the round-57 direct statement gate and plant at the frozen base.
2. Implement only the compiled masked-curvature/selector/flux order and add
   binding source-order, mask, eager/JIT, gradient, and revert-failure tests.
3. Re-run the direct statement gate, ORCA2 kt=1..10 ladder, GYRE ten-step and
   year gates, OVERFLOW, LOCK_EXCHANGE, generic, DINO/tank/card gates, and the
   standing Decision 43/45/55/59 comparison.
4. Run a separate read-only `codex exec` review of the diff.  If it cannot
   initialize, record `independent review unavailable in-sandbox` exactly.
5. Run the default and round receipt citation gates with a real shifted-line
   plant, focused tests, one push-gate-equivalent battery, the 170-test card
   battery, and one `tests/ocean/fidelity -n 12` battery after confirming no
   other pytest battery is active.

## Frozen OPEN

If the statement lands, retry the separately held source-ordered QCO/RK
statement only in a new preregistered round.  After the step-level walk closes,
perform Decision 52's owed **independent-start** ORCA2 ladder and then rank
month-scale magnitudes.  Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: land the measured NEMO source-ordered UP3 statement under the standing
shared gates.  
UNASKED: none.
