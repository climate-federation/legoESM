# ORCA2 round 64 receipt — vector-invariant ZAD walk

Date: 2026-09-28

Base: `1026f84027c730f4f61b70863e69c38c3ba9d9a3`

Disposition: **HELD; no model statement lands**

Claim label: **given NEMO's recorded operands**

Sea ice, its six selectors, and the ORCA2 card's `unmeasured_features` tuple
remain frozen.  This round changes no file under `packages/`, no card selector,
no initial state, and no NEMO arithmetic statement.

## Answer

ZAD is bit-exact.  The literal source-order fp64 replay differs in `0 / 413030`
active owned U faces and `0 / 415175` active owned V faces across the two MPI
ranks; both maximum absolute differences are `0.0`.  Rank 0 contributes
`226236` U and `226637` V faces; rank 1 contributes `186794` U and `188538` V.
A one-ULP U mutation adds one unequal active cell on each rank, proving the
comparison fires.

Together with round 63's exact KEG replay, this discharges both compiled
children of `dyn_adv` on the acquired boundary.  The rounds 62-63 framing that
the nonzero after-VOR→after-ADV movement itself was an ORCA2 mismatch is
**RETRACTED**: those numbers measure NEMO's own KEG+ZAD update between two
different program points.  They do not compare legoESM with NEMO and therefore
cannot name a legoESM defect.

No package change follows from this result.  The step-level walk returns to the
already-open QCO mixed boundary; after it closes, Decision 52's independent
initial state and the month-scale ORCA2 magnitude ranking retain their standing
order.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R64-P1 admission | **CONFIRMED** | Two rank-tagged 21-field records parse through physical EOF; effective Stokes drift is zero; ocean and ice restarts are byte-identical to the parent. |
| R64-P2 literal ZAD replay | **CONFIRMED bit-exact** | U `0 / 413030`, V `0 / 415175`, maximum `0.0`, dtype fp64. |
| R64-P3 controls | **CONFIRMED** | Header, field-order, truncation, restart, producer-stamp, and one-ULP ZAD plants all refuse. |
| R64-P4 first-non-bit disposition | **CONFIRMED retraction** | KEG and ZAD are each exact replays; boundary movement was a within-NEMO physical update, not a cross-model residual. |
| R64-P5 landing | **CONFIRMED** | No file under `packages/` changed; no physics statement lands. |

## Executed card and compiled source

The committed resolved-card census reports zero disagreements: ORCA2-zps and
GYRE-zco execute vector-invariant momentum advection; OVERFLOW-zps and LOCK-zco
execute flux-form UP3.  This is therefore an ORCA2/GYRE statement, scored here
on the ORCA2 record.

The acquired compiled dispatcher selects the C2 vector arm, calls KEG, then
calls ZAD, and records its endpoint at
`ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynadv.f90:136-143`.
The replay follows the executing `ln_vortex_force=.FALSE.` branch, including
surface-zero carries, area-times-`ww`, two-point transport sums, vertical
velocity differences, live QCO U/V thickness divisors, carried-interface
sums, and the bottom update at
`ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynzad.f90:102-137`.

The gate parses each record from its own magic, header integers, and per-array
name/rank/shape/payload fields through physical EOF.  It neither predicts a
whole-file byte count nor reconstructs carried state.  Every input and output
array in the scored replay is float64.

## Shared-card gate

The round changes measurement code, tests, citation anchors, and documentation
only.  The final `packages/` tree is byte-identical to the base, so ORCA2,
GYRE, DINO, OVERFLOW, LOCK_EXCHANGE, and generic-card trajectories cannot move.
The GYRE trajectory gate is therefore not applicable under the standing
shared-statement rule; no model statement changed.

## Review and validation

Validation and the required separate review are recorded in the final commit.

## OPEN

1. Return to the QCO mixed boundary measured in rounds 48-60, now without the
   retracted UP3/ORCA2 or vector-boundary framing, and name its first actual
   ORCA2 statement under the compiled vector-invariant program.
2. Then execute Decision 52's independent-start ORCA2 ladder and label every
   resulting number **independent**.
3. Once that ladder exists, rank month-scale ORCA2 errors by magnitude before
   walking more bit rows.
