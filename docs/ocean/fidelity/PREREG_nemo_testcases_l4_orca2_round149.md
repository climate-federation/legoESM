# Preregistration — ORCA2 round 149 carried V reciprocal substitution

Date: 2026-10-05. Frozen base: `0c0ceb3dae`.
Claim class: rung-0 results are **independent**. No configuration, initial
state, forcing, sea-ice, stabiliser, carried-state form, threshold, or
`unmeasured_features` choice is authorised.

## Source statement and scope

At the end of each external substep NEMO computes `hvr_e` from the updated
V-face depth at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-767`, includes
that array in the seven-field V-grid association at `dynspg_ts.f90:770-779`,
and carries the result into the next substep.  The next substep consumes that
carried reciprocal in the explicit V bottom-stress product at
`dynspg_ts.f90:680-699`.

Round 148 proved the seven post-association arrays bit-exact, but the substep-2
trace reconstructs `entry_inverse_v` from the compact SSH and differs from the
recorded carried `j001_hvr_e` on 68 full-domain northern faces, maximum
`0.03332976059679253`, first `(j,i)=(147,29)`.  The active-only score is exact.
This is the first non-bit recorded operand, but it is not upstream of the same
substep's midpoint transport or continuity: those statements execute earlier
at `dynspg_ts.f90:502-591`.

The one-variable arm substitutes NEMO's recorded substep-entry `hvr_e` only,
indexed by the scan's own substep counter.  It does not change V depth,
transport, continuity, SSH, configuration, or the model's ordinary carried
state.  It is a private/default-off record arm, not a production fix.

## Frozen protocol

1. Reuse the admitted round-95/97 rank-complete substep record and the
   round-146/148 passive production-JIT trace on CPU, fp64/libm, x64.
2. Extend the existing private test-hook and round-146 gate; do not add a
   second solver, callback, public selector, or card field.
3. Require observer passivity, the seven post-call rows, literal EEN
   coefficients, and round 148's frozen 30/68 control census to remain exact.
4. Build the override sequence from `i000_hvr_e` followed by the prior
   substep's recorded `jNNN_hvr_e`.  Refuse a wrong-frame plant and one ULP in
   an otherwise exact substituted cell.
5. Score substeps 1 and 2 in the frozen round-129 source order, including all
   full-domain operands.  Do not run either ORCA2 ladder unless the later
   midpoint-depth/transport/continuity chain also closes without an earlier
   loss.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R149-P1 | The observer, seven-field association, and literal EEN coefficients remain exact. | Every prerequisite row is bit-exact. | Any prerequisite bit moves. |
| R149-P2 | The control reproduces the first substep-2 entry-reciprocal debt. | `entry_inverse_v` is unequal on 68 full-domain cells with maximum `0.03332976059679253`; every earlier recorded substep-2 operand is exact. | Census/max differ or an earlier row is non-bit. |
| R149-P3 | The recorded carried `hvr_e` closes that boundary by itself. | Arm `entry_inverse_v` has zero unequal full-domain cells and the wrong-frame/ULP plants refuse. | Any bit remains or either plant stays green. |
| R149-P4 | This substitution does not own the earlier-executed midpoint/continuity chain. | Arm retains 30 unequal `mid_depth_v`, 68 unequal `transport_v`, 68 unequal `continuity_dv`, and 68 unequal `after_ssh` cells at substep 2. | The chain closes, changes census unexpectedly, or an earlier exact row moves. |
| R149-P5 | A measurement-only result leaves production unchanged. | Hook is private/default-off; no production card/deck/default/carry change remains; verdict HELD. | Any production behavior or configuration changes. |

## Controls and terminal rule

The gate must refuse the wrong-entry-frame plant, a non-vacuous substituted
`hvr_e` ULP, the existing observer/post-call/passivity/sign/registry plants,
and a reordered override registry.  A confirmed R149-P3 with R149-P4 leaves
the next statement at `mid_depth_v`; it does not authorize a landing or either
ladder.  Any unexpected closure is retained as a refutation and walked in
source order before further claims.

Pre-implementation search: the repository already has per-substep record
overrides for Coriolis and pressure-gradient operands in
`barotropic_latlon_cgrid.py`; round 149 extends that pattern and the existing
round-146 gate rather than adding a new harness.

ASKED choices: none. UNASKED choices: empty.
