# Preregistration — ORCA2 round 146 batched external-mode association

Date: 2026-10-04. Frozen base: `23d4980b8040a2de7395b55053c5974b4ebd5ef0`.
Claim class: rung-0 results are **independent**; rung-7 results are **given
NEMO's recorded entry**. No configuration, initial-state, forcing, sea-ice,
stabiliser, carried-state, threshold, or `unmeasured_features` choice is
authorised.

## Source statement and scope

The compiled rung-0 oracle updates `ua_e`, `va_e`, `hu_e`, `hv_e`, `hur_e`,
`hvr_e`, and `ssha_e`, then passes all seven arrays through one `lbc_lnk`
statement at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.
The T-pivot V association executed by that call is at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:684-721`.
The next substep consumes the associated velocity/depth operands before
forming unmasked metric transports at `dynspg_ts.f90:560-591`.

This round changes no public selector. A private trace may expose the exact
pre/post arrays. A private arm may apply the complete seven-array association
at the existing compact carry boundary. Partial-field or downstream
transport/divergence arms are forbidden.

## Frozen protocol

1. Reuse the admitted round-95/97 substep record and round-144 independent
   step-1..10 growth record. CPU, production JIT, fp64/libm, x64.
2. Extend the existing private barotropic-substep trace; do not add a second
   solver or callback. Publish unmasked pre/post U/V velocity, U/V live depth,
   both reciprocals, and SSH for the one batched association.
3. Compare the traced returned state with an ordinary step using
   `np.array_equal` for T, S, u, v, ssh, `uu_b`, and `vv_b` before accepting
   any observer number. Plant one ULP in a post-association operand and require
   refusal.
4. Apply the complete association as one private arm. Score the round-129
   substep-1/2 source-order registry, the independent rung-0 step-1..10 ladder,
   and the given-entry rung-7 step-1..10 ladder. Compare every moved row with
   the committed controlled comparator.
5. A production landing is eligible only if the complete call, without the
   round-129 transport/divergence rewrites, closes the bracketed boundary,
   loses no exact/AT-BAR row, does not move first-over-bar earlier, and does not
   restore the registered rung-0/rung-7 ~31 PSU salinity exposure. If eligible,
   run the standing GYRE, DINO, tank, generic-card, citation, and test gates.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R146-P1 | The extended observer is passive. | Every returned state array is bit-identical to the ordinary step. | Any returned-state bit moves. |
| R146-P2 | The seven post-association arrays differ from their pre-association forms only on the periodic/fold boundary images prescribed by the compiled call. | At least one boundary bit moves; no non-boundary bit moves. | No bit moves, or any interior bit moves. |
| R146-P3 | The reconstructed post-association arrays reproduce NEMO's recorded `ua_e`, `va_e`, `hu_e`, `hv_e`, `hur_e`, `hvr_e`, and `ssha_e` for substep 1 over the full recorded domain. | Seven rows have zero unequal cells. | Any row has one unequal cell. |
| R146-P4 | Applying only the complete batched association closes round 129's substep-2 first active residual. | `continuity_du` has zero unequal active cells and the following `after_ssh` row is bit-exact. | Either row remains non-bit, or an earlier exact row moves. |
| R146-P5 | The complete statement does not recreate the held multi-statement tracer compensation. | Neither ladder approaches the registered ~31 PSU kt=10 stage-3 S maximum; no exact/AT-BAR row is lost and first-over-bar is unchanged. | The ~31 PSU exposure returns, any exact/AT-BAR row leaves, or first-over-bar moves earlier. |
| R146-P6 | Shared-card gates accept the complete statement if it lands. | GYRE satisfies its standing year/ladder predicate; DINO, tanks, and generic cards do not worsen outside their registered allowances. | Any standing gate refuses. |
| R146-P7 | A failed complete-call prediction leaves production unchanged. | Arm reverted, no net production/card/deck diff, verdict HELD with the failed row. | Any partial association or downstream rewrite remains. |

## Controls and terminal rule

The gate must refuse a planted post-association ULP, a planted passivity bit,
and a reordered seven-field registry. The citation gate must pass against the
compiled record and a rigid line-shift plant must fire. A landing requires the
full standing gates; otherwise this is a measurement-only HELD round.

ASKED choices: none. UNASKED choices: empty.
