# Preregistration — ORCA2 round 151 raw reference-depth causal arm

Date: 2026-10-05. Frozen base: `01feb01bd9`.
Claim class: rung-0 results are **independent**. No configuration, initial
state, forcing, sea-ice, stabiliser, carried-state form, threshold, or
`unmeasured_features` choice is authorised.

## Source statement and scope

NEMO forms the midpoint V-face depth from frozen `hv_0` and the already-exact
area-weighted midpoint sea surface at compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:542-545`, then uses it
in `zhV`, the south difference, and `ssha_e` at `dynspg_ts.f90:568-591`.
Round 150 split every operand: only the reconstructed frozen reference depth
is non-bit (30/26,640 V cells, maximum 899 m); an offline replacement with the
card's raw NEMO `hv_0` makes the depth bit-exact, while its descendants remain
unmeasured under a causal production-JIT arm.

This round adds one private/default-off hook to the existing barotropic
substep function.  The hook replaces only the frozen `H_u_ref/H_v_ref` slots
of `_nemo_ssh_avg_prep`; dynamic SSH products, reciprocal areas, masks,
configuration, histories, carry, and all source associations remain the
ordinary production values.  The arm receives the card's already-loaded raw
NEMO `hu_0/hv_0`.  `None` must leave the production jaxpr and trajectory
unchanged.

## Frozen protocol

1. Reuse the admitted round-95/97 rank-complete substep record and the
   round-146/150 CPU production-JIT fp64/libm trace.
2. Extend the existing private hook plumbing; do not add a public selector,
   card field, callback, second solver, or independently re-derived depth.
3. Require observer passivity, the seven post-association arrays, eight EEN
   coefficients, and the 30/68/68/68 control census to reproduce.
4. Run one arm with raw `hu_0/hv_0` replacing only the reference-depth slots.
   Score substep-2 `mid_depth_v`, `transport_v`, `continuity_dv`, and
   `after_ssh` over their complete recorded domains in source order.
5. Refuse a one-ULP perturbation in an arm depth and a malformed raw-depth
   tuple.  Unit tests must prove the default is exact and the override changes
   only the two registered prep slots.
6. If the four-row chain is not bit-exact, retain the failed prediction and
   name the first remaining non-bit boundary.  If it is exact, do not land a
   production fix until both ORCA2 ladders and the registered compensating
   salinity/tracer exposure have been measured.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R151-P1 | The passive prerequisites and control reproduce. | Observer, association, and EEN rows are exact; substep-2 control is 30/68/68/68 at the registered maxima. | Any prerequisite or control row moves. |
| R151-P2 | The unset hook is production-passive and the arm is one-variable. | Default prep is array-identical; the arm changes only `H_u_ref/H_v_ref`, with metrics and fold mask exact. | Default bits move or another prep slot changes. |
| R151-P3 | Raw NEMO reference depth closes the midpoint V depth in the running solver. | `mid_depth_v` becomes 0 unequal cells with all earlier rows exact. | Any midpoint V-depth cell remains unequal or an earlier row moves. |
| R151-P4 | The same one-variable arm closes the immediate V continuity chain. | `transport_v`, `continuity_dv`, and `after_ssh` each become 0 unequal cells. | Any target row remains non-bit. |
| R151-P5 | Measurement plumbing does not change an ordinary shared-card trajectory. | GYRE ten-step rows and day-30 snapshots are byte-identical to this round's base. | Any ordinary GYRE state bit moves. |

## Controls and terminal rule

The malformed-tuple and one-ULP arm plants must both fire.  A production fix
is not authorised merely because the causal arm closes: the approximately
31 PSU salinity exposure remains a hard veto until its registered gate and
both ORCA2 ladders pass.  This round may therefore end HELD with a confirmed
causal statement.

Pre-implementation search: the round-146 gate already runs private causal
arms and the barotropic substep function already accepts per-substep record
overrides.  The round extends those paths and reuses `_nemo_ssh_avg_prep`;
no existing raw-reference-depth hook was found.

ASKED choices: none. UNASKED choices: empty.
