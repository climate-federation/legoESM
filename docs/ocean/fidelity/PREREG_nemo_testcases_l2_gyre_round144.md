# Preregistration — NEMO testcase L2 GYRE round 144

Date: 2026-09-21

Incoming lane tip: `a41bc5ed36574da1532d9369bee81a80c901a2c9`

This document is frozen before running a new wind-operand arm. Evidence will
live under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round144/`.

Round 143 proved under production JIT that NEMO's exact post-drag pair followed
by legoESM's live wind update remains unequal on all 580 U and 570 V wet faces,
at `1.3293041661996061e-13` and `1.0882954083811340e-13 m s-2`. Replacing the
post-wind pair makes both incoming rows BIT. This round walks only the inputs
and association of that wind statement; it changes no production physics.

## P0 — compiled order and controls

The admitted target writes the direct wind operands at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-246`, then evaluates
the U and V additions at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:248-251`, with the written
order `post_drag + ((r1_rho0 * stress) * live_inverse_depth)`. The record then
writes the post-wind pair at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:252-260`.

Before any scientific row, the gate must re-admit the exact 1,486,548-byte
Round-140 record and reproduce every Round-143 ordinary, completed-RHS,
post-depth, post-drag, and post-wind control row exactly. The private hook must
default to `None`, be absent from `LatLonCGridOceanConfig`, preserve unowned
faces, and remain BIT between callback and independently compiled plain arms
across all returned-state leaves and actual external-call operands. A changed
record, control maximum, default path, or public selector REFUTES P0 and stops
the walk.

## P1 — one-family wind operand walk

Extend the existing Round-83/143 production-step gate. Add one private,
default-off wind-operand override at the already materialized wind boundary;
do not create another harness or callback. Every arm starts from NEMO's exact
post-drag U/V pair and leaves all later production arithmetic live:

1. live operands (the Round-143 post-drag control);
2. exact NEMO scalar density reciprocal only;
3. exact NEMO U/V face stresses only;
4. exact NEMO U/V live inverse depths only;
5. all three exact operand families with legoESM's current written association;
6. NEMO's exact post-wind pair as the terminal calibration control.

For each arm register incoming U/V and final U/V: scored cells, unequal cells,
maximum and RMS absolute difference, and first unequal index with signed-zero
sensitivity. Also compare each live operand against its recorded counterpart
before interpreting a substitution. No claim may be inferred from subtracting
two residuals.

Frozen prediction: density reciprocal is BIT and its arm is inert; stress is
the first non-bit operand family; replacing stress alone makes both incoming
rows BIT. Inverse depth and all-input arms are calibration rows expected BIT.
The prediction is REFUTED if density moves either row, stress is BIT, stress
does not make both incoming rows BIT, or an earlier family closes the boundary.
Every failed clause remains in the receipt.

If all three operand families are BIT yet the all-input arm is non-BIT, compare
the production expression against a direct transcription of the compiled
left-associated statement and name the written association as the first
non-bit statement only if that directed association closes both incoming rows.
Otherwise name no statement and retain the unresolved boundary.

A closed-row-registry omission plant and a one-ULP change to one consumed wet
stress word must each exit nonzero with `STATUS PLANT-FIRED`; the ULP plant must
move a registered incoming row through the complete production step.

## P2 — landing and campaign scope

The operand walk is diagnostic. A source-exact production candidate, if one is
named, is not landed without the full Decision-43 ladder/month checks,
Decision-45 360-day checks, every moved-row registry, and DINO measurement if
the statement is shared. Otherwise the expected status is `HELD`.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, or immutable before arm changes. LOCK_EXCHANGE and OVERFLOW execute no
changed production path. ORCA2 remains `UNMEASURED-WITH-SPEC` for this GYRE
developed-entry diagnostic. `ACQUISITION_NEEDED` and `DECISION_NEEDED` are
`NONE`.

## P3 — review and citations

A separate read-only Codex pass must try to refute the same-run operand
ancestry, native-face mapping, override placement, one-family isolation,
written association, callback/plain identity, registry completeness, and ULP
propagation. `DO NOT SHIP` blocks the diff. Every citation is mapped by the
receipt citation gate, whose shifted-citation plant must exit nonzero.
