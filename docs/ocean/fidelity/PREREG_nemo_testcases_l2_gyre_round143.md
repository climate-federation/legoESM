# Preregistration — NEMO testcase L2 GYRE round 143

Date: 2026-09-21

Incoming lane tip: `478f8abf2b70a106f9910190ca66199767bc3a63`

This document is frozen before scoring a new downstream directed arm. Evidence
will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round143/`.

Round 142 replaced legoESM's completed three-dimensional momentum RHS by
NEMO's same-step pair without observing the model RHS. That arm reduced the
incoming U maximum from `4.2854247978022983e-13` to
`1.3307884389737387e-13 m s-2` and the incoming V maximum from
`4.433308633699682e-13` to `1.089946783815101e-13 m s-2`, but all wet faces
remained non-bit. The completed RHS is therefore only a partial contributor.
This round follows the Round-142 OPEN section and changes no production
physics.

## P0 — compiled order and admitted boundaries

The exact compiled Round-140 program accumulates the three-dimensional RHS at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169`, writes that
completed pair at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:182-192`, computes the
depth-average at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`, applies the
baroclinic drag at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-239`, and applies
wind at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-260`.

Before any new arm, re-admit the exact 1,486,548-byte Round-140 record and all
stored controls. Reproduce Round 142's ordinary and completed-RHS-directed
incoming/final rows exactly. The default-off hooks must be absent from the
public configuration. Their no-override path must be BIT against an
independently compiled ordinary production step across all 23 returned-state
leaves and at the actual external-call U/V operands.

Any changed digest, replay, inherited byte, ordinary maximum, Round-142
directed maximum, returned-state cell, external operand, or public selector
REFUTES P0 and stops the scientific comparison.

## P1 — no-observer downstream substitutions

Extend the existing Round-83 walk and Round-142 developed-state bridge; do not
create another stepper or callback. Add two private, default-`None`,
non-configurable hooks:

1. replace only the owned native U/V post-depth-average faces, then leave the
   model's drag and wind arithmetic live; and
2. replace only the owned native U/V post-drag faces, then leave the model's
   wind arithmetic live.

The second arm replaces the already accumulated depth/drag/wind value at the
post-drag boundary and then adds the model's already-computed wind increment
once. It does not move or reorder the default production path. Both arms
preserve excluded boundary faces. The admitted post-wind/incoming pair is the
terminal control through the existing incoming override.

Register ordinary, completed-RHS, post-depth, post-drag, and post-wind arms,
each with incoming U/V and final U/V rows. Every row reports scored and unequal
cells, maximum and RMS absolute difference, and first unequal index with
signed-zero sensitivity. Each directed arm must be BIT between its callback
and independently compiled plain execution; its callback final fields must be
BIT against the actual external-call fields.

Frozen prediction:

1. The post-depth arm is the first downstream boundary that removes the
   `1e-13 m s-2` magnitude: both incoming maxima decrease by at least `1e3`
   from the completed-RHS arm and are at most `1e-18 m s-2`.
2. The post-depth arm may retain last-bit drag/wind association differences;
   the post-drag arm followed by the live wind term makes both incoming rows
   BIT.
3. The post-wind terminal arm makes both incoming rows BIT and is a calibration
   control, not an attribution.

Prediction 1 is REFUTED if either post-depth incoming maximum is above
`1e-18`, improves by less than `1e3`, or worsens. Prediction 2 is REFUTED if
either post-drag incoming row is non-bit. Prediction 3 is REFUTED if either
terminal row is non-bit. If post-depth is already BIT, the last-bit clause is
simply unnecessary; the depth-average family remains the first closing
boundary. If post-depth misses the magnitude criterion but post-drag passes,
the drag family is first. If only post-wind passes, wind is first. No result is
promoted from subtraction of residuals.

A registry-omission plant and a one-ULP change to one consumed wet post-depth U
word must each exit nonzero with `STATUS PLANT-FIRED`; the ULP plant must move a
registered downstream incoming row through the full production step.

## P2 — interpretation and stop condition

A closing boundary names only an owner family. It does not name an internal
statement or a day-240 owner. If post-depth is first, the next round walks the
compiled depth-average statement from its recorded e3, RHS, mask, and
reciprocal operands under production JIT. If post-drag is first, the next round
walks the compiled drag inputs and association. If post-wind is first, it walks
the compiled wind operands and association.

No cumulative HPG/LDF/VOR/KEG/ZAD acquisition is authorized unless the
post-RHS chain closes. Even after closure, no operator earns a day-240 carry
without a source-exact statement and the Decision-45 year arm. No landing is
expected this round.

## P3 — review, citations, and scope

A separate read-only Codex pass must try to refute native-face mapping,
excluded-face preservation, default-path identity, arm-local callback
passivity, exact placement of each override, one-time wind application,
registry completeness, production propagation of the plant, compiled-source
order, and the boundary-versus-statement distinction. `DO NOT SHIP` blocks the
diff. Every compiled-source citation is mapped by the receipt citation gate;
its shifted-citation plant must exit nonzero.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, or immutable before arm changes. The ladder, month, year, DINO,
LOCK_EXCHANGE, OVERFLOW, tanks, and ORCA2 are not rerun because this is private
diagnostic instrumentation. ORCA2 remains `UNMEASURED-WITH-SPEC` for the GYRE
developed-state registry. `DECISION_NEEDED` is `NONE`; expected status is
`HELD`.
