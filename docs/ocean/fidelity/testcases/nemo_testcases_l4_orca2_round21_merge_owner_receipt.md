# NEMO testcase Lane 4 — ORCA2 card round 21 merge-owner receipt

Date: 2026-09-25

Parent: `d79d85e53b9d2a7fcc7b8a73545a5d5fd39e3e74`

Status: **HELD.**  The two merge contributors are both real, but they are not
complete.  Given the independent ORCA2 initial state with Decision-52's
recorded NEMO SSH entry, substituting the pre-merge fold layout and the
bridge-carried NEMO live vorticity-thickness operands together restores six
of the 185 changed scored rows.  It leaves 179 rows different from round 20;
the first changed score is kt=2 entry T.  The ORCA2 ladder is not re-certified,
and Decisions 58 and 54 do not start.

No file under `packages/` changed.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain unchanged: `staged_gm_eiv`,
`linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every ladder number below is labelled **independent with Decision-52 SSH**.
No given-NEMO-entry solver number is mixed into the table.

## 1. Compiled statement and controls

The record's compiled build creates `e3f_0vor` from the masked four-cell
reference-thickness average, applies its F-point halo exchange, and only then
fills remaining zeros from `e3f_3d` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937`.
It creates the RK3 live F-point stretch from four area-weighted SSH values,
`r1_hf_0`, and `r1_e1e2f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The executing EEN vorticity statement divides by the product of those two
operands and `fe3mask` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:556`.

The baseline and four controls use the same clean committed tree, card,
inputs, record, ten steps, fp64 policy, and CPU backend.

- Fold-only restores the pre-merge fold descriptor through the already
  committed merge control.
- Thickness-only restores the implementation immediately before
  `ddb70da1a4`: all reference thickness, mask, area, F-depth, and `fe3mask`
  operands come from the bridge-carried NEMO record.  The current call's
  card-state thickness and mask arguments are accepted and deliberately
  ignored.  No model configuration or state is changed.
- Combined installs both controls.
- The plant installs the combined arm and advances every nonzero substituted
  live thickness by one representable fp64 value.

The result gate verifies that every top-level context other than the candidate
trajectory and worktree stamp is identical across the five current-tree arms.
The plant changes 185 scored rows, so the substitution channel is not deaf.

## 2. Ten-step result

| arm | rows different from round 20 | kt=1 stage-2 u max | kt=1 stage-2 v max | kt=10 entry T max |
|---|---:|---:|---:|---:|
| round 20 | 0 | `0.06463348861831608` | `0.03401478477520298` | `3.9430791763114783` |
| merged baseline | 185 | `0.06463349988292608` | `0.03401471577804818` | `3.947126188631776` |
| fold-only | 185 | `0.06463349183839187` | `0.03401471577804818` | `3.947126188631776` |
| thickness-only | 185 | `0.06463349666286919` | `0.03401478477520298` | `3.9471160991501986` |
| combined | 179 | `0.06463348861831608` | `0.03401478477520298` | `3.9471160991501986` |

The merged baseline's full `candidate_trajectory` document is exactly equal
to the merge receipt's admitted document, so R21-P1 is an instrument
reproduction rather than a new baseline.  Against the 185 registered rows,
fold-only moves 27 scores toward round 20, 24 farther, and leaves 134 at the
same max-error distance.  Thickness-only moves 118 toward and 67 farther.
Combined moves 110 toward, 69 farther, and restores six complete row summaries.

R21-P4 is **REFUTED**.  The combined arm's first residual score is kt=2 entry
T.  Its maximum, unequal-cell count, and first unequal index equal round 20,
but its mean absolute error over unequal cells is
`0.0004382862650997294` rather than `0.00043828626509972854`.  A row summary
cannot identify which candidate bits moved; it only proves the two documents
are not equal.  The largest remaining max-error movement is kt=10 stage-3 v:
`0.3556678149969952` in round 20 versus `41.44545607370846` in the combined
arm.  Therefore the residual is not merely the precision of the first scalar
summary, and the next round must instrument the kt=1 stage-3 to kt=2 entry
transition directly before naming a third statement.

## 3. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R21-P1 | **CONFIRMED** | The baseline exactly reproduces the merge document and all three frozen target values. |
| R21-P2 | **CONFIRMED** | Fold-only reproduces `0.06463349183839187` at kt=1 stage-2 u and does not restore round 20. |
| R21-P3 | **CONFIRMED** | Thickness-only changes 185 rows, first at kt=1 stage 2, and moves 118 registered rows toward round 20. |
| R21-P4 | **REFUTED** | Combined leaves 179 rows different; the two named contributors are incomplete. |
| R21-P5 | **CONFIRMED** | Contexts are identical and the one-ULP field plant changes 185 rows. |
| R21-P6 | **CONFIRMED** | No production file changed and no statement is eligible to land. |

No prediction was rewritten after measurement.  R21-P4 remains in the
preregistration and in the result gate as a failing prediction.

## 4. Review and gates

The required separate `codex exec --sandbox read-only` review was attempted
at the committed measurement tip.  It failed before reading the diff because
the in-process app-server client could not initialize on a read-only
filesystem.  Verdict: **independent review unavailable in-sandbox**.

The result gate exits 2 with `HELD_THIRD_OWNER`; that nonzero exit is its
registered scientific verdict, not a crash.  Its direct tests pass 2 / 2.
The receipt citation gate and test batteries are recorded below after their
final runs.

## Choices

ASKED: substitute the two registered merge owners one variable at a time and
re-certify the ORCA2 ladder before Decisions 58 and 54.

UNASKED: none.  No configuration value, state field, stabilizer, sea-ice
selector, scoring definition, or production statement changed.

## OPEN

1. Round 22 instruments the exact candidate fields across the kt=1 stage-3 to
   kt=2 entry transition, then walks the first changed operation.  The row
   summaries are insufficient to name that statement.
2. Decision 58 and then Decision 54 remain ordered after ladder
   re-certification.  Decision 57 remains reported and untouched.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on the scheduled independent
   initial-state completion.
6. The seven inherited duplicate citation-map literal keys remain open.
