# NEMO testcase Lane 4 — ORCA2 card round 21 preregistration

Date: 2026-09-25

Parent: `d79d85e53b9d2a7fcc7b8a73545a5d5fd39e3e74`

Status: **PREREGISTERED BEFORE ROUND-21 SCIENTIFIC SCORING.**

Round 21 reconciles the ORCA2 ten-step ladder after the GYRE-lane merge.  It
changes no configuration and lands no model statement merely because a
diagnostic arm restores an older trajectory.  Every number is labelled
**independent with Decision-52 SSH**.  The six sea-ice selectors and the
card's `unmeasured_features` tuple remain frozen.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round21/`.

## Executed NEMO statements

The admitted ORCA2 build creates the frozen EEN reference thickness by the
masked four-cell average at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-919`,
applies the F-point halo exchange and only then fills remaining zeros from
`e3f_3d` at `:935-937`.  It creates the live RK3 F-point stretch from the
four area-weighted sea-surface values, `r1_hf_0`, and `r1_e1e2f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The executed vorticity statement divides by the product of those two operands
and `fe3mask` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:555-557`.

The merge receipt registered two contributors to the ladder movement:

1. GitHub main's tripolar storage-layout classification.  The committed
   one-variable control restores only the pre-merge fold descriptor.
2. The GYRE lane's rebuild of the live vorticity thickness from card state.
   The new control restores only the pre-rebuild, bridge-carried NEMO
   operands used by the same cited formulas.  It does not change the card,
   state, solver, fold descriptor, or any other operator.

The four arms are baseline, fold-layout-only, vorticity-thickness-only, and
both substitutions together.  All use the same committed tree, inputs,
record, ten steps, fp64 policy, and CPU platform.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R21-P1 | The merged baseline reproduces the merge receipt exactly. | `LADDER_MEASURED`; kt=1 entry and stage 1 unchanged; kt=1 stage-2 `u=0.064633499882926077`, `v=0.034014715778048182`; kt=10 entry `T=3.947126188631776`. | Any differing value or status: stop for instrument drift. |
| R21-P2 | The existing fold-layout-only control reproduces its registered movement and does not restore round 20. | kt=1 stage-2 `u=0.064633491838392`, with at least one remaining row unequal to round 20. | No movement, a different registered value, or complete restoration. |
| R21-P3 | Restoring only the bridge-carried NEMO vorticity-thickness operands moves at least one stage-2 momentum row toward round 20 and leaves kt=1 entry and stage 1 bit-identical. | At least one of the 185 moved rows changes toward round 20, with no earlier movement. | Zero moved rows, movement first at entry/stage 1, or every changed row farther from round 20. |
| R21-P4 | The two registered contributors are complete: applying both substitutions restores round 20's 200 ladder rows bit-for-bit. | All checkpoint field rows equal round 20, including kt=1 stage-2 `u=0.064633488618316082`, `v=0.034014784775202977`, and kt=10 entry `T=3.9430791763114783`. | Any residual row.  Rank the first and largest residual and HOLD certification. |
| R21-P5 | Both controls are one-variable and non-vacuous. | Constructor/config leaves are identical; each arm changes only its named runtime operand; reverting either substitution changes at least one ladder row; a one-ULP plant in the substituted vorticity operand changes the ladder. | Any config difference, extra changed operand, or deaf plant. |
| R21-P6 | No production statement is eligible merely from these substitutions. | Measurement either restores the old ladder diagnostically or leaves registered debt; no `packages/` file changes. | A first non-bit NEMO statement is isolated and a single NEMO-cited correction passes the complete ORCA2/GYRE/card landing bar. |

Failed predictions remain **REFUTED** in the receipt and are never rewritten.
The exact comparison excludes only provenance fields (`worktree`) and compares
every candidate-trajectory content leaf.

## Stop and landing rules

- If R21-P1 fails, no substitution result is interpreted.
- If R21-P4 fails, the merged ladder remains uncertified; the first and largest
  residuals become the next walk boundary.
- A diagnostic return to round 20 is not a reason to revert a faithful shared
  change.  Rule 12 keeps the shared change and registers the exposed ORCA2
  debt unless its own executed statement is shown wrong.
- Decision 58 and Decision 54 begin only after this reconciliation is complete,
  in later rounds.  Decision 57, the ranked slow-forcing walk, the fold-row
  mask/wind debt, and independent initial state remain untouched.
- No NEMO run is required or permitted in this round.

## Choices

ASKED: substitute the two registered merge owners one variable at a time and
re-certify the ORCA2 ladder before Decisions 58 and 54.

UNASKED: none.  No configuration value, carried-state field, stabilizer,
sea-ice selector, or scoring rule is changed.
