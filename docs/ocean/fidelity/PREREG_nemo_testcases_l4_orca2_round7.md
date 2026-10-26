# NEMO testcase Lane 4 — ORCA2 card round 7 preregistration

Date: 2026-09-22

Parent: `90ab3e871a4d7cfbc11441b345b007016726d348`

Status: **PREREGISTERED BEFORE ANY ROUND-7 MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card and round 6's OPEN items 1
and 2.  The six-entry sea-ice registry is frozen and stays out of scope.
Decision 52's labels are binding: every number below is either **given NEMO's
entry** (the twin that loads NEMO's recorded operand) or **independent**
(legoESM's own initial state).  The two are never mixed in one table.

## The two transcriptions this round owes

**(a) The independent ORCA2 initial state.**  Round 6 named the first
full-domain non-bit statement as kt=1 entry temperature and attributed it to
the active ORCA_R2 hand alterations the card omits.  This round transcribes
that branch: the Alboran Sea temperature and salinity decrements and the Red
Sea deep-temperature assignment that the executed initial-condition reader
applies to the time-interpolated input fields before the land mask, at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:218-253`.  The
index arithmetic is read off the compiled global-to-local map at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/mppini.f90:1586-1594`, not
inferred from the residual pattern.

**(b) The EOS-80 stratification branch.**  The record's namelist selects
EOS-80 (`ln_eos80 = .true.`); the production stratification helper accepts
only the simplified and TEOS-10 forms and refuses the card's selection.  The
compiled expansion-coefficient routine runs ONE polynomial branch for both
TEOS-10 and EOS-80 and the stratification assembly itself carries no equation
-of-state branch at all, so the transcription is a coefficient-set selection
on the existing evaluator, not a new formula.  The card already names
`eos80`; nothing in any selector or namelist mirror changes, and the GYRE card
stays on TEOS-10.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R7-P1 | The compiled expansion-coefficient branch is shared between TEOS-10 and EOS-80, and the stratification assembly has no equation-of-state branch. | The compiled routine's polynomial case names both forms and the assembly routine contains no such selection. | Either routine carries a separate EOS-80 formula. |
| R7-P2 | The EOS-80 coefficient set already committed in the model reproduces the compiled initialization block exactly, all 52 density, 36 thermal, 34 haline coefficients plus the four normalization constants. | A mechanical extraction from the compiled file equals the committed set on every entry, bit for bit. | Any single coefficient or normalization constant differs. |
| R7-P3 | The omitted hand alterations own the ENTIRE independent kt=1 entry temperature and salinity mismatch. | After transcription, independent kt=1 entry temperature and salinity are bit-identical on all 799,200 cells each. | Any cell of either field remains unequal. |
| R7-P4 | The alteration region is the inner-domain one-based box i=140..154, j=101..109 for the Alboran pair and i=147..159, j=87..96 for the Red Sea assignment, derived from the compiled index map with a two-cell halo. | R7-P3 confirms with exactly these bounds and no shift. | Any index shift, transpose, or level change is needed to reach bit identity. |
| R7-P5 | Adding the EOS-80 arm leaves the GYRE trajectory bit-identical. | Base and tip GYRE ladder comparisons give zero differing rows and array-equal residuals, and the 30-day member snapshots are byte-identical. | Any differing row, unequal residual array, or differing snapshot digest. |
| R7-P6 | With both transcriptions landed, the ORCA2 ladder passes the kt=1 entry and reaches at least the first Runge-Kutta stage, so the first non-bit statement moves strictly later than round 6's. | The first non-bit row is at or after kt=1 stage 1 and carries a cited compiled owner and a registered kt=10 magnitude. | The ladder still stops before stage 1, or an entry field other than sea-surface height remains non-bit. |

Failed predictions stay in the receipt as **REFUTED**.

## Controls and stop rules

- Reverting the hand-alteration transcription must restore round 6's recorded
  1,283 unequal temperature cells and 720 unequal salinity cells.
- Perturbing one representable kt=1 temperature value must make entry identity
  refuse.
- Perturbing one extracted EOS-80 coefficient must make the coefficient audit
  refuse.
- A rigid two-line shift of a compiled citation must fail the citation gate.
- Decision 52's sea-surface-height bridge remains the ONLY explicit entry
  operand replacement; the residual independent sea-surface-height difference
  stays owned by the out-of-scope initial sea-ice category configuration and is
  not "fixed" this round.
- No configuration, carried-state, stabilizer, sea-ice selector, default, or
  NEMO source change is authorized.  The EOS-80 arm is an ADDITION to an
  allowed set, never a change of any default.

## Frozen card registry

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```
