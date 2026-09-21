# NEMO testcase Lane 4 — ORCA2 card round 1 preregistration

Date: 2026-09-21

Parent: `dda3f3257dad5a7f86e1177f934afc531e6567d9`

Status: **PREREGISTERED BEFORE ROUND-1 MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card.  SI3 is not an
implementation target in this round.  In particular, this round will not edit
the card's frozen `unmeasured_features` tuple, select an ice configuration, or
interpret an SI3 difference as permission to change sea ice.

## Frozen records and comparison order

The pinned oracle root is
`variant_orca1ice_phase2x_a_10step_np2`
(`VARIANT_ORACLE_ORCA1ICE`); twin B is its reproducibility witness.  The
accepted five-category comparison root is
`variant_icebergs_off_phase2v_tke_a_10step_np2`
(`VARIANT_ORACLE_V2`).  The ORCA2 input deck is
`/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0`.

The ordered round-1 walk is:

1. rerun the pinned-root admission and frozen record/schema controls;
2. compare the current card's initial `T`, `S`, `u`, `v`, and `ssh` to the
   pinned root's kt=1 step-entry frame;
3. stop at the first non-bit field in that frame;
4. only for magnitude carry, compare the two admitted NEMO roots' entry and
   three stage frames through kt=10.  This is a
   `NEMO_ROOT_DIFFERENTIAL_NOT_LEGOESM_TRAJECTORY`, not a candidate result;
5. inventory the exact per-step surface-input records required to run the
   legoESM card through kt=10.  No substituted, reused, or inferred forcing
   frame is admissible.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R1-P1 | The ORCA1-ice A/B root remains admitted. | 116/116 streams decode; A/B are raw-byte identical; all admission plants bind. | Any missing/extra stream, twin difference, parser failure, or inert plant. |
| R1-P2 | At kt=1 the current card and `VARIANT_ORACLE_ORCA1ICE` have exact `T`, `S`, `u`, and `v`; `ssh` is the first non-bit field. | Four `0 / n` rows, followed by nonzero `ssh`; `ssh` max absolute error is in `[0.015, 0.016]` m. | Any earlier non-bit field, exact `ssh`, or magnitude outside the interval. |
| R1-P3 | The kt=1 `ssh` debt is the category-dependent initial snow/ice/pond mass adjustment, not an ocean time-step statement. | The current five-category card remains exact to `VARIANT_ORACLE_V2` at kt=1, while the pinned one-category root differs only in `ssh`; the compiled `iceistate` branch sums category mass and applies the global adjustment to both time levels. | The card differs from V2, a non-SSH field differs from ORCA1-ice, or the compiled branch does not execute the stated adjustment. |
| R1-P4 | The admitted-root SSH offset remains order `1.5e-2` m at the kt=10 entry. | V2-vs-ORCA1-ice kt=10 entry `ssh` max absolute difference is in `[0.014, 0.017]` m. | Magnitude outside the interval.  Other kt>1 field rankings are descriptive and will be labeled post-hoc. |
| R1-P5 | The accepted records cannot support a mechanically exact legoESM kt=1..10 trajectory because only the kt=1 post-`sbc` ocean-surface-input frame was captured. | The pinned root has kt=1 and lacks one or more of kt=2..10. | All ten exact per-step surface-input frames are present and schema-valid. |

P2 and P4 use maximum absolute error over the rank-0-owned domain after the
two-cell NEMO halo is removed.  Bit identity is `np.array_equal`, with no
tolerance.  The kt=1 plant changes one `T` value by one representable step;
the gate must reject it with a named `REFUSE` condition.

## Landing and stop rules

- A single-statement ocean landing is eligible only if all ten exact surface
  frames exist, the first non-bit statement is an ocean statement, every moved
  row is registered, the ORCA2 ladder constraints pass, and the mandated GYRE
  base/tip trajectory proof is exact.
- If P5 confirms, round 1 stops for a record acquisition.  The acquisition
  must be a fail-closed operator-run script, preserve the pinned ORCA1-ice
  configuration, add only a WRITE-only per-step surface recorder, run twins,
  and require inherited-record passivity.
- If P2/P3 confirm, changing the card's initial SSH would choose between the
  one-category oracle initialization and the card's frozen five-category
  initialization.  That is a configuration choice, so no such edit may land
  without a user decision.
- Failed predictions remain in the receipt as `REFUTED`; they are never
  rewritten after measurement.

## Frozen card registry

The expected tuple, in order, is:

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```

Any tuple change is an immediate gate failure.
