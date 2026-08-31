# Preregistration: DINO cold-start Euler operator peel

Date: 2026-08-31. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **FROZEN BEFORE NEW STAGE SCORES OR PHYSICS CHANGES.** The v5
initialization artifact admitted every geometry, profile, anchor, and resolved
state row with zero differences. The next legal target is therefore the first
failed Euler row, `conditional_euler_T_after_trazdf`, whose max error is
`1.1374146413256625e-4 degC`.

This peel is CPU-only and consumes the already recorded raw first-step dumps in
`RUN_KT2`; it does not execute NEMO, MPI, or a GPU.

## Executed oracle control flow

DINO enters `cfgs/DINO/MY_SRC/stpmlf.F90` with `l_1st_euler=.TRUE.`.
Lines 134--137 set `rDt=rn_Dt` and `r1_Dt=1/rDt`; initialization has already
collapsed `Nbb=Nnn`. The executed tracer order is `tra_sbc` at line 498,
`tra_qsr` at 505, `tra_adv(Nbb,Nnn,Naa)` at 528,
`tra_ldf(Nbb,Nnn)` at 548, and `tra_zdf(Nbb,Nnn,Nrhs,Naa)` at 551.
The split-explicit momentum corrector is not skipped on this Euler step:
line 578 calls `mlf_baro_corr(kstp,Nnn,Naa,uu,vv)`. Only after filtering and
level rotation do lines 685--688 restore `rDt=2*rn_Dt` and clear
`l_1st_euler`. Thus the faithful first step is not a generic Euler shortcut:
it is the MLF operator with collapsed levels, one-step coefficients, and the
normal barotropic corrector.

## Frozen first-divergence order

The committed probe must score the following rows in order and stop assigning
an owner at the first over-bar row. All are pointwise max-absolute comparisons
on the same registered NEMO wet cells/faces used by the v5 probe. The bar is
`1e-15` in the row's stated units; no new Rule-1b waiver may be invented here.

| order | row | NEMO dump / owner | units |
|---:|---|---|---|
| 0 | `time_level_collapse` | `Nbb == Nnn`, `rDt=2700` receipt | exact boolean |
| 1 | `T_after_trasbc`, `S_after_trasbc` | stage 14 | degC/s, PSU/s |
| 2 | `T_after_traqsr`, `S_after_traqsr` | stage 17 | degC/s, PSU/s |
| 3 | `T_after_traadv`, `S_after_traadv` | stage 20 | degC/s, PSU/s |
| 4 | `T_before_traldf`, `S_before_traldf` | stage 22 | degC/s, PSU/s |
| 5 | `T_after_traldf`, `S_after_traldf` | stage 23 | degC/s, PSU/s |
| 6 | `T_after_trazdf`, `S_after_trazdf` | stage 21 | degC, PSU |
| 7 | `U_wind_slow_forcing` | registered `utau`/stage-7 operands | m/s2 |
| 8 | `V_hpg_slow_forcing` | registered HPG/stage-7 operands | m/s2 |

Rows 1--5 compare the shared RHS accumulator, not a state update. Row 6
reconstructs NEMO's Euler `Naa` input from the admitted `Nbb=Nnn` state and
stage-23 RHS, then runs or independently checks the implicit vertical solve.
It may not merely compare the final field again. Rows 7--8 are reached only
after the tracer owner is resolved or honestly dispositioned.

For every failed cumulative RHS row the probe must also print an incremental
residual against the immediately preceding row. A cumulative difference is
not evidence that the last-called operator owns it.

## Repair discipline and controls

At the earliest failed row, read the executed NEMO operator source and quote
`file:line` before changing code. Reuse an existing faithful selector when the
campaign already has one. A new selector is legal only when the difference is
oracle-defined, defaults faithful on the two DINO oracle cards, preserves all
non-oracle defaults byte-for-byte, and has a fail-closed construction guard.

The committed probe must include controls that can fail:

- omit the surface tracer RHS once;
- use `2*rn_Dt` once on the collapsed Euler row;
- swap the directly sampled U-face stress for the cell-centred projection;
- perturb one live V-HPG accumulator operand by one representable value; and
- plant one live wet-cell mismatch before the final gate.

Each control must move a nonzero live population or make the gate reject the
artifact. A zero-valued plant is invalid.

## Frozen outcomes and launch gate

- `EULER_AT_BAR`: every row 0--8 is at bar and every control fires.
- `EULER_DEBT_<FIRST_ROW>`: the earliest failed row has an identified executed
  owner and a source-backed fix/design disposition.
- `EULER_BLOCKED_<FIRST_ROW>`: the necessary oracle operand was not recorded;
  the missing dump and exact follow-up run are named, and no launch arm emits.

The independent-year prediction remains frozen from the parent prereg:
day-360 SST RMS should collapse from `0.39 degC` toward the bridged
`~0.01 degC` class. A human-owned arm is emitted only for `EULER_AT_BAR` (or a
pre-existing signed Rule-1b clearance), and must be 11,520 fp64 steps,
standalone member 0, public `nemo_dino_kamm_mlf`, with `--snap-final`.
`<=0.02 degC` confirms, `>=0.10 degC` refutes, and the interval is reported as
inconclusive.
