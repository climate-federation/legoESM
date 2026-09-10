# S-21: the stage-1 momentum transport (`zub`) — enumeration, scaling,
# preregistered predictions

Frozen BEFORE the free-run arm was scored, at legoESM `bd4097f89` (branch
`fidelity/nemo-branch-isomorphism-audit`, worktree `/tmp/wt-branch-iso`), fp64
(`PrecisionPolicy.fp64()` + `JAX_ENABLE_X64=1`, every compared array printed
`float64`), CPU.  Cards `OVERFLOW-zps` (`dt = 10 s`, `ln_zad_Aimp = .true.`,
`momentum_advection = flux_form`, `momentum_time_integrator =
tracer_time_integrator = rk3_ws`) and `LOCK_EXCHANGE-zco`.  Oracle dumps
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/{overflow,lock}_kt1_10/`.

## 0. The lead

The UP3 upwind-selector round (`02179b0eb..aa923bede`) removed the OVERFLOW
kt=2 stage-3 baroclinic `u` error (`2.5988e-07 -> 4.5517e-10`) and left P7
REFUTED: kt=10 `u` `2.644298e-05 -> 2.644302e-05` and SSH
`9.237391e-05 -> 9.237444e-05` were unchanged, so the kt>=3 walk has another
owner.  That round's second finding, recorded UNMEASURED, is S-21.

## 1. Rule 0 — what NEMO advects with at stage 1

`stprk3.F90:186` runs `stp_2D(kstp, Nbb, Nbb, Naa, Nrhs)`, which produces the
external time-mean transport `un_adv`, then `:195` runs
`stp_RK3_stg(1, kstp, Nbb, Nbb, Nrhs, Naa)` — stage 1 has `Kmm == Kbb == Nbb`.

Inside the stage, ONE transport triplet is built per stage and used by every
stage (`stprk3_stg.F90:255-275`), with `n_baro_upd = np_HYB` (`:44`) selecting
the `CASE(np_LIN, np_HYB)` branch:

```fortran
zub(ji,jj) = un_adv(ji,jj)*r1_hu(ji,jj,Kmm) - uu_b(ji,jj,Kmm)   ! :267
zFu(ji,jj,jk) = e2u(ji,jj)*e3u(ji,jj,jk,Kmm)                  &  ! :273
   &          * ( uu(ji,jj,jk,Kmm) + zub(ji,jj)*umask(ji,jj,jk) )
```

and stage 1 is handed exactly that transport:

```fortran
CASE ( 1 )                                                       ! :311
   IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )   ! :315
```

The 3-D `Krhs` slot stage 1 accumulates into already holds `stp_2D`'s Kbb
terms: `dyn_hpg` ASSIGNS it (`stp2d.F90:128`; `dynhpg.F90:294` "RK3 case:
dyn_hpg always called first"), `dyn_ldf` (`:131`) and `dyn_vor` (`:146`) add,
and `dyn_adv_up3` at `:172` is called WITH `pUe`/`pVe`, which
`dynadv_up3.F90:158-202` routes to the 2-D barotropic seed ONLY — it never
touches the 3-D `Krhs`.  So NEMO's stage-1 3-D momentum RHS is
`hpg(Kbb) + ldf(Kbb) + vor(Kbb) + adv_up3(u(Kbb), zFu with zub)`.

## 2. The legoESM executing site, and the difference as a formula

`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`:

- `:4087-4093` — the step-entry `self.tendencies(...)` call.  It passes NO
  `momentum_flux_transport_velocity`, so the horizontal momentum advection uses
  the default `Q = h u(Kbb)`.  Its depth mean becomes `F_slow_{u,v}`, which
  seeds the barotropic solve at `:4826`; the baroclinic remainder is
  `du_dt_pert` (`:4211`).
- `:5011-5017` — stage 1 steps with `du_dt_pert` (plus the Aimp vertical UP3
  `_vert0`, which is built from `_g0` and DOES carry the correction).
- `:4439-4454` — stages 2 and 3 go through `_mom_pert_ws`, which passes
  `momentum_flux_transport_velocity = u_in + (Hu_avg/H_u - mean(u_in))`; that
  bracket is NEMO's `zub` term for term, and
  `_NEMOWSRK3TestHooks.momentum_transport_reconcile` defaults to `True`.

So, per unit `e2u`, with `zub = un_adv/hu(Kbb) - uu_b(Kbb)`:

```
  NEMO   stage 1:  F_u = e3u(Kbb) ( u(Kbb) + zub )
  legoESM stage 1: F_u = e3u(Kbb)   u(Kbb)
  difference:      dF_u = e3u(Kbb) zub * umask                     (dF_v likewise)
```

and, the UP3 T-point flux being linear in the transport once the branch is
selected by the advected-velocity pair (`dynadv_up3.F90:166,169-170`, S-44),

```
  d(du/dt)|stage1 = -1/(4 e1u e2u e3u) [ d_i( (dF_u,i + dF_u,i+1)(zui - g1 zl_u) )
                                       + d_j( (dF_v,i + dF_v,i+1)(...)          ) ]
```

i.e. the extra advection of `u` by the depth-uniform barotropic correction
`zub`.  It is EXACTLY zero at kt=1 (the cards start from rest, so `zui`, the
curvature and the F-point advected pair all vanish and the transport multiplies
zero), and live from kt=2.

The arm is the private hook
`_NEMOWSRK3TestHooks.legacy_stage1_no_transport_advection`; NEMO has no such
switch.  The correction is applied as the DIFFERENCE of `_mom_pert_ws`
evaluated with and without the transport, so every term the step-entry call
carries and the stage helper does not cancels exactly — the arm is
one-variable and its legacy side is bit-identical to `bd4097f89`.

## 3. Scaling (measured before the arm, fp64)

Stage-1 RHS difference at the OVERFLOW kt=2 entry (`expose_momentum_stage=1`
A/B; `dRHS = 3/dt * du1`, the depth mean replaced by the identical barotropic
target in both arms, so only the baroclinic part survives):

| quantity | value |
|---|---:|
| stage-1 `u1` difference | `1.045340e-06 m/s` |
| implied stage-1 RHS difference | `3.136020e-07 m/s^2` (face 21, k=24) |
| depth-mean part | `5.27e-19` (replaced) |
| top faces | 21 `3.14e-07`, 20 `1.64e-07`, 19 `1.44e-07`, 22 `3.73e-08` |
| vertical structure | LINEAR in depth, sign change at k=12 (`-2.77e-07` at k=0 to `+3.14e-07` at k=24) |

That is 12x the stage-3 RHS error the UP3 round removed (`2.60e-08 m/s^2`) —
but it enters at stage 1, where it reaches the step's exit only through two
further stages, not through the `x dt` of a stage-3 term.

One-step response from each free-run entry state (hook A/B on the SAME entering
state), against the CURRENT legoESM-vs-NEMO residual at that entry.  The probe
is calibrated: its `u` and `ssh` residual columns reproduce the receipt's
kt=3/10 rows exactly (`9.079022e-09`, `3.774823e-09`, `2.644302e-05`,
`9.237444e-05`), and its `T` column is the receipt's row times the registered
20 K normalizer.

OVERFLOW-zps:

| kt | field | residual now | S-21 one-step response | corr | share |
|---:|---|---:|---:|---:|---:|
| 3 | u | `9.079022e-09` | `2.664653e-11` | `0.3795` | `0.003` |
| 3 | T | `1.557794e-10` | `2.903811e-11` | `-0.178` | `0.186` |
| 3 | ssh | `3.774823e-09` | `0.0` exactly | — | `0.000` |
| 4 | u | `1.683855e-07` | `8.850816e-12` | `0.0145` | `0.000` |
| 5 | u | `2.503112e-06` | `1.115184e-11` | `0.0028` | `0.000` |
| 10 | u | `2.644302e-05` | `5.115082e-12` | `0.0081` | `0.000` |
| 10 | ssh | `9.237444e-05` | `0.0` exactly | — | `0.000` |

LOCK_EXCHANGE-zco: kt=3 `u` `7.233653e-15` vs response `4.290815e-17`
(share `0.006`), kt=10 `u` `1.314126e-11` vs `1.982328e-16`, ssh `0.0` exactly
at every kt.

The SSH zero is STRUCTURAL, not small: the barotropic solve runs before the
stage ladder, is seeded by the step-entry `F_slow` that this term does not
touch, and `_replace_stage_mean` re-imposes the identical depth mean at every
stage — so the stage-1 transport cannot move `ssh` within a step at all.  The
SSH walk is the largest OVERFLOW row at kt=10.

## 4. Predictions (frozen)

Baseline = the `bd4097f89` rows already in
`nemo_testcases_l1_phase3_receipt.md` ("Trajectory ... after" columns).

| # | card | quantity | prediction | REFUTED if |
|---|---|---|---|---|
| Q1 | both | kt=2 `T`, `S`, `u`, `v`, SSH | BIT-IDENTICAL — the term multiplies the advected velocity, which is exactly zero at kt=1 from rest | any row moves at all |
| Q2 | OVERFLOW | kt=3 `u` | `9.079022e-09`, moving `< 1e-10` (one-step injection `2.66e-11`) | moves `> 1e-9` |
| Q3 | OVERFLOW | kt=3 SSH | BIT-IDENTICAL `3.774823e-09` (the term cannot move ssh in one step; kt=3 is one S-21-live step after kt=2) | any move |
| Q4 | OVERFLOW | kt=10 `u`, SSH | `2.644302e-05` and `9.237444e-05` move `< 1%` | either moves `> 10%` |
| Q5 | OVERFLOW | kt=60 `u`, `T`, SSH | move `< 5%`; no direction claimed | — |
| Q6 | LOCK | kt=2 stage sweep + kt=2 rows | stay AT BAR (`< 1e-15`), bit-identical | any row leaves the bar |
| Q7 | LOCK | kt=3, kt=10, kt=60 | `u` moves `< 1%`; `T`, SSH bit-identical at kt=3 | `u` moves `> 10%` |
| **Q8 (OWNERSHIP, the question of this round)** | OVERFLOW | does S-21 own the kt>=3 walk? | **NO** — the walk drops `< 2x` at kt=10 in both `u` and SSH | the walk drops `>= 2x` at kt=10, which would CONFIRM ownership |
| Q9 | OVERFLOW | the six 6120-step statistics rows (`final_temperature_histogram_tv`, `final_water_mass_census`, `instantaneous_u_linf`, `plume_descent_m`, `plume_front_km`, `temperature_linf`) | NOT RUN under refutation: with a per-step injection `<= 0.3%` of the `u` residual and exactly `0` in ssh, the 6120-step decorrelated state cannot attribute them, and the campaign's own fp32 floor already exceeds the effect. Run ONLY if Q8 confirms | — |

Rule 8: nothing is reverted because a row worsens.  Under Q8-refutation the
fix is NOT landed in this round either — it is a real, cited faithfulness
debt whose landing needs its own statistics round, and it is reported as such.

## 5. Instrument finding, separate from S-21: `plume_descent_m`

`nemo_testcase_full_statistics.py:797-801` reduces `plume_descent_m` to
`max(centres[active & (T <= 15 C)])`, and `:1112-1120` scores
`candidate = max_t |L64 - N2|` against `spread = max_t |N4 - N2|`.  On the
registered times `(0, 8.5 h, 17 h)` the final-time value is effectively
two-valued — the cold plume either still has a cell on the deep bottom or does
not:

| arm | t=0 | t=8.5 h | t=17 h |
|---|---:|---:|---:|
| N2 (NEMO, registered namelist) | 490.0 | 1986.58 | **1989.97** |
| N4 (NEMO, alternative namelist) | 490.0 | 1969.52 | **490.34** |
| L64 before the UP3 fix | 490.0 | 1969.63 | 1989.80 |
| L64 after the UP3 fix | 490.0 | 1987.95 | **490.17** |

So NEMO's OWN two arms already differ by the whole basin (`1499.63 m`), and the
UP3 round's `16.96 -> 1499.8 m` is the candidate crossing to the other branch:
`1499.80` against a spread of `1499.62`, i.e. OUTSIDE by `0.18 m` inside a
`1500 m` step.  The same event reads WITHIN-SCHEME-SPREAD through
`plume_front_km` (`117.13` vs spread `121.93`).  The row is a
threshold-crossing artifact of a metric whose own reference spread spans its
entire dynamic range; it is not evidence of a physical degradation, and it can
neither confirm nor refute any fix.

NOT FIXED here.  Any robust reducer (a volume-weighted descent, a percentile,
or a deeper time sampling) is a new scientific choice that changes this row and
must be preregistered as its own round — ASK, not a silent metric edit.
