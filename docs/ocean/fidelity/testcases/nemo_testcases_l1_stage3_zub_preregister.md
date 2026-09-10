# S-35 — stage-3 `zub` on the wrong side of `dyn_zdf`: preregistered scaling and predictions

Frozen BEFORE any arm was run, at legoESM `b9e8e60b6`
(branch `fidelity/nemo-branch-isomorphism-audit`, clean tree),
fp64 (`PrecisionPolicy.fp64()` + `JAX_ENABLE_X64=1`), CPU.
Cards: `OVERFLOW-zps` and `LOCK_EXCHANGE-zco`, the certified
`key_qco + key_RK3` NEMO testcase pair; oracle dumps under
`/data/abyssal/dbalwada/nemo-testcases-l1/phase3/{overflow,lock}_kt1_10/`.

## 1. What NEMO does (Rule 0 — quoted, not inferred)

`nemo_5.0.2/src/OCE/stprk3_stg.F90`:

```
430:         IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration
431:
432:         !
433:         !                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1
434:         !
439:         DO_2D( 0, 0, 0, 0 )             ! barotropic velocity correction
440:            zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)
441:            zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_0(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)
442:         END_2D
443:         DO_3D( 0, 0, 0, 0, 1, jpkm1 )   ! corrected horizontal velocity
444:            uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)
445:            vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)
446:         END_3D
```

Facts read off those lines, each of which CORRECTS a description in the
S-35 registry row and map:

1. **The correction runs at EVERY stage**, not only stage 3 — the `:433`
   banner says so explicitly ("All stages ... at Kaa = N+1/3, N+1/2 or N+1")
   and the `DO_2D`/`DO_3D` block sits OUTSIDE any `kstg` guard.  So
   legoESM's per-stage `_replace_stage_mean` at stages 1 and 2 DOES have a
   NEMO counterpart; the map's "one site" is a statement about the source,
   not about the number of executions.
2. **`dyn_zdf` runs at stage 3 ONLY** (`:430`, `IF( kstg == 3 )`).  So the
   before/after ordering question is vacuous at stages 1 and 2, and exists
   at stage 3 alone.
3. **The weights are the REFERENCE thicknesses `e3u_0` and `r1_hu_0`**
   (`:440`), NOT `e3u(Kaa)`.  The correction is an ADDITIVE, column-uniform
   shift of the whole column (`:444`, `+ zub*umask`), so it changes ONLY the
   `e3u_0`-weighted depth mean and leaves the baroclinic anomaly untouched
   by construction.
4. Time level: everything is at `Kaa` — `uu_b(Kaa)` against `uu(:,:,:,Kaa)`
   — i.e. the post-`dyn_zdf` after-level velocity at stage 3.

## 2. The two legoESM sites, and which one executes on each card

| site | symbol | file | when it runs |
|---|---|---|---|
| (a) BEFORE the implicit solve | `_replace_stage_mean` | `ocean_model_latlon_cgrid.py:4818` (called `:4987`, `:5003`, `:5021` for stages 1/2/3) | every `momentum_time_integrator="rk3_ws"` card, unconditionally (only the private `stage_barotropic_correction` test hook disables it) |
| (b) AFTER the implicit solve | `_impose_mean` + `_fixed_depth_means` | `ocean_model_latlon_cgrid.py:6403-6407`, `:6483-6497`, helper `:6580` | gated on `barotropic.nemo_stage_mean_imposition and _apply_implicit_vmix` |

INSTANTIATED, not read off a default (Rule 10) — both cards print
`nemo_stage_mean_imposition = False`, `implicit_vertical_mixing = True`,
`A_v = 1.0e-4 m^2/s`, `eta(t=0) = 0`.  So on OVERFLOW and LOCK **only site
(a) executes**: the correction is applied on the WRONG side of the implicit
vertical solve.  `_fixed_depth_means` uses `eta = 0` thicknesses, i.e. NEMO's
`e3u_0`/`r1_hu_0`; `_replace_stage_mean` uses the live `min_cell_to_uface`
thickness at `Kbb`, which is IDENTICAL to the reference at `kt = 1` because
`ssh(Kbb) = 0` there.

## 3. Scaling BEFORE the owner claim (Rule 4 / compute-discipline)

The two orders differ by a COLUMN-UNIFORM shift and nothing else, so the
reorder's leverage on any residual is bounded by the DEPTH-MEAN part of that
residual.  Measured with the gate's own reference weights
(`hu0 = min_cell_to_uface(h(eta=0))`), on the faithful arm's kt=1 stage-3
state against `oracle_stage_kt00000001_s3.bin`:

| card | stage-3 u residual (L-inf, wet U faces) | depth-mean part | baroclinic part | mean / full |
|---|---|---|---|---|
| OVERFLOW-zps | `2.598797930308122e-07` | `4.7531423241764514e-15` | `2.5987979778568926e-07` | `1.83e-08` |
| LOCK_EXCHANGE-zco | `2.1388041289210555e-10` | `1.929879867024198e-17` | `2.1388039364751699e-10` | `9.02e-08` |

Mechanism for why the depth-mean part is at the roundoff floor: with
`A_v = 1e-4`, zero bottom drag, no implicit surface stress and no-flux
boundary conditions, the implicit vertical momentum solve CONSERVES the
thickness-weighted column integral of `u`.  A conserved depth mean commutes
with a column-uniform shift, so applying `zub` before or after `dyn_zdf` is a
no-op to roundoff on these two cards.  ~5e-15 on O(1) velocities is a few
tens of ulp of fp64.

## 4. Predictions (frozen)

Arm = one variable, `barotropic.nemo_stage_mean_imposition = True` on an
otherwise byte-identical copy of each card's config (this re-imposes the
stage-3 depth mean AFTER the implicit solve, which is NEMO's `:430 -> :439`
order; no public selector and no card is changed).  Everything else — grid,
z_coord, dt, oracle dumps, masks, metric, window — held byte-identical to the
committed stage-sweep protocol.

| # | card | quantity | prediction |
|---|---|---|---|
| Z1 | OVERFLOW-zps | stage-3 u movement (arm vs faithful, L-inf) | `< 1.0e-13 m/s` |
| Z2 | OVERFLOW-zps | stage-3 u residual vs oracle | stays `2.59880e-07 m/s` to 6 significant figures |
| Z3 | OVERFLOW-zps | kt=2 u residual vs oracle | stays `2.59880e-07 m/s` to 6 significant figures |
| Z4 | OVERFLOW-zps | kt=2 T movement | `< 1.0e-12 K` |
| Z5 | OVERFLOW-zps | kt=2 SSH movement | `< 1.0e-14 m` |
| Z6 | LOCK_EXCHANGE-zco | stage-3 u movement | `< 1.0e-15 m/s` |
| Z7 | LOCK_EXCHANGE-zco | kt=2 u residual vs oracle | stays `2.13880e-10 m/s` to 6 significant figures |
| Z8 | LOCK_EXCHANGE-zco | kt=2 T and SSH movement | `< 1.0e-12 K` and `< 1.0e-14 m` |

## 5. Refute / confirm condition

The hypothesis under test is **"the stage-3 `zub` site order owns the
OVERFLOW kt=2 velocity debt (`2.598797930308122e-07 m/s`, normalized, currently
UNOWNED)"**.

- **CONFIRMED** iff the arm moves the OVERFLOW stage-3 u by at least
  `2.6e-08 m/s` (0.1x the residual) AND improves it.  Only then does the
  order become the unbranched `rk3_ws` behaviour and S-35 collapse to one
  symbol.
- **REFUTED** iff the movement is below `2.6e-08 m/s`.  The scaling above
  already predicts a movement 5 to 7 orders of magnitude under that
  threshold, so this preregistration is a prediction of REFUTATION.  On
  refutation the task stops: no other owner is tried, S-35 stays
  `ARTIFICIAL_BRANCH` with its two sites, and the debt stays UNOWNED and
  baroclinic.

If the arm moves anything by more than `1.0e-12 m/s` on LOCK — where the
oracle `ssh` is identically zero and the reference and live weights coincide
exactly — the ARM ITSELF is suspect (instrument check, Rule 3), not the
physics.
