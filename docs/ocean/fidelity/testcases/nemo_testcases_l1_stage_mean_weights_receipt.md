# OVERFLOW-zps stage depth-mean WEIGHTS: receipt — FAITHFUL AND INERT

Preregistration `nemo_testcases_l1_stage_mean_weights_preregister.md`
(`07cad331d`, frozen before the arm).  Base `eccaf6007`; fix `9a23ea2c3`.
fp64 (`PrecisionPolicy.fp64()` + `JAX_ENABLE_X64=1`), CPU.  Baseline for every
"before" row: the phantom round's committed gates and statistics under
`phantom_velocity/after/`, byte-identical protocol.

## Verdict

**LANDED on Rule 0, and MEASURED INERT.  The preregistration's refute condition
is MET and its scaling estimate is RETRACTED.**

NEMO removes a REFERENCE-weighted depth mean from every WS-RK3 stage velocity
(`stprk3_stg.F90:440`, divisor built once at `domain.F90:145`); legoESM removed
a LIVE-weighted one.  Those are different operators wherever a face's two
columns differ in depth.  Adopting NEMO's rule changes the OVERFLOW trajectory
by nothing beyond the fifth significant digit at every one of 300 rows, and
changes all six 6120-step statistics by EXACTLY ZERO.  It stays landed because
it is NEMO's rule, not because it moved a number (the user principle: one NEMO
routine, one legoESM implementation).

## Rule 0

```fortran
DO_2D( 0, 0, 0, 0 )             ! barotropic velocity correction              ! stprk3_stg.F90:439
   zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)  ! :440
END_2D                                                                        ! :442
DO_3D( 0, 0, 0, 0, 1, jpkm1 )   ! corrected horizontal velocity               ! :443
   uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)           ! :444
END_3D                                                                        ! :446

hu_0(:,:) = hu_0(:,:) + e3u_0(:,:,jk) * umask(:,:,jk)                         ! domain.F90:145
```

legoESM: `_replace_stage_mean` used `sum(u*h_u_pre)/H_u_pre` with
`h_u_pre = min_cell_to_uface(h_k_pre)` (`ocean_model_latlon_cgrid.py:4369`).
The minimum is taken over two columns whose free-surface Jacobians differ, so
its per-level argmin can switch sides and the live weights are not a rescale of
the reference ones.  Equal-depth neighbours make the two coincide, which is why
the flat shelf measures one ULP and only the staircase moves.

## Predictions vs outcomes

| # | prediction | outcome | verdict |
|---|---|---|---|
| L2-P1 | OVERFLOW kt=10 `u` drops `>= 1.5x` | `5.422602e-06 -> 5.422582e-06` (`1.000004x`) | **REFUTED** |
| L2-P2 | OVERFLOW kt=60 `u` drops `>= 1.3x` | `3.238532e-05 -> 3.238533e-05` (`1.0000x`) | **REFUTED** |
| L2-P3 | kt=2 and kt=3 `u` move `< 5%` | both EXACTLY unchanged (`7.064252e-12`, `4.181444e-09`) | CONFIRMED |
| L2-P4 | LOCK BIT-IDENTICAL on every row | 300 rows: 59 better, 197 unchanged, 44 worse — but every move is at `1e-17`, e.g. kt=9 `ssh` `1.691355e-17 -> 1.669671e-17` | **REFUTED as written**; the residual is re-association, not operator change |
| L2-P5 | census does not close, moves `< 5.206e-04` | moves by EXACTLY `0.0` | CONFIRMED |
| L2-P6 | the unit test fails on reverted code | see "Non-vacuity" | CONFIRMED |
| L2-P7 | DINO reach NIL | CONFIRMED: the only DINO recipe is `nemo_dino_v1`, which selects `momentum_time_integrator="rk3"`, not `rk3_ws`, so no DINO card enters this branch.  **But a THIRD consumer exists and P7 did not enumerate it** — see "Reach" | CONFIRMED for DINO, INCOMPLETE as a reach statement |

**Refute condition MET** (neither kt=10 nor kt=60 `u` improved by `1.2x`).
**RETRACTED: the preregistration's scaling estimate.**  It predicted an induced
stage-velocity error of `1e-7`-`1e-6 m/s` from `shear x dw`, using a slope
shear of `5e-2 m/s`.  The first ten steps do not have that shear — the plume
has barely left the shelf — so the true induced error is orders smaller.  The
lesson is the one the skill already names: size a candidate against the shear
IN THE WINDOW BEING SCORED, not against the mature flow.

## Trajectory, before -> after (`stage_mean_weights/after_*.json`)

| kt | T | u | SSH |
|---:|---|---|---|
| 2 | `7.815970e-15` -> same | `7.064252e-12` -> same | `1.050549e-14` -> same |
| 3 | `5.725731e-12` -> same | `4.181444e-09` -> same | `1.509461e-13` -> same |
| 10 | `1.522637e-09` -> `1.522594e-09` | `5.422602e-06` -> `5.422582e-06` | `6.266335e-07` -> `6.265017e-07` |
| 20 | `1.112020e-08` -> `1.111966e-08` | `1.671404e-05` -> `1.671405e-05` | `1.556077e-06` -> `1.555955e-06` |
| 40 | `1.418330e-07` -> same | `2.307055e-05` -> same | `7.042411e-06` -> `7.042354e-06` |
| 60 | `1.183988e-05` -> same | `3.238532e-05` -> `3.238533e-05` | `1.234092e-05` -> `1.234089e-05` |

OVERFLOW kt<=60, 300 rows: **100 better, 89 unchanged, 111 worse**.
LOCK kt<=60, 300 rows: 59 better, 197 unchanged, 44 worse.
DISCLOSED, not reverted (Rule 8): the "worse" rows are all at ratio `1.0000`
except a handful of `S` rows at `1e-16` (e.g. OVERFLOW kt=10 `S`
`6.090366e-16 -> 8.120488e-16`) where the absolute values are at the last bits
of the field.  No row changes by more than the fifth significant digit; the
better/worse split is arithmetic re-association, not a physical direction.

kt=1..10 gates were also run on both cards with the same protocol; the same
picture holds (OVERFLOW 9 better / 28 unchanged / 13 worse; LOCK 5 / 42 / 3).

## Statistics, 6120 steps (`stage_mean_weights/stats/overflow_statistics.json`)

| metric | before | after | fp32 floor before | fp32 floor after | NEMO spread | verdict |
|---|---:|---:|---:|---:|---:|---|
| `final_temperature_histogram_tv` | `0.0330051` | `0.0330051` | `0.0062277` | `0.011261` | `0.044231` | WITHIN-SCHEME-SPREAD (unchanged) |
| `final_water_mass_census` | `0.00943798` | `0.00943798` | `0.00052063` | `0.0014132` | `0.0033621` | **OUTSIDE** (unchanged) |
| `instantaneous_u_linf` | `0.657229` | `0.657229` | `0.086588` | `0.30491` | `0.67269` | WITHIN-SCHEME-SPREAD (unchanged) |
| `plume_descent_m` | `1.43044` | `1.43044` | `0.0036066` | `0.036013` | `1499.6` | WITHIN-SCHEME-SPREAD (unchanged) |
| `plume_front_km` | `2.96245` | `2.96245` | `0.13553` | `0.20625` | `121.93` | WITHIN-SCHEME-SPREAD (unchanged) |
| `temperature_linf` | `0.319981` | `0.319981` | `0.053768` | `0.097966` | `0.35674` | WITHIN-SCHEME-SPREAD (unchanged) |

Status `OUTSIDE -> OUTSIDE`; counts `{OUTSIDE 1, WITHIN-SPREAD 5}` unchanged.
Every fp64 candidate distance is identical to every printed digit.  The fp32
FLOOR rows DO move (e.g. `u_linf` floor `0.0866 -> 0.3049`) because the fp32
arm is a different rounding path and this change re-associates its arithmetic;
that is a floor, not a candidate, and it is disclosed rather than read as a
result.

## Reach — the incomplete part of L2-P7

`rk3_ws` is selected at exactly three sites, found by grep and named here:
`nemo_testcase_recipe.py:66` (the two certified L1 cards, both gated above) and
**`nemo_recipe.py:970`, `build_nemo_gyre_recipe`** — a third consumer the
preregistration did not enumerate.  That card explicitly relies on this site
(`nemo_stage_mean_imposition=True`, its own comment citing `stprk3_stg:440`).
Its reach is **UNMEASURED** in this round.  It is not currently gateable: all
four `test_nemo_gyre_*` tests FAIL at the BASE tree `fd4a798c7` with
`ValueError: pgf_quadrature="nemo_trapezoid" is the hpg_sco recurrence and
requires pgf_scheme="nemo_sco"`, i.e. the GYRE card does not construct there,
with or without this change.  Named as an open item, not silently shipped.

## Non-vacuity and controls

- `tests/ocean/unit/test_nemo_ws_stage_mean_weights.py`, 2 tests.  The
  staircase test asserts the faithful and legacy arms give DIFFERENT states
  after 5 steps; reverting the fix makes the default live-weighted, the arms
  coincide, and it fails with `moved == 0.0`.  This was OBSERVED, not assumed:
  the first version stepped ONCE and failed at HEAD with exactly that message,
  because at kt=1 the free surface is flat and the two weightings are equal by
  construction on every card.  The flat-bottom control passes at `< 1e-14`.
- the private hook is a true one-variable ablation: with
  `--arm-legacy-live-stage-mean-weights` the OVERFLOW kt=1..10 gate reproduces
  the committed pre-fix run on **50 of 50 rows, bit for bit**.
- **9 test failures in `test_rk3_ws_and_mxl3.py`, `test_nemo_recipe.py` and
  `test_nemo_ab3am4_filter.py` are PRE-EXISTING**: all nine reproduce
  identically in a scratch worktree at the base commit `fd4a798c7` with this
  change absent (same `pgf_quadrature` ValueError).  Not from this round.
- 75 passed / 9 pre-existing failures / 1 xfailed across the WS-RK3, NEMO
  recipe, AB3-AM4 filter, fidelity-card, face-mask-rank, face-thickness,
  stage-mean-weights, census-probe and testcase-recipe suites.

## Figures

Regenerated with the committed plotter from THIS round's 6120-step states, at
`/data/abyssal/dbalwada/nemo-testcases-l1/figures_full_census/`:

| figure | sha256 |
|---|---|
| `overflow_zps_full_temperature_sections.png` | `d8a6444fda295a11f03d4aed824d76343509c3ea5ff7a7b713b68dcc82186cde` |
| `overflow_zps_full_statistical_metrics.png` | `f9d3805148fdc2c732528afc4b5d9c7181770f79395315b8c8dd1588da1da3ef` |
| `lock_exchange_zco_full_temperature_sections.png` | `5278a17476d810e82a5fda255297d9e5978e729a47459008d8bd4665328e357e` |
| `lock_exchange_zco_full_statistical_metrics.png` | `4da32eb6955084e66424801181cd8731f44e25280b9c209e2d64e9bf24c0d608` |

DISCLOSED: the OVERFLOW panels use this round's fp64/fp32 states and scorer
JSON; the LOCK panels reuse the older `full_statistical` LOCK states and report,
because no LOCK 61200-step run has been made since.

## What remains open, ranked

1. The census row itself: still `OUTSIDE`, still unattributed.  The chaos null
   (census receipt §5b) shows the case is not chaotic, so it IS a systematic
   operator difference; which operator is open.  Vertical and lateral tracer
   mixing are excluded by the oracle's own output (`votkeavt` exactly `0.0`),
   leaving the FCT advection, the advective BBL and the `ln_zad_Aimp` partition.
2. Whether `ln_zad_Aimp` ever activates during the descent.  Prior rounds
   measured it inert at kt=1 (`Cu_v = 1.66e-3` against a `0.8` threshold); the
   only oracle-side bound at the end of the run is a 17-hour MEAN `Cu_v` of
   `0.072`, which a time mean makes weak.
3. The `rk3_ws` GYRE consumer above, and the pre-existing `pgf_quadrature`
   validation failure that currently blocks gating it.
4. Open item 2 of the phantom round (the UP3 curvature `umask`,
   `dynadv_up3.F90:142-143`), untouched.
