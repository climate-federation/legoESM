# OVERFLOW-zps stage depth-mean WEIGHTS: preregistration for the reference-weight arm

Frozen before the arm.  Base tree `eccaf6007` (the census round's tip; no model
code changed there).  fp64, CPU.  The candidate was measured by the committed
`nemo_testcase_census_map_probe.py faces` command and cross-checked by an
independent reviewer probe that reproduced its values to the last digit.

## Rule 0 -- NEMO's rule, quoted

```fortran
!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1
DO_2D( 0, 0, 0, 0 )             ! barotropic velocity correction                       ! stprk3_stg.F90:439
   zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)  ! :440
   zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_0(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)  ! :441
END_2D                                                                                    ! :442
DO_3D( 0, 0, 0, 0, 1, jpkm1 )   ! corrected horizontal velocity                           ! :443
   uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)                        ! :444
   vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)                        ! :445
END_3D                                                                                    ! :446
```

with the divisor built ONCE from the REFERENCE thicknesses:

```fortran
hu_0(:,:) = hu_0(:,:) + e3u_0(:,:,jk) * umask(:,:,jk)      ! domain.F90:145   (r1_hu_0 = 1 / hu_0)
```

So the depth-mean operator NEMO removes from every WS-RK3 stage velocity is
`M_ref(x) = SUM(e3u_0 * x) / hu_0` -- **reference** thicknesses, fixed for the
whole run, and `M_ref(umask) = 1` exactly by construction.

legoESM removes a LIVE-weighted mean instead: `_replace_stage_mean` uses
`sum(u * h_u_pre) / H_u_pre` with
`h_u_pre = min_cell_to_uface(h_k_pre)` (`ocean_model_latlon_cgrid.py:4370`),
`h_k_pre` being the ssh-stretched thickness, and the stage TARGET is formed
with the same live pair.

The two are NOT algebraically equal.  Under z*/qco a single column's levels all
carry one `1 + r3u`, so a column-wise reference mean and a column-wise live
mean coincide -- but `h_u_pre` is a MIN over two columns whose Jacobians
differ, so the per-level argmin can switch sides and the live weights are not a
uniform rescale of the reference ones.

## Measured candidate (`census_map/faces.json`)

Per-level weight difference `|h_u_live/H_u_live - e3u_0/hu_0|`, free run:

| kt | max over wet faces | wet levels `> 1e-12` | at the `k=24` injection faces 20/21/22 |
|---:|---:|---:|---:|
| 1 | `0.0` | 0 | `0.0` |
| 2 | `2.323655e-09` | 26 | `6.94e-18` |
| 5 | `3.761720e-06` | 105 | `6.94e-18` |
| 9 | `8.609751e-06` | 167 | `6.94e-18` |
| 10 | `7.570658e-06` | 173 | `6.94e-18` |

Scale-compatibility, computed BEFORE the arm: both weight vectors sum to 1 over
a column, so the induced error in the removed mean is `sum((u - ubar) * dw)` --
only the SHEAR is exposed.  With column shear of order `5e-2 m/s` on the slope
and `dw` of order `8e-6` on a handful of levels, the induced stage-velocity
error is `1e-7`-`1e-6 m/s`, i.e. `1e-6`-`3e-6` normalized -- the same order as
the OVERFLOW kt=10 `u` row (`5.42e-06`).  Scale-compatible with the walk.

It is NOT scale-compatible with the `k=24` family (one ULP there, because gate
faces 20/21/22 have EQUAL bathymetry on both sides so the min picks one uniform
Jacobian), and this arm therefore predicts NO movement at kt=2.

## Predictions

| # | prediction |
|---|---|
| L2-P1 | OVERFLOW kt=10 `u` `5.422602e-06` drops `>= 1.5x` |
| L2-P2 | OVERFLOW kt=60 `u` `3.238532e-05` drops `>= 1.3x` |
| L2-P3 | OVERFLOW kt=2 `u` `7.064252e-12` moves by `< 5%` and kt=3 `u` `4.181444e-09` by `< 5%`: the `k=24` faces have equal-depth neighbours, so this arm has no leverage there (measured one ULP) |
| L2-P4 | LOCK_EXCHANGE-zco is BIT-IDENTICAL on every trajectory row, kt=1..60, both because its bottom is flat (every face's two columns have equal depth, so the min-rule picks one uniform Jacobian) and because that makes the two weightings identical there |
| L2-P5 | `final_water_mass_census` does NOT close: it moves by less than the fp32 precision floor `5.206e-04`, leaving the row OUTSIDE.  A move `>= 0.00336` would be a surprise and would be reported as one |
| L2-P6 | the new unit test FAILS on a tree with only this change reverted |
| L2-P7 | DINO reach NIL: every DINO recipe resolves `momentum_time_integrator='euler'`, so no DINO card enters the `rk3_ws` branch.  Verified by grep, not assumed |

## Refute condition

If neither the kt=10 nor the kt=60 OVERFLOW `u` row improves by at least
`1.2x`, the candidate is REFUTED as an owner of the walk.  The faithfulness
change still stands on Rule 0 (it is NEMO's rule either way) but is reported as
faithful-and-inert, and the scaling estimate above is retracted as too
optimistic.

## The arm

ONE variable: the depth-mean operator's weights, applied consistently at both
sites of the same operator (the mean REMOVED from each stage velocity, and the
TARGET that replaces it -- they are the same operator `M` and mixing two
weightings would be incoherent).  Default moves to NEMO's rule; a private
`_NEMOWSRK3TestHooks.legacy_live_stage_mean_weights` restores the live-weight
rule for the one-variable comparison.  NEMO has no such switch, and no public
selector is added.

## Choices, named

* the hook's default is the FAITHFUL value and the legacy behaviour is behind
  the private flag, matching the two prior rounds (`legacy_2d_stage_face_mask`,
  `legacy_hadv_min_face_thickness`).  A default that preserved the old
  behaviour would be a bug with a knob.
* the `v`-half changes with the `u`-half; on both cards the three-row channel
  makes it inert, as in the phantom round.
* nothing else changes: no scheme selection, no namelist value, no limiter, no
  public config field.
