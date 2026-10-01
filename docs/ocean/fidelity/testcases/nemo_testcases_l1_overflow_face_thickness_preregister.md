# OVERFLOW-zps stage face thickness and stage qco factor — preregistration

Date: 2026-09-01.  Scope: the certified OVERFLOW-zps and LOCK_EXCHANGE-zco
NEMO WS-RK3 cards, step 1 stages and the resulting `kt=2` entry.  Written and
committed BEFORE either arm was implemented or run.  Every number below is
produced by the committed stage-sweep gate's `face_thickness_and_qco_scaling`
block from the oracle kt=1 dumps alone (no legoESM arm), fp64, so the
predictions are reproducible and were frozen without seeing an arm result.

## Rule 4 pre-implementation search

Searched for an existing NEMO `r3u`/`e3u(Kmm)` face-thickness builder before
writing one: `grep -rn "r3u\|qco" packages/ocean/legoesm/ocean/` found
`vertical.py:140 nemo_qco_live_face_geometry_from_operands` and
`vertical.py:197 nemo_qco_live_face_thicknesses`, the shared canonical
`dom_qco_r3c` operand builders already used by the DINO ldfslp/dynzad and the
`wzv_call2_evaluation="nemo_literal"` tracer path
(`ocean_model_latlon_cgrid.py:5234`).  **Both arms reuse that helper; no
second implementation of the rule is written.**  The wrapper
`nemo_qco_live_face_thicknesses` pulls its operands off `z_coord.nemo_*`
fields the L1 testcase cards do not carry, so the arms call the
`_from_operands` primitive directly with operands assembled from the card's
own grid and reference thicknesses.

## Source verification (Rule 0)

* `domqco.F90:189-232` `dom_qco_r3c_RK3`, line 219-222:
  `pr3u(ji,jj) = 0.5*( e1e2t(ji,jj)*pssh(ji,jj) + e1e2t(ji+1,jj)*pssh(ji+1,jj) ) * r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)`.
  This is an `e1e2t`-weighted mean of **ssh**, divided by `hu_0` — NOT the
  mean of the two `r3t`, because `r3t` divides by each column's own `ht_0`.
* `domzgr_substitute.h90:127`: `e3u(i,j,k,t) = E3u_0(i,j,k) * (1 + r3u(i,j,t)*umask(i,j,k))`.
* `stprk3_stg.F90:272-273`: the stage transport is
  `zFu = e2u*e3u(ji,jj,jk,Kmm)*( uu(Kmm) + zub*umask )`, and the same triplet
  feeds `dyn_adv` (`:315,331-334`) and `tra_adv` (`:463,494,519`).
* `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:184`: `pe3u(:,:,:) = pe3t(:,:,:)`.
  Measured on the card's own `mesh_mask.nc`: on every wet U face,
  `e3u_0 == min(e3t_0_i, e3t_0_{i+1})` **exactly** (0 of 16900 faces differ).
  So the reference thickness is already faithful; only the STRETCHING differs.
* `ocean_model_latlon_cgrid.py:1064` (`_nemo_ws_stage_transport`, reached from
  `:4830,:4844,:4862` on both cards because
  `momentum_time_integrator == tracer_time_integrator == "rk3_ws"`):
  `hu_stage = min_cell_to_uface(h_stage)` with
  `h_stage = e3t_0*(1 + eta/H)` (`vertical.py:1127 compute_layer_thickness`).
  `min(e3t_0_i*(1+r3t_i), e3t_0_{i+1}*(1+r3t_{i+1})) != e3u_0*(1+r3u)`; the
  difference is first order in the ssh difference across the face.
* `stprk3_stg.F90:373-378` (`key_qco`, reached because
  `ln_dynadv_vec=.false.` and `key_qco` is set, so `lk_linssh` is false):
  `uu(Kaa) = ( (1+r3u(Kbb))*uu(Kbb) + rDt*(1+r3u(Kmm))*uu(Krhs) ) / (1+r3u(Kaa)) * umask`,
  and `dynzdf.F90` (`dyn_zdf` body lines 80-85) applies the identical factor at
  stage 3.  `ocean_model_latlon_cgrid.py:4940-4944,4851-4856,4868-4871` writes
  `u_raw = u0 + stage_dt * RHS` with no factor.  The TRACER analogue is already
  faithful (`:1215 _stage` divides `h_k_old*base - dt*flux_div` by `h_stage`,
  which is exactly NEMO `:552-554`), which is why the stage-1 tracer operand
  row is bit-exact at 0.0 while the stage-1 velocity row is not.

## Scaling BEFORE any owner label

### H1 — stage face thickness (`min` of stretched T thicknesses vs `e3u_0*(1+r3u)`)

Kmm ssh per stage from the oracle dumps; `Δe3u` = legoESM rule minus NEMO
rule on the 16900 wet U faces; the oracle's own per-stage tracer increment is
100 % advective on this card (K_h = K_v = 0, no surface forcing), so the
predicted T movement is the relative transport error times that increment.

| stage (Kmm level) | max abs Δe3u [m] | max rel Δe3u | oracle stage max abs ΔT [K] | predicted T movement [K] |
|---|---|---|---|---|
| 1 (Kbb, ssh ≡ 0)   | 0.0000e+00 | 0.0000e+00 | 3.4518e-04 | 0.0000e+00 |
| 2 (N+1/3)          | 1.3200e-03 | 6.5999e-05 | 6.0182e-04 | 3.971e-08 |
| 3 (N+1/2)          | 1.9800e-03 | 9.8999e-05 | 3.6980e-03 | 3.661e-07 |
| **sum**            |            |            |            | **4.058e-07** |

Faithful `kt=2` T residual to beat: **2.5485e-07 K**.  predicted / target = 1.59.

### H2 — stage velocity qco factor

With `uu(Kbb) = 0` at kt=1 the omitted factor leaves
`u_lego - u_nemo = u_nemo * (r3u(Kaa) - r3u(Kmm)) / (1 + r3u(Kmm))`.  The
stage barotropic correction (`stprk3_stg.F90:433-446`, legoESM
`_replace_stage_mean`) reinstalls the depth mean, so only the **baroclinic**
part of that error survives into the scored row.

| stage | max abs r3u(Kaa) (wet) | max abs r3u(Kaa)−r3u(Kmm) | predicted baroclinic u movement [m/s] | measured faithful residual [m/s] | pred/meas |
|---|---|---|---|---|---|
| 1 | 3.4495e-05 | 3.4495e-05 | 3.7635e-12 | 3.7618e-12 | 1.000 |
| 2 | 5.1743e-05 | 1.7248e-05 | 6.1965e-11 | 6.9081e-11 | 0.897 |
| 3 | 1.0349e-04 | 5.1743e-05 | 6.3212e-10 | 2.5984e-07 | 0.002 |

### LOCK_EXCHANGE-zco — both hypotheses are inert here, stated in advance

The oracle's `ssh` is identically zero at all three kt=1 stages (0 non-zero
cells of 134x7 in every stage dump) and legoESM's own `eta` after kt=1 is
4.78e-28 m.  With `ssh = 0` both `r3t` and `r3u` vanish and the flat-bottom
`e3t_0` is horizontally uniform, so the two face-thickness rules and the two
velocity updates are **algebraically identical**.  Neither hypothesis can own
LOCK's 9.7656e-11 stage-2 / 2.1388e-10 stage-3 / kt=2 u residual.

## Frozen predictions and decision rules

A confirming arm must move its target by a factor consistent with the scaling
above, within 2x, and must improve the residual.

**P1 (H1, OVERFLOW-zps).**  Turning the stage face thickness from the `min`
rule to NEMO's `e3u_0*(1+r3u)` moves the `kt=2` T field by
**2.03e-07 .. 8.12e-07 K** (4.058e-07 within 2x) and lowers the `kt=2` T
residual below 2.5485e-07 K.
REFUTED if the movement is below 2.03e-07 K or above 8.12e-07 K, or if the
residual does not improve.

**P2 (H1, OVERFLOW-zps, u).**  H1 reaches u only through momentum advection
and the diagnosed `w`; the stage-2 u residual is already 90 % accounted for by
H2, bounding H1's stage-2 u contribution at ~1e-11.  Predicted `kt=2` u
movement is **below 2.6e-08 m/s** (< 0.1x the 2.5984e-07 residual).  H1 is
therefore **preregistered as REFUTED for the u residual**; a movement above
2.6e-08 m/s would refute this reading instead.

**P3 (H2, OVERFLOW-zps).**  Applying the qco stage factor drives the stage-1
baroclinic u residual **below 1.3e-13 m/s** (>= 30x improvement on 3.7618e-12)
and the stage-2 baroclinic residual **below 1.4e-11 m/s** (>= 5x improvement on
6.9081e-11).  The `kt=2` u row is predicted to move only ~6.3e-10 m/s and will
**not** clear the bar; H2 is preregistered as **REFUTED as the stage-3 / kt=2
owner**.
REFUTED as the stage-1/2 owner if either residual fails to improve by the
stated factor.

**P4 (both, LOCK_EXCHANGE-zco).**  Both arms move every LOCK row by less than
1e-15 m/s (identically zero to roundoff).  A larger movement refutes the
"ssh ≡ 0 ⇒ inert" reading above and invalidates P1-P3's one-variable claim.

## What the arms are

Both changes belong to the unbranched NEMO WS-RK3 scheme identity; no public
config selector is added and legacy schemes keep their existing path.  The
one-variable ablation controls live only in the private
`_NEMOWSRK3TestHooks`:

* `legacy_stage_min_face_thickness=True` restores `min_cell_to_uface` /
  `min_cell_to_vface` in the stage transport (H1's control).
* `omit_stage_qco_factor=True` restores the unweighted
  `u_raw = u0 + stage_dt*RHS` stage update (H2's control).

Each hook is exercised by the stage-sweep gate, and the gate ships a planted
control that changes its exit code.
