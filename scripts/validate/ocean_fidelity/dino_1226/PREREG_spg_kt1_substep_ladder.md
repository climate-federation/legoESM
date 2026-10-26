# PREREG — NEMO `dyn_spg_ts` at kt=1 from rest, statement by statement

Registered BEFORE the measurement, against `cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90`
(the compiled source) and `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py`.
Card: `nemo_dino_kamm_mlf` standalone, one Euler step from rest.

Resolved oracle configuration, read from the run's own log rather than the deck:
`ln_bt_fw = F`, `nn_bt_flt = 2`, `ln_bt_auto = T` -> `nn_e = 23` (ocean.output:1102),
`icycle = 45` (ocean.output:1265), `rDt_e = rn_Dt/nn_e = 117.3913043478 s`
(dynspg_ts.f90:1203 — note `rn_Dt`, NOT the Euler step's `rDt`), `ln_dynadv_vec = T`
(vector form), `ln_dynvor_een = T`, `ln_scal_load = F` so `zldg = grav`,
`ln_wd_dl = F`, `ln_apr_dyn = F`, `ln_tide = F`, `ln_bdy = F`.

`l_1st_euler` changes exactly three things and none of them is a separate code path:
`rDt = rn_Dt` (stpmlf.f90:131-133), `ll_init = T` (:241) so the barotropic sub-state is
re-initialised and the two-substep AB/AM ramp fires, and `ll_fw_start = T` (:242) which
moves the boxcar centre from `2*nn_e` to `nn_e`, giving `icycle = 45` instead of 68.

## The ordered table

`N` = NEMO statement in execution order. `L` = the legoESM statement that stands in for
it. Verdict is what the source says, before any number was taken.

| # | NEMO (`dynspg_ts.f90`) | legoESM (`barotropic_latlon_cgrid.py`) | verdict |
|---|---|---|---|
| N1 | :489 `sshn_e = pssh(Kbb)`, `un_e = puu_b(Kbb)`, `hu_e = hu(Kbb)` — CENTRED seed, BEFORE level | `_baro_seed` `eta_init/u_init/v_init` from `_barotropic_before_state`, `barotropic_seed_evaluation='nemo_literal'` | MATCH. From rest all are 0 |
| N2 | :562 `ua_e = za1*un_e + za2*ub_e + za3*ubb_e` | :1425 `U_mid = za_i[0]*U_bar_c + za_i[1]*Ub_c + za_i[2]*Ubb_c` | MATCH incl. the `ll_init` ramp rows `(1,0,0)`, `(1,0,0)` |
| N3 | :574 `zsshp2_e = za1*sshn_e + za2*sshb_e + za3*sshbb_e` | :1441 `eta_mid = za_i[0]*eta_c + ...` | MATCH |
| N4 | :587 `zhup2_e = hu_0 + r1_2*r1_e1e2u*(e1e2t*zsshp2_e + e1e2t*zsshp2_e)*ssumask` | `_ssh_avg_face_depths(eta_mid)` (`barotropic_face_depth='nemo_ssh_avg'`) | MATCH in form. **`hu_0` is where R0 bites** — see below |
| N4b | `hu_0` built from `e3u_0`, i.e. from the 3-D `e3t_0` | built from `z_coord.h_partial`, i.e. from the 1-D `dz_ref` | **DIFF** — the two ladders are not the same one below level 25 |
| N5 | :607 `zhU = e2u*ua_e*zhup2_e` | `nemo_literal_metric_transports` | MATCH |
| N6 | :627 `ssha_e = ( sshn_e - rDt_e*( ssh_frc + zhdiv ) )*ssmask` | :1489 `eta_c - dt_s*div_flux + dt_s*F_slow_eta*mask` | **DIFF in association** — NEMO sums the two forcings THEN multiplies once. Degenerate on DINO, where `ssh_frc` is identically zero (no E-P); a live defect on any card with a freshwater flux |
| N7 | :636 `un_adv += wgtbtp2(jn)*zhU*r1_e2u`, one divide by `SUM(wgtbtp2)` at :882 | `nemo_literal_accumulate_transport` + `_transport_divisor` | MATCH — raw weights, one divide |
| N8 | :651 `zsshu_a` from `ssha_e` | `_ssh_avg_face_depths(eta_new)` | MATCH |
| N9 | :662 `ts_bck_interp` -> `zsshp2_e = za0*ssha_e + za1*sshn_e + za2*sshb_e + za3*sshbb_e` | :1509 `eta_pgf = zb_i[0]*eta_new + ...`, `flt2` literals `(0.614,0.285,0.088,0.013)` + the `jn=1,2` ramp | MATCH |
| N10 | :671 `zu_spg = -zldg*(zsshp2_e(i+1)-zsshp2_e(i))*r1_e1u` | `_nemo_literal_barotropic_pressure_gradient` | MATCH (`zldg == grav` because `ln_scal_load=F`) |
| N11 | :676 `dyn_cor_2D(ua_e, va_e)` — EEN, both components from the SAME mid-step velocities, coefficients frozen at `dyn_cor_2D_init(Kmm)` | `een_barotropic_coriolis(U_mid, V_mid, een_pre)` | MATCH in form. Coefficients ride `r1_hu(Kmm)` and `e3u/e3v(Kmm)` — **R0 again** |
| N12 | :714 `zu_trd += zCdU_u * un_e * hur_e` — substep-START velocity, reciprocal depth | :1571 `-drag_r_u*U_bar_c / max(H_u, min_water_col)` | **DIFF**: NEMO multiplies by a stored reciprocal, legoESM divides. `a*(1/b) != a/b`. ~1 ulp; zero at substep 1 from rest |
| N13 | :732 `ua_e = ( un_e + rDt_e*( zu_spg + zu_trd + zu_frc ) )*ssumask` | :1587 `(U_bar_c + dt_s*(_cor_u + _drag_u + _pgf_u + F_slow_u_i))*u_mask` | MATCH in association given `_cor_u`/`_drag_u` are exactly 0 at substep 1 |
| N14 | :778 `hu_e = hu_0 + zsshu_a`, `hur_e = ssumask/(hu_e + 1 - ssumask)` | next substep's `_ssh_avg_face_depths(eta_c)` | MATCH except on substep 1, where NEMO's `hu_e` is the qco `hu(Kbb) = hu_0*(1+r3u)` rather than the additive average. Degenerate from rest |
| N15 | :830 swap `ubb_e<-ub_e<-un_e<-ua_e`, `sshbb_e<-sshb_e<-sshn_e<-ssha_e` | the `ab3_za is not None` carry rotation | MATCH |
| N16 | :862 `puu_b(Kaa) += wgtbtp1(jn)*ua_e`, :874 `pssh(Kaa) += wgtbtp1(jn)*ssha_e`, RAW weights | `U_sum_c + w_i*U_bar_new`, `eta_sum_c + w_i*eta_new` | **DIFF**: `w_i` is the PRE-NORMALISED weight (measured `sum(w_filter) = 1.0000000000000002`), NEMO carries raw `wgtbtp1 = 1` and divides once. Same defect the transport side already fixed in round 58, never applied to the primary |
| N17 | :884 `puu_b(Kaa) /= SUM(wgtbtp1)`, `pssh(Kaa) /= SUM(wgtbtp1)` | `U_sum_f / w_total`, `eta_sum_f / w_total` | DIFF, same row as N16 |
| N18 | exit: vector form leaves `puu_b(Kaa)` a VELOCITY (:855-863, the `ln_dynadv_vec` branch never divides by depth) | `primary_transport_average=False` | MATCH |

`ts_wgt` for this step (`ll_av=T`, `ll_fw=T`, `nn_bt_flt=2`, `nn_e=23`, dynspg_ts.f90
CASE(2)): `jic = 23`, `wgtbtp1(jn)=1` for `|jn-23| < 23` i.e. `jn = 1..45`, so `Kpit =
icycle = 45` and the primary window is a UNIFORM boxcar. `wgtbtp2(jn) = 46-jn`,
`SUM(wgtbtp2) = 1035`. legoESM's window shape agrees; only the normalisation timing
(N16/N17) differs.

## Preregistration

**First substep predicted to differ: substep 1, statement N13** — the velocity update —
because from rest every other operand at `jn=1` is identically zero (`ssh_frc == 0` makes
`ssha_e(1) == 0`, hence `zu_spg == 0`; the rest seed makes `dyn_cor_2D` and the bottom
stress 0), so `ua_e(1) = rDt_e * zu_frc * ssumask` and the whole substep reduces to the
frozen forcing. Sea level at substep 1 is predicted to be EXACTLY zero on both sides.

**Predicted size**: unknown before measuring `zu_frc`; the falsifier is that
`eta_exit` at substep 1 is nonzero on either side, or that the two velocities agree to
0 cells (which would exonerate the forcing and move the owner into the substep
recurrence).

**How the prediction resolved.** Sea level at substep 1 is exactly zero on both sides —
0 cells unequal, as predicted. The velocity is NOT the same, and the reason is upstream
of every row in this table: the frozen forcing is divided by a water column that is not
NEMO's. See `spg_kt1_barotropic_ladder.py` section 2, rung R0.

## The owner, and what the evidence can and cannot say

`R0` is upstream of every rung in the table: `hu_0`, `zhup2_e`, the depth-weighted mean
that builds `zu_frc`, and the EEN Coriolis coefficients all stand on the layer
thicknesses. legoESM runs on `e3t_1d`; NEMO runs on `e3t_0`; they differ by up to 70.4 m
per level on 94134 of 342134 wet cells and by up to 104.2 m (4.0%) in the u-face column
total on 2503 of 9758 wet faces.

What this does NOT establish is that the barotropic loop's own statements are innocent.
A reviewer's objection, accepted: the coordinate gap and any forcing gap are not
independent, because the depth-weighted mean that builds `zu_frc` uses those very
thicknesses. "The coordinate, not the loop" is a distinction this evidence cannot draw.
The discriminator is the one the record cannot currently supply — drive legoESM's 45
substeps with NEMO's own dumped `zu_frc`/`zv_frc`/`ssh_frc` and compare the exit — and it
is blocked on the rank-tagged re-run.

## Retractions (Rule 11)

**1. The vertical-coordinate mechanism was wrong; the numbers were right.** This
document first said NEMO's DINO stretches its ladder to each column's own bathymetry,
citing `ocean.output:402` ("zgr_lib: zgr_sco : define full vertical s-coord. system using
the 2d bathymetry field") and `ln_hpg_sco = T`. An independent review refuted it and the
measurement agrees: `ln_zco_nam = .true.`, `zgr_sco_mi96` is called on `zflat(:,:) =
zHmax = 4000` m (`usrdef_zgr.F90:107-118`), `mesh_mask` carries `VertCoord = 'zco'`, and
the column-to-column spread of `e3t_0` over the 342134 wet cells is **exactly 0.0 at
every level**. The log line is printed inside `zgr_sco_mi96`, which both branches call,
and `ln_hpg_sco` names the pressure-gradient scheme, not the coordinate. Prose is a
pointer, never a citable fact — and this is the second time in this campaign that rule
has been paid for. The correct statement: `e3t_0` is horizontally uniform and equals
`e3t_1d` for k = 1..25, then diverges (ratio 0.979, 0.945, 0.924, 0.915, 0.919, 0.935,
0.965, 1.009, 1.070, 1.148 at k = 26..35). legoESM's `dz_ref` reproduces `e3t_1d` to
4.5e-13 — it is faithful to the wrong one of NEMO's two ladders.

**2. A near-retraction that was itself wrong.** The reconstruction of NEMO's `zu_frc` as
`depth_mean(utrd_hpg) + wind` was almost withdrawn on finding `utrd_hpg` identically
zero, which looked like a dead diagnostic bucket. It is not: DINO's initial temperature
and salinity are functions of depth and LATITUDE only (`usrdef_istate.F90:173-174`), so
the zonal pressure gradient really is exactly zero at kt=1, while `vtrd_hpg` is 2.49e-06.
Recorded because the reverse error — publishing a number built on a zero-filled bucket —
is one this campaign has made before.
