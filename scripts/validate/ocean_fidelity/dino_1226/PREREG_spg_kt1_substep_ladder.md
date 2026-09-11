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

---

## ROUND 42 — the rank-tagged record arrived, and the table above was scored

The re-run this document said it was blocked on exists
(`/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump`, 720 files =
16 ranks x 45 substeps). `spg_kt1_barotropic_ladder.py` now stitches it and
scores R1 and R3 substep by substep.

### The instrument was calibrated first, and it is not a formality

Two statements of NEMO's, rebuilt from NEMO's own dumped arrays with legoESM
absent, both at **0 cells unequal**:

| check | statement | result |
|---|---|---|
| C1 | the swap, `dynspg_ts.f90:830`/`:838` — substep n+1's ENTRY must be substep n's EXIT | 0 of 44 joins x 3 fields |
| C2 | `ts_bck_interp`, `:662-665` — `zsshp2_e = za0*ssha_e + za1*sshn_e + za2*sshb_e + za3*sshbb_e` | 0 of 45 substeps |

C2 is the one that matters. It fails if a tile is placed at the wrong offset,
if a header field is read in the wrong order, if a substep is mis-numbered, or
if the two `zsshp2_e` assignments are confused (`:574` writes the AB3 mid-step
eta and `:664` OVERWRITES it; the dump at `:926` sees only the second).
`--plant-reader` moves one cell of C2's own prediction by 1 ulp and the gate
then refuses to score anything below it.

**C2 also MEASURES `ll_init`.** The identity only closes if substeps 1 and 2
use the forward-backward and AB2-AM3 ramp rows, which is the `ll_init = .TRUE.`
branch of `ts_bck_interp`. It does close, at 0 cells.

### How the prediction resolved

The prereg predicted the first break at substep 1, statement N13, because from
rest the substep collapses to the frozen forcing. **That is what happened**, and
the entry rows at substep 1 are 0 cells, so legoESM's operator ran on NEMO's own
operands and the break is the operator's inputs, not inherited drift.

But the break is NOT in the loop. Both sides satisfy their own closed identity
at 0 cells — NEMO's `ua_e(1) = rDt_e*zu_frc` and legoESM's
`u_exit(1) = dt_s*slow_u` — so `slow_u/slow_v` IS `zu_frc/zv_frc`, and the
substep-1 residual is the forcing arriving different, multiplied by `rDt_e`:

| row | cells != | max\|d\| | NEMO's own rms | ratio |
|---|---|---|---|---|
| R1 `zu_frc` | 8568 | 6.617e-23 | 2.418e-08 | 2.7e-15 (~12 ulp) |
| R1 `zv_frc` | 9868 | 8.414e-11 | 1.414e-06 | **6.0e-05** |
| R3 substep 1 `v_exit` | 9868 | 9.877e-09 | — | = 117.391 x 8.414e-11 exactly |

**The barotropic loop's own statements are exonerated at substep 1.** The gap
is almost entirely MERIDIONAL, and at kt=1 from rest `utrd_hpg` is identically
zero while `vtrd_hpg` is 2.49e-06 — so the owner is the depth-weighted mean of
the baroclinic momentum RHS that builds `zv_frc`, upstream of `dyn_spg_ts`.
Localising it further needs a rank-tagged `dynhpg` dump; this record's
`hpg_dump_dv.bin` is written from `dynhpg.F90`, which the rank-tagging regex
does not reach, so it is still a 16-rank interleave. NOT MEASURED, named.

### Retractions (Rule 11)

**3. R0 is no longer the owner, and this document said it was.** The section
above ("The owner, and what the evidence can and cannot say") attributed the
substep-1 break to the layer thicknesses: `e3t_1d` vs `e3t_0`, "up to 70.4 m
per level on 94134 of 342134 wet cells". That gap is CLOSED — R0a, R0b and R0c
all read 0 cells unequal on the current branch. The residual survived the fix,
so the thicknesses were never what it was made of.

**4. "the split-explicit solver's SECOND step is the named owner" is not
supported at kt=1.** The barotropic loop reproduces NEMO's arithmetic exactly
given NEMO's operands; what it is handed differs. Whether the kt=2 gap has a
second, independent owner inside the loop is a separate question, and round 42
found one that is NOT a history: see below.

### The kt=2 window — RETRACTED WITHIN THE ROUND (Rule 11)

**The table below was wrong and the finding it carried is withdrawn.** The
numbers as first published said legoESM ran 91 barotropic substeps at kt=2
against NEMO's 68, and called that "this round's one open decision". It was a
bug in the INSTRUMENT, not in the model: `kt2_leapfrog_gate.py` re-derived the
loop with `_compute_weights(..., substep_scale=<default 1>)`, having read
`substep_scale` from `mc.barotropic.barotropic_substep_scale`, a field that
exists nowhere in the package, so the getattr silently returned 1. The
leap-frog path passes **2** (`ocean_model_latlon_cgrid.py:10474`), and the
captured `n_substeps = 46` was itself the tell — the forward-Euler path passes
23. Both independent reviews found it, and the measurement with the real
value is:

| | substeps | primary boxcar nonzero on |
|---|---|---|
| NEMO kt=2 | 68 | 24..68 |
| legoESM kt=2 | **68** | **24..68** |

**AT BAR.** The same defective line was in this gate
(`spg_kt1_barotropic_ladder.py`), where it was right at kt=1 only because the
Euler path really does pass 1; both now take `substep_scale` from the CALL.

The correct reading of NEMO's own statements below still stands, and it is
worth keeping because it is what the kt=2 record's prediction P1 tests:

### The kt=2 window as NEMO builds it (the reading, which was never in doubt)

`ll_fw_start` is TRUE only at `kt == nit000` with the Euler start
(`dynspg_ts.f90:228-232`); at `kt == nit000 + 1` with `ln_bt_fw = .FALSE.` NEMO
RESETS it and calls `ts_wgt` again (`:245-250`), moving the boxcar centre from
`nn_e = 23` to `2*nn_e = 46`.

| | substeps | primary boxcar nonzero on | integrated span |
|---|---|---|---|
| NEMO kt=1 | 45 | 1..45 | 5283 s |
| NEMO kt=2 | **68** | **24..68** | 7983 s |
| legoESM kt=1 | 45 | 1..45 | 5283 s |
| legoESM kt=2 | 68 | 24..68 | 7983 s |

legoESM doubles `n_substeps` 23 -> 46 with the timestep and passes
`substep_scale = 2`, which halves the half-width; NEMO holds `nn_e` at 23 and
moves the centre. Different bookkeeping, same window.

STILL UNCALIBRATED, and named: the `ll_fw = .FALSE.` branch of the port has no
number in any record to check it against — `ll_spg_dump` is `kt == nit000`
only, so nothing states NEMO's kt=2 `icycle` out loud. Prediction P1 of
`nemo_dino_kt2_rankdump/run.sh` is exactly that measurement. Until it runs, a
port wrong in the same way legoESM is wrong would read AT BAR.

### Decision 33 (carrying NEMO's barotropic histories in NEMO's form)

**Zero-sized on this card, read from the oracle and measured.**
`ll_bt_av = .TRUE.` whenever `nn_bt_flt /= 3` (`dynspg_ts.f90:208-209`) and
`ll_init = ll_bt_av` (`:214`) — both OUTSIDE the `IF( kt == nit000 )` block, so
`:463-470` re-zeroes `sshb_e/sshbb_e/ub_e/ubb_e/vb_e/vbb_e` on EVERY step.
There is nothing to carry. legoESM already matches: the `nemo_boxcar_ab3`
branch sets `_ab3_hist = None` and `ramp=True` every step
(`barotropic_latlon_cgrid.py:2366-2374`). Measured at kt=1 by C2 above; the
kt=2 confirmation is prediction P2 of
`nemo_dino_kt2_rankdump/run.sh`, which is prepared and not run.
