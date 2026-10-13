# PREREGISTRATION — TSUNAMI lane, round 1 (survey + card + acquisition)

Date 2026-10-08. Lane tip at start `0575754c061c`. Frozen before any NEMO
record exists; nothing below has been measured. The predictions are scored
next round against the record `run.sh --run` produces.

## 0. Premise correction (read first)

The round brief says TSUNAMI isolates ORCA2's split-explicit external mode.
**It does not run ORCA2's program.** `cpp_TSUNAMI.fcm` compiles
`key_qco key_xios key_vco_1d` and no `key_RK3`, so NEMO's main loop calls
the leapfrog step `stp_MLF`, not `stp_RK3`. Every other campaign case
(GYRE, ORCA2, VORTEX, LOCK_EXCHANGE, OVERFLOW) compiles `key_RK3`.
TSUNAMI's own `MY_SRC/stpmlf.F90` then reduces that step to the external
mode alone. The barotropic sub-step loop is shared code between the two
programs; the window set-up before it and the closing statements after it
are not (they sit in opposite arms of `#if defined key_RK3` in
`dynspg_ts.F90`). See section 6.

## 1. Survey: resolved switches (namelist_cfg over namelist_ref, cpp keys)

| switch | resolved | source | does TSUNAMI's step execute it? |
|---|---|---|---|
| cpp keys | key_qco, key_xios, key_vco_1d; **no key_RK3** | cpp_TSUNAMI.fcm | selects stp_MLF |
| domain | 201 x 201 x 2 (one wet level) | usrdef_nam kpi/kpj/kpk | yes |
| rn_dx, rn_dy | 10 km | namelist_cfg | yes |
| ln_Iperio, ln_Jperio | .true., .true. | namelist_cfg | yes |
| nn_fcase, rn_ppgphi0 | 0 (f-plane), 38.5 deg | namelist_cfg | yes |
| rn_0xratio, rn_0yratio | 0.2, 0.4 (bump at global i=40, j=80) | namelist_cfg | yes (initial ssh) |
| rn_Dt, nn_itend | 1000 s, 100 | namelist_cfg | yes |
| ln_dynspg_ts | .true. | namelist_cfg | yes: the whole program |
| ln_bt_fw, nn_bt_flt, rn_bt_alpha | .true., 1 (boxcar), 0 | namelist_ref | yes |
| ln_bt_auto, rn_bt_cmax | .true., 0.8 -> nn_e = 6 | namelist_ref; dynspg_ts auto rule | yes |
| rn_atfp | 0.1 | namelist_ref | yes, only in the un_adv filter (dead for ssh/uu_b) |
| ln_dynvor_een, nn_e3f_typ | .true., 0 | namelist_cfg / namelist_ref | yes, only in the barotropic Coriolis |
| ln_dynadv_OFF, ln_traadv_OFF, ln_traldf_OFF, ln_dynldf_OFF | all .true. | namelist_cfg | routines never called |
| ln_hpg_sco | .true. | namelist_cfg | never called |
| ln_zdfcst (rn_avm0, rn_avt0) | .true. (1.2e-4, 1.2e-5) | namelist_cfg / namelist_ref | zdf_phy never called |
| ln_zad_Aimp | .false. | namelist_ref | never called |
| rn_shlat | 0 | namelist_cfg | no coast exists |
| ln_drg_OFF | .true. | namelist_cfg | zero drag |
| ln_seos (+ all coefficients) | .true., namelist_ref coefficients | namelist_cfg / namelist_ref | density never evaluated |
| ln_usr sbc, nn_fsbc | .true., 1: all fluxes zero | usrdef_sbc | yes (zeros) |
| ln_rstart | .false. (start at rest, l_1st_euler) | namelist_ref | yes |
| nn_hls | 2 | namelist_ref | no formula depends on it |

## 2. What the case changes in NEMO's step program

`diff -u` against `src/OCE`:

- **stpmlf.F90 (the program that runs).** Removes every physics call:
  no eos/bn2, no zdf_phy, no ldf, no ssh_nxt, no wzv, no dyn_adv/vor/ldf/hpg,
  no dyn_zdf, no dyn_atf, no ssh_atf, no tracer step, no restart write, no
  stp_ctl. What remains is: sbc; the "after" quasi-Eulerian metric from the
  ssh currently in the after slot; zero the 3-D trend; dyn_spg (the
  split-explicit external mode); dia_wri; rotate the three time levels;
  switch from Euler to leapfrog time step after kt = 1.
- **stprk3.F90.** Drops the `#if defined key_RK3` guard and the same physics.
  It COMPILES but is never called, because nemogcm calls stp_MLF without
  key_RK3.
- **diawri.F90.** Output only (adds the barotropic velocities to the
  IOIPSL files, drops RK3-only workspace). Writes no model state.

Consequences the card must reproduce, all read off the code (PLAUSIBLE until
measured):

1. The only prognostic state is ssh and the barotropic velocities. The 3-D
   velocity is never advanced (nothing writes its after slot).
2. **The metric lags the surface height.** The metric is rebuilt each step
   from the ssh that sits in the after slot BEFORE dyn_spg overwrites it,
   and the slots rotate, so the metric dyn_spg reads as "now" at step k was
   built from the ssh of step k-4 (from the initial ssh for k <= 4), while
   the "now" ssh is that of step k-1. ORCA2's RK3 program has no such lag.
3. Every step re-zeroes the barotropic history and re-runs the cold-start
   window (boxcar averaging, ll_init true every step).
4. The window average is a transport average divided by the after face depth
   (flux-form branch: momentum advection is OFF, so not vector form).

## 3. Geometry the card carries (proved from usrdef_zgr, depth_e3, domzgr)

The flat bottom gives uniform z levels, not sigma: usrdef_zgr declares a
z-coordinate, sets the bottom level to jpkm1 = 1 everywhere, and key_vco_1d
substitutes every reference thickness by the 1-D ladder. With rn_domszz = 100
and jpk = 2: zd = 100; gdepw_1d = (0, 100); gdept_1d = (50, 150); e3t_1d =
e3w_1d = (100, 100) exactly, unchanged by the e3 -> depth round trip. The
time-varying thicknesses are 100 * (1 + ssh/100) through key_qco. Horizontal
scale factors are 10 000 m everywhere. No land: tmask = 1 at level 1.

## 4. Named blockers (the card declares all five; its execution gate refuses)

| id | what is missing | NEMO citation | smallest card-selected addition |
|---|---|---|---|
| B1 | a step program that runs the MLF external mode ALONE (no 3-D update, no ssh filter) | stpmlf.F90:111-134, nemogcm.F90:186; dynspg_ts MLF arms | a new whole-step identity selected only by this card that calls the existing split-explicit window once per step on the carried ssh / uu_b / vv_b, with the MLF window set-up and closing statements |
| B2 | the lagged quasi-Eulerian metric | stpmlf.F90:118 + slot rotation :131-134; dynspg_ts.F90:486-490, :355 | the B1 identity carries the metric's own time-level slots instead of recomputing it from the current ssh |
| B3 | card-level j-periodicity | namelist_cfg ln_Jperio | legoESM's y-wrap is a process-global halo flag, not card data; a card field threaded to the barotropic operators |
| B4 | i-periodic seam on the NEMO-literal barotropic arms | namelist_cfg ln_Iperio | none expected (the lat-lon model is zonally periodic) but every certified card has a closed ring, so the EEN/continuity arms have never run across an open seam: measure, do not assume |
| B5 | a one-level column | usrdef_nam.F90:98 | the coordinate now builds behind an opt-in flag; no model step has run on one level |

Also missing in legoESM, but subsumed by B1 (TSUNAMI never calls the
routines): an OFF arm for momentum advection and for tracer advection.

## 5. Frozen predictions and falsifiers (scored next round)

Geometry identity gate (mesh_mask.nc against the card, interior 201 x 201):

- **G1** glamt, glamu, gphit, gphiv equal the card's kilometre arrays bit for
  bit. Falsifier: one unequal cell.
- **G2** e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f all exactly 10000.0.
- **G3** ff_t = ff_f = f0 = 9.078896742484248e-05 bit for bit (legoESM's NEMO
  omega and libm sin). Falsifier: a last-bit difference.
- **G4** e3t_0 = e3u_0 = e3v_0 = e3w_0 = 100.0 on both records; gdept_1d =
  (50, 150), gdepw_1d = (0, 100); tmask 1 at level 1 and 0 at level 2.
- **G5** the kt = 1 entry ssh equals the card's initial ssh bit for bit on all
  40 401 cells, 325 non-zero, maximum 0.1 at 0-based (j, i) = (79, 39).
  Falsifier: any unequal cell. Registered risk: VORTEX's Gaussian tail
  differed by 1-2 last bits between two exp implementations; a cosine tail
  could do the same.
- **G6** ocean.output resolves nn_e = 6 (run.sh refuses otherwise).

kt = 1 ladder (records only; the card cannot execute until B1-B5 close):

- **L1** at kt = 1 entry, the three ssh slots are identical (the initial
  bump), the uu_b and vv_b slots are zero, and the three r3t slots are
  identical. Falsifier: any slot differs.
- **L2** (the B2 claim) at kt = 2 entry, the metric in the "now" slot is
  still the initial ssh divided by 100, not step 1's ssh divided by 100. At
  kt = 5 it is step 1's. Falsifier: either equality fails.
- **L3** the kt = 1 sub-step record carries icycle = 8, primary weights 1 at
  sub-steps 4..8 and 0 elsewhere (sum 5), and rDt_e = 1000/6.
- **L4** un_adv is not time-filtered at kt = 1 and is filtered with
  rn_atfp = 0.1 from kt = 2 (dynspg_ts.F90:920). Falsifier: the kt = 1 value
  equals the filtered form.

## 6. ORCA2 pointer

ORCA2 rung 0 runs the RK3 program (cpp_ORCA2 compiles key_RK3) with
nn_bt_flt = 3. TSUNAMI runs the MLF program with nn_bt_flt = 1. What
transfers: the sub-step loop body (continuity, ssh extrapolation, pressure
gradient, EEN barotropic Coriolis, transport accumulation) is the same code
in both. What does not: the window set-up and closing statements (opposite
cpp arms), the time filter, and the lagged metric (B2), which ORCA2 does not
have. A bit-exact TSUNAMI loop body is evidence for ORCA2's loop body only
for statements that do not read the filter weights. A key_RK3 build of this
same case would run the case's own stprk3.F90, i.e. ORCA2's program with
every physics term off; that is a configuration change and is asked, not
made (DECISION_NEEDED in the receipt).
