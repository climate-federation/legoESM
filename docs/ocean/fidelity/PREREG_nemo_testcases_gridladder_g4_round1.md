# Grid ladder rung G4 — round 1 preregistration (gridladder_r17)

Base: 748d98775 (rungs G1, G2, G3 certified; G3 on its as-built initial
state, Decision 120 pending). No NEMO record of G4 exists; nothing here was
measured against NEMO. Evidence directory:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round17/.
"ORCA2" = `cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg`; "ref" =
`cfgs/SHARED/namelist_ref`; NEMO citations are to the record binary's
compiled sources, `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/<file>.f90:lines`.

## 1. The rung, and the operator's picks inside Decision 117

G4 = the certified G3 program with ONE switch, ORCA2's surface boundary
condition, and NO sea ice. It is NEMO's OMT-0 minus ice: G4 + ice is the
rung that meets the ORCA2 lane's OMT-0 from below.

Operator picks (the user may redirect): every forcing value exactly as
ORCA2 resolves it (table below), with `nn_ice = 0`; the same forcing files,
weights, start date (nn_date0 = 010101, ref) and time interpolation;
G3's as-built initial state (no T/S damping; Decision 120 pending);
everything else identical to the certified G3 deck.

| group.key | G3 | G4 | source |
|---|---|---|---|
| namsbc.nn_fsbc | 2 | 2 | ORCA2:80 |
| namsbc.ln_usr | T (zero fluxes) | F | ref:184 |
| namsbc.ln_blk | F | **T** | ORCA2:83 |
| namsbc.nn_ice | 0 | **0** (ORCA2 says 2) | ORCA2:85; ice is the next rung |
| namsbc.ln_traqsr | F | **T** | ORCA2:89 |
| namsbc.ln_ssr | F | **T** | ORCA2:90 |
| namsbc.ln_rnf | F | **T** | ORCA2:91 |
| namsbc.nn_fwb | 0 | **2** | ORCA2:92 |
| namsbc.ln_dm2dc, ln_apr_dyn, ln_wave, ln_flx, ln_abl, ln_cpl, ln_ice_embd, nn_lsm | F/0 | F/0 | ref:185-213 (ORCA2:94 for ln_wave) |
| namsbc_blk (all) | unread | ORCA2:97-115 verbatim: ln_NCAR; sn_wndi/sn_wndj (6 h, bicubic weights, Uwnd/Vwnd rotation), sn_qsr/sn_qlw (24 h), sn_tair/sn_humi/sn_slp (6 h), sn_prec/sn_snow (monthly), bilinear weights, no time interpolation, climatological | ORCA2:100-114 |
| namsbc_blk (unstated) | — | ref: rn_zqt = rn_zu = 10 m, nn_iter_algo = 5, no skin, rn_pfac = rn_efac = 1, ln_crt_fbk F, ln_humi_sph T, ln_tair_pot F | ref:239-256 |
| namtra_qsr | unread | ORCA2:133-149: ln_qsr_rgb, nn_chldta = 1, nn_chlprfl = 1, sn_chl monthly interpolated; ref rn_abs = 0.58, rn_si0 = 0.35 | ORCA2:136-148, ref:429-430 |
| namsbc_ssr | unread | ORCA2:151-157: nn_sssr = 2, rn_deds = −166.67 mm/day, ln_sssr_bnd T, rn_sssr_bnd = 4; ref nn_sstr = 0, nn_sssr_ice = 1, sn_sss = 'sss_data' monthly interpolated | ORCA2:153-156, ref:448-465 |
| namsbc_rnf | unread | ORCA2:159-175: ln_rnf_mouth T, rn_hrnf = 15 m, rn_avt_rnf = 1e-3, rn_rfact = 1, sn_rnf monthly interpolated, sn_cnf once; ref ln_rnf_depth/tem/sal/icb F | ORCA2:161-174, ref:474-477 |
| namsbc_apr | unread | unread (ln_apr_dyn = F in ORCA2 and ref) | ref:211 |
| namsbc_fwb | unread | ref: rn_fwb0 = 0, nn_fwb_voltype = 1 → forced to 2 without ice | ref namsbc_fwb; sbcfwb.f90:156 |

Files NEMO opens (copied into each run directory, SHA-256 in run.sh):
the seven CORE-II normal-year files (u_10, v_10, ncar_rad, t_10, q_10,
ncar_precip, slp `.15JUNE2009_fill.nc`), `weights_core2_orca2_bicub.nc`,
`weights_core2_orca2_bilin.nc`, `chlorophyll.nc`, `sss_data.nc`,
`runoff_core_monthly.nc` (flux and mouth mask, sbcrnf.f90:356, :575-576),
plus G3's domain and T/S files. ORCA2's `runoffs` file (sn_s_rnf,
sn_t_rnf, sn_dep_rnf) is never opened: its three switches are off
(sbcrnf.f90:376, :388, :400); it is absent from the input set.

## 2. Ice-free consequence

With `nn_ice = 0` the bulk formulae run over open ocean everywhere,
including where ORCA2's ice initialisation would put ice (SST within 2 K of
freezing): 3,456 of the domain file's 16,433 wet surface cells (1,279 north,
2,177 south; card arithmetic on G3's initial state, not a NEMO number).
Nothing clips SST at freezing (15 surface cells start below it), there is
no ice-ocean stress, no brine or melt flux, and SSS restoring applies
everywhere (nn_sssr_ice is moot).

## 3. Deck switches that changed besides the forcing family

**None.** Outside &namsbc, &namsbc_blk, &namtra_qsr, &namsbc_ssr and
&namsbc_rnf, the G4 deck's assignments equal G3's (committed test; run.sh
refuses otherwise). Not moved, stated so nobody reads them as forcing:
`ln_tsd_dmp`/`ln_tradmp` stay F (Decision 120; ORCA2 has both T),
`ln_trabbc` stays F (ORCA2's geothermal flux is a bottom condition, not
surface forcing), `nn_ice` stays 0. The reference run's restart list is
rendered as 2,4,..,20 (run-length line only) so it holds the restart at
whatever record length NEMO allows.

## 4. NEW code exercised (none of it has met a NEMO record on this ladder)

1. **fldread with weights and rotation**: bilinear/bicubic interpolation of
   the CORE fields onto ORCA2 and the U/V pair rotation (fldread.f90,
   geo2ocean.f90); time interpolation of the monthly chlorophyll, SSS and
   runoff.
2. **NCAR bulk formulae**: absolute 10 m wind module (sbcblk.f90:674-676,
   NOT relative to the surface current), q_sat at the SST (:694),
   `turb_ncar`'s stability iteration, nn_iter_algo = 5 (:706), air density,
   stress `utau = taum/wndm * U10` (:796-805), the turbulent and longwave
   heat fluxes and evaporation in blk_oce_2 (:856-944), qsr = (1−albo)·SW
   (:683-687).
3. **Application**: the wind stress to the momentum trend and the
   barotropic forcing; qns/qsr/emp to tracers and sea level (trasbc.f90,
   sbcmod.f90); sbc called at kt = 1, 3, 5, … (nn_fsbc = 2) and held
   between.
4. **Runoff**: surface volume flux into the level-1 divergence
   (ln_rnf_depth = F branch, sbcrnf.f90:279-283) and the
   river-mouth extra diffusivity `avt += 2·rn_avt_rnf·rnfmsk·wmask` for
   jk = 2..nkrnf (zdfphy.f90:353-355, nkrnf from sbcrnf.f90:512-515) — the
   first non-constant vertical diffusivity on this ladder.
5. **Penetrating shortwave**: RGB with chlorophyll data, analytic profile
   (traqsr.f90:1133 np_RGBc), heating below level 1.
6. **SSS restoring**: emp (and the matching qns) damping toward sss_data,
   bounded at 4 mm/day, switched off near river mouths by
   `coefice*(1-2*rnfmsk)` (sbcssr.f90:187-199; data read at :150).
7. **Freshwater budget nn_fwb = 2**: without a restart the correction is
   rn_fwb0 = 0 and stays 0 in year 1 (sbcfwb.f90:289-293, :302-318); the
   kt = 1 global sum runs and adds exactly zero (sbcfwb.f90:328-329).

## 5. The record this acquisition gives, and its limit

- Binaries: G1's round-4 pair, unchanged (SHA-256 pinned; no build).
  ORCA2_OMIP_L4 compiles `key_si3 key_qco key_vco_1d3d key_RK3`; the G1
  binaries the same minus key_si3, which nn_ice = 0 never reaches. The
  forcing sources carry no cpp key and no G1 MY_SRC file touches them
  (usrdef_sbc runs only under ln_usr = T).
- Runs: 2-step smoke (reference binary), the 100-step reference (NEMO
  stops it; its completed steps set the record length, capped at 10,
  even), then twins a (restart at the end), b (even restarts), c (odd
  restarts) with the record binary at that length. Admission is G1's
  checker plus G3's judgments, and NEMO's own log must state every G4
  switch.
- **Limit.** The frames hold no surface flux. NEMO's RK3 restarts hold the
  surface tracer fluxes `sbc_hc_b`/`sbc_sc_b` (trasbc.f90:175-176),
  `fraqsr_1lev` (traqsr.f90:246) and the mean surface fields
  (sbcssm.f90:185-192), but not the wind stress (sbcmod.f90:590 writes it
  under lk_SWE only). Stress is witnessed only through level-1 u, v.
  If the walk needs it, a forcing observer is a new record binary
  (next round's ask, not built here).

## 6. Predictions (frozen before any G4 record exists)

- G4-P1 (log). NEMO's log states ln_usr = F, ln_blk = T, nn_ice = 0,
  ln_traqsr = T, ln_ssr = T, nn_fwb = 2, ln_rnf = T, 31 levels, and the T/S
  read with damping off. Falsifier: any absent (run.sh refuses).
- G4-P2 (boundary). NEMO stops the 100-step run in stp_ctl at kt = 11 on
  |V| > 10 m/s in the deep interior, as G3 (forcing acts at the surface;
  G3's stop came from the pressure gradient at level 27). Falsifier: any
  other stop step or condition. PLAUSIBLE.
- G4-P3 (first non-bit row). The card cannot run forcing yet, so the first
  comparison is an offline replay of the bulk on NEMO's kt = 1 entry: the
  step-2 restart's `sbc_hc_b` (kt = 1 heat flux) is NOT bit-identical to
  legoESM's NCAR bulk evaluated on the same inputs. Falsifier: equal on
  every wet cell. PLAUSIBLE (libm transcendentals in turb_ncar's
  stability functions, q_sat, the interpolated inputs).
- G4-P4 (consequence). In any forced run, level-1 T, S, u, v and sea level
  are all unequal at kt = 1, the first unequal deep level is at most
  nkrnf (river-mouth mixing) or below (penetrating light). Falsifier: a
  bit-identical kt = 1. PLAUSIBLE.
- G4-P5. No direction predicted for certification.

## 7. The card

ORCA2G4-sbc: the G1/G2/G3 builder on the G4 deck root. Its model
configuration equals G3's field for field except the penetrating light,
which is ORCA2's own RGB selection (committed test); it names the forcing
`ncar_core_sbcblk` / `nemo_fld_read` like the ORCA2 card. The rung table
refuses a G4 deck whose forcing switches or data lines differ from ORCA2's,
and a G1-G3 deck that states any forcing data. Declared not
execution-ready: record not admitted, and the forcing family (bulk,
application, runoff + mouth mixing, RGB with chlorophyll, SSS restoring,
nn_fwb = 2) is listed as unmeasured.

## 8. Question for the user

SSS restoring (nn_sssr = 2, −166.67 mm/day) is a damping, like the T/S
damping of Decision 120. Built as ORCA2's deck has it (ON).

## Dated correction — 2026-10-15

Lines 14–15 above used the operator's incorrect description of the ORCA2
lane's OMT-0.  Decision-103 OMT-0 has no surface forcing and no sea ice; it is
closest to G3, not G5.  G4 and G5 instead climb the separate forcing/ice axis
toward the shipped ORCA2 program.  This correction does not change the frozen
G4 deck, predictions, or falsifiers.
