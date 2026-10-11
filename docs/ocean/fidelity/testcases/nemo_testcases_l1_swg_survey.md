# NEMO 5.0.2 `tests/SWG` (shallow-water gyre) — oracle survey for the legoESM card

Status 2026-10-11: oracle BUILT and RUN (operator); legoESM card NOT STARTED (lane
`fidelity/nemo-testcases-swg`, branched from the TSUNAMI lane tip 1acdf3fad).
Every line number below is from the shipped source under
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`; the compiled branch to cite in
receipts is `tests/SWG_OMIP_L1/BLD/ppsrc/nemo/`.  Everything marked VERIFY was
read off the source by the operator and must be re-read by the round that uses it.

## Why this case

SWG is NEMO's own single-layer, reduced-gravity, wind-driven square-basin gyre: the
closest shipped NEMO analogue of the barotropic Stommel–Munk configuration used for
online-learning studies (Maddison 2026, JAMES; Brolly 2026, PNAS).  It is the cheap
rung of the online-learning proof-of-concept: one layer, 80x80 wet cells at the
shipped resolution, ~2 s per 10 oracle steps.  The 2026-09-27 ladder survey marked
it BLOCKED for three reasons; this survey replaces that row.

## The oracle is NOT the primitive-equation engine

`tests/demo_cfgs.txt:13` builds SWG with components `OCE SWE`.  `src/SWE/` carries
its own `nemogcm.F90`, `stp_oce.F90`, `stpctl.F90`, `stpmlf.F90`, `stprk3.F90`.
The shipped keys (`tests/SWG/cpp_SWG.fcm`) are `key_xios key_qco key_vco_3d key_RK3`;
the oracle build drops `key_xios` (arch `arch-conda-scalarmath`, no XIOS), compiled
keys `key_qco key_vco_3d key_RK3` (`tests/SWG_OMIP_L1/cpp_SWG_OMIP_L1.fcm`).

The SWE `nemogcm.F90` init sequence (grep of `CALL` lines) is: nemo_init -> domain_cfg,
usr_def_nam, dom_init, istate_init, sbc_init, ldf_dyn_init, dyn_adv_init, dyn_vor_init,
dyn_ldf_init.  There is NO `dyn_spg_init` (so `dynspg.F90:238-239`'s "dynspg_exp not
implemented with RK3" stop never fires — the shipped deck integrates under RK3, measured
2026-10-11), NO `zdf_phy_init` and NO tracer step: `&namzdf`/`&namzdf_tke` entries in the
deck are inert, `&namdrg ln_OFF=.true.` and the "layer drag" is `rn_rfr` inside the
stepper.  `usrdef_istate.F90:59-61` starts from rest with `pts = 0` ("not used in SWE");
`usr_def_istate_ssh` (`:69-87`) sets `pssh = 0`.

## The step program: `src/SWE/stprk3.F90` (RK3, three stages)

Stage lengths: `rDt = rn_Dt/3` (`:112`), `rn_Dt/2` (`:178`), `rn_Dt` (`:244`).
Per stage, in compiled order:

1. `uu(:,:,:,Nrhs) = 0; vv(:,:,:,Nrhs) = 0`
2. `CALL dyn_adv( kstp, Nbb, K, uu, vv, Nrhs )` — K = Nbb at stage 1 (`:120`), Nnn at
   stages 2-3 (`:185`, `:252`).  Vector form (`ln_dynadv_vec=T`, `np_VEC_c2`,
   `dynadv.F90:79-82`): `dyn_keg` with `nn_dynkeg=0` (C2 KE gradient) then `dyn_zad`
   With one wet level vertical advection is identically zero: compiled
   `dynzad.f90:111-114` initializes the carried term to zero and its only
   interior loop has the empty bound `1:jpk-2`; `:141-146` subtracts that zero.
3. `CALL dyn_vor( kstp, K, uu, vv, Nrhs )` — `ln_dynvor_ens=T` (`np_ENS`,
   `dynvor.F90:72,135`), routine `vor_ens` at `dynvor.F90:539-669`, with
   `ln_dynvor_msk=T` (relative vorticity multiplied by fmask, `:67`).  Total vorticity
   (planetary + relative) in one call.  **Round-1 correction:** `ncor` was the
   wrong predicted branch.  The five-argument calls at compiled
   `stprk3.f90:149,204,261` dispatch to `dyn_vor_3D`; its executed ENS arm calls
   `vor_ens(...,ntot,...)` at compiled `dynvor.f90:250-251`.  The `ncor` arm at
   `:201-228` belongs only to split trend diagnostics, not the SWE step.
4. `CALL dyn_ldf(...)` — ONLY at stage 3 (`:254`); at stages 1-2 it is inside
   `#if defined key_RK3all` (`:122-124`, `:187-189`) and `key_RK3all` is NOT in the
   build.  Laplacian, iso-level, constant `ahm = 1/2 * rn_Uv * rn_Lv = 500 m2/s`
   (`nn_ahm_ijk_t=0`, `rn_Uv=2.0`, `rn_Lv=500` "at 1/4deg"; ocean.output:512-524).
5. Pressure gradient, inline (`:128-129`, `:194-195`, `:259-260`):
   `zrhs_u = -grav*(ssh(ji+1,jj,K) - ssh(ji,jj,K))*r1_e1u`, same for v with `r1_e2v`,
   where `grav = rn_modified_grav = 0.02 m/s2` (`usrdef_nam.F90:109`; ocean.output:131).
6. Wind and layer drag — ONLY at stage 3 (`:262-265`; stages 1-2 again under
   `key_RK3all`): `+ z1_2rho0*(utau_b + utauU)/e3u(Nnn) - rn_rfr*uu(Nbb)` (note: the
   drag uses the BEFORE velocity `Nbb` even at stage 3, and the wind is the mean of the
   before and current stress; for the steady SWG wind `utau_b == utauU` once
   `sbcmod.F90:570` has copied it at nit000).  `rn_rfr = 1e-7 1/s`.
7. `uu(Nrhs) += zrhs_u; vv(Nrhs) += zrhs_v` (`:138-139`).
8. `CALL ssh_nxt( kstp, Nbb, K, ssh, Naa )` (`:144`, `:210`, `:273`) —
   `sshwzv.F90:75-147`: `div_hor(kt, Kbb, Kmm)` then
   `ssh(Kaa) = (ssh(Kbb) - rDt*(r1_rho0*emp + SUM_k e3t(Kmm)*hdiv))*ssmask` (`:123`);
   Compiled `usrdef_sbc.f90:84-87` sets `emp`, `sfx`, `qns`, and `qsr` to zero;
   compiled `sshwzv.f90:145-147` contains the `emp+zhdiv` update (the alternate
   surveyed branch is absent after preprocessing).
9. `CALL dom_qco_r3c( ssh(Naa), r3t, r3u, r3v, r3f )` (`:146`, `:212`, `:275`) — the
   quasi-Eulerian thickness ratios for the after state (`key_qco`, `key_vco_3d`).
10. Velocity update, vector form (`:150-153` and the stage-2/3 copies):
    `uu(Naa) = uu(Nbb) + rDt*uu(Nrhs)*umask` (always from `Nbb`, the step entry).
11. `lbc_lnk` on `uu/vv(Naa)` (sign -1) and `r3u/r3v(Naa)` (`:167-168`), then the time
    level swap.

End of step: `dia_wri`, restart write (`:311-313`), `stp_ctl`.

## Geometry and forcing (`tests/SWG/MY_SRC`)

- `usrdef_nam.F90`: namelist `&namusr_def` (`:82-87`): `nn_SWG=4` (grid spacing
  `ze1 = rn_dx/nn_SWG = 25 km`, `:101`), `jpkglo=2` (one wet level), `rn_theta=0`
  (rotation), `nn_gc=2` ghost cells, `rn_domsiz=2000 km`, `rn_dx=100 km`,
  `rn_tau=0.20 N/m2`, `rn_modified_grav=0.02`, `rn_rfr=1e-7`, `rn_beta=2e-11`,
  `rn_f0=0.7e-4`, `nn_dynldf_lap_typ=1`, `rn_abp=1e-2`, `nn_cnp=1`, `rn_fsp=0`.
  `rn_abp/nn_cnp/rn_fsp/r1_abp` (penalisation) are read (`:117`) but NOT consumed
  anywhere under `src/OCE` or `src/SWE`.  Round 1's compiled-tree census found
  only their declarations and namelist read at `usrdef_nam.f90:56-60,143`; they
  are inert in this build.
- `usrdef_hgr.F90:80-146`: Cartesian coordinates in METRES, `plamt = zim05*ze1*cos -
  zjm05*ze1*sin - zlam0` etc. (`:119-135`), uniform `e1 = e2 = ze1` (`:142-145`).
  Coriolis `pff_f = rn_f0 + rn_beta*ABS(pphif)`, `pff_t` likewise (`:154-157`) —
  note the ABS.  Domain origin `zlam0/zphi0` rounded to the grid (`:100-105`).
- `usrdef_zgr.F90`: 1-D ladder `pdepw_1d = (0, 500)`, `pdept_1d = (250, 750)`
  (`:110-114`) -> ONE 500 m layer; flat bottom `k_bot = jpkm1` (`:191`); wet where
  `10 km < gphiv < 2010 km` and `10 km < glamu < 2010 km` (`:194-205`), plus the rule
  that a cell with <= 1 wet 4-neighbour becomes land (`:206-213`); zco scale factors
  (`zgr_zco`).  Result: 80x80 wet T cells inside the written 84x84 data domain
  (nn_hls=2).  Round 1 clarified that the computed domain is 88x88 and the
  NetCDF data domain removes two MPP halo rows per side: written T coordinates
  begin at -12.5 km, and the wet block occupies zero-based rows/columns 1..80.
- `usrdef_sbc.F90:84-106`: `utau = -rn_tau*cos(theta) * COS(pi*gphit/2000 km)`,
  `vtau = +rn_tau*sin(theta) * COS(...)` -> with `rn_theta=0`: zonal single-gyre wind,
  `-0.2` at the south wall, `+0.2` at the north wall, curl single-signed.  `taum = |tau|`.
  Compiled `usrdef_sbc.f90:84-87` verifies `emp=qns=sfx=qsr=0` (restart carries
  before-level copies of the first three).
- `usrdef_fmask.F90`: ORCA-only branches; for SWG the f-mask is the default free-slip
  one (`rn_shlat = 0`, ocean.output:284).
- `&namrun`: `nn_itend=172800`, `rn_Dt=3600 s` (the RK3 deck; the "10 ans - 30 min"
  comment is stale: 172800 x 3600 s = 19.7 years).  `nn_write=480`.

## Oracle records on disk (operator probe, 2026-10-11, NOT a committed acquisition)

`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round0_probe/kt10/`:
the shipped deck with `nn_itend=10`, `nn_stock=1`, `nn_write=1`, `ln_meshmask=.true.`,
2 s wall.  Files: `mesh_mask.nc`, `output.init.nc`, `SWG_000000NN_restart.nc`
(NN = 01..10; each holds the AFTER state of step NN: `sshn`, `un`, `vn` (nav_lev=2),
`uu_n`, `vv_n`, `tn`, `sn`, `utau_b`, `vtau_b`, `qns_b`, `emp_b`, `sfx_b`, `rdt`, `kt`,
84x84), hourly `grid_T/U/V/W` output, `ocean.output` (the resolved namelist echo),
`manifest.sha256` (binary + namelists).  `.../round0_probe/year10/`: the shipped
172800-step run (10-day output), for statistics.
Reference binary: `tests/SWG_OMIP_L1/BLD/bin/nemo.exe` (keys above).  A proper
STAGE-ENTRY record (the TSUNAMI pattern: additive `MY_SRC` writer on `src/SWE/stprk3.F90`,
`tsunami_r1_step_record.F90` format) is round 1's acquisition.

## What the legoESM lane lacks for this card (read off the TSUNAMI-lane tip)

1. `vorticity_scheme` on the lat-lon C-grid lane accepts only `al81`, `een_total`,
   `een_planetary`, `ene`, `ene_total` (`ocean_pe_latlon_cgrid.py:2625`); NEMO's
   `vor_ens` (`dynvor.F90:539-669`) with `ln_dynvor_msk` is not transcribed.
2. No per-stage single-mode explicit free surface: the lane's RK3 programs run the
   split-explicit external mode (`dynspg_ts`); the SWE program advances `ssh` with
   `ssh_nxt` inside every stage and takes the pressure gradient inline from the stage's
   `ssh`.  This is a NEW step program (`swe_rk3`), not a switch on the existing one.
3. Stage-3-only forcing: Laplacian viscosity, the `(utau_b+utauU)/2` wind, and the
   `rn_rfr` linear drag on the BEFORE velocity.
4. Reduced gravity `g = 0.02`: cards carry `constants=NEMO_CONSTANTS_CONFIG`
   (`nemo_recipe.py:58`); the card must `_replace(g=0.02)` and prove it reaches every
   consumer (pressure gradient, any CFL/diagnostic that reads g).
5. Beta-plane with `|y|` and grid rotation: `create_beta_plane_cgrid_geometry` exists
   (MITgcm barotropic card, VORTEX); `ABS(pphi)` and `rn_theta` must be transcribed,
   not assumed.
6. One-level `key_vco_3d` + qco: thickness `e3 = e3_0*(1+r3)` after every stage's
   `ssh_nxt`; TSUNAMI's card already runs one level under `key_qco`/`key_vco_1d`.

Template: `build_tsunami_zco_card` / `TsunamiResolvedNamelist` /
`TSUNAMI_DEVIATIONS` / `TSUNAMI_UNMEASURED` in `nemo_testcase_recipe.py:2894-3200`,
`nemo_testcase_l1_tsunami_ladder.py`, `nemo_testcase_l1_tsunami/{run_rk3.sh,
check_records.py, tsunami_r1_step_record.F90, stprk3_rk3_step_record.patch}`,
tests `test_nemo_tsunami_card.py`, `test_nemo_tsunami_ladder.py`.

## Deviations expected (document, never hide)

- `key_xios` dropped (toolchain), as for every oracle build.
- None else intended: the deck is the shipped `EXPREF/namelist_cfg`; the 10-step
  acquisition changes only `nn_itend`, `nn_stock`, `nn_write`, `ln_meshmask`.
