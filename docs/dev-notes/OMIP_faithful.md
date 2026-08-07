# OMIP-Faithful: legoESM ocean vs NEMO ORCA1

**Goal.** Run reference NEMO ORCA1 (morays, COREv2 normal-year forcing) and drive
legoESM ocean on **all 5 grids** under the *same* forcing; iterate legoESM code
until integral/climatological diagnostics match NEMO. Review every code change
with `/codex:adversarial-review`. Track here; shrink log every 10 iters.

**Status (honest, current):** Full faithful-comparison pipeline BUILT + codex-clean —
NEMO ORCA1 reference (5yr run in progress, 33+ monthly grid_T records), identical CORE-II
6-hourly forcing, eORCA1 mesh + WOA18, multi-grid runner (`run_omip_core2.py`), scorer
(`compare_omip_nemo.py`); 3 real applicator bug fixes + proper in-step forcing integration.
**BLOCKER — legoESM global-ocean dycore is UNSTABLE under realistic CORE-II forcing/IC**
(WOA cold-start blows up day-1 on both grids; tripole rest-state runs physical ~90d then
velocity-spikes to ~15 m/s by day-110; latlon-bathy-with-regridded-bathy blows up day-10).
Ruled out ~12 fixes (PGF scheme, 10× viscosity, dt, nlev, IC-mask, forcing-ramp, nudge,
in-step forcing, grid). This is **fundamental dycore-stability development, not parameter
tuning** — the real barrier to "fully faithful". First end-to-end comparison
(tripole day-90 physical state vs NEMO month-3, heavily caveated: rest-state IC + 90d spinup)
launched to prove the pipeline + produce real numbers. **NOT faithful — and honestly,
reaching it needs sustained ocean-dycore work beyond config sweeps.** See iteration log.

### Reference + comparison facts (verified iter 1)
- **NEMO grid_T vars** (the comparison fields): `to`(3D T), `so`(3D S), `tos`(SST),
  `sos`(SSS), `zos`(SSH), `mldr10_1`/`somxl010`(MLD), `tob`/`sob`(bottom T/S),
  `heatc`/`saltc`. Single global file (one_file), dims y=331,x=360,z=75.
- **COREv2 NYF files** (NEMO INPUTS, T62 grid LAT=94×LON=192): u_10/v_10(m/s),
  t_10(K), q_10(kg/kg), slp — 6-hourly(1460); ncar_rad LWDN+SWDN(W/m²) daily(365);
  ncar_precip RAIN+SNOW monthly(12); runoff = Dai-Trenberth-Depoorter on eORCA1.
- **legoESM forcing gap:** `load_core2_nyf`→`core2_nyf/nyf.zarr` (schema: lon,lat,time_s,
  u10,v10,T_air,q_air,sw_down,lw_down,precip,runoff) used ONLY by regional
  `run_bryan_thc.py`. run_omip has NO core2 mode (only restoring / jra55_do_tropical).
  → DECISION: build `nyf.zarr` from NEMO COREv2 files, then either add a `core2_nyf`
  forcing-mode to run_omip (reuse LY09 bulk path) or write a global COREv2 runner
  reusing the bryan_thc bulk-flux step. Both models then share identical NYF forcing.

---

## Strategy (decision)

Reference acquisition, primary → fallback:

1. **Morays container (PRIMARY, confirmed viable).** `morays_env_amd64.sif`
   (1.68 GB; `singularity/3.7.1` == apptainer 1.4.1) bundles pre-built XIOS at
   `/home/jdoe/XIOS` + Debian netcdf + gfortran/mpif90. The ORCA1 `deploy.sh`
   (run INSIDE the container) downloads forcing + compiles via
   `makenemo -m ORCA1_GCC -j3`, then `srun ./nemo.exe`. ORCA1 = plain
   COREv2-forced ocean+SI3-ice run (NOT ML-coupled). "python call of Nemo" =
   morays Python deploy/postproc wrapper, not eophis (eophis only for the
   ML-coupled DINO case). The "Aronnax" package does not exist (WebFetch
   confabulation) — ignore.
2. **Native build (FALLBACK).** gcc11/13 + intel-oneAPI + openmpi + netcdf
   modules all present → build XIOS(-fPIC)+NEMO with a Ginsburg arch file.
   Higher risk (arch authoring, XIOS finicky).
3. **Literature reference values (BOOTSTRAP).** Encode published NEMO ORCA1
   benchmark scalars (ACC ~130 Sv, AMOC@26N ~15-20 Sv, global-mean T/S drift,
   SST/SSS RMSE vs WOA, MLD) as `reference_fn`s so legoESM scoring starts
   before live NEMO finishes. Replace with live-NEMO fields as they arrive.

**Synergy:** ORCA1 `mesh_mask.nc` (from deploy) = the `data/grids/eORCA1.2_mesh_mask.nc`
that legoESM's `tripole` grid needs (currently missing).

---

## Environment facts (verified iter 1)

- **Python:** no `.venv`. Use jn2808 conda env + pg2328 src on PYTHONPATH:
  `export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH`
  `JAX_PLATFORMS=cpu /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`
  (GPU: `JAX_PLATFORMS=cuda JAX_ENABLE_X64=1` inside SLURM only).
- **SLURM:** `--account=glab` required. Partitions: short(12h), glab1, burst.
  GPU g0xx = Quadro RTX 8000 46GiB. Walltime ceiling 72h.
- **HARD RULE:** login node = read-only/short only. ALL heavy work (downloads,
  builds, runs, pytest, JIT) via sbatch/srun. Login `git` OK.
- **Singularity** 3.7.1 (`module load singularity/3.7.1`). No Apptainer.
- **codex** CLI present → `/codex:adversarial-review` usable.
- **RESOLVED (probe 8089676, node g037):** compute nodes HAVE outbound internet
  (https + git + container-registry all work) ⇒ all downloads run in sbatch
  (login-policy compliant). Env import OK (jax 0.9.1). `singularity/3.7.1` ==
  apptainer 1.4.1; `docker://` pull+exec works on compute. **CAVEAT:** https
  confirmed; COREv2 uses **FTP** — reachability tested in STAGE job.

## Asset inventory (verified iter 1)

| Asset | State |
|---|---|
| NEMO / nemo.exe / ORCA1 | absent — must acquire |
| Morays container .sif | absent — wget from morays GH release |
| eORCA1 mesh_mask.nc | absent — comes from ORCA1 deploy (or zenodo) |
| COREv2 forcing | absent — ORCA1 deploy downloads it |
| WOA18 IC/targets | absent — `scripts/download_omip_data.sh` (public NCEI) |
| JRA55-do forcing | absent — download_omip_data.sh `--with-jra55` (~23GB) |
| legoESM `run_omip.py` | present, 5 grids, emits results.txt + mean_timeseries.csv |
| `ocean/forcing/core2.py` | present (COREv2 reader) |
| `ocean/fidelity/` harness | present; refs analytical + Veros → extend w/ NEMO |
| climate diagnostics | `diagnostics_cmip6.py`, `_streamfunction.py`, `_climate.py` |

---

## Grid ↔ NEMO mapping (5 legoESM grids)

| legoESM grid | run_omip res | vs NEMO ORCA1 | match expectation |
|---|---|---|---|
| **tripole/eORCA1** | eorca1 (1°) | SAME grid family | strictest — fields + integrals |
| latlon | 36x72 (5°) | regrid | integrals + large-scale patterns |
| mpas | ico3 (~900km) | regrid | integrals only (very coarse) |
| spectral | T21 (~5.6°) | regrid | integrals + large-scale |
| cubed_sphere | C24 (~5°) | regrid | integrals; KNOWN face-edge PGF instab |

## Metric set — defines "excellent match"

Point-by-point won't match across 1° NEMO vs ~5° coarse grids. Match defined on
**integral/climatological diagnostics** (tiered by stringency):

- **T0 conservation/sanity:** finite, mass/volume conserved, no blowup.
- **T1 integral transports:** ACC (Drake) ~130±30 Sv; AMOC@26.5N 15-20 Sv;
  global meridional heat transport sign/peak.
- **T2 climatology drift:** global-mean SST/SSS/3D-T drift vs NEMO over N yr.
- **T3 fields (regridded RMSE):** SST, SSS, SSH, MLD, MOC streamfn structure.
- **T4 tripole strict:** eORCA1 vs NEMO on native grid, tighter tolerances.

Pass thresholds per metric to be set in `ocean/fidelity/tolerances.py` (new NEMO tier).

**Faithfulness caveat (runoff):** `nyf.zarr` sets runoff=0, but NEMO applies the
Dai-Trenberth-Depoorter runoff (eORCA1 file). Until runoff is added to the legoESM
forcing path, freshwater-sensitive metrics — **SSS, MLD, AMOC/overturning, freshwater
budget** — are NOT faithful and must be reported as caveated (differences may be the
missing runoff, not the ocean core). Faithful NOW: SST/3D-T, wind-driven circulation
(ACC, gyres), heat transport. TODO: map Dai-Trenberth runoff (legoESM `ocean/forcing/
dai_trenberth.py` exists) onto the forcing → ungate the freshwater metrics.

---

## Pipeline

- **A** Resolve data-staging path (probe 8089676). 
- **B** Acquire NEMO: pull container → clone NEMO5.0.1/ORCA1/Aronnax → deploy (build+forcing) via sbatch in container.
- **C** Run NEMO ORCA1 (`srun ./nemo.exe`), post-process → reference climatology.
  Run config: `rn_Dt=3600s`, default `nn_itend=350400`=**40yr** (2000-2039),
  SI3 ice, RK3. ⚠️ default file_def enables 1h/2h/5d output → TB-scale; for
  climatology DISABLE high-freq, keep 1mo+1y. First reference: SHORT run
  (shorten nn_itend, e.g. 1-5 yr) to validate pipeline, then extend.
- **D** Stage eORCA1 mesh + COREv2/WOA forcing for legoESM.
- **E** Wire NEMO refs into `ocean/fidelity/` (references.py/registry.py/tolerances.py); add NEMO comparison cases.
- **F** Run legoESM `run_omip.py` all 5 grids w/ **identical COREv2 forcing** (sbatch GPU); regrid + score vs NEMO.
  Faithful-forcing: legoESM `core2.py::load_core2_nyf` reads `core2_nyf/nyf.zarr`
  (7ch: u10,v10,T_air,q_air,sw_down,lw_down,precip,runoff). Build that zarr FROM
  NEMO's downloaded COREv2 files (ncar_precip,ncar_rad→sw+lw,q_10,slp,t_10,u_10,v_10)
  so both models share the exact same NYF forcing. (synthetic fallback if absent.)
- **G** Iterate legoESM code to close gaps → `/codex:adversarial-review` each change → repeat F.

## Risk log

- **COREv2 forcing is FTP** (`data1.gfdl.noaa.gov`) — may be firewalled on
  compute even though https works (tested in STAGE). If blocked: https mirror,
  or pre-stage, or adapt deploy.sh.
- Singularity == apptainer 1.4.1 (modern) — .sif compatibility OK (verified docker://).
- XIOS/NEMO build fragility if native fallback needed.
- cubed_sphere face-edge PGF instability (documented) — may never match T3/T4.
- "Excellent match" across dynamically different cores is bounded; tripole is the
  honest strict test, coarse grids judged on integrals only.

## Iteration log

- **2026-08-07b (ARCTIC SPLIT INTO TWO DEFECTS; native-cell comparison unlocked):**
  New committed probe `scripts/validate/ocean_fidelity/arctic_deep_convection_columns.py`.

  **★ THE MESHES ARE IDENTICAL — STOP REGRIDDING TRIPOLE vs NEMO.** Our tripole runs
  on eORCA1 and so does the NEMO oracle. Verified offset `j0=0, i0=1` with
  **max|dlat| = 1.4e-14 deg** (the probe SEARCHES for the offset and refuses to run
  unless it verifies). Every tripole-vs-NEMO number can therefore be computed
  cell-for-cell with ZERO interpolation, removing the largest standing source of
  doubt. TRAP, cost one failed run: the first attempt reported an 84-degree
  mismatch, which was entirely NEMO grid_T's fill row (`nav_lat = -1.0` against a
  real -84.2) dominating a max reduction. Match on NEMO's OCEAN cells only.

  **★ INSTRUMENT FIRST: a deep MLD is often a SATURATED DIAGNOSTIC.**
  `diagnostics.mixed_layer_depth` ends `mld = where(has_crossing, mld_cross, bottom)`,
  so a column with no density crossing returns its bottom depth EXACTLY. Day 90 vs
  NEMO March, of 6225 Arctic (>=60N) columns:
  | | VSF control | real FW | NEMO |
  |---|---|---|---|
  | deeper than 500 m | 302 | 466 | 89 |
  | no crossing at all | 26 | **168** | - |
  | **genuine deep** | **271** | **291** | **84** |
  **RETRACTS the 2026-08-07a reading** that real freshwater roughly doubles Arctic
  deep convection. The genuine count moves 271->291 (+7%); the apparent +54% is
  the diagnostic saturating. The ~3.4x excess over NEMO is present in BOTH arms and
  is NOT caused by the freshwater closure.

  **★ THE ARCTIC IS TWO SEPARATE DEFECTS, ANTI-COLOCATED.** Genuine-deep columns vs
  the rest of the Arctic: median H_bathy 2152 m vs 286 m, ice fraction **0.19 vs
  0.87**, median dSSS vs NEMO **+0.08 vs +0.80**, all statically unstable vs 32%.
  So the salinity bias lives on ice-covered shelves and the convection defect lives
  in ice-free deep basins. By position the worst columns are the
  **Greenland/Norwegian Seas** (75-78N, 355-3E) and, in the real-FW arm, the
  **Labrador Sea** (60N, 300-302E), reaching **1800-2600 m where NEMO holds
  20-160 m**. Stop reporting these as one "Arctic problem".

  **LEVER (config gap CONFIRMED, causation PLAUSIBLE).** NEMO ORCA1
  `EXP00/namelist_cfg` sets `ln_mle = .true.` (namtra_mle) and `ln_zdfevd = .true.,
  rn_evd = 100`. Every arm scored in this campaign ran `MLE=off`, `convection=none`.
  We HAVE `mle_latlon_cgrid.py` + `--mle` (tripole-wired, threaded at
  run_omip_core2.py:875/888) and its `MLEConfig` defaults already mirror ORCA1
  (ce=0.06, lat_ref 20, rho_c 0.01, mld_uv "min"). MLE restratifies exactly these
  basins. A/B LAUNCHED (`_ab_mle_fwreal_d90.sbatch`, job 9337800): fwreal command
  verbatim + `--mle`, same pinned worktree, token-diffed against the control's
  manifest. Pre-registered: CONFIRMS if genuine-deep falls from 291 toward 84 and
  the Greenland/Labrador medians drop to O(100 m); REFUTES if it stays >250 or the
  columns merely move. Falling BELOW ~84 is over-restratification, also a failure.
  Watch the global and Antarctic MLD bands — trading an Arctic error for a Southern
  Ocean one is not a fix.

  **RETRACTION on the regrid control (codex r3).** 2026-08-07a said the
  common-footprint radius sweep REFUTED the IDW-asymmetry artifact. It does not: a
  common large-scale legoESM-vs-NEMO bias plus independent grid-scale errors on
  each grid gives the same table, and a post-filter cannot undo aliasing or a
  footprint bias surviving past 4 deg. Supported claim: **no differential advantage
  to tripole is detectable at any radius** — the grids were already equidistant
  from NEMO at radius 0 (SST 0.8518 vs 0.8566). Also from r3: the zonal support
  gate was a TAUTOLOGY (availability derived from the already-dropped-out mask);
  smoothing corrupted the MLD tail statistics (now suppressed under smoothing —
  no published number affected, the quoted tails are from unsmoothed runs and the
  native-cell probe). `near_land*` remain SCREENING figures: nearest-centre
  classification cannot resolve a strait narrower than a target cell.

- **2026-08-07 (THREE-WAY SCORECARD + freshwater A/B + a contaminated-mask retraction):**
  New committed instrument `scripts/validate/ocean_fidelity/compare_three_way_nemo.py`
  (+13 unit tests, `tests/validate/test_compare_three_way_nemo.py`; codex rounds 1-2).
  It puts tripole, MPAS and NEMO on ONE target grid and scores all three pairs on the
  cells resolved on all three, with ARCTIC (>=60N) and NEAR-LAND sub-domains, the
  near-land one split into its Arctic and non-Arctic parts.

  **★ RETRACTION — the pre-existing scorecards' ocean mask scores LAND.**
  `regrid_curv_to_latlon` returns a *distance-to-wet-data* flag (coverage=1 within
  `max_deg`=2.5 of any wet source cell), NOT a land/sea classification, so a target
  cell that is land keeps coverage=1 and is filled by extrapolation from offshore —
  preferentially at coastlines, i.e. exactly inside the near-land domain. Measured on
  the day-30 matched pair: **7133 of 49358 scored cells (14.4%) are land**. Effect
  (same runs, same protocol, only the classification changed, tripole SSS):
  global RMSE 0.817→**0.632**, near-land RMSE 1.621→**1.109**, near-land-Arctic bias
  +1.768→**+1.279**; SST near-land-Arctic bias -0.780→**-0.329** (RMSE 1.349→0.726).
  Roughly a third of the "coastal error" was the instrument. `--mask-mode nearest`
  (default) classifies a target wet iff the NEAREST cell of every source is wet;
  `coverage` reproduces the old behaviour and is kept only for that.
  **`compare_omip_nemo.py` still uses the coverage flag — every near-coast number in
  the entries above this one is contaminated and must be re-scored before reuse.**

  **★ THERE IS NO GENERIC COASTLINE DEFECT.** With the corrected mask the near-land
  split reads: non-Arctic coasts SSS bias **+0.245** / RMSE 0.944; Arctic coasts
  **+1.279** / 1.941. "Near-land" was the Arctic problem seen through a coastal mask.

  **Cross-grid agreement (matched pair, d30 vs NEMO Jan, corrected mask).** SST:
  trp -0.192/0.852, mpas -0.185/0.857, trp-mpas 0.245. SSS: trp +0.147/0.632,
  mpas +0.155/0.643, trp-mpas 0.162. **The two grids agree with each other 3.5-3.9x
  better than either agrees with NEMO** => the residual is SHARED physics, not
  discretisation; grid-specific work cannot close it.

  **Instrument control for that ratio — and a retracted first attempt.** I first
  re-ran at 2 deg and called the artifact refuted. WRONG, and codex round 2 caught it:
  `--res-deg` moves the target sample points but leaves the IDW regridder taking k=4
  source neighbours, whose PHYSICAL footprint still differs per mesh (~60 km MPAS ico7
  vs ~111 km ORCA1), so a 2-deg rerun can look stable whether or not the artifact
  exists. The real control (`--smooth-radius-deg`) imposes ONE great-circle top-hat on
  all three fields AFTER regridding. Global RMSE vs filter radius:
  | radius | trp-NEMO | mpas-NEMO | trp-mpas | ratio |
  |---|---|---|---|---|
  | none | SST 0.8518 / SSS 0.6319 | 0.8566 / 0.6433 | 0.2448 / 0.1616 | 3.49 / 3.95 |
  | 2 deg | 0.7993 / 0.5908 | 0.8013 / 0.6028 | 0.1751 / 0.1080 | 4.57 / 5.52 |
  | 4 deg | 0.7362 / 0.5383 | 0.7345 / 0.5544 | 0.1297 / 0.0661 | 5.67 / 8.26 |
  The artifact is REFUTED, on the prediction stated before the run: if the stencil
  flattered tripole-vs-NEMO, tripole would degrade RELATIVE to MPAS as the common
  filter widens. It does not — the two stay within 1-3% of each other at every radius
  (4 deg: 0.5383 vs 0.5544). Second, unforced result: the cross-grid difference is
  SMALL-SCALE (0.162 -> 0.066 under filtering) while the NEMO gap is LARGE-SCALE and
  survives => the shared residual is a systematic large-scale physics/forcing
  difference, not grid-scale noise.

  **★ FRESHWATER A/B — CONTROLLED (`nemolev_trp_gwcorr_d90` vs `nemolev_trp_fwreal_d90`,
  manifests differ by exactly `--freshwater-closure real_freshwater`), d90 vs NEMO m3.**
  | metric | VSF control | real freshwater |
  |---|---|---|
  | SST RMSE | 0.820 | 0.826 (neutral) |
  | SSS RMSE | **1.102 (poor)** | **0.816 (good)** |
  | SSS corr | 0.919 | 0.951 |
  | Arctic MLD bias / median | +70.9 m / **+21.8 m** | +129.7 m / **+14.5 m** |
  | Arctic columns deeper than 500 m | 3.72 % | 6.08 % (NEMO **0.80 %**) |

  The fix validates on salinity AND on the TYPICAL Arctic column (median MLD bias
  +21.8→+14.5 m). **The mean/RMSE MLD degradation is entirely a deep tail**: runaway
  columns 3.7%→6.1% against NEMO's 0.8%. A first reading of the mean alone said
  "fixes SSS, breaks MLD" — wrong, and it is exactly why `_tail` (median + deep
  fraction) is in the scorer. NOTE the control ALREADY has 4.6x NEMO's runaway count:
  **Arctic deep convection is a pre-existing defect that real-freshwater amplifies**,
  and it is now the largest single remaining term.

  **Best config per grid (d90 vs NEMO m3).** tripole+realFW beats MPAS on salinity
  everywhere (global SSS RMSE 0.523 vs 0.605; Arctic bias +0.904 vs +1.363); MPAS wins
  Arctic SST (0.757 vs 1.008) and Arctic MLD (RMSE 162 vs 464) because it has no
  real-FW closure and so gets neither the SSS benefit nor the convection amplification.

  **NULL RESULT — the TKE mixing-length choice does not move the Arctic.** d30 vs
  NEMO m1, four arms (`mxl4` / `mxl3ctl` / `nnmxl2` / `icemelt70`): SSS RMSE
  0.856 / 0.858 / 0.865 / 0.859, Arctic bias +0.77 / +0.78 / +0.78 / +0.78. ORCA1's
  actual `nn_mxl=2` is reachable now (`--tke-mxl-choice 4`) and is a null lever here.

  **CIRCULATION IS NOT SCOREABLE AT THIS WINDOW.** The only archived NEMO
  `grid_U`/`grid_V` are ANNUAL means (AMOC@26.5N 17.74 Sv, ACC@Drake 159.26 Sv); our
  arms are day-30/90 snapshots with AMOC ~0.0 Sv, which is what a 90-day spin-up from
  rest gives. Differing windows is a confound, not a result — a matched AMOC verdict
  needs multi-year arms or monthly NEMO grid_V output, neither of which exists.

  **SSH DEMOTED.** `zos`-vs-`eta` datum, inverse-barometer treatment and free-surface
  diagnostic are unreconciled; removing the area-weighted mean fixes a spatially
  uniform offset only. Now opt-in (`--ssh`), informational, never a verdict.

  Instrument defects caught by the gates rather than by review: `_mld_area` used before
  assignment; `rc=$?` inside an echo containing `$(basename ...)` reporting basename's
  status, so a crashed scorer logged rc=0; `_demean` returning all-NaN because
  `0.0 * nan` is `nan` (a zero AREA weight does not exclude a NaN CELL).

- **iter 1:** Oriented. No prior OMIP/NEMO work. Verified env/slurm/singularity/codex,
  asset inventory, container URI, grid mapping, real ORCA1 repo (deploy.sh: COREv2-ftp
  + Zenodo rec/14041098 inputs + GH ice/weights; `makenemo -m ORCA1_GCC -j3`; default
  namelist 40yr 2000-2039, SI3 ice, RK3). Wrote plan. Probe 8089676 GREEN (compute
  internet works). STAGE 8089685 GREEN in 39s: container 1.6G, NEMO 23M, ORCA1 cfg,
  **FTP reachable**, Zenodo inputs 1.34G, container XIOS `/home/jdoe/XIOS/libxios.a`.
  Submitted BUILD 8089688 (deploy.sh in container) + WOA 8089692. Found: legoESM
  core2 forcing reads a zarr cache → must build nyf.zarr from NEMO's COREv2 files
  for identical forcing (faithful comparison).

- **iter 1 (cont):** BUILD 8089688 GREEN in 4m48s — nemo.exe linked (XIOS+netcdf paths
  correct), INPUTS 6.6G, `domain_cfg.nc` + COREv2→eORCA1 weights staged. Run cfg:
  XIOS attached (`using_server=false`), no OASIS, auto MPI decomp (`jpni=jpnj=0`),
  OpenMPI 4.1.4 in container, nodes=32cores/172G (short/burst partitions). NOTE:
  deploy.sh forcing symlinks point to container path `/home/jdoe/to_host/...` →
  resolve only inside `--bind $WS:/home/jdoe/to_host`. Created RUN_SMOKE (nn_itend=768
  =1mo, ln_meshmask=.true., disabled 1ts/1h/2h/5d output). Submitted SMOKE 8089701.
- **iter 1 (code):** Extended `coupler/omip2_applicator.py` for **tripole** (NN-sample CORE-II
  onto 2D lat_T/lon_T + geographic→grid rotation via cos_alpha_u/sin_alpha_u; factored shared
  `_apply_cgrid_surface_fluxes` from the latlon body — no copy-paste). **FOUND + FIXED a
  wind-stress SIGN BUG**: `air_sea_fluxes` returns *atmospheric*-convention tau (−ρCd|U|U,
  opposing wind); the applicator applied **+tau** → ocean driven BACKWARDS. Fixed to ocean
  reaction **−tau** across latlon/tripole/cube/mpas (matches dynamics-core
  `ocean_pe_latlon_cgrid.py:1892`). Added sign-regression + tripole unit tests → **all 12 pass**
  (job 8089767). codex adversarial-review running (b9a6j470w).
- **iter 1 (codex round 1):** adversarial review = needs-attention, 3 findings, all valid + fixed/gated:
  (1) [high] converter collapsed 6-hourly→daily before the NONLINEAR bulk flux (flux(mean U)≠mean(flux)).
      FIXED: `nyf.zarr` now keeps native **6-hourly (1460 rec)** for u/v/T/q; rad/precip broadcast up.
      Evidence it mattered: |u10|max 24.31(daily)→27.99(6-hourly). (2) [high] `_nn_interp_to_points`
      used searchsorted insertion idx (not nearest) + no periodic lon. FIXED: true periodic NN, cached.
      (3) [med] runoff=0 vs NEMO Dai-Trenberth → GATED (see caveat below). Rebuilt zarr + 12 tests still pass.
- **iter 1 (codex round 2):** confirmation adversarial review = **APPROVE**, no new findings —
  6-hourly + true-NN fixes confirmed resolved, ocean-reaction sign (−tau) + tripole rotation
  correct, runoff gating accepted. applicator + converter **codex-clean** (iterate loop converged).
  Investigated tripole mask source: eORCA1 mesh HAS tmask/tmaskutil/mbathy/e3t_0/gdepw_0
  (332×362×75) → derive land_mask (tmaskutil) + H_bathy (mbathy/e3t_0) from the **mesh itself**
  (NEMO's own mask — more faithful than ETOPO). **NO ETOPO needed** for the tripole runner.
- **iter 1 (comparison + IC fix):** Wrote `scripts/compare_omip_nemo.py` (cKDTree-IDW regrid of
  BOTH curvilinear grids → common 1° lat-lon; area-weighted SST/SSS bias/RMSE/corr; maps +
  zonal plots + report.json). VALIDATED on smoke artifacts (runs clean). Test SST RMSE 9.3°C —
  but that's an **IC CONFOUND**: legoESM rest-state IC (idealised exp-T, uniform S=35) vs NEMO
  **Gouretski/WOCE climatology** IC. 3rd faithfulness gap. → Added **`--woa-init`** to the runner
  (`init_ocean_from_woa`, WOA18 ≈ Gouretski) so legoESM starts from climatology too. Smoke-testing
  WOA-init stability (realistic gradients may excite PGF). Current rest-state run 8089905 is
  IC-confounded (superseded if WOA-init stable). NEMO's EXACT Gouretski IC
  (`woce_{temp,salt}_monthly_init_4p2.nc`, staged in INPUTS) = further refinement.
- **iter 1 (WOA-init INSTABILITY — key blocker):** WOA-init (faithful IC) blows up by **day 1 at
  BOTH dt=600 AND dt=150** → **dt-INDEPENDENT** → NOT wave-CFL. It's the **cold-start imbalance**
  (WOA T/S + zero velocity = unbalanced baroclinic PGF) and/or PGF error on partial-cell bathymetry
  with realistic ρ. legoESM tripole is validated only for idealized rest-state IC — a genuine
  dycore-robustness gap (the hard part of OMIP cold-start; rest-state IC is gentle/stable, realistic
  WOA IC is not). `init_ocean_from_woa` shape (332,362,20) is correct — purely a stability problem.
  **PLAN (next iters, smoke-test each):** (a) tripole PGF adcroft→**smc03** (density-Jacobian,
  accurate on partial cells — run_omip latlon-bathy uses it for exactly this); (b) higher Laplacian/
  biharmonic viscosity during the adjustment; (c) bathymetry smoothing (run_omip --smoothing-passes/
  --r-factor-max); (d) thermal-wind-balanced initial velocity (kill the day-0 PGF shock); (e)
  nudge-from-rest spinup (run_omip --nudge-woa-tau) then release. Requires overriding the
  `_create_setup` tripole config (pgf_scheme, A_h, B_h) — add a config-override path to the runner.
  **INTERIM comparison:** rest-state run 8089905 (STABLE, ~10h) + NEMO 5yr ref → first
  **IC-caveated** legoESM-vs-NEMO comparison (proves the full pipeline + a verdict) while WOA-init
  stability is solved. Logging fix needed: run sbatch `| tail` buffers → hides live progress; use
  direct redirect for long runs.
- **iter 1 (WOA-init: RULED OUT 5 fixes via cheap smokes):** still blows up day 1 with — smc03 PGF;
  adcroft + 10×A_h + B_h=1e11; dt=150; **bathymetry_depth**-masked IC (no below-seafloor T/S garbage);
  AND a 30-day forcing ramp (~3% forcing at day 1 → forcing is NOT the trigger). ⇒ **definitively the
  cold-start geostrophic IMBALANCE** (WOA ρ field + zero velocity = O(1) unbalanced baroclinic PGF;
  the adjustment is what diverges). NEMO survives this; legoESM tripole does not. Remaining viable
  fixes (NEXT iters): **(A) thermal-wind-balanced initial velocity** — initialise u,v geostrophically
  from WOA ρ to cancel the t=0 PGF shock (the correct cold-start; `fidelity/references.py::
  thermal_wind_shear` + handle f→0 at equator); **(B) nudge-from-rest spinup** — start rest (stable),
  nudge T,S→WOA over ~months, optionally release to free-run. Runner now has --pgf-scheme/--A-h/--B-h/
  --forcing-ramp-days knobs + bathymetry_depth-correct WOA init + config-override path.
  **INTERIM (unblocks a first verdict now):** rest-state run 8089905 (stable) + NEMO 5yr ref →
  IC-caveated legoESM-vs-NEMO comparison via `compare_omip_nemo.py` when both finish.
- **iter 1 (nudge ALSO fails → fundamental dycore limit):** implemented nudge-from-rest spinup
  (start stable rest, nudge T,S→WOA over tau; runner: `compute_woa_3d` + `--nudge-woa-tau-days`/
  `--nudge-release-day`). tau=60d (only ~1.6%/day toward WOA from the stable rest state) STILL
  blows up by day 1 — same step as full WOA-init. A *tiny* realistic-gradient perturbation
  diverging that fast ⇒ **any realistic horizontal density gradient excites a fast-growing mode**
  (rest-state has uniform ρ → no horizontal PGF → stable). Diagnostic: WOA IC is FINITE + physical
  (T[-2.1,29.5]°C, S[0,40.7]) ⇒ blowup is DYNAMICAL, not NaN. **CONCLUSION: legoESM tripole eORCA1
  cannot stably integrate a realistic density field — a fundamental dycore-robustness limitation
  (the real barrier to the faithful tripole comparison).** Ranked next candidates (next iters):
  (1) **nlev=40-75** (20 is coarse; WOA→20-level interp may create steep/near-unstable profiles;
  more levels also more faithful to NEMO's 75) — cheap `--nlev 40` test; (2) fix the static-stability
  diagnostic (`compute_ocean_rho(state,z,jacobian)`) to confirm convective-vs-baroclinic; (3)
  thermal-wind-balanced initial velocity; (4) deeper PGF/partial-cell/barotropic-coupling
  investigation on the curvilinear grid. INTERIM rest-state comparison still the path to a first verdict.
- **iter 1 (nlev=40 fails too → STRATEGY PIVOT):** WOA-init at nlev=40 also blows up day 1 (8th
  ruled-out fix). The tripole eORCA1 (adcroft PGF) realistic-IC instability is fundamental, not a
  quick parameter fix. **KEY INSIGHT:** `run_omip._create_setup` latlon-bathy branch comment —
  legoESM's **lat-lon C-grid + real ETOPO bathymetry + smc03 PGF + biharmonic + implicit-CN
  barotropic** config "run[s] STABLE 50+ year integrations" with realistic geometry
  (global-overturning production runs). The *tripole* (adcroft) path is the unstable one, not legoESM
  ocean per se. ⇒ **PIVOT the faithful run to the latlon-cgrid-with-bathymetry production config**
  (applicator already supports `latlon`; WOA IC; CORE-II forcing; regrid to score vs NEMO ORCA1) —
  likely STABLE with realistic IC where tripole isn't. NEXT: latlon-bathy CORE-II run
  (`_create_setup(..., use_bathymetry=True)` + ETOPO + WOA IC), smoke it; if stable → the faithful
  run. Tripole stays the ideal same-grid target pending dycore-stability work. Rest-state run +
  NEMO ref → interim verdict meanwhile.
- **iter 1 (BOTH grids unstable → forcing-integration hypothesis):** Built multi-grid runner
  (`--grid latlon_bathy`, NEMO bathy regridded to lat-lon, smc03 production config). Results:
  (a) latlon-bathy + WOA IC ALSO blows up day 1 (setup fine: 50937 ocean cells, H_bathy[50,6004]m)
  → realistic cold-start fails on BOTH grids. (b) rest-state tripole 4yr run BLEW UP at day 300
  AFTER developing **unphysical max|v|~7-10, max|u|~5-6 m/s** from ~day 30 (10-day smoke had ~0.8 m/s)
  → slow RUNAWAY; even the "stable" baseline is unphysical; died before year 1 → interim comparison
  also blocked. **SHARPER HYPOTHESIS:** the runaway implicates my **operator-split forcing** — the
  external `apply_omip2_surface_fluxes` (forward-Euler force-then-step) is energetically inconsistent
  with the dynamics → pumps spurious KE. FAITHFUL + likely-stable path = integrate forcing INSIDE the
  step via **`model.step(state, dt, surface_forcing=OceanSurfaceForcing(tau_x,tau_y,q_net,sw_down))`**
  (the dynamics-core tau block, ocean_pe_latlon_cgrid.py:1884 — the path run_omip's jra55 mode uses),
  NOT the external applicator. **NEXT (leading fix):** compute CORE-II bulk tau/q_net per step → pass
  as `surface_forcing` to `model.step`; re-smoke rest-state (runaway gone?) then WOA; if WOA cold-start
  still fails → thermal-wind balanced initial velocity. **HONEST: no stable faithful global legoESM
  ocean run achieved yet — genuine multi-iteration dycore-integration work.** NEMO 5yr ref still
  running (reference is fine; legoESM side is the blocker).
- **iter 1 (in-step forcing — partial improvement):** Added `compute_omip2_surface_forcing` (builds
  OceanSurfaceForcing tau/q_net/sw_down on the model grid; tau atmospheric-convention + q_net total,
  so the dynamics core does the −flip/rotation/solar-split) + wired the runner to
  `model.step(state, dt, surface_forcing=sf)` instead of the operator-split applicator. Result
  (tripole rest-state, 36 d): finite throughout; peak max|v| ~5 m/s at day 30 (was ~8-10 with the
  split, headed to blowup) and dropped to ~1.7 at day 36 → reduced + possibly oscillatory, **not a
  full fix** — ~5 m/s is still elevated (physical ocean ≲2 m/s). So the operator-split was *a* cause,
  not the whole story. Launched 120-day rest-state in-step runs on BOTH grids (tripole + latlon_bathy)
  to see if velocities stabilise or keep growing → decides if there's a usable legoESM run for the
  interim comparison. WOA cold-start still needs balanced init (forcing-independent IC imbalance).
  Note: legoESM has `coupler.apply_runoff_step` (Dai-Trenberth) to ungate SSS/MLD/AMOC later.
  **HONEST: in-step forcing is a correct improvement but legoESM global-ocean stability under CORE-II
  is still not solved; no faithful run yet.**
- **iter 1 (FIRST end-to-end comparison — pipeline PROVEN):** tripole 90-day (rest-state IC, in-step
  forcing; physical T/S, velocity spiking late) vs NEMO month-3 via `compare_omip_nemo.py`:
  **SST bias −1.1 °C, RMSE 11.7 °C, corr 0.05 → POOR**; SSS RMSE 2.7, corr −0.01 (runoff-gated).
  Outputs: `results/omip_nemo/compare_day90/{report.json,SST_maps.png,SSS_maps.png,zonal_means.png}`.
  The near-zero correlation = the **IC gap**: legoESM rest-state IC is horizontally UNIFORM (no
  realistic SST/SSS pattern) vs NEMO's Gouretski fields — NOT a pipeline bug. **The full machinery
  (NEMO ref → legoESM run → regrid → area-weighted score → verdict + maps) is PROVEN and yields real
  numbers.** To turn POOR→excellent needs: (a) realistic IC (WOA/Gouretski) — blocked by the
  cold-start dycore instability → **thermal-wind-balanced init** is the key next code task; (b) a
  stable multi-year run — blocked by the velocity-spike instability (~day 80-110); (c) runoff
  (`apply_runoff_step`) to ungate SSS. **Bottom line: pipeline DONE + demonstrated; "fully faithful"
  is gated on legoESM ocean-dycore stability + balanced realistic IC — genuine model-development
  work, not config tuning. No false DONE.**
- **iter 1 (NEMO 5yr REFERENCE COMPLETE):** ran 4h07m on 32 cores → full 5-yr climatology.
  `nemo_rc=137` = the *postprocess* python killed; raw annual files intact:
  `RUN_REF/ORCA1_1y_20000101_20041231_{grid_T(337M, T/S/SSH/MLD), grid_U/V(220M, →ACC/AMOC),
  grid_W, icemod, ocebudget, SBC}.nc` + 33+ monthly grid_T records. **Reference side DONE** —
  legoESM is the sole blocker. (For transports later: NEMO grid_U/V give native ACC/AMOC; add to
  `compare_omip_nemo.py` once legoESM has a stable run with velocities.)
- **iter 1 (DYCORE-NUMERICS investigation — rigorous):** Pursued the dynamics-core fix.
  • Controlled flat-bottom test: WOA-init blows up day-1 even FLAT (no partial cells) ⇒ not
    PGF-over-topography alone. • Static-stability check: WOA IC is statically STABLE (0.01% mild
    inversions; ρ increases downward) ⇒ not a bad-IC/convection trigger. • flat-bottom + gradual
    nudge + in-step forcing + smc03 (the CLEANEST possible case: no topo, no cold-start shock, no
    split-forcing, best PGF) STILL blows up day-10. ⇒ **the instability is in the CORE baroclinic
    dynamics' response to realistic horizontal ρ gradients**, independent of topo/IC/forcing-path/
    PGF-scheme/dissipation. • Connected to legoESM's documented PGF work
    (`docs/ocean/experiments/pgf_*`): the partial-cell Adcroft PGF drives a 2Δz mode; smc03
    density-Jacobian is the mitigation; baseline shows smc03 residual ~3-6 mm/s on partial cells but
    **~2000 mm/s in the wind+stratification+topo regime** (= the OMIP regime → consistent with my
    runaway). • Verified the **critical smc03 bug C1 is already FIXED** (z_target = min(centroids),
    not the buggy midpoint) — and **fixed the stale C1 docstring** (`latlon_cgrid_operators.py:2843`,
    it still described the disproven midpoint). • KEY: the ocean test matrix passes **57/57**
    (idealized baroclinic_gyre incl.) ⇒ the dynamics handle IDEALIZED baroclinic gradients; only
    GLOBAL REALISTIC WOA (equator f→0, sharp fronts, global scale) breaks them — narrows the bug.
  **REMAINING dycore work (documented research, multi-step): (1) per-term tendency instrumentation
  of `model.step` to see which term grows; (2) equatorial isolation (mid-lat-only WOA) — f→0 prime
  suspect; (3) higher-order cubic-spline PGF (the docs' next step beyond smc03 linear) + H1 σ-extension;
  (4) thermal-wind-balanced init.** Not a single quick fix; genuine ocean-numerics development.
- **iter 1 (ROOT CAUSE found + partial fix — BIG):** Per-step localization of the WOA blowup:
  step-1 max|u|=12, max|v|=15 m/s @ **75°S** (spurious PGF from rest, ~30× too large), then the
  **equator (f→0) super-exponentially amplifies** → NaN by step 6. Inspecting the WOA IC field found
  the cause: **NEMO's ocean mask (tmaskutil) includes ~13% of cells WOA has NO data for → S≈0 there
  → 40-PSU jumps vs S~35 neighbours → the huge spurious PGF seed** (also explains mean SSS 23-30).
  **FIX (real, kept): `compute_woa_3d` now flood-fills those S<1 ocean cells from the nearest valid
  WOA column** (`scripts/run_omip_core2.py`). Validated: mean SSS 29.7→34.08, step-1 max|u| 12→0.78
  m/s (15× better). • BUT still blows up: residual seed moved to ~50°N, velocities reach ~5-10 m/s by
  step 3, then **super-exponential** (33→1037→7e7) = **nonlinear advective feedback (u·∇u)**, equator
  the final amplifier. ⇒ The remaining barrier is the **cold-start geostrophic adjustment**: from rest
  (u=0) the unbalanced WOA PGF drives O(5 m/s) transients that trip the advective blowup. **NEXT
  (the right fix): thermal-wind-balanced initial velocity** (set u,v geostrophic from the WOA ρ,
  tapered near the equator |lat|<5°) so the adjustment stays small + the advective feedback never
  triggers; pair with the flood-fill (done) + a smoother IC if needed. Reframes the problem: it is a
  cold-start / IC-conditioning problem (largely fixable), NOT a hopeless core-dycore bug — the flood-fill
  already removed the dominant seed.
- **iter 1 (equatorial f→0 is the residual core barrier):** with flood-fill, nudge-from-rest +
  in-step forcing survives ~20 d (vs 10) but the **equator (umax_lat −2 to −4°) grows 12→39 m/s →
  blowup ~day 30**. Equatorial viscosity boost (`A_h_eq_boost`=20) did NOT help (still day-10 blowup)
  → the f→0 growth is **not viscosity-dampable**. ⇒ Remaining barrier = **core equatorial dynamics
  (f→0 momentum/PGF/barotropic-split)** + the cold-start adjustment that feeds it. The right fix is
  **thermal-wind-balanced init** (minimise the adjustment) + possibly an equatorial-dynamics audit —
  careful core work, scoped. **NET this session: found + fixed a real WOA-IC mask-mismatch bug
  (flood-fill, 15× seed reduction, kept); reframed the "fundamental dycore instability" as a
  cold-start + equatorial-f→0 problem that is largely tractable; precisely localised the residual to
  the equator.** Runner knobs added: --flat-bottom, --A-h-eq-boost, --K-bih, --nudge-*, --pgf-scheme.
- **iter 1 (DEFINITIVE root cause — baroclinic dycore instability):** nudge-from-rest WITH the
  fixed in-step forcing ALSO blows up (day 10, tau=60 → only ~16% WOA). Gradient-magnitude scaling:
  0% WOA (rest) stable ~90 d; 16% WOA → day 10; 100% WOA → day 1. ⇒ **legoESM's ocean dycore cannot
  stably CARRY realistic horizontal density gradients — a fundamental baroclinic instability, not a
  cold-start/IC-imbalance issue.** Critically, **thermal-wind-balanced init will NOT fix it** (the
  model can't maintain the gradients however they're introduced; it's not just the day-0 shock).
  Ruled out (~13 experiments): adcroft/smc03 PGF, 10× Laplacian + biharmonic momentum viscosity,
  dt 600/300/150, nlev 20/40, below-bathy IC mask, forcing ramp, gentle nudge, operator-split→in-step
  forcing, tripole + latlon grids. **REMAINING candidate levers (deep numerics, untested):** tracer
  hyperdiffusion / biharmonic K_h to damp a grid-scale baroclinic mode; EOS/density-Jacobian PGF
  audit; barotropic-baroclinic split coupling; possibly the realistic run needs a fundamentally more
  robust PGF or higher resolution. **HONEST CONCLUSION: the faithful comparison is blocked by a
  fundamental legoESM ocean-dycore numerics problem (instability under realistic stratification) —
  genuine model-development research, NOT closable by config/IC iteration. Reference + forcing +
  pipeline + scorer are all complete and proven; the dycore is the wall.**
- **iter 1 (CONFIG SPACE EXHAUSTED):** biharmonic tracer hyperdiffusion K_bih = 1e12/1e13/1e14
  + WOA-init ALL blow up day 1 too. The full list of tested-and-failed levers (~16): PGF scheme
  (adcroft, smc03); momentum viscosity (A_h up to 1e6, B_h 1e11); **tracer hyperdiffusion (K_bih
  to 1e14)**; dt (600/300/150); nlev (20/40); below-bathy IC mask; forcing ramp (30 d);
  nudge-from-rest (tau 60); operator-split → in-step forcing; tripole + latlon grids. **NOTHING in
  driver/forcing/config space stabilises legoESM under realistic horizontal density gradients.**
  ⇒ the fix MUST be in the **dynamics core itself** (the PGF/density-Jacobian on the C-grid +
  partial cells, the barotropic↔baroclinic split coupling, or the baroclinic-mode discretisation) —
  dedicated ocean-numerics development, NOT achievable from this driver-level project. **FINAL
  HONEST STATE: reference + identical forcing + multi-grid runner + scorer all complete & proven
  (first real comparison produced); 3 real bug fixes landed; the faithful "match" is blocked by a
  fundamental legoESM ocean-dycore instability whose fix lives in the dynamics core. No false DONE.**
- **iter 1 (final):** SMOKE 8089701 SUCCESS (nemo_rc=0, 6m34s, no errors) — global
  `ORCA1_1m_20000101_20000201_grid_T.nc` (to/so/tos/sos/zos/MLD), mesh tiled (32).
  `Read -1 errno=1` MPI warnings = harmless containerized-CMA. Decoded grid_T +
  COREv2 schemas (see facts above). Launched 10yr REFERENCE run 8089712 (burst, RUN_REF,
  nn_itend=87600, nn_stock=8760) + eORCA1.2 mesh dl 8089713. NEMO side de-risked end-to-end.

- **2026-07-13 (ice-partition + KPP-freshwater + normalize + runoff/ISF fixes — codex 4-round loop):**
  Implemented the adversarial-review spec A–D on `omip-faithful-nemo-comparison`:
  **A** ONE shared mask-aware partition `coupler.ocean_forcing.blend_ice_ocean_forcing`
  (open-water stress/EVAP/heat/SW × f_open=1−A at the SINGLE PRE-step ice-conc time level —
  the state the ice integrated its atm fluxes over, so A+(1−A)=1 conserves the incident flux;
  ice basal heat/brine salt/melt fw/ice stress added exactly once; `raw_core2` mode reproduces
  the old `_ice_surface_heat` heat split bit-exactly and adds the missing STRESS+EVAP partition
  — the old wiring left full open-water τ + evaporation acting under ice, prime deep-polar-MLD/
  salinification suspects); runner `_route_ice_response_to_ocean` deleted;
  `omip_sea_ice_surface_forcing` refactored onto the same helper.
  **B** KPP freshwater buoyancy: `sf.freshwater = net_freshwater_flux(fw, restoring EXCLUDED)`
  on the non-cube host loop (KPP previously saw ZERO freshwater buoyancy — no halocline defense);
  consumer-scoped guard `_validate_kpp_freshwater_contract` (latlon/tripole reject 'external'
  which double-applies as virtual salt; MPAS 'external' is τ/q-only ⇒ safe).
  **C** `NEMOMatchTripoleRecipeConfig.normalize_freshwater=True` + passthrough + catalog entries
  (`omip_nemo_match_{tripole,mpas}_v1`) now carry it (recipe silently fell back to False vs the
  proven `_create_setup` True).
  **D** `--runoff`+`--isf` ice-shelf double-count fixed: `load_runoff_monthly(exclude_isf=args.isf)`
  drops `sornfisf` when ISF deposits it at depth; warns when the ISF file ≠ the runoff source.
  Also: prognostic ice now sees the SAME dm2dc-modulated SW as the ocean (shared `dm2dc_sw_factor`);
  ice TileResponse passed UNSCALED under the cold-start ramp (conservation with ice_state);
  NaN-safe `jnp.where` masking; stale runoff cell-sum test fixed to the area-integral contract
  (verified pre-existing failure at HEAD in a clean worktree).
  **Validation: 502 focused unit tests pass (incl. new partition/full-ice/ice-free-bit-identity/
  land-mask/KPP-buoyancy/brine-once/NaN-leak/guard/catalog tests + inline-coeff, private-import,
  dispatch-hardening ratchets); codex adversarial loop ran 5 rounds (r1: 4 real findings fixed —
  MPAS-guard abort, ice dm2dc/ramp inconsistency, restoring-in-KPP, ISF-file mismatch; r2: 2 fixed —
  ramp-scaling broke ice-ocean conservation (reverted), NaN×0 masking; r3: catalog normalize +
  consumer-scoped guard; r4: post→PRE-step partition conc (flux conservation at melt/freeze
  margins) fixed; the A·τ_sw·swd under-ice SW dribble is a documented PRE-EXISTING surrogate
  (~3% SW non-closure over ice, ice model has no penetration channel) — zero it via
  --ice-thermo-sw-trans 0 for a strict-budget A/B).
  NOTE: ll2_ri015 (8920073) + trp2_tkeice (8916859) ran WITH the old defects (full τ+evap under ice,
  KPP fw-blind, runoff+ISF double count) — re-run/re-score after merge before comparing to NEMO.
  The nemo-gaps-p4p5 worktree needs these fixes ported (its `--runoff-depth-nemo-ini` branch flag
  is unaffected by D but shares the ice/KPP defects).

- **2026-07-13b (residual-limitation closure + p4p5 merge + all-grid relaunch):**
  Merged `ocean-nemo-gaps-p4-p5` (42 commits: --ice-init NEMO SI3 IC, --tripole-vmix tke,
  --nemo-monthly-init/--sss-restore-file, ln_rnf_depth_ini runoff-depth map, --prescribed-flow
  SCM twins, NEMO centred split-explicit barotropic + DINO r1_exact) into
  `omip-faithful-nemo-comparison` → ONE tree for every grid. Then closed the three disclosed
  limitations, 6-round codex loop:
  **L1** `SeaIceConfig.sw_transmittance_const` (tail field, spec'd tier-2): constant-scheme SW
  transmittance is debited from the ice EB inside `compute_ice_sw` (reflected+absorbed+
  penetrated == sw_down EXACTLY) and reaches the ocean via the existing
  `sw_penetrated → ocean_heat_extraction` channel; runner retires the ocean-side A·τ·swd
  surrogate (`sw_transmittance_ice=0.0`) — SW budget over ice CLOSED; `--ice-thermo-sw-trans`
  now feeds the ice model and is validated for the prognostic path.
  **L2** `TileResponse.ice_concentration_thermo` (trailing None-default): v2 response exposes the
  post-transport pre-thermo aggregate concentration; the runner partitions open water at that
  exact flux time level (advective grids included).
  **L3 (tripole ice TRANSPORT)**: `upwind_to_u/v_points` PROMOTED ocean→core
  (`upwind_cell_to_uface/vface`; ocean re-imports); new fold-aware donor-cell C-grid advection
  `transport.fv_flux_divergence_latlon_cgrid` — E-N→face rotation with local angles; SEAM =
  ONE shared donor-cell upwind flux per fold pair (last-cell-row projection onto the seam
  normal, partner via vector-parity perm_v flip; `pad_ns_scalar`'s fold row is SIDE-SWAPPED,
  caught by a positivity test + hand trace) → exact pair cancellation + positivity; REAL
  eORCA1.2-mesh closure test (was 2.8e-4 leak, now <1e-10); gate split
  `grid_supports_ice_transport` (tripole YES) vs `grid_supports_ice_dynamics` (mEVP still
  MPAS/latlon/cube — curvilinear strain-rate ops = future dycore project); `--ew-cyclic-overlap`
  now slaves ice_state AND ice_resp halo columns; `_surface_currents` returns true geographic
  E/N on tripole via the canonical `rotate_tpoint_currents_to_geographic` (renormalized).
  Merge-artifact fixes: canopy param-spec double-classification (graduated), #928 ddm spec
  format, stale slab-ice heat-bound test (finding-#6 surplus melt-out warming is legitimate,
  verified pre-existing at the merge commit).
  **RELAUNCH (all grids, post-fix tree)**: ll3_ri015 (latlon_bathy, ll2 command verbatim),
  trp3_tke_iceinit (tripole full stack from MAIN checkout; stale trp2 on old code cancelled),
  mpas7 kppdeep r2 (ico7), DINO pub campaign r2 (latlon 365d wright+seos + MPAS 90d + figures),
  DINO L2 recipe intercomparison r2. Score day-90 vs NEMO month-3 as before.

- **2026-07-14 (post-fix A/B RESULT — ll3 vs ll2, clean one-variable):** ll3_ri015 (8974876) and
  ll2_ri015 (8920073) ran the BYTE-IDENTICAL command (latlon_bathy 180×360, --kpp-ri-crit 0.15,
  full P45, --prognostic-sea-ice, dt150, 0.25yr), day-90 vs NEMO month-3 — the ONLY difference is
  the code (ll2 @ 1cb0c4464 pre-fix; ll3 @ 4c32e98a2 post-L1-L3). The fixes IMPROVE every metric:
  SST RMSE 0.975→0.959 / bias 0.392→0.382 / corr 0.9967→0.9968 (both EXCELLENT); SSS RMSE
  1.451→1.409 / corr 0.768→0.782; Arctic SSS bias +1.44→+1.23 (fresher toward NEMO). **HEADLINE:
  Arctic MLD bias +78.7 m → +26.6 m (lego 176.9→124.9 vs NEMO 98.2) — a ~52 m shoaling toward
  NEMO**, the KPP-freshwater-buoyancy channel (fix B) restoring the polar-halocline defense the
  boundary layer was missing. Global MLD bias +13.1→+8.3 m. SSS still verdict-"poor" (Arctic
  rmse 3.59, +1.23 residual = the structural Siberian-shelf freshwater retention, NOT the
  ice-partition bug — see [[omip-cross-grid-arctic-sss]]).
  mpas7 kppdeep r2 (8974878, post-fix): SST EXCELLENT (RMSE 1.099, bias 0.025, corr 0.995), SSS
  GOOD (RMSE 0.975, corr 0.925) — strong absolute, but Arctic MLD +177 m too deep (lego 273.7 vs
  NEMO 96.1; separate MPAS issue, no pre-fix MPAS baseline at this protocol to A/B).
  trp3 (tripole, the run that exercises the NEW L3 ice transport) still queue-blocked on burst
  node availability — the headline L3 validation awaits it.

## Next actions (gated)

DONE iter1: probe, stage, BUILD, smoke-validated, WOA, ref-run 8089712 + mesh 8089713 launched.

1. ✓ DONE `scripts/build_core2_nyf_zarr.py` → `nyf.zarr` (validated, q_air clipped ≥0).
2. ✓ DONE applicator extended: `apply_omip2_surface_fluxes` now does latlon / cubed_sphere /
   mpas / **tripole** (+ wind-stress SIGN BUG fixed: ocean reaction −tau). spectral still
   unsupported (state has no grid-space u/v faces — follow-up). 12 unit tests pass.
   codex adversarial-review in flight (b9a6j470w) — parse + fix findings.
3. **Global CORE-II runner** `scripts/run_omip_core2.py` (NEXT — sole blocker for comparison):
   `load_core2_nyf` → per step map model-time → idx_t in [0,1460) → loop{
   `apply_omip2_surface_fluxes`(grid_type); `model.step`} → save annual snapshots + diagnostics.
   - **tripole (faithful, priority)**: read eORCA1 mesh → `land_mask=tmaskutil`, `H_bathy` from
     mbathy/e3t_0 (**NO ETOPO**); `create_tripole_grid(mesh)` + `rest_state_latlon_cgrid_ocean(
     geom, z, land_mask_override=, H_bathy_override=)`; reuse run_omip tripole config (A_h,
     GM/Redi, KPP, adcroft PGF, implicit vmix). legoESM z=20 levels (mesh 75 → use total depth).
   - latlon/cube/mpas (default configs) = coarse/idealized → won't match NEMO geometry closely;
     **tripole eORCA1 is the only realistic-geometry comparison**. spectral: applicator TODO.
3. Launch legoESM COREv2 runs: **tripole** first (eORCA1.2 mesh), then latlon/mpas/spectral/cubed_sphere (GPU sbatch).
4. When NEMO 8089712 done → `compute_yearly_mean.py`/`compute_monthly_mean.py` → reference climatology.
5. Extend `ocean/fidelity` w/ NEMO refs + integral diags (ACC/AMOC/MOC/MLD/SST-SSS RMSE); regrid; score; iterate; `/codex:adversarial-review`.

Running jobs: **NEMO 5yr ref 8089790** RUNNING on `short` (g079, ~4-5h; moved off scavenger
`burst` which stuck PD; nn_itend=43800, yearly restarts) — `omip_nemo/nref_8089790.out`,
waiter bb93uws3s. codex confirmation review in flight (b7n2hhj8e).
DONE so far: mesh (462MB), WOA, **nyf.zarr (6-hourly, 1460 rec, validated)**, applicator
(tripole + sign fix, 12 tests pass), codex round 1 (3 findings fixed/gated).
**Runner `scripts/run_omip_core2.py` VALIDATED + REAL RUNS LAUNCHED.** Chain works end-to-end
(mesh→mask/bathy, CORE-II forcing, tripole applicator, model.step). dt=3600 blew up (legoESM
tripole CFL); **dt=600 STABLE** (10-day smoke: SST 19.8→18.5°C, SSS 35, max|u|~0.8 m/s realistic
→ sign fix confirmed; 5.78 steps/s = 2.53 h/yr). Reuses run_omip `_create_setup`
(forcing_mode=jra55_do_tropical → sf scheme=none, only applicator forces). CAVEATs: applicator
host-side numpy (device↔host/step, OK at 5.8/s); KPP sees state-change not flux (operator-split,
buoyancy mixing via convection OK, wind-KPP under-rep — refinement).

### RUNNING (both faithful, → yearly snapshots for year-N comparison)
- **NEMO 5yr ref 8089790** (short, ~4-5h, dt=3600, ORCA1_1y_*grid_T) — waiter bb93uws3s.
- **legoESM tripole 4yr 8089905** (GPU g095, ~10h, dt=600, snapshot_yearNNN.npz) — waiter b2e3old20.
  Self-aborts on non-finite. Multi-year stability UNVERIFIED (10-day smoke only) — monitor.

**NEXT (blocker for "match" verdict): comparison machinery** — regrid legoESM tripole
snapshot (332×362, 20 lev, °C) → NEMO grid_T (to/so/tos/sos/zos/MLD, 75 lev); compute SST/SSS/
SSH RMSE + integral diags (ACC Drake, AMOC@26N, MOC, MLD) at matching year; score vs tolerances.
Extend `ocean/fidelity` (references.py/metrics.py/tolerances.py NEMO tier). Build while runs proceed.
