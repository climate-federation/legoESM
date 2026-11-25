# ORCA2X long-run comparison design

Status: **DECISION 160 TAKEN (2026-10-24); ACQUISITION DONE; 30-day pilot
run.**  Window = 30-day pilot (240 steps at `rn_Dt = 10800` s).  Metric card =
core + dynamics with NEMO's own `zdf_mxl` mixed-layer depth `hmlp`.  Weights
use NEMO's interior/fold-duplicate masks.  Bars are a separate, later user
decision.  No shipped card or NEMO configuration is changed by this lane.

## Fixed comparison program

The candidate is rung 10's existing comparison card: the shipped ORCA2 card
plus the comparison-only recorded initial ice (DECISION 157), implicit
ice-ocean top drag with the recorded coefficient (DECISION 150), under-ice TKE
mixing length fed the recorded SI3 thickness (DECISION 154), and lane A's L9
route with the eddy-induced velocity (round 325).  There is no new physics in
this lane.

At every 10,800-s ocean step the runner replays the fields already decoded by
`nemo_testcase_l4_orca2_round1_ladder_gate._surface_forcings`:

- `emp`, `rnf`, `qsr`, `qns`, `sfx`, and the two-tracer `rnf_tsc`;
- native `utau` and `vtau`, plus `taum`;
- sea-ice fraction `fr_i` and the implicit ice-ocean drag coefficient
  `rCdU_ice`; and
- SI3 mean ice thickness `hm_i`, written where `zdf_tke` reads it
  (`ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/zdftke.f90:195`:
  "CALL orca2x_longrun_hmi_write( kt, nn_mxlice, ln_mxl0 )").

The surface record keeps the reviewed complete exchange schema (stresses,
heat, freshwater/salt/runoff, ice fraction/mass/flux, `rCdU_ice`, iceberg
exchange, runoff tracer content), so the existing reader, extent checks and
field-order admission apply unchanged.  Static geothermal heat flow, the
time-interpolated damping target and the solar chlorophyll forcing come from
the admitted deck; chlorophyll is the 12-record `chlorophyll.nc` climatology
interpolated at the step midpoint with NEMO's cyclic no-leap month centres.
The record also keeps the resolved `ocean.output`, including the analytic
initial-ice settings; no SI3 state is reconstructed from an assumed
configuration.

## Acquisition (done)

The operator ran the 30-day acquisition at producer commit `966b6f9012`;
the record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/acquisition/orca1ice_longrun_30day_np2`.
Its staged namelist runs 240 steps with a restart every 8
(`orca1ice_longrun_30day_np2/namelist_cfg:21`: "nn_itend    =      240";
`orca1ice_longrun_30day_np2/namelist_cfg:23`: "nn_stock    =      8").
It was re-admitted with the current tool (`run.sh --admit-existing --days 30`),
which binds every surface, `hm_i`, daily restart and log file by SHA-256.

The inherited observers that would otherwise grow without bound stop after
step 10 with observer-only early returns, each reading
"IF( kt > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer":
`ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icestp.f90:297`
(SI3 exchange), `ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icethd.f90:309,327,339`
(thermodynamic and ZDF-input frames) and
`ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icedyn_adv_pra.f90:602`
(Prather frames); the barotropic frame writer has the same cap on `kstp`
(`ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/stprk3.f90:314`).
Admission requires the first ten barotropic/stage/entry/SI3 bytes and every
new surface and `hm_i` frame to equal round 326.  The enumerated footprint is
6.249 GiB for 30 days (8.249-GiB disk guard); a one-year record would be
52.151 GiB.

## Sampling and comparison schedule

Replay inputs are instantaneous and recorded every step.  Comparison states
are NEMO's daily restarts, which hold the whole-step entry (`Kbb`) state:
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/restart.f90:176-180` writes
"CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )" through
"CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )".  The
candidate runner advances in eight-step `integrate_scan` batches and saves the
matching instantaneous state after each day.  The restart reader is checked
bit for bit against the record's step-entry frames (day d restart = step
8d+1 entry) wherever both exist (days 1-7).  Days 1-30 are compared daily;
"monthly" is the 30-day mean.  Trajectory departure is descriptive only; the
long-run verdict is statistical and awaits bars.

## Metric card (DECISION 160)

All horizontal reductions use NEMO's unique interior wet T-cell area
`e1t*e2t*tmask_i`; three-dimensional reductions also use the live
partial-cell thickness.  Bias = candidate - NEMO; RMS = root of the weighted
mean squared difference.

1. SST, SSS and SSH: daily area-weighted bias and RMS; 30-day mean difference
   maps.
2. Temperature and salinity: volume-weighted bias and RMS for 0-100 m,
   100-1,000 m and below 1,000 m (a partial cell is split at the exact band
   interface), plus per-level area-weighted profiles so a band cannot hide
   cancellation.
3. Surface kinetic energy (collocated at T points from the two adjacent wet
   U and V faces with face-area weights), barotropic streamfunction RMS and
   pattern correlation, and ACC transport through the closed Drake Passage
   section.
4. Mixed-layer depth: NEMO's `zdf_mxl` `hmlp`, evaluated identically on both
   states (below).
5. Sea-ice-relevant ocean fields: SST and SSS bias/RMS over ice-covered cells
   (`fr_i > 0` at that step).  `fr_i`, `hm_i`, `rCdU_ice` and the under-ice
   heat/freshwater fluxes are replay inputs, identical by construction, and
   are not scored as prognostic skill.

### Mixed-layer depth

`hmlp` is computed in `zdf_phy`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:330`:
"CALL zdf_mxl( kt, Kmm )"), which the RK3 step calls with the entry level
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:183`:
"CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )") from the entry-state buoyancy
frequency (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:175`:
"CALL bn2    ( ts(:,:,:,:,Nbb), rab_b, rn2b, Nbb )").  The criterion is
`rho_c` (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:45`:
"REAL(wp), PUBLIC ::   rho_c = 0.01_wp"), converted and integrated as

- `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:114-118`:
  "zN2_c = grav * rho_c * r1_rho0", then
  "hmlp(ji,jj) =  hmlp(ji,jj) + MAX( rn2b(ji,jj,jk) , 0._wp ) * (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))"
  and "IF( hmlp(ji,jj) < zN2_c )   nmln(ji,jj) = MIN( jk , ikt ) + 1";
- `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:122-123`:
  "iik = nmln(ji,jj)" and
  "hmlp(ji,jj) = ((gdepw_1d(iik  ) ) *(1._wp+r3t(ji,jj,Kmm))) * ssmask(ji,jj)".

The loop starts at `nlb10`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domzgr.f90:384-386`:
"zrefdep = 10._wp - 0.1_wp * MINVAL( e3w_1d )" and
"nlb10 = MINLOC( gdepw_1d, mask = gdepw_1d > zrefdep, dim = 1 )"), which is
W level 2 (10 m) on ORCA2.  The record does not write `hmlp`, so the
instrument recomputes `rn2b` with the card's own NEMO `bn2` path on each
saved state (NEMO restart and candidate snapshot alike) and applies the loop
above literally.  It is therefore NEMO's definition on NEMO's state, not
NEMO's online output.

### Reduction geometry

The compact `148 x 180` arrays strip the MPI and cyclic halos, but the
northernmost T-pivot row (`j=147`) folds onto itself: 99 of its 180 stored T
points duplicate another, 35 of them wet.  Counting every stored cell once
double-counts 262,015.55 km2.  Every reduction therefore uses NEMO's unique
interior masks:

- `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dommsk.f90:247,263-264`:
  "ssmask (:,:) = MAXVAL( tmask(:,:,:), DIM=3 )",
  "CALL dom_uniq( tmask_i, 'T' )" and
  "tmask_i (:,:) = ssmask(:,:) * tmask_i(:,:)".  `dom_uniq` marks a point
  unique when the fold exchange leaves it unchanged
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domutl.f90:131-132`:
  "CALL lbc_lnk( 'domwri', puniq, cdgrd, 1._wp )" and
  "lluniq(:,:,1) = puniq(:,:) == ztstref(:,:)").
- NEMO's own global sums weight by the same mask
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lib_fortran.f90:1508`:
  "zztmp = ptab(ji+iisht,jj+ijsht) * smask0_i(ji,jj)"), and the
  reconstruction reproduces NEMO's logged area
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/sbcfwb.f90:212`:
  "IF (lwp) WRITE(numout,*) 'Total Domain area (km**2):', area/1000._wp/1000._wp";
  `orca1ice_longrun_30day_np2/ocean.output:1662`:
  "Total Domain area (km**2):   357005237.10546613").
- U and V faces use `umask*puniq_U` and `vmask*puniq_V`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domwri.f90:118,120`:
  "CALL dom_uniq( zprw, 'U' )" and
  "zprt(ji,jj) = ssumask(ji,jj) * zprw(ji,jj)").
- A 3-D T-cell weight at level `Kmm` is
  `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/trdglo.f90:108`:
  "zvm = e1e2t(ji,jj) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))) * tmask(ji,jj,jk) * tmask_i(ji,jj)".
- Live thickness ratios come from each model's own SSH:
  `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domqco.f90:257,266`:
  "pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)" and
  "pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &", with
  `ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domain.f90:198,212`:
  "ht_0(:,:) = ht_0(:,:) + e3t_3d(:,:,jk) * tmask(:,:,jk)" and
  "r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:) + 1._wp -  ssmask (:,:) )".
  Volume weights use NEMO's `r3t`; each model's transports use its own
  `r3u`.

The Drake transport is the signed eastward U-face flux at raw ORCA2 column
`i=106` (zero-based; 67.000 W), summed over rows `j=20..37` and all 30 levels
and divided by 1e6.  With
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dommsk.f90:210`:
"umask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji+1,jj  ,jk)", row `j=20`
(Antarctic Peninsula) and row `j=37` (South America) are dry at every level and
`j=21..36` carry every wet face, so the section is closed.

The barotropic streamfunction is psi(j,i) = -(sum over rows j' <= j of the
column U transport), zero at the Antarctic coast, placed at F points.  RMS and
pattern correlation use `e1f*e2f` over unique F points with a wet surface U
face on either side.

## Bars

Not set here.  The pilot reports the numbers; the user decides bars next.
