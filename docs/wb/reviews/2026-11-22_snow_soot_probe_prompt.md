# Snow soot/dust probe — instrument review + detail-choice vote (2026-11-22)

Question (user-approved measurement): is our production spring snow albedo over 45-70N land brighter than
CLM5.0 SNICAR (the CAM6 land), with and without CLM5.0's own BC+dust loading in snow?

Files (worktree /glade/derecho/scratch/pg2328/snow_soot_wt, branch probe/snow-soot-snicar-2026-11-22):
- scripts/validate/clm50_snicar_oracle.F90 — stub modules + driver; compiles the UNMODIFIED
  /glade/u/home/branson/cesm2.1.5/components/clm/src/biogeophys/SnowSnicarMod.F90 (CLM5.0, Toon solver,
  snicar_optics_5bnd_c090915.nc = clm5_0 default, aging table snicar_drdt_bst_fit_60_c070416.nc).
  shr_assert.h replaced by empty macros. Runs SnowAge_grain each 1800 s step through a schedule, SNICAR_RT daily.
- scripts/validate/probe_snow_albedo_snicar.py — reads CESM2 CMIP6 historical.001 CLM5.0 monthly history
  (SNOFSR{VD,VI,ND,NI}/FSDS{..} = flux-weighted snow-only albedo; SNOBCMSL/MCL, SNODSTMSL/MCL, H2OSNO_TOP,
  H2OSNO), builds the oracle, and ages OUR production snow albedo (alpha_max 0.8077, alpha_min 0.5207,
  tau 150 melt-equivalent days, 5000 K Arrhenius clock; spectrally flat, no zenith term — this is what the
  production two-leaf path applies to both VIS and NIR bands) through the same temperature history.
- Pre-registration is in the probe docstring (thresholds fixed by the requester, not up for vote).

## A. Instrument review (find defects)
Check: stub correctness (anything SNICAR_RT/SnowAge_grain reads that the stub sets wrongly), argument
layout/bounds into SNICAR_RT, species order (bcphi, bcpho, ocphi, ocpho, dst1-4; DO_SNO_OC=.false.),
diffuse call passing coszen=0.5, history-ratio validity (SNOFSR* is spval when snl>=0 so it is averaged only
over snow-layer times while FSDS* is averaged over all times), deposition split, whether ours is the
production path, whether the verdict logic matches the pre-registration, anything that would make the
number silently wrong.

## B. Detail choices — vote each (keep / alternative), one line each with reason
1. History window MAM 1979-1983 (alt: 1979 only; 1979-1988).
2. Mask: cell-month FSNO >= 0.9 and landfrac >= 0.5 (alt: FSNO >= 0.5; >= 0.99).
3. BC hydrophilic/hydrophobic and dust 4-bin split from CMIP6 aerosol-deposition file ratios (MAM, 45-70N)
   (alt: all hydrophilic; report all-phi/all-pho bracket).
4. Two-layer pack: top layer = history-mean H2OSNO_TOP, lower = rest of mean H2OSNO; concentrations top =
   SNOBCMSL/H2OSNO_TOP, lower = (MCL-MSL)/(H2OSNO-H2OSNO_TOP), ratio of area-weighted means
   (alt: single layer with column-mean concentration).
5. Layer density 250 kg/m3 (sets dz, enters CLM aging density index) (alt: 200; 300).
6. Synthetic schedule: 60 d at 255 K dT/dz 20 K/m dry; 30 d at 265 K 10 K/m dry; 15 d at 273.15 K liquid
   fraction 0.05; fresh radius from air T 255 K; no new snowfall (alt: propose).
7. Score = mean over days 61-105 (spring + melt), cosz 0.45 for the direct beam (also report 0.3, 0.6).
8. Broadband for BOTH models with the CLM5.0 history's VIS/NIR x direct/diffuse SW weights.
9. W/m2 = albedo difference x CESM2 history mean FSDS over the same mask.
10. Self-checks fatal: dirty never brighter than clean; radius non-decreasing without snowfall.
Answer A as a defect list (severity, fix), then B as 10 lines "N: KEEP|ALT <which> — reason".
