# AMIP tropical moisture campaign — 2026-09-15 (session AMIP3-watervapor)

Target: prw, pr, evaporation and boundary-layer RH in the tropics vs ERA5.
Reference run dd_ctl (production deck, days 80-110). Sibling plan with the
radiation/polar iterations: `amip_bias_campaign_plan_2026-09.md`.

## Measured chain (all CONFIRMED, tropics 20S-20N)
- 1000 hPa RH 82 vs 71 % (q +2.2 g/kg, T correct); sea-air humidity deficit
  0.61 of ERA5; ocean evaporation 0.84 of ERA5 with trades 1.31x too strong.
- Total-water ledger (sigma 0.7-1.0): microphysics +3.63 kg/m2/day = rain
  entering the band minus surface rain (4.43); rain water path 0.16 kg/m2 so
  that convergence is evaporation, not storage.
- Convection hands the microphysics 8.5 kg/m2/day of rain that ALREADY
  survived the IFS downdraft + sub-cloud evaporation (physstate_conv_precip,
  days 85 and 101 agree within 3 %).
- Direct kernel call (rain_evap_timestep.py, day 110): Morrison rain
  evaporation 4.14 kg/m2/day below sigma 0.7 + 1.22 in 0.3-0.7. EPSR*dt
  p99 = 0.04: no explicit-step overshoot (codex item 3 refuted).
- Literature (codex R1): IFS sends the survivor straight to the surface;
  MG1/MG2 and Jakob-Klein 2000 evaporate only in the precipitating,
  cloud-free fraction; CoMorph-A passes convective rain to microphysics but
  with a prognostic precipitation fraction. Sub-cloud evaporation in CRMs /
  IFS is 10-30 % of surface rain (GLM); here it is 93 %.

## Iteration A — survivor convective rain to the surface (commit 1eb97261a)
Arm `wv_sfcrain` (job 27469338): 5 days from dd_ctl's day-80 checkpoint,
`--convective-rain-to-surface`, otherwise the pa_ctl deck byte-for-byte
(REPO 1eb97261a = 598d2fd28 + switch + probes). Control `pa_ctl`
(598d2fd28, same EXTRA, same restart).
Pre-registered, days 81-85, arm minus control, tropics 20S-20N:
- CONFIRM: lowest-level / 1000 hPa RH falls >= 2 points; tropical-ocean
  hfls rises >= 3 W/m2 (evaporation ratio to ERA5 +>= 0.03); prw falls
  >= 1 kg/m2; pr changes by less than +1 mm/day net of the first-day pulse.
- REFUTE: |dRH| < 0.5 point AND |d hfls| < 1 W/m2 -> the third evaporation
  pass is not what sustains the sub-cloud moist bias.
- FIX-FIRST: blow-up, negative water, or rsut moving > 10 W/m2 in the
  window (a cloud-cover shock, not a moisture result).
Scoring: window_diff.py --ctl pa_ctl --arms wv_sfcrain --start 80 --end 85;
profile_rh_split on the day-85 checkpoints; land_ocean_rain for the
evaporation ratio.
Default decision for the user after the arm: `convective_rain_to_surface`
False (legacy) -> True (IFS route) in the production deck.

## Queue (both reviewers' order)
1. This arm; then 30 days if confirmed.
2. Vapour-and-phase process ledger (rain evap / cloud evap / deposition /
   sedimentation split) — codex's instrument; the direct kernel call above
   already gives the rain-evaporation term.
3. Partial-cloud condensation (H2, structural): offline PDF check first
   (does CLUBB's ADG1 PDF support cloud water at sub-saturated mean RH in the
   500-700 hPa bulge?), then Tiedtke-1993-style source/sink or CLUBB rcm as
   macrophysics. Existing sundqvist.py does NOT do this (needs grid-mean
   supersaturation, codex).
4. Precipitation-fraction / clear-fraction conditional rain evaporation
   for the stratiform rain that remains in q_r (H4).
