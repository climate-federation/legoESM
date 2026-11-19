# Pre-registration: layered snowpack + soil freeze/thaw vs bulk (2026-10-06)

Change under test (ONE switch, one code package): `--land-snow-scheme layered`
on branch feat/layered-snow-freeze-thaw (468cef302 + this PR): the 5-layer pack
solved with the soil in one heat column, soil freeze/thaw under it with an
energy-exact phase correction, CLM5 densification (destructive metamorphism,
Vionnet overburden, wind drift, Slater/wind fresh-snow density), the pack's
thickness on the physical snow-covered fraction.  Soil freeze/thaw is ON in
both arms (production deck).  Everything else = production deck of the run's
repo.  The bulk-path code of this branch is byte-identical to 468cef302 in
offline land steps (1368 bulk arrays: freeze/thaw on and off, two surface
schemes; measured at 89f28d614, whose model code the launch commit carries
unchanged).

Pairs: arm and control from the SAME mv3y_vl checkpoint copy, day 275 (Oct 3
00Z) and day 280 (Oct 8 00Z), both to day 365 (Dec 31).  The bulk checkpoint
seeds the pack from its snow water (all ice at min(top soil T, 0 C), 250 kg/m3).
Controls: snow-heat's switch-off arms sh2_off_a / sh2_off_b (468cef302 + its
switch off), valid only if its 1-day gate is byte-identical to 468cef302;
otherwise lf_ctl_a / lf_ctl_b (plain 468cef302, bulk).  A shared control is
valid only if, after day 1, scripts/validate/amip_config_diff.py --against
<sh2_off_x> --run <lf_on_x> --expect land_snow_scheme,land_snow_insulation
exits 0 (the two resolved configurations differ in nothing else); else the
lf_ctl pair runs.

Scoring: 45-70N land (sftlf >= 0.5, cos-lat), first 7 days of each run
discarded.  Windows: Nov 1-30 and Dec 11-31 2001, vs ERA5 on the SAME days
(same-day scorer) and arm minus control on the same days.  ClimateEval too.

Primary (soil, the direct test): Dec 11-31 soil temperature at 17.5 cm
(linear interpolation between the nodes at 13.6 and 27.6 cm), arm - control
  CONFIRM >= +3 K; REFUTE < +1 K.
Secondary (air): Dec 11-31 tas arm - control
  CONFIRM >= +2 K; REFUTE < +1 K; HARMFUL <= -2 K (air cooling: an insulated
  snow surface decoupled from the soil can cool at night; sign uncertain).
Also reported, no verdict: Nov 1-30 of both; 45-70N land snow mass and snow
depth arm vs control; top-1 m soil ice; skin temperature vs ERA5 skt.
Validity: Dec 11-31 45-70N land snow mass arm vs control within +-20%; if
outside, a tas change is attributed to snow MASS, not insulation.
Report both pairs separately; a verdict needs both pairs on the same side.
Day-1 gate, automatic (bundle stage 1, check_day1.py): each arm runs one day,
then must show every array finite, the resolved config selecting its snow
scheme with soil freeze/thaw on, and (layered) the snow layers present, their
water equal to the snow water, and the pack-seeding warning in the log.  Any
failure stops all arms at day 1.  Held land column-steps (budget not closed)
are printed for arm and control, not gated.
Scoring pipeline: nh-cold's E0' scorer (/scratch/b/b381103/nhcold/e0p_score.sh:
cmor_window.py, score_sameday.sh, t2m_window_metrics.py from wt_nhcold
scripts/validate/amip_bias) for tas; soil temperature from the daily
checkpoints (land_ml_T_soil), linear between the 13.6 and 27.6 cm nodes.
