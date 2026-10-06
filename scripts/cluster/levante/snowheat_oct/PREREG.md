# Pre-registration: one-layer snow thermal node A/B (2026-10-03)

Change under test (ONE variable): `--land-snow-insulation` (branch
feat/land-snow-insulation). Snowpack gets a surface-node temperature, ice heat
capacity c_ice*SWE and Jordan (1991) insulation (bulk density 250 kg/m3,
k = 0.223 W/m/K) between its surface and the top soil node, solved with the
soil. Everything else = production deck of the run's repo.

Pairs: arm/control from the SAME mv3y_vl checkpoint copy, day 245 (Sep 3 00Z)
and day 255 (Sep 13 00Z), both to day 365 (Dec 31). Controls on base commit
e94825017, shared with the snow-albedo arm, valid only if the 1-day
switch-off gate (sh_gate_off vs sh_gate_base from day 245) is bit-identical in
every checkpoint array.

Scoring: 45-70N land (sftlf >= 0.5, cos-lat), first 7 days of each run
discarded. Windows: Nov 1-30 and Dec 11-31 2001, vs ERA5 on the SAME days
(same-day scorer) and arm minus control on the same days. ClimateEval too.

Primary (soil, the direct test):  Dec 11-31 soil temperature at 17.5 cm
(linear interpolation between the nodes at 13.6 and 27.6 cm), arm - control
>= +5 K.
Secondary (air, user bands): Dec 11-31 tas arm - control
  CONFIRM >= +2 K;  REFUTE < +1 K;  HARMFUL: <= -2 K (air cooling).
  Prediction +2..+6 K (GLM: +1..+3 more likely; codex: sign uncertain).
Validity: Dec 11-31 45-70N land snow mass arm vs control within +-20%; if
outside, any tas change is attributed to snow MASS, not insulation.
Report both pairs separately; a verdict needs both pairs on the same side.
