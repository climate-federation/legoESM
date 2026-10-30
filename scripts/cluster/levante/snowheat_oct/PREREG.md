# Pre-registration: snow thermal layer A/B on the land-stress baseline (2026-10-06)

> **Re-pinned 2026-10-16 (user decision):** #1880 and #1881 are merged, so
> every arm, including both base gate arms, now runs ONE frozen tree at main
> (`wt_snowheat_ab_main`, commit recorded per arm by `setup_arms.sh`). The OFF
> arms are main's production deck, not the earlier 468cef302 control. Main's bulk-snow rules differ from
> 468cef302 (unused evaporation energy to sensible heat, sublimation at the
> ice latent heat), so the offline evidence below predates them. The 1-day
> gate also tests that main restarts cleanly from the mv3y_vl checkpoints.
> The estimand is therefore the switch's effect under main's snow
> energetics: never pool or difference these arms against pre-merge
> (468cef302) arms. The re-pin followed the merge alone; no arm result
> informed it.

Change under test (ONE variable): `--land-snow-insulation` vs
`--no-land-snow-insulation`, both arms on the SAME frozen tree
(wt_snowheat_ab_main = main with #1880 and #1881 merged). The bulk
snowpack gets one layer: snow-surface temperature, ice heat capacity
c_ice*SWE and Jordan (1991, CLM5) insulation of the whole pack (fixed density
250 kg/m3, k = 0.2235 W/m/K, whole-cell cover) between its surface and the
soil, solved with the soil (freeze/thaw on, as in production). Everything else
= the production deck of that tree (land stress on, soil freeze/thaw on).

Pairs: ON/OFF from the SAME mv3y_vl checkpoint copy, day 275 (2001-10-03 00Z)
and day 280 (2001-10-08 00Z), each to day 365. The OFF arms are the plain
production default only if the 1-day gate (explicit switch-off vs the plain
default, and the plain default twice, all on the same tree) is
byte-identical, and a 1-day switch-ON run differs from switch-OFF in an entry
both checkpoints carry (else the switch does nothing and the A/B is void).
The ON path must also be byte-reproducible (two 1-day ON runs) and survive a
restart (ON restarted at day 276 equals ON run continuously to day 277), since
a long arm that is killed is resubmitted and resumes from its own newest
readable checkpoint (procedure in bundle.sbatch). The 2-day ON run's day-276 state must equal the 1-day ON run's. The same
2-day/restart pair runs with the switch OFF (an ON-only failure is the snow
layer, a failure of both is the model), and an ON restart with the saved
snow-layer temperature lowered 2 K on snow columns must end day 277 with a
finite snow temperature that is colder on average over snow columns than the
unperturbed restart's (the restart reads the layer). bundle.sbatch
re-checks all nine before launch.

Scoring: 45-70N land (sftlf >= 0.5, cos-lat weights), the first 7 days of
each run discarded. Windows: Nov 1-30 and Dec 11-31 2001, against ERA5 on the
SAME days (same-day scorer) and ON minus OFF on the same days; ClimateEval
too. The verdict uses the MEAN OF THE TWO PAIRS (single-pair noise ~2 K, per
nh-cold); each pair is also reported.

Offline evidence behind the predictions (S2, EXPERIMENT_LOG [snow-heat]):
replaying the Nov 27 state of the control code for 15 days with fixed
captured forcing, the switch halved the upward soil heat flux (12.6 -> 5.9
W/m2), kept +5.5 MJ/m2 more soil heat, warmed the top soil +4.6 K, and made
the snow SURFACE 0.6 K colder (insulation withholds soil heat from the
surface). No atmosphere feedback there.

Primary (mechanism, soil): Dec 11-31 soil temperature at 17.5 cm (linear
between the 13.6 and 27.6 cm nodes), ON - OFF, pair mean.
  Prediction +5 to +12 K (an Oct start retains heat before the soil cools).
  Mechanism FAILS if < +2 K.
Secondary (the cold bias, air): Dec 11-31 tas, ON - OFF, pair mean.
  CONFIRM (switch fixes part of the bias) >= +2 K;  NULL |dT| < 1 K;
  REFUTE < +1 K;  HARMFUL <= -1 K (colder air).
  Prediction -1 to +0.5 K (offline: colder snow surface at fixed air).
  Also reported: Nov 1-30 tas, soil heat content (top 1 m) and ground heat
  flux series, and the same-day bias of each arm against ERA5.
Validity: Dec 11-31 45-70N land snow mass ON vs OFF within +-20%; outside
it, a tas change is attributed to snow MASS, not insulation.
Run validity: every run reaches day 1 in its node log, each manifest records
the tree's commit, the snow-layer energy (snow node) stays finite and no snow
surface is exported above freezing.
The default (on/off) stays the user's decision after the result.
