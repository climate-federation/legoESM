# AMIP bias campaign plan — 2026-09-14 (dual-reviewed: codex + GLM-5.2)

Reference run: `dd_ctl` (current production deck: Bechtold + CLUBB diagnostic +
biharmonic moisture filter + zenith-angle ocean albedo; res6, 30 levels;
days 80-110 from the old-deck day-80 restart). Scored on matched months
against CERES-EBAF / ERA5 / GPCP / ESACCI by `scripts/validate/amip_bias/`.

## Bias table (model - obs, dd_ctl)

| region        | rsut  | rlut  | rsutcs | rlutcs | clt   |
|---------------|-------|-------|--------|--------|-------|
| ITCZ 10S-10N  | +34.4 | -5.4  | -1.4   | -21.2  | +20.6 |
| trades 10-30N | +28.9 | -10.5 | -2.7   | -20.7  | +19.1 |
| trades 10-30S | +25.5 | -14.7 | +0.9   | -24.1  | +27.3 |
| Sc Peru       | +12.4 | -8.2  | -1.1   | -15.9  | +24.0 |
| Sc Namibia    | +24.8 | -14.2 | -3.5   | -20.1  | +35.2 |
| Sc California | -9.1  | +15.0 | +0.3   | -6.7   | -15.2 |
| SO stormtrack | +7.4  | +1.3  | +0.9   | -13.6  | +15.2 |
| NH midlat     | -2.3  | +11.4 | +0.3   | -7.3   | -10.2 |
| poles 60-90   | -22.5 | +4.6  | -19.4  | -3.8   | -40.7 |
| GLOBAL        | +12.8 | -2.1  | -2.9   | -15.3  | +6.6  |

RH split (`profile_rh_split`): the tropical RH excess at 500-700 hPa is
HUMIDITY (dq part +0.4..+0.65) with a cold part of -0.06..-0.2; above 300 hPa
the cold bias (-0.5) and humidity (+0.85) cancel to a moderate RH excess.
Near-surface: +0.11..+0.21 RH, all humidity.

Reviewer consensus (independent, same conclusions):
- Structural at 1.1 deg / L30, do not tune: California Sc (no resolved
  inversion), grid-scale humidity noise (keep the biharmonic), the saturation
  plateau's SHAPE.
- Wasted arms: any further entrainment / detrainment / fall-speed ladder on the
  moist bias; global CLUBB retunes for the Sc seesaw; cover-only arms;
  5-day arms for boundary-layer moisture, evaporation or snow (need >= 2 weeks);
  repeating the Latin hypercube before a gradient gives the direction.
- One unfixed provenance point: CERES-EBAF begins 2000, so all 1979 scores
  are against a seasonal climatology.

## Iterations (one hypothesis each; CONFIRM / REFUTE pre-registered)

0. **Penetrative downdraft ventilates the boundary layer** (both reviewers'
   first measurement). Arm `dd_on` re-launched to day 110 on the capped
   transport (job 27451412); control `dd_ctl`. Window days 91-110, tropics.
   CONFIRM: RH at 1000 hPa falls >= 3 points, evaporation ratio to ERA5 rises
   >= 0.03, RH500 median <= 0.88, tropical rain change <= +0.2 mm/day.
   REFUTE: drying <= 1 point, or rain > +0.5, or RH500 > 0.92.
   Gradient: none (binary lever). Blow-up mechanism of the uncapped flux is
   NOT identified; the cap is the guard, this arm is also its 30-day test.
1. **Offline radiation adjoint on the dd_ctl day-110 state** (replaces a
   6-arm finite-difference ladder). Reuse `_wave2_grad/grad_sensitivity.py`
   + `_albedo_hifi/harness_hifi.py` (FD-verified 0.5 %) and
   `amip_bias/rlutcs_kernel.py`: d(rsut, rlut, rlutcs)/d{cover per layer,
   LWP, visible-ice fraction, fsd, r_eff} and d(rlutcs)/dq(z).
   Pre-registration: the cover term must carry >= 50 % of the tropical +34,
   else the lever is brightness; rlutcs -20 must map to 500-300 hPa.
2. **Joint cover + inhomogeneity direction from (1)**: condensate-aware cover
   PAIRED with fsd at the gradient-selected ratio. 5-day paired screen, then
   30-day. CONFIRM: high cloud 5 -> >= 12 %, |d rlut| <= 2, d rsut <= +2 and
   tropical rsut improves >= 3. REFUTE: any gate fails -> #10 is structural,
   defer to a prognostic-cover project.
3. **Polar clear-sky (-19..-22 W/m2 = surface albedo)**: reconcile the canopy's
   absorbed shortwave with the exported (snow-aged) albedo BEFORE any albedo
   lever; offline surface-energy residual < 0.1 W/m2; then snow-ageing
   timescale and a two-category sea-ice albedo, 30-day. CONFIRM: polar rsutcs
   bias -19 -> >= -8. REFUTE: < 50 % recovery.
4. **Convective momentum transport (Gregory 1997) — implemented in bechtold,
   `enable_cmt` reachable from no run config, and its tendency is not applied
   on the MPAS lane** (edge winds). Wire it, then 30-day arm IF iteration 0
   confirms the boundary-layer -> trades -> overturning chain. CONFIRM: trades
   <= 1.15x ERA5, ascent excess <= 25 %, RH500 <= 0.85.
5. **Land ET partition**: refit the calibration tables with the two-leaf plant
   model ON, by gradient (`train_land_params_era5.py`, offline). 30-day arm.
   CONFIRM: arid ET ratio <= 1.15, humid >= 0.93, neither tas bias worse by
   0.2 K.
6. **One-step tendency gradient on the production MPAS lane** (screening
   instrument, replaces LHS/FD ladders for every later iteration): call the
   un-jitted step (`_step_jit.__wrapped__`) with the physics rebuilt inside
   the loss from traced `apply_param_overrides` leaves. Known blockers
   (codex): physics_fn is a static jit argument (primitive_eq_mpas.py:930),
   GWD factory `float()` casts (physics_pipeline.py:4001), carry side-channels
   skipped under trace (primitive_eq_mpas.py:897), host callbacks in
   orchestration (model_driver.py:10600). Gate: 1/8/32-step serial gradients
   finite, <= 1 % directional-FD disagreement, memory measured. Sharded
   gradient is known NOT finite — serial only.

Baseline for ESMValTool: `base90_r6` (job 27451176), current deck from
1979-01-01 to day 90, for ClimateEval TIMERANGE 197901/197903 (the arms'
day-80 restarts are old-deck states and have no complete month).

## Decisions that need the user (none taken)
- Iteration 2/3/4/5 each end in a production-config change; each will be put
  as one line "current -> proposed" when its arm passes its gate.
- Whether to spend the ~40 GPU-h per 30-day arm serially or as a paired
  batch across accounts.

## Iteration 1 result (2026-09-14, dd_ctl day 110, 2000 columns, 2 times of day, n_sub=1 adjoint, FD check on n_c 0.4 %)

Gradients of global TOA flux per e-fold of the knob [W/m2]:

| knob            | d rsut | d rlut |
|-----------------|--------|--------|
| cloud_fsd       | -66.7  | +107.9 |
| r_eff (liquid)  | -30.0  | +1.5   |
| LWP (q_c)       | +7.9   | -0.6   |
| N_c             | +7.1   | -0.3   |
| condensate floor| +7.7   | -1.8   |
| IWP (q_i)       | -0.02  | -0.3   |

Reading: ice is radiatively invisible (as before); fsd cannot rise above 1.0
and LOWERING it raises rsut, so fsd cannot pair with a cover fix; the
brightness knobs are worth <= 8 W/m2 per e-fold against a tropical +34, so
the reflected-shortwave excess is COVER (humidity), not brightness — the
pre-registered threshold for iteration 2's "cover term >= 50 %" is met by
elimination. Iteration 2 as written (cover paired with fsd) is REFUTED; a
cover fix must arrive with the humidity fix or it worsens SW.
Clear-sky OLR kernel (ref1979 profile, 2048 columns): humidity alone covers
65 % of the -15.3 W/m2, temperature 23 %, both 94 %.
Kernel rerun with dd_ctl's OWN bias profile (not ref1979's): the current deck
is WARM aloft (model - ERA5 = +2.4 K at 500, +6.7 K at 300, +7.0 K at 250 hPa)
and holds 2.5-4x ERA5's vapour above 500 hPa; humidity alone covers 106 % of
the clear-sky OLR deficit and the warm bias offsets 46 %. The old deck's cold
upper troposphere is gone; the clear-sky OLR error is now entirely humidity.

## Stability finding (2026-09-14)
The current deck's COLD START (base90_r6, from ERA5 1979-01-01) blew up at
day 4.0 with the downdraft transport OFF — the same day the downdraft arm
blew up from a day-80 restart. The blow-up state is fully NaN. Onset lies
between steps 2592 and 3024 (saturation adjustment drained 201 points, 2 g/kg,
at 2592; 2.8 million held land column-steps by 3024). A 5-day cold-start
rerun with 3-hourly checks (base90_diag) is running to catch the origin.
Surviving runs of the same deck: dd_ctl, sc_*, r3_* (all restarts, daily
checks), dd_diag (restart + transport + cap + clamp, 3-hourly checks).

## Results 2026-09-14 evening
- Iteration 0 REFUTED at 30 days (dd_on vs dd_ctl, days 101-110): tropical
  evaporation +0.3 % (gate >= +3 %), rsut +1.8 W/m2 tropics (worse), rain
  +0.27 mm/day, Peru evaporation +7 W/m2 and rsut +3.9. The penetrative
  downdraft does not ventilate the boundary layer at this resolution. Lever
  stays off. The CFL cap alone carried the arm through the day it blew up
  twice before (cap cures the restart blow-up: PLAUSIBLE, one run).
- COLD-START INSTABILITY of the production deck: base90_r6, base90_diag and
  the checkpoint-cadence control all blow up at day 3.1-4.0 from the ERA5
  1979-01-01 start (top-level winds 78 -> 446 m/s from day 2.8, model-top
  temperature falls to 168 K, ice number explodes). Each single revert
  survives 5 days with the top warming to 190 K like the old deck: dt 75 s
  (+rad every 48 steps), Louis turbulence, or the Laplacian filter in place of
  the biharmonic. The instability needs dt 112.5 + CLUBB + biharmonic
  together. No ESMValTool baseline exists until one of them is chosen or the
  instability itself is fixed. Restarted runs (day 80) are unaffected.
- Iteration 6 instrument WORKS: K=24-step remat adjoint on the production
  MPAS lane, 132 tunables in one backward pass, 67 GiB, 19 min. FD-verified
  on epsilon_deep for free-tropospheric vapour (0.8 %). rh_crit gradients
  FAIL the FD check (factor ~2; the cover threshold is piecewise) — do not use
  them. Ranked free-troposphere vapour sensitivities per e-fold (g/kg per
  45 min): cloud_depth_deep +0.010, epsilon_deep -0.0067,
  cloud_depth_shallow_max -0.0065, rhebc_land -0.0054, rain_vent_f2 +0.0052,
  M_b_max +0.0044, tau_bl +0.0040. 45 of 132 tunables reach the vapour at
  all. K=72 needs 506 GiB (remat per step); K=33 attempted.
- Iteration 3 fix drafted (uncommitted, land package): canopy absorbs and the
  driver exports ONE broadband albedo, snow layered on each band's base.
  Behaviour change to put to the user: the export no longer adds the
  prognostic dry-soil brightening (reviewers: double count with the
  soil-colour bands).

## Iteration 3 arms (2026-09-15, pre-registered before any number exists)
Diagnosis (codex, file:line in _tools/_codex_polar_diag.txt; GLM arithmetic):
sea-ice albedo is a flat 0.65 over the ice fraction only (surface_utils.py:108,
model_driver.py:10755); land snow ages 0.81 -> 0.52 on 3.67 days with the
temperature clock OFF (A=0, calendar ageing); glacier bands 0.82/0.62 are
darkened to 0.52 by the same overlay; the ice package's temperature/pond optics
(ice/shortwave.py:272,380) are not wired to this lane. GLM's cap arithmetic:
Arctic land snow ~+20, Arctic sea ice ~+14, Antarctic ~+4 W/m2 recoverable.
The replay attribution instrument FAILS its own Arctic control (+26 W/m2) and
is not used.
Arms, 5 days from dd_ctl's day-80 restart, all on the reconciled land albedo
(8065cb0e9): pa_ctl (reconciliation only), pa_tau150 (snow ageing 150 d),
pa_ice080 (sea-ice albedo 0.80). Scored with window_diff days 81-85, arm minus
pa_ctl, poles 60-90 rsutcs.
CONFIRM: tau150 recovers >= +8 W/m2 polar clear-sky (GLM: bias -19.4 -> ~-8);
ice080 >= +5. REFUTE: < +3 (the lever is not reaching the flux). Reject any
arm whose polar cloud cover moves > 2 points or whose global rsutcs moves
outside the polar caps by > 0.5. Note pa_ctl itself measures the
reconciliation's atmospheric effect against dd_ctl (expected ~0 at TOA: only
the land's absorption changed).
Result (days 82-84, sidecar means, arm minus pa_ctl, poles 60-90 clear-sky
reflected SW): snow ageing 150 d **+7.3 W/m2** (gate +8: at the gate within a
3-day window's noise; polar cloud +0.4 points, NH midlat +2.6, global +1.5);
sea-ice albedo 0.80 **+2.7** (gate +5: REFUTED at this size — the late-March
window barely lights the Arctic ice, so this is a lower bound on its
March-April value, not a null). No arm moved cloud cover. Both together
recover ~+10 of the -19.4; the aged-snow floor (0.52 vs a dry-snow 0.65-0.70)
is the next candidate and is a calibrated scalar fitted under the broken
absorption. 30-day pair launched: pa30_ctl vs pa30_both (tau 150 d + ice
0.80), days 101-110 against dd_ctl's windows.
30-day result (pa30_both vs pa30_ctl, days 80-110, published months vs CERES):
poles 60-90 clear-sky rsut bias -15.7 -> -2.7 W/m2; all-sky -22.3 -> -11.1;
polar clt bias -46 -> -50 points (the remaining all-sky deficit is CLOUD);
global rsut bias +10.2 -> +12.6; polar tas -0.5 K. Iteration 3 CONFIRMED for
the surface term. Production change (tau 150 d, ice 0.80) put to the user.
Next polar lever is cloud cover, not albedo.

## Cold-start instability: diagnosed (2026-09-15/16)
Mechanism (CONFIRMED by a discriminating restart arm): diagnostic CLUBB's
explicit TKE production runs away in one warm-pool anvil column (7-9N 163E,
110-270 hPa) on day 2 of the cold start, seeded by a vapour spike the old
Laplacian used to damp; the dynamics blows the winds up on day 2.8-3.4 and the
uncompensated post-step vapour floor manufactures ice (N_i 4e11/kg). Tapering
CLUBB to zero above 150 hPa (scheme field, now reachable as
clubb_trop_cloud_top_press) from the day-2 state runs clean to day 5 (peak
wind 65 vs 357 m/s). Restart twin at day 80 (tp_ctl / tp_taper150, days
82-84): the taper is climate-neutral to within 0.3 W/m2 in every region
(global rsut +0.01, ITCZ -0.3, cloud +0.06 points). Reviewer caveat (GLM):
150 hPa is inside the anvil layer; a production-term limiter would be the
targeted fix. Decision pending: production clubb_trop_cloud_top_press
None -> 15000 Pa (restores cold starts, neutral on restarts).
