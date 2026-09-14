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
