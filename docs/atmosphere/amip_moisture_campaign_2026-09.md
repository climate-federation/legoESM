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

## Iteration A result — 5-day pair (wv_sfcrain − pa_ctl, days 80-85 published mean)
Tropics 20S-20N unless stated. Control values in brackets.
- prw −4.6 kg/m2 (ocean −5.2) [55.3]  → CONFIRM gate met (≥ −1).
- pr: daily tropical mean arm/control 7.6/4.1, 6.7/4.9, 5.8/4.9, 5.5/5.6, 5.4/4.8
  mm/day: a drain pulse decaying to +0.5 by day 5 → gate met.
- Ocean hfls +0.4 W/m2 (evspsbl +0.01 mm/day) [101 W/m2] → gate NOT met at 5 days.
  Land hfls +21.7, land evap +0.75 mm/day, land tas +1.0 K (more rain reaches the
  ground; less cloud).
- RH at 1000 hPa: ITCZ 0.86→0.85, trades 0.86→0.86 → gate NOT met; REFUTE
  condition (|dRH1000| < 0.5 pt and |d hfls| < 1) formally met at 5 days.
  RH at 850 hPa: ITCZ 0.71→0.58 (ERA5 0.73), trades 0.49→0.45: the layer the
  re-evaporated rain was moistening is the CLOUD layer (700-850), not the
  surface layer.
- q/q_ERA5: ITCZ 700 hPa 1.40→1.15, 600 1.76→1.53, 500 2.43→2.22; trades-N
  600 hPa 2.81→2.25. Mid-level bulge shrinks by a quarter to a third.
- Lower troposphere WARMS: dT bias at 850 hPa ITCZ +1.0→+2.7 K, trades +2.3→+3.0
  (the removed evaporative cooling); upper troposphere +4→+6 K at 250 hPa.
- clt −31 points [85.5 → 54.6; ERA5 ~65]; rsut −38.7 W/m2 (ITCZ −52, bias +34
  → −18); rlut +6.9 global, +18 ITCZ; dTOA +7.9 global. The pre-registered
  FIX-FIRST trip (rsut > 10) fired: the RH-diagnosed cover collapses with the
  4.6 kg/m2 drying. Whether it settles is the 30-day question.
Reading: the third evaporation pass DID sustain the lower-free-troposphere
moist bias (700-850 hPa) and about a third of the bulge above it; it did NOT
(in 5 days) sustain the 1000 hPa surface-layer excess or the evaporation
deficit. H1 as stated about the sub-cloud layer: NOT supported at 5 days;
the BL memo's caveat (BL/evaporation need >= 2 weeks) applies.
30-day pair launched from the same restart: wv_ctl30 (27471783, flag off,
this worktree) and wv_sfcrain30 (27471784), checkpoint-days 5.
Pre-registered for days 91-110: CONFIRM = ocean evaporation ratio to ERA5
0.84 → ≥ 0.90 and RH1000 bias +11 → ≤ +7 and tropical clt within 10 points of
ERA5 and |rsut bias| ≤ 15 W/m2; REFUTE = evaporation ratio +< 0.02 or clt/rsut
overshoot larger than the original bias persisting through days 100-110.

## Sub-cloud total-water ledger (1 day from day 80, production deck, kg/m2/day, tropical OCEAN)
| band (sigma)  | turbulence | convection | microphysics | dynamics |
|---------------|-----------:|-----------:|-------------:|---------:|
| 0.95-1.00     |     +0.75  |    -0.68   |     +0.51    |   -0.58  |
| 0.90-0.95     |     +1.85  |    -1.89   |     +0.49    |   -0.44  |
| 0.70-1.00 (led3, Louis deck, day 100) | +3.57 | -6.21 | +3.15 | -0.50 |
Reading: surface evaporation (3.5) enters the lowest 450 m and 80 % of it is
mixed UP by the diffusion scheme through sigma 0.95; the convection scheme
draws only 0.7 from the lowest band and 1.9 from 900-950 hPa — most of its
6.2 removal is from the 700-900 hPa cloud layer. The sub-cloud layer is
vented by turbulent mixing, not by the mass flux; the surface humidity is
then set by the mixed-layer balance, which is where the transfer-law
equilibrium argument (iteration B) applies.

## Iteration B — ocean surface-layer corrections (commits da7cca173, 2dd03c721)
Claim (codex: equilibrium sign CONFIRMED, attribution PLAUSIBLE; GLM had
called the height error "wrong sign" on the fixed-state flux): with
K = rho*C_E*U ~2x too large, q_a = q_s - E/K sits too close to q_s.
Arms, 5 days from the day-80 restart vs pa_ctl: wv_qsal (sea-water q_sfc at
p_s; 27473012), wv_zref (real input height + potential temperature; 27473013),
wv_sfcboth (both; 27473014).
Pre-registered (days 81-85, tropical ocean, native lowest level):
- wv_qsal: q_a falls by ~0.8 g/kg (the q_s correction) with E within 5 %.
- wv_zref / both: lowest-level RH falls >= 4 points, E changes < 5 %
  (ventilation-limited) — CONFIRM; E falls > 10 % with RH < 1 point — the
  transfer law was NOT the maintainer (REFUTE); E rises > 10 % — the BL is
  not ventilation-limited.
- Any arm: sensible heat and stress change too (T, wind inputs re-labelled);
  report them; tas diagnostics still use 10 m + fresh q_s (known, unchanged).

## Iteration B result — 5-day arms vs pa_ctl, tropical OCEAN (days 80-85 mean; native lowest level at day 85)
| quantity                    | control | sea-water q_sfc | real input height |
|-----------------------------|--------:|----------------:|------------------:|
| hfls [W/m2]                 |  101.0  |   -11.9         |   -12.7           |
| hfss [W/m2]                 |   14.9  |    +1.2         |    -7.1           |
| lowest-level q [g/kg]       |  18.57  |   -0.30         |   -0.42           |
| lowest-level RH             |  0.851  |   -0.004        |   +0.016          |
| lowest-level T [K]          | 298.99  |   -0.20         |   -0.68           |
| prw [kg/m2]                 |  57.1   |   -0.6          |   -1.3            |
| pr [mm/day]                 |  4.54   |   -0.26         |   -0.30           |
Both corrections cut evaporation by 12-13 % and dry the surface air by only
0.3-0.4 g/kg; the height correction also halves the sensible heat flux and
cools the surface layer, so its RH RISES. The pre-registered REFUTE condition
(E falls > 10 % with < 1 RH point of drying) is met: the transfer law is NOT
what maintains the moist surface layer; it was compensating for it. Whether
the two corrections are still adopted (they are the physically consistent
inputs) is a user decision — with them the evaporation deficit vs ERA5 grows
from 0.84 to ~0.73 until something else dries the boundary layer.
Reading (PLAUSIBLE, to be tested by the 30-day survivor-rain pair): the
surface humidity is high because the humidity contrast across the boundary-
layer top is weak — the cloud layer above it was kept moist by the rain
re-evaporation — so mixing and convective venting export too little. If so,
q_low in wv_sfcrain30 falls by >= 1 g/kg and evaporation rises during days
10-30 without any surface-layer change.

## Iteration C — IFS shallow closure ON (existing switch, default off "pending its own A/B")
Both reviewers' next candidate for the surface-layer excess: the sub-cloud
moist-energy closure (cumastrn ZDHPBL) sets the shallow mass flux from the
surface supply instead of CAPE. Arm wv_shallow (5 days from day 80,
--bechtold-use-ifs-shallow-closure, else pa_ctl deck). Known caveats (codex):
the MPAS bridge feeds it the previous step's SH+LH without the radiative
term; the cape_weight^2 * M_b_max cap and the carry relaxation still bind.
Pre-registered (tropical ocean, days 81-85 / day 85 native level): CONFIRM =
lowest-level q falls >= 0.5 g/kg AND hfls rises >= 5 W/m2; REFUTE = |dq| <
0.1 g/kg and |dhfls| < 2 W/m2 (then check offline whether the cap muted the
target before discarding the closure).
Existing Louis / CLUBB-diagnostic pair at day 86 (sc_louis / sc_clubbd):
lowest-level q 18.50 vs 18.59 g/kg, RH 0.843 vs 0.850 — the turbulence
scheme choice moves the tropical-ocean surface humidity by 0.1 g/kg only.

## Iteration C result — IFS shallow closure: NULL (5 days vs pa_ctl, tropical ocean)
hfls +0.3 W/m2, lowest-level q -0.01 g/kg, every other field < 0.1 of its unit.
The fluxes DO reach the scheme (no inert-closure warning in the log); the
closure is muted downstream by the mass-flux cap:

## Finding — the cloud-base mass flux is capped in most of the tropics
dd_ctl day 110, relaxed updraught mass-flux carry (physstate_conv_prog_profile),
tropical ocean: 86 % of columns active; the median, p75, p90, p95 and p99 of the
profile maximum are all exactly 0.0200 kg/m2/s; 71 % of active columns within
5 % of the cap; area mean 0.015. Production M_b_max = 0.02 is the LOWER bound of
its declared range (0.02-0.15); the scheme's standalone default is 0.05; IFS
uses the CFL limit dp/(g dt) ~ 2-3 kg/m2/s. Codex: the cap acts on the whole
profile (closure scale limited to M_b_max/max(profile)), so the pile-up is at
the profile maximum and the cloud-base flux is bounded through it; the
relaxation (1/17 per step) cannot create the pile-up. Observed trade-cumulus
mass flux ~0.02-0.04 (Barbados, Klingebiel 2021), deep regions higher.
Reviewers: GLM launch 0.05 and 0.10; codex launch 0.05 first — both launched
in parallel (same wall-clock, stability observed either way).
Arms wv_mb005 (27478724), wv_mb010 (27478725): 5 days from day 80 vs pa_ctl,
--params deck = production + M_b_max. Pre-registered (tropical ocean):
CONFIRM = lowest-level q falls >= 0.5 g/kg AND hfls rises >= 5 W/m2, stable;
REFUTE = |dq| < 0.1 and |dhfls| < 2; FIX-FIRST = blow-up. Guardrails: RH500
and prw (detrainment moistening), rain, the overturning.
