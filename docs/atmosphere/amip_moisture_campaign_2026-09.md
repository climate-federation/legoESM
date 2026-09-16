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

## Iteration A — 30-day result (wv_sfcrain30 − wv_ctl30, April file = days 90-110)
| quantity              | ocean ctl | arm−ctl | land ctl | arm−ctl |
|-----------------------|----------:|--------:|---------:|--------:|
| prw [kg/m2]           |   58.5    |  −9.0   |  52.9    |  −5.1   |
| hfls [W/m2]           |  103.6    |  +4.0   |  76.3    | +16.3   |
| evspsbl [mm/day]      |   3.58    |  +0.14  |  2.64    |  +0.56  |
| pr [mm/day]           |   4.44    |  +0.07  |  5.35    |  +0.62  |
| clt [%]               |   83.2    | −28.2   |  79.7    | −34.5   |
| rsut [W/m2]           |  118.6    | −37.7   | 139.0    | −41.6   |
| rlut [W/m2]           |  245.3    | +21.2   | 243.0    | +19.0   |
| rlutcs [W/m2]         |  267.2    |  +8.0   | 267.1    |  +6.9   |
| tas [K]               |  299.3    |  −0.1   | 298.8    |  +1.0   |
| lowest-level q, day 110 [g/kg] | 19.16 | −0.13 | | |
| lowest-level RH, day 110       | 0.860 | +0.002 | | |
Gates: prw CONFIRMED (about 60 % of the tropical-ocean excess removed);
evaporation ratio 0.84 → ~0.87 (gate ≥ +0.03: met, marginal); RH1000 NOT met;
the cloud-cover / radiation swing is a compensation exposed: the RH cover
scheme (rh_crit 0.85) was tuned against the moist column, so the drier column
now under-predicts cover (tropical rsut bias ~+30 → ~−8, rlut −8 → +13).
The "downstream" hypothesis (drying the cloud layer dries the surface layer
over weeks) is REFUTED at 30 days.
Decision for the user (RULE 3): production default `convective_rain_to_surface`
False → True, together with a cover re-tune by the cloud-cover session.

## Cap arms result (5 days vs pa_ctl, tropical ocean)
| quantity            | ctl   | M_b_max 0.05 | M_b_max 0.10 |
|---------------------|------:|-------------:|-------------:|
| mean active M_b [kg/m2/s], day 85 | 0.018 (78 % at cap) | 0.033 (31 %) | 0.038 (5 %) |
| hfls [W/m2]         | 101.0 | −2.4 | −2.4 |
| lowest-level q      | 18.57 | −0.09 | −0.03 |
| lowest-level RH     | 0.851 | +0.006 | +0.007 |
| prw [kg/m2]         | 57.1  | −0.3 | −0.4 |
| pr [mm/day]         | 4.54  | +0.22 | −0.06 |
| clt [%]             | 86.8  | −2.0 | −5.4 |
| rsut [W/m2]         | 124.7 | +4.2 | −0.3 |
The cap was binding and now is not, the closures act, and the surface layer
does not dry: the sub-cloud venting is not limited by the mass flux. REFUTED
as the surface-layer maintainer (fifth lever). Stable in both arms.
Score so far for the tropical-ocean surface layer (q 18.6 g/kg, RH 0.85):
rain re-evaporation, transfer-law corrections, Louis/CLUBB, shallow closure,
mass-flux cap — all null within 0.1-0.4 g/kg. The moist layer spans
1000-925 hPa; the 850 hPa level dried in the 30-day route arm while 925-1000
did not; 700-500 hPa is still 1.15-1.6x ERA5 after the route fix.

## Round 5 synthesis (codex xhigh + GLM), 2026-09-15 late
Agreed: the surface excess is the lower-troposphere humidity expressed at the
surface — venting acts on the humidity contrast across the boundary-layer
top, and the replacement air (925 hPa +12 RH points; 700 hPa still +15-60 %)
is itself too moist. Disagreements: none on mechanism; codex wants the
post-step numerics excluded first, GLM wants a free-troposphere nudging arm.
Reference check: observed trade-wind mixed-layer q at 26-27 C is 15-17 g/kg
(EUREC4A 15.5, RICO 15.6, BOMEX 16.8); model 18.6 (mixing ratio; 18.3 as
specific humidity) is 1.5-3 g/kg too moist depending on regime; RH 0.85 vs
~0.80 is the more robust statement.
Done: the biharmonic moisture filter + its floor measured offline on the
day-110 state: −0.003 kg/m2/day in sigma 0.95-1.0 (tropical ocean), floor
creates 0.0000 — numerical moistening REFUTED (threshold 0.2).
Running: wv_mb010sh (shallow closure ON with the cap released to 0.10,
27480130) — the one untested combination.
Next code (GLM drafting, codex + Claude review): CMT wired to MPAS edge
winds (bridge hands the scheme reconstructed cell winds, projects du/dv back
to edge normals through a shared voronoi helper; `bechtold_enable_cmt`);
5-day arm gates (codex): trade ratio 1.31 → ≤ 1.15 AND lowest-level q −0.5
g/kg CONFIRM; wind corrected with |dq| < 0.15 refutes fast moisture
sensitivity. Sign caution (Savazzi 2022): CMT can strengthen or weaken the
near-surface wind depending on the shallow/deep split.
Structural item still open: partial-cloud condensation (H2) for the
remaining 700-500 hPa excess.

## Iteration D — CMT on the MPAS lane (commit 2bfdf5413)
Arm wv_cmt (5 days from day 80 vs pa_ctl, --bechtold-enable-cmt, cmt_c_u =
cmt_c_d = 0.7, mass-flux cap at the production 0.02). Pre-registered
(tropical ocean, days 81-85 / day 85): CONFIRM = |u| at 1000-850 hPa ratio to
ERA5 1.31 -> <= 1.15 AND lowest-level q -0.5 g/kg (RH -2 pts); wind corrected
but |dq| < 0.15 refutes fast moisture sensitivity to the trades; wind NOT
reduced (or increased, Savazzi 2022) = the CMT sign question, report it.
Instrument: profile_rh_split (ua bias by level) on the published window.
Note the cap: with M_b at 0.02 in 78 % of columns the CMT is proportional to
a capped flux; a second arm with the cap at 0.10 follows if the sign is right.

## Shallow closure ON + cap 0.10 (wv_mb010sh, 5 days vs pa_ctl)
Tropical ocean: hfls +2.9 W/m2 (evaporation +3 %), lowest-level q −0.14 g/kg,
RH unchanged (T −0.15 K), prw −0.2, clt −5.8, rsut −2.6; land: hfls +11,
pr +0.8 mm/day. The largest surface response of the six levers, still a
third of the CONFIRM gate — inconclusive, not a fix.
CMT arm relaunched (wv_cmt, 27481448) on the conserving kernel after codex's
review of the wiring (8fdecb0ac); the first launch ran a kernel that leaked
column momentum at the surface and was cancelled.
Standing conclusion for the surface layer after six levers: no single
parameterization switch on this deck moves the tropical-ocean surface
humidity by more than 0.15 g/kg in 5 days or 30 days; the excess lives in
the 1000-925 hPa structure the 30-level uniform-sigma grid (≈33 hPa, ≈300 m
layers in the boundary layer) cannot resolve a mixed layer / transition layer
/ inversion in — a vertical-resolution arm is the natural next test and a
user decision (shared with the UTLS session's vertical-grid work).

## Iteration D result — CMT (wv_cmt vs pa_ctl, 5 days, conserving kernel)
Tropical-ocean wind speed 1000/925/850 hPa: 5.32/6.71/5.90 → 5.32/6.73/5.82
m/s; hfls +0.2, lowest-level q 0.00, prw −0.05, pr +0.28 ocean. REFUTED as the
trade-wind fix at the production mass flux (cap 0.02): the momentum flux is
proportional to a capped M_u and the shear-only Gregory closure; the 31 %
wind excess is dynamical. (A CMT arm with the cap at 0.10 would be ~2-5x
stronger; not launched — the cap arm itself moved nothing at the surface.)

## Decisions 2026-09-16 (user)
1. Convective rain to the surface ADOPTED (production default true) with the
   cover threshold re-tuned by the cloud-cover session on the wv_ctl30 /
   wv_sfcrain30 pair (day-110 state). 2. bechtold_M_b_max 0.05 ADOPTED.
3. Boundary-layer-refined vertical grid: GO (then partial-cloud condensation
   if it does not move the surface layer).

## Iteration E — boundary-layer-refined sigma grid (commit 2e35f7dad)
Grid: 30 levels, density bump refine 3 at sigma 0.95, width 0.06: 7 layers
above sigma 0.9 (~14 hPa, ~130 m) vs 3 uniform; the rest 33 → 40 hPa (the UTLS
coarsens, the opposite of the thin-UTLS grid that blew up for the UTLS
session). State: dd_ctl day 80 remapped with the conservative PPM tool
(remap_mpas_checkpoint.py, inventories conserved; carries interpolated).
Arm wv_bl5 (27485700): 5 days, dt 112.5, production deck + new defaults, from
the remapped state — STABILITY screen first (the deck's refined-UTLS restart
blew up on day 1-3). If stable: wv_bl30 (30 days) vs wv_ctl30 (uniform L30,
same restart, same days). Pre-registered (tropical ocean, days 91-110):
CONFIRM = lowest-level q falls >= 0.5 g/kg AND RH falls >= 2 points AND
evaporation rises >= 3 W/m2 with the 925 hPa RH excess halved; REFUTE =
|dq| < 0.15 g/kg; FIX-FIRST = blow-up. Caveat: the control's lowest level is
150 m, the arm's ~65 m — score on the same PRESSURE (ERA5 1000 hPa) and on
the lowest 100 hPa mean, not the lowest model level alone.

## Iteration E — stability screen (wv_bl5, 5 days, new deck, dt 112.5): STABLE
Max wind 57.8 m/s, hard-saturation drains 4-5 points/step (same as the
uniform grid). Day 85, tropical ocean, matched PRESSURE layers:
| run (deck)                    | lowest 100 hPa q / RH | 925-1000 hPa q / RH |
|-------------------------------|----------------------:|--------------------:|
| pa_ctl (legacy)               | 16.58 / 0.849 | 17.47 / 0.849 |
| wv_sfcrain (rain route)       | 16.53 / 0.845 | 17.48 / 0.849 |
| wv_bl5 (refined + new deck)   | 16.34 / 0.858 | 17.38 / 0.881 |
q slightly lower, RH higher (cooler lowest layers) after 5 days from the
remapped state — adjustment, not a verdict. 30-day pair launched on the SAME
new deck: wv_bl30 (27486649, refined) vs wv_ctl30b (27486650, uniform L30),
both from dd_ctl day 80 (remapped / native), checkpoints every 5 days.

## Cover re-tune (cloud-cover session, 2026-09-16) — PROVISIONAL, not in config
cloud_rh_crit 0.85 → 0.80 closes ~+5 of the −8 W/m2 tropical rsut overshoot
of the dried column (offline RRTMGP ladder + coupled 5-day pair at 0.82:
ITCZ rsut +3.2, clt +4.5, rain/prw unchanged; coupled response ~40 % of the
offline ladder). The +11-13 W/m2 rlut excess is not an rh_crit matter (high
cloud ~0 on the dried column): a high-cloud lever follows. User decision:
wait for that lever and re-tune once; PR #1756 carries the rain route and
M_b_max 0.05 only.

## Iteration E result — refined BL grid (wv_bl30 − wv_ctl30b, adopted deck, days 90-110)
| quantity (tropical ocean)     | uniform | refined − uniform |
|-------------------------------|--------:|------------------:|
| hfls [W/m2]                   | 101.1   | +3.4  |
| evspsbl [mm/day]              | 3.49    | +0.12 |
| prw [kg/m2]                   | 44.6    | −1.5  |
| pr [mm/day]                   | 4.06    | +0.34 |
| clt [%]                       | 55.3    | +11.1 |
| rsut [W/m2]                   | 83.3    | +10.9 |
| rlut [W/m2]                   | 273.6   | +2.3  |
| day 110, lowest 100 hPa q / RH / T | 16.31 / 0.837 / 296.49 | −0.08 / +0.019 / −0.36 |
| day 110, 925-1000 hPa q / RH / T   | 17.51 / 0.859 / 295.88 | −0.19 / +0.022 / −0.20 |
Surface-layer gate: REFUTED (q −0.1 to −0.2 g/kg, RH +2 points: the resolved
layers are cooler, not drier). Side result: the resolved boundary-layer top
restores +11 points of tropical low cloud, which offsets the cover collapse
of the rain route (rsut bias −8 → +3, clt 55 → 66 vs ESACCI ~65) — a
cloud/radiation win to weigh with the cloud-cover session's re-tune, and the
UTLS session must judge the coarsened upper levels (33 → 40 hPa).
Adopted-deck effect (wv_ctl30b vs wv_sfcrain30, i.e. M_b_max 0.05 vs 0.02 on
the rain route): lowest 100 hPa q 16.82 → 16.31 (−0.5 g/kg), RH 0.849 →
0.837 at day 110 — the cap change dries the surface layer over 30 days
where the 5-day arm showed nothing.
Per the user's order: 3a did not move the surface layer → 3b, partial-cloud
condensation, next.

## Iteration F — partial-cloud condensation (H2), design 2026-09-16
Both reviewers converge on the same family: a prescribed UNIFORM total-water
PDF equilibrium adjustment (codex, xhigh; exactly Sundqvist's cover relation
c = 1 − sqrt((1−RH)/(1−rh_crit)) with a condensate budget) — GLM's "(a)
Sundqvist partial condensation". Codex corrections adopted over GLM's sketch:
no separate clear-fraction evaporation sink (it would evaporate an exact
equilibrium), no detrainment cloud-fraction source for a diagnostic PDF
(detrained q_c enters q_t), radiation consumes the same PDF diagnosis at its
own state and bypasses the diagnostic condensate floor, the post-step hard
drain stays. Ranking: uniform-PDF > complete Sundqvist tendency > CLUBB ADG1
macrophysics > prognostic Tiedtke fraction.
Order: liquid kernel + AD/conservation tests (GLM coding now) → offline
capacity check on the day-110 state at 700/600/500 hPa → mixed-phase
ownership (single owner of vapour ↔ cloud transfer; note constants have
L_s ≠ L_v + L_f, a pre-existing enthalpy inconsistency) → Morrison wiring
under a static switch → shared radiation diagnosis → 5-day pair → 30-day
pair. Gates (codex): 5-day P(RH>0.98) −25 % rel., 500-700 hPa vapour bias
−10 %, prw −0.5, cover displaced < 10 pts, rsut/rlut not worse by > 5;
30-day: vapour bias −25 %, prw bias −25 % and ≥ 1 kg/m2, cover within 10
pts, |rsut|,|rlut| ≤ 15. Refutation: RH falls by warming, or vapour becomes
condensate that re-evaporates, or the excess persists with credible
sub-saturated clouds.

## Adopted deck scored against the references (wv_ctl30b, April = days 90-110)
| region        | prw bias old → new | rsut old → new | rlut old → new | clt old → new |
|---------------|-------------------:|---------------:|---------------:|--------------:|
| ITCZ 10S-10N  | +14.1 → −0.7  | +32.6 → −12.6 | −5.8 → +25.7 | +20.9 → −17.2 |
| trades 10-30N | +12.2 → +4.1  | +24.4 → +12.0 | −10.5 → +5.9 | +19.6 → +6.7 |
| trades 10-30S | +13.4 → +4.6  | +22.5 → −0.4  | −14.2 → +5.1 | +26.7 → +6.2 |
Day-110 tropical RH at 600 hPa: old deck 0.858 (P(RH>0.9) 66 %) → adopted
0.524 (8 %) vs ERA5 0.446; 700 hPa 0.466 vs 0.495 (now slightly dry).
Vapour ratio to ERA5 (ITCZ): 850 0.76, 700 0.89, 600 1.18, 500 1.61, 400
2.49, 300 3.13; 1000 hPa 1.12 (trades 1.18-1.23). Temperature bias grew with
the drying: +3 K at 850, +4-5 at 500, +6.5-7.5 K at 300-250 (the UTLS
session's item; at fixed RH the upper-troposphere vapour excess is largely
that warm bias).
Refined-grid run (wv_bl30) vs references: ITCZ prw −1.6, 600 hPa RH 0.425
(slightly dry), but trades rsut +31 / clt +15 — the resolved boundary layer
over-produces trade cloud; not adopted.
Offline capacity of the uniform-PDF condensation on the adopted deck's
day-110 state: the 550-650 hPa cells above rh_crit are 8-20 %; one
adjustment moves ~0.025 g/kg there. The remaining mid-level excess sits
BELOW the condensation threshold (RH 0.5-0.85), where partial condensation
cannot act; H2's value is now cover/condensate consistency and the 500 hPa
band, not the bulk vapour bias.
Remaining moisture biases on the adopted deck: upper troposphere (≥ 500 hPa,
riding the warm bias), trades prw +4, surface layer +12-23 % q (RH +5 pts),
evaporation ~0.87 of ERA5.

## Iteration G — the plume never terminates (2026-09-16 evening)
Main merged into the branch (acb1203a4; one additive test conflict; gates 325 +
5220 passed). NOTE for every later pair: main brought three peer-session
AMIP changes (van Leer vertical advection default, cold-start fix + Antarctic
snow, CLUBB upper limit); arms launched from this worktree are no longer on
the adopted deck's physics. The adopted deck is pinned in the worktree
`wt_wv_ledger` at 196af6b6d (physics-identical to wv_ctl30b's b26e5d8dc).
Codex on the earlier "cap dries the surface layer over 30 days" attribution:
CONFOUNDED — the wv_ctl30b / wv_sfcrain30 pair also differed in the ice
albedo and land-snow ageing merged from main between the two launches.

Lowest 3 km, day-110 snapshot on native levels vs ERA5 April climatology,
ocean only (q g/kg / RH / T K); figure maps/bl_profiles_adopted_deck.png:
ERA5 trades 10-30N: 1000:13.6/0.76/295.9 975:13.2/0.81/293.9 950:12.1/0.80/292.4 925:10.6/0.74/291.3 900:9.3/0.68/290.4 875:8.2/0.61/289.6 850:7.2/0.54/288.8
model  trades 10-30N: 985:16.2/0.88/296.3 952:13.8/0.84/293.8 919:11.5/0.77/292.0 886:8.5/0.60/291.2 853:6.0/0.41/291.3 820:4.5/0.28/291.4
ERA5 ITCZ:  1000:17.5/0.80/299.2 975:17.2/0.88/297.2 950:16.5/0.90/295.5 925:15.1/0.86/294.4 900:13.8/0.82/293.3 875:12.7/0.78/292.3 850:11.6/0.75/291.2
model ITCZ: 981:19.5/0.88/299.3 948:16.7/0.84/296.8 915:14.1/0.78/294.9 882:10.3/0.55/294.6 849:8.0/0.43/294.2 816:7.0/0.39/292.8
Reading (codex: pattern PLAUSIBLE, my percentages overstated by the level
matching; GLM: structure CONFIRMED): surface layer too moist, 815-886 hPa
cloud layer too dry and 2-3 K too warm, near-isothermal 886-820 hPa in the
trades. Undilute parcel from the model's ITCZ level-1 state is +1.65 K at
850, +2.5 at 500, +3.8 at 300 hPa above the parcel from ERA5's 1000 hPa
state (measured bias +3.1/+3.8/+6.5) — the boundary-layer theta_e excess is a
large part of the free-tropospheric warm bias (both reviewers: PLAUSIBLE, not
attribution; entraining plumes and ERA5's own sub-adiabatic profile cut it).

1-day total-water ledger on the adopted deck (day 110-111, kg/m2/day, ocean;
arms wv_led2_080_090 / 090_095 / 095_100, pinned worktree):
| region, band | turbulence | convection | microphysics | dynamics |
|---|---:|---:|---:|---:|
| ITCZ 0.80-0.90 | +0.59 | -2.54 | +0.27 | +1.69 |
| ITCZ 0.90-0.95 | +1.65 | -0.79 | +0.03 | -0.87 |
| ITCZ 0.95-1.00 | +1.13 | -0.41 | -0.05 | -0.66 |
| trades N 0.80-0.90 | +1.13 | -0.99 | -0.00 | -0.05 |
| trades N 0.90-0.95 | +1.42 | -0.32 | -0.13 | -0.93 |
| trades N 0.95-1.00 | +0.97 | -0.15 | +0.01 | -0.81 |
| trades S 0.80-0.90 | +1.20 | -1.90 | +0.02 | +0.65 |
Active convective columns only (trades N, 0.80-0.90): turbulence +1.68,
convection -1.73. The convection scheme is a net total-water SINK of the
810-910 hPa cloud layer everywhere in the tropics; the diffusion scheme
supplies it; convection vents almost nothing from the surface layer
(-0.15 to -0.4). (Codex: the ledger sums all phases, so the convection term
includes rain export; a vapour-only split is a follow-up.)

The carried updraught mass-flux profile (day 110, active ocean columns,
normalized by the column maximum; "p: fraction of columns with M/Mmax>0.1 /
mean normalized M"):
trades 10-30N: 887:0.08/0.04 854:0.22/0.08 788:0.60/0.26 688:1.00/0.92 589:1.00/0.99 490:1.00/0.97 391:1.00/0.92 291:1.00/0.84 192:0.97/0.70
ITCZ:          915:0.00/0.01 849:0.58/0.23 816:0.93/0.36 685:1.00/0.90 586:1.00/0.99 487:1.00/0.84 388:1.00/0.68 290:1.00/0.54 191:0.98/0.37
The mass flux grows 15-50x from cloud base to a maximum at ~590 hPa, stays
within 30 % of it up to 300 hPa and reaches 190 hPa in 96-99 % of active
columns — in the trades as in the ITCZ; no trade column has a top below
750 hPa (53-68 % of trade columns active). With the closure/cap acting on
the profile maximum, the cloud-base flux is 0.002-0.012 kg/m2/s (observed
trade cumulus 0.02-0.04).
Offline call of the production scheme on the same state (scratch probe,
zero carry => instantaneous profile): SAME shape (trades N 886:0.06/0.02
688:1.00/0.67 588:1.00/0.95 192:0.92/0.50), so it is not the carry averaging
intermittent deep events. `buoyancy_death_memory=True` (the P2 "plume revives
above an inversion" switch, unreachable for Bechtold from any run) changes
nothing (192 hPa: 0.92/0.50 -> 1.00/0.71): the plume is never killed at the
inversion in the first place — REFUTED as the mechanism.
Undilute parcel from level 1 (the scheme's own LCL/LNB routines): FIRST level
of neutral buoyancy at the trade inversion, median depth 1.7 km in 10-30N
(39 % below 1.5 km, 84 % below 3 km), 2.9 km in the ITCZ.

CODE (read): the plume kernel has no updraught termination (the IFS
kinetic-energy profile `_ifs_updraft_ke_profile` is computed but used only for
in-plume precipitation conversion and the closure's mean velocity), no
organized detrainment where buoyancy is negative (the port's own comment lists
the missing cuascn pieces: the 0.75·M per-layer detrainment limiter, the ZMFMAX
redistribution, the negative-buoyancy ZOCUDET detrainment, the shallow/mid
"detrainment = entrainment" tie), and a soft local buoyancy filter
(sigmoid, 2 K width) that lets a plume pass a 1-2 K inversion at 12-38 %
strength and regrow above it by entrainment (epsilon 1.75e-3 (1.3-RH) f_scale,
larger in the model's dry cloud layer) against a turbulent detrainment of only
7.5e-5 /m. Claim (PLAUSIBLE, code-read + profile fingerprint): this is why no
shallow-cumulus regime exists over the tropical ocean, why the cloud layer is
dried and the upper troposphere moistened by convection everywhere, why the
cloud-base flux is starved, and a large part of why the free troposphere is
warm. Next: IFS-faithful termination + organized detrainment designed with
codex + GLM, coded by GLM, 5-day screen pre-registered on the mass-flux top
distribution, 850 hPa q, 300-500 hPa vapour, ITCZ rain/prw, rsut/rlut.

### Iteration G — plume buoyancy and IFS kinetic energy captured offline (day 110, 6000 ocean columns 30S-30N, eager call of the production scheme)
| region | active | cloud base p50 | min B_u 780-920 hPa p10/50/90 [K] | KE at base p50 | min KE in the CIN layer p10/50/90 [m2/s2] | columns whose KE stays > 0 through CIN | LFC p50 | first KE<=0 above the LFC p10/50/90 |
|---|---:|---:|---|---:|---|---:|---:|---|
| ITCZ | 0.60 | 882 hPa | -1.30/-0.67/-0.25 | 0.13 | -2.06/-1.09/+0.38 | 0.16 | 782 hPa | 158/323/455 hPa |
| trades N | 0.65 | 820 hPa | -2.55/-1.63/-0.43 | -1.49 | (floor)/-1.96/+0.79 | 0.14 | 687 hPa | 157/324/789 hPa |
| trades S | 0.69 | 884 hPa | -2.55/-1.19/-0.15 | 0.23 | (floor)/-0.91/+2.95 | 0.33 | 784 hPa | 226/423/794 hPa |
Reading: every active plume is negatively buoyant by 0.3-2.5 K between cloud
base and a level of free convection 100-130 hPa higher (the model's warm
815-886 hPa layer is a CIN layer even in the ITCZ); the soft local filter
(sigmoid, 2 K) lets 12-40 % of the flux through and the plume regrows above
the LFC (max buoyancy +1.8 K ITCZ, +0.8-1.0 K trades) up to 300-450 hPa,
where the IFS kinetic-energy budget first turns negative — the 150 hPa gate
(p_conv_top_pa) is what sets today's tops. The IFS termination rule applied
with the port's cloud-base kinetic energy (~0.1-0.2 m2/s2) would kill 84 %
of ITCZ plumes inside the CIN layer: the rule needs IFS's cloud-base kinetic
energy / test-parcel treatment and the shallow test ascent, not the KE gate
alone. Both reviewers' ordered set (codex r9 pending, GLM): (1) absorbing KE
termination threaded through closure/carry, (2) shallow/deep from an
entraining test ascent with the 200 hPa depth criterion, (3) organized
detrainment + shallow detrainment tie later; offline day-110 gate before any
arm. Probe outputs: amip_runs/_wv/plume_ke_day110.txt.
