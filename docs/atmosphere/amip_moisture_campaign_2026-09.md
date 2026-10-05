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

### H2 kernel (commit 5ee4ef31b) codex review: FIX-FIRST, deferred behind the plume work
Findings (amip_runs/_wv/codex_h2_kernel.md): the width cap D = min((1-rh_crit) s, q_t) changes
the advertised PDF (cover 1/6 vs 7/30 at rh_crit 0.25, q_t = 0.6 s; harmless for rh_crit >= 0.5);
the 1e-12 floor breaks the bracket at s = 0 (reachable in float32) and keeps a finite width as
rh_crit -> 1; "exact gradient" overstated (exact only at a smooth root with inactive clipping);
the enthalpy identity needs l_over_cp = L_v/c_pd; wiring must hand Morrison the signed transfer
as condensation (negative branch included) and not re-apply T_new; six test mutations named that
the tests would not catch. Not wired; fixes queued for GLM before any wiring.

## Iteration H — oracle receipts (OpenIFS main 8f6f722, local copy under docs/references/openifs_arpifs, uncommitted) and the ship list
Full audit: amip_runs/_wv/codex_r11_oracle.md (codex xhigh, reading the local
cubasen/cuascn/cuentr/cumastrn/cubasmcn/sucumf sources; my own reading of
cubasen.F90:330-735 and cuascn.F90:395-730 agrees). What IFS does that the port does not:
- cubasen: a DEPARTURE SEARCH — surface test parcel launched with w^2 = (1.2 A^(1/3))^2 + 0.1
  (A from u_* and the surface buoyancy flux; the convection call passes u_* = 0.1),
  T/q excess from the surface fluxes clipped to [0.2, 1] K and [1e-4, 5e-4] kg/kg; elevated
  departures with +0.2 K, +1e-4 and w^2 = 1 (1 m/s); a mixed departure parcel within 60 hPa
  of the surface; shallow-test mixing eps = 0.8/z + 2e-4 (ENTSTPC1/2), deep-test mixing
  mu = min(1, 0.4 ENTRORG dz min(1, (q_s/q_s,lowest)^3)); the velocity recurrence
  w+^2 = [w-^2 (1 - 2 mu) + 2 b dz]/(1 + 2 mu) (cubasen.F90:513-557); cloud base at the first
  condensation, the test top where w^2 < 0 (:615-628); PWUBASE = sqrt(w^2 at the base)
  (:657-668); deep iff p_base - p_top >= RDEPTHS = 200 hPa (sucumf.F90:170), KTYPE 1/2
  from that (cumastrn.F90:513), mid-level (KTYPE 3) from cubasmcn (RH > 0.8, -omega/g launch).
- cuascn: PKINEU(base) = 0.5 PWUBASE^2 (:401); per layer, in order: turbulent D = DETRPEN M dz
  (cuentr.F90:147, 7.5e-5), D = min(D, 0.75 M) (:487), E = lagged organized entrainment
  (:679-691: M min[0.4, ENTRORG (1.3 - min(1,RH)) dz min(1, q_s/q_s,base)^3], zero if
  B <= -0.2 K), for KTYPE >= 2: E *= ENTSHALP = 2 and D = E (:504-510), D *= (1.6 - min(1,RH))
  (:514), CFL redistribution with M_max = dp RMFCFL/(g dt) (:499-520), transport + saturation
  adjustment (cuadjtq), KE recurrence K = max(-1000, [K- (1 - d) + dPhi (1/3) B_mean/T_v]/(1 + d))
  with d = min(1, 1.94875 (E or D)/max(1e-8, M-)) (:647-668), negative-buoyancy organized
  detrainment D = max(D, M- (1 - (1.6 - min(1,RH)) sqrt(clip(K/max(1e-3, K-), 0, 1))))
  (:669-676), then ACCEPT the level iff K > 0 and M > 0 and (B > -2 K or dT_env/dz < -3e-3 K/m)
  else zero M and K, D = M- (all incoming mass detrained), deposit the condensate, KCTOP stays
  at the last accepted level, the label 0 prevents any revival (:698-715); KTYPE <= 2 also
  stop where the adjustment leaves q_u unchanged (:720-726).
- cumastrn: deep closure ZMFUB1 = ZCAPE ZMFUB/(ZHEAT ZXTAU) with the plume-buoyancy pressure
  integral, tau = depth/(2 + min(15, w_mean)) x resolution factor, 720-10800 s; one column
  scale factor against the per-level CFL/RMFLIA = 2 bounds; NO inter-timestep profile
  relaxation, no per-level cap, no fixed top gate.
Port departures (one line each): a single undilute parcel with +0.5 K / +1e-3 excess, the
LNB/height class blend and a fixed KE seed (bechtold.py:2175-2269, _plume.py:608-625); KE
computed after the plume and never terminating it (bechtold.py:2546, _plume.py:667-874);
prescribed eps/delta without the lagged organized entrainment, the tie, the redistribution or
organized detrainment, condensate delivered as delta*M*q_c (bechtold.py:2430-2488, 2853);
relaxation into a carry (tau 1800 s), per-level cape_weight^2 M_b_max clip, 150 hPa gate
(bechtold.py:2773-2822).
Ship list (codex; GLM codes, codex + Claude review): (1) this record + the replay contract;
(2) departure search / test ascent (cubasen) with the paired-sounding tests; (3) coupled
main ascent with the ordered E/D, KE, acceptance and terminal deposition (cuascn);
(4) closure consumers on the diagnosed window, one column scale, no carry/clip/gate on the
faithful path — (2)-(4) are ONE physics PR, "do not ship the KE gate alone"; (5) mid-level
branch; (6) the day-110 replay evidence.
Replay contract (frozen before the replay): active = column max of the carried profile
> 1e-4 kg/m2/s on the adopted deck's day-110 checkpoint (the baseline-active denominator;
columns that become inactive count as failures); top = smallest pressure with
M/M_max > 0.1; prediction 70 % of baseline-active trade-ocean columns (10-30N and 10-30S
separately) with p_top > 750 hPa, < 60 % refutes; >= 80 % of baseline-active ITCZ columns
must keep transport above their frozen LFC; water/enthalpy budgets closed with the terminal
deposition; eager/JIT parity; finite-difference gradients away from switching levels.

## Iteration H — cubasen port: resolution ladder (2026-09-16 night)
Module packages/atmosphere/legoesm/atmosphere/physics/convection/_ifs_test_ascent.py (GLM from the local
source; codex rounds r12/r13 + Claude fixes applied; unwired, no tests yet; jax.grad still non-finite on
one path). Known-answer sounding (mixed layer theta 300 K / 17 g/kg below 950 hPa; above: undilute
pseudo-adiabat of that parcel minus 1 K, RH 0.8; isothermal 200 K above 150 hPa), source-literal gate,
uniform sigma:
| levels | layer | verdict | departure | base | top | w_base |
|---|---|---|---|---|---|---|
| 30 | 33 hPa | shallow | surface | 962 | 895 hPa | 1.5 m/s |
| 60 | 16 hPa | DEEP | 988 hPa | 954 | 650 hPa | 3.4 |
| 100 | 10 hPa | DEEP | 998 | 957 | 582 | 3.1 |
| 137 | 7 hPa | DEEP | 1002 | 958 | 558 | 2.5 |
The IFS trigger works as designed once the layers resolve the lifting condensation level (<= 16 hPa);
on 33 hPa layers every elevated parcel dry-lifts a whole layer and dies (kinetic-energy loss ~ h^2).
The trade sounding stays shallow (top ~802 hPa) at every resolution. On the model's day-110 states
(uniform 30-level AND the BL-refined grid of Iteration E) the trigger finds 58-87 % shallow columns
(tops 820-930 hPa) and no deep in the ITCZ: the model's warm 900-800 hPa layer, the product of the
non-terminating plume, blocks the test parcels — expected for a corrupted state, only a coupled run
tells whether it relaxes. Codex (r13, HOLD): an in-layer LCL split is a defensible adaptation but not
L137-equivalent; complete sub-stepping is structurally closer; both change more than the ascent
(CAPE, base/top bookkeeping, condensate halving). Decision for the user: refined vertical grid
(<= 16 hPa through the lower troposphere, converging with the UTLS session's grid work) versus a
sub-stepped trigger on 30 levels.

## Decision 2026-09-16 (user): "3 both — code with GLM, codex and Claude review"
Sub-stepped test ascent on the 30-level grid as the stopgap AND a refined vertical grid
(<= 16 hPa layers through the lower troposphere) as the target; the faithful main ascent
(cuascn) port proceeds in parallel. Grid work to be coordinated with the UTLS session.

### Iteration H — sub-stepped trigger, first results (2026-09-16 late)
GLM added in-layer sub-stepping to the test ascent (cfg.test_ascent_substeps, cfg.test_substep_condensate;
n = 1 reproduces the source path exactly). Claude review fix: the sub-interface buoyancy was compared
against the mid-cell environment (spurious -0.4 K). Result on the known-answer deep sounding (L30):
n = 1, 2, 4, 8 -> never deep (the trade sounding stays shallow, top 827, at every n). Level-by-level
comparison with the native L60 run (which triggers deep from its 971 hPa departure): the L60 parcel
launches from a half level whose cuinin humidity is the MIXED-LAYER air (16.9 g/kg) and enters the cloud
layer 3.8 g/kg moister than the environment; on L30 the first elevated departure (962 hPa) launches from
half level 945 whose cuinin humidity is already the cloud-layer air (12.8 g/kg; the rule takes the level
above), so the parcel is 2-4 g/kg drier and dies within 300 m. Sub-stepping the ascent cannot repair the
departure sampling. Candidate stopgap (to be validated by codex): run the SOURCE-LITERAL departure
search on a vertically refined copy of the column (environment interpolated between parent levels,
2-4 sub-layers), map base/top/type back to the parent levels — equivalent to the fine-grid oracle up to
the interpolation; requires the departure loop as a lax.scan (the unrolled loop at 120+ levels exhausts
compile memory) and a 2x test showed the interpolation itself must be done in the right variables (a
first attempt with piecewise-linear T/q in pressure gave no convection at all — under investigation).
Peer (UTLS session) drafted the target grid L45 (9 log layers 2-109 hPa, 18 x 33 hPa to 703 hPa,
18 x 16.5 hPa to the surface): the ported trigger classifies the deep sounding deep on it (top 662 hPa)
and the trade sounding shallow — sufficient for the trigger; awaiting the user's approval of the grid.

### Iteration H — refined-column trigger (the stopgap) passes its gate (2026-09-16, 23:00)
Codex r14: prefer the refined column over in-layer sub-stepping (deleted); departure loop as nested
lax.scans (GLM; bit-identical; 120-level compile 10 s / 1 GB, model-state jit 2 s). Wrapper
`ifs_departure_search_refined` (GLM): s and q reconstructed linearly in pressure between parent full
levels, hydrostatic geopotential, the unchanged source search on the refined column, indices snapped
back to parent levels; `column_refine` = 1 is the identity. Gate on the frozen soundings:
| case | native L60 | L30 x1 | L30 x2 | L30 x4 | native L100 |
|---|---|---|---|---|---|
| deep: ktype / top | 1 / 650 hPa | 2 / 895 | 1 / 625 | 1 / 591 | 1 / 582 |
| trade: ktype / top | 2 / 802 | 2 / 827 | 2 / 794 | 2 / 794 | 2 / 805 |
Departure within one parent layer, top within 50 hPa of native: PASSED. Model day-110 state (x2 / x4):
ITCZ none 44/34 % shallow 49/55 % deep 7/11 % (deep tops 190-490 hPa); trades N deep 7/12 %, S 23/36 %
(mostly congestus depth, tops 390-760 hPa). Open: eager/jit parity 17.8 at x2 on one output (a
switching column), non-finite gradient path, no unit tests yet, codex review of the wrapper pending.

### Status 2026-09-17 01:00 (codex out of credits until 2026-09-21 10:18 — Claude-only review meanwhile)
Landed on the branch (unwired, tests for the trigger only): `_ifs_test_ascent.py` (cubasen port, nested
scans, refined-column stopgap, 10 tests passing + 1 strict xfail for the gradient hazard) and
`_ifs_ascent.py` (cuascn/cuentr port: ordered increments, KE, organized detrainment, termination with
terminal deposition, in-plume precipitation and rain fallout; liquid-only; four Claude fixes on GLM's
draft: scan xs, cuadjtq call, unflagged-column writes, KLAB lifecycle). Smoke: deep sounding plume
962 -> 794 hPa, M peaks 1.5x base, rain forms; trade plume 962 -> 861 hPa.
Next (GLM codes, Claude reviews, codex when back): (1) tests for the main ascent (frozen soundings,
mass/energy flux consistency, termination, no-condensation stop); (2) the wiring design: IFS closure
consumers on the diagnosed window (cumastrn ZMFUB1 = ZCAPE ZMFUB/(ZHEAT ZXTAU), one column scale
against the CFL/RMFLIA bounds, no profile relaxation/clip/gate on the faithful path), the flux-form
tendencies from the ascent's fluxes + terminal deposition, the existing IFS downdraft and sub-cloud
evaporation kept, behind `BechtoldConfig.use_ifs_ascent` (default False until the arm decides);
(3) the day-110 replay contract (Iteration H) and the 5-day screen on the adopted deck; (4) the L45
grid arm once the user approves the grid (UTLS session drafts the layout).

### Cover re-tune status (cloud session, 2026-09-17 ~02:00) — still PROVISIONAL, nothing in config
High-cloud lever (condensate-aware cover floor) on the wv_sfcrain30 day-105 restart: q_ref 1e-4
overshoots (ITCZ rsut +28 / rlut -24); q_ref 3e-4 gives rsut +17 / rlut -12 (ITCZ), SW-heavy; their
interim recommendation if the PR must close: q_ref 3e-4 with rh_crit 0.85 (tropics ~rsut +9 / rlut -7);
a cold-only (T < ~235 K) variant is being tested for a LW-heavy lever (~half a day). Per the user's
decision the joint rh_crit re-tune waits for that; note the new convection path will move the
cloud-layer humidity again, so the re-tune should follow the 5-day screen of the faithful scheme.

### Iteration I — the cumastrn closure lands, and the port's gate debt is cleared (2026-09-17)
`_ifs_closure.py` (GLM, Claude-reviewed; codex review pending, its CLI is out of credits until
2026-09-21): first-guess cloud-base mass flux, deep CAPE closure, shallow sub-cloud MSE closure,
single ZMFS rescaling of the once-computed ascent profiles. Ten unit tests on the frozen
deep/trade soundings, all passing; the sub-cloud test is shown to fail when the index fix is
reverted.

Review found eight defects. The load-bearing one: the module read the cloud-base/top/departure
indices as 1-based IFS levels in the deep closure and as our own 0-based surface-last indices
everywhere else, so every mask in the CAPE closure and in the sub-cloud MSE integral was off by
one level. The ascent module settles the convention (it writes `M.at[ar, k_cbot]` and sets
`k_ctop = j`). Others: ZDH2 used the plume temperature at the cloud base where cumastrn:896 uses
IKB-1; the source's `LDCUM=.FALSE.` on the shallow ZDHPBL<=0 branch was dropped; a dead boolean
factor; a needless two-function split; a re-implemented ZTAURES (now the shared public
`mass_flux.ifs_ztaures` — and GLM's first version of that helper dropped the `dx<=0 -> 1.0`
sentinel, which would have moved every production Bechtold column from 1.0 to 20.2, since the
default `dx_m` is 0.0); and a `__param_spec__` keyed by source line with a duplicate key.

CALLER CONTRACT (measured, not read off the code): cumastrn runs CUASC with the first-guess
ZMFUB, so ZHEAT is linear in PMFU and ZMFS = ZMFUB1/ZMFUB is a ratio to that same base. The first
test harness handed the closure an ascent whose cloud-base flux was 46x smaller than the first
guess; ZMFUB1 then saturates at ZMFMAX on every deep column (measured on the deep sounding:
ZCAPE 2389, ZHEAT 0.1221, ZXTAU 736 s, so the closure asks for 26.6x the first guess against a
10x cap). **The wiring must launch the ascent with the closure's first guess**, or the CAPE
closure is inert and the cap decides everything.

Gate debt: all five AST ratchets were RED on this branch before today, including for the three
modules landed on 2026-09-16 (`_ifs_ascent`, `_ifs_test_ascent`, `_ifs_tendencies`). Cleared:
module-level param specs and physics contracts in the repo schema, verbatim Fortran literals
hoisted to named constants with provenance, `273.15` replaced by the canonical constant, and the
seven private symbols `_ifs_ascent` imported from `_ifs_test_ascent` promoted to public names.
5143 gate tests pass.

Separate, NOT introduced here: `tests/unit/test_bechtold.py` has 17 failures (NaN gradients in
the Bechtold grad tests, the IFS CAPE-closure and downdraft leaves). Identical list at
b636d812a in a clean worktree, so they predate this work; they need their own triage before the
faithful path is wired.

Next: the wiring design (closure + tendencies behind `BechtoldConfig.use_ifs_ascent`, default
False), with the first-guess contract above respected, then the day-110 replay and the 5-day
screen.

### Iteration I.2 — codex's oracle review finds two wrong numbers in the closure (2026-09-17)
Codex came back from its usage limit and reviewed the closure against the ACTUAL OpenIFS source,
which is vendored in this repo at `docs/references/openifs_arpifs/phys_ec` (gitignored; per
worktree). Every finding below was re-read at the quoted line before acting.

| item | source | port had | effect |
|---|---|---|---|
| ZORCPD | cumastrn:411 `1.0/RCPD` | `g/RCPD` | environmental stability in ZHEAT inflated by g; ZHEAT is the closure denominator, so deep mass flux suppressed ~10x |
| tendency pairs | :494-496 use PTENT/PTENQ, :757-767 use PTENTA/PTENQA | one pair for both | sub-cloud supply and the CAPE correction cannot both be right when other physics is active |
| NJKT2 | `DO JK=NJKT2,KLEV` (:490, :732), 60 hPa (sucumf:280-285) | whole column | stratospheric advection reaches ZDQCV |
| PWMEAN | cuascn:863 `MAX(1e-2, PWMEAN/MAX(1,ZDPMEAN))` | neither floor | wrong turnover time on weak columns, and sqrt at zero = infinite derivative |
| ZMFS floor | :975-979, inside the binding branch | every active column | a legitimately tiny ratio is raised to 1e-10 |

Two process notes. (1) The green gate run of Iteration I proved nothing for this module: codex
found the param-spec gate SKIPS it, because its float defaults are names rather than literals.
The spec is now keyed by the config class as the schema requires. (2) GLM's first NJKT2
implementation took the cutoff at the SURFACE end of the column, which collapsed every integral
onto the bottom layer and drove ZHEAT to its floor; caught by the new tests, fixed, and the
direction is now spelled out in a comment at both sites.

Tests: 15, all passing under `JAX_ENABLE_X64=1`, on a stretched grid (the uniform fixture could
not distinguish the two cloud-base interface pairs). Four of the new expectations were themselves
wrong and are corrected: the module lowers one global ZMFS instead of clamping per level; the
below-base taper is built from the UNSCALED input flux; and sum(M_u) is legitimately flat in T
whenever the RMFLIA bound rather than the CAPE closure sets ZMFS.

Retraction from Iteration I: the claim that violating the first-guess caller contract makes the
closure "saturate at ZMFMAX" is too strong. Codex's algebra: for a purely amplitude-scaled ascent
the factor cancels between ZHEAT and the final rescaling, so saturation only follows once a cap
binds or the ascent SHAPE changes. The contract still holds — the ascent must be launched at the
first guess — but for that reason, not this one.

Also settled today: `tests/unit/test_bechtold.py` is 98-passed under `JAX_ENABLE_X64=1` and
17-failed without it, identically on main. Those 17 are a precision-mode artefact of running a
numerics suite in float32, not branch damage. The residual question is real though: those
gradients are non-finite in float32, which matters if this scheme is ever trained there.

### Iteration J — the chain is wired, and it is NOT yet a faithful scheme (2026-09-17)
`BechtoldConfig.use_ifs_ascent` (default False, static python gate) routes the whole ported chain
in place of the legacy Bechtold reduction. It runs end to end. It is also, measurably, not right
yet, and the switch is now labelled INCOMPLETE in the config.

Two measurements and one review did the work:

1. First smoke run returned EXACTLY ZERO. Cause: the shallow cloud-base mass flux comes from
   ZDHPBL, the sub-cloud integral of the TOTAL physics tendencies, and cumastrn:578 switches the
   column off when that supply is not positive. Nothing supplies that pair here.
2. Threading what does exist (radiation + dynamics) woke it up — into a +-11000 K/day DIPOLE
   across the two lowest levels, at every resolution and timestep tried (30/60/120 levels,
   dt 112.5 and 1800 s), while everything above was O(100) K/day. The dipole's size matches the
   raw absolute static-energy flux differenced over one layer (28000 K/day), which is what a
   missing environmental subtraction looks like.
3. Codex's oracle review named it: the chain omits **CUFLXN**, which cumastrn runs between the
   closure and the tendencies (cumastrn.F90:1104 then :1226). cuflxn.F90:250-251 subtracts the
   environmental transport and :297-328 constructs the below-cloud-base heat and moisture fluxes.
   Without it the tendency module gets raw plume fluxes plus a tapered below-base mass flux whose
   heat/moisture fluxes were never set.

Also open from the same review, all P1: the humidity convention differs across the
trigger/ascent boundary (the trigger converts to specific humidity internally and returns
`q/(1-qu-lu)`); KTYPE is never reclassified against the ACTUAL ascent top
(cumastrn.F90:634-641), so a plume diagnosed deep but stopping shallow still gets the deep CAPE
closure; the detrained condensate and precipitation are dropped from the host water budget while
their latent heating is kept; the dynamics tendencies become de-facto mandatory on this path
though their public default is None; and the early return bypasses the legacy downdraught and
sub-cloud evaporation instead of keeping them, contradicting what the config comment claimed.

ARCHITECTURAL FINDING behind (1): this pipeline is PROCESS-SPLIT — every parameterization sees the
same input state and the tendencies are summed afterwards — and turbulence runs AFTER convection.
So there is no "physics accumulated so far" to hand the convection scheme, and the turbulent part,
which is the dominant sub-cloud supply for shallow convection, cannot be threaded without
reordering the pipeline. Radiation and dynamics are what exist at that call site and are what the
chain now receives.

Next, in order: port CUFLXN, fix the humidity convention at the trigger/ascent seam, reclassify
KTYPE after the ascent, return the condensate and precipitation to the host, and only then the
day-110 replay and the 5-day screen.

## Iteration K — the trigger/ascent seam, the KTYPE retype, and the first-guess contract (2026-09-17)

Three oracle defects closed on the faithful chain, each diagnosed by codex against the vendored
Fortran, coded by GLM, and reviewed by codex and GLM in turn.

**1. The seam was specific humidity all along.** The ported trigger applied `q/(1+q)` on entry and
`q/max(1-q-l, 0.5)` on exit (twice each, counting the refined wrapper). The oracle has neither:
PQEN, PQENH, PQU and PLU are one moist-mass basis end to end (cubasen.F90:72 and :62,
cuinin.F90:187 and :211, cubasen.F90:677-678, cuascn.F90:526/534/570-572/618). The two conversions
are exact inverses when there is no condensate, so the returned vapour was unchanged and the defect
was invisible from outside — it lived INSIDE, where the environment was 1.67% drier than the
saturation curve it was compared against. Correcting it deepens the deep fixtures by 30-70 hPa,
lowers their cloud base by 17 hPa and raises precipitation generation 30%; the trade-cumulus tops
do not move, because they are inversion-limited rather than humidity-limited.

Both column fixtures also built their humidity as a fraction of `saturation_mixing_ratio`. On a
specific-humidity API that is a 1.7% moist bias in the FIXTURE, and it makes the module-wrong and
fixture-wrong hypotheses produce identical diffs — GLM's finding. They now use
`saturation_specific_humidity`, which is genuinely `w_s/(1+w_s)`.

**2. KTYPE is now reclassified against the actual ascent top** (cumastrn.F90:635-641), between the
single ascent and the final closure, on the realised half-level pressure depth against RDEPTHS. The
source resets nothing else there and never runs a second ascent.

**3. The first guess and the rescale now divide by the same number.** ifs_closure rebuilds ZMFUB
internally and forms ZMFS = ZMFUB1/ZMFUB (cumastrn.F90:963), so it has to rebuild it with the
PRE-reclassification type — codex's P1. Fixing that exposed a second, older defect: the chain built
its ZDH from the FULL-level environment at cloud base while ifs_closure used the HALF-level one, so
the flux that launched the ascent and the flux the closure divided by were different quantities.
The oracle uses ZTENH/ZQENH (cumastrn.F90:570-571). Both sites now use the half-level environment,
and two tests pin the identity.

That last correction costs mass flux on the 30-level fixture: peak heating falls from just over 1 to
0.33 K/day. MEASURED, not inferred — on that column the retype is inert (realised depth 6.8 kPa
against a 20 kPa split, KTYPE 2 -> 2) and the rescale is ZMFS = 0.96, so the half-level ZDH is the
whole effect. The fixture has a sharp humidity step at 950 hPa, which is why the half-level humidity
at cloud base is 0.0129 against 0.017 at the full level.

REFUTED in review, both by reading the Fortran: ZDHPBL is NOT gated on KTYPE==2 (cumastrn.F90:493
gates on LDCUM and the sub-cloud level range only), and ITOPM2 is KCTOP, not KCTOP-2 (:637).

STILL OPEN on this path, in order: the detrained condensate and the convective precipitation are
dropped from the host water budget while their latent heating is kept (codex: the vapour sink is
already in PTENQ via the flux-divergence form, so the host loses sum(PLUDE) + surface precipitation
per unit time); the early return bypasses the legacy downdraught and sub-cloud evaporation; the
ported ascent does not return an updated LDCUM (CUASCN has it INOUT, cuascn.F90:389 and :627),
harmless only while there is no KTYPE=3 branch; plitot is zero, there is no convective momentum
transport, and ustar is a fixed 0.1 m/s.

ALSO OPEN, found by GLM while reviewing the seam fix: the trigger's sub-layer refinement does not
converge to the native-grid answer. On the deep sounding, L30 refined by 1/2/4 gives 895/557/523 hPa
against a native L60 top of 616 hPa, and the native-vs-refined gap widened from 25 to 59 hPa with
the seam fix. The gate test now documents that rather than certifying agreement.

## Iteration L — the host water budget, and the grid refinement (2026-09-18)

**The chain was destroying water.** The ported cudtdqn already removes the detrained
condensate and the convective precipitation from the vapour tendency (cudtdqn.F90:343-347), so
a host that receives neither loses that water while keeping its latent heating. Measured on the
active deep fixture: 0.051 mm/day on one weak column, which in a convecting tropical column is
the whole convective rain rate. Both are now returned, converted with the same expression the
tendency module builds its own ZDP from, and the column budget closes to 3 parts in 10 million.
On this chain — no downdraught, no sub-cloud evaporation, PSNDE = 0 — the surface rate equals
the sum of PDMFUP; when any of those lands, the rain tendency must become the per-level net.

**The sub-layer refinement had two defects.** Its half levels ran at fractions 1/r..1 per parent
with the surface appended again, which dropped the model top, displaced every interface by one
sub-layer and left a ZERO-THICKNESS bottom cell with its full level exactly at the surface. And
the reconstruction interpolated between parent FULL levels with the weight clipped to [0, 1],
while the sub-layers of a parent straddle its centre — so the shallower half were pinned to the
parent value and only the deeper half interpolated, toward the next layer down. On the last
parent it read out of bounds. Measured: up to 38% of a parent layer's water created or
destroyed, 1.25% of the column, and up to 31.5% error against the analytic profile with a mean
MOIST bias of +3.5e-4 kg/kg.

Now a Δp-conservative piecewise-linear finite-volume reconstruction with a minmod limiter.
Per-parent water conserved to 2e-7, profile error 2.3% at r = 2, mean bias down to +1.0e-4.

**The spuriously moist environment was driving spuriously vigorous convection.** Controlled A/B
on the reconstruction alone: the trade column keeps its mass-flux profile and cloud top exactly
but loses 22% of its kinetic energy and, rising more slowly through a drier cloud layer, carries
16% more condensate and rains 37% more; the deep column's ascent is shallower (760 -> 827 hPa)
with 28% less mass flux, 43% less kinetic energy and 32% less rain. The refined cloud top moves
557 -> 658 hPa at r = 2 against a native L60 answer of 616, so the refined-native gap falls from
59 to 42 hPa.

**Owner decision (2026-09-18): keep the layer-average semantics, consistently.** A level value is
the mean over its layer, located at the layer centre derived from the half levels, never the
caller's full-level pressure. `_layer_centre` is now the single expression for that, used by both
arms of the refinement wrapper — the r = 1 shortcut previously forwarded the caller's p_full, so
the anchoring was a function of r. The geopotential gets the same treatment, because forwarding
the caller's while re-anchoring the pressure reintroduces the same sub-layer shift. The
source-literal search is unchanged; its saturation calls divide by PAPRSF as the source does.
Codex refuted an audit claim here: the NJKT1/NJKT2 bounds and the optional cell-centre
mixed-layer gate are PORT-LOCAL, not source-literal, and already receive layer centres through
the wrapper.

**OPEN, cause unknown.** Neither ladder converges — L30 refined r = 1/2/4/8 gives
895/658/625/591 hPa, native L30/L60/L120 gives 895/616/536 — and the refined-native gap GROWS
with effective resolution, 42 hPa at 60 levels and 89 at 120. A resolution-sensitive departure
search and the point-sample-versus-layer-mean semantics of the fixtures are both candidates and
neither has been discriminated. The gate test records this instead of certifying agreement.

## Iteration M — the surface-layer reference height, and a controlled pair

The evaporation deficit turned out to sit on a pending decision rather than on
physics nobody had looked at. Commit da7cca173 (2026-09-15, "opt-in ocean
surface-layer corrections — real input height, sea-water q_sfc") added two
switches and left both off, its own message saying "both default False (user
decision pending)". No committed configuration has set either since.

What the height switch does. On the MPAS path the similarity solver is handed
the lowest full level's wind, temperature and humidity, but with
`surface_z_ref_model_level` False it is told those values came from 10 m. They
come from about 147 m. Calling the model's own flux routine twice on identical
soundings — `scripts/validate/amip_bias/zref_height_factor.py` — the
mislabelling inflates latent heat by 1.13 to 1.30 depending on regime and
surface stress by 1.16 to 1.54. A neutral-limit estimate of 1.5 was too high;
stability corrections eat about a fifth of it.

The controlled pair. Two five-day branches from the pinned day-105 state, one
variable, both corrections against neither, both with the process ledger on.
Global, area-weighted, kg/m2/day at day 110:

                       off      on    change
  evaporation        2.278   2.008     -11.9%
  convective rain    2.014   1.782     -11.5%
  stratiform rain    0.795   0.804      +1.1%
  transport          0.145   0.136
  column drying      0.840   0.961

Tropical ocean surface layer at the same day: specific humidity 19.48 against
18.70 g/kg, so the corrected arm closes roughly a third of the excess over
ERA5's 17.2 in five days, and the air cools by 1.03 K.

How to read it. The immediate evaporation penalty was 24% on day one and 12%
by day five, so half of it was already recovered as the surface layer dried —
which is the signature GLM's review predicted if the inflated coefficient was
CAUSING the moist bias rather than merely offsetting it. But five days is the
transient, not the answer: precipitation moves from 95% to 87% of the
reference and relative humidity gets WORSE, 0.877 to 0.897, because the
cooling outpaces the drying. Whether the arm recovers depends on whether the
drying continues at 0.12 kg/m2/day for the thirty-odd days it would take to
remove a bias of 4 to 8 mm, and this pair cannot say.

Open, and needing a decision: a thirty-day pair is the only thing that settles
whether these corrections are net-positive. The cold drift of about 1 K in
five days is the risk to watch in it.

## Iteration N — where the convective rain's vapour is debited (2026-09-21)

The claim, reviewed before code (codex read cudtdqn/cuascn/cuflxn; GLM from
knowledge): the Bechtold port debited the vapour that becomes in-plume rain,
and released its latent heat, in proportion to each level's vapour mass, so
43.5% of the ITCZ's convective rain water (1.60 of 3.68 kg/m2/day, day 135
banded ledgers) was taken from, and its heat put into, the layers below
sigma 0.83 where the plume never condensed; ~0.8/day and ~30 W/m2 (2-3 K/day)
landed below cloud base. IFS assembles the sink where the rain forms. Both
reviewers confirmed the mechanism against the source.

Fix (`BechtoldConfig.rain_vapor_sink`, default "formation"; "vapour_mass"
keeps the legacy spread for the A/B; `--bechtold-rain-vapor-sink`): the debit
and its heating follow the rain-formation profile (precip_frac * M_u), capped
per level at 0.9 of the post-transport vapour, excess redistributed within the
formation support, any remainder reducing the rain and heating together so
rain == sink holds exactly per column. The user moved the production default
in the same change. Five review rounds (codex + GLM each): NaN float32
gradients from 1e-30 denominators (fixed, non-vacuous test), a physical flux
floor for the masks, sub-floor columns emit no rain and debit nothing, and
the offline probe records the helper's cap scale directly (0 of 9401 formation
levels binding on day 135: the cap does not act in production).

Pre-registered 1-day arm (wv_sfcon_rvs_* vs wv_sfcon_led_*, day 135->136,
ITCZ ocean): q850 ratio 0.71 -> 0.78, T850 bias +1.54 -> -0.04 K, sub-cloud
convective term -0.86 -> +0.01 (profile side CONFIRMED); convective rain
3.69 -> 1.89 (-49%, outside the +-25% band). The rain halving rides on the
carried relaxed mass flux: swapping the control's memory into the arm's state
gives 5.15, its T+q gives 0.74.

Five-day pair (wv_sfcon_rvs5 vs wv_sfcon_leg5, days 136-140, one variable):
evaporation transient CONFIRMED (global hfls 56.5 -> 60.4 vs legacy 60.3);
q850 ratio 0.71 -> 0.83, T850 +1.47 -> -0.58 K, q925 0.87 -> 0.97; the
mid-level bulge shrinks (600 hPa 1.04 -> 0.94, 500 hPa 1.38 -> 1.12,
PLAUSIBLE until repeated). Convective rain INCONCLUSIVE by the
pre-registration (day-140 ratio 0.63, between REFUTE 0.60 and CONFIRM 0.75,
recovering slowly from 0.51). Pair scores, 5-day means: tropics pr RMSE
6.50 -> 4.63 mm/d (-29%), bias +0.30 -> -0.38; prw RMSE 10.89 -> 9.98;
global pr RMSE 4.29 -> 3.35. Figures: qprofile_rvs5_vs_leg5.png,
pair5_pr_hfls_timeseries.png, ledger_layers_wv_sfcon_rvs.png.

Open after this iteration: the ITCZ convective rain settles ~35% below the
legacy scheme, so whether the closure (rprcon / CAPE relaxation) was tuned
against the misplaced heating is the next user decision; pre-existing 1e-30
divisors in the downdraft evaporation and rescale (NaN gradients at zero
rain, codex) are not touched here; the 30-day continuation is not run.
