# Cloud-cover / invisible-ice arms — pre-registration (2026-09-05)

Launcher: `scripts/cluster/levante/submit_cloud_cover_arms.sh` (5 arms; `SPLIT_D=1` adds two).
Scorers: `scripts/validate/amip_bias/window_diff.py --ctl cld_ctl --d0 80 --d1 85 cld_ctl_twin cld_mixed cld_xurandall cld_pre1519`
and `scripts/validate/amip_bias/cloud_layers.py <arm> --days 81,82,83,84,85`.

Base: `rhebc90_r6` (MPAS res6, production config, w7_eps_hi params), restart
from the pinned `checkpoint_day_0080` + its accumulator sidecar (copied into
each arm), run to absolute day 86 so the day-85 sidecar is not the terminal
one the driver deletes. Scored window: days 81–85 (March, one accumulator
bucket). Subcolumn sampling is a deterministic stratified table (no RNG), so
restarts pair exactly.

**What 5 days measures.** TOA fluxes respond instantly; upper-troposphere
T/q relax over 15–30 days. The window therefore measures the radiative
FORCING of each change on a near-identical state, not the adjusted climate
(GLM review). That is what ranks levers; adjusted values need the 30-day
follow-up on the winner.

Offline day-80 priors each arm was chosen on (ice reaching the solver g/m²,
global high cover %, ITCZ high cover %):

| arm | change (one flag) | ice→solver | high % | ITCZ high % |
|---|---|---|---|---|
| cld_ctl | none | 2.1 | 3.3 | 8 |
| cld_ctl_twin | none (identical twin) | 2.1 | 3.3 | 8 |
| cld_mixed | `--cloud-saturation-scheme mixed_phase` | 30.6 | 69 | 99.5 |
| cld_xurandall | `--clouds xu_randall` (condensate-aware cover; liquid saturation) | 22.0 | 11 | 27 |
| cld_pre1519 | overlap none + constant chi=1 (two optics switches, see launcher) | 32.6 | 3.3 | 8 |

xu_randall's ice visibility is set by its `alpha_xr` (default 100): alpha/10
→ 0.2 g/m², high 1.5 %; alpha×10 → 30.2 g/m², high 41 %. The arm is one
point on a tuning axis, not a physical prediction of 27 % (GLM discriminator,
measured offline).

Control absolute (rhebc90_r6 days 71–80, `window_diff`): rsut 114, rlut 233,
CRE_LW 16.5 W/m² global (CERES ≈ 27, deficit ≈ −10.5), ITCZ CRE_LW 30.

## Noise floor first
`cld_ctl_twin − cld_ctl` must be 0.00 in every field (bit-restart). Anything
else voids the pairing of every arm and is the first finding. There is no
separate drift guard against rhebc90_r6's own earlier window: the equinox
insolation trend (~0.3 W/m²/day) is the size of any sensible tolerance.
Paired 5-day σ is NOT known a priori; GLM's plausible values (global rlut
1–1.5, rsut 1.5–2.5; ITCZ 5° box rlut 8–15; polar clt 5–10 %) are the
scale the thresholds below are read against, not measured.

## Measurements and tests (arm − ctl, days 81–85)

- **cld_pre1519 — a measurement, not a test.** Cover is flag-invariant, so
  |Δclt| < 1 cannot fail, and 30 g/m² of newly visible ice (τ_LW ≈ 2) makes
  Δrlut ≪ −5 near-certain (GLM). The number reported is the radiative weight
  of the hidden ice: Δrlut, Δrsut, ΔCRE_LW global and ITCZ. It closes the
  open question of how much of #1519's −22.7 W/m² rsut was deleted ice.
- **cld_mixed — a bound.** ITCZ high cover is 99.5 % before any dynamics, so
  it cannot pass as a fix; it bounds what "cover sees ice" does to CRE_LW and
  rsut. Report ΔCRE_LW, Δrsut; if Δrsut > +15 with ITCZ clt > 95 % the
  stand-alone lever is refuted (expected).
- **cld_xurandall — the candidate lever.** Existence: ΔCRE_LW > +4 global
  (≈3σ on GLM's scale). **Adequacy** against the −10.5 deficit: ΔCRE_LW ≥ +8
  with Δrsut within ±8 global and ITCZ clt < 90 %. Existence without
  adequacy → the alpha axis is the next sweep (offline first: alpha 200,
  400), not another 5-day set.
- **Polar cover.** cld_mixed polar clt ≥ +15 % (offline prior 43 → 64 low
  cover) is ~1.5–2σ on GLM's scale: read as supporting, not confirming, that
  the −41 % polar deficit is the same defect. The checkpoint-based
  `cloud_layers` cover (not the 5° TOA field) is the stronger statistic here.
- Any NaN, or a twin difference ≠ 0, is a code finding first.

Cost: 5 jobs × ~2.6 GPU-h (res6 ≈ 2.3 sim-days/h); +2 with `SPLIT_D=1`.

Out of scope here (own arms later): the near-surface humidity bias that
drives the shallow marine cover (sigma 0.92–0.98); the missing BL-top /
shallow-cumulus cover mechanism; #1715 (N_c units).
