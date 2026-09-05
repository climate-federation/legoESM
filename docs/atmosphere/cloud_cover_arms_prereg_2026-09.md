# Cloud-cover / invisible-ice arms — pre-registration (2026-09-05)

Launcher: `scripts/cluster/levante/submit_cloud_cover_arms.sh`.
Scorers: `scripts/validate/amip_bias/window_diff.py --ctl cld_ctl --d0 80 --d1 85 cld_mixed cld_xurandall cld_pre1519`
and `scripts/validate/amip_bias/cloud_layers.py <arm> --days 81,82,83,84,85`.

Base: `rhebc90_r6` (MPAS res6, production config, w7_eps_hi params), restart
from the pinned `checkpoint_day_0080` (copied into each arm), run to absolute
day 85. Window days 81–85 (March, one accumulator bucket). Few-day arms are
sufficient for cloud responses (user directive 2026-09-05); this set ranks
levers, it does not tune them.

Offline snapshot (day 80) numbers each arm was chosen on — the arms measure
the coupled response, these are the priors:

| arm | change (one variable) | ice reaching solver g/m² | global high cover % | ITCZ high % |
|---|---|---|---|---|
| cld_ctl | none | 2.1 | 3.3 | 8 |
| cld_mixed | `--cloud-saturation-scheme mixed_phase` | 30.6 | 69 | 99.5 |
| cld_xurandall | `--clouds xu_randall` (condensate-aware cover, liquid saturation) | 22.0 | 11 | 27 |
| cld_pre1519 | overlap none + constant chi=1 (pre-#1519 optics) | 32.6 | 3.3 | 8 |

Control absolute (days 71–80 window, `window_diff`): rsut 114, rlut 233,
CRE_LW 16.5 W/m² global (CERES ≈ 27), ITCZ CRE_LW 30.

## Confirms / refutes (arm − ctl, days 81–85, paired window)

- **Ice is radiatively invisible (mechanism).** cld_pre1519 must move rlut
  by more than −5 W/m² globally (more high-cloud trapping) with cover
  unchanged (|Δclt| < 1 %). Refuted if |Δrlut| < 2: the "deleted ice" then
  has no radiative weight and the #1519 attribution question is closed the
  other way.
- **cld_mixed.** Expect the ice visible and the LW cloud effect recovering
  (ΔCRE_LW > +6 global), at the cost of a large SW brightening (Δrsut
  > +20) from overcast anvils. It CONFIRMS "cover blind to ice is the root"
  and simultaneously REFUTES mixed_phase as a stand-alone fix if Δrsut > +15
  with ITCZ clt → > 95 %.
- **cld_xurandall.** The candidate lever: ΔCRE_LW > +4 with Δrsut within
  ±8 and ITCZ clt not above 90 %. If it delivers the LW recovery without
  the SW blow-up it becomes the next production A/B (30 days). If Δrsut >
  +15 it is the same overcast failure by another route.
- **Polar cover.** cld_mixed polar clt +15 % or more (offline prior 43 → 64
  low cover) confirms the −41 % polar deficit is largely the same defect.
- Any arm that NaNs or whose ctl window differs from rhebc90_r6's own days
  71–80 by more than 3 W/m² in rsut is a code-drift finding first, a physics
  result second.

Cost: 4 jobs × ~2.5 GPU-h (res6 ≈ 2.3 sim-days/h).

Out of scope here (needs its own arms): the near-surface humidity bias that
drives the shallow marine cover (sigma 0.92–0.98), the missing BL-top /
shallow-cumulus cover mechanism, the #1715 N_c unit inconsistency.
