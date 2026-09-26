# Cloud-cover / invisible-ice A/B — PR #1714 pre-registration (2026-09-05)

Extracted from `scripts/cluster/levante/submit_cloud_cover_arms.sh` as it stood on
`main` before the 2026-09-08 launcher replaced it (see the merge commit on
PR #1724). The script was superseded; **this record was not**, and a merge
commit message is not somewhere anyone finds an experiment's registration six
months later. Recovered on GLM's review of that merge.

```

Cloud-cover / invisible-ice A/B: 5-day paired arms (4 + a control twin) off the res6 AMIP
run's pinned day-80 state (PR #1714 pre-registration, 2026-09-05).

Measured on rhebc90_r6 checkpoints with scripts/validate/amip_bias/
cloud_layers.py (offline, CONFIRMED):
  * 90 % of the prognostic cloud-ice mass sits in layers the sundqvist cover
    scheme calls clear, because RH is measured against LIQUID saturation; all
    of it is ice-saturated (RH_ice >= 0.85), 95 % at RH_ice >= 1.
  * two redundant gates then remove it from the radiation: the fsd=1
    two_region factor on a cf-floored in-cloud tau, and the cf=0 subcolumn
    mask.  Solver input: 1.9 of 32.6 g/m2.  The pre-#1519 optics (overlap
    none, chi=1) saw all of it.
  * mixed_phase saturation alone makes the ice visible but the ITCZ
    overcast aloft (high cover 8 -> 99.5 %); xu_randall (condensate-aware
    cover) with liquid saturation makes 68 % of the ice visible at 27 % ITCZ
    high cover (offline snapshot numbers; the arms measure the response).

Controlled comparison: every arm starts from the SAME pinned checkpoint,
COPIED into its own directory (never a newest-wins pointer), the params
file copied likewise, byte-identical EXTRA except the single variable
under test, same absolute TARGET_DAYS.  Score with
  scripts/validate/amip_bias/window_diff.py --ctl cld_ctl --d0 80 --d1 85 cld_*
and scripts/validate/amip_bias/cloud_layers.py <arm> --days 85.
set -euo pipefail

ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}
SRC="${ROOT}/rhebc90_r6"
PIN="${SRC}/checkpoint_day_0080.npz"
PIN_ACCUM="${SRC}/cmor_accum_day_0080.npz"
PARAMS="${REPO}/config/amip/params/w7_eps_hi_tuning.yaml"   # what rhebc90_r6 ran
Absolute 80 (pinned) + 6.  The scored window is days 81-85; the run must go
ONE day past it because the driver deletes the sidecar of the FINAL day at
clean finalization (model_driver: cmor_accum_day_<final>.npz unlinked once
the NetCDF is written), so a run ending at 85 would leave window_diff.py
nothing to read (codex review).  Day 85's sidecar is then an ordinary
per-checkpoint sidecar and survives.
TARGET_DAYS=86

```
