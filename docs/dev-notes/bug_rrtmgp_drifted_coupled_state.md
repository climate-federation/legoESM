# Known limitation: rrtmgp TOA fluxes corrupt on the drifted coarse-coupled state

**Status (2026-06-15):** OPEN. Tracked follow-up. NOT a regression — pre-existing.
The model does **not** crash; SST/`tas` stay physical and there is no NaN. The
corruption is in the **radiation diagnostics** (TOA fluxes) and is **silent**.

## Symptom

A coarse (C18/L20, dt=450 s) fully-coupled run with the full-physics default
(`radiation=rrtmgp`, `clouds=sundqvist`, convection/turbulence/GWD/microphysics
on) produces **unphysical TOA radiation fluxes after ~day 15**:

- `rsdt` (TOA incident SW) area-mean 330 → ~360-420, with cells up to **1072
  W/m²** — above *any* possible solar flux (daily-mean max ~490, instantaneous
  max ~`S_0`≈1361·cosθ).
- `rsut` (TOA upwelling SW) → ~320 (planetary albedo ~0.8, unphysical).
- `rlut` (OLR) → ~0, with cells **negative** (down to −51 W/m²). Negative OLR
  is impossible.

The **held radiation field itself** is the garbage (it equals
`PhysicsPipeline.compute_radiation_core`'s output), so the radiation *solver*
produces these values — it is NOT a diagnostic/regrid/accumulator artifact.

## Time evolution (job 8489297, diag_days=10 → held printed each segment)

```
held@day10: sw_down_toa mean 331.8 (max 488.7)   lw_up_toa 304 (>0)    CLEAN
held@day20: sw_down_toa mean 698.5 (max 1072.5)  lw_up_toa -28.7        CORRUPT
held@day30: sw_down_toa mean 417.8 (max 576.1)   lw_up_toa 10.5 (min -10.2)
```

The whole SW field inflates ~2× roughly uniformly (so the flux *divergence* /
heating rate stays ~0 — which is why the model stays "stable"). It develops
between day 10 and 20 and oscillates after.

## What it is NOT (ruled out, with evidence)

- **Microphysics water-trap.** `--microphysics kessler` (q_c bounded, pr>0) AND
  `none` both corrupt at 30 days. (The microphysics fix DID fix `pr=0` — a
  separate, real bug; see commit `56e0e436`.)
- **Thick clouds.** A q_c sweep through `compute_radiation_core` (up to
  q_c=2e-2) gives `rsdt` *low* and `rlut` *positive* — the OPPOSITE of the run.
- **Cold/humid drift (uniform).** A cold+humid sweep (T_sfc 295→240, T_top
  220→160, q_v up to 4e-2) is clean down to T_top=160 K.
- **Diagnostic / regrid / accumulator path.** The held field (pre-regrid) is
  already garbage.
- **Long single segment.** `diag_days=10` (3 segments) corrupts too — it is
  state-dependent, not segment-length-dependent.

## What it IS

The radiation solver fails on the **evolved real coupled state** at ~day 15-20.
The trigger is a *specific* feature of that state — a localized extreme cell, a
vertical structure (inversion / near-isothermal layer), cloud ice, or an
unclamped negative tracer — that smooth synthetic states do not reproduce.
Note `rrtmgp.py` clamps `p`, `q_v` (≤0.99), `o3`, cloud paths, `cos_zenith` —
but does **not** clamp layer temperature to the gas-optics valid range, a
likely-missing guard worth checking first.

## Reproduce

```
sbatch scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_validate.sbatch
# (or any run_coupled.py rrtmgp run with --days 30) and read the CMOR rlut:
#   negative cells = corrupt.  A 2-day run is clean.
```

## Recommended fix (deliberate, not rushed)

1. Instrument a run to **dump the day-15-20 state arrays** (T, q_v, q_c, q_i,
   cloud fraction, r_eff, p) at the cell where `sw_down_toa` is max / `lw_up_toa`
   is negative.
2. **Bisect** which field/cell drives `compute_radiation_core` super-physical
   (feed the dumped state back into the offline probe — it WILL reproduce once
   the real state is used, since the held field proves the solver produces it).
3. Add the **missing input clamp** in the rrtmgp wrapper (candidates: layer T to
   the k-table range, a negative-tracer floor, an SSA/asymmetry bound in the
   two-stream) and/or fix the upstream cloud-optics that feeds it.
4. Validate with a **clean 30-day run** (no negative `rlut`, `rsdt`≈340).

## Workaround today

Short runs (< ~10-15 days) are clean. For longer runs use gray radiation
(`--radiation gray`, no cloud-radiation coupling) until this is fixed.

See `[[cmip6_coupled_infra_session]]`.
