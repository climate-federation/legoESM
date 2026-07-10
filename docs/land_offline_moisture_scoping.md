# Scoping: fixing the offline soil-moisture / summer-warm bias

## Problem (diagnosed 2026-07-07)

Offline ERA5-forced `multilayer_land` runs a **too-dry root zone** in wet-season
mid-latitudes, driving a **summer warm bias** (North America JJA **+1.0 K**, model
root-zone moisture **0.21** vs ERA5 **0.31** m³/m³). Ruled out, with tests:

- **Bare-soil evaporation** — strengthening the resistance *warms* (removes latent
  cooling) and barely wets (0.215→0.219 across exp 2→10). Not the cause.
- **Bottom boundary condition** — free-drainage vs zero-flux give **identical** 0.209.
  The root zone doesn't drain out the deep (6.375 m) bottom; **CLM5's aquifer/water-table
  would not help.**
- **Soil retention** — porosity 0.42, fc 0.167 = textbook van-Genuchten **loam field
  capacity at −33 kPa**; same as CLM5/HTESSEL. Not a parameter error.

**Root cause = the offline forcing framework, not a parameter.** ERA5's 0.31 sits *above*
field capacity — real soil stays wet from **seasonal replenishment** (spring snowmelt
saturates it; it dries slowly through summer). The trainer's **Stage-A equilibration under
ANNUAL-MEAN forcing drains straight to field capacity (0.21)** and cannot hold water above
fc without the seasonal forcing that puts it there. The **partial moisture freeze** (deep
layers pinned to the Stage-A equilibrium, only the root zone evolves in Stage B) — added
for numerical stability (fully-evolving moisture drains thin cells to wilting → ET collapse
→ +50 °C runaway) — locks in that dry equilibrium.

## Why the offline mean ≠ the seasonal reality

`swvl_annual_mean(ERA5) = mean(wet spring ~0.40, dry summer ~0.20) ≈ 0.31`, whereas the
annual-mean *forcing* produces a single steady state at the drainage floor (fc ≈ 0.21). The
nonlinearity (snowmelt pulse → slow drawdown) is lost when precip is time-averaged first.

## Options

### A. Seasonal Stage-A spin-up (medium lift, medium risk)
Replace the constant annual-mean Stage-A equilibration with a **multi-year seasonal
spin-up** on the monthly forcing (the same `fstack` Stage B already uses), so the spring
snowmelt pulse saturates the soil and the summer starts wet.
- **Cost**: Stage A becomes `n_spinup_years × 12 × SPM` steps instead of `_EQ_STEPS` at one
  forcing — the AD scan (training) grows ~2–5×; checkpoint the spin-up (`jax.checkpoint`)
  to bound backward memory.
- **Risk**: reintroduces the drain-to-wilting instability the partial-freeze avoided. Needs
  a stability treatment first (see D). Without it, thin/sandy cells still blow up.
- **Payoff**: directly targets the mechanism — the one change that should move these biases.

### B. Un-pin the Stage-B deep moisture (small lift, high risk)
Let all layers evolve in Stage B instead of pinning the deep reservoir.
- **Cost**: small code change (drop the `deep`/`deep_psi` pin at lines ~450).
- **Risk**: HIGH — this is exactly what caused the +50 °C runaway; do NOT do it without D.

### C. Coupled / interactive forcing (large lift, low risk, definitive)
Run the land in the coupled ESM (or with sub-daily real forcing), so precip timing,
snowmelt, and the atmosphere are interactive. This is the standing conclusion for *every*
offline bias this session (skin-T, albedo, moisture): the offline annual-mean framework is
the shared limiter. Highest fidelity, biggest effort; use `run_amip --use-multilayer-land`
(the differentiable land-in-coupler path already exists, PR #650).

### D. Stability prerequisite for A/B — implicit / bounded Richards (small–medium lift)
The instability is an explicit-drainage stiffness (thin top layer + high K). Fixes that
un-block seasonal moisture: (1) an adaptive/sub-stepped or **implicit** Richards drainage
(the CFL cap at richards.py:362 is a band-aid), (2) a mass-conserving moisture floor at
`theta_r` with the deficit carried by the ponding cell, (3) hydraulic-redistribution or a
capillary-rise term so a dry top layer refills from below overnight.

## Recommendation

**D → A.** First make the Richards drainage unconditionally stable (implicit/bounded, D),
then switch Stage-A to a seasonal spin-up (A) and let Stage-B moisture evolve. That removes
the annual-mean equilibrium that pins the soil to field capacity, which is the actual cause.
Validate against the ERA5 `swvl1/swvl2` target (now loadable via `fetch_era5_chunked`,
`lam_sm` active) and the N-America JJA skin-T bias. C (coupled) remains the gold-standard
cross-check. Do NOT prescribe ERA5 moisture (masks the bug) and do NOT tune bare-soil
evaporation or the bottom BC for this — proven ineffective above.

Effort estimate: D ≈ 1–2 focused sessions (implicit tridiagonal drainage already exists in
`timestepping/tridiagonal.py`); A ≈ 1 session once D lands; both need the codex review +
the RCE/stability gates.
