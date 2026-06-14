# Scope — EKE (eddy kinetic energy) closure

> Status: **scoping only** (not implemented). A confirmed must-build block for Veros ACC
> apples-to-apples: `enable_eke=True` in `veros/setups/acc/acc.py`, and legoESM has no EKE
> module (`grep -ri eke src/legoesm/ocean` finds only fidelity-side mentions). Per doctrine
> rule H + the apples-to-apples rule, ACC needs it because Veros runs ACC *with* it.

## What it is

Eden & Greatbatch (2008) parameterize mesoscale eddies via a **prognostic eddy-kinetic-energy
field** `E` (one scalar per cell, like a tracer) whose budget is

```
∂E/∂t = ∇·(K_iso ∇E)        # isopycnal diffusion of EKE
        + advection(E)        # by the resolved + eddy flow (Veros: superbee)
        + P                   # production: the GM/Redi eddy buoyancy flux * N²-ish source
        − ε                   # dissipation: c_eps * E^{3/2} / L  (L = mixing length, ≥ lmin)
```

and the **GM coefficient becomes prognostic**: `kappa_GM = c_k · L · sqrt(E)` (a
Visbeck-style relation) instead of a constant. Veros ACC settings: `eke_c_k=0.4`,
`eke_c_eps=0.5`, `eke_lmin=100 m`, with superbee advection + isopycnal diffusion of E enabled.

This is a bigger block than flux-form momentum: it adds a **new prognostic state variable** and
**couples back into GM/Redi** (which currently takes a constant `kappa_GM`).

## What to reuse

- **Advection of E**: the existing tracer advection dispatch (superbee/TVD/etc. in
  `ocean/advection.py`) — E advects like a tracer.
- **Isopycnal diffusion of E**: the GM/Redi isopycnal-diffusion machinery
  (`gm_redi_*` / `_gm_redi_common`) — E diffuses along isopycnals like a tracer.
- **Visbeck coefficient**: `compute_visbeck_kappa_gm` in `_gm_redi_common.py` already computes
  an EKE-style length scale / coefficient — the EKE prognostic closure generalizes the source
  that feeds it. (Today `GMRediConfig.visbeck` is the closest existing hook.)
- **Vertical structure / N²**: `compute_buoyancy_frequency` (eos), already used by Richardson.

## What is genuinely new

1. **A new state field** `eke` on the ocean state pytrees (lat-lon C-grid + MPAS), threaded
   through init, the step loop, restart I/O, and the channel-packing/diagnostics — this is the
   cross-cutting part (every `SegmentCarry`/state constructor + I/O path).
2. **`ocean/physics/lateral_mixing/eke.py`** — the Eden-Greatbatch prognostic tendency
   (production from the GM flux, dissipation `c_eps·E^{3/2}/L`, mixing length with `lmin` floor)
   + the `kappa_GM = c_k·L·√E` diagnosis.
3. **`EKEConfig`** NamedTuple (`c_k`, `c_eps`, `l_min`, advection scheme, iso-diffusion κ) + a
   dispatch in GM/Redi so `kappa_GM` can be **eke-prognostic** instead of constant
   (`raise ValueError` on unknown literal, per dispatch discipline).
4. Coupling: the GM/Redi tendency reads the prognostic `kappa_GM`; the step integrates `eke`
   alongside T/S (positivity-preserving — E ≥ 0).

## Truth-tier tests required (gate — BEYOND oracle-matching)

- **Tier 0**: EKE budget closure (∫∂E/∂t = ∫production − ∫dissipation + boundary fluxes to
  machine ε in a closed domain); **positivity** (E stays ≥ 0 under the integrator).
- **Tier 1**: equivariance of the E budget under the bridge transforms; differentiability
  (`jax.grad` through the prognostic E + kappa_GM coupling, finite/nonzero).
- **Tier 2**: an idealized baroclinic channel (Eady/ACC-like) where eddy energy spins up to a
  steady level; check kappa_GM responds to E and the eddy buoyancy flux is down-gradient.
- **Tier 3** (last): ACC tier-2 comparison with EKE active vs Veros's E field + kappa_GM.

## Risks / magnitude

- **Cross-cutting state change** is the hard part (new field on every state/carry/I/O path; the
  `SegmentCarry` discipline in CLAUDE.md applies — update every constructor + the compiled-
  segment tests).
- Positivity under advection+dissipation (E must not go negative — clip-free if possible, or a
  justified floor).
- Coupling stability: prognostic kappa_GM can feed back; needs the idealized-channel test to
  confirm it doesn't run away.
- **Rough size: ~800–1200 LOC** (state plumbing ~300, closure module ~300, config+dispatch ~50,
  tests ~300–500). Larger than flux-form momentum; its own focused PR. **Recommend building
  flux-form momentum first** (smaller, self-contained, drives the visible `du_adv` delta), then
  EKE.
