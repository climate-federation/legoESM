# MITgcm barotropic-gyre oracle: the residual is the free-surface SPLIT, not the momentum scheme

**Status (2026-06-17, revised):** root cause *localized by direct measurement* to the
**barotropic free-surface predictor–corrector projection** in legoESM's split solver. An
earlier version of this note (commit `5ba07e726`) attributed the residual to a non-energy-
conserving *momentum* scheme and prescribed an Arakawa/Sadourny Coriolis + enstrophy-conserving
advection. **That prescription is wrong** and is corrected below: the Coriolis is *already*
energy-neutral, the advection is provably irrelevant to the source, and MITgcm itself uses a
non-energy-conserving flux-form momentum scheme yet stays laminar. The real source is the
operator-split free surface.

## The gap (measured on-box, both models)

legoESM reproduces MITgcm `tutorial_barotropic_gyre` to **0.9997 eta pattern correlation at 10
steps**, but the multi-year equilibria diverge:

| | legoESM (recipe) | MITgcm (75 000 steps on-box) |
|---|---|---|
| equilibrium `|u|max` | turbulent, **0.15–0.37** | laminar, **0.031** (peak 0.033) |
| spin-up | overshoots ~0.19yr, goes eddying | monotone to 0.031 by 0.38yr, holds 2.85 yr |

MITgcm's own per-step monitor (`out_eq.txt`, `monitorFreq=1`) shows `dynstat_uvel_max` rising
*monotonically* to ~0.030 and never exceeding **0.033** over the whole 2.85-yr run — perfectly
laminar, no overshoot. legoESM at the same wall-clock is already 0.064 → 0.107 → turbulent.

## What was ELIMINATED this session (all by direct measurement)

| candidate | test | verdict |
|---|---|---|
| **Wind forcing** | legoESM applied stress vs `windx_cosy.bin` | bit-exact (max diff 3.7e-9) — not it |
| **Advection stencil** | legoESM `centered` flux vs MITgcm `mom_u_adv_uu.F` | identical `0.25(Qw+Qe)(uw+ue)` |
| **Advection scheme** | spin-up under upwind / centered / vector-invariant | faithful `centered` is *worse* (0.37); all over-energize |
| **Dissipation deficit** | KE budget at rough & laminar states | dissipation is *large* (`D=-1.1e9`, exceeds wind input); balances wind exactly at the laminar state |
| **Coriolis** | instantaneous power `∫u·(f k×u)` (uniform grid → exact) | **machine-zero** (`|P|/(f·KE)=4.5e-8`); legoESM's Sadourny Coriolis already conserves energy |
| **Centered advection** | instantaneous power `∫u·adv` | **machine-zero** (`P=1e-13`); legoESM's centered flux-form is energy-neutral |
| **Time integration** | inviscid `E(T)/E0` vs `dt` (1200→150 s) | growth **converges** to +20.5% as `dt→0` → spatial, not a time-truncation error |

## The source (measured): the free-surface split projection

A **total mechanical-energy** budget (`E = KE + ½g∫η²`; PE is only ~0.1 % of KE here) shows a
genuine spurious source under inviscid + unforced dynamics:

- at the **rough/turbulent** state: `dE/dt > 0`, inviscid-unforced growth **+20.5 %/5.6 days**
  (`S/W ≈ 3.2`, i.e. the dynamics inject ~3× the wind work);
- at MITgcm's **smooth laminar** state: `S ≈ 0` (`|S|/|W| = 0.06`) and legoESM is *stable*.

So the source is **gradient-/grid-scale-activated** — negligible on smooth fields, dominant once
grid-scale structure appears. Spin-up from rest excites those modes and legoESM runs to a
turbulent attractor while MITgcm stays laminar.

Localization, step 1 — advection-independent: the semi-discrete source `dE/dt` is **byte-identical**
(`+5.2518e5`, 6 sig figs) under **upwind, centered, and vector-invariant** momentum advection.
Advection contributes *exactly nothing*.

Localization, step 2 — it is the **Coriolis ⟷ implicit-free-surface coupling**. Re-running the
inviscid + unforced semi-discrete `dE/dt` with **`f=0`** (no Coriolis) gives **exactly machine-zero**
(`dE/dt = 0.0`); with the real β-plane `f` it is `+5.24e5`. So:

> The free-surface projection *by itself* conserves energy exactly (the discrete grad/div pair is
> adjoint for the actual masked flow — `f=0` proves it). The spurious source appears only when
> there is **Coriolis-induced divergent flow** for the implicit free-surface step to project. The
> Coriolis term is itself energy-neutral (`∫u·(f k×u)=0` to machine precision), but the **sequence
> "apply explicit Coriolis → project onto the free-surface-balanced state"** is not energy-
> conserving in legoESM's C-grid implicit solver. The error is `dt`-independent (semi-discrete) and
> scales with the field's divergent content — hence gradient/grid-scale-activated.

This is the one structure **MITgcm does not have**: MITgcm is **unsplit** for a single layer and
its specific explicit-Coriolis → `cg2d` sequencing keeps the laminar Munk gyre at 0.031.

**Verified NOT the fix (this session):** moving Coriolis out of the solver's forward-backward
predictor into the AB2 explicit tendency `F_slow` (`coriolis_scheme="explicit_ab2"`, the solver's
FB-Coriolis gated off) leaves the leak **byte-identical** (`+5.2429e5`) — the placement of the
explicit Coriolis (FB vs AB2-`F_slow`) does not matter, because either way the Coriolis-induced
divergence is what the projection mishandles. Likewise an "unsplit single-layer path" built on the
existing implicit solver does **not** help: with the recipe's `θ_eta=θ_pgf=1.0` the predictor's
old-η PGF cancels *exactly* (in both the corrector and the η-equation), so the θ=1 implicit-CN solve
is **already algebraically the unsplit fully-implicit free surface** — and it still leaks. The
leak is intrinsic to how the C-grid implicit free-surface step balances the Coriolis-driven flow.

Symptom severity: bridging MITgcm's laminar 0.031 equilibrium into legoESM and integrating, legoESM
**cannot hold it** — `|u|max` drifts 0.031 → 0.11 (and climbing) toward the turbulent attractor.

## The fix (scoped — a real dycore task, NOT a config/wiring change)

Make the **C-grid implicit free-surface step conserve total energy in the presence of Coriolis** —
i.e. an energy-conserving coupling of the (energy-neutral) Coriolis operator with the free-surface
projection, so the discrete `KE + ½g∫η²` is conserved when `f≠0`. This is genuinely a dycore
problem (the projection of the Coriolis-divergent flow must be energy-orthogonal), and the
oracle-faithful target is MITgcm's unsplit explicit-Coriolis → `cg2d` sequencing.

What the fix is **NOT** (all ruled out by measurement): an Arakawa/Sadourny energy-conserving
*Coriolis* (already energy-neutral); *enstrophy-conserving advection* (byte-identical source across
all advection schemes; MITgcm's own flux-form isn't conserving yet stays laminar); a *time-scheme*
change (source converges as `dt→0`); the *PGF/free-surface split* (`f=0` conserves exactly, and θ=1
is already unsplit-equivalent); or the *explicit-Coriolis placement* (FB vs AB2 identical).

The regression test `tests/ocean/unit/test_barotropic_energy_conservation.py` pins the inviscid-
unforced KE growth so the fix has a target to drive to ~0. NOTE its IC carries `f≠0`; an `f=0`
control conserves to machine zero and could be added to bracket the defect.

## Reproduce

```
# MITgcm reference (needs gfortran; conda env `mitgcm-build`); 75 000-step run dumps + out_eq.txt:
python scripts/data/generate_mitgcm_barotropic_gyre_reference.py --ref-root <ref> --optfile <...>
# Investigation probes (this session) in scripts/tmp/:
#   _gyre_equilibrium_headtohead.py   legoESM spin-up to 75 000 steps (turbulent 0.15–0.37)
#   _gyre_scheme_variants.py          advection upwind/centered/vector-invariant (all over-energize)
#   _gyre_total_energy_budget.py      KE+PE budget: S/W≈3.2 at rough state
#   _gyre_coriolis_power.py           Coriolis power = machine-zero
#   _gyre_advection_power.py          centered advection power = machine-zero
#   _gyre_dt_scaling.py               growth converges as dt→0  => spatial source
#   _gyre_semidiscrete_budget.py      dE/dt byte-identical across advection => free-surface split
```
