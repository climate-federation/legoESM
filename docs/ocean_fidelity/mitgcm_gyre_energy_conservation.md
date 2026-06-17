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

The decisive localization: the semi-discrete source `dE/dt` is **byte-identical** (`+5.2518e5`,
6 sig figs) under **upwind, centered, and vector-invariant** momentum advection. Advection
contributes *exactly nothing*; Coriolis is energy-neutral. By elimination the source is the one
remaining common element — the **barotropic free-surface predictor–corrector**:

> legoESM operator-splits the single layer into a "slow" explicit predictor (advection, FB
> Coriolis, old-η PGF, forcing) followed by an **implicit free-surface solve** that projects the
> predicted transport onto the balanced/continuity-satisfying state. That projection is **not
> energy-orthogonal**: it injects KE+PE at a rate that scales with the divergent (gravity-wave)
> content of the field. A projection error is exactly `dt`-independent and gradient-activated —
> matching every measurement above.

This is the one structure **MITgcm does not have**: MITgcm is **unsplit** for a single layer —
one symmetric elliptic `cg2d` free-surface solve whose discrete gradient/divergence pair is
adjoint *by construction*, so the free-surface KE↔PE exchange conserves energy. legoESM's split
predictor (explicit `gradient_*_cgrid`) + corrector (`divergence_cgrid` in the elliptic RHS) does
not preserve that adjointness on a closed basin (`gradient_x_cgrid` also assumes periodic
longitude, while the gyre is a closed box).

## The fix (scoped — NOT an Arakawa momentum scheme)

Make the **barotropic free-surface scheme energy-conserving**, one of:

1. **Energy-conserving split predictor–corrector** — choose the corrector projection so the
   discrete `KE + ½g∫η²` is conserved by the free-surface exchange (discretely-adjoint
   gradient/divergence with the closed-basin boundary, consistent time-weighting). Targeted fix
   in `barotropic_implicit_latlon_cgrid.py` + the `gradient_*`/`divergence_cgrid` metric.
2. **Unsplit single-layer path** — one implicit free-surface solve on the full momentum tendency,
   matching MITgcm exactly (the oracle-faithful option; larger change).

What the fix is **not**: an Arakawa/Sadourny energy-conserving *Coriolis* (already energy-neutral)
or *enstrophy-conserving advection* (provably irrelevant — the source is byte-identical across all
advection schemes, and MITgcm's own flux-form advection is not energy-conserving yet stays laminar).

The regression test `tests/ocean/unit/test_barotropic_energy_conservation.py` pins the inviscid-
unforced KE growth so the fix has a target to drive to ~0.

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
