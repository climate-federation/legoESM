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

## UPDATE 2026-06-18 (iteration 3): the leak is OPERATOR-LEVEL, not the split structure

Per the user's request, a **bit-faithful MITgcm UNSPLIT single-layer stepper** was built and run
(`scripts/tmp/_gyre_unsplit_mitgcm_faithful.py`): explicit ``Gu`` = advection(centered) +
Coriolis(face-f) + flux-divergence viscosity + no-slip sidedrag + wind (NO η-PGF) → AB2(abEps=0.01)
→ ``u* = u+Δt·Gu`` → implicit Helmholtz η-solve → ``u^{n+1}=u*−gΔt∇η``. **It also goes turbulent**
(|u|max 0.026→0.14 by 0.38yr), tracking MITgcm *exactly* at 0.1yr (0.0256 vs 0.026) then diverging.
AB2 vs forward-Euler made NO difference (byte-identical trajectory). So:

> **The split predictor-corrector velocity inconsistency was NOT the cause** — a clean unsplit
> single-velocity scheme leaks the same. Both legoESM schemes (split *and* faithful unsplit) run
> the gyre turbulent; MITgcm at the same A_h=400 stays laminar at 0.031. The discrepancy is
> therefore at the **OPERATOR** level (an energy source legoESM's discretization has that
> MITgcm's lacks), present regardless of time scheme or split/unsplit structure.

Further eliminations (iteration 3):
- **Viscosity operator is not the lever**: ``vector_laplacian`` (= MITgcm's grad(div)−curl(curl)
  default) gives a trajectory *identical* to ``flux_divergence`` (0.064 vs 0.064 at 0.38yr) — as
  expected for the near-non-divergent gyre.
- **It is a source-vs-dissipation balance**: 4× viscosity (A_h=1600) on the unsplit stepper makes
  it laminar (|u|max≈0.024, stable) — i.e. ~3× more dissipation than A_h=400 is needed to absorb
  legoESM's spurious source, whereas MITgcm's A_h=400 suffices. (4×A_h is masking, not faithful.)

Net: every targeted single-change fix has been falsified by measurement — vertex-f Coriolis, AB2,
unsplit structure, viscosity operator, and all default knobs. The +12 % inviscid leak is real,
first-order, dt-independent, advection-independent, and lives in the operator *composition* in a way
that none of the isolated-operator tests (all energy-neutral/adjoint) capture. **STILL OPEN.**

**Recommended rigorous next step** (the definitive "oracle as reference" approach, not yet done):
generate a MITgcm reference at a *rough mid-spin-up* state (not the smooth equilibrium, where the
earlier per-term diff already matched) and compare per-term momentum tendencies legoESM-vs-MITgcm
at that state — the term that diverges there is the source. This needs MITgcm intermediate dumps
(a fresh reference run), so it is scoped as the next investigation rather than another quick toggle.

## UPDATE 2026-06-17 (iteration 2): the "Coriolis" attribution below is WRONG

A clean **same-state f-toggle** (one rough gyre state, scale the grid ``f`` by 0/0.5/1.0)
shows the inviscid leak is **independent of Coriolis**: ``f=0`` leaks IDENTICALLY to ``f=1``
(``+5.2429e5`` both). The earlier ``f=0`` result that pointed at Coriolis used a *different*,
smoother ``f=0`` spin-up state — a contaminated control. So the Coriolis attribution in the
"## The source" section below is **superseded**.

What iteration 2 established (solid, by measurement):
- **Not Coriolis** (same-state f-toggle identical; a Sadourny vertex-f energy-conserving Coriolis
  — now in ``coriolis_cgrid_energy_conserving``, committed — does not change the gyre).
- **Not advection**: the thickness-weighted (physical) momentum-advection power ``⟨h_u u, adv⟩``
  is machine-zero (so is the unweighted), and the semi-discrete ``dE/dt`` is byte-identical
  across upwind/centered/vector-invariant.
- **Not the free-surface solver in isolation**: calling the barotropic solver alone (``f=0``,
  ``F_slow=0``) is mildly *dissipative* (−0.14 %/200 steps), and the Helmholtz operator
  ``A = I − gΔt²∇·(H∇)`` is **exactly symmetric** (rel asym ~1e-15) → grad/div ARE discrete
  adjoints.
- **Not a default knob**: ``fix_eta_drift``, ``barotropic_diffusion_alpha``, the barotropic
  time filter all toggle to byte-identical leak.
- **Not a time-scheme error**: the leak *converges* as ``dt→0`` (clean centered-advection
  dt-scan +11.3→12.1 %), i.e. it is a **first-order, spatial** effect; AB2 does not fix it.

The defect is therefore in the **composition** advection→``F_slow``→barotropic predictor-corrector:
a single inviscid step's ``dKE = ⟨Hu, du⟩ = +7.6e8`` (∝Δt, first-order) is NOT matched by the PE
change (``dPE = +5.7e5``, ~1300× smaller) — KE rises with no corresponding PE drop, even though
the individual PGF/continuity operators are adjoint. The leading hypothesis is a
**momentum-advection ↔ continuity mass-flux inconsistency** (the momentum flux-form advection
uses an ``h_k``-weighted transport that is not the same discrete mass flux the free-surface
continuity uses, so the predictor-corrector is not energy-conserving for the coupled system —
the "continuity-consistent advection" requirement of Arakawa / MOM6). This was NOT yet isolated
or fixed; it needs a careful from-scratch energy budget of the exact stepped scheme, not the
black-box probing above. **The fix is a scoped dycore task, still open.**

Delivered this session: the energy-conserving vertex-f Coriolis option (correct + unit-tested,
necessary-not-sufficient) and the elimination chain above.

## The source (measured): the free-surface split projection  [SUPERSEDED — see UPDATE above]

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

## UPDATE 2026-06-18 (iteration 3b): per-term diff at spin-up states — the limitation

Ran the per-term comparison against MITgcm's own `momU`/`momV` diagnostic dumps (which bundle
UVEL/VVEL/ETAN + Um_Advec/Cori/Diss/dPhiX/USidDrag at iters 5000…70000 — the spin-up):
- **TOTAL advection energy power** `∫(u·advU + v·advV)`: MITgcm **1e-13** (machine zero), legoESM
  **2.4e-10** — both essentially conserving on these smooth states (the earlier U-only −1e-6 was
  just the U↔V transfer, balanced by V).
- **Coriolis power**: machine-zero in both. legoESM's Coriolis values match MITgcm's exactly.
- Raw per-term *value* diff hits the known **baroclinic-vs-barotropic trap** (legoESM's direct
  `_bc_..._advection` is the ~0 baroclinic-perturbation part; the barotropic advection flows through
  the depth-mean `F_slow`), so a naive value-correlation is not meaningful (ratio 0.001, anti-corr).

**Conclusion / honest limitation**: on every *comparable* (smooth, laminar) state MITgcm visits, the
per-term tendencies match and conserve energy in both models — there is no isolable "wrong operator".
legoESM's excess energy only manifests once its trajectory develops grid-scale structure at the
marginally-resolved (Munk δ≈1.7 cell) western boundary current, an attractor **MITgcm never enters**,
so a direct per-term oracle diff *there* is structurally impossible. The residual is a subtle
nonlinear-stability difference of the under-resolved WBC, not a single faithfully-fixable term. The
oracle stands at its strong tiers (10-step eta 0.9997 + per-term tendency match on smooth states);
the equilibrium turbulence is documented as an open dycore-stability item.
