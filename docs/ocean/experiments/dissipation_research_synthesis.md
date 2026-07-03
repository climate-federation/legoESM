# Damping the ETOPO Barotropic Standing Mode — Synthesis

**Date**: 2026-05-03
**Inputs**: `dissipation_research_ocean.md` + `dissipation_research_dycore.md` + observed
ETOPO failure modes from the 30-day and 90-day runs in `tropical_omip_plan.md`.

## What both experts agree on

The failure is the **trapped topographic-Rossby / shelf-edge mode** (Rhines 1969 bottom-trapped wave):
deep ocean cells next to a steep continental slope at low-to-mid latitude, where the wave's group
velocity is killed by coarse Δx and weak |f|. Bathymetry smoothing only relocates the trapping
latitude — it does not address the mode itself. Both reports converge on three of the same
physical levers (drag, slope-targeted viscosity, mode-selective dissipation) but frame them
differently: the ocean expert gives them as physical parameter changes (DRAG_BG_VEL, slope
biharmonic) while the dycore expert gives them as numerical projections (depth-mean-only
hyperviscosity, divergent vs rotational anisotropy).

## Unified recommendation, ordered by leverage × cost

### Tier A — surgical, ≤50 LOC, both experts endorse

| # | Recommendation | LOC | Source | Why |
|---|---|---|---|---|
| **A1** | **`DRAG_BG_VEL = 0.1 m/s` background floor in quadratic bottom drag** | ~10 | ocean #1 | Drag scales with `√(u²+v²+u_bg²)`; at standing-mode amplitudes (≈0.05 m/s) effective drag jumps 3×. MOM6 OM4 default. JAX-clean. |
| **A2** | **Biharmonic hyperviscosity on depth-mean only**, applied via existing `F_slow_u/v` channel of the implicit-CN solver | ~50 | dycore #1 | Standing mode IS the depth-mean signal; baroclinic perturbation `u' = u_3d − ⟨u⟩_z` is mathematically untouched. MOM6 `BT_VISCOSITY=5e3 m⁴/s` precedent. Most surgical option — does not damp resolved geostrophy. |
| **A3** | **Tighten `r_factor_max` 0.20 → 0.10** (Beckmann–Haidvogel cap) with cosine taper | ~5 | ocean #2 | Pre-processing only. Cuts the partial-cell PGF residual that *seeds* the mode by ~2×. NEMO ORCA1 / MOM6 OM4 standard. |

These three address (in order): the **damping** (A1), the **mode-selective sink** (A2), and the **seed** (A3). Together they cost <100 LOC and don't change physics in any region except where they target.

### Tier B — moderate effort, high specificity

| # | Recommendation | LOC | Source | Why |
|---|---|---|---|---|
| **B1** | **Slope-foot biharmonic enhancement**: multiply biharmonic coefficient by `1 + α·tanh(|∇H|/H / r₀)` in the deepest 2-3 cells (combine ocean #3 + dycore #2 — same physics) | ~30 | both | MOM6 OM4 does this at the African shelf, ITF, and Ryukyu — *literally our failure points*. legoESM already has `smagorinsky_viscosity_cgrid` so just need a depth+slope mask multiplier. |
| **B2** | **Standing-mode diagnostics** (`χ` 2-grid-point variance + `P_bt` purity index) emitted every step | ~40 | dycore #4 | 5-10-day early warning before max\|u\| spikes. Distinguishes time-discretisation-mode (RAW filter helps) from spatial-discretisation-mode (PGF fix needed). Cheap (one global reduction each) and unblocks principled gating of A2/B1 by χ thresholds. |

### Tier C — last resorts, only if Tier A+B don't close the case

| # | Recommendation | LOC | Source | Why |
|---|---|---|---|---|
| **C1** | Smith–McWilliams 2003 **anisotropic viscosity** (`ν_div ≫ ν_rot`) | ~150 | dycore #3 | The "right" classical fix — selectively damps divergent modes (where the standing mode lives) over rotational modes (where geostrophy lives). Higher cost than A2 because it touches every layer. Use only if A2 leaks into baroclinic flow. |
| **C2** | **AFV-PPM PGF migration** (Adcroft–Hallberg–Hill 2008) — replace SMC03 with PPM-reconstructed Wright EOS integrals | ~300 | ocean #4 | Highest impact on the *seed* of the instability, but biggest lift. The principled fix; items A1–A3 + B1 are the production-pragmatic fix. Defer until needed. |
| **C3** | **RAW filter on η** (`α=1, ν=0.1`) | ~20 | dycore #5 | Only if B2's `χ` diagnostic shows the mode is in the time discretisation, not the spatial discretisation. Otherwise pure symptom mitigation. |

### Explicitly de-prioritised (both reports agree)

- **Beckmann–Döscher BBL** — wrong target (tracer, not momentum)
- **Holloway "Neptune" topographic stress** — too intrusive on mean flow at 1° OMIP
- **z̃ / Klingbeil ALE** — multi-week project, entangles with GM/Redi
- **Demange 2019 filtered free surface** — wrong mode (acts on temporal, not spatial aliasing)
- **Shapiro filter on η** — production-deprecated since SM05; less selective than A2
- **Plain isotropic biharmonic increase** — damps resolved 24°N flow before damping the mode
- **Lee-wave drag (Trossman/Naveira-Garabato)** — needs Goff–Arbic roughness dataset; defer to coupled

## Concrete implementation sequence

If we implement these in order and run a 90-day test after each, the working hypothesis is:

1. **A1 alone** ⇒ extends stable run from day 70 → ~day 150 (drag scales with column |u|, so 3× at low |u| helps but doesn't kill the mode)
2. **A1+A2** ⇒ stable indefinitely at 1° (depth-mean hyperviscosity directly removes the standing mode's energy)
3. **+A3** ⇒ reduces `χ` baseline (less seed energy from PGF residual), letting us reduce A2's coefficient
4. **+B1** ⇒ cleanup at remaining steep-slope hotspots (Antarctic Peninsula shelf, Bering sea floor)
5. **+B2** ⇒ scientific monitoring of how close we are to the edge, every run

We should build **B2 first** even though it's listed in Tier B — having `χ` and `P_bt` diagnostics for the *next* test (whatever fix we try) is much more informative than running blind. With diagnostics in place, every subsequent test costs the same wall clock but produces 10× the diagnostic information.

## Suggested next concrete step

Implement **B2 (diagnostics) + A1 (DRAG_BG_VEL) together**: ~50 LOC total, both fully differentiable,
neither touches the operator structure. Run a 90-day test. Read `χ(t)` and `P_bt(t)` traces. If
`χ` grows linearly before max|u| does → time-discretisation origin (try C3 RAW filter). If `P_bt`
saturates at a single column without `χ` growth → spatial-discretisation origin (do A2+B1).
This single experiment determines the rest of the path.

## References to the underlying reports

- **`dissipation_research_ocean.md`** — 7-section production-practice review with NEMO ORCA1 / MOM6 OM4
  parameter values, code paths, and per-candidate match assessment
- **`dissipation_research_dycore.md`** — 10-candidate dycore-numerics review with mathematical forms
  and computational-mode diagnostic prescription
