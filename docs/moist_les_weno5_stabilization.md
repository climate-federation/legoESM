# Stabilizing WENO5 moist LES

## TL;DR (corrected root cause — read this first)
The earlier "WENO5 destabilizes moist convection" was a **misdiagnosis**. The
real blocker was **float32 precision loss in the WENO-Z smoothness indicators**
(β ∝ squared scalar differences) at a near-discontinuity — e.g. the DYCOMS θ_l
inversion (~9 K per cell) — which NaNs at cloud onset. **WENO5 in float64 is
stable on this near-inviscid core WITHOUT any added momentum dissipation**, and
it directly improves cloud amount:

| case | van-Leer (f32) | WENO5 (f64) | WENO5 (f64) + θ-hyperdiff |
|------|----------------|-------------|---------------------------|
| DYCOMS LWP | ~2 g/m² (collapses) | ~28 | **~48** (ref 50–80) |
| DYCOMS cover | ~0.2 | 0.99 | 1.0 (ref ~1.0) |
| BOMEX cover | 0.068 | 0.094 (ref 0.10–0.15) | — |

**Recipe:** `--scalar-advection weno5` with **float64** (omit `--f32`,
`JAX_ENABLE_X64=1`). Add a modest `--theta-hyperdiff` (scale-selective k⁴ on θ
only) to further maintain the stratocumulus. The momentum-side operators below
(`w_hyperdiff`, `div_damping`) were the originally-scoped hypothesis; they do
NOT fix the f32 blow-up (it is precision, not dynamics) and are **not required** —
kept as opt-in tuning knobs. `make_grid` warns on WENO5+float32.

---
## (Original analysis — momentum-side dissipation)

Pseudo-spectral incompressible plane LES (`spectral_les_plane.py`), moist
coupling (`spectral_les_moist.py`). Context: the van-Leer scalar default is
stable but over-diffusive (cloud cover/LWP too low, worst for DYCOMS
stratocumulus). WENO5 scalars are sharp but destabilize moist convection. This
note documents the **codex-vetted stabilization path**: WENO5 scalars + targeted
**momentum-side** dissipation, all opt-in.

## Why WENO5 alone destabilizes this near-inviscid core
This core has **no numerical hyperdiffusion and no acoustic damping** — the only
momentum dissipation is the SGS eddy viscosity, which nearly vanishes in the
weakly-sheared cloud layer. So the resolved dynamics are essentially inviscid at
the grid scale.

The moist feedback loop is: condensation → latent heating → buoyant `w` →
adiabatic lifting → more condensation. With **van-Leer** scalar transport the
cloud/heating fields are smeared (numerically diffuse), so the latent heating
has no grid-scale (2Δ) structure and the loop stays smooth. With **WENO5**
(5th-order, low diffusion) the cloud edges and latent heating are **sharp at the
grid scale**, exciting a 2Δ `w` mode. Because the momentum dynamics cannot
dissipate that mode, it amplifies through the latent-heat feedback until the run
NaNs at cloud onset (~0.5 h in BOMEX/DYCOMS). Empirically this happens for WENO5
in *any* direction (pure WENO5 and the WENO5-horizontal/van-Leer-vertical hybrid
both blow up) — i.e. it is a **momentum**-side problem, not a scalar-scheme
problem.

## Why momentum dissipation, not scalar diffusion
Adding diffusion back to the **scalars** (e.g. reverting to van-Leer, or adding a
scalar Laplacian) would re-smear the cloud field and **throw away the entire
reason for WENO5** — the sharp moisture/cloud structure that lifts cloud cover
and LWP. The right fix removes the grid-scale **`w`** energy the feedback excites
while leaving the resolved scalars sharp. So we add dissipation to the
**momentum** equation only:

* **`w_hyperdiff_coeff` (ν₄)** — horizontal biharmonic hyperdiffusion on `w`:
  `dw/dt −= ν₄·(∇²_h)²w = −ν₄·k⁴·ŵ` (spectral, horizontal-only, `_w_hyperdiffusion`).
  Scale-selective: it damps the 2Δ `w` mode hard (∝k⁴) and barely touches the
  energy-containing eddies, so the resolved convection and the WENO5 scalar
  sharpness survive. **This is the operative lever.**
* **`div_damping_coeff` (α)** — `du/dt += α·∇(∇·u)` (`_divergence_damping`).
  Provided for completeness/compressible-style tuning, but on this
  **incompressible projection** core `∇·u≈0` after every pressure solve, so it is
  **largely redundant** here. Prefer `w_hyperdiff`.

Both default **off**; van-Leer remains the stable default. Turn them on only with
`--scalar-advection weno5`.

## Tuning
Let `Δ = min(dx, dy)` and `dt` the step. The hyperdiffusion damps the highest
resolved mode at rate `γ = ν₄·(π/Δ)⁴`; explicit stability needs `γ·dt < 2`.

* **`ν₄` (w_hyperdiff_coeff)** — start at
  `ν₄ ≈ 0.25 · Δ⁴ / (π⁴ · dt)` → **≈1×10⁵ m⁴/s** at `dx=100 m, dt=2 s`
  (≈4×10³ at `dx=35 m, dt=0.5 s` for DYCOMS). This damps the 2Δ mode over a few
  steps while keeping `γ·dt ≈ 0.25 ≪ 2`. If it still NaNs, raise ×3; if cloud is
  over-smoothed (LWP/cover drop back toward the van-Leer values), lower ×3.
* **`α` (div_damping_coeff)** — if used at all, `α ≈ 0.05 · Δ² / dt` (m²/s).
  Expect ~no effect here (projection core).
* **CFL** — WENO5 is less dissipative, so run a **lower advective CFL, 0.25–0.35**
  (e.g. `--dt 0.5` for DYCOMS `dx=35 m`, `--dt 1` for BOMEX `dx=100 m`). The
  hyperdiffusion CFL `γ·dt<2` is satisfied by the `ν₄` formula above.

## Recommended starting configs
```
# DYCOMS-II RF01 (dx=35 m, dt=0.5 s)
--scalar-advection weno5 --w-hyperdiff 4e3 --dt 0.5
# BOMEX (dx=100 m, dt=1 s)
--scalar-advection weno5 --w-hyperdiff 1e5 --dt 1
```

## Test matrix (`scripts/tmp/run_stab_matrix.sh`)
DYCOMS first (the worst van-Leer case), then BOMEX:
| Run | scalars | w_hyperdiff | div_damping |
|-----|---------|-------------|-------------|
| A | van_leer | off | off | baseline |
| B | weno5 | off | off | expect NaN (control) |
| C | weno5 | on | off | **the candidate** |
| D | weno5 | off | on | (expect NaN — div damping ~no-op) |
| E | weno5 | on | on | |

## Success metrics (DYCOMS-II RF01, Stevens et al. 2005, hours 2–4)
A run "wins" if, relative to the van-Leer baseline, it is **stable for ≥4 h** AND
moves these toward the reference:
* **LWP** 50–80 g/m² (van-Leer collapses to ~2; the key failure to fix)
* **cloud cover** ≈ 1.0 (overcast; van-Leer ~0.2)
* **z_i** 840–870 m, entrainment ~0.4 cm/s (van-Leer over-entrains/thins)
* well-mixed θ_l/q_t; ⟨w'²⟩ single mid-BL peak ~0.4 m²/s²
* total-water conservation error small (diagnostic `total_water` drift)
* `max_cfl < 1`, `max|w|` physical (≲1–2 m/s), no positivity violations
  (`qv_min ≥ 0`).

Diagnostics bundle: `spectral_les_moist.moist_diagnostics`
(cloud_frac, lwp, max_w, w_var, tke, max_cfl, total_water, qv/qc bounds).
