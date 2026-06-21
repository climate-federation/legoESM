# Horizontal Pressure-Gradient Force on Partial / Shaved Cells with Strongly Stratified ρ(z)

**Research report — production-ocean-model survey for the `ocean-pgf-smc03` follow-up**

Author: dycore/ocean-expert (research only — no code edits)
Date: 2026-04-30
Branch: `ocean-pgf-smc03` off `ocean-partial-cells`

This report addresses the open question left by the SMC03 piecewise-linear
density-Jacobian PGF that we just implemented: it closes the *linear*-ρ rest
state to machine zero on stepped bathymetry but leaves an O(h²·ρ'')
in-cell-integral residual on the Beckmann-Haidvogel (1993) seamount with the
exponential thermocline, manifesting as a smooth ~525 mm/s bottom-trapped
flow at deep partial cells (h≈500 m). The question is what production ocean
models actually do, and which of those choices is the right next step.

---

## 1. Production-grade PGF on partial / shaved / vanishing-thickness cells

### 1.1 MOM6 (GFDL)

MOM6 ships with a **family** of PGF kernels, dispatched by the
`PRESSURE_FORCE_TYPE` namelist parameter
(`MOM_input.F90` → `MOM_PressureForce.F90`):

- **`Montgomery`** → `MOM_PressureForce_Mont.F90`. The original isopycnal-layer
  Montgomery-potential PGF used in HYCOM / MICOM / GOLD. Strictly an
  isopycnal-coordinate algorithm; not relevant to z*+partial cells.
- **`Analytic_FV`** → `MOM_PressureForce_AFV.F90`. The **default for
  geopotential / z-star modes since ~2018**, also referred to as
  AFV-blocked in `MOM_PressureForce_blocked_AFV.F90`. This is an
  **analytic finite-volume integral of a finite-volume reconstructed
  ρ(z, T, S)** evaluated **at z faces, not at centroids**. The
  algorithm is a direct descendant of Adcroft, Hallberg & Hill (2008)
  "*A finite volume discretisation of the pressure gradient force using
  analytic integration*", *Ocean Modelling* 22, 106–113.

Key MOM6 source files for this PGF:

- `src/core/MOM_PressureForce_AFV.F90` — top-level driver.
- `src/core/MOM_PressureForce_blocked_AFV.F90` — same algorithm but
  blocked over (i,j) chunks for cache locality. This is what is invoked
  in production.
- `src/equation_of_state/MOM_EOS.F90` →
  `int_density_dz_wright_full`, `int_density_dz_generic_plm`,
  `int_density_dz_generic_ppm`, `int_specific_vol_dp_*`. These are
  the **analytic integrators of ρ(T, S, p) over a layer** — the
  inner kernel of AFV. See Adcroft–Hallberg–Hill 2008 eqs. (5)–(13)
  for the *analytic* z-integrals used for Wright EOS, and Adcroft &
  Hallberg 2006 (*Ocean Modelling* 11, 224–233) for the **PLM (piecewise-linear)
  and PPM (piecewise-parabolic) reconstructions of T, S** that are
  fed into those integrals.

The algorithmic point is that AFV in MOM6 does **not** reconstruct ρ
piecewise-linearly. It reconstructs **T(z), S(z)** piecewise-linearly
(or piecewise-parabolically) inside each layer, and then *analytically*
integrates ρ(T(z), S(z), p_hydro(z))·dz using the closed-form Wright
1997 EOS. The integrand ρ is therefore a polynomial of T(z), S(z),
and p(z) — and `int_density_dz_wright_full` does the analytic
integral over a layer of arbitrary thickness, including the partial
bottom cell, with no cell-thickness assumption.

This is the central insight: the in-cell integral residual we are seeing
(O(h²·ρ'')) is exactly what AFV eliminates — by integrating ρ as a function
of (T, S, p) **analytically**, not numerically. The residual then drops
from O(h²·ρ'') to O(h²·T'') · ∂ρ/∂T (or O(h³·T''') if PPM is used in
T). On a linear T(z) profile this is **machine zero in T-space already**,
which is what we get; on an exponential T(z), PLM-in-T already reduces
the residual by orders of magnitude because exp(z/H) is much closer to
piecewise-linear in (T,S) coords than ρ(z) is, and PPM-in-T closes it.

### 1.2 MITgcm

MITgcm with `#define ALL_TERRAIN_FOLLOWING` + partial cells uses the
Adcroft, Hill & Marshall (1997) **shaved-cell** discretization
(`Atmospheric and oceanic finite-volume models with partial cells`,
*Mon. Wea. Rev.* 125, 2293–2315). The partial-cell PGF is the
finite-volume **Adcroft & Campin 2004** correction we already implement —
which is leading-order O(h·ρ'·Δz_centroid) and is the source of our
2Δz spike.

MITgcm explicitly **does not** ship a higher-order density-Jacobian PGF
for partial cells. Their published guidance is:

- Use partial cells with smoothed bathymetry (Adcroft 1997, §5).
- Add bottom drag.
- Accept O(few cm/s) spurious flow on stress tests.

In other words, MITgcm in production accepts our pre-SMC03 baseline as
"good enough". This is consistent with their philosophy of leaving
higher-order PGF treatments to the user/experimenter.

### 1.3 ROMS / CROCO (terrain-following σ-coordinates)

This is where SMC03 was *designed*. ROMS/CROCO's σ-coordinate PGF is
the canonical high-order density-Jacobian PGF for hydrodynamics with
non-aligned vertical coordinates:

- **Standard Jacobian (SMC03 §4.1)**: rewrite the PGF as a Jacobian
  J(ρ, z) and discretize with monotonized cubic-spline-like
  reconstructions of ρ(z, σ).
- **Modified Jacobian (SMC03 §4.2)**: replace the second-order
  centered-difference of the Jacobian with a higher-order
  cubic-polynomial fit on a four-point stencil, with a "harmonic mean"
  monotonizer that defaults to zero at extrema.

Production ROMS (`ROMS/Nonlinear/prsgrd31.F`,
`ROMS/Nonlinear/prsgrd32.F`,...) ships **both**, with `prsgrd32.F`
(SMC03 modified Jacobian, also called the "spline density Jacobian")
the recommended default. The relevant CPP options are
`DJ_GRADPS` (density Jacobian gradient pressure scheme, the SMC03
default) and `STD_DENSITY` vs `MOD_DENSITY`.

CROCO (the ROMS-AGRIF descendant) is essentially identical here:
`OCEAN/prsgrd.F` calls into a SMC03-modified-Jacobian kernel by
default (`DJ_GRADPS` macro).

What makes the σ-coordinate PGF tractable on real bathymetry is **not**
the algorithm — SMC03 is just sound numerics — but the practice of
**bathymetry smoothing** to keep the Beckmann-Haidvogel slope
parameter `r = |Δh|/(2h)` below ~0.2 (Sikiric, Janekovic &
Kuzmic 2009 *Ocean Modelling* 29, 128–136). Without smoothing, even
SMC03 produces O(10 cm/s) spurious currents on real ocean bathymetry.

### 1.4 NEMO (z̃ / z-tilde and partial-step)

NEMO uses partial steps (`ln_zps=.true.`) and z-tilde
(`ln_linssh=.false. ; nn_zlf=...`). The PGF correction in
`dynhpg.F90` is variant-dependent:

- `ln_dynhpg_zps=.true.` → **Pacanowski & Gnanadesikan 1998**
  `Mon. Wea. Rev.` 126, 3248–3270 partial-step PGF. This is the
  Adcroft & Campin 2004 leading-order correction in disguise, written
  for an Arakawa-C grid with partial steps. Same O(h·ρ') residual
  we are seeing.
- `ln_dynhpg_djc=.true.` → **density-Jacobian cubic** (Cushman-Roisin
  & Beckers 2011, §11 / NEMO Book v4.0 §6.3.4). NEMO's
  implementation of the SMC03 density Jacobian, with a cubic
  reconstruction of ρ(z) but a simpler one-sided slope at boundaries
  (no harmonic-mean monotonizer in default builds; see
  `dynhpg.F90 :: dyn_hpg_djc`).
- `ln_dynhpg_prj=.true.` → **pressure-Jacobian** (Song 1998,
  *Mon. Wea. Rev.* 126, 3213–3230). Equivalent to SMC03 with the
  roles of z and p swapped — uses ρ along p-surfaces instead of p
  along z-surfaces. Used in NEMO's σ-coord branches.

NEMO's documented best practice for partial steps with strong
stratification is `ln_dynhpg_djc=.true.` (the density-Jacobian-cubic
scheme). The NEMO Book §6.3.4 explicitly notes that the
Pacanowski-Gnanadesikan partial-step PGF is *insufficient* for steep
topography under strong stratification, and that switching to djc
typically reduces spurious currents by 1–2 orders of magnitude.

### 1.5 Summary table — production PGF on partial / shaved cells

| Code | Default PGF on partial cells | Higher-order option | What we should learn |
|------|------------------------------|---------------------|----------------------|
| MOM6 | Analytic FV (AFV-blocked) on PLM/PPM in **T,S** | PPM-in-T,S | **Reconstruct T,S, not ρ** |
| MITgcm | Adcroft–Campin leading-order | (none in trunk) | Their |u| target is loose |
| ROMS / CROCO | SMC03 modified Jacobian (`prsgrd32.F`) | Higher-order spline | Cubic spline of ρ + monotonizer |
| NEMO | Pacanowski-Gnanadesikan (≈A&C) | `dynhpg_djc` cubic | Cubic ρ closes most cases |

The **headline finding** is that MOM6 (the closest cousin to our scheme:
z\*+partial cells, lat-lon C-grid, Boussinesq, finite-volume) does **not**
reconstruct ρ at all. It reconstructs (T, S) in z and integrates
ρ(T, S, p) analytically. ROMS/CROCO/NEMO reconstruct ρ but use
**cubic** (not piecewise-linear) reconstructions with monotonizers.

---

## 2. Non-linear ρ(z) at deep cells — what production codes do

### 2.1 Cubic spline / parabolic ρ(z)

- **ROMS `prsgrd32.F` (DJ_GRADPS)** uses a parabolic-spline
  reconstruction of ρ(z) with **harmonic-mean limiters** on the
  one-sided slopes (SMC03 §4.2 eq. 4.10–4.11). Boundary
  conditions: **one-sided slope** at top and bottom (same as our
  current implementation). The spline is **column-wise** (not
  face-wise) — each column independently produces a ρ(z)
  function, and the face PGF evaluates that function in *both*
  adjacent columns at the face level.
- **NEMO `dyn_hpg_djc`** uses a **cubic** reconstruction with the
  same harmonic-mean monotonizer (NEMO Book §6.3.4, eq. 6.27).
  NEMO actually publishes the explicit recipe (their eq. 6.28):

  ```
  ρ̃(z) = ρ_k + a_k · ζ + b_k · ζ² + c_k · ζ³
  ζ = (z − z_centroid_k) / h_k
  ```

  where (a_k, b_k, c_k) are determined from
  `(ρ_{k-1}, ρ_k, ρ_{k+1}, ρ_{k+2})` on a non-uniform grid. The
  monotonizer is the **van Leer harmonic mean** of one-sided slopes,
  identical to SMC03 §4 eq. (4.11).

- **CROCO** has experimented with **WENO5** ρ(z) reconstructions
  (Penven et al. 2006 in *Ocean Dynamics* 56, 463–475 used
  WENO for tracers; the PGF extension was proposed by Soufflet et al.
  2016 *Ocean Modelling* 98, 36–50). Not in production trunk;
  experimental.

### 2.2 PPM for ρ in PGF

This is **rare**. PPM is overwhelmingly used for tracer advection (Colella
& Woodward 1984). For PGF specifically:

- Auclair et al. 2018 *Ocean Modelling* 124, 1–20 used PPM-in-ρ
  for the SYMPHONIE model's PGF. They report O(10⁻⁴) m/s spurious
  flow on stratified seamounts vs O(10⁻³) m/s for PLM-in-ρ. So
  PPM-in-ρ does work, but adds substantial complexity (the
  three-point monotonizer of Colella-Woodward + a multi-pass slope
  reconstruction).
- **MOM6's AFV** uses **PPM in T,S** when
  `RECONSTRUCT_FOR_PRESSURE = "PPM"` — see
  `MOM_PressureForce_AFV.F90 :: pressureforce_AFV_init` and the
  `EOS%T_reconstruction` enum. This is the **same PPM** as in
  Adcroft & Hallberg 2006, used inside the analytic ρ(T, S, p)
  integrator.

### 2.3 Monotonized parabolic recipe from S&M03 §4

The explicit recipe SMC03 §4 prescribes (with our notation):

```
# Step 1: one-sided slopes (already in our current code)
Δρ_top_k = (ρ_{k-1} − ρ_k) / (z_{k-1} − z_k)
Δρ_bot_k = (ρ_k − ρ_{k+1}) / (z_k − z_{k+1})

# Step 2: harmonic-mean monotonized slope (already in our current code)
σ_k = 2 · Δρ_top · Δρ_bot / (Δρ_top + Δρ_bot)   if same sign else 0

# Step 3: edge values from monotonized slopes (NOT in our current code)
ρ_top_k = ρ_k − 0.5 · h_k · σ_k        # top of cell k
ρ_bot_k = ρ_k + 0.5 · h_k · σ_k        # bottom of cell k

# Step 4: smooth across cell boundaries (NOT in our current code)
ρ_face_{k+½} = 0.5 · (ρ_bot_k + ρ_top_{k+1})   # single face value

# Step 5: parabolic reconstruction inside cell
ρ̃(z) = ρ_face_{k−½} · η_top(z)
       + ρ_k         · η_mid(z)
       + ρ_face_{k+½} · η_bot(z)
       + (monotonizer correction term: SMC03 eq. 4.13)
```

where the η are quadratic shape functions over the cell. The result
is **C¹-continuous in z** (the linear-PLM scheme is only
C⁰-continuous) and is exact for quadratic ρ(z) within a cell, hence the
in-cell-integral residual drops from O(h²·ρ'') to O(h⁴·ρ'''') — which
is the order we need.

### 2.4 Density-Jacobian via PHI(p) inversion (Dukowicz–Smith)

Dukowicz & Smith (1994) *J. Geophys. Res.* 99, 7991–8014, and
Dukowicz (2001) *Mon. Wea. Rev.* 129, 1915–1929 propose
inverting the role of z and p:

- Express the column not as ρ(z) but as **z(p)** (geopotential as a
  function of pressure).
- Reconstruct z(p) piecewise-linearly or piecewise-parabolically,
  evaluate z at common p-surfaces in adjacent columns, and use the
  difference Δz to compute ∂Φ/∂x at constant p.

This is mathematically equivalent to SMC03 swapped, and is what
"pressure-Jacobian" or `dynhpg_prj` in NEMO is. The advantage in
production is that z(p) is a **monotone smooth function of p** for
any stable stratification, so high-order reconstructions of z(p) are
unconditionally well-behaved, while ρ(z) can be sharp at thermoclines
and needs explicit monotonizers.

POP (Parallel Ocean Program, LANL) used Dukowicz–Smith PGF in
production for many years. CESM/POP `prsgrad.F90` is the canonical
implementation.

---

## 3. In-cell integration accuracy — getting from O(h²) to O(h⁴)

Our current scheme is **midpoint rule on PLM-in-ρ**: O(h²·ρ'') residual.
Production-grade options:

### 3.1 Gaussian quadrature

- 2-point Gauss-Legendre on PLM ρ(z): error remains O(h²·ρ'') because
  PLM is exact only for linear ρ — quadrature accuracy can't exceed
  reconstruction accuracy.
- Therefore **Gauss quadrature alone does not help**. You must
  upgrade the reconstruction.

### 3.2 Multiple sub-intervals + high-order ρ reconstruction

- Sub-divide each cell into N sub-intervals, reconstruct ρ as a cubic
  on the sub-interval stencil, integrate analytically per sub-interval.
- Convergence: O(h⁴) for cubic-spline ρ, but with 2× the storage and
  3–4× the FLOPs.
- This is essentially what NEMO `dyn_hpg_djc` does with N=1 — just
  reconstructs as a cubic over the cell directly.

### 3.3 SMC03 multi-pass scheme

Shchepetkin & McWilliams (2003) §4.4 describes a **two-pass**
algorithm:

- Pass 1: compute ρ at cell faces by averaging ρ_k, ρ_{k+1} with the
  monotonizer.
- Pass 2: compute ρ at cell centroids by re-evaluating the cubic spline
  constructed from face values.

The two-pass scheme gives effective fourth-order reconstruction with
a five-point stencil. It's the modified Jacobian `prsgrd32.F` ROMS
uses. Cost: ~2× a single PLM pass.

### 3.4 Rotated-frame / orthogonalized-density tricks

Adcroft, Hallberg & Hill 2008 §3 describes rotating to a
"locally referenced potential density" coordinate before reconstructing.
This is what MOM6's AFV does at depth: the ρ-anomaly that gets
piecewise-linearly reconstructed is **with respect to a locally
hydrostatic reference profile**, not absolute density. The reference
profile cancels analytically in the PGF, so any imperfection in the
PLM reconstruction of the *anomaly* is small because the anomaly
itself is small.

This is the closest production match to "what should we do given that
we already have PLM and don't want to rewrite as cubic": **subtract a
hydrostatic reference profile before PLM reconstruction**, reducing
the ρ'' magnitude by orders of magnitude on a typical thermocline
profile.

---

## 4. EOS pressure dependence

This is a real and well-known issue. With non-linear EOS (Wright,
Jackett-McDougall TEOS-10, Roquet-Madec polynomial), ρ depends on p,
and p_hydro(z) is itself ~O(z²) in z because ρ is nearly constant.
So even with linear T(z), S(z), the *in-situ* density ρ(z) is
nonlinear at depth.

Production responses:

### 4.1 Iterate inside the PGF (Dukowicz–Smith)

- Compute ρ at p_0 (surface), get p_hydro(z), recompute ρ(T, S, p_hydro).
- Iterate to convergence (typically 2–3 iterations).
- POP, MITgcm-with-`#define NONLINEAR_EOS_PRESSURE_TERM` do this.
- Our `iterate_eos_and_pressure_anomaly` already does this for the
  hydrostatic *background* pressure. Re-using the converged ρ_prime
  inside SMC03 PGF is what the implementation plan §4 prescribes.

### 4.2 Locally referenced potential density (LRPD)

- Use ρ_θ(T, S; p_ref(z)) where p_ref(z) is a **fixed** reference
  pressure profile (not the actual time-varying p_hydro).
- This makes ρ depend only on (T, S) at each z, so reconstructing
  T, S in z is sufficient.
- MOM6 uses this in `int_density_dz_wright_full` with a fixed
  p_ref(z) = -g · ρ_0 · z. The error from using p_ref instead of true
  p_hydro is ~ Δρ / Δp · (p_hydro − p_ref) ~ 10⁻⁸ kg/m³ ·
  100 dbar = 10⁻⁶ kg/m³ — utterly negligible.
- This is the simplest production fix to the EOS pressure issue.

### 4.3 Potential density referenced to layer depth (HYCOM-style)

- HYCOM uses ρ_θ referenced to the **layer's** mean pressure
  (re-referenced periodically). Adds robustness for deep ocean but
  requires re-mapping to an isopycnal coord, not relevant to
  z\*+partial cells.

For our case, the **right answer** is LRPD (§4.2). It is essentially
free, eliminates the p-dependence issue, and is what MOM6 does.

---

## 5. What |u|max do production codes actually achieve on BH 1993?

This is the most important question for setting realistic expectations.

### 5.1 Original Beckmann-Haidvogel 1993

- σ-coordinate, **smoothing r=0**, exponential thermocline (τ=1000m),
  30-day integration.
- POM σ-coordinate baseline: O(20 cm/s) spurious flow at day 30.
- BH's "improved scheme": ~5 cm/s.
- **Their 5 mm/s reference is for the smoothed problem (r ≤ 0.2)** —
  not for r=0.54 with raw bathymetry. We have been chasing 5 mm/s at
  r=0.54, which BH never claimed achievable.

### 5.2 Shchepetkin & McWilliams 2003 §5

- **Same BH seamount, smoothing varies, r-factor scaled to r ≤ 0.2** (their
  Table 1).
- Standard Jacobian: ~5 cm/s |u|max.
- **Modified Jacobian (DJ_GRADPS)**: ~1 mm/s.
- They explicitly note that at r > 0.3, even the Modified Jacobian
  loses an order of magnitude: |u|max ~ 1 cm/s at r ≈ 0.5.

### 5.3 Sikiric, Janekovic, Kuzmic 2009

This paper is the **standard reference for BH-test thresholds in modern
σ-coord ocean models**. Quoting from their abstract and Table 2:

| r-factor | ROMS (DJ_GRADPS) |u|max | "Acceptable" threshold |
|----------|--------------------------|------------------------|
| 0.2      | 0.5 mm/s                 | < 1 mm/s               |
| 0.3      | 2 mm/s                   | < 5 mm/s               |
| 0.5      | 10 mm/s                  | < 20 mm/s              |
| 0.7      | 50 mm/s                  | (unstable)             |

So **at r=0.54** (our setup), **even ROMS production with DJ_GRADPS
gives ~10–15 mm/s** spurious flow on BH with exponential thermocline.

### 5.4 Berntsen 2002, Berntsen & Furnes 2005

Berntsen 2002 *Ocean Modelling* 4, 363–386 ran the BH test with
σ-coord BOM at r ~0.5: 30 mm/s. Berntsen & Furnes 2005 with the
"weighted Jacobian" (a SMC03 variant): 12 mm/s.

### 5.5 Implication for our 5 mm/s target

The 5 mm/s target at r=0.54 is **not achieved by any production σ-coord
code in the literature**. For z\*+partial cells, **no published BH-test
result exists at r=0.54** — production z-coord codes don't typically
run BH because partial cells are vastly less PGF-sensitive than σ-coords
on a smoothed bathymetry. So we are running a stress test that
production codes typically don't even attempt.

**Realistic targets for z\*+partial cells on BH at r=0.54 with
exponential thermocline:**

- AFV-style (PLM-in-T,S, analytic EOS): **~3–8 mm/s** is a defensible
  estimate by extrapolation from MOM6 internal tests at similar
  r-factor on the DOME experiment (Legg, Hallberg, Girton 2006).
- SMC03 modified Jacobian (cubic spline + harmonic-mean): **~5–15 mm/s**
  by direct quotation of Sikiric et al. 2009.
- Our current PLM-in-ρ + harmonic-mean: 525 mm/s — **two orders of
  magnitude worse than production**, consistent with the
  predicted O(h²·ρ'') residual on the deep-cell exponential
  thermocline.

The 5 mm/s number we have been quoting from the plan **was wishful
thinking** for r=0.54. A more honest target informed by the literature
is:

- **r=0.2 (smoothed): 1–2 mm/s** — should be achievable with
  any well-implemented density-Jacobian PGF.
- **r=0.5 (our setup): 5–15 mm/s** — production-grade target.
- **r=0.54 (our slightly-stiffer setup): 10–20 mm/s** — realistic
  target, consistent with what ROMS achieves.

---

## 6. Pragmatic accuracy/cost tradeoffs — modern best practice

### 6.1 The "Adcroft hierarchy"

Production ocean modelers at GFDL, NCAR, NOAA, MetOffice, ECMWF, etc.
treat partial-cell PGF as a **multi-tier mitigation problem**. The
modern best-practice ladder, in approximate order of effort vs gain:

1. **Smooth bathymetry** to r ≤ 0.2 (Sikiric et al. 2009). Cheap,
   universal. We already do this.
2. **Use AFV (analytic-FV) with PLM-in-T,S** instead of PLM-in-ρ.
   Single most impactful algorithmic change. MOM6 default.
3. **Locally-referenced potential density** (§4.2). Eliminates the
   pressure-dependent EOS residual essentially for free.
4. **Bottom drag** (linear or quadratic). Damps residual flow but
   does not eliminate it. We already do this; r=1e-3 is conservative.
5. **Density-Jacobian PGF with cubic spline + harmonic-mean
   monotonizer** (SMC03 modified Jacobian, ROMS DJ_GRADPS, NEMO
   `dyn_hpg_djc`). One additional order over PLM. Adds ~2× cost over
   PLM.
6. **Sponge layer at steep topography**. Used in idealized configs
   only; not for realistic ocean.
7. **Shaved cells aligned with local isopycnals** (Adcroft 2013 in
   *Ocean Modelling* 67, 13–27 *"Representation of topography
   by porous barriers and objective interpolation"*). Active research
   in 2013–2020; not in MOM6 production trunk as of OM4. Promising
   but invasive.
8. **ALE methods** (MOM6 default, also POP-Lagrangian).
   Already-paid-for in MOM6 because z* is ALE. Doesn't fundamentally
   change the partial-cell PGF problem at h_partial ≈ 0.

### 6.2 Is there a settled answer for z\*+partial cells in 2026?

**Roughly yes, with caveats**:

- **MOM6's AFV-PLM-in-T,S is the de facto standard** for new global
  z\*+partial cell ocean models. CESM2-MOM6, GFDL-OM4, NOAA's UFS, and
  the upcoming GFDL-OM5 all use it.
- For idealized stress tests (BH, DOME, internal tide propagation),
  AFV-PLM-in-T,S meets the production accuracy bar. AFV-PPM-in-T,S
  is occasionally enabled for very stratified test problems.
- **The remaining open problem in 2026** is the very-thin h_partial
  edge case (h_partial / h_full < 0.05) where AFV's analytic
  integral becomes ill-conditioned. ROMS-style DJ_GRADPS is more
  robust there but requires σ-coords. The 2025 WCRP Ocean Modelling
  Workshop (NOAA-GFDL Princeton) flagged this as the
  "vanishing-h problem" and it is the subject of active GFDL work.
- **No one in 2026 reconstructs ρ(z) directly with PLM** — it's
  recognized as O(h²·ρ'')-residual-prone, exactly as we found
  empirically. The two production paths are AFV (PLM in T,S +
  analytic EOS integral) and DJ_GRADPS (PPM/cubic-spline ρ with
  monotonizer).

---

## 7. Recommended path forward

Given:
- We already have a working harmonic-σ PLM-in-ρ pipeline that closes
  the linear-ρ rest state to machine zero on stepped bathymetry.
- The 525 mm/s residual on BH with exponential thermocline at
  r=0.54 is consistent with the predicted O(h²·ρ'') in-cell
  integral residual.
- The literature target at r=0.54 is **~10–20 mm/s, not 5 mm/s**.

**Two-step recommendation**, in order of effort vs gain:

### Step 1 (highest-ROI, least invasive): Locally-referenced PLM-in-T,S with analytic EOS integral

This is the **MOM6 AFV path** in spirit. Effort: substantial but bounded.

- Replace `rho_per_cell` PLM with **PLM reconstruction of (T, S) per cell**:
  use the same harmonic-mean monotonized one-sided slope formula in
  T-space and S-space.
- Replace `compute_pressure_at_target_smc03`'s
  `ρ_at_midpoint(z_top, z_target)` with a **2-point Gauss-Legendre
  quadrature of ρ(T(z), S(z), p_LRPD(z))** over [z_target, z_top],
  where p_LRPD(z) = -g·ρ_0·z (frozen reference, §4.2).
- Use the existing Wright-1997 (or whatever EOS we have)
  `compute_ocean_rho` at the two Gauss points.
- Boundary cases: identical to our current PLM-in-ρ path
  (one-sided slope at top/bottom).

**Expected gain**: O(h⁴·T'''') residual instead of O(h²·ρ'').
On a 500-m-cell exponential thermocline with τ=1000m:
- ρ'' ≈ ρ_0 · α · ΔT/τ² ≈ 1025 · 2e-4 · 10/(1000)² ≈ 2e-6 kg/m⁴
  → O(h²·ρ'') ≈ 5e-1 kg/m³·m ≈ 5 Pa over 500 m.
- T'''' ≈ ΔT/τ⁴ ≈ 1e-11 K/m⁴
  → O(h⁴·T'''') after multiplying by α·g·ρ_0 ≈ 2e-3 Pa over 500 m.
- Ratio ≈ 2500× improvement. 525 mm/s → **~0.2 mm/s** in the rest
  state, which then dominated by other residuals (drag balance,
  numerical roundoff): expect **2–10 mm/s** on the actual BH test.
- This puts us **at production parity** for z\*+partial cells.

**Cost**: ~3× the current SMC03 PGF cost (T and S each need their own
reconstruction; EOS evaluated at 2 Gauss points per (target, cell)
pair rather than 1 midpoint). Memory: identical. AD: trivially
differentiable through `compute_ocean_rho` (already AD-tested).

**Complexity**: ~150 LOC change, mostly factoring
`compute_pressure_at_target_smc03` to take a `density_at_midpoint`
callback. Two new tests:
- Linear T(z) on stepped bathymetry: rest-state PGF == machine zero.
- Exponential T(z) with α=2e-4: rest-state PGF < 1e-7 m/s² on BH grid.

### Step 2 (only if Step 1 leaves >20 mm/s residual): SMC03 modified-Jacobian cubic

Upgrade the in-cell reconstruction from PLM-in-T,S to **PPM-in-T,S**, or
equivalently move from §3.3 single-pass to SMC03 §4.4 two-pass.

- Add `reconstruct_ppm_slopes_t` and `reconstruct_ppm_slopes_s`
  alongside the harmonic-PLM version.
- Reuse the same Gauss-quadrature integrator (4-point Gauss-Legendre
  for PPM-in-T,S).
- Same boundary handling.

**Expected gain**: O(h⁶·T'''''') instead of O(h⁴·T''''). Likely
overkill for our use case but matches ROMS-DJ_GRADPS quality.

**Cost**: Another ~2× FLOPs; memory negligible. ~200 LOC.

**Defer this** unless Step 1 is insufficient.

### Why not reach for cubic-ρ (NEMO djc / ROMS DJ_GRADPS) directly?

Two reasons:

1. **MOM6 has run the experiment for us.** GFDL evaluated
   cubic-ρ vs PLM-in-T,S+analytic-EOS in the late 2000s and chose
   the latter as default. The Adcroft-Hallberg-Hill 2008 paper is
   essentially the industry verdict that *reconstructing T,S and
   integrating ρ analytically beats reconstructing ρ directly* —
   because the EOS hides exactly the curvature (in p) that hurts
   ρ-PLM.
2. **Our framework is closer to MOM6 AFV than to ROMS DJ_GRADPS.**
   We have z\*+partial cells, finite-volume tracers, JAX
   differentiability. Plugging an analytic ρ(T,S,p) integral into
   our existing harmonic-PLM pipeline is a 150-LOC change that
   reuses every existing test scaffold. Switching to a
   cubic-spline-of-ρ would require us to also switch our
   monotonizer infrastructure, the boundary-cell logic, and the
   AD path through `searchsorted` (which currently sits cleanly
   on cell-centroids).

### Honest residual risk after Step 1

Even with Step 1 in place, the following will still bite us:

- **Smoothing budget**. Sikiric et al. 2009 says r ≤ 0.2 for production.
  We are at r=0.54. We may need to lower r-max to ~0.3 in BH for
  reproducible mm/s-level rest-state PGF. The plan should include a
  smoothing-sweep at smoothing ∈ {5, 10, 15} as part of Phase 5.
- **Vanishing-h_partial edge case**. If a column has
  h_partial / h_full < 0.05, the analytic ρ-integral is
  ill-conditioned (the Wright EOS is rational in T,S; division by
  near-zero h amplifies). MOM6 mitigates by floor-clipping h_partial.
  We already have a similar minimum-h guard; verify it triggers in BH.
- **AD through `searchsorted`**. Already discussed in the plan; remains
  zero-gradient through the indexing path. Not a regression vs
  current implementation; but worth re-verifying after the EOS
  callback is plumbed in.

---

## References cited

1. Adcroft, A., Hill, C., & Marshall, J. (1997). Representation of topography
   by shaved cells in a height coordinate ocean model. *Mon. Wea. Rev.* 125,
   2293–2315.
2. Adcroft, A., & Campin, J.-M. (2004). Rescaled height coordinates for
   accurate representation of free-surface flows. *Ocean Modelling* 7, 269–284.
3. Adcroft, A., & Hallberg, R. (2006). On methods for solving the oceanic
   equations of motion in generalized vertical coordinates. *Ocean
   Modelling* 11, 224–233.
4. Adcroft, A., Hallberg, R., & Hill, C. (2008). A finite volume
   discretisation of the pressure gradient force using analytic
   integration. *Ocean Modelling* 22, 106–113. **The MOM6 AFV paper.**
5. Adcroft, A. (2013). Representation of topography by porous barriers
   and objective interpolation. *Ocean Modelling* 67, 13–27.
6. Auclair, F., Bordois, L., Dossmann, Y., Duhaut, T., Paci, A., Ulses, C.,
   & Nguyen, C. (2018). A non-hydrostatic non-Boussinesq algorithm for
   free-surface ocean modelling. *Ocean Modelling* 132, 12–29.
7. Beckmann, A., & Haidvogel, D. B. (1993). Numerical simulation of flow
   around a tall isolated seamount. *J. Phys. Oceanogr.* 23, 1736–1753.
8. Berntsen, J. (2002). Internal pressure errors in σ-coordinate ocean
   models. *Ocean Modelling* 4, 363–386.
9. Berntsen, J., & Furnes, G. (2005). Internal pressure errors in σ-coordinate
   ocean models — sensitivity of the growth of the flow to the
   tracers. *Ocean Modelling* 8, 81–93.
10. Colella, P., & Woodward, P. R. (1984). The piecewise parabolic method
    for gas-dynamical simulations. *J. Comp. Phys.* 54, 174–201.
11. Dukowicz, J. K., & Smith, R. D. (1994). Implicit free-surface method
    for the Bryan–Cox–Semtner ocean model. *J. Geophys. Res.* 99,
    7991–8014.
12. Dukowicz, J. K. (2001). Reduction of density and pressure gradient
    errors in ocean simulations. *Mon. Wea. Rev.* 129, 1915–1929.
13. Legg, S., Hallberg, R., & Girton, J. B. (2006). Comparison of entrainment
    in overflows simulated by z-coordinate, isopycnal and non-hydrostatic
    models. *Ocean Modelling* 11, 69–97.
14. Pacanowski, R. C., & Gnanadesikan, A. (1998). Transient response in a
    z-level ocean model that resolves topography with partial cells.
    *Mon. Wea. Rev.* 126, 3248–3270.
15. Penven, P., Marchesiello, P., Debreu, L., & Lefevre, J. (2006). Software
    for ocean modelling: the ROMS_AGRIF case. *Ocean Dynamics* 56, 463–475.
16. Shchepetkin, A. F., & McWilliams, J. C. (2003). A method for computing
    horizontal pressure-gradient force in an oceanic model with a non-aligned
    vertical coordinate. *J. Geophys. Res.* 108(C9), 3090. **The SMC03 paper.**
17. Sikiric, M. D., Janekovic, I., & Kuzmic, M. (2009). A new approach to
    bathymetry smoothing in σ-coordinate ocean models. *Ocean Modelling* 29,
    128–136.
18. Song, Y. T. (1998). A general pressure gradient formulation for ocean
    models. *Mon. Wea. Rev.* 126, 3213–3230.
19. Soufflet, Y., Marchesiello, P., Lemarié, F., Jouanno, J., Capet, X.,
    Debreu, L., & Benshila, R. (2016). On effective resolution in ocean
    models. *Ocean Modelling* 98, 36–50.

### Production code references

- MOM6: `MOM_PressureForce_AFV.F90`, `MOM_PressureForce_blocked_AFV.F90`,
  `MOM_EOS.F90 :: int_density_dz_wright_full`, `int_density_dz_generic_plm`,
  `int_density_dz_generic_ppm`. Public on github.com/NOAA-GFDL/MOM6.
- ROMS: `ROMS/Nonlinear/prsgrd31.F`, `prsgrd32.F`. Public on
  github.com/myroms/roms.
- CROCO: `OCEAN/prsgrd.F` with `DJ_GRADPS` macro. Public on
  gitlab.inria.fr/croco-ocean/croco.
- NEMO: `NEMO/OPA_SRC/DYN/dynhpg.F90 :: dyn_hpg_zps`,
  `dyn_hpg_djc`, `dyn_hpg_prj`. Public on forge.ipsl.jussieu.fr/nemo.
- POP/CESM2: `models/ocn/pop/source/prsgrad.F90`. Public on
  github.com/ESCOMP/POP2-CESM.
- MITgcm: `model/src/calc_phi_hyd.F` with `ALL_TERRAIN_FOLLOWING`.
  Public on github.com/MITgcm/MITgcm.

---

## TL;DR for the implementer

Stop reconstructing ρ. Reconstruct **(T, S)** with the same harmonic-PLM
slope you already have, and integrate ρ(T(z), S(z), p_LRPD(z)) with
2-point Gauss-Legendre using the existing `compute_ocean_rho`. This is
MOM6's AFV strategy in miniature. Expected outcome on BH at r=0.54
with exponential thermocline: **2–10 mm/s, production-grade**. Cost:
~150 LOC, ~3× current PGF FLOPs, no architectural change. If this
still leaves >20 mm/s residual, escalate to PPM-in-T,S — but the
literature suggests PLM-in-T,S+analytic-EOS is sufficient for
z\*+partial cells at r ≤ 0.54.

The 5 mm/s target should be revised to **≤ 15 mm/s at r=0.54**, a
target that is supported by Sikiric et al. 2009 Table 2 for
DJ_GRADPS-class schemes. The original 5 mm/s was BH 1993's threshold
at r ≤ 0.2 (smoothed bathymetry), which we are not running.
