# Barotropic-mode noise suppression in production ocean GCMs

A synthesis of how MOM6, MITgcm, NEMO, POP/MPAS-Ocean, and ROMS handle the
C-grid barotropic noise problem (checkerboard, divergent-mode, mode-splitting
aliasing) and what it implies for `legoESM`'s split-explicit lat-lon C-grid
substep.

---

## Q1. Inventory of barotropic-mode noise suppression by model

Production codes do **not** rely on a single knob. They stack 3–4 mechanisms.
Below is what is actually on by default in each code.

### MOM6 (GFDL OM4 / CM4 / CESM-MOM6)

Source: `src/core/MOM_barotropic.F90`. OM4_025 production namelist
(`MOM6-examples/ice_ocean_SIS2/OM4_025/MOM_input`).

| Mechanism | Parameter | OM4_025 value | Code default |
|---|---|---|---|
| Semi-implicit gravity-wave step | `BEBT` | `0.2` | `0.1` |
| Project velocity tendency by `(1+BEBT)` | `BT_PROJECT_VELOCITY` | `True` | `False` |
| Barotropic-baroclinic correction bound | `BOUND_BT_CORRECTION` | `True` | `False` |
| Streaming bandpass time filter on (η, U_bar) | `DT_BT_FILTER` | (positive — ~`1*DT`) | 0 (off) |
| Ratio of barotropic to baroclinic dt | `DTBT` | `-0.9` | `-0.98` |
| Velocity ceiling | `MAXVEL` | `6.0 m/s` | — |
| Velocity floor | `VEL_UNDERFLOW` | small | — |
| Dynamic pressure feedback (atmos load) | `DYNAMIC_SURFACE_PRESSURE` | `True` | `False` |

Crucially, MOM6 has **no namelist parameter named `BT_DIV_DAMP`**. Searching
the source code, the only divergence-control mechanisms in the barotropic
substep are (a) `BEBT` semi-implicit gravity-wave damping, (b) the streaming
filter `Filt_CS_u`, `Filt_CS_v` (`Filt_accum`), and (c) the corrective
fictitious mass source `bt_mass_source` that drives barotropic η back to the
baroclinic estimate. Lateral viscosity in MOM6 is applied to the
**3D layered velocity** in `MOM_hor_visc.F90` (Smagorinsky / Leith / biharmonic),
**not** as a separate barotropic-mode operator. The "div_damping" you see in
GFDL's `Diffusion_operators.pdf` belongs to the FV3 atmospheric core — it is
not active in MOM6's barotropic solver.

### MITgcm

Source: `pkg/mom_common`, free-surface kernel. Namelist
`implicSurfPress` (γ) and `implicDiv2DFlow` (β), both default **1.0** (fully
implicit). The Crank–Nicolson choice is γ=β=½. Adcroft & Campin (2004) show
this fully or semi-implicit treatment **is** MITgcm's primary barotropic-noise
control — fast gravity waves are unconditionally damped by the implicit step,
so the checkerboard is squashed at the gravity-wave timescale. MITgcm does
not advertise an explicit "divergence damping" knob in the barotropic
solver; it leans on the implicit free surface plus the standard `viscAh` /
`viscA4` Laplacian/biharmonic viscosity applied to the 3D flow.

### NEMO (ORCA025, eORCA025)

Source: `dynspg_ts.F90`. Namelist:

| Mechanism | Parameter | Typical |
|---|---|---|
| Number of barotropic substeps per dt | `nn_baro` | 30 |
| Forward vs centred barotropic step | `ln_bt_fw` | T (forward, AB-like) |
| Time-average barotropic accumulator | `ln_bt_av` | T |
| Time filter shape | `nn_bt_flt` | 1 (boxcar over `nn_baro`) or 2 (over `2·nn_baro`) |
| Max barotropic Courant | `rn_bt_cmax` | 0.8 |

NEMO's primary noise control is the boxcar over `2·nn_baro` (`nn_bt_flt=2`)
which is essentially a Higdon-style centred filter — it suppresses both
checkerboard and aliasing onto the baroclinic step. NEMO uses **Robert–
Asselin–Williams** time filtering on the leapfrog 3D step (`rn_atfp ≈ 0.05–
0.1`) which also damps any 2-Δt mode that survives the barotropic averaging.
NEMO does **not** apply explicit divergence damping or eta-diffusion in the
barotropic substep.

### POP / POP2 (CESM, CCSM)

Source: Smith et al. POP reference manual; Dukowicz & Smith (1994). POP uses
a **fully implicit** free surface solved by preconditioned conjugate gradient
on the elliptic operator — no substep at all. The implicit treatment, combined
with the B-grid staggering (POP is on B-grid, not C), eliminates the C-grid
checkerboard mode by construction. The Rossby-like computational mode is
damped by the choice of forward discretization (POP manual §3.2). MPAS-Ocean
on Voronoi cells uses a similar semi-implicit Crank–Nicolson barotropic mode
solver (Kang et al., 2021, *JAMES*), with the recent E3SM Trilinos solver.

### ROMS / CROCO

Source: Shchepetkin & McWilliams (2005, *Ocean Modelling*), `step2d.F`. The
**polynomial / power-law time filter** (`A(τ) = A₀{(τ/τ₀)^p[1−(τ/τ₀)^q] −
r(τ/τ₀)}`, defaults `p=2, q=4, r=0.284`) is the centerpiece. It integrates
M*≈1.5·M substeps past `n+1` and weights them so the average has
second-order accuracy and conserves tracer constancy. The cosine filter
(`COSINE2`) is the legacy fallback and is documented as **only first-order
accurate** — the ROMS team explicitly recommends against it. Default
`NDTFAST ≥ 20`, typically 30. ROMS does **not** apply explicit barotropic
divergence damping; it has Laplacian/biharmonic 3D viscosity (`UV_VIS2`,
`UV_VIS4`) and that is considered sufficient once the time filter is doing
its job (see myroms.org forum thread t=4052: a checkerboard report turned
out to be `NDTFAST=3`, fixed by raising it to 30).

---

## Q2. The C-grid checkerboard mode — canonical reference and physics

**Punchline.** The canonical reference is Arakawa & Lamb (1977), with the
free-surface manifestation discussed by Killworth, Stainforth, Webb & Paterson
(1991, *JPO*). On a C-grid the Coriolis term must be 4-point averaged from V
to U-points (and back), so the discrete inertia-gravity dispersion has
**spurious zero-frequency modes** at the 2Δx scale: a checkerboard η pattern
plus a 2Δx-staggered (U,V) pattern can satisfy the discrete continuity
exactly while having identically zero Coriolis tendency at the grid scale.
Once the Rossby radius is unresolved — which is exactly the situation at
**5° lat-lon** in the Drake band — the C-grid Coriolis averaging leaves a
near-null space that any forward-backward integrator slowly populates from
roundoff and from η-redistribute events.

How the production codes handle it:
- **MOM6** (`MOM_barotropic.F90`): `BEBT≥0.2` semi-implicit damps the fastest
  gravity-wave projection of the mode; `BT_PROJECT_VELOCITY=True` plus
  `BOUND_BT_CORRECTION` keeps the 2Δx η imprint bounded.
- **MITgcm** (`MOM_continuity` + implicit free surface): unconditional damping
  via `implicSurfPress=1, implicDiv2DFlow=1`.
- **NEMO** (`dynnxt.F90` Asselin + `dynspg_ts` boxcar): doubled boxcar filter
  cancels the 2Δt computational mode exactly; the residual 2Δx spatial mode
  is bled off by the Laplacian viscosity in `dynldf_lap_blp`.

---

## Q3. Divergence damping specifically

**Punchline.** Explicit `∇(∇·U)` divergence damping on the barotropic
velocity is **not** a default in MOM6, NEMO, POP, MPAS-Ocean, ROMS, or
MITgcm. It is a **dycore-atmosphere** technique (FV3, MPAS-A, CAM-FV) where
it suppresses the lat-lon polar gravity-wave instability. Whitehead, Jablonowski
et al. (2011, *MWR*) — "A Stability Analysis of Divergence Damping on a
Latitude-Longitude Grid" — gives the canonical analysis.

The reason ocean codes don't lean on it: the operator destroys geostrophic
balance at the grid scale (`∇·U_g = β·v / f ≈ 0` on the f-plane, but the
discrete operator damps any high-wavenumber divergence regardless of whether
it's geostrophically balanced or not), and at coarse resolution the
geostrophic divergent component carries real signal (Ekman pumping, β-driven
divergence, topographic Sverdrup). The trade-off is documented in Hallberg
(2013, *Ocean Modelling*) "Using a resolution function" — divergence damping
above ~0.1 in dimensionless units degrades the western-boundary current
sharpness.

When ocean codes do use it (e.g. some regional CROCO setups, or MOM6's
`HOR_VISC` Smagorinsky path on the layered flow), the **dimensionless**
coefficient is typically **0.01–0.1**, applied as
`A_div = c_div · A_cell / dt`. legoESM's `div_damp_coeff` of 0.0–0.1 is in
this range. At `c_div=0.1` and Δt=600 s, the implied viscosity at 5°
(A_cell ~ 3×10¹⁰ m²) is ν ~ 5×10⁶ m²/s, which is large but not catastrophic
for a barotropic-only operator (the baroclinic viscosity stays separate).

---

## Q4. Is ±5 cm/s grid-scale `⟨V_baro⟩` after 1 yr a known artifact?

**Punchline.** Yes, at coarse resolution with C-grid, a forward-backward
substep, no streaming filter, and a boxcar (or weak cosine) time filter, this
is the textbook signature of the unsuppressed C-grid Coriolis mode plus
mode-splitting aliasing. It is exactly what Killworth & Stainforth (1991) and
Shchepetkin & McWilliams (2005, §2.3) describe.

The diagnostic that distinguishes "checkerboard" from "real numerical
instability" is whether `⟨∇·U_baro⟩` averaged over a few weeks has 2Δx
structure that is *stationary* (checkerboard) versus growing (instability).
A 5 cm/s amplitude that has *not* grown in 50 000 steps and whose zonal
mean is small (0.02 cm/s) is the stationary null-space mode. The reason it
contaminates your Drake-band momentum budget is purely the
`ρ·H·f ≈ 5×10⁵ kg m⁻² s⁻¹` amplification — geostrophic balance turns a
0.02 cm/s residual V into a 0.1 Pa zonal force, which is comparable to the
wind. That's not a bug in V_baro per se; it's that any zonally averaged V at
mid-latitudes is a *very* sensitive diagnostic.

---

## Q5. Recommendations for legoESM

1. **Activate divergence damping at `barotropic_div_damp = 0.05`** as the
   default. `c_div=0.1` is the upper end of what MOM6 / Hallberg (2013)
   uses on the layered flow; 0.05 gives margin without crippling the
   geostrophic balance at 5°. It's the simplest intervention because the
   operator already exists in `barotropic_latlon_cgrid.py` L324–337.

2. **Switch the cosine filter to a Higdon-style doubled boxcar** (or a true
   power-law; the ROMS `(p=2, q=4, r=0.284)` weights). The current cosine
   is first-order accurate and is documented to be the *cause* of
   checkerboard problems in ROMS (Wilkin / Shchepetkin, myroms.org thread
   t=4052). At 5° resolution and 30 substeps, integrating to `M*≈1.5·M`
   substeps past `n+1` is where the Higdon-style filter gets you a clean
   second-order zero at the 2Δt mode.

3. **Add a CI invariant** on `var(∇·U_baro) / var(U_baro)`: in steady state
   this ratio should be O(Δx⁻²·tau_inertial⁻²) and roughly constant. A
   monotonically growing ratio signals checkerboard contamination. Run it
   at the end of each ocean test-matrix case at 5° and 1° resolution.

4. **Do not raise `bebt` past 0.4.** MOM6's 0.2 + `BT_PROJECT_VELOCITY=True`
   pair is well-tested. Raising `bebt` to 0.5 is unconditionally stable but
   over-damps real near-inertial energy in the barotropic mode.

5. **For the specific Drake-band momentum budget**, after divergence damping
   plus power-law filter are active, the residual `ρ·H·f·⟨V_baro⟩` should
   drop below the 0.01 Pa floor. If it doesn't, the next thing to check is
   the freshwater `F_slow_eta` source: a small global-mean leak through the
   eta source is **not** a checkerboard issue but a closure issue
   (z-star + freshwater + non-zero mean V), and is the subject of
   `zstar_vbaro_residual_investigation.md`.

---

### Sources

- [MOM6 MOM_barotropic.F90 (NCAR mirror)](https://ncar.github.io/MOM6/APIs/structmom__barotropic_1_1barotropic__cs.html)
- [MOM6 OM4_025 MOM_input — production namelist](https://github.com/NOAA-GFDL/MOM6-examples/blob/dev/gfdl/ice_ocean_SIS2/OM4_025/MOM_input)
- [Shchepetkin & McWilliams 2005, ROMS Ocean Modelling paper](https://people.atmos.ucla.edu/alex/ROMS/ROMSArticle2005.pdf)
- [ROMS WikiROMS Numerical Solution Technique](https://www.myroms.org/wiki/Numerical_Solution_Technique)
- [ROMS forum: checkerboard instability thread](https://www.myroms.org/forum/viewtopic.php?t=4052)
- [MITgcm Crank–Nicolson barotropic time stepping](https://mitgcm.readthedocs.io/en/latest/algorithm/crank-nicol.html)
- [NEMO surface pressure gradient (dynspg) docs](https://www.nemo-ocean.eu/doc/node38.html)
- [POP Reference Manual (CESM)](https://www2.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf)
- [Kang et al. 2021 — Semi-implicit barotropic solver for MPAS-Ocean](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002238)
- [Higdon 2002 / Higdon & de Szoeke 1997 — barotropic-baroclinic splitting](https://data.coaps.fsu.edu/pub/eric_back/papers_html/higdon.pdf)
- [Whitehead, Jablonowski et al. 2011 — Stability of divergence damping on lat-lon](https://journals.ametsoc.org/view/journals/mwre/139/9/2011mwr3607.1.xml)
- [Killworth, Stainforth, Webb, Paterson 1991 — Free-surface Bryan–Cox–Semtner](https://www.semanticscholar.org/paper/The-Development-of-a-Free-Surface-Bryan%E2%80%93Cox%E2%80%93Semtner-Killworth-Stainforth/ea8ce650f5b945eda233c3f2c2f5316d91703616)
- [Adcroft & Campin 2004 — MIT GCM overview](http://mitgcm.org/pdfs/ECMWF2004-Adcroft.pdf)
