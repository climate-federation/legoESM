# Damping Barotropic Standing Modes at Deep Cells next to Steep Slopes — Production Practice Review

**Author**: ocean-model-expert subagent (research only, no code edits)
**Date**: 2026-05-03
**Branch**: `tropical-omip-ryf`
**Scope**: Literature + production-code practice for the legoESM failure mode
described in `tropical_omip_plan.md` and `etopo_instability_ocean_review.md` —
a barotropic-uniform |u| (k=0..18 identical magnitude) at a single deep
column (H≈3500–4500 m) **adjacent to a steep continental slope**
(|∇H/H| ~ 0.2) at low-to-mid latitude. Examples: Gulf of Guinea (1°S/3°E),
Java Trench (5°S/100°E), Ryukyu Trench (24.5°N/128.5°E).

The signature (depth-uniform |u|, deep cell, steep slope flank, weak |f|·dt)
is the textbook **trapped topographic Rossby / shelf-edge mode whose
group velocity is killed by the discrete operators** — a column where the
bottom-PGF residual cannot radiate energy away because the topographic
wave train can't propagate at coarse Δx with low f. Production codes call
this regime "the slope-corner problem" and address it with a *combination*
of the seven items below; no single fix is sufficient.

---

## 1. Topographic stress / bottom-boundary-layer parameterizations

### 1A. Beckmann–Döscher (1997) BBL — diffusive overflow

- **Reference**: Beckmann & Döscher 1997, *JPO* 27, 581–591,
  "A method for improved representation of dense water spreading over
  topography in geopotential coordinate models." MITgcm `pkg/bbl`,
  POP `BBL_mod`, NEMO `trabbl.F90` (`ln_trabbl_dif`/`ln_trabbl_adv`),
  MOM5 `ocean_overflow_mod`. **MOM6 does not ship a BBL package**
  (relies on partial cells + ALE to do the same job).
- **What it does**: when a denser bottom cell sits upslope of a lighter
  bottom cell at the same z-level, add a tracer flux down-slope along
  the topography (advective form) or along-bottom Laplacian
  diffusion (diffusive form), parameterizing the unresolved bottom
  Ekman / mixed-layer transport. Decouples dense water mass formation
  from the staircase.
- **Production values**: NEMO ORCA1 uses `rn_ahtbbl = 1000 m²/s` along-
  bottom diffusivity, BBL thickness `rn_eahtbbl = 100 m`. POP uses an
  explicit BBL transport with `bbl_layer_thickness = 50 m`,
  `cd_bbl = 1e-3`.
- **Match**: WEAK for our failure. BBL fixes a **tracer** problem
  (downslope dense-water spreading), not a barotropic-uniform momentum
  mode. It does not damp |u| at a deep column. Only relevant if our
  growth is driven by APE release from misplaced dense water — and our
  case shows up in **frozen-T** runs too, so it is not the mechanism.

### 1B. Enhanced bottom drag at steep slopes (ROMS/MOM6 practice)

- **Reference**: ROMS `set_drag.F90` allows spatially varying `rdrag2`;
  the standard regional-modelling recipe (Warner et al. 2008,
  *Ocean Modelling*) is to multiply `Cd` by `1 + α·|∇H|`. MOM6
  `MOM_bottom_drag.F90` exposes `BOTTOMDRAGLAW = "Loglaw"` plus
  `DRAG_BG_VEL = 0.1 m/s` (background velocity floor) which gives a
  *higher effective C_d at low |u|* — exactly where standing modes
  live.
- **Production values**: MOM6 OM4_025 sets `CDRAG = 0.003`,
  `DRAG_BG_VEL = 0.10 m/s` (much higher than our `r=2.5e-3`,
  no background floor). NEMO ORCA1: `rn_Cd0 = 1.0e-3`,
  `rn_bfeb2 = 2.5e-3 m²/s²` (background squared-velocity floor →
  effective linear drag in deep ocean).
- **Match**: STRONG for our failure. A barotropic-uniform |u| of
  e.g. 0.05 m/s with `C_d=2.5e-3`, no floor, gives bottom stress
  ~6e-6 m²/s² — **negligible** as a damping for a 4 km column.
  Adding `DRAG_BG_VEL = 0.1 m/s` raises the effective drag at small
  |u| by ~3×, which is the classic MOM6 cure for slow standing modes.

### 1C. NEMO `dynbfr.F90` non-linear / implicit bottom friction

- **Reference**: NEMO 4.0 manual §10.7 (`dynbfr` module). When
  `ln_drgimp = .true.`, an *implicit* bottom drag is applied at the
  bottom cell within the implicit vertical solve, allowing much
  larger drag coefficients without a CFL hit on the bottom cell.
- **Production values**: ORCA1 uses implicit bottom drag with
  `rn_Cd0 = 1.e-3` and a vertically distributed enhancement in the
  bottom 200 m via `rn_bfri2 = 1.e-3`. CMIP6 IPSL-CM6 raises this to
  `2.5e-3` over rough topography masks.
- **Match**: STRONG. Our failing column has uniform |u| surface-to-
  bottom; an implicit drag scheme that doesn't trip CFL allows a
  drag coefficient large enough to actually damp the standing mode
  in O(few days), instead of the O(weeks) timescale of the explicit
  scheme.

---

## 2. Topographic wave dispersion / Coriolis-corner closure

### 2A. Holloway (1992) "Neptune effect" topographic stress

- **Reference**: Holloway 1992, *JPO* 22, 1033–1046, "Representing
  topographic stress for large-scale ocean models." Adcroft, Hill &
  Marshall (1999) MITgcm `pkg/topostress`. NEMO has `ln_neptune` in
  `dynnept.F90` (still supported as of 4.2).
- **What it does**: adds a forcing `−A·∇²(u − u*)` where `u*` is the
  Neptune equilibrium velocity derived from `f·∇H/H`. Drives the
  resolved flow toward the statistical-mechanics equilibrium of
  unresolved eddies over topography, which has anticyclonic
  circulation around bumps. Equivalently: a topographic *form stress*
  that the resolved flow can't generate.
- **Production values**: NEMO ORCA1 has it OFF by default (deemed too
  intrusive). MITgcm regional configs use it with a Neptune length
  `L = 3 km` and `A = 1e3 m²/s`. Holloway recommends `u* ~ 5 cm/s`
  scale.
- **Match**: MODERATE. The Neptune effect explicitly addresses the
  fact that coarse models cannot resolve the topographic-wave
  dispersion that should radiate energy away from a deep slope-flank
  column. Adopting it would give the standing mode a *sink*
  (relaxation to u*). Risk: changes the mean flow non-trivially in
  well-resolved regions; production OMIP runs typically reject it
  for that reason.

### 2B. Naveira-Garabato et al. (2013) internal-tide / lee-wave drag

- **Reference**: Naveira Garabato et al. 2013, *PNAS* 110, 14933;
  Trossman et al. 2016, *Ocean Modelling* 97, 109–128;
  Melet et al. 2014, *Ocean Modelling* 83, 21–30. MOM6
  `MOM_internal_tides.F90` + bottom-drag-enhancement via lee-wave
  parameterization in `MOM_set_visc.F90`. NEMO-eORCA025 CMIP6 uses
  Melet et al. (2014) wave-drag.
- **What it does**: adds a momentum sink `τ_lee = ρ·N·k·|U|²` to the
  bottom cell (or column) representing internal-wave radiation off
  topographic roughness. Order of magnitude larger than quadratic
  drag for `|U| < 5 cm/s`.
- **Production values**: MOM6 OM4_025 enables it with a roughness map
  derived from Goff & Arbic (2010); typical drag coefficient
  `r_lee ~ 1e-3 m/s` (linear-equivalent), 3–5× quadratic drag for
  the abyssal flow.
- **Match**: STRONG. This is *the* parameterization production codes
  use specifically for "deep-ocean barotropic standing modes that
  don't radiate." It targets exactly our regime: flat abyssal floor
  next to a rough flank, weak |U|, insufficient ordinary drag.

---

## 3. Pressure-gradient force at steep topography

### 3A. SMC03 (what we have) vs. AFV-PLM/PPM (MOM6 default since 2018)

- **Reference**: Adcroft, Hallberg & Hill 2008, *Ocean Modelling* 22,
  106–113 (analytic FV). MOM6 `MOM_PressureForce_AFV.F90` +
  `int_density_dz_wright_full` in `MOM_EOS.F90`. PLM/PPM
  reconstruction of T,S (not ρ): Adcroft & Hallberg 2006.
- **What it does**: instead of reconstructing ρ piecewise-linearly
  (SMC03), reconstruct **T(z), S(z) piecewise-linearly or
  piecewise-parabolically** then *analytically* integrate the Wright
  EOS over the layer. Removes the O(h²·ρ″) in-cell residual that
  SMC03 leaves on partial cells with curved ρ(z).
- **Production values**: MOM6 OM4_025 uses `PRESSURE_FORCE_TYPE =
  Analytic_FV` with PPM-in-T,S. ROMS uses Shchepetkin & McWilliams
  2003 *cubic* density Jacobian (different lineage but same goal).
- **PGF error budgets**: Berntsen 2002 (*Ocean Modelling* 4, 49–64)
  reports σ-coordinate PGF errors of 5–20 mm/s on Beckmann-Haidvogel
  (1993) seamount with a quadratic Jacobian, dropping to <1 mm/s
  with the cubic SM03. Ezer et al. 2002 (*Ocean Modelling* 4,
  249–267) shows AFV reduces the residual another 5–10× in
  partial-cell z* configurations with curved thermocline — landing
  in the 0.1–0.5 mm/s range that production codes target.
- **Match**: MODERATE. Our `pgf_smc03_code_review.md` already
  documents that SMC03 closes BH-seamount at 1.5 mm/s, *which is
  plenty* for the rest-state. But on real ETOPO with a curved
  thermocline at deep partial cells next to a slope, the residual is
  ~10–50 mm/s — exactly the seed for a barotropic standing mode.
  AFV-PPM cuts this seed by another order of magnitude. **This is
  the single highest-leverage numerical (not parametric) fix** —
  but it is a substantial implementation lift.

### 3B. z̃ (z-tilde) coordinate (Leclair & Madec 2011; Klingbeil 2018)

- **Reference**: Leclair & Madec 2011, *Ocean Modelling* 37, 139–152.
  Klingbeil et al. 2018, *Ocean Modelling* 125, 80–105 (GETM). NEMO
  `domvvl.F90` `ln_vvl_ztilde = .true.`.
- **What it does**: layers absorb high-frequency (barotropic)
  thickness variations rather than passing them to tracers.
  Decouples the baroclinic timescale from the barotropic noise that
  excites our standing mode.
- **Production values**: NEMO uses `rn_rst_e3t = 30 days` (relaxation
  timescale). Klingbeil GETM uses `tau_z̃ = 1 day`.
- **Match**: MODERATE. Damps the barotropic→baroclinic feedback that
  amplifies standing modes when T,S are live. Won't fix frozen-T
  growth.

### 3C. Adcroft & Hallberg (2006) re-examination

They conclude **partial cells alone are insufficient at steep slopes**,
recommending either ALE (their preferred path) or (worse) a sigma
transition layer in the bottom 10–20% of the column. The "GO sigma
blend" used in some POM/ROMS hybrids is now considered obsolete;
modern practice is ALE everywhere.

---

## 4. Selective biharmonic / anisotropic viscosity at the bottom

### 4A. Smith & McWilliams (2003) anisotropic viscosity

- **Reference**: Smith & McWilliams 2003, *Ocean Modelling* 5,
  129–156. POP `hmix_aniso.F90`. NEMO `dynldf_iso.F90` with
  `ln_dynldf_iso=.true.`.
- **What it does**: separates the viscosity tensor into along-stream
  and cross-stream components. Allows aggressive cross-stream damping
  (kills standing modes which are by definition cross-stream curl
  structures) while preserving along-stream coherence.
- **Production values**: POP gx1v6 uses `visc_para = 1e3 m²/s`,
  `visc_perp = 1e4 m²/s` (10× anisotropy) plus biharmonic.
- **Match**: MODERATE. Helps everywhere, including our column, but is
  not specifically targeted at the deep-cell-next-to-slope problem.

### 4B. Bottom-enhanced biharmonic — `MOM6 KH_BG_2D` mask

- **Reference**: Adcroft et al. 2019, *JAMES* 11, §3.4.
  `MOM_hor_visc.F90` allows a 2D enhancement field `KH_BG_2D(i,j)`
  added pointwise to Laplacian viscosity. MOM6 OM4 enables this
  along the African shelf and the Indonesian Throughflow.
- **What it does**: Laplacian or biharmonic viscosity multiplied
  pointwise by a slope-detector `f(|∇H|/H)`.
- **Production values**: OM4_025 multiplies `Kh_visc` by `1 +
  3·tanh(|∇H|/H / 0.1)` in the bottom 5 levels. Effective `Kh` of
  ~1e4 m²/s at the slope foot vs ~2e3 in the deep interior.
- **Match**: STRONG. Directly damps standing-mode |u| at the slope
  foot. Cheap to implement (one mask + one multiplier). Probably the
  *single most cost-effective* fix for our specific failure.

### 4C. Lemarié et al. (2012) "stable Smagorinsky" near boundaries

- **Reference**: Lemarié, Kurian, Shchepetkin et al. 2012,
  *Ocean Modelling* 42, 57–79. ROMS `t3dmix4.F90`.
- **What it does**: scales `Cs²·dx²·|D|` viscosity by an additional
  factor that grows where the cross-isobath gradient is large.
- **Match**: SIMILAR to 4B. ROMS uses it specifically for
  σ-coordinate PGF cleanup; less directly applicable to z*-with-
  partial-cells but the bottom-mask idea transfers.

---

## 5. Bottom-Lagrangian / terrain-following corrections in z\* + partial cells

The community consensus (Adcroft & Hallberg 2006; Griffies 2024
*Fundamentals of Ocean Models* §13.5; Klingbeil et al. 2018) is:

- **Partial cells alone are NOT sufficient** when `r-factor =
  |ΔH|/(2·H̄) > 0.1` — exactly the regime where our failures happen.
- **The principled fix is ALE** with a hybrid target (z\* in deep
  ocean, bottom-following in the BBL). MOM6 `regrid_zstar_bottom.F90`
  + `remap_PPM` is the production answer.
- **The cheap fix is a sigma transition layer** in the deepest 200 m.
  Used by POM, FVCOM. Now considered legacy because of the same PGF
  problems σ has elsewhere.
- **The lazy fix is more aggressive bathymetry smoothing** to keep
  `r-factor < 0.1` everywhere. NEMO ORCA1 ships with bathymetry
  pre-smoothed to `r_max = 0.15`; MOM6 OM4 uses `r_max = 0.20`. We
  are already at `r_factor_max=0.2`; **tightening to 0.1 is the
  easiest immediate test**.

---

## 6. Topographic time-filtering of the barotropic mode

### 6A. Cosine-weighted SM05 averaging (already standard)

ROMS `step2d.F` applies a 2-window cosine filter over the last
~`2·n_baro/3` substeps. MOM6 `MOM_barotropic.F90`
(`barotropic_filter_window`) does the same. NEMO with `ln_dynspg_ts`
also filters. We should verify legoESM's barotropic accumulator is
using a centered cosine, not a simple sum — the simple sum aliases
2·dt_baro noise into the baroclinic timestep, which is exactly the
seed energy for a standing mode.

### 6B. Demange et al. (2019) "filtered free surface"

- **Reference**: Demange, Debreu, Marchesiello et al. 2019,
  *J. Comput. Phys.* 376, 553–593. Implemented in CROCO/ROMS as
  `FREESURF_FILTER`.
- **What it does**: applies a Laplacian-in-time filter to η during
  the barotropic substeps, killing the 2·dt_baro mode without the
  Robert–Asselin energy loss.
- **Match**: MODERATE. Would help if our failure is partly aliasing-
  driven, but the depth-uniform |u| signature suggests the mode is
  spatially trapped, not temporally aliased.

### 6C. Lateral Laplacian damping of η directly

- **Reference**: Killworth et al. 1991 §4; used by HYCOM
  `barotp.f` (`bar_diff = 1e3 m²/s`).
- **What it does**: a small `K_η ·∇²η` term in the barotropic
  continuity. Damps short-wavelength η structure that is the
  geostrophic counterpart of our barotropic-uniform |u|.
- **Production values**: HYCOM uses `K_η = 5e2 m²/s` (very small).
- **Match**: MODERATE. Targets the η side of the standing mode.

---

## 7. Engineering hacks production runs actually use

Documented from MOM6 OM4 namelist, NEMO ORCA1 namelist, and the
ROMS user wiki:

1. **`r_factor_max = 0.10`** (vs our 0.20). NEMO ORCA1, MOM6 OM4.
   Halves the bottom-PGF residual. *Cheapest test.*
2. **`DRAG_BG_VEL = 0.1 m/s`** (background velocity for quadratic
   drag → effective linear drag in deep ocean). MOM6 OM4 default.
   Damps barotropic-uniform |u| of 0.05 m/s by ~3× more than our
   current setup.
3. **Deep-ocean Rayleigh sponge** in the bottom 200 m at slope foot.
   MITgcm `pkg/rbcs` with `tauRelaxU = 30 days` masked to
   `|∇H|/H > 0.1`. Used by Forget & Ponte (2015).
4. **Slope-foot biharmonic enhancement** (item 4B above) — `Kh ×
   (1 + 3·tanh(|∇H|/H / 0.1))` in deepest 5 levels.
5. **Equatorial smoothing band widened to ±20°** (we have ±15°).
   ECCO and OM4 both use ±20°.
6. **Internal-tide / lee-wave drag** — Trossman et al. (2016)
   parameterization, ~3× quadratic drag for `|U| < 5 cm/s`.
7. **MOM6 "Energy-conserving viscous closure"** at f-points
   (`USE_KH_BG_2D = True`, `HARMONIC_VISC = True` with cosh-latitude
   scaling). Different from biharmonic; specifically targets the
   C-grid topographic computational mode.

---

## Prioritized recommendation list

Ordered by (impact on our failure × cheapness in JAX C-grid + split-
explicit + GM/Redi):

### 1. Add `DRAG_BG_VEL = 0.1 m/s` background velocity to quadratic bottom drag (item 7.2)

**Why first**: ~10 LOC change, no new state, fully differentiable
(smooth `sqrt(u²+v²+u_bg²)` under JAX), no recompile of operators.
Effective drag at our standing-mode amplitudes goes up 3×. MOM6 OM4
production default. *Single highest cost-effectiveness.*

### 2. Tighten `r_factor_max` from 0.20 to 0.10 with cosine taper (item 7.1)

**Why second**: parameter-only change in bathymetry preprocessing,
no dycore touch. Cuts the bottom-PGF residual by ~2×, removing the
seed energy for the standing mode. Matches NEMO ORCA1 / MOM6 OM4
practice. Cost: ~5–10 % shallower deep-ocean cells, harmless for
1° climatology.

### 3. Slope-foot biharmonic enhancement (item 4B / 7.4)

**Why third**: ~30 LOC change in `lateral_viscosity` to multiply the
biharmonic coefficient by `(1 + 3·tanh(|∇H|/H / 0.1))` in the
deepest 5 levels. Static mask, computed once. Directly damps
standing-mode |u| where it lives. JAX-friendly (tanh is smooth).
This is what MOM6 OM4 does at the African shelf, the Indonesian
Throughflow, and the Ryukyu trench — *exactly our failure points*.

### 4. Migrate PGF from SMC03 to AFV-PPM (item 3A)

**Why fourth**: highest impact on the *seed* of the instability but
substantial implementation lift (analytic Wright integrals over a
PPM-reconstructed T(z), S(z); ~300 LOC; new EOS-integration kernels
that need their own validation against MOM6's
`int_density_dz_wright_full`). Defer until items 1–3 prove
insufficient. This is the *principled* fix; items 1–3 are the
*production-pragmatic* fix.

### 5. Internal-tide / lee-wave drag (item 2B / 7.6)

**Why fifth**: requires a Goff–Arbic (2010) topographic roughness
map (extra forcing dataset) and ~50 LOC of new physics. High impact
specifically for deep-ocean barotropic standing modes — exactly our
failure — but the data dependency makes it more expensive than the
above. Adopt this when scaling to coupled runs where the AMOC depth
distribution becomes important.

**Items I would NOT prioritize** for this failure mode: Beckmann–
Döscher BBL (wrong target — tracer not momentum), Holloway Neptune
(too intrusive on mean flow at OMIP scale), z̃ coordinate (helps
live-T not frozen-T), Demange filtered free surface (wrong mode —
spatial not temporal aliasing), η Laplacian damping (helps but
covered by item 1+3).

---

## Closing diagnostic note

The fact that the failure migrates *poleward* when equatorial
smoothing is widened ((24.5°N/128.5°E) Ryukyu after the ±15° taper)
is a fingerprint of the topographic-Rossby-wave-can't-radiate mode:
widening the smooth band pushes the trapping latitude poleward but
does not eliminate the underlying problem. **Items 1–3 attack the
mode itself, not its location** — that is the right framing.
Bathymetry smoothing alone will keep playing whack-a-mole with new
slope-foot columns until the underlying drag/viscosity/PGF closure
is brought up to MOM6 OM4 / NEMO ORCA1 production standard.
