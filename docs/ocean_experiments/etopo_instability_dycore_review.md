# ETOPO 30-day Rest-State Instability — Dycore-Expert Review

**Reviewer**: dycore-expert subagent
**Branch**: `ocean-partial-cells`
**Subject**: Should we ship GM/Redi + biharmonic + Smagorinsky to "fix"
the live-ETOPO 30-day NaN, or is there a genuine discretization bug
hiding under the slow frozen-T growth?

**TL;DR — verdict**: Plan **(b/c hybrid)**, not **(a)**.  The slow
frozen-T growth (e-fold 6–20 days) is too fast to be a resolved
topographic Rossby wave at 3° and too slow to be a generic CFL
instability — it is, with high prior probability, a **partial-cell
PGF/face-thickness inconsistency** at the active-vs-inactive face
that production codes (MOM6, MITgcm) actively *fix*, not damp.  Adding
biharmonic + Smagorinsky now would mask a known class of bug.  The
live-T NaN is a separate, dominant **TS–PGF feedback** that likely
needs at least horizontal Laplacian/biharmonic tracer mixing to be
controllable on coarse ETOPO regardless — but that should be added
as physics, not as a stability crutch.  Concrete dycore audit list
below.

---

## 1. What sort of mode fits a 6–20-day e-fold on a frozen-T,
   stratified, partial-cell, real-bathymetry rest state?

### 1.1 Ruling out resolved physical modes

At Δλ = 3° (~330 km at the equator, ~165 km at 60°N) and 20 levels,
the gravest **resolved** topographic Rossby wave on real bathymetry has
period ≳ 30–100 days (Rhines & Bretherton 1973; LaCasce 2017,
*J. Phys. Oceanogr.* 47, eq. 2.7).  These are *neutral* under linear
theory; the model can only render them *unstable* via discrete
truncation error.  The frozen-T growth rate (e-fold 6–20 days) is
**within an order of magnitude of the slowest physical timescale** in
the system, which is exactly the regime where discrete error of the
*pressure-gradient* and *thickness-weighted continuity* terms shows
up as a slow exponential mode rather than as a CFL blowup.

The frozen-T case also rules out the obvious physical instabilities
(baroclinic / symmetric / inertial): T,S do not evolve, so no
APE-to-EKE pathway exists.  The energy source for the growing flow
**must** be an inconsistency between the PGF and either (a) the
continuity / thickness-weighted mass flux divergence, or (b) the
barotropic-baroclinic split.  This is a **discrete adjoint
inconsistency**: in a continuous Boussinesq system at rest, the
prognostic energy equation has dE/dt = 0 identically; if discrete
PGF · u + continuity · p does not telescope, the discrete energy
budget gains a term proportional to the rest-state PGF residual times
the flow it generates, which is precisely the exponential signature.

### 1.2 The frozen-T E1/E3 evidence

E1 (uniform T,S) over ETOPO is stable: 4.2 mm/s, no growth.  Therefore
the partial-cell *geometry alone* does not drive the mode.

E3 (flat-bottom, sin²(lat) T) is bounded at 96 mm/s but does not NaN:
horizontal density gradients alone, on a flat z* grid, excite a
saturated mode (likely a numerical western-boundary trapped wave or
the well-known Arakawa-C grid f-plane geostrophic adjustment artefact;
Adcroft 1995, *Numerical algorithms for use in a dynamical model of
the ocean*, §6.3).  Bounded — saturates against bottom drag.

The killer combination is **stratification + partial-cell
geometry over real bathymetry**.  Frozen-T then exponentiates at
~10-day e-fold, live-T at ~2-day e-fold.  This is the dycore
signature, not a parameterization gap.

### 1.3 Most plausible mechanism

Ranking against the candidate list in the prompt:

1. **Highest prior — partial-cell PGF residual at level transitions
   between adjacent columns with different `bot_level`.**  The SMC03
   piecewise-linear reconstruction in `latlon_cgrid_operators.py`
   eliminates the rest-state residual *within a cell* with linear
   ρ(z), but the bottom-active boundary condition forces
   `σ_bot = Δρ_top` (one-sided slope), and the harmonic-mean
   `σ_k` at the cell *just above* the partial seafloor uses
   `Δρ_bot = (ρ_k − ρ_{k+1})/(z_k − z_{k+1})` where `(z_k − z_{k+1})`
   is the *partial* spacing — different in adjacent columns when
   `bot_level` differs.  Two columns at the same level k can therefore
   reconstruct different ρ at the face-target depth, leaving a
   per-face PGF residual that scales with curvature of ρ(z) and the
   *difference* in partial-cell thickness across the face.  ETOPO
   has many adjacent-column `bot_level` jumps; the seamount BH test
   has very few (one ridge).  This is consistent with: **BH passes,
   ETOPO fails, the residual has the right cross-bathymetry
   structure, and it is invisible in E1 (uniform T) and E2 (no
   topography)**.  See §3.2 for a concrete prediction.

2. **Highest prior (tied) — `min(h_W, h_E)` face-thickness
   reconstruction at active/inactive transitions, used both in the
   continuity flux and in `U_bar` (the depth average that defines
   the baroclinic perturbation).**  Adcroft, Hill & Marshall (1997),
   *Mon. Wea. Rev.* 125, eq. 13–15, define `hFacW = min(hFacC_L,
   hFacC_R)` for the *flux* face thickness, but this is not the
   thickness used to compute the *kinetic-energy density* nor the
   one used in the depth-mean projector.  Inspection of
   `ocean_pe_latlon_cgrid.py:842–854` shows the same `h_u =
   min_cell_to_uface(h_k)` is used for (i) mass-flux divergence,
   (ii) `U_bar = Σ u·h_u / Σ h_u`, and (iii) the PV thickness
   `h_vtx` at vertices is the *4-cell average* of cell-center
   thicknesses, **not the min**.  This is an inconsistency: the
   flux-form continuity equation and the vector-invariant momentum
   equation use *different* effective layer thicknesses on partial
   cells.  In a flat-bottom z* limit they coincide (h_u = h_v = h_k);
   on partial cells with adjacent-column `bot_level` mismatch they
   do not.  This is the classic root cause of the "spurious flow on
   stratified seamount" problem that Pacanowski & Gnanadesikan
   (1998), *Mon. Wea. Rev.* 126, eq. 5–6, fix by *consistently*
   using the *same* `min`-rule layer thickness in PV/h, KE, and the
   momentum flux.  Our q-at-vertex `h_vtx = 0.25*(...)` does not
   do that.

3. **Medium prior — barotropic-baroclinic split with implicit-CN
   barotropic + perturbation baroclinic on partial cells.** The
   slow-forcing equation in `ocean_tendency_common.py`/
   `barotropic_common.py` is depth-averaged with the *same* `min`-rule
   `h_u`.  The Hallberg & Adcroft (2009), *Ocean Modelling* 29, §3.2
   column-sum invariant requires `Σ_k h_u(k) = Σ_k h_k|_face` to hold
   *exactly* for the slow forcing to project cleanly onto the
   barotropic mode.  With `h_u = min(h_W, h_E)` per level, this is
   satisfied identically only when both columns have the same
   `bot_level`.  When they do not, the *column-summed* min-thickness
   is **strictly less than** either column's bottom depth on the
   face, and the slow-forcing residual leaks into the barotropic
   solver.  See §3.3.

4. **Low–medium prior — Coriolis FB on partial-cell levels.**  The
   forward-backward Coriolis update is unconditionally stable for
   inertial oscillations *on a uniform grid*; on a partial-cell
   grid the discrete Coriolis tendency scales with `f` only and is
   thickness-independent, so this is unlikely to be a root cause
   on its own.  But the *interaction* with #2 above is real: if
   `U_bar` is wrong because of inconsistent `h_u`, Coriolis acts on
   a wrong projector and the geostrophic balance the model is
   trying to settle into is itself inconsistent.

5. **Low prior — KE gradient, momentum advection, vert advection.**
   Frozen-T case shows monotonic growth without a sharp grid-scale
   structure (E3 saturates against drag in the same regime).  The
   2Δz signature documented in `density_jacobian_pgf_plan.md` was
   the *BH* mode, which the SMC03 PGF closed.  ETOPO's failure
   mode is broader and slower — that is consistent with a continuity
   / barotropic-projector inconsistency, not with a column-local
   PGF spike.

---

## 2. Literature on "naked" rest-state runs at coarse resolution
   over real ETOPO

Production codes do **not** rely on GM/Redi/Smagorinsky to suppress
rest-state residual flow over realistic bathymetry.  They rely on
three discrete consistency conditions:

1. **MITgcm** (Adcroft, Hill, Marshall 1997 §5; Adcroft & Campin
   2004, *Ocean Modelling* 7, eq. 18–22): partial-cell PGF uses the
   "shaved" face reference depth, *plus* the layer thicknesses used
   for mass continuity, KE, and PV are derived from the *same* `hFacW`
   /`hFacS` arrays.  That is the canonical lever; without it MITgcm
   blows up on stratified ETOPO at 1° in days.

2. **MOM6** (Adcroft et al. 2019, *JAMES* 11, §2.4; Griffies et al.
   2020 GMD paper, §4.6): uses **finite-volume Lagrangian remap**
   (ALE) so layer thicknesses adapt to the flow and the partial-cell
   geometric inconsistency at the *fixed* z* boundary disappears.
   For OMIP runs at 1°, MOM6 *does* run with rest-state checks
   passing without GM/Redi (GM is added to make the *physical*
   eddy-permitting comparison sensible, not to cure stability — see
   Hallberg 2013, *Ocean Modelling* 72, §6 for the explicit
   distinction).

3. **NEMO** (Madec et al. 2017, NEMO 4.0 manual §6.5–6.6): the
   "step-z" and "partial-step" PGF schemes both include the
   density-Jacobian face reference; rest-state residuals are
   documented to be a few mm/s at 1° over ETOPO *without* GM (NEMO
   reference manual, §6.6.2 and §17.3).

4. **ROMS** (Shchepetkin & McWilliams 2003, *J. Geophys. Res.* 108,
   §4): the density-Jacobian PGF *is* the foundational paper for what
   you implemented; ROMS is a σ-coordinate model where the analogous
   issue is the steep-bathymetry hydrostatic-error mode.  ROMS rest
   states pass with the SMC03 PGF, *no* viscous closure required, on
   realistic bathymetry (S&M03 §6, fig. 7).

So no, the literature does *not* support adding viscous closures as
a substitute for getting the discretization right.  GM/Redi suppresses
*physical* diapycnal flux from numerical isopycnal-slope error;
biharmonic suppresses grid-scale enstrophy cascade; Smagorinsky
parameterizes shear-driven sub-grid mixing.  None of them fix a
**rest-state PGF/continuity inconsistency**.  Adding them to a model
that has such an inconsistency hides the bug for a while, then it
re-emerges as a "model drift" or "spurious upwelling" once you start
running with forcing.

---

## 3. Concrete things to check before adding any closure

### 3.1 Verify the slow frozen-T mode is grid-scale or column-pair-scale

Plot frozen-T `u` at day 10 and 20 with two diagnostics:
- Spatial map of `|u|` at level 10 (sub-thermocline).  If the mode
  lives *along* contours of `bot_level` (the boundary between columns
  with different deepest active level), that is dispositive evidence
  for either #1 or #2 above.
- Vertical profile at the columns where `|u|` is max.  If `u` has a
  **bottom-trapped** structure (max at the deepest active level,
  decaying upward) it is the partial-cell PGF residual (#1).  If `u`
  is **depth-uniform** with a sign that flips across the
  `bot_level`-discontinuity, it is the barotropic-projector leak (#2/
  #3).

This test is cheap (one diagnostic dump) and decides the next step.

### 3.2 Audit the SMC03 σ at the bottom-active cell with partial
   thickness

Inspect `reconstruct_harmonic_slopes` (lines 1995–2041): the
boundary branch `has_top & ~has_bot → σ = Δρ_top` uses
`(ρ_above − ρ_self)/(z_above − z_self)` where `z_self` is the
partial-cell centroid.  In an adjacent column with deeper
`bot_level`, the analogous cell at the *same* k is a *full* cell
with a different centroid, so its `Δρ_top` uses a different
denominator.  At the face-target depth `min(z_c_W, z_c_E)`, the
two reconstructions agree exactly only if ρ(z) is linear *and*
both σ values come from the same denominator — which they do not at
the bottom-active level.  For nonlinear ρ(z) (centroid-aware T-init
in the original case is monotone but curved), this leaves a
non-zero residual that scales with the *curvature* of ρ(z) and the
*difference* in partial-cell thickness.

**Test**: re-run frozen-T with **linear T(z)** init (constant
`dT/dz`).  If frozen-T then drops to <5 mm/s over 30 days, that is
proof the bottom-cell σ is the residual driver and the fix is to
either (i) use the same one-sided slope in adjacent columns at a
shared face, or (ii) use a *cell-interface* (Lagrange) reconstruction
that sees both columns' density values at the face simultaneously.
This is the design Engwirda & Kelley (2016), *GMD* 9, §3, push for
on unstructured grids.

### 3.3 Audit the column-sum invariant on ETOPO

Compute `Σ_k h_u(k)` and compare to `min(H_W, H_E)` at every u-face
(both should equal the wet-column depth at that face per Adcroft–
Hill–Marshall 1997 eq. 11).  If `Σ_k min(h_W(k), h_E(k))` deviates
from `min(Σ h_W, Σ h_E)` anywhere with `bot_level_W ≠ bot_level_E`,
that quantifies the slow-forcing leak.  This is a one-line numpy
diagnostic, no code change.

If the deviation is non-zero, the fix is the **two-thickness
convention**: use `h_u^flux = min(h_W, h_E)` at each level for mass
flux, but `h_u^col` such that `Σ h_u^col = min(H_W, H_E)` (i.e. give
the residual mass back to the bottom-active level on the shallower
side).  See Pacanowski & Gnanadesikan (1998), *Mon. Wea. Rev.* 126,
eq. 5; Adcroft & Campin (2004), eq. 22.  This is the classic "missing
mass" fix; it costs ~10 lines and is consistent across PGF, KE, PV,
and the depth-average.

### 3.4 Verify the implicit-CN barotropic well-posedness on ETOPO

Compute the condition number of the discrete Helmholtz operator on
ETOPO bathymetry (sparse, ~n_lat·n_lon nodes; one-time cost).  Coastal
H~50 m vs. abyss H~5000 m gives a ratio of 100; CN-implicit handles
that fine *if* the operator is built from the correct depth array.
Confirm `H_u` in the barotropic solver matches `Σ_k h_u(k)` from §3.3.
This is checkbox-level but it has bitten every C-grid ocean group.

---

## 4. The live-T NaN is a separate problem

Frozen-T grows to 51 mm/s in 30 days, no NaN.  Live-T (E4 and the
original) NaN at day 16–19 with e-fold ~2 days.  The live-T loop is:

  PGF residual → spurious u → tracer advects T,S → new ρ gradient →
  larger PGF → faster u → ...

This is a *positive feedback*, not a new instability mechanism.  The
seed is the rest-state PGF residual; the runaway is the tracer
advection feeding it.  Fix the seed (§3.2/§3.3) and the runaway
should slow dramatically.  A residual amount of horizontal tracer
mixing (Laplacian, B_h ~ 1e3 m²/s at 3°) is justifiable on physical
grounds (sub-grid eddy mixing is real at coarse resolution; this is
the GM/Redi case), but only after the seed is gone.  Otherwise you
are tuning the closure to suppress a numerical signal, which sets a
trap when you later run at higher resolution and the seed shrinks
unevenly with grid spacing.

---

## 5. Verdict

**Plan (b) with a bounded (c) safety net.**  Concretely:

1. Run §3.1 (one diagnostic dump) — identifies whether the frozen-T
   mode is partial-cell-PGF or barotropic-projector.  ~1 hour.
2. Run §3.2 (linear T(z) frozen-T re-run) — definitive on whether
   bottom-cell σ is the residual driver.  ~1 hour.
3. Run §3.3 (column-sum invariant audit on ETOPO) — definitive on
   whether `min(h_W, h_E)` consistency is broken.  ~10 minutes.
4. Based on (1)–(3), implement the *single* discretization fix that
   matches the diagnostic.  Likely candidates, in order of probability:
   (a) consistent partial-cell thickness across PGF, mass flux, KE,
       PV (Pacanowski & Gnanadesikan 1998); (b) cell-interface
       density reconstruction at the bottom-active face (Engwirda &
       Kelley 2016); (c) "missing mass" two-thickness convention
       (Adcroft & Campin 2004 eq. 22).
5. Re-run frozen-T and live-T.  Frozen-T target: <5 mm/s/30 days
   (the BH target).  Live-T target: bounded, no NaN at 30 days.
6. *Then* add GM/Redi as **physics**, not as **stability**, with
   coefficients set from the literature (Gent-McWilliams κ_GM ~
   600–1000 m²/s; Redi κ_iso ~ 600 m²/s) at 1°–3° (Griffies et al.
   2015 OMIP-2 protocol).  Do not tune them to keep the model
   stable.  If they need to be larger than the literature values to
   suppress drift, the discretization is still wrong.
7. Do not add Smagorinsky or biharmonic momentum closure unless
   step (5) shows residual grid-scale enstrophy cascade.  If you
   *do* need biharmonic, set B_h from the Griffies-Hallberg (2000)
   *Mon. Wea. Rev.* 128 §4 grid-Reynolds-number rule; do not
   tune to "make ETOPO not blow up".

If steps (1)–(3) reveal nothing actionable (i.e., we have already
done the right things and the residual is below what we can resolve
with this PGF), then move to plan (c) with a **documented**
numerical-closure justification: state the residual flow scale, state
the GM/biharmonic coefficient values used, state that they are
literature values not stability-tuned, and put a regression test on
frozen-T 30-day rest-state at <50 mm/s.  Never (a).

---

## References

- Adcroft, Hill, Marshall (1997), *Mon. Wea. Rev.* 125, 2293–2315.
  Partial-cell PGF and `hFac` machinery.
- Adcroft & Campin (2004), *Ocean Modelling* 7, 269–284.
  Rescaled vertical coordinate; eq. 18–22 are the consistent
  partial-cell thickness convention.
- Pacanowski & Gnanadesikan (1998), *Mon. Wea. Rev.* 126, 3248–3270.
  Two-thickness convention; eq. 5–6.
- Shchepetkin & McWilliams (2003), *J. Geophys. Res.* 108(C9), 3090.
  Density-Jacobian PGF; §4 is the formulation we implemented.
- Hallberg (2013), *Ocean Modelling* 72, 92–103.  Resolution-aware
  GM coefficients; §6 distinguishes "physics" from "stability".
- Hallberg & Adcroft (2009), *Ocean Modelling* 29, 15–26.  Column-
  sum invariant in barotropic split.
- Griffies & Hallberg (2000), *Mon. Wea. Rev.* 128, 2935–2946.
  Biharmonic viscosity grid-Reynolds-number rule.
- Engwirda & Kelley (2016), *GMD* 9, 4577–4598.  Cell-interface
  density reconstruction on unstructured meshes.
- LaCasce (2017), *J. Phys. Oceanogr.* 47, 2715–2725.  Topographic
  Rossby waves on real bathymetry.
- Madec et al. (2017), NEMO 4.0 reference manual, §6.5–6.6, §17.3.
- Adcroft et al. (2019), *JAMES* 11, e2019MS001726.  MOM6 ALE.
- Griffies et al. (2020), *GMD* 13, 3231–3296.  OMIP-2.
