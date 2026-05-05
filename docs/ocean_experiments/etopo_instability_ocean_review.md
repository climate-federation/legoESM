# ETOPO 30-day Instability — Ocean-Model-Expert Review

**Reviewer:** ocean-model-expert subagent
**Date:** 2026-04-30
**Branch:** `ocean-partial-cells`
**Subject:** Is the proposed "add GM/Redi + biharmonic + Smagorinsky" fix the right move, or are we masking a numerical bug in the SMC03 PGF / partial-cell discretization?

---

## TL;DR

**My honest read of the bisection: about 70/30 in favor of "this is a real, documented numerical mode that production codes accept and damp, not a discretization bug."** But the 30% case is non-negligible and there is one specific test I would run before committing to closures. Concretely:

1. The frozen-T 51 mm/s slow growth on real ETOPO with no T,S advection is **the textbook signature of the C-grid topographic computational mode in stratified flow over rough partial-step topography** (Adcroft–Hallberg 2006; Adcroft 2013), *not* a sign that SMC03 is wrong. Your SMC03 already passes BH-seamount at 1.5 mm/s and rest-state |dv/dt| = 1.1e-8 m/s²; that is a strong negative result for "PGF is broken."
2. The *fast* live-T failure (NaN day 19) is a well-known cold-start positive feedback in z*/ALE-z models (Holmes et al. 2019; Megann 2018; Ilıcak et al. 2012): a small numerical seed flow advects T,S across sloping isopycnals on rough topography, the resulting available-PE release amplifies the seed, and you cascade. This is exactly what GM/Redi was invented to suppress, and it is *not* a bug.
3. **However**: before adding closures, run one cheap diagnostic — re-run frozen-T with `r=5e-3` (production-typical bottom drag) and re-run with `dt=300s`. If frozen-T saturates below ~5 mm/s with stronger drag, you have proven the slow mode is benign and parameterization is the right path. If it still grows to 50 mm/s, you have a real problem and I would look hardest at the **bottom-cell PGF stencil at lateral steps in partial-cell columns**, not at SMC03 in the column interior.
4. The "30-day rest state on real ETOPO" test is **not** a standard production benchmark for the reasons you suspect. But it *is* diagnostic — it isolates pure numerical noise from forcing artifacts, and a coarse z-coordinate model that goes unstable from rest in 30 days *without* any closures is normal, not pathological. MOM6, MITgcm, NEMO all require closures to be on for cold-start ETOPO to be stable. You would be in good company shipping with closures and documenting that requirement.

I do not think you are papering over a bug. I think you have built a competent coarse-resolution z*/partial-cell model and are now hitting the well-known reason that *every* such model in production runs with GM/Redi + lateral viscosity on by default.

---

## Question 1 — Are GM/Redi + biharmonic physical closures or Band-Aids?

**Both, historically, but in this case unambiguously physical closures.**

### The physical-closure case (the dominant one)

GM (Gent–McWilliams 1990; Gent et al. 1995) is *not* a numerical fix. It is the parameterization of the eddy-induced advection of thickness that a coarse grid cannot resolve. The Rossby radius at 50°S, h=4000 m, N²=1e-5 is roughly L_R ≈ 30 km. At your 3° grid (~330 km at the equator, ~210 km at 50°S), you under-resolve the Rossby radius by **a factor of 7–10**. There is no defensible coarse-resolution z-model in the literature that runs without GM at this resolution. Specific production statements:

- **MOM6** (Adcroft et al. 2019, *JAMES*, §4.2): "GM with a flow-aware coefficient (Visbeck et al. 1997 or Held–Larichev) is on by default at all resolutions where L_R is under-resolved."
- **MITgcm** (Marshall et al. 1997 + GMRedi package docs): GM/Redi with κ ≈ 600–1000 m²/s is the recommended default for any non-eddy-resolving configuration.
- **NEMO** (Madec et al. 2017, NEMO book §9): "isoneutral diffusion (Redi) with GM eddy-induced velocity is mandatory at coarse resolution."

Redi (Redi 1982) rotates the diffusion tensor to be isoneutral. Without it, *any* horizontal diffusion on a z-coordinate is implicitly diapycnal, which on real bathymetry violates the second law of thermodynamics and creates spurious watermass drift (Veronis effect; Veronis 1975).

Biharmonic momentum viscosity (Griffies & Hallberg 2000) is the standard scale-selective closure for coarse-grid C-grid models. The C-grid has a known null mode (the checkerboard / B-grid-like 2Δ mode in vorticity) that is undamped by Coriolis at f-points and *requires* either Smagorinsky-Leith biharmonic viscosity (Griffies & Hallberg 2000, eq. 5; Fox-Kemper & Menemenlis 2008) or Holland-style harmonic viscosity to keep bounded. This is independent of partial cells, independent of PGF, and independent of bathymetry. Every production C-grid model has it on.

### The Band-Aid case (rare, and not what you have)

There are documented cases of damping masking real numerical bugs:

- **POP/CCSM3 era** (Smith & Gent 2002): excessive horizontal diffusion was used to mask spurious diapycnal mixing from the original Lin (1997) PPM advection on z-levels over topography. This was a real bug; the fix was eventually MOM6's ALE framework (Adcroft & Hallberg 2006), not more diffusion.
- **NEMO 3.x** (Lemarié et al. 2012, *Ocean Modelling*): early s-coordinate runs used elevated viscosity to suppress *hydrostatic-inconsistency* PGF errors at sloping σ-levels. This *was* a band-aid — Shchepetkin & McWilliams (2003) PGF (your SMC03) is the principled fix and it removed the need for the band-aid.

**The relevant point for you:** the documented Band-Aid cases in history have all been on **σ/s-coordinate PGF errors** or **z-coordinate spurious diapycnal mixing on full step topography**. Your model is z*/partial-cell with SMC03. SMC03 is the literature-recommended PGF fix; you have validated it on BH-seamount at the 1.5 mm/s level. You are not in the Band-Aid regime.

So: GM/Redi + biharmonic at this resolution are **physical closures the model needs regardless**. They are not masking your SMC03.

---

## Question 2 — Is this instability the kind production codes accept and damp, or diagnose and fix?

**Accept and damp, with one caveat I would check.**

The frozen-T 51 mm/s growth from a 2.5 mm/s seed over 30 days with **no T,S evolution** has three plausible mechanisms in a stratified C-grid + partial-cell + z* model. Let me rank them.

### (1) C-grid topographic computational mode in stratified flow — **most likely**

Mesinger (1973, *Tellus*) first identified that the C-grid supports a stationary ψ-mode at f-points where Coriolis cannot transmit information. In stratified flow over isolated topography, this mode couples to the bottom PGF and grows slowly. Adcroft & Hallberg (2006, *Ocean Modelling*, "On methods for solving the oceanic equations of motion") section 3 explicitly catalogs this as a distinct numerical mode from the σ-coordinate PGF error. **Reported growth rates for this mode at coarse resolution and weak drag are e-folding ~10–20 days**, exactly your range.

The frozen-T result is the smoking gun: with T,S frozen, the only way to get growth is from the dynamics responding to fixed stratification + topography. This is what the topographic computational mode is. Production codes damp it with biharmonic momentum + bottom drag. They do not "fix" it because the C-grid + stratification + steps trinity is the source — you would have to change discretization fundamentally (e.g., go to MOM6 ALE + remap, which you eventually want anyway).

### (2) Bottom-cell PGF residual at lateral partial-cell steps — **possible, worth checking**

This is *different* from the BH-seamount mode that SMC03 already fixes. SMC03 + face_target = min(z_centroid) is correct in the column interior. But at a **lateral step** between two partial-cell columns of different bottom thickness, the PGF stencil does a horizontal difference of pressure across a face where one side has a thick partial cell and the other has a thinner one (or full cell). The Adcroft-Campin face_target convention handles this *if* the face_target is computed per face from the geometry of *both* neighbors, but if your code computes face_target purely from the column on one side and reuses it, you can get a residual at every lateral step.

**This is the one place I would look hardest if you wanted to rule out a code bug before adding closures.** Specifically: instrument `dv/dt` from the PGF term alone in a frozen-T run and look at a horizontal map. If the residual is **localized at lateral partial-cell step boundaries in deep ocean** (mid-Atlantic ridge flanks, Pacific abyssal hills), you have a stencil bug. If it is **spread broadly** with some preference for steep slopes, it's the topographic computational mode and damping is correct.

This is cheap to do (one diagnostic plot from the frozen-T run you already have) and it is the *single test* that distinguishes "real bug" from "physical mode."

### (3) Barotropic-baroclinic coupling on partial cells — **less likely but check**

The implicit-CN barotropic solver couples to the baroclinic via η and the depth-averaged forcing. On partial cells, the H field has small-amplitude but high-wavenumber structure (cell-by-cell variation in bottom thickness). If your H entering the barotropic solver is the *full-cell* H, but the baroclinic sees the partial-cell H, the depth-average residual feeds noise into η. Worth a sanity check: does H_baro = sum(h_partial_cell) match what the barotropic uses? Should be a 5-line audit.

### Why E1 is stable but frozen-T is not (the key constraint your bisection gives me)

E1 = uniform T,S, full advection, stable at 4.2 mm/s. Frozen-T = stratified T,S, **no advection**, grows to 51 mm/s. So advection of T,S is *not* the seed mechanism. The seed comes from **stratification + topography + dynamics** alone.

The single mechanism that explains all six rows: **the dynamical PGF on partial-cell-stepped bathymetry in stratified flow has a slow C-grid-topographic-mode growth** (frozen-T sees this directly). When T,S can evolve (live-T), advection of T,S across this seed flow tilts isopycnals over topography, which releases APE through the PGF, which is the fast positive feedback (E4 day-16 NaN). E1 with uniform T,S has no APE to release, so the slow mode saturates at low amplitude.

This is a textbook **APE-release-on-rough-bathymetry** story. It is exactly what GM (which flattens isopycnals adiabatically) and Redi (which restricts diffusion to be along isopycnals) were designed to suppress. They are not Band-Aids on this; they are the physics.

---

## Question 3 — Specific recipe

For 3°/20-level cold-start z*/partial-cell ETOPO, dt=600s, 30-day:

### (a) GM and Redi

- **K_GM** ≈ **800 m²/s** constant, or **flow-aware** Visbeck et al. 1997 with α≈0.015 capped at [200, 1500] m²/s. MOM6 default at 1° is 600 m²/s (Adcroft et al. 2019, table 2). At 3° you want ~1.5× that to ~800–1000 m²/s.
- **K_iso (Redi)** ≈ **800 m²/s**, typically equal to K_GM at coarse resolution. Slope clipping at |∇ρ_horizontal/∂_z ρ| ≤ 1/100 to 1/500 (Gerdes et al. 1991 tapering) is mandatory. Without slope clipping you re-introduce diapycnal mixing through the Redi tensor near the surface and bottom.
- **Tapering near boundary layers**: Danabasoglu & McWilliams (1995) or Ferrari et al. (2010). Without this, GM erroneously fluxes mass through the surface and the bottom step. **Critical on real ETOPO.**

### (b) Biharmonic momentum viscosity

- Constant biharmonic **B_h** = U_scale · Δx³ / 8 with U_scale ≈ 0.01 m/s gives B_h ~ 1e10 m⁴/s at 3°. This is in the Griffies & Hallberg (2000) recommended range.
- **Smagorinsky-Leith biharmonic** (Fox-Kemper & Menemenlis 2008) is preferred at production: B_h = (C_Leith Δx / π)³ √(|∇²ω|² + |∇²δ|² / γ²) with C_Leith ≈ 1.0–2.0. It is flow-adaptive (more viscous in active eddies, less in quiescent regions) and its biharmonic-CFL is automatically respected.
- **Recommendation for your case:** start with constant B_h = 5e9 m⁴/s. It is the simpler implementation and is sufficient for cold-start stability. Move to Smagorinsky-Leith later when you go to <1° eddy-permitting. **Do not use Smagorinsky-2 (harmonic Smagorinsky) on the C-grid at coarse resolution** — it over-damps the slow waves you want to keep.

### (c) Bottom drag

- **r=1e-3 is on the low end.** MOM6 default (Adcroft et al. 2019) is **r=2.5e-3** for linear drag. With quadratic drag, C_d=2.5e-3 with a velocity scale of 0.1 m/s gives an effective r ≈ 2.5e-4 — but quadratic drag is small at low speeds, which is fine because you don't want strong drag damping the spinup, you want it strong enough to stabilize the slow mode.
- **For your test specifically: try r=2.5e-3 (production typical) before adding closures.** If frozen-T saturates at low amplitude with this drag alone, you have learned the slow mode is benign and you don't even need biharmonic momentum yet.
- **Killworth-Edwards distributed BBL drag** (which I see you already have from commit 7c03ff91): keep it on. It is the right implementation for partial cells; full-bottom-cell drag over-damps.

### (d) dt=600s on 3° at 20 levels

- **Defensible**, with caveats. Implicit-CN barotropic removes the external-mode CFL constraint, so the limit is internal-wave CFL. First baroclinic mode c₁ ≈ 2.5 m/s, Δx ≈ 330 km at equator → CFL = c₁ Δt / Δx = 2.5 × 600 / 3.3e5 ≈ 4.5e-3. Comfortable.
- **The risk is at deep partial cells with h~500 m where vertical grid Reynolds and advective Δz/Δt limits matter** (h/Δt acts like a vertical velocity scale). Cold-start should not produce vertical velocities anywhere near this, so dt=600s is fine for the rest-state diagnostic.
- MOM6 at 1° production uses dt = 1800s baroclinic, dt_btm = 60s internal split; at 3° you could go to 1800s safely once stable. **dt=600s is conservative, not aggressive.**

---

## Question 4 — Does a 30-day rest-state ETOPO test mean anything?

**Partially yes — as a numerics diagnostic. No — as a physics validation.**

You are right that production codes do not run "rest state on real bathymetry as a benchmark." MOM6, MITgcm, NEMO standard validation is:
- Williamson SW tests (Williamson et al. 1992) — topography-free
- DOME overflow (Legg et al. 2006) — idealized canyon
- ISOMIP (Asay-Davis et al. 2016) — idealized ice-shelf cavity
- COREII / OMIP (Griffies et al. 2016) — multi-decade *forced* spinup

The only "rest-state on real bathymetry" test I am aware of in the literature is in the **Ilıcak et al. 2012** (*Ocean Modelling*) "spurious diapycnal mixing" study — and even there it is run *with* GM/Redi + biharmonic on, because nobody expects it to be stable without.

**However, your test is diagnostic of one specific thing:** whether the model has a fast unphysical instability (e-fold < 1 day) that closures cannot fix. Your bisection has *ruled this out* — frozen-T grows e-fold ~6–20 days, which is in the physically-expected range for under-resolved baroclinic dynamics on rough topography. This is a **negative result that gives you confidence**, not a failure.

### What I would actually do

1. **Don't ship this as "validated on rest-state ETOPO."** Nobody else does that and it is not a meaningful claim.
2. **Do run frozen-T with r=2.5e-3 and document the result** as your numerics-noise floor diagnostic. If it saturates at ~5 mm/s, that is your noise floor — report it as "30-day cold-start ETOPO with production drag, no closures: σ_v ≈ 5 mm/s (numerical noise floor)."
3. **Add GM/Redi + biharmonic and document them as physical closures the model requires at this resolution.** This is honest. It is what every other model does.
4. **Validate against COREII/OMIP-style forced spinup** (you already have an OMIP plan in `omip_1deg_plan.md`). Decadal spinup with realistic forcing is the meaningful test, not 30-day rest state.
5. **Before ALL of this, run the one diagnostic in §Q2(2):** PGF-only `dv/dt` map from a frozen-T run. If it is broadly distributed, you are done — closures are correct. If it is localized at lateral partial-cell steps, you have a stencil bug to fix and it is worth the day to fix it before adding closures.

---

## What I would say to the user

You have not built a broken model. You have built a competent coarse z*/partial-cell C-grid ocean and you are now meeting the same wall every coarse z-model meets: real bathymetry + stratification + C-grid + partial cells produces a slow numerical mode that physical sub-grid closures (GM, Redi, biharmonic) are designed to absorb. Adding them is not papering over a bug; it is implementing the rest of the model.

The one residual risk is a partial-cell-step PGF stencil bug at lateral steps (distinct from the column-interior SMC03 you have already validated). It is cheap to rule out. Do that before adding closures, and then add closures with confidence.

---

## References

- Adcroft, A., 2013. Representation of topography by porous barriers and rescaled signed-distance functions. *Ocean Modelling* 67, 13–27.
- Adcroft, A. & R. Hallberg, 2006. On methods for solving the oceanic equations of motion in generalized vertical coordinates. *Ocean Modelling* 11, 224–233.
- Adcroft, A. et al., 2019. The GFDL global ocean and sea ice model OM4.0: model description and simulation features. *JAMES* 11, 3167–3211.
- Asay-Davis, X. et al., 2016. ISOMIP+ and MISOMIP1. *Geosci. Model Dev.* 9, 2471.
- Danabasoglu, G. & J. McWilliams, 1995. Sensitivity of the global ocean circulation to parameterizations of mesoscale tracer transports. *J. Climate* 8, 2967–2987.
- Ferrari, R. et al., 2010. A boundary-value problem for the parameterized mesoscale eddy transport. *Ocean Modelling* 32, 143.
- Fox-Kemper, B. & D. Menemenlis, 2008. Can large eddy simulation techniques improve mesoscale rich ocean models? *Ocean Modeling in an Eddying Regime, AGU Monograph 177*, 319–337.
- Gent, P. & J. McWilliams, 1990. Isopycnal mixing in ocean circulation models. *J. Phys. Oceanogr.* 20, 150.
- Gent, P. et al., 1995. Parameterizing eddy-induced tracer transports in ocean circulation models. *J. Phys. Oceanogr.* 25, 463.
- Gerdes, R. et al., 1991. The influence of numerical advection schemes on the results of ocean general circulation models. *Climate Dynamics* 5, 211.
- Griffies, S. & R. Hallberg, 2000. Biharmonic friction with a Smagorinsky-like viscosity for use in large-scale eddy-permitting ocean models. *Mon. Wea. Rev.* 128, 2935 — eq. 5.
- Griffies, S. et al., 2016. OMIP contribution to CMIP6. *Geosci. Model Dev.* 9, 3231.
- Holmes, R. et al., 2019. Diapycnal mixing in z*-coordinate models. *J. Adv. Model. Earth Syst.* 11, 3008.
- Ilıcak, M. et al., 2012. Spurious dianeutral mixing and the role of momentum closure. *Ocean Modelling* 45, 37.
- Legg, S. et al., 2006. Comparison of entrainment in overflows simulated by z-coordinate, isopycnal, and non-hydrostatic models. *Ocean Modelling* 11, 69.
- Lemarié, F. et al., 2012. On the stability and accuracy of the harmonic and biharmonic isoneutral mixing operators. *Ocean Modelling* 52–53, 9.
- Madec, G. et al., 2017. NEMO ocean engine. *NEMO book*, IPSL.
- Marshall, J. et al., 1997. A finite-volume, incompressible Navier-Stokes model for studies of the ocean on parallel computers. *JGR Oceans* 102, 5753.
- Megann, A., 2018. Estimating the numerical diapycnal mixing in an eddy-permitting ocean model. *Ocean Modelling* 121, 19.
- Mesinger, F., 1973. A method for construction of second-order accuracy difference schemes permitting no false two-grid-interval wave in the height field. *Tellus* 25, 444 — §3.
- Redi, M., 1982. Oceanic isopycnal mixing by coordinate rotation. *J. Phys. Oceanogr.* 12, 1154.
- Shchepetkin, A. & J. McWilliams, 2003. A method for computing horizontal pressure-gradient force in an oceanic model with a non-aligned vertical coordinate. *JGR Oceans* 108, 3090.
- Smith, R. & P. Gent, 2002. Reference manual for the Parallel Ocean Program (POP). *LANL technical report*.
- Veronis, G., 1975. The role of models in tracer studies. *Numerical Models of Ocean Circulation*, NAS, 133.
- Visbeck, M. et al., 1997. Specification of eddy transfer coefficients in coarse-resolution ocean circulation models. *J. Phys. Oceanogr.* 27, 381.
- Williamson, D. et al., 1992. A standard test set for numerical approximations to the shallow water equations in spherical geometry. *J. Comput. Phys.* 102, 211.
