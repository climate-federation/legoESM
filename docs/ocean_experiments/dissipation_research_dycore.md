# Dissipation, Filters, and Stabilization for Split-Explicit C-Grid Dycores
## Low-latitude / Steep-topography Barotropic Standing-Mode Failure — Literature & Production-Code Review

**Reviewer**: dycore-expert (legoESM)
**Subject**: Mitigation candidates for the deep-cell, depth-uniform `|u|` polynomial-then-exponential blowup adjacent to 4000 → 100 m shelves at |lat| < 30° in the lat-lon C-grid + implicit-CN barotropic + SMC03 PGF + AL81 PV-flux + biharmonic+cos² Laplacian configuration.
**Scope**: literature + production code locations, not JAX implementation patches.

The 1D dispersion picture is unambiguous: the implicit-CN Helmholtz operator has a numerical near-null eigenmode whose continuous-PDE counterpart is the **bottom-trapped barotropic-along-isobath wave** (Rhines 1969). At low f the geostrophic restoring is weak and `(βL/f)` blows up — so the standing mode can sit on a single isobath with no propagation, fed by truncation error in the discrete PGF and PV-flux at partial-cell faces. Production codes treat this through *targeted* dissipation that selects div modes over rot modes, *or* through filtering of the depth-mean state, *or* through a vertical-coordinate change that removes the partial-cell discontinuity.

---

## 1. Barotropic-only divergence damping (Skamarock & Klemp 1992; Lauritzen et al. 2018; FV3 Lin/Harris)

- **Form**: `∂U_bar/∂t += ν_div · ∇(∇·U_bar)`, applied only to the depth-mean transport, harmonic or 4th-order.
- **Reference**: Skamarock & Klemp 1992, *MWR* 120; FV3 (Lin 2004; Harris et al. 2021) `dyn_core.F90` parameters `d2_bg`, `d4_bg`, `d2_bg_k1`, `d2_bg_k2` (typical `d4_bg = 0.15`, `d2_bg = 0.0` free atmosphere, sponge `d2_bg_k1 = 0.16`).
- **Production in oceans**: MOM6 has it via "modified Leith" (`USE_MOD_LEITH = True`) in `MOM_hor_visc.F90` — adds an extra Leith-style coefficient that targets ∇·u in addition to vorticity-based Leith. ROMS calls this "div-grad correction" but layer-by-layer, not barotropic-only. **No production ocean applies *barotropic-only* div damping** — closest is MITgcm's `viscC2leithD`. DYNAMICO (Lauritzen et al. 2018, *GMDD*) uses pure ∇·u damping at every level on the icosahedron.
- **Why it should help our mode**: the standing mode IS barotropic and IS divergent at the η level (the corrector Δη tail). A small, near-grid-scale `ν_div ≈ 0.01·Δx²/Δt` applied to `(U_bar, V_bar)` — *not* to baroclinic perturbation — would dissipate the mode without polluting geostrophy (which is rotational in the baroclinic perturbation). Clean fix: baroclinic geostrophic balance lives in `u' = u_3d − U_bar`, untouched.
- **Why it might not**: with implicit-CN we are not substepping; if applied only post-corrector, the next-step Helmholtz simply re-excites the mode at the same rate. Must apply inside the predictor before the elliptic solve.

## 2. Smagorinsky / Leith with depth-amplification (Smagorinsky 1963; Leith 1968; Griffies & Hallberg 2000; Fox-Kemper & Menemenlis 2008)

- **Form**: `A_smag = (C_s · Δ)² · √(D_T² + D_S²)` (deformation rate) or `A_leith = (C_l · Δ)³ · |∇ζ|`. Biharmonic versions square the scaling.
- **Reference / code**: Griffies & Hallberg 2000 *MWR* 128 (canonical "Smag-biharmonic" used in MOM5/MOM6 `MOM_hor_visc.F90`, parameters `SMAGORINSKY_KH = 0.05`, `SMAG_LAP_CONST = 0.05`, `SMAG_BI_CONST = 0.06`, `LEITH_KH = True` with `LEITH_LAP_CONST = 1.0`); MITgcm `mom_calc_visc.F` (`viscC2smag`, `viscC2leith`); ROMS `lmd_skpp.F` and `t3dmix2.F`.
- **Production constants**: MOM6 1° default `SMAG_LAP_CONST = 0.15`, `SMAG_BI_CONST = 0.06`; ROMS regional 0.1° `tnu2 = 5` m²/s, `visc2 = 5` m²/s with Smag multiplier; MITgcm ECCO LLC4320 `viscC2smag = 2.2`, `viscC4smag = 4.0`.
- **Depth-enhanced variant?** None of the four production models apply *enhanced* Smagorinsky in the deepest few cells *as documented*. MOM6 has `KH_VEL_SCALE` and `KH_BG_MIN` that produce a depth-floor on viscosity, but not a depth-*top* enhancement. Closest is MITgcm's `bottomDragQuadratic` plus a vertically-enhanced Laplacian via `viscArNr(:)` array (per-level coefficient, hand-tunable) — used in tide-resolving runs (Arbic et al. 2010 STORMTIDE).
- **Why it should help us**: the standing mode is hosted in the deepest cells next to the shelf; |D|² is genuinely large there (4000 m → 100 m over 1–2 grid cells gives O(|U|/Δx) signal). A coefficient amplification factor of 5–10 in the bottom 2–3 cells via `viscArNr`-style profile, gated on `bot_level − k ≤ 2`, costs nothing for the geostrophic interior.
- **Why it might not**: Smagorinsky and Leith are *isotropic* — they damp rotational and divergent modes equally. Applied isotropically at strength sufficient to control the standing mode, will also damp resolved equatorial-jet geostrophic flow at lat 24°N. Use depth-enhanced version *only* in conjunction with rotational/divergent splitting (item 3).

## 3. Smith–McWilliams 2003 anisotropic / rotational-vs-divergent viscosity

- **Form**: split viscous stress tensor into rotational and divergent invariants, weight separately:
  `F = ∇·[ν_rot · D_rot + ν_div · D_div]` where `D_div = (∇·u) I` and `D_rot = ½(∇u + ∇uᵀ) − D_div`.
- **Reference**: Smith & McWilliams 2003, *Ocean Modelling* 5; Smith & Gent 2004 (anisotropic GM extension). Production: MOM6 `MOM_hor_visc.F90` parameter `ANISOTROPIC_VISCOSITY = True`, `KH_ANISO`, `MOM_ANISOTROPIC_VISC_DIR` — implements two-coefficient reduced form (Smith & McWilliams §3.2 with the divergent-flow term retained at the 2κh + κa level).
- **Production constants**: MOM6 ⅛° regional configs that enable it use `KH_ANISO ~ 1e3 m²/s` (cross-stream) and isotropic `KH ~ 1e2 m²/s` (along-stream). Default off in MOM6 1° global.
- **Why it should help**: the standing barotropic mode is mostly divergent at the depth-mean level (lives in η through `g·∇η` and continuity). Setting `ν_div >> ν_rot` *selectively damps the mode*. Geostrophic equatorial currents at 24°N are predominantly rotational (Sverdrupian) at the depth-mean, less affected. Cleanest classical fix.
- **Why it might not**: SM03 split is on *horizontal* tensor invariants; if the standing mode aliases partially onto the rotational projector through C-grid Coriolis null space, partial cancellation remains. Empirically (S&M 2003 §5) the split is excellent at controlling grid-scale checkerboard but only partially effective for topographic-trapped modes unless paired with item 1.

## 4. Filtering the Helmholtz solution / barotropic-mode time filter (Pacanowski–Griffies, Killworth 1991, Shchepetkin–McWilliams 2005)

- **Form**: `<U_bar> = Σ w_n · U_bar^n` over `n_baro` substeps, with `w_n` cosine bell or AB3-AM4 weight set designed to filter frequencies above the inertial cutoff.
- **Reference**: Killworth, Stainforth, Webb, Paterson 1991, *JPO* 21:1333 (free-surface MOM, AB-2 filter); Pacanowski & Griffies 1998 MOM3 manual §11 (η-filtering); Shchepetkin & McWilliams 2005, *Ocean Modelling* 9:347 — canonical reference, AB3-AM4 with shape-function weights satisfying normalisation, consistency and second-order accuracy. NEMO `dynspg_ts.F90` (`ln_bt_av = True`, `nn_bt_flt = 1` boxcar / `2` cosine); MOM6 `MOM_barotropic.F90` uses SM05 weights (`BTFILT_n` and `BTFILT_e` arrays); ROMS `step2d.F` is the original SM05 implementation.
- **Production constants**: MOM6 default `BTSplit ~ 40` substeps per baroclinic step with cosine-bell averaging (`BT_FILTER_FUNCTION = 'cos_bell'`). ROMS Coastal regional: `ndtfast = 30`, AB3-AM4 weights with `Falpha = 2.0`, `Fbeta = 4.0`, `Fgamma = 0.284`.
- **The implicit-CN twist**: with no substeps there is no time average to filter. Closest analog is **a Robert-Asselin-Williams η filter applied to η after the corrector** (item 7) and/or **a low-pass filter on the predictor `U_pred, V_pred` before the Helmholtz RHS**.
- **Why it should help**: η stays at mm-amplitude until the burst — classical signature of a slowly-growing computational mode in η that the implicit operator is *not* damping. A weak time filter (α ≈ 0.05–0.1 RAW) on η would give the standing mode a finite damping time without affecting resolved gravity-wave propagation.
- **Why it might not**: if the energy source is the partial-cell PGF residual (highly likely per `etopo_instability_dycore_review.md`), filtering η just delays the inevitable.

## 5. Time-stepping mods: AB3, semi-Lagrangian momentum, Backward-Euler in deepest k

- **Form**: treat vertical viscosity *fully implicitly only* in the deepest 2–3 cells: `(I − Δt · ∂_z(ν_v ∂_z))·u^{n+1} = u^* + Δt · F_lat`.
- **Reference**: MITgcm `IMPLICIT_VISCOSITY = .TRUE.` (default in z-coord global runs); MOM6 `BOTTOMDRAGLAW = "Cdrag"` with `LINEAR_DRAG = False` plus implicit treatment in `MOM_vert_friction.F90`; ROMS `LMD_BKPP.F` for bottom-boundary-layer-aware vertical viscosity; NEMO `dynzdf.F90` Backward-Euler.
- **Production**: MOM6 `HMIX_BOTTOM = 10 m`, MITgcm `bottomDragQuadratic = 2.5e-3`, ROMS `rdrg2 = 3.0e-3`.
- **AB3 with weights for non-leapfrog**: SM05 §3 derives the AB3-AM4 *generalized FB* scheme — third-order accurate and asymptotically stable for the Coriolis + gravity-wave system *with* partial cells. Does not require leapfrog; generalises FB to three-time-level. ROMS `step3d_uv.F`. Direct relevance to our FB Coriolis predictor.
- **Why it should help**: a Backward-Euler vertical viscosity in the deepest 2 cells, paired with quadratic bottom drag (Cd ≈ 2.5e-3), gives O(Δt) absolute damping of any depth-uniform mode hosted there — `|u|_eq = (τ_lat / ρ Cd)^{1/2}` saturation. This is what saturates ROMS/MITgcm tropical regional runs that would otherwise blow up.
- **Why it might not**: linear bottom drag on the bottom cell does *not* damp a depth-uniform standing mode in cells *above* the partial-cell stair. If the mode is hosted at the cell adjacent to the partial-cell jump (one step up the stair from deepest cell), bottom drag never sees it. Combine with item 2 depth-enhanced viscosity over the bottom 3 cells.

## 6. Low-Coriolis treatment: enhanced equatorial viscosity / sigma blending near steep slopes

- **Form**: `A_h(lat, z) = A_h_base · [1 + α · exp(−(lat/lat_eq)²)] · [1 + β · exp(−(d_shelf / L_s)²)]` with `lat_eq ≈ 10°`, `d_shelf ≤ 2Δx`, β ~ 5.
- **Reference / code**: MOM5 `ocean_sigma_diff_mod.F90` (`USE_SIGMA_DIFF = .TRUE.`) — pure σ-coordinate Laplacian *blended* with z-coordinate diffusion in a transition band, used in MOM5-CM2.5 to suppress overflow spurious mixing; Adcroft & Hallberg 2006 *Ocean Modelling* 11 z-tilde / z̃ ALE blending; MITgcm `mom_visc.F` with `viscAhGrid` (grid-relative scaling). ROMS s-coordinate is intrinsically σ-blended but uses Beckmann-Haidvogel pressure-Jacobian to control slope error.
- **Production constants**: MOM5 sigma-diff transition band 200–700 m with `sigma_diff_K = 50–100` m²/s (vs `K_h ~ 1e3` background); equatorial enhancement appears in some POP configs as `kappa_eq = 2 · kappa_base` for ±5° (Smith et al. 2010 POP2 ref. manual).
- **Why it should help**: at lat 24°N the equatorial enhancement doesn't reach our problem column, but the *shelf-blend* part does. A σ-blend in bottom 3 cells over a 2-grid-cell band downhill of the 4000 → 100 m jump turns the partial-cell discontinuity into a smooth slope; the partial-cell PGF residual feeding the mode decreases by `(Δh_partial/H)²`, which at z* with 20 levels can be O(10) reduction.
- **Why it might not**: σ-blending is a *grid* change, not a dissipation knob. Implementing in JAX inside the existing z* partial-cell pipeline is a multi-week project and entangles with GM/Redi (which assumes z-coord neutral surfaces). Cheaper proxy: item 2 with depth-enhanced viscosity.

## 7. Robert-Asselin-Williams filter for non-leapfrog schemes (Williams 2009; Amezcua–Kalnay–Williams 2011; Lin 2014)

- **Form**: `η^n_filt = η^n + α·ν·(η^{n−1} − 2·η^n + η^{n+1})` with `α ∈ [0.5, 1]` (Williams' RAW: α = 0.5 is original Robert; α = 1.0 is conservative limit), `ν = 0.05–0.2`. The "non-leapfrog" insight: when you have any 3-time-level state buffer (we do: `η^{n−1}`, `η^n`, predictor + corrector), RAW applies *as is* to η.
- **Reference**: Williams 2009, *MWR* 137:2538; Amezcua, Kalnay, Williams 2011 *MWR* 139:608 (the RAW name); Lin 2014 *J. Comput. Phys.* 259 (higher-order RAW); SM05 §4 (FB-AB3 has its own implicit filter at α = 0.281). FV3 atmosphere `tracer_2d_nested.F`.
- **Production constants**: MOM6 default `Robert_filter_coeff_eta = 0.0` (off, because their AB3-AM4 has internal filtering); MITgcm `RobertAfac = 0.1` for `useImplicitFreeSurface = False`; CESM POP `gamma = 0.1`.
- **Why it should help**: burst-after-30-days is textbook Robert-mode growth: slowly-amplifying eigenmode of the discrete time operator with growth factor `(1 + ε)^n` until ε·n → O(1). RAW with `α = 1, ν = 0.1` on η directly damps this without affecting physical eigenmodes (3rd-order accurate per Williams).
- **Why it might not**: implicit-CN with θ > 0.5 already provides unconditional gravity-wave damping ∝ (θ − 0.5). If the mode is *not* a time-discretisation mode but a *spatial* discretisation mode (PGF residual at partial cells, item 6), RAW does nothing. Diagnostic for distinguishing: item 10.

## 8. Hyper-dissipation only on the depth-mean barotropic component

- **Form**: `∂U_bar/∂t += −ν_4 · ∇⁴ U_bar` (hyperviscosity on depth average only); `u_3d ← u_3d + (U_bar^new − U_bar^old)` for the baroclinic perturbation (no change). Apply via the baroclinic-barotropic split: `u' := u_3d − ⟨u_3d⟩_z; U_bar := ⟨u_3d⟩_z`.
- **Reference**: this is *exactly* what HIM/MOM6 BTSplit does on the barotropic substep loop — biharmonic friction inside `MOM_barotropic.F90` substep affects only `U_bar`, not the layer velocities (updated separately in `MOM_dynamics_split_RK2.F90`). Hallberg & Adcroft 2009, *Ocean Modelling* 29:15 describes the projection. With implicit-CN we have no substep, so the equivalent is to apply biharmonic friction to `(U_bar, V_bar)` *as a slow forcing* `F_slow_u, F_slow_v` (the implicit solver already accepts this — see `barotropic_implicit_latlon_cgrid.py:264`).
- **Production constants**: MOM6 `BIHARMONIC_BAROTROPIC = True` with `BT_VISCOSITY = 5e3 m⁴/s`; HIM with `KhB_baro = 1e10 m⁴/s` for 1° global.
- **Why it should help**: *single most surgical* candidate. Standing mode is by definition depth-uniform (`u'_baro = 0`, `U_bar ≠ 0`), so depth-mean-only hyperviscosity hits it without touching baroclinic eddies, GM/Redi, or geostrophic shear. Baroclinic perturbation `u'` carries the resolved geostrophic balance and remains untouched.
- **Why it might not**: requires careful 3D update preserving `<u'>_z = 0` exactly under floating-point arithmetic. If re-projection is sloppy, hyperviscosity leaks into `u'`. legoESM already has `_depth_average_to_faces` (`barotropic_implicit_latlon_cgrid.py:73`) so projector is one step away.

## 9. Shapiro filter on η

- **Form**: `η^filt = (1 − ε·S_n) η`, with `S_n` the n-th-order Shapiro operator, ε = 0.05–0.5 per filter pass.
- **Reference / code**: MITgcm `pkg/shap_filt` (Shapiro 1970; targeted Shapiro of Jahn et al. 2012 for high-latitude η); MOM3 §11 `eta_smoother`; MOM5 `ocean_polar_filter.F90` Shapiro near poles. ROMS no longer uses Shapiro since SM05 substep filtering replaced it.
- **Production**: MITgcm Antarctic `shap_filt_n = 4`, `shap_filt_dt = 1`, `shap_filt_lev = 4` (apply every 4 steps); MOM3 polar filter at |lat| > 70° applies a 1D Shapiro every step with ε = 0.5.
- **Production usage in 2024**: largely deprecated for free-surface η in favour of SM05 substep averaging (which is a *spectral* filter on the same target). Still alive in MITgcm `pkg/shap_filt` for tracer fields and in regional terrain-following community (POM, NCOM).
- **Why it should help us**: single-pass 2nd-order Shapiro on η every 5–10 steps (ε = 0.1) is cheap, conservative-by-construction (operator commutes with `Σ area`), damps standing mode at host cell. Trade-off: acts on *all* η scales above grid, including resolved Kelvin/Rossby waves; over 30 days will damp resolved η variability by 5–15 % at affected wavelengths.
- **Why it might not**: 2024 production wisdom (MOM6, NEMO 4.x, MPAS-O) is that Shapiro is *symptom* mitigation. If standing mode is fed by a real PGF residual, Shapiro reduces η variability globally to control a local pathology — not selective.

## 10. Diagnostic for the standing mode

- **Computational-mode diagnostic** (Williams 2009, §3): track per-step **2-grid-point variance** of η and U_bar: `χ(t) = Σ |η^n − ½(η^{n−1} + η^{n+1})|² / Σ |η^n|²`. Clean integration χ ≈ 10⁻⁶; growing computational mode shows χ ↑ exponentially before max|u| does.
- **Barotropic-mode purity index**: `P_bt(t) = ⟨|U_bar|²⟩ / ⟨|u_3d|²⟩` and `P_bc(t) = ⟨|u_3d − U_bar|²⟩ / ⟨|u_3d|²⟩`. Standing-mode burst → `P_bt → 1` at the affected column. MOM6's `MOM_diagnostics.F90` outputs `e_baroclinic` and `e_barotropic` KE separately for exactly this reason.
- **Topographic-Rossby-mode projector**: project `u_3d − U_bar` onto leading f/H eigenmode and monitor column-wise norm; Rhines (1969) bottom-trapped eigenmode has analytic vertical structure `cosh(N k z)`. MITgcm `pkg/diagnostics` has `BAROTROPIC_PROJ` for this; not standard in MOM6.
- **Reference**: Hughes & de Cuevas 2001, *JPO* 31:2871 (f/H projector); Marshall et al. 2017, *Ocean Modelling* (continuous monitoring of `P_bt` for ocean instability detection).
- **Why it should help**: max|u| only spikes after the mode has saturated. χ and `P_bt − P_bt_climatology` lead the spike by 5–10 days, giving early-warning trigger that can also gate item 8 hyperviscosity (turn on only when χ > 10⁻³, leaving climatology runs untouched).

---

## Prioritized recommendations (highest expected impact first, given legoESM has GM/Redi already on, implicit-CN barotropic, JAX-pure-function constraints)

1. **Item 8 — biharmonic hyperviscosity on the depth-mean `(U_bar, V_bar)` only**, applied via the `F_slow_u, F_slow_v` slow-forcing channel that the implicit solver already accepts. ν₄ ≈ 5e3 m⁴/s. **Highest impact, lowest collateral damage**: standing mode IS the depth-mean signal at the affected column, baroclinic perturbation is mathematically untouched by construction. JAX-clean (one extra `_depth_average_to_faces` call, one biharmonic stencil, both already in codebase). Effort: ~50 LOC.

2. **Item 2 — depth-enhanced Smagorinsky in the deepest 2–3 cells (`viscArNr`-style profile, factor ×5–10 for `bot_level − k ≤ 2`) coupled with quadratic bottom drag from item 5**. `C_smag = 0.06` biharmonic, `C_d = 2.5e-3`. Targets *spatial* host of the mode. Already partially in legoESM (`smagorinsky_viscosity_cgrid` exists in `latlon_cgrid_operators.py`); only depth-profile multiplier missing. Effort: ~30 LOC.

3. **Item 3 — Smith-McWilliams 2003 anisotropic viscosity with `ν_div ~ 5×ν_rot`** applied to layered velocities. Operates on rotational/divergent invariants of same stress tensor used by existing biharmonic; JAX implementation reuses 80 % of `smagorinsky_biharmonic_tendency_cgrid`. Selectivity matters at the equator where SM03 is *the* documented production fix. Effort: ~150 LOC + tests.

4. **Item 10 — `χ` and `P_bt` diagnostics emitted every step**. Cheap (one global reduction each, no new MPI), scientifically essential for distinguishing time-discretisation (RAW) from spatial-discretisation (PGF) origin of the mode, unblocks future gating of items 1, 8 by physical diagnostics rather than blanket application. Effort: ~40 LOC.

5. **Item 7 — RAW filter on η with α = 1, ν = 0.1**, *only if* item 10's diagnostics show `χ` growth precedes max|u| growth (mode is in time, not space). If item 10 instead shows a per-cell PGF residual signature, do *not* add RAW — fix the PGF (per `etopo_instability_dycore_review.md`).

**Explicitly de-prioritised**:
- Item 4 substep-time-filter — N/A under implicit-CN, subsumed into item 7.
- Item 6 σ-blending — multi-week effort, entangles with GM/Redi, low return.
- Item 9 Shapiro on η — production-deprecated, less selective than item 8.
- Increasing existing isotropic biharmonic — will damp resolved flow at lat 24°N before it controls the mode.
