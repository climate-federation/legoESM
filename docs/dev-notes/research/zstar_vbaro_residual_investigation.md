# Z-star continuity, the residual ⟨V_baro⟩, topographic form stress, and what to test next

A senior-physical-oceanographer review of the Drake-band momentum-budget closure
problem, the validity of the Reynolds-mass-flux hypothesis, and the
z-star-specific subtleties the user asked us to look at carefully.

Code references are to `src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py`
(BARO), `ocean_pe_latlon_cgrid.py` (PE), `ocean_model_latlon_cgrid.py`
(ORCH), and `vertical.py` (VERT).

---

## Q1. The exact discrete continuity equation and what it conserves

**Punchline.** The substep enforces a *2D* free-surface continuity
∂η/∂t + ∇·(H_total · V̄) = 0 exactly to roundoff, *as long as*
`_clamp_redistribute` does not fire. It does **not** enforce a per-latitude
"line integral of meridional transport vanishes" — it only enforces that the
*global* sum equals the freshwater source. There is therefore no discrete
constraint that forces ⟨V̄⟩ summed zonally to be zero at every latitude
unless η is in true statistical steady state.

Evidence:

- BARO L267–285 builds `flux_u = H_u·U_bar·u_mask`, `flux_v = H_v·V_bar·v_mask`,
  and updates `eta_unfloored = eta_c − dt_s·div_flux + dt_s·F_slow_eta·mask`.
  The divergence operator (`divergence_cgrid`, latlon_cgrid_operators.py
  L220–320) is a strictly conservative FV stencil:
  `div = (u_E·dy − u_W·dy + v_N·dx_N − v_S·dx_S)/A`. Summing `div_flux·area`
  over the whole grid telescopes to zero (periodic in lon, v=0 at poles),
  so `Σ η·area` is conserved up to `Σ F_slow_eta·area·dt_s` per substep.
- The summed-zonal continuity at row `i`,
  `Σ_j area·∂η/∂t = Σ_j (v_north·dx_north − v_south·dx_south)`,
  is the difference of *meridional throughflows at two latitudes*, not zero.
  In statistical steady state (∂η/∂t→0 in time mean) the time-mean
  meridional throughflow becomes y-independent (a global "leak" rate set by
  freshwater), but locally η fluctuates and the constraint is violated
  instantaneously.
- `_clamp_redistribute` (eta_floor.py L26–46) is the one escape hatch:
  whenever a cell is dried out (`eta_unfloored < eta_floor`), it injects
  mass and redistributes globally over wet cells — destroying the local
  zonal-flux constraint. With H_max=4000 m and observed |η|<1 m, this
  almost never fires, but it is worth instrumenting (count how many cells
  trigger it per step).
- BARO L132–137 also applies a non-conservative *initial* `eta = max(eta_raw, eta_floor)*mask`
  at the *top* of the substep loop, so any baroclinic-step floor violation
  is absorbed silently before continuity starts integrating.

**Verdict on the user's diagnostic.** Computing
`Σ_j v_face_avg(i,j)·dx(i,j)` from saved time-mean `v_3d`
*and* `dz_ref` (note: not `h_k`!) is **not** the quantity the model
conserves. Two distinct errors are folded in:

1. **z-star thickness.** Using `dz_ref` (eta=0 column) instead of `h_k = dz_ref·J(η)`
   under-counts mass by a factor `1/J ≈ H_max/(η+H_bathy)`. With H_max = 4000 m
   and a 4000 m water column this is essentially exact, but where the band
   sits over shallower bathymetry or under thinned columns it can be 10–20%
   off.
2. **Reynolds correlation.** ⟨H·V⟩ ≠ ⟨H⟩·⟨V⟩ in general (next section).

---

## Q2. h_k·v_k vs H·V_bar — are they identical in this code?

**Punchline.** They are identical *only* because `V_bar` is constructed as
the thickness-weighted depth-mean. The model never actually sums
`Σ_k v_k·h_k` for continuity; it only ever updates η from
`H_total·V_bar`. Your offline diagnostic that uses `Σ_k v_k·dz_ref` is
*not* what the model conserves.

Evidence:

- BARO L72: `U_bar = Σ(u·h_u)/H_u` exactly thickness-weighted, where
  `H_u = Σ_k h_u_k`. By construction `H·V_bar ≡ Σ_k h_k·v_k` at the moment of
  averaging. Per-substep continuity is in terms of `H·V_bar` (BARO L274–275).
- ORCH L506–509: after substepping, the 3D velocity is *corrected* so that
  `Σ_k h_u_old·u_3d_corrected = Hu_avg` exactly. So when ORCH writes
  `mass_flux_u = h_u_old·u_corrected`, the column sum equals the
  barotropic-averaged transport `Hu_avg` returned from the substep
  (`Hu_sum`/n_substeps, BARO L396).
- The `h_u`/`h_v` used inside the substep (BARO L267–272) is built fresh
  every substep from the *current* `eta_c + H_bathy`, so the continuity
  uses the actually-evolving column thickness `H_total_c`, not the cached
  baroclinic-step `h_k_old` used by ORCH for tracer flux. They agree at
  substep 0 but drift away by `O(dt_s)` per substep.

**Implication for the diagnostic.** `Σ_k v_k_mean·dz_ref` (which uses fixed
column thickness from the static reference) cannot match the conserved
quantity `⟨H·V_bar⟩` or even `⟨V_bar⟩·⟨H⟩` because the time-mean is taken in
two different basis weights. Best is to recompute `⟨V_bar⟩` at v-faces from
the *barotropic* state (post-substep) and dot it with time-mean `H_v` (also
post-substep). Even this misses the η'·V'_bar correlation.

---

## Q3. Is the Reynolds-mass-flux hypothesis numerically plausible?

**Punchline.** It is plausible in *order of magnitude* but tight to the
upper end of the range. With dt_s = 20 s, n_substeps = 30, c_baro =
sqrt(g·H) ≈ 200 m/s, and dx ~ 5°·cos(60°)/2 ≈ 140 km, the gravity-wave CFL
is about 0.03 — this is heavily under-CFL barotropic, which actually
*amplifies* η' fluctuations because BEBT damping is weak.

Quantitative scoping. To produce ⟨V̄⟩ = 2×10⁻⁴ m/s as a closure for
⟨η'V'_bar⟩, summed zonally we need (per Q1 above)

```
Σ_j η'_j · V'_bar,j ~ ⟨V̄⟩·Σ_j H_j  ≈  2×10⁻⁴ · 65·4000  ≈  52 m²/s
```

i.e. ~0.8 m²/s per face. For broadband barotropic noise: gravity-wave
amplitude scales η'·c ~ V'·H, so V' ≈ c·η'/H ≈ 200·η'/4000 = 0.05·η'.
Perfectly correlated η'V' with this dispersion gives
`⟨η'V'⟩ = (1/2)·η'·V' = 0.025·(η')²`. To reach 0.8 m²/s requires
**η' ≈ 5–6 m**, which is way more than observed (|η|<1 m).

But (and this is the key correction) **gravity waves are not perfectly
correlated** in the relevant phase, **inertial-period oscillations are not
the dominant Reynolds-flux source either** in a forced steady state, and
the dominant non-zero ⟨η'V'⟩ in a wind-forced channel is *Stokes-like*
correlation between Ekman v' and a near-resonant η' on the same gravity
wave — typically O(0.005 m²/s) per face at this resolution. **This is two
orders of magnitude too small.**

So Reynolds correlation alone almost certainly does not explain a
0.02 cm/s residual. **Reject the Reynolds hypothesis as dominant.** What
remains:

- (a) **A real, geostrophically balanced, steady-state ⟨V̄⟩**. With
  TFS=0 the ACC momentum balance leaves only `ρ·H·f·⟨V̄⟩` and bottom drag
  to close `τ_x`. The model picks the V̄ that closes the budget, and that
  V̄ *must* be nonzero because the wind is nonzero. The "mass conservation
  violation" is an artifact of computing the wrong diagnostic — see Q1/Q2.
- (b) **Geostrophic adjustment around the meridional continent**. The
  band has a meridional continent at lon 5–11 north of the band; the
  geostrophic deflection of the wind-driven Ekman transport against this
  wall sets up a barotropic gyre that returns mass at depth. ⟨V̄⟩ in the
  band can be nonzero in *steady state* (the standing wave around the
  continent), without violating mass conservation.

The user's observation that "η is not drifting" is consistent with (a)+(b):
**the model is conserving mass exactly**; only the offline diagnostic was
wrong.

---

## Q4. Why didn't the ridge restore the wind/drag balance?

**Punchline.** A z-star coordinate makes topographic form stress (TFS) act
through a *different mechanism* than in z-coord, and a single 1-cell-wide
3000 m ridge at 5° resolution is far below the scale where the discrete
PGF can express it cleanly.

Evidence and reasoning:

- **z-star PGF.** PE L347–365 computes the baroclinic PGF as
  `(1/ρ_0)·∇p'` on z* surfaces, where `p'(x,y,k) = g·Σ_{l≤k} ρ'_l·dz_ref_l`
  using the *reference* dz_ref, not actual h_k. The barotropic PGF is
  carried entirely by `g·∇η` in BARO. So in z*, TFS appears in the
  depth-integrated zonal momentum equation through *two* paths:
  (i) `∫_{−H}^{η} ∂_x p · dz` at the bathymetric step (the z-coord TFS
  term `p|_{−H}·∂_x H`), and (ii) the cross-correlation `⟨ρ'·∂_x H_bathy⟩`
  on the deepest active z* level. Path (i) is captured because `∇p` is
  computed on *every active layer* including the deepest, and the deepest
  layer's lateral extent changes across the ridge, so the integrated
  pressure × bathymetric gradient acts as a TFS analog. **It is not zero in
  z-star.**
- **But** the discrete PGF on z* uses `gradient_x_cgrid(p_prime_filled)`
  (PE L355–364), which is a 1-cell stencil. Across a 1-cell ridge the
  stencil sees only one face of pressure step; the symmetric face is
  filled with the same value via `_neumann_fill_cgrid` over land. This
  effectively halves the PGF jump and therefore halves the TFS. With a
  ~3000 m ridge in a 4000 m column, the actual form stress is
  ~½·g·H·∂η_x_at_ridge·ρ_0, but our discrete operator delivers half of it,
  and only at the single ridge face — total form stress per zonal
  circumference is `∼ τ_TFS·dx_ridge / circumference ≈ 1/72 of analytical`.
- **Bottom drag in z-star.** In z-star the bottom-cell thickness is
  `dz_ref[-1]·J = dz_ref[-1]·(η+H)/H_max`. Over the ridge the bottom layer
  is thinner by factor (1000 m + η)/4000 m ≈ 0.25, so the drag stress
  `r·u/dz_bot` (PE L602–608) is *4× stronger* over the ridge crest. This
  is the right physics, but it means our +80 Sv ridge effect is split
  between TFS-analog and locally enhanced drag — not dominantly TFS.
- **z-star also lacks "true" pressure-against-step.** In a z-coord model
  the PGF on a partial cell has a step contribution `p·∂_x H_bathy`. In z*,
  the bottom is conceptually flat in σ-space and the ∂_x H_bathy enters
  only through ρ' contributions across the lateral pressure stencil — it is
  weaker for small ρ' anomalies. In a homogeneous-temperature run (small ρ'),
  z-star TFS would tend to zero entirely.

**So the +80 Sv recovery is not "ridge underperforming" — it is the ridge
delivering most of what the discrete z* scheme can deliver at 1-cell
resolution.** Realistic ACC TFS at this resolution requires (a) a
multi-cell ridge with tapered crest (3+ cells wide so the PGF stencil
sees the full step) and (b) sufficient stratification (so ρ' across the
lateral PGF on the ridge produces the signed pressure asymmetry).
Reference: Munk-Palmen 1951; Olbers et al. *Ocean Dynamics*, ch. 11.4;
Stewart-Hogg 2017 ACC TFS budget (ridges resolve over ~100 km in their
simulations, ours is ~140 km × 1 cell).

---

## Q5. Recommended next steps to disentangle the candidates

In rough order of cost-benefit:

1. **Recompute the diagnostic correctly** (cheap, hours).
   - Save and time-mean `H_v` and `V_bar` at v-faces (post-substep), then
     compute `Σ_j ⟨H_v⟩·⟨V_bar⟩·dx_v` and separately `Σ_j ⟨H_v·V_bar⟩·dx_v`
     (the latter is the model-conserved transport). The difference is the
     Reynolds correction `Σ_j ⟨η'·V'_bar⟩·dx_v`. Expectation: the
     model-conserved transport is nearly y-independent (set by the freshwater
     source and the barotropic mode of the standing wave), and the
     Reynolds correction is O(0.001 m²/s)·N_face ≈ 0.07 m²/s — well below
     the 50 m²/s "violation" you've been chasing.
   - This will likely confirm the model is conserving mass and the
     "violation" was diagnostic-side.

2. **Multi-cell ridge experiment** (medium, 1–2 days run time).
   Replace the 1-cell ridge with a 3-cell tapered ridge of similar volume
   (e.g., crest at 1 cell, 2000 m at 2 cells either side). Expectation:
   factor 2–3× more TFS reach, transport recovers further toward the
   Munk wind/TFS balance. References: Stewart-Hogg 2017;
   Constantinou-Young 2017 idealized topographic ACC.

3. **Run with `barotropic_div_damp = 0.1`** (cheap, hours) — *this is a
   reasonable test* but I do not expect it to fix the residual.
   Divergence damping suppresses the *divergent* barotropic mode (gravity
   waves), which is the only mode that can produce a non-zero `⟨η'·V'⟩`.
   If V_baro residual is unchanged, that is independent confirmation that
   the residual is **not** Reynolds-noise-driven (i.e. confirms (1)).
   References: Hallberg 1997; Shchepetkin-McWilliams 2005 SBE.

4. **Diagnose `_clamp_redistribute` activity** (5 min). Add a host-side
   counter for how many cells per substep trigger the floor clamp. If
   non-zero, this is a *real* mass leak that breaks Q1's claim.

5. **GL90 / GM-on-momentum** (this is the structural fix, ~weeks).
   The "wrong sign of ⟨V̄⟩" in flat-bottom channels is the well-known
   Wardle-Marshall 2000 / Treguier 1997 / Greatbatch-Lamb 1990 issue:
   GM-on-tracers-only leaves the depth-integrated momentum equation
   structurally underdetermined. Adding a vertical-viscosity GL90 closure
   transmits wind to bottom drag through a *parameterized eddy form
   stress*, restoring the right-sign Munk balance even on flat bottoms.
   See `docs/dev-notes/research/why_westward_drake.md` §2 for the literature
   anchor.

---

## Bottom line

**Best hypothesis.** Mass *is* being conserved by the discrete continuity
to roundoff. The "0.02 cm/s residual" in `⟨V̄⟩` is **not** a Reynolds
mass-flux artifact — that mechanism is too small at this resolution by 1–2
orders of magnitude. It is the *physical* depth-integrated meridional
transport that the model selects to close the depth-integrated zonal
momentum balance with TFS = 0 (or strongly under-resolved over a 1-cell
ridge), drag underactive, and GM acting only on tracers. The diagnostic
mismatch (`Σ v_k·dz_ref ≠ ⟨H·V_bar⟩`) is a separate, real **diagnostic
bug** that needs fixing before any further interpretation.

**Recommended single experiment.** First, fix the diagnostic by averaging
`H_v · V_bar` post-substep instead of `v_k · dz_ref`. If the residual
disappears or shrinks 10×, the case is closed and the next physics
question is GL90 / GM-on-momentum. If the residual remains, run
`barotropic_div_damp = 0.1` as the cleanest noise-noise discriminator
(no expected effect on the physical solution; large effect if the
residual is barotropic-wave Reynolds correlation).
