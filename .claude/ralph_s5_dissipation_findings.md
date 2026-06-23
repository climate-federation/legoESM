# Ralph §5 dissipation-closure loop — running findings

Gate: `compare_oceananigans_gridmode_decay.py` (finite-amplitude 2Δx-in-lon mode decay,
GMD_REFDIR=gridmode_decay_A0.1 SEED_AMP=0.1) → GATE = decay_ratio_lego/decay_ratio_oracle;
baseline **1.040** (target 1.0). Strong gate: `compare_oceananigans_baroclinic_adjustment.py 24/30`
(baseline NaN day 18; day-12 max|u| 3.39 vs oracle 1.47). Env hooks on both drivers:
VORTCOR_METRIC, VORTCOR_ZETA, WENO_DIV_SMOOTH.

## Iteration 1 — vorticity flux + time integration RULED OUT
Gated config options added (default OFF, bit-identical): `vortcor_enstrophy_metric`,
`vortcor_reconstruct_zeta` (state.py + `_bc_pv_flux`).

- **Sadourny Δx metric-weighted transport** (`vortcor_enstrophy_metric`, v̂=0.25·Σ(Δx_v·v)/Δx_u,
  matches Oceananigans ℑxᶠᵃᵃ(ℑyᵃᶜᵃ,Δx_q·v)·Δx⁻¹): gate 1.040→**1.039** (NULL). Negligible at the
  equatorial mode; a no-op on the uniform Cartesian baroclinic grid. RULED OUT.
- **Direct-ζ reconstruction** (`vortcor_reconstruct_zeta`, flux=v̂·ζᴿ vs PV q=ζ/h·(h·v)):
  bickley gate 1.040→**1.042** (NULL); baroclinic day-12 3.39→3.26 (marginal), still NaN day 18.
  η reaches ~3.8%/H at finite amplitude but the q-vs-ζ difference is 2nd-order. RULED OUT.
- **D-term self-upwinding** (weno_divergence_smoothness=standard = Oceananigans OnlySelfUpwinding):
  gate 1.040→**1.040** (NULL).
- **AB2 time integration**: legoESM ab2_epsilon=0.1 == Oceananigans QuasiAB2 χ=0.1, same form
  (3/2+χ)Gⁿ−(1/2+χ)Gⁿ⁻¹. MATCHES — ruled out.

**Conclusion so far:** the vorticity-flux reconstruction and time integration are faithful;
NONE of the faithful vorticity-flux structural variants move the gate. Consistent with the
per-node tendency match (0.997). The finite-amplitude under-dissipation is NOT in the
vorticity flux. (This adds to the prior effort's exhausted levers: WENO kernel, smoothness
family, FS solver, Coriolis scheme, timestep, spherical metrics, D-term order.)

## Iteration 2 — KE-gradient RULED OUT (structurally identical)
Read Oceananigans `bernoulli_head_U/V` (vector_invariant_self_upwinding.jl:61-90):
`(δKuᴿ + δKvˢ)·Δx⁻¹` where δKuᴿ = WENO-upwind of δx(u²) biased by sign(û) [SELF, dissipative]
and δKvˢ = SYMMETRIC (centered) interp of δx(v²) [CROSS, non-dissipative]. legoESM's KE-grad
(ocean_pe_latlon_cgrid.py:1462) is `dKE_u2_dx_at_uface [WENO-upwind ∂x⟨u²⟩] + 0.5·dvsq_dx
[centered ∂x⟨v²⟩]` — STRUCTURALLY IDENTICAL (self-upwind + cross-centered). FAITHFUL, ruled out.

## CONVERGED CONCLUSION (every INTERIOR momentum operator is faithful)
Across this loop (iter 1-2) + the entire prior effort, EVERY interior operator in the
vector-invariant WENO momentum tendency has been verified faithful or structurally identical to
Oceananigans: vorticity flux (per-node 0.997; metric/ζ-direct null), KE-gradient (identical),
D-term (decoupled self-upwinding null), WENO kernel (coefficient-faithful), smoothness family
(decoupled), Coriolis scheme, free-surface solver, time integration (AB2 ε=χ=0.1), timestep,
spherical metrics. NONE is the finite-amplitude under-dissipation lever. ⇒ there is NO single
faithful INTERIOR-operator fix. The bickley ~4% gate is benign (bickley stays finite); the
CATASTROPHE is the WALLED case (§5 / baroclinic_adjustment NaN). Both this loop and the prior
effort point to the ONE remaining structural difference: the WALL REPRESENTATION — legoESM's
masked-land rows + Neumann-fill vs Oceananigans' Bounded topology (even-reflection mirror halo →
ζ_wall=0 free-slip + near-wall WENO order reduction). That is a multi-session structural rework
(prior effort: order-reduction ALONE was negative; needs mirror-halo + order-reduction TOGETHER),
NOT a loop-iteration operator fix.

## ITERATION 2 — MAJOR REFRAME: the blow-up is INTERIOR + 3D-specific (NOT the wall mode)
DECISIVE diagnostic (`scripts/tmp` probe of baroclinic_adjustment max|u| location per day):
the blow-up is at lat-row 22-26 of 48 = the FRONT/JET CENTER (INTERIOR), NEVER the N/S walls
(rows 0-3/45-48), every day to NaN day 16. ⇒ the catastrophe is NOT the wall representation —
it is an INTERIOR finite-amplitude eddy runaway at the front. This OVERTURNS the prior
wall-mode hypothesis (and matches the prior wall-budget note that the 2dx structure was "in the
JET CENTER").

KEY: bickley (2D single-layer) is BENIGN (~4% gate, stays finite); baroclinic (3D, 8 levels)
BLOWS UP interior. The difference is the 3D-SPECIFIC terms — VERTICAL momentum advection and the
baroclinic PGF — which the ENTIRE bickley-focused effort (this loop + prior) could NEVER exercise.
THIS is the prime untested suspect.

VERTICAL MOMENTUM ADVECTION mismatch FOUND: oracle WENOVectorInvariant(vorticity_order=9) uses
vertical_order=5 (WENO5) on the FULL horizontal momentum w·∂u/∂z. legoESM's weno9 path caps
vertical at WENO5 (matches order) BUT advects u_prime = u − U_bar (the baroclinic PERTURBATION,
ocean_pe_latlon_cgrid.py:2120-2122), OMITTING the w·∂U_bar/∂z redistribution term (per the
upwind_perturbation doctrine, line 146-149). Oceananigans advects the FULL u. This omission is a
concrete, untested candidate for the interior finite-amplitude runaway. ALSO test the baroclinic
hydrostatic PGF at finite amplitude. NEXT ITERATION: build a vertical-momentum A/B (full-u vs
perturbation) and test on the baroclinic blow-up (the STRONG, now-correctly-targeted gate).

## Next candidates (untested)
1. KE-gradient (bernoulli head) upwinding: is legoESM's WENO KE-grad matching Oceananigans'
   OnlySelfUpwinding `kinetic_energy_gradient_scheme`? (test on the STRONG baroclinic gate —
   it IS sensitive: reconstruct_zeta moved day-12 3.39→3.26.)
2. The free-surface/barotropic coupling at finite amplitude (prior "complementary flaws").
3. Whether the residual is a COMBINATION of small effects (no single decisive lever) → would
   imply the structural-obstruction conclusion (no single faithful operator fix).

Use the STRONG gate (baroclinic blow-up day + day-12 max|u|) as the primary signal; the
bickley ~4% gate is weak/near-noise for single-lever discrimination.
