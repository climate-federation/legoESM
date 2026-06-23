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

## Next candidates (untested)
1. KE-gradient (bernoulli head) upwinding: is legoESM's WENO KE-grad matching Oceananigans'
   OnlySelfUpwinding `kinetic_energy_gradient_scheme`? (test on the STRONG baroclinic gate —
   it IS sensitive: reconstruct_zeta moved day-12 3.39→3.26.)
2. The free-surface/barotropic coupling at finite amplitude (prior "complementary flaws").
3. Whether the residual is a COMBINATION of small effects (no single decisive lever) → would
   imply the structural-obstruction conclusion (no single faithful operator fix).

Use the STRONG gate (baroclinic blow-up day + day-12 max|u|) as the primary signal; the
bickley ~4% gate is weak/near-noise for single-lever discrimination.
