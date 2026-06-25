# Barotropic Wind-Driven Gyre Test Case for Ocean Dynamical Core Validation

## Purpose

This test case validates the ocean dynamical core's ability to produce a steady-state wind-driven circulation in an enclosed basin. It is the most fundamental forced ocean test and directly follows the rest-state test in the validation hierarchy. It tests: land/wall boundary conditions, wind stress forcing, Coriolis and beta-effect implementation, free surface solver, lateral friction operators, and steady-state convergence. An analytical solution exists in the linear (Munk) limit, enabling quantitative error assessment.

**References**: Stommel (1948), Munk (1950), Bryan (1963), MITgcm tutorial `verification/tutorial_barotropic_gyre/`, Pedlosky (1987) §5.11, Vallis (2017) Ch. 19.

---

## 1. Governing Equations

The barotropic shallow water equations on a beta-plane:

```
∂u/∂t - fv = -g ∂η/∂x + τ_x/(ρ₀H) + A_h ∇²u
∂v/∂t + fu = -g ∂η/∂y + A_h ∇²v
∂η/∂t + H(∂u/∂x + ∂v/∂y) = 0
```

where f = f₀ + βy is the Coriolis parameter on a beta-plane.

In steady state, the barotropic vorticity equation reduces to:

```
βv = curl(τ)/(ρ₀H) + A_h ∇⁴ψ       (Munk balance)
βv = curl(τ)/(ρ₀H) - r ζ            (Stommel balance)
```

where ψ is the barotropic streamfunction and ζ = ∂v/∂x - ∂u/∂y.

---

## 2. Domain and Grid

### 2.1 Physical Domain

| Parameter | Value | Notes |
|-----------|-------|-------|
| Basin extent (x) | L_x = 1200 km | Zonal |
| Basin extent (y) | L_y = 1200 km | Meridional |
| Depth | H = 5000 m | Flat bottom, uniform |
| Reference latitude | φ₀ = 30°N | Mid-latitude beta-plane center |
| f₀ | 2Ω sin(30°) = 7.29 × 10⁻⁵ s⁻¹ | Coriolis parameter at center |
| β | 2Ω cos(30°)/a ≈ 1.98 × 10⁻¹¹ m⁻¹s⁻¹ | Meridional gradient of f |

### 2.2 Horizontal Grid

**On lat-lon or Cartesian grids (MITgcm-style)**:
- 60 × 60 grid cells → Δx = Δy = 20 km
- Higher resolution variants: 120×120 (10 km), 240×240 (5 km) for convergence testing

**On cubed-sphere or icosahedral grids (legoESM)**:
The rectangular basin must be embedded within the global grid using a land mask. This is the critical test — the rest-state images show that land masking is currently broken on several grids. Options:

- **Beta-plane approximation on global grid**: Define f = f₀ + β(y - y₀) locally within the basin region, mask everything outside as land. This is the cleanest comparison to the analytical solution.
- **Full sphere with basin**: Place a rectangular basin at a specific location on the sphere (e.g., 0-12°E, 24-36°N). Use the full spherical Coriolis parameter. No analytical solution, but converges to a known numerical reference. Tests the real-world configuration more faithfully.

**Recommendation**: Start with the beta-plane approach embedded in the global grid. This isolates the boundary condition issues from coordinate geometry issues.

### 2.3 Vertical Grid (Z-star Specific)

This is where the MITgcm setup must be adapted for z-star. Key considerations:

**Number of levels**: MITgcm uses a single level for their barotropic tutorial (effectively a 2D problem). For z-star, we need multiple levels to verify that the barotropic mode is correctly represented across the water column.

**Recommended vertical grid**: Use N_k = 10 uniformly spaced levels, each Δz = 500 m.

**Stratification**: The fluid is **homogeneous** — uniform temperature T₀ = 20°C, uniform salinity S₀ = 35 PSU everywhere, at all levels, for all time. No buoyancy forcing, no surface heat/freshwater fluxes.

**Why this matters for z-star**:

1. **Every level must produce the same horizontal velocity.** In a barotropic flow, u(x,y,z) = u(x,y) independent of depth. If different z-star levels show different velocities, the vertical coupling (pressure gradient communication between layers) is broken.

2. **The baroclinic PGF must be exactly zero.** With uniform density, the only horizontal pressure gradient is from the free surface tilt: PGF_x = -g ∂η/∂x at every level. The density contribution -(1/ρ₀)∫(∂ρ/∂x)dz must vanish because ∂ρ/∂x = 0 everywhere. Any nonzero baroclinic PGF is a bug.

3. **Z-star stretching with the free surface.** In z-star, each layer thickness is h_k = Δz_k(1 + η/H). For the gyre, η ~ O(1 m) and H = 5000 m, so the stretching is O(2 × 10⁻⁴) — negligible but nonzero. The model must handle this consistently.

4. **Vertical velocity should be essentially zero.** In a truly barotropic flow on a flat bottom, w = 0 everywhere (the flow is depth-independent and non-divergent at each level). Any significant vertical velocity indicates spurious baroclinic modes being excited.

5. **ALE remapping should be a no-op.** With uniform density and barotropic flow, there is no reason for the ALE algorithm to move layer interfaces. If remapping is active and producing non-trivial transport, it may introduce spurious mixing (irrelevant here since the fluid is homogeneous, but worth monitoring).

**Diagnostic to verify barotropicity**: Compute the standard deviation of u across vertical levels at each (x,y) point. Should be zero to machine precision:

```
σ_u(x,y) = std_z[u(x,y,z_k)] → should be O(10⁻¹⁵) m/s
```

If σ_u is O(10⁻⁶) or larger, the vertical structure is contaminated.

---

## 3. Forcing

### 3.1 Wind Stress

**Single gyre** (MITgcm default):
```
τ_x(y) = -τ₀ cos(π y / L_y)
τ_y = 0
```
with τ₀ = 0.1 N/m². This produces a single anticyclonic gyre.

**Double gyre** (more common in MOM6 community):
```
τ_x(y) = -τ₀ cos(2π y / L_y)
τ_y = 0
```
This produces a subtropical (anticyclonic) and subpolar (cyclonic) gyre separated by an eastward jet at mid-basin. More interesting dynamically but harder to compare to analytical solutions.

**Recommendation**: Start with the single gyre (MITgcm-standard) for comparison to the Munk analytical solution. Add the double gyre as a follow-up test for the nonlinear regime.

### 3.2 Application of Wind Stress in Z-star

The wind stress is applied as a body force to the **top layer only**:

```
∂u/∂t = ... + τ_x / (ρ₀ Δz₁)
```

where Δz₁ is the thickness of the top z-star layer. This is how MITgcm handles it. The stress then communicates downward through the pressure gradient (barotropic mode) and vertical viscosity.

**Alternative**: Distribute the stress uniformly across all layers as τ_x/(ρ₀H). This is equivalent for the barotropic mode and avoids any vertical shear transients during spinup. MITgcm applies it to the top layer, but for a barotropic-only test, distributing it uniformly is cleaner.

**Recommendation**: Apply to top layer (consistent with MITgcm). Verify that the barotropic mode communicates the stress downward through the free surface adjustment within a few inertial periods.

---

## 4. Dissipation

### 4.1 Munk Configuration (Lateral Viscosity)

| Parameter | Value | Notes |
|-----------|-------|-------|
| A_h | 400 m²/s | Horizontal Laplacian viscosity |
| Munk layer width | δ_M = π(A_h/β)^(1/3) ≈ 100 km | Must be resolved: δ_M > 2Δx |
| Bottom drag | r = 0 | Free-slip bottom |
| Lateral BCs | No-slip | u = 0, v = 0 at walls |

### 4.2 Stommel Configuration (Bottom Drag)

| Parameter | Value | Notes |
|-----------|-------|-------|
| A_h | 0 (or minimal for stability) | No lateral viscosity |
| Bottom drag | r = 10⁻⁴ s⁻¹ | Linear Rayleigh friction |
| Stommel layer width | δ_S = r/β ≈ 50 km | Must be resolved: δ_S > 2Δx |
| Lateral BCs | Free-slip | ∂u/∂n = 0 at walls |

**Recommendation**: Run both. The Munk case has a cleaner analytical solution and tests the viscous operator. The Stommel case tests the bottom drag implementation.

### 4.3 Stability Constraints

For explicit Laplacian viscosity:
```
S_Lh = 4 A_h Δt / Δx² < 0.3
```

With A_h = 400 m²/s and Δx = 20 km: Δt < 300,000 s. Not a binding constraint.

The CFL constraint for an expected max velocity of ~2 m/s:
```
S_a = |u_max| Δt / Δx < 0.5 → Δt < 5000 s
```

**Recommended time step**: Δt = 1200 s (consistent with MITgcm).

---

## 5. Initial Conditions

| Field | Value |
|-------|-------|
| u(x,y,z,t=0) | 0 m/s |
| v(x,y,z,t=0) | 0 m/s |
| η(x,y,t=0) | 0 m |
| T(x,y,z,t=0) | 20°C (uniform) |
| S(x,y,z,t=0) | 35 PSU (uniform) |

The ocean starts from rest. The wind stress spins up the gyre over ~1-3 years (barotropic Rossby wave crossing time ~ L_x/c_R where c_R = βL_d² for barotropic mode, which is very fast — O(days) for the barotropic mode, but the frictional adjustment takes longer).

---

## 6. Boundary Conditions

| Boundary | Momentum | Tracers |
|----------|----------|---------|
| Lateral walls (all 4 sides) | No-slip (Munk) or Free-slip (Stommel) | No-flux: ∂T/∂n = 0 |
| Bottom | Free-slip (Munk) or Linear drag (Stommel) | No-flux |
| Surface | Wind stress τ_x(y) | No flux (no surface T/S restoring) |

**Critical for legoESM**: The rest-state images show that land boundary conditions are broken on most grids. This test will immediately expose whether no-flux and no-slip/free-slip conditions are correctly implemented at land-ocean boundaries.

---

## 7. Analytical Solution (Munk, Linearized)

With momentum advection switched off (linear equations), the steady-state free surface height is:

```
η(x,y) = (τ₀ f₀)/(ρ₀ g H β) · π sin(πy/L_y) · (1 - x/L_x)
          × [1 - exp(-x/(2δ_M)) · (cos(√3 x/(2δ_M)) + (1/√3)sin(√3 x/(2δ_M)))]
```

where δ_M = (A_h/β)^(1/3).

**Interior Sverdrup transport**:
```
V_Sv = curl(τ)/(ρ₀ β) = (τ₀ π)/(ρ₀ β L_y) sin(πy/L_y)
```

Maximum interior velocity ~ τ₀π/(ρ₀βHL_y) ≈ 1.3 × 10⁻³ m/s.

**Western boundary current velocity** ~ V_Sv × L_x/δ_M ≈ 0.016 m/s (for the linear case; much stronger with nonlinear terms).

---

## 8. Diagnostics and Validation Criteria

### 8.1 Qualitative Checks (Visual)

1. **SSH field** at steady state: should show a single gyre with western intensification. The SSH maximum should be in the basin interior, with the strongest gradient on the western wall.
2. **No grid imprint**: On the cubed-sphere, the gyre should be smooth with no evidence of panel boundaries in the SSH or velocity fields.
3. **No land-mask artifacts**: No spurious velocities, SSH anomalies, or mass accumulation along the basin walls.

### 8.2 Quantitative Metrics

| Metric | How to Compute | Expected Value | Threshold |
|--------|---------------|----------------|-----------|
| Sverdrup transport | ∫v dx at mid-basin | τ₀π sin(πy/L_y)/(ρ₀β) | Within 10% of analytical |
| Western BC transport | ∫v dy at x = 0-100 km | Equal and opposite to Sverdrup | Mass conservation check |
| Max SSH (linear) | max(η) | Compare to analytical Munk | L2 error < 5% at Δx=20 km |
| Max SSH (nonlinear) | max(η) | Compare to MITgcm reference | Cross-model consistency |
| Mass conservation | ∫∫η dA over time | Should be constant | Drift < 10⁻¹⁵ per step |
| Energy equilibrium | d(KE+PE)/dt | Should → 0 at steady state | < 10⁻³ of wind input |
| Barotropicity | std_z[u(x,y,z_k)] | 0 | < 10⁻¹⁰ m/s |
| Convergence rate | L2 error vs Δx | ~2nd order for FV schemes | Rate ≥ 1.5 |

### 8.3 Convergence Test Protocol

Run at minimum 3 resolutions: Δx = 40, 20, 10 km.
1. Compare linearized (no advection) solution to Munk analytical at each resolution.
2. Compute L2 norm of (η_numerical - η_analytical) weighted by area.
3. Compute convergence rate: log(e₁/e₂)/log(Δx₁/Δx₂).
4. Expected: 2nd order for FV schemes.

### 8.4 Time Evolution Diagnostics

1. **Spinup curve**: Plot domain-integrated KE vs time. Should grow from zero and plateau at O(1-3 years).
2. **Energy balance**: Wind power input = τ·u (integrated) should equal viscous dissipation at steady state.
3. **Maximum velocity timeseries**: Should approach a constant. If growing without bound → instability.

---

## 9. Common Failure Modes

| Failure | Symptom | Likely Cause |
|---------|---------|--------------|
| No circulation develops | η stays at zero, u=v=0 | Wind stress not being applied, or applied but balanced by erroneous drag |
| Circulation in wrong direction | Anticyclonic when should be cyclonic (or vice versa) | Sign error in wind stress or Coriolis |
| No western intensification | Symmetric gyre | β=0 (Coriolis gradient not implemented) |
| Mass loss/gain | η drifts globally | Free surface solver not conservative, or boundary leakage |
| Grid imprint in SSH | Panel boundaries visible (cubed sphere) | Halo exchange, metric terms, or PGF errors at panel edges |
| Velocity spikes at walls | Large \|u\| at first wet point | Incorrect no-slip/free-slip BC implementation |
| Baroclinic contamination | Vertical shear in u | Spurious density gradients created, or incorrect pressure coupling between levels |
| Spurious flow at land boundaries | Velocity parallel to walls where should be zero | Land mask not applied to velocity or pressure fields |
| Checkerboard noise | 2Δx oscillation in η | Pressure-velocity mode, missing stabilization |

---

## 10. Relationship to MOM6 Double Gyre

The MOM6 `ocean_only/double_gyre/` example differs from this test in important ways:

1. **MOM6 uses stacked shallow water (isopycnal layers)**, not z-star. The density is uniform within each layer by construction. There is no spurious diapycnal mixing issue.

2. **MOM6 double gyre is typically 2-layer** with reduced gravity g' at the interface. This creates a baroclinic mode (first internal mode) in addition to the barotropic mode.

3. **MOM6 double gyre uses the double-gyre wind** (cos(2πy/L)) rather than single gyre.

4. **Resolution in MOM6 studies is typically 1/4° to 1/64°**, much higher than the 20 km used here, because the focus is on mesoscale eddy dynamics and parameterization.

For legoESM validation, the barotropic single-layer test described here should come **before** the MOM6-style double gyre, because it isolates the boundary condition and momentum equation issues without any vertical structure complications.

---

## 11. Extension Path

After the barotropic gyre passes:

1. **Stommel + Munk comparison**: Same setup, switch between bottom drag and lateral viscosity configurations. Both should produce western intensification with the correct boundary layer widths.

2. **Double gyre**: Change wind to cos(2πy/L). Look for jet separation, recirculation gyres. No analytical solution but rich dynamics.

3. **Baroclinic gyre (2-layer)**: Add stratification with two density classes. Requires the z-star grid to maintain a sharp interface. Tests the baroclinic PGF and vertical structure.

4. **Continuously stratified gyre**: Linear T(z) initial condition, full thermocline development. Tests ALE remapping, spurious mixing, ventilated thermocline dynamics. This is the MITgcm baroclinic gyre tutorial equivalent.

5. **Resolution convergence and eddy-permitting**: Increase resolution to permit mesoscale eddies. Compare eddy statistics to the MOM6 double gyre literature (Perezhogin et al. 2024).
