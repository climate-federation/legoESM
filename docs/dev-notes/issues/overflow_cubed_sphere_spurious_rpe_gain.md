# Overflow on cubed-sphere: small panel-edge surface-tracer overshoot (strict PE gate fail)

## Symptom
`scripts/matrix/run_ocean_test_matrix.py --only overflow --grid cubed_sphere`
fails `pe_rel_final < 0`: the PE proxy drifts slightly POSITIVE (PE_rel
+6.5e-7 quick / +1.04e-5 @1.0d). `latlon` overflow passes.

## Precisely localized (instrumented, the real/monolithic stable path)
T_max grows **slowly and monotonically** above the global initial max 20.0 °C:
19.98 → 20.00 (step ~47) → 20.10 (step 288), **always at a cube PANEL EDGE
(i=0), SURFACE layer (k=0)**. Tracer mean is conserved to **1e-14**; it is NOT a
blowup (the run completes, `ok=True`). So this is a tiny, accumulating
**panel-edge surface-tracer non-monotonicity** (~1e-4 °C/step), surfaced by a
strict-sign gate on a crude domain-PE proxy (`_compute_rpe`,
experiments.py:1236, integrates `rho(T)·z_ref·dz_ref`, an admitted
approximation, not true sorted RPE).

## Empirically ruled out — it is NOT the tracer transport
Six fixes were each implemented, run through the matrix, and **reverted** —
EVERY one left PE_rel byte-identical (≈1.0449e-05 @1.0d):
1. Flux-form vertical tracer advection (÷h_k) vs advective −w·∂q/∂z.
2. Thickness-weighted horizontal FCT (arithmetic h_face), ÷h_k.
3. FCT reusing the EXACT synced PPM face mass flux from
   `cgrid_mass_flux_divergence` (constant-tracer preservation by construction).
4. C-face wet/rock masking (already present).
5. True-geometry PE diagnostic (actual eta-stretched z* `h`/centroid vs `z_ref`).
6. dt-aware Zalesak limiter (`q_td = q + dt·dq_low`, limit `dt·ad`).

(3) and (6) were Codex-confirmed as the root cause; the matrix disproved both.
The byte-identical PE_rel across all six proves the overshoot does **not**
originate in the horizontal FCT, the vertical advection, the limiter, or the
thickness weighting — the FCT limiter likely never even fires for this case.

## PINNED root cause (instrumented isolation + Codex-confirmed, 2026-06-14)
It is the **horizontal tracer DIFFUSION**, not the advection. Decisive isolation
on the real (monolithic) path with `pgf_scheme="zero"` (zero pressure force) and
zero initial velocity:
- `K_v=0` AND `K_h=0` → NO overshoot (T_max = 19.98 = exact IC max).
- `K_h` only (5e6, K_v=0) → overshoot (20.0247). `K_v` only → none (19.979).

So `laplacian_viscosity_3d(tracer_flat, grid, config.K_h)`
(`ocean_pe_cdgrid.py:635-639`, = `K_h·divergence_3d(gradient_x_3d, gradient_y_3d)`
in `ocean/physics/mixing.py`) is the source. A down-gradient Laplacian must obey
the discrete maximum principle and can NEVER create a new extremum — so the
cube **A-grid centered-gradient → vector-divergence** form (with vector-halo
rotation at panel seams) **violates the discrete maximum principle at the
non-orthogonal cube-corner metric**, going locally ANTI-DIFFUSIVE at i=0. The
default `K_h = 5e6 m²/s` (anomalously large for a TRACER diffusivity — other
ocean configs use ~0–1e3) amplifies it. Codex confirmed: the lat-lon tracer path
uses a maximum-principle-preserving **face-flux** Laplacian (`laplacian_cgrid`),
whereas the cube uses A-grid grad→div.

This is why all six advection/limiter fixes, momentum viscosity (A_h→1e5), dt
(300/100/30) and PGF had ZERO effect — the overshoot was in the diffusion the
whole time.

## Fix (deferred — needs cube-imprint visual validation)
Replace the cube tracer horizontal diffusion with a **maximum-principle-
preserving face-flux Laplacian** (down-gradient two-point face fluxes, like the
lat-lon `laplacian_cgrid`) so it cannot create extrema at non-orthogonal cube
corners — and/or correct the anomalous `K_h=5e6` tracer default. Per CLAUDE.md +
Codex this is a cube tracer-transport change requiring cube-imprint visual
validation (W2 / tracer-field PNGs at the panel boundary) before it is trusted —
not closable headlessly. Blast radius: all cube ocean cases.

## Divergent-duplicate runner bug — RESOLVED (2026-06-14)
The **modular** `scripts/matrix/ocean_test_matrix/experiments.py:run_overflow`
blew up to **NaN in ~6 steps** while the monolithic
`run_ocean_test_matrix.py:run_overflow` ran stably. The earlier "different
unstable IC (`ov_ic`)" diagnosis was **WRONG**: the library IC
(`legoesm.ocean.experiments.overflow.create_initial_conditions`) is
**byte-identical** to the old monolithic `_create_rest_state` + `_init_overflow`
across T/S/H_bathy/land_mask/eta/u/v on every grid (verified 2026-06-14).

The real cause was the **modular `_create_ocean_setup` cube branch**, which had
been extracted *without* the cube cold-start stabilization the monolithic
carries. The **dominant** stabilizer is the barotropic substep count:
`n_barotropic_substeps=30` (modular) blows up; `=60` (monolithic) is stable — a
panel-edge barotropic gravity-wave CFL limit. Ablation (overflow IC, zero
forcing): substeps=60 alone → stable; raised K_h alone → NaN@6; conservation
fixer alone → NaN@6. (Consistent with the panel-edge-velocity localization
above; it is NOT the tracer transport.)

**Fix:** the cube stabilization now lives in one shared block,
`ocean_test_matrix.setup.cube_matrix_ocean_config_kwargs` (60 substeps, raised
A_h/K_h, conservation fixer, `cube_light_diffusion` for wave tests, explicit
K_h override). BOTH `_create_ocean_setup` functions call it (monolithic cube
`OceanConfig` verified byte-identical across the full param matrix). The
overflow IC is de-duplicated to the single library `create_initial_conditions`
— the script-local `_init_overflow` is deleted. Both drivers now produce
**identical** results (cube PE_rel=6.5042e-07, latlon PE_rel=-5.7333e-08).
Guarded by `tests/ocean/unit/test_overflow_runner_parity.py`.

## Status
overflow/cubed_sphere is a **known, documented** tiny panel-edge overshoot
(content conserved to 1e-14; T_max ≤ ~20.1 °C; PE_rel=+6.5e-7 quick) caught by a
strict `pe_rel_final<0` sign gate on a crude PE proxy. Root cause is the cube
horizontal tracer **diffusion** (`laplacian_viscosity_3d` A-grid grad→div
violating the discrete max principle at non-orthogonal cube corners; K_h=5e6
amplifies) — NOT the tracer transport (six transport fixes had zero effect).
Both matrix drivers now run it identically and stably; the remaining overshoot
fix (max-principle-preserving face-flux Laplacian) is deferred pending
cube-imprint visual validation. `latlon` is the conservation-faithful reference.

## Reproduce
```
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --only overflow --grid cubed_sphere --days 1.0
```
