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

## Most likely actual cause (not yet fixed)
A panel-edge effect UPSTREAM of the tracer limiter — a tiny seam-inconsistent
**C-grid normal velocity** (`dgrid_to_cgrid` / duogrid vector exchange at i=0)
or a **halo value** at the i=0 ghost cell that biases the upwind/PPM face value
each step. Surface-only (k=0) points at the free-surface / rigid-lid edge
coupling. This needs instrumented probing of `u_c` and the tracer halo AT the
i=0 seam (not more transport-scheme changes), plus mandatory cube-imprint
visual validation (CLAUDE.md) before any fix is trusted — neither closable
headlessly.

## Separate bug found
The **modular** `scripts/matrix/ocean_test_matrix/experiments.py:run_overflow`
is a divergent DUPLICATE of the monolithic `run_ocean_test_matrix.py:5894`
runner the CLI actually uses (`RUNNERS`, line 6261/8134). The modular one uses a
different IC (`ov_ic` vs `_create_rest_state` + `_init_overflow`) that is
**unstable — it blows up to NaN in ~6 steps**. It appears unused by the CLI;
it should be removed or reconciled with the monolithic runner.

## Status
overflow/cubed_sphere is a **known, documented** tiny panel-edge overshoot
(content conserved to 1e-14; T_max ≤ ~20.1 °C). Root cause narrowed to the
panel-edge velocity/halo (NOT the tracer transport — six transport fixes had
zero effect). `latlon` is the conservation-faithful reference.

## Reproduce
```
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --only overflow --grid cubed_sphere --days 1.0
```
