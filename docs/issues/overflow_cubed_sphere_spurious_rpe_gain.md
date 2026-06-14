# Overflow on cubed-sphere: small positive PE-proxy drift (strict-sign gate fail)

## Symptom
`scripts/matrix/run_ocean_test_matrix.py --only overflow --grid cubed_sphere`
fails the `pe_rel_final < 0` gate: the PE proxy drifts slightly POSITIVE
instead of decreasing as the dense plume descends.

| duration | PE_rel_final | mean-T drift | T range (init [0,20] °C) |
|----------|--------------|--------------|--------------------------|
| 0.10 d (quick) | +6.47e-7 | ~1e-15 | — |
| 0.50 d (full)  | +4.48e-6 | ~5e-15 | [0.00, 20.04] |
| 1.00 d         | +1.04e-5 | ~2e-14 | [0.00, 20.10] |

`latlon` overflow passes (PE_rel < 0, no overshoot).

## What it is NOT (empirically ruled out, 2026-06-14, each tested + Codex-reviewed)
The original "non-thickness-weighted tracer transport / non-conservation"
hypothesis is **wrong**. Tracer **content is well conserved** (mean-T drift
~1e-14). Four conservation fixes were implemented and each left PE_rel
essentially unchanged (≈4.4e-6 @0.5d) — all reverted:

1. **C-face wet/rock masking** — already present in the code; no effect on overflow.
2. **Flux-form vertical tracer advection** (÷h_k vs advective −w·∂q/∂z): 4.4791e-6 → 4.4770e-6.
3. **Thickness-weighted horizontal FCT** (arithmetic h_face, ÷h_k): 4.48e-6 → 4.38e-6.
4. **Exact shared mass flux** (FCT reuses `cgrid_mass_flux_divergence`'s synced
   PPM face flux so the tracer mass-flux divergence == the w-diagnosis one →
   discrete continuity / constant-tracer preservation): 4.38e-6 — no further change.

So the residual is NOT bulk horizontal/vertical advective non-conservation.

## Most likely actual cause (narrowed, not yet fixed)
A **tiny, growing non-monotone overshoot** (T_max 20.04 → 20.10 °C over 1 day,
i.e. ~0.04→0.10 °C above the global initial max) — a *local monotonicity*
violation, not a content-conservation one. The FCT limiter's local bounds
(`operators_cdgrid.py` ~859-869) use only the **4 face-adjacent** neighbours
(halo-1); at the 8 cube **corners / panel seams** a cell's diagonal/cross-seam
neighbours are not in the bounds, so the PPM reconstruction can produce a small
overshoot there that the limiter does not clip. Accumulated over the run this
is the ~1e-5 positive drift in the PE proxy.

Two compounding test-harness factors:
- `_compute_rpe` (experiments.py:1236) is a **crude PE proxy**, NOT true sorted
  RPE: it integrates `rho(T)·z_ref·dz_ref·area` on the REFERENCE (J=1) geometry,
  ignoring the eta-stretched z* layer positions.
- the gate is **strict-sign** (`pe_rel_final < 0`), so ANY positive drift —
  including ~1e-5 cube-seam numerical noise — fails, while latlon (no seams)
  squeaks negative.

## Fix paths (deferred — need visual cube-imprint validation)
- **(preferred) cube-corner-aware FCT limiter**: include the diagonal/cross-seam
  neighbours in the `q_min`/`q_max` bounds (halo-2 corner fill) so monotonicity
  holds at panel corners. Touches `_cgrid_fct_fluxes_2d` (`operators_cdgrid.py`);
  per CLAUDE.md cube-tracer changes MUST be visually verified (cube imprint /
  seam grid-scale noise is not caught by norms) — cannot be signed off headlessly.
- **(diagnostic) replace the crude PE proxy with true sorted RPE on the actual
  z* geometry** and reconsider the strict-sign gate vs a small tolerance, so the
  test reflects real conservation rather than ~1e-5 seam noise. Do NOT merely
  loosen the gate to mask a real corner-monotonicity residual.

Until then `overflow/cubed_sphere` is a **known, documented** tiny-positive-drift
limitation (tracer content conserved to ~1e-14; overshoot ~0.1 °C). The `latlon`
overflow is the conservation-faithful reference.

## Reproduce
```
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --only overflow --grid cubed_sphere --days 1.0
```
Watch `PE_rel` grow slightly positive and `T_max` exceed 20 °C by ~0.1 °C while
the mean-T drift stays ~1e-14.
