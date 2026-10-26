# LMIP global differentiability runs — plan (2026-10-26)

Status: PLANNED, nothing built yet. Branch `land/lmip-calibration`.

## Goal

Two global land-only runs on Derecho with `scripts/run/run_lmip_biophys.py`,
real CRU-JRA forcing, 2-degree lat-lon grid, 1-hour step:

1. **Forward run, 1 month** (720 steps). Plain model run, standard output tapes.
2. **Reverse pass, 1 month.** Same run, plus the gradient of a scalar
   objective with respect to chosen inputs, computed by reverse-mode AD
   through the whole month.

Forward-mode AD (JVP) is out of scope: the canopy Newton solver
(`core/nonlinear.py::make_implicit_newton_solver`) and the soil tridiagonal
solve are `custom_vjp`, so `jax.jvp`/`jacfwd` raise
`TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.
Measured 2026-10-26 on a 4-step loam column: primal and `jax.grad` work, `jvp`
raises. This has been true since the canopy solver was introduced
(2026-04-12); it is not a regression. The unmerged branch
`fix/1736-hvp-while-loop` enables forward-over-reverse (HVPs), not plain JVP.

## Run 1 — forward, 1 month

Works today; needs only a config. Copy `config/lmip/lmip_biophys_soil_ic.yaml`
and change:

- `forcing.year_start` / `year_end`: the chosen year (same value)
- `time.n_steps: 720`, `time.dt: 3600.0`, `time.start_doy`: month start
- `forcing.data_dir`: the CRU-JRA directory on Derecho
- `restart.from`: a spun-up land restart (a cold start makes a 1-month window
  meaningless for soil water)

```
JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \
    --config config/lmip/<new>.yaml --output-dir <scratch>/lmip_fwd_1mo
```

## Run 2 — reverse pass, 1 month (NOT BUILT)

The driver has no gradient path. Work needed:

- **Decisions still open (user):**
  1. Objective: e.g. global area-weighted mean ET over the month, or a
     per-cell field (then one reverse pass per scalar, or a VJP with a chosen
     cotangent map).
  2. Differentiate with respect to: calibrated plant parameters (Vcmax, g1,
     root depth, ...), forcing (precip, air T), or initial soil state.
- **Implementation sketch:** a `--grad` mode that wraps the per-year
  `lax.scan` in a loss function, applies parameter overrides INSIDE the loss so
  the leaves are traced (SegmentForcing / `apply_param_overrides` doctrine),
  `jax.checkpoint` on the step body for memory, and writes the gradient
  maps/values as NetCDF next to the tapes.
- **Known AD hazards in the driver:**
  - The per-column NaN-revert guard (`jnp.where` back to the previous state)
    silently zeroes the gradient through reverted steps. Report the revert
    count with every gradient.
  - The canopy implicit solver zeroes its gradient on steps where Newton did
    not converge (max_iters). Expect a few columns (seen in the column demos:
    tall tropical canopies at night). Report the unconverged count.
  - Tape accumulators and host-side flushes must stay outside the
    differentiated function.
- **Validation before quoting any number:** finite-difference check of the
  gradient on >=1 parameter (central difference, ratio ~1), and gradients
  finite and non-zero.
- **Compute:** memory of a global 2-degree, 720-step reverse pass is unknown;
  measure with a 24-step run first, then scale. Large-memory GPU nodes likely.

## Prior work on this branch

Single-column and global demos in `scripts/run/land_sensitivity/` (reverse-mode,
AD checked against finite differences). `demo_global_param_map.py` already
runs a global reverse pass on synthetic forcing; reuse its objective and
chunking logic rather than re-deriving them.
