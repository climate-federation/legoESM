# Diff_CHANGELOG — CLM-ML-JAX canopy differentiable integration

Session log for making the legoESM ↔ CLM-ML-JAX multilayer-canopy integration
end-to-end `jax.grad`-differentiable and bug-free. Scope:
`docs/land/clm_ml_differentiable_integration_scope.md`.

Convention: newest entries on top. Each entry = what changed, why, how verified.

---

## 2026-07-13 — Session start: environment repair + baseline

**Environment (this Linux/burg host, not the report's Mac `.venv`):**
- `clm-ml-jax` is an editable install (`.local/.../__editable__.clm_ml_jax-0.1.0.pth`)
  → upstream edits under `/burg-archive/home/al4385/clm-ml-jax/src` take effect live.
- `legoesm` is **not** pip-installed here; it resolves via `PYTHONPATH` over the 8
  `packages/*` roots (federation layout), matching the report's run recipe.
- **Blocker found:** shared `.local` had `numpy 1.26.4` but `jax 0.9.2` requires
  `numpy>=2.0` → every `import jax` failed (`np.dtypes.StringDType` missing).
- **Fix (isolated, does NOT mutate shared `.local`):** created
  `.venv_diffwork` (`python -m venv --system-site-packages`) and installed
  `numpy>=2.0,<2.3` (+ `equinox`) into it; its site-packages shadow the stale
  `.local` numpy for this venv only. `jax.grad` works (CPU fallback; cuDNN absent).
- netCDF4 (built vs numpy 1.x) emits a benign `ndarray size changed` ABI
  RuntimeWarning under numpy 2.x; harmless for reads (forward smoke test passes).
  The upstream repo's pytest escalates it to error → run its tests with
  `-W ignore::RuntimeWarning -o filterwarnings=`.
- Runner helper: `scratchpad/run.sh` (sets venv + PYTHONPATH + JAX_ENABLE_X64 + CPU).

**Baseline verified:**
- `tests/land/unit/test_canopy.py::TestCLMMLInterface::test_smoke_daytime` PASSED
  (78 s) → forward (non-diff) path is healthy on this env.

**Key architecture finding (drives the plan):**
- In diff mode (`grid != None`), upstream `MLCanopyFluxes` **returns early**
  (MLCanopyFluxesMod.py ~L766) BEFORE `_CanopyFluxesDiagnostics`. That function is
  what computes `shflx_canopy / lhflx_canopy / gppveg_canopy / rnet_canopy /
  swveg_canopy / taveg_canopy / stflx_air_canopy / stflx_veg_canopy` — exactly the
  fields `_extract_surface_fluxes` reads. So activating diff mode naively yields
  stale/`spval` outputs. Fix: make `_CanopyFluxesDiagnostics` diff-safe and run it
  in diff mode (scope item E, done upstream rather than duplicating ~480 LOC of
  flux-aggregation numerics in legoESM — honours the "no duplicate numerics" rule).
