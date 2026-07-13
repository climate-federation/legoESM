# Scope: Make the legoESM CLM-ML-JAX canopy integration differentiable (2026-07-13)

## Goal
Enable end-to-end `jax.grad` through the CLM-ML-JAX multilayer canopy **as wired into
legoESM** — i.e. make `compute_clm_ml_canopy_fluxes(forcing, ...) -> SurfaceFluxOutput`
differentiable w.r.t. the atmospheric forcing and (optionally) trainable canopy
parameters, so gradients flow through the land column in the ESM's autodiff graph.

## Key discovery (corrects the earlier "not differentiable" claim)
The `clm-ml-jax` repo (AyaLahlou/clm-ml-jax, local clone == GitHub main) **already
supports a differentiable mode.** It is *not* a dead Fortran/NumPy port. The forward-only
vs differentiable behaviour is selected by **two toggles**, both currently OFF in the
legoESM interface:

1. **Per-call `grid` argument** → `MLCanopyFluxesMod.py:300`:
   `_diff_mode = grid is not None`. When a `GridInfo` is passed, the model:
   - runs `@jax.jit` / `lax.scan` paths (jnp throughout),
   - reads structural ints (`ncan/ntop/nbot/p`) from `grid` instead of `int(tracer)`,
   - gates out host-syncing Python checks behind `if not _diff_mode:`.
   Same switch in `MLLeafPhotosynthesisMod.py:2324`, `MLRungeKuttaMod.py:98`.
2. **Module global `MLclm_varctl.DIFFERENTIABLE_MODE`** (default `False`, `MLclm_varctl.py:38`)
   — imported by `MLCanopyTurbulenceMod.py:75` to skip `endrun`, diagnostic file I/O, and
   other tape-breaking ops. Must be set `True`.

Reference calling convention: **`make_clm_ml_forward(...)`** (`MLCanopyFluxesMod.py:2063`)
builds a closure `forward(mlcanopy_inst) -> scalar` that is `jax.grad`-able; exercised by
`clm-ml-jax/tests/test_differentiability.py` (grad of `sum(shair)+sum(etair)` w.r.t.
`mlcanopy_inst` state; finite + FD-checked). `MLCanopyFluxes` also accepts trainable
`vcmaxpft_jax` and `g1_MED_jax` as JAX arrays → Vcmax25 and Medlyn g1 are differentiable.

## Why the legoESM wiring is currently forward-only
`packages/land/legoesm/land/canopy/clm_ml_interface.py`:
- **Calls `MLCanopyFluxes(...)` WITHOUT `grid=`** (line 1277) → `_diff_mode = False`.
- **Never sets `DIFFERENTIABLE_MODE = True`.**
- **Marshals inputs through NumPy** in `_build_stubs` (502), `_init_mlcanopy` (805): builds
  `np.zeros/np.full` arrays, some wrapped back into `jnp.array(...)` as **constants** — so
  the numeric forcing leaves are *disconnected* from the traced `forcing` pytree.
- **Side-effecting host init inside the call**: `_ensure_clm_initialized()` (1158; netCDF4
  RSL lookup load + global `MLpftcon` init), `_setup_clm_topology` (136), `_setup_clm_time`
  (254) mutate module globals and touch the host — fine as one-time static setup, but they
  must not sit inside the differentiated region.

`differentiable: False` in the physics contract and the `xfail` in
`tests/land/unit/test_canopy.py::test_grad_lhflx_wrt_T_lowest` describe **this wiring**,
not the model's capability. The xfail's comment ("requires a JAX-native reimplementation")
is **outdated** — the JAX-native diff path already exists upstream.

## Required changes (file-by-file)

### A. `clm_ml_interface.py` — split static setup from the traced step
- Hoist `_ensure_clm_initialized`, `_setup_clm_topology`, `_setup_clm_time`, RSL/netCDF
  load, and structural-int extraction (`GridInfo`) into a **one-time setup** that runs
  before `jax.grad` tracing and returns concrete Python ints + constant lookup tables.
- Provide a **pure per-step forward** `f(forcing_pytree, params) -> SurfaceFluxOutput` that
  contains ONLY jnp physics (no `np.`, no `float(tracer)`, no `.item()`, no I/O).

### B. `clm_ml_interface.py` — build `GridInfo` and pass it
- Extract `GridInfo(p, ncan, ntop, nbot)` from the warm-start template `mlcanopy_inst`
  (concrete ints), mirroring `make_clm_ml_forward` (2098-2104).
- Set `MLclm_varctl.DIFFERENTIABLE_MODE = True` in the setup phase (static Python bool).
- Call `MLCanopyFluxes(..., grid=grid, _o2ref_py=<concrete float>)`.

### C. `clm_ml_interface.py` — make the forcing→state mapping traced (jnp)
- The forcing entry points the model reads must carry **traced jnp** derived from the
  `forcing` argument:
  - `mlcanopy_inst.*_forcing` (`uref/tref/swskyb/swskyd/..._cur_forcing`, set via
    `.at[p].set(jnp_value)` — see MLCanopyFluxesMod.py:414-421), and
  - `atm2lnd_inst.forc_*` (`forc_u_grc`, `forc_t_downscaled_col`, `forc_solad...`,
    `forc_pco2`, `forc_po2`, ... MLCanopyFluxesMod.py:959-965).
- In `_build_stubs`/`_init_mlcanopy`, arrays that receive forcing must be `jnp.zeros(...)`
  populated from the traced `forcing`, **not** `np.full/np.zeros` constants. Purely
  structural/index arrays (`patch.column`, `snl`, `dz`, `nbedrock`, lat/lon) may stay
  constant.

### D. `clm_ml_interface.py` — keep output extraction on the tape
- `_extract_surface_fluxes` (925) already uses jnp; audit for any `float()`/np on the
  `mlcanopy_new` → `SurfaceFluxOutput` path and remove.

### E. Repo (optional, avoids duplication) — return full state, not a scalar
- `make_clm_ml_forward` returns a scalar loss. legoESM needs the full `mlcanopy_type`
  output to extract multiple fluxes. Either (E1) add an upstream
  `make_clm_ml_forward_state(...)` / `reduce=None` variant returning `inst`, or (E2)
  replicate its ~15-line grid-extraction closure inside legoESM and call `MLCanopyFluxes`
  directly. **Recommend E2** (keeps the repo untouched; the closure is tiny).

### F. `config.py` / contract
- Flip `__physics_contract__["differentiable"]` to `True` **only** for the diff-mode path;
  add a `CLMMLCanopyConfig.differentiable: bool = False` gate so production stays
  forward-only (fast, no grad tape) and training opts in.

## Design decisions
- **Single-column first.** The diff path assumes single site (`grid.p = filter[0]`,
  scalar `ncan`). Restrict M1 to `ncol == 1`; add `vmap` over columns in M3.
- **Trainable params** injected as `vcmaxpft_jax`/`g1_MED_jax` (already supported) via the
  config pytree inside the loss (SegmentForcing doctrine — traced leaves in the loss,
  static floats in production).
- **netCDF4/f90nml** stay as install deps; RSL lookup is one-time setup, outside the tape.

## Risks / unknowns (verify during M1 spike)
1. **Non-smooth ops → NaN grads.** Hard clips / `jnp.where` boundaries can give NaN
   reverse-mode grads. Precedent: the native two-leaf `test_prognostic_lai_jax_grad_through_feedback`
   XPASS/NaN issue at the `LAI_CRIT=2` clip. CLM-ML has many clips (btran, stomatal,
   longwave) — expect to smooth or `custom_jvp` a few.
2. **Compile time & memory.** The model is thousands of ops; first-trace `jit` compile and
   reverse-mode tape storage may be heavy. Measure; consider `remat`/checkpointing.
3. **Global mutable CLM state** (`MLclm_varctl`, `clm_instMod`) — must be treated as static
   constants; any per-step-varying global read on the traced path breaks jit caching.
4. **Diff-mode coverage.** Upstream diff tests only check grad w.r.t. `mlcanopy_inst` state
   → scalar. Grad w.r.t. *forcing* through the full flux extraction exercises paths the
   repo's tests do not; new break points are likely.
5. **Forward/diff numeric parity** must be verified (same fluxes to tolerance between
   `grid=None` and `grid=GridInfo`).

## Milestones + validation
- **M1 (spike, ncol=1):** static setup + traced `f(T_lowest) -> sum(lhflx)`; assert
  `jax.grad` finite, non-zero, and matches finite-difference (loose tol). Confirm
  forward/diff flux parity. *Gate: FD check passes.*
- **M2:** full `SurfaceFluxOutput` differentiable w.r.t. all forcing fields + trainable
  `vcmax`/`g1`; unit tests per flux (shflx/lhflx/gpp) grad finite + FD.
- **M3:** `vmap` to `ncol>1`; wire diff-mode into `multilayer_land.step_*` so end-to-end
  land-column `jax.grad` works; add an integration grad test.
- **M4:** flip contract `differentiable: True` (gated), replace the `xfail` with a real FD
  gradient test, fix the outdated xfail comment, update docs + CHANGELOG.

## Anchor references
- Interface: `packages/land/legoesm/land/canopy/clm_ml_interface.py`
  (`compute_clm_ml_canopy_fluxes` 1085, `_build_stubs` 502, `_init_mlcanopy` 805,
  `_extract_surface_fluxes` 925, call site 1277).
- Repo: `clm-ml-jax/src/multilayer_canopy/MLCanopyFluxesMod.py`
  (`MLCanopyFluxes` 168, `_diff_mode` 300, `make_clm_ml_forward` 2063, `GridInfo` build 2099);
  `MLclm_varctl.py` (`GridInfo` 22, `DIFFERENTIABLE_MODE` 38);
  `tests/test_differentiability.py` (reference grad + FD check).
