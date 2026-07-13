# Diff_CHANGELOG — CLM-ML-JAX canopy differentiable integration

Session log for making the legoESM ↔ CLM-ML-JAX multilayer-canopy integration
end-to-end `jax.grad`-differentiable and bug-free. Scope:
`docs/land/clm_ml_differentiable_integration_scope.md`.

Convention: newest entries on top. Each entry = what changed, why, how verified.

---

## 2026-07-13 — Codex round-5 [P2]s

- **Solar-geometry guard:** differentiating the WHOLE `AtmToSurface` pytree makes
  `forcing.cos_zenith` a tracer → the host `np.array(cos_zenith)` would raise a
  cryptic `TracerArrayConversionError`. Solar geometry (lat/lon/doy/cos_zenith)
  is NON-differentiable by design (host-side CLM orbital setup — you don't train
  through the Sun's position). Added a `jax.core.Tracer` check that raises a clear,
  actionable `ValueError` in diff mode. Verified: concrete arrays pass, a
  differentiated leaf is detected. Physical forcing leaves (T/q/u/v/sw/lw/p/co2)
  stay fully differentiable.
- **Pin clm-ml-jax [P2] — accepted known limitation:** the package is not on PyPI
  (local/gitignored checkout; see session-report gap #1). It cannot be pinned to a
  PyPI/Git revision from here. Enforcement is the runtime capability guard (raises
  if the installed build lacks diff-diagnostics) + the `[canopy]`-extra doc note.
  Fully resolving requires publishing clm-ml-jax (out of this integration's scope).

## 2026-07-13 — Codex round-4 [P1]: thread grid_info through the land rollout (M3)

`step_multilayer_land` / `step_multilayer_land_with_diagnostics` /
`_step_multilayer_land_impl` gain a `clm_ml_grid_info=None` param, forwarded to
`compute_clm_ml_canopy_fluxes(grid_info=...)`. A multi-step differentiable
rollout through the documented `MultiLayerLandConfig` path now works: extract a
concrete `GridInfo` once from the warm-start state (`extract_clm_ml_grid_info`),
close over it before the `jax.grad` scan, and pass it via `clm_ml_grid_info=`.
Non-CLM-ML schemes and forward-only runs ignore it (default `None`).

## 2026-07-13 — Codex rounds 2–3 fixes

- **Round-2 [P1]** (forward-path API coupling): `grid=`/`vcmaxpft_jax=`/
  `g1_MED_jax=` are now forwarded to `MLCanopyFluxes` ONLY when used (`**_opt_kwargs`),
  so the default `differentiable=False` call keeps the original `_o2ref_py`-only
  signature and does not depend on the newer upstream API.
- **Round-3 [P1]** (multi-step tracer): in a differentiated rollout of ≥2 canopy
  steps the carried `canopy_state.mlcanopy` is a tracer, so `int(ncan_canopy)`
  would raise `ConcretizationTypeError`. Added:
  - public `extract_clm_ml_grid_info(state) -> GridInfo` (concrete ints from a
    warm state), and a `grid_info=` param on `compute_clm_ml_canopy_fluxes` to
    thread them through every diff step;
  - a clear `RuntimeError` (instead of a cryptic ConcretizationError) if the
    structural ints are traced and `grid_info` was not supplied.
  Tests: fast `extract_clm_ml_grid_info(None)` guard + `grid_info` pass-through
  parity (matches the state-derived diff result to 1e-9) in the slow diff test.
- **Round-3 [P2]** (can't pin non-PyPI clm-ml-jax): documented the required
  diff-diagnostics revision in the `[canopy]` extra comment; the runtime
  capability guard is the enforcement.
- **M2 spike OOM:** the two-grad spike aborted in LLVM JIT (`Cannot allocate
  memory`) — NOT a correctness bug but scope risk #2 (reverse-mode tape of the
  ~20-layer canopy is compile/memory-heavy; two accumulated `jit_scan`
  executables in one process exhausted contiguous JIT memory despite 170 GB free
  / unlimited ulimit). Re-running the SW-grad and Vcmax-grad in SEPARATE
  processes (M1-sized peak) to validate.

## 2026-07-13 — Codex review round 1: fix [P1] (silent stale outputs on old clm-ml-jax)

`codex review --base 281f2b0ce` raised one **[P1]**: `differentiable=True`
depends on the upstream diagnostics fix (`clm-ml-jax` commit `517e044`), but that
package is not on PyPI (local install), so `clm-ml-jax>=0.1.0` cannot pin it — a
clean env with an older build would return before `_CanopyFluxesDiagnostics` in
diff mode and silently yield stale `shflx/lhflx/gpp/rnet`.
- **Fix:** runtime **capability guard** in the `_diff_mode` branch — probes
  `inspect.signature(_CanopyFluxesDiagnostics)` for the `grid` parameter and
  raises a clear, actionable `RuntimeError` if absent (update clm-ml-jax). Turns
  a silent wrong-answer into a loud failure. Verified the probe returns `True`
  against the patched build.

## 2026-07-13 — Durable test + contract flip; ratchet status

- Replaced the strict-xfail `test_grad_lhflx_wrt_T_lowest` with a real
  differentiable-mode test (warm-start → forward/diff parity → grad + FD),
  marked `slow`; added a `differentiable=True ∧ ncol>1 → ValueError` test. Fixed
  the obsolete "requires JAX-native rewrite" comment.
- `__physics_contract__["differentiable"] = True` (gated on the config).
- **Ratchets:** `test_no_inline_physics_coeffs` / `test_no_saturation_reimpl`
  pass; my `clm_ml_interface.py` + `config.py` edits are clean (kept `# coeff-ok`
  escapes, `alpha=min(dt/(10·86400),1)` computed once — no literal-count growth).
  `test_no_hardcoded_constants` has ONE failure in
  `packages/land/legoesm/land/clm_surface_map.py`
  (`_TUNED_PFT_ROOT_DEPTH_MULTILAYER`) — **pre-existing, unrelated** (file
  untouched here; introduced by PRs #906/#893). Left for the owning feature to
  annotate/baseline; flagged so it is not mistaken for diff-work fallout.

## 2026-07-13 — No regression + M2 trainable-param wiring

- **Regression gate GREEN:** `tests/land/unit/test_canopy.py -m "not slow"` →
  **25 passed, 1 xfailed** (113 s). The jnp SW partition, traced soil inputs, and
  upstream diagnostics change did NOT alter any forward-path result. The
  `differentiable=False` xfail (`test_grad_lhflx_wrt_T_lowest`) still correctly
  raises under `jax.grad`, confirming the forward-only path is untouched and the
  diff gate is what flips behaviour.
- **M2 wiring:** `compute_clm_ml_canopy_fluxes` gains optional `vcmaxpft_jax`
  (per-PFT Vcmax25 `(mxpft+1,)`) and `g1_medlyn_jax` overrides, forwarded to
  `MLCanopyFluxes(vcmaxpft_jax=, g1_MED_jax=)`. These are TRACED leaves injected
  from the loss (SegmentForcing doctrine) → Vcmax25 (and Medlyn g1 when
  `gs_type==0`) become differentiable/trainable. Validated by `spike_m2.py`
  (grad wrt sw_down through the jnp partition + grad wrt Vcmax25, both FD-checked).

## 2026-07-13 — M1 GATE PASSED ✅ (forward/diff parity + grad + FD)

Spike `scratchpad/spike_m1.py` (single column, warm-start → diff forward → grad):
- **Forward/diff parity** (same warm state, `differentiable=True` vs `False`):
  `shflx/lhflx/G_soil/sw_net/lw_up/gpp/T_surface` all match to **rel ≈ 1e-14**
  (machine precision) → running `_CanopyFluxesDiagnostics` in diff mode with
  `grid=` reproduces the forward physics exactly.
- **`jax.grad(sum lhflx) / d(T_lowest) = 5.848`** — finite, non-zero.
- **Finite-difference check:** analytic `5.84768` vs central-FD `5.84784`,
  **rel_err 0.003%** (well under the 10% gate).
- Cost (CPU, no GPU): warm-up cold call 34 s; diff forward 37 s; the grad's
  first-trace `jit_scan` compile ~4 min → grad wall-time ~7 min (scope risk #2,
  compile-bound; correctness unaffected). The durable test is marked `slow`.
- `mlcanopy` ncan resolved to 20 layers (htop fallback 5 m → dz_short grid), not
  the nominal `nlevmlcan=9` — pre-existing upstream layering, unrelated to diff.

## 2026-07-13 — Diff-mode activation (upstream + interface + config)

**Upstream `clm-ml-jax` (editable install; separate git repo):**
`src/multilayer_canopy/MLCanopyFluxesMod.py`
- `_CanopyFluxesDiagnostics` gains `grid=None`. In diff mode it reads `ncan/ntop`
  from `grid` (concrete pre-traced ints) instead of `int(mlcanopy_inst.*)`, and
  the four host-syncing `abs(err) … endrun` energy-balance checks are skipped
  (they concretise a traced scalar). Dropped the `float()` cast on `tref_forcing`
  in the `flux_profile_type==1` path (LatVap already accepts a jnp scalar). Flux
  arithmetic identical in both modes.
- `MLCanopyFluxes`: removed the diff-mode **early return** that skipped
  diagnostics; diagnostics now runs in BOTH modes (with `grid=`), so the
  canopy-integrated outputs `shflx/lhflx/etflx/gpp/rnet/swveg/albcan/taveg/
  stflx_air/stflx_veg` — exactly what `_extract_surface_fluxes` reads — are
  populated on the grad tape. The sun/shade merge (already pure-JAX) also runs in
  both modes, keeping the prognostic `tleaf/lwp` warm-start carry correct.
- `DIFFERENTIABLE_MODE` is **vestigial** (imported in `MLCanopyTurbulenceMod` but
  never read; every module switches on `grid`). Left in place; the interface
  still sets it for intent/forward-compat. So the scope's "must set
  DIFFERENTIABLE_MODE=True" is now really "pass `grid=`".

**legoESM `clm_ml_interface.py`:**
- `_estimate_beam_fraction` + `_sw_partition` rewritten **jnp-native** (were
  `np.asarray`), so `d(swsky*)/d(sw_down)` (incl. the clearness-index `f_dir`
  dependence) stays on the tape. `# coeff-ok:` Erbs escapes preserved. The
  standalone `_estimate_beam_fraction` unit tests still pass numpy in and
  `float()` out — values unchanged.
- `_build_stubs`: dropped `float()` on the traced soil inputs (`t_soisno` from
  `T_soil`/`T_soil_top`, `hk_l` from `K`), so `d(flux)/d(T_soil,psi,theta)` flows.
- `compute_clm_ml_canopy_fluxes`: added the static `_diff_mode` gate
  (`config.differentiable ∧ warm-started ∧ ncol==1`); a multi-column diff request
  is a hard `ValueError` (no silent degrade). Cold start always runs forward to
  build the vertical structure. Builds `GridInfo(p,ncan,ntop,nbot)` from the warm
  template (concrete ints) and passes `grid=` to `MLCanopyFluxes`; sets
  `MLclm_varctl.DIFFERENTIABLE_MODE`. The `t_a10` acclimation running mean is now
  traced jnp in diff mode (was `np.array(forcing.T_lowest)` → would raise on a
  tracer). Geometry (`cos_zenith`/lat/lon/doy) stays a non-differentiated host
  constant, by design.

**Config `CLMMLCanopyConfig`:** added `differentiable: bool = False` static gate
(bool → not `__param_spec__`-eligible; no spec entry needed). Production stays
forward-only; training opts in.

**Known deferred (M3):** structural fields (`htop`/LAI/SAI/root) stay host
constants; multi-column diff is via `vmap` (not yet wired into
`multilayer_land.step_*`). Geometry is intentionally non-differentiated.

**Push:** this host has no GitHub credentials (no `gh`, no token, no credential
helper) → `git push` fails auth. Commits accumulate locally; push with `!git push`
from a credentialed context.

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
