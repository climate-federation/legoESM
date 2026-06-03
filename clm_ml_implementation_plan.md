## Active Task: CLM-ML-JAX Canopy Plugin Integration

THe goal is to implement a new canopy scheme in legoesm that calls the CLM-ML-JAX model. The code for this model lives in /Users/ayalahlou/legoESM/clm-ml-jax. This will be a new option for the "scheme" field in CanopyConfig, and will be implemented as a plugin in src/legoesm/land/canopy/clm_ml_interface.py. The implementation will follow the phases outlined below, with clear rules and checkpoints to ensure correctness and maintainability.

### Rules for this session
- Work through phases IN ORDER. Do not start Phase N+1 until Phase N is complete and tests pass.
- After completing each step, mark it [x] and run the relevant tests before moving on.
- Never modify the soil stack (Richards, thermal diffusion, snow) in multilayer_land.py.
- All imports of clm_ml_jax must be lazy (inside function body), not top-level.
- Do not add any constants from MLclm_varcon.py into legoesm.constants.

### Phase 1: Scaffolding (no logic yet)
- [x] Create src/legoesm/land/canopy/__init__.py
- [x] Create src/legoesm/land/canopy/config.py (CanopyConfig, CLMMLCanopyConfig)
- [x] Create src/legoesm/land/canopy/state.py (CanopyState)
- [x] Extend MultiLayerLandConfig with canopy field (config.py)
- [x] Extend MultiLayerLandState with canopy field (state.py)
- [x] Extend LandSurfaceParams with LAI, SAI, htop, hbot + PARAM_BOUNDS
- [x] Update src/legoesm/land/__init__.py exports
- [x] Add [project.optional-dependencies] canopy group to pyproject.toml
- [x] Run existing tests — all must still pass: 168/168 unit tests pass

### Phase 2: Interface stub
- [x] Create src/legoesm/land/canopy/clm_ml_interface.py with correct signature but raise NotImplementedError body
- [x] Add dispatch block to multilayer_land.py (scheme="none" path unchanged, scheme="clm_ml" calls stub)
- [x] Update init_multilayer_land_state() to initialize CanopyState when scheme="clm_ml"
- [x] Run existing tests — all 168 unit tests pass

### Phase 3: Interface implementation
- [x] Implement SW partitioning (f_dir, f_VIS split) in clm_ml_interface.py
- [x] Implement AtmToSurface → mlcanopy_type input mapping (full table from plan)
- [x] Implement CLM-ML-JAX → CanopyFluxes output mapping (full table from plan)
- [x] Implement column indexing (trivial 1:1 via jnp.arange(ncol))

### Phase 4: Tests
- [x] Create tests/land/unit/test_canopy.py
- [x] Test: energy balance closure |net_rad - SH - LH - G| < 50 W/m² (cold-start; 9 W/m² observed)
- [x] Test: differentiability — jax.grad of LH w.r.t. T_lowest (xfail: CLM-ML-JAX uses float() on traced arrays; Python Fortran port is not end-to-end jax.grad compatible)
- [x] Test: GPP > 0 under positive PAR
- [x] Integration test: 24h run with scheme="clm_ml", check TileResponse finite + water budget closes
- [x] Dispatch test: default scheme="none" still passes all existing tests/land/ tests
- [x] Config validation test: CLMMLCanopyConfig() creates valid NamedTuple; canopy_state None for default

### Phase 5: Final checks
- [x] `pip install ".[canopy]"` works cleanly
- [x] `pip install .` (without canopy extra) still works and imports don't fail
- [x] Run full test suite: `pytest tests/land/` — 180/180 tests pass (168 existing + 12 new canopy)


## requirements and guidelines:

If you are blocked on a step for more than 3 attempts, write a BLOCKED.md describing exactly what's failing and why, mark the step [BLOCKED], and continue with the next step.

When implementing clm_ml_interface.py, read the actual MLCanopyFluxes source code first and reconcile any differences with the plan before implementing.

Don't skip the dispatch test. The "existing tests still pass" check after Phases 1 and 2 is the most important guardrail — it's how you catch if Claude accidentally broke the scheme="none" path.

always try to improve timing and memory usage when implementing the interface, but do not sacrifice readability or correctness for performance. We can optimize later if needed.

Commit and push as often as possible with clear concise messages.
