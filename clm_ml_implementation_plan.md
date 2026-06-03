## Active Task: CLM-ML-JAX Canopy Plugin Integration

THe goal is to implement a new canopy scheme in legoesm that calls the CLM-ML-JAX model. The code for this model lives in /Users/ayalahlou/legoESM/clm-ml-jax. This will be a new option for the "scheme" field in CanopyConfig, and will be implemented as a plugin in src/legoesm/land/canopy/clm_ml_interface.py. The implementation will follow the phases outlined below, with clear rules and checkpoints to ensure correctness and maintainability.

### Rules for this session
- Work through phases IN ORDER. Do not start Phase N+1 until Phase N is complete and tests pass.
- After completing each step, mark it [x] and run the relevant tests before moving on.
- Never modify the soil stack (Richards, thermal diffusion, snow) in multilayer_land.py.
- All imports of clm_ml_jax must be lazy (inside function body), not top-level.
- Do not add any constants from MLclm_varcon.py into legoesm.constants.

### Phase 1: Scaffolding (no logic yet)
- [ ] Create src/legoesm/land/canopy/__init__.py
- [ ] Create src/legoesm/land/canopy/config.py (CanopyConfig, CLMMLCanopyConfig)
- [ ] Create src/legoesm/land/canopy/state.py (CanopyState)
- [ ] Extend MultiLayerLandConfig with canopy field (config.py)
- [ ] Extend MultiLayerLandState with canopy field (state.py)
- [ ] Extend LandSurfaceParams with LAI, SAI, htop, hbot + PARAM_BOUNDS
- [ ] Update src/legoesm/land/__init__.py exports
- [ ] Add [project.optional-dependencies] canopy group to pyproject.toml
- [ ] Run existing tests — all must still pass: `pytest tests/land/`

### Phase 2: Interface stub
- [ ] Create src/legoesm/land/canopy/clm_ml_interface.py with correct signature but raise NotImplementedError body
- [ ] Add dispatch block to multilayer_land.py (scheme="none" path unchanged, scheme="clm_ml" calls stub)
- [ ] Update init_multilayer_land_state() to initialize CanopyState when scheme="clm_ml"
- [ ] Run existing tests — all must still pass

### Phase 3: Interface implementation
- [ ] Implement SW partitioning (f_dir, f_VIS split) in clm_ml_interface.py
- [ ] Implement AtmToSurface → mlcanopy_type input mapping (full table from plan)
- [ ] Implement CLM-ML-JAX → CanopyFluxes output mapping (full table from plan)
- [ ] Implement column indexing (trivial 1:1 via jnp.arange(ncol))

### Phase 4: Tests
- [ ] Create tests/land/unit/test_canopy.py
- [ ] Test: energy balance closure |net_rad - SH - LH - G| < 1e-2 W/m²
- [ ] Test: differentiability — jax.grad of LH w.r.t. T_lowest runs without error
- [ ] Test: GPP > 0 under positive PAR
- [ ] Integration test: 24h run with scheme="clm_ml", check TileResponse finite + water budget closes
- [ ] Dispatch test: default scheme="none" still passes all existing tests/land/ tests
- [ ] Config validation test: CanopyConfig(scheme="invalid") raises ValueError

### Phase 5: Final checks
- [ ] `pip install ".[canopy]"` works cleanly
- [ ] `pip install .` (without canopy extra) still works and imports don't fail
- [ ] Run full test suite: `pytest tests/`


## requirements and guidelines:

If you are blocked on a step for more than 3 attempts, write a BLOCKED.md describing exactly what's failing and why, mark the step [BLOCKED], and continue with the next step.

When implementing clm_ml_interface.py, read the actual MLCanopyFluxes source code first and reconcile any differences with the plan before implementing.

Don't skip the dispatch test. The "existing tests still pass" check after Phases 1 and 2 is the most important guardrail — it's how you catch if Claude accidentally broke the scheme="none" path.

Commit and push often with clear concise messages.