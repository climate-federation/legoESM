pyproject.toml:111:    "pytest-timeout",
pyproject.toml:304:[tool.pytest.ini_options]
.github/workflows/mpi-distributed.yml:52:              python -m pytest -q tests/distributed/test_halo_mpi.py
.github/workflows/mpi-distributed.yml:62:              python -m pytest -q tests/distributed/test_plane_pencil_mpi.py
.github/workflows/mpi-distributed.yml:75:              python -m pytest -q tests/distributed/test_latlon_2d_pad_wall_mpi.py
.github/workflows/mpi-distributed.yml:88:              python -m pytest -q tests/distributed/test_latlon_transpose_ad_mpi.py
.github/workflows/mpi-nightly.yml:47:            python -m pytest -q tests/distributed/test_ocean_mpi_conservation.py -k longrun
.github/workflows/ci.yml:84:        run: python -m pytest --collect-only tests/ 2>&1
.github/workflows/ci.yml:105:      # leaks across the pytest session for subsequently-imported modules.
.github/workflows/ci.yml:108:          python -m pytest tests/unit/ -x --timeout=300 \
.github/workflows/ci.yml:143:        # `bash -c` is required so the glob expands before pytest
.github/workflows/ci.yml:144:        # sees the arguments; otherwise pytest gets the literal
.github/workflows/ci.yml:155:          python -m pytest "${files[@]}" -x --timeout=300 \
.github/workflows/ci.yml:169:          python -m pytest tests/grids/test_fv3_native_grid_phase1.py \
.github/workflows/ci.yml:186:          python -m pytest tests/grids/test_fv3_native_swcore_phase4.py \
.github/workflows/ci.yml:202:          python -m pytest tests/unit/test_cmor_experiments_restart.py \
.github/workflows/ci.yml:218:          python -m pytest tests/atmosphere/ -x --timeout=600 \
/home/b/b381103/.local/bin/uv

 succeeded in 511ms:
?? .physics-validator/conservative_clamp/
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
eaf61e8d6 fix(mpas): close codex round-2 blockers — mass-fix ordering + f16 refusal
901b1d69f fix(mpas): close the codex FIX-FIRST findings on the conservative clamp
aab9be8dc docs(claude): attribution gates from the 2026-07 failure set
0b0c5ab6c Merge remote-tracking branch 'origin/main' into mpas-stability-campaign
fc7e7dce8 fix(mpas): stop the tracer clamp inventing water — column-conserving positivity borrow

codex
| Item | Status | Evidence |
|---|---|---|
| A — mass-fix reorder | **RESOLVED** | Serial fixes `p_s` before either floor; the intervening code reads/writes only `T` and tracers. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:969) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:992) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1015) MPI has the same ordering. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1105) `sfc_diag`/`phys_state_out` are finished before the fix, while `cast_pytree` is after both positions, so neither gains a new observation of corrected `p_s`. The serial fixer reads only `p_s` from old/new state and mesh area—not `T` or tracers—so the unchanged pre-step `state` baseline remains correct. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1084) Anchor semantics also do not change: `target_mass` is computed before `_step_jit`, and the fixer substitutes it for `state_old` exactly as before. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:790) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1089) Exact conservation remains subject to the helper’s pre-existing degenerate-column and storage-cast roundoff caveats. |
| B — f16 refusal | **RESOLVED** | `eps_eff > 1e-6` rejects f16, whose `sqrt(tiny)` is ~7.8e-3. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:165) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:170) The supported accepted policy dtypes are bf16/f32/f64; their dtype-derived effective threshold is ~1e-19 (bf16/f32) or `1e-30` (f64 default), not material at mixing-ratio scale. [precision.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/precision.py:50) A caller may explicitly choose `eps=1e-6`, allowing at most that tolerance; this is parameter-selected, not an accepted-dtype hole. |
| C — MPI `dsigma` mismatch | **RESOLVED** | The factory compares both `dsigma` arrays on the host and raises before setup collectives/JIT step creation. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:819) |

New non-blockers: the MPI factory docstring still describes the old `physics → floor → fix_mass` order. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:754) Also, the new error text calls `eps_eff` “sqrt(tiny)” even when caller-supplied `eps` dominates. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:172)

VERDICT: SHIP
tokens used
75,945
| Item | Status | Evidence |
|---|---|---|
| A — mass-fix reorder | **RESOLVED** | Serial fixes `p_s` before either floor; the intervening code reads/writes only `T` and tracers. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:969) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:992) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1015) MPI has the same ordering. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1105) `sfc_diag`/`phys_state_out` are finished before the fix, while `cast_pytree` is after both positions, so neither gains a new observation of corrected `p_s`. The serial fixer reads only `p_s` from old/new state and mesh area—not `T` or tracers—so the unchanged pre-step `state` baseline remains correct. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1084) Anchor semantics also do not change: `target_mass` is computed before `_step_jit`, and the fixer substitutes it for `state_old` exactly as before. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:790) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1089) Exact conservation remains subject to the helper’s pre-existing degenerate-column and storage-cast roundoff caveats. |
| B — f16 refusal | **RESOLVED** | `eps_eff > 1e-6` rejects f16, whose `sqrt(tiny)` is ~7.8e-3. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:165) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:170) The supported accepted policy dtypes are bf16/f32/f64; their dtype-derived effective threshold is ~1e-19 (bf16/f32) or `1e-30` (f64 default), not material at mixing-ratio scale. [precision.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/precision.py:50) A caller may explicitly choose `eps=1e-6`, allowing at most that tolerance; this is parameter-selected, not an accepted-dtype hole. |
| C — MPI `dsigma` mismatch | **RESOLVED** | The factory compares both `dsigma` arrays on the host and raises before setup collectives/JIT step creation. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:819) |

New non-blockers: the MPI factory docstring still describes the old `physics → floor → fix_mass` order. [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:754) Also, the new error text calls `eps_eff` “sqrt(tiny)” even when caller-supplied `eps` dominates. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:172)

VERDICT: SHIP
