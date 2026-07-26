packages/coupler/legoesm/driver/physics_pipeline.py
packages/coupler/legoesm/driver/run_config_yaml.py
packages/ml/legoesm/tuning.py
tests/unit/_params_reachability_baseline.py
tests/unit/golden/yaml_to_experiment/williamson_test2.json
tests/unit/golden/yaml_to_experiment/williamson_test5.json
tests/unit/test_flag_reachability_audit.py
tests/unit/test_homogeneous_ice_nucleation_wiring.py
tests/unit/golden/yaml_to_experiment/williamson_test2.json
tests/unit/golden/yaml_to_experiment/williamson_test5.json

codex
1. RESOLVED — default Kessler/Thompson/P3 is byte-identical: `_touched` is empty and the resolver returns the original leaf object unchanged. Manual resolver check confirmed object identity for all three. [physics_pipeline.py:3150](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3150) [physics_pipeline.py:3281](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3281)

2. STILL-OPEN — migration equality is exact, and goldens are correctly updated, but the policy is a trap: a fresh explicit `1e-8` is silently migrated to `1e-3` before strict validation, so it is not refused. The warning’s “Set it explicitly to suppress” is impossible for `1e-8`. Version-gate legacy migration or reject it explicitly. [config.py:2918](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2918) [config.py:2136](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2136) [williamson_test2.json:158](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/golden/yaml_to_experiment/williamson_test2.json:158)

3. STILL-OPEN — MPAS ordering does not clobber hard-sat fields: the second replacement changes only Morrison leaves. But it is inside `if _msub is not None`; `microphysics="none"` has no subconfig, so touched Morrison scalars bypass the hard gate. FV has the same hole via its early return. `ExperimentConfig(microphysics="none", morrison_dep_coeff=3e-4).validate_strict()` passes and FV resolves `(None, None)`. [model_driver.py:6083](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6083) [model_driver.py:6117](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6117) [physics_pipeline.py:3189](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3189)

4. STILL-OPEN — `rel_tol=1e-6` silently drops deliberate valid retunes. For example, requested `dep_coeff=0.0010000005` is marked touched, but resolves as `0.001` because `isclose` prevents `_replace`. Float32 noise warrants a much tighter tolerance; user-specified calibration values should not disappear. [physics_pipeline.py:3162](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3162) [config.py:1123](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1123)

5. RESOLVED — the five `--params` mappings exist, their reachability-baseline entries are removed, and strict validation is finite plus spec-bounded for all five. [run_config_yaml.py:190](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/run_config_yaml.py:190) [_params_reachability_baseline.py:265](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/_params_reachability_baseline.py:265) [config.py:2136](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2136)

6. STILL-OPEN — newly exposed tuning-guide inconsistency: it advertises `rime_coeff` through `5.0` and `agg_coeff` down to `1e-5`, while strict/spec bounds are `[0,2]` and `[1e-4,1e-2]`. Now that these knobs are live, advertised tuning proposals can be rejected. [tuning.py:723](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/ml/legoesm/tuning.py:723) [tuning.py:760](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/ml/legoesm/tuning.py:760) [config.py:2138](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2138)

VERDICT: FIX-FIRST
tokens used
134,269
1. RESOLVED — default Kessler/Thompson/P3 is byte-identical: `_touched` is empty and the resolver returns the original leaf object unchanged. Manual resolver check confirmed object identity for all three. [physics_pipeline.py:3150](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3150) [physics_pipeline.py:3281](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3281)

2. STILL-OPEN — migration equality is exact, and goldens are correctly updated, but the policy is a trap: a fresh explicit `1e-8` is silently migrated to `1e-3` before strict validation, so it is not refused. The warning’s “Set it explicitly to suppress” is impossible for `1e-8`. Version-gate legacy migration or reject it explicitly. [config.py:2918](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2918) [config.py:2136](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2136) [williamson_test2.json:158](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/golden/yaml_to_experiment/williamson_test2.json:158)

3. STILL-OPEN — MPAS ordering does not clobber hard-sat fields: the second replacement changes only Morrison leaves. But it is inside `if _msub is not None`; `microphysics="none"` has no subconfig, so touched Morrison scalars bypass the hard gate. FV has the same hole via its early return. `ExperimentConfig(microphysics="none", morrison_dep_coeff=3e-4).validate_strict()` passes and FV resolves `(None, None)`. [model_driver.py:6083](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6083) [model_driver.py:6117](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6117) [physics_pipeline.py:3189](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3189)

4. STILL-OPEN — `rel_tol=1e-6` silently drops deliberate valid retunes. For example, requested `dep_coeff=0.0010000005` is marked touched, but resolves as `0.001` because `isclose` prevents `_replace`. Float32 noise warrants a much tighter tolerance; user-specified calibration values should not disappear. [physics_pipeline.py:3162](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/physics_pipeline.py:3162) [config.py:1123](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1123)

5. RESOLVED — the five `--params` mappings exist, their reachability-baseline entries are removed, and strict validation is finite plus spec-bounded for all five. [run_config_yaml.py:190](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/run_config_yaml.py:190) [_params_reachability_baseline.py:265](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/_params_reachability_baseline.py:265) [config.py:2136](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2136)

6. STILL-OPEN — newly exposed tuning-guide inconsistency: it advertises `rime_coeff` through `5.0` and `agg_coeff` down to `1e-5`, while strict/spec bounds are `[0,2]` and `[1e-4,1e-2]`. Now that these knobs are live, advertised tuning proposals can be rejected. [tuning.py:723](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/ml/legoesm/tuning.py:723) [tuning.py:760](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/ml/legoesm/tuning.py:760) [config.py:2138](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:2138)

VERDICT: FIX-FIRST
