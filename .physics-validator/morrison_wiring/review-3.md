
codex
2. STILL-OPEN — direct construction rejects `1e-8`, but every JSON load rewrites it before validation, with no schema/version distinction. A fresh explicit JSON config is therefore silently migrated, not refused. [migration](packages/coupler/legoesm/driver/config.py:2932) [JSON loader](packages/coupler/legoesm/driver/config.py:2989) [strict bounds](packages/coupler/legoesm/driver/config.py:2145)

3. RESOLVED — `validate_strict` rejects any non-default Morrison scalar unless the scheme is Morrison, including `none`; normal driver setup invokes strict validation. [cross-field gate](packages/coupler/legoesm/driver/config.py:2156) [driver validation](packages/coupler/legoesm/driver/model_driver.py:835)

4. STILL-OPEN — the new cross-field check defines “touched” with exact inequality, while threading uses `isclose(..., rel_tol=1e-6)`. Thus a float32-noise value such as `1e-3*(1+4.7e-8)` is rejected on Kessler/none as an override, though the resolver treats it as default and applies nothing. This contradicts the claimed single tolerance. [exact comparison](packages/coupler/legoesm/driver/config.py:2149) [resolver tolerance](packages/coupler/legoesm/driver/physics_pipeline.py:3171)

6. RESOLVED — all five catalog ranges now lie inside the strict gates; notably rime tops out at `2.0` and aggregation starts at `1e-4`. [catalog](packages/ml/legoesm/tuning.py:723) [strict gates](packages/coupler/legoesm/driver/config.py:2137)

`ExperimentConfig._field_defaults` is the correct defaults source. A legacy `1e-8` + `microphysics="kessler"` load does not explode: migration first changes it to the `1e-3` class default, so strict validation sees it as untouched. [order](packages/coupler/legoesm/driver/config.py:2924) [default comparison](packages/coupler/legoesm/driver/config.py:2149)

New issue: item 4’s strict-vs-resolver tolerance mismatch is not covered by the added tests, which exercise the below-tolerance case only on Morrison. [test](tests/unit/test_homogeneous_ice_nucleation_wiring.py:356)

VERDICT: FIX-FIRST
tokens used
105,534
2. STILL-OPEN — direct construction rejects `1e-8`, but every JSON load rewrites it before validation, with no schema/version distinction. A fresh explicit JSON config is therefore silently migrated, not refused. [migration](packages/coupler/legoesm/driver/config.py:2932) [JSON loader](packages/coupler/legoesm/driver/config.py:2989) [strict bounds](packages/coupler/legoesm/driver/config.py:2145)

3. RESOLVED — `validate_strict` rejects any non-default Morrison scalar unless the scheme is Morrison, including `none`; normal driver setup invokes strict validation. [cross-field gate](packages/coupler/legoesm/driver/config.py:2156) [driver validation](packages/coupler/legoesm/driver/model_driver.py:835)

4. STILL-OPEN — the new cross-field check defines “touched” with exact inequality, while threading uses `isclose(..., rel_tol=1e-6)`. Thus a float32-noise value such as `1e-3*(1+4.7e-8)` is rejected on Kessler/none as an override, though the resolver treats it as default and applies nothing. This contradicts the claimed single tolerance. [exact comparison](packages/coupler/legoesm/driver/config.py:2149) [resolver tolerance](packages/coupler/legoesm/driver/physics_pipeline.py:3171)

6. RESOLVED — all five catalog ranges now lie inside the strict gates; notably rime tops out at `2.0` and aggregation starts at `1e-4`. [catalog](packages/ml/legoesm/tuning.py:723) [strict gates](packages/coupler/legoesm/driver/config.py:2137)

`ExperimentConfig._field_defaults` is the correct defaults source. A legacy `1e-8` + `microphysics="kessler"` load does not explode: migration first changes it to the `1e-3` class default, so strict validation sees it as untouched. [order](packages/coupler/legoesm/driver/config.py:2924) [default comparison](packages/coupler/legoesm/driver/config.py:2149)

New issue: item 4’s strict-vs-resolver tolerance mismatch is not covered by the added tests, which exercise the below-tolerance case only on Morrison. [test](tests/unit/test_homogeneous_ice_nucleation_wiring.py:356)

VERDICT: FIX-FIRST
