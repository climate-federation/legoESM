- F1: fixed. Canonical MPAS configs fail fast otherwise, matching the sole hook site ([config.py:1907](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:1907), [model_driver.py:7001](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:7001)).

- F2: fixed. Morrison-only validation plus `q_i is None` liquid fallback are internally energy-consistent ([config.py:1922](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:1922), [model_driver.py:342](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:342)).

- F3: finding remains. The seed is mathematically consistent with `mi0`, but is uncapped: every drained ice mass creates new 10-µm crystals ([model_driver.py:381](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:381), [model_driver.py:7030](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:7030)). Morrison caps Cooper nucleation at `N_i_nuc_max=5e5 m^-3` ([morrison.py:338](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:338)); a default 5-K all-ice capped drain adds `8.46e8 kg^-1`, about `1.7e8 m^-3` at ρ=0.2—338× that ceiling. This directly perturbs deposition and sedimentation. Add a physically bounded seed policy and a cap-binding test.

- F4: accepted. The frozen `L_eff` solve and same-`w(T0)` routing are consistent with Morrison’s `L_s` deposition convention; the −700 J kg⁻¹ residual is pre-existing constants inconsistency, not newly introduced.

- F5: accepted as a documented operator-split edge. The temporary above-freezing `q_i` is subsequently subject to Morrison melting and its heat limiter ([morrison.py:663](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:663)).

- F6: fixed. The fp32 subprocess and T/p FD coverage are present ([test_hard_sat_ice_curve.py:125](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_hard_sat_ice_curve.py:125), [test_hard_sat_ice_curve.py:347](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_hard_sat_ice_curve.py:347)).

OVERALL VERDICT: findings remain: uncapped N_i seeding can exceed Morrison’s intended ice-number ceiling by >300× per default capped adjustment.
