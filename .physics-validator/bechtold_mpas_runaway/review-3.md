Not quite: retain the settled/experiment-gated split, but add one settled sub-defect to B. It does not require another factorial axis—cell 01 already includes it.

- **B3 — spectral collapse:** `ext_earth` is band-resolved, but the loader integrates altitude then averages the spectral-band axis into one scalar AOD ([external.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/forcing/external.py:493), [external.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/forcing/external.py:569)). The same gray layer OD is then added in every RRTMGP LW g-point ([two_stream.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:463)). This loses the source spectral dependence independently of the extinction→absorption and vertical-placement errors.

Everything else converges:

- A is confirmed; “~3×” is the uncapped closure-scale interpretation, not a universal exact ratio.
- B is conditionally live exactly as stated: the full-transient deck opens the external-forcing gate ([model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6157)), then supplies LW OD when active ([model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6483)). Its sign remains experiment-gated.
- F1 is a collection blocker: the removed keyword is instantiated at module import ([test_bechtold_column_conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_bechtold_column_conservation.py:47)); thus all three tests are unreachable.
- The leaf-only convection closure conclusion, F2/F3/F4, F7, CMT interpretation, and temperature-floor correction stand. The MPAS temperature floor is strictly lower-only ([primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:894)).
- The existing ledger is indeed dry-enthalpy-only and is not wired into the dedicated MPAS run path; it therefore cannot resolve the requested MPAS full-energy attribution.

So: same experiment-gated root-cause list; amend settled finding B to include the spectral-collapse defect. No further separate code finding beyond that.
