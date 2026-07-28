## Verdict

Do not merge as-is. The operator is conservative with valid fixed positive `dp`, but the driver does not enforce the conditions needed for its positivity/CFL claims.

- **BLOCKER — the CFL guard is not the CFL of the applied operator.** The true row sum is

  \[
  S_{c,k}=\sum_e \frac{dv_e}{A_cdc_e}\frac{dp_{e,k}}{dp_{c,k}},
  \quad dp_e=\tfrac12(dp_c+dp_n).
  \]

  The driver guards only \(g_{\max}=\max_c\sum_e dv_e/(A_cdc_e)\) at 0.5, omitting the dynamic, level-dependent \(dp_e/dp_c\) factor. [model_driver.py:6053](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6053)-[6067](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6067)

  The claimed x2 headroom only works if every \(dp_n/dp_c\le3\), since \(dp_e/dp_c=\frac12(1+dp_n/dp_c)\le2\). No such invariant is checked or implied by config validation. A low-pressure cell surrounded by cells with \(dp_n/dp_c=4\) has \(S=2.5g\); an accepted \(\nu dt g_{\max}=0.5\) gives diagonal coefficient \(1-1.25<0\), producing negative `q_v`.

- **BLOCKER — default hybrid coordinates make this worse and can make `dp <= 0` at plausible surface pressures.** MPAS defaults to `vertical_coord="hybrid"` and constructs `make_hybrid_levels`. [config.py:61](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:61)-[66](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:66), [model_driver.py:884](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:884)-[891](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:891)

  Here \(A=\eta-B+\epsilon(1-\eta)\), \(B=\eta^3\), so

  \[
  dp_k=p_{\rm ref}\Delta A_k+p_s\Delta B_k.
  \]

  Near the surface, \(\Delta B/\Delta\eta\approx3\), so the layer becomes non-positive below roughly \(p_s\approx2p_{\rm ref}/3\), about 67 kPa. [vertical.py:730](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:730)-[778](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:778), [vertical.py:885](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:885)-[904](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:904)

  Thus `dp_edge/dp_cell` can become arbitrarily large while both surface pressures remain finite, and `div / dp_cell_3d` can yield Inf/NaN. [operators_voronoi.py:1283](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/operators_voronoi.py:1283)-[1287](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/operators_voronoi.py:1287)

  The RHS clips a local `p_s` copy, but the returned state is not pressure-floored; the smoother consumes the raw post-step state. [primitive_eq_mpas.py:223](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:223)-[229](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:229), [primitive_eq_mpas.py:869](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:869)-[915](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:869)-[915), [model_driver.py:6622](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6622)-[6626](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6626)

  A negative value then survives the smoother/drain and is clipped on a later MPAS step, destroying the claimed water conservation. [primitive_eq_mpas.py:898](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:898)-[906](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:898)-[906)

- **The static analysis incorrectly treats all coordinates as pure sigma.** `pressure_at_half(p_s)` is \(p_s\sigma\) only for `SigmaCoordinate`; hybrid uses \(A p_{\rm ref}+B p_s\). [vertical.py:91](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:91)-[98](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:91)-[98), [vertical.py:587](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:587)-[594](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:587)-[594)

  The smoothing uses true hybrid `dp`, while the hard-saturation post-step explicitly uses pure-sigma pressure. [model_driver.py:6622](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6622)-[6625](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6622)-[6625), [model_driver.py:241](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:241)-[294](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:241)-[294). That does not by itself break transfer of `q_v` to `q_c`, but the “same pressure convention” justification is false.

What does hold, conditionally:

- **Conservation:** for finite nonzero fixed `dp`, yes. During just this substep `p_s` is calculated once and not changed, so
  \[
  \sum_{c} A_cdp_c\,lap_c=\sum_e(\mathrm{sign}_{c1}+\mathrm{sign}_{c2})F_e=0.
  \]
  The omitted \(1/g\) is a harmless global factor. This is only a substep result, not a full-model-step invariant.

- **Sign and constants:** correct for positive `dp`. The mesh defines `+` as outward from `cellsOnEdge[0]`, while the gradient is `q[c2]-q[c1]`; a maximum therefore has negative Laplacian. [voronoi.py:874](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/voronoi.py:874)-[888](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/voronoi.py:874)-[888), [operators_voronoi.py:520](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/operators_voronoi.py:520)-[534](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/operators_voronoi.py:520)-[534). `lap(const)=0` also holds unless `dp=0`, where weighted form is `0/0`.

- **Padding, scope, and ordinary shapes:** correct for finite arrays. `edgesOnCell` masking is implemented; `np` is imported; the local import is in scope; `p_s: (nCells,)` produces `dp: (nCells,nlev)`; copying the tracer dict and `Field.replace` are valid. [operators_voronoi.py:507](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/operators_voronoi.py:507)-[517](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/operators_voronoi.py:507)-[517), [model_driver.py:16](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:16)-[18](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:16)-[18)

- **No-op/JIT:** `0.0` skips the new block completely, so runtime numerics are unchanged. This eager post-step does not cause retracing. [model_driver.py:6037](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6037)-[6039](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6037)-[6039), [model_driver.py:6619](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6619)-[6630](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6619)-[6630)

- **Strict validation and CLI:** correct for ordinary configurations: cd-grid is refused, MPAS is accepted, bounds are finite `[0,1e8]`, and the argument is wired through. [config.py:1757](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:1757)-[1835](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:1757)-[1835), [run_amip.py:1248](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/run/run_amip.py:1248)-[1255](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/run/run_amip.py:1248)-[1255), [run_amip.py:1731](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/run/run_amip.py:1731)-[1734](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/run/run_amip.py:1731)-[1734)

Test weaknesses:

- The positivity test samples only 3–7 kPa `dp`, so \(dp_n/dp_c<2.34\); it cannot test the missing-ratio failure or hybrid `dp`. [test_mpas_qv_smoothing.py:42](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:42)-[46](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:42)-[46), [94](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:94)-[108](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:94)-[108)

- The test does not exercise the driver guard; it manually duplicates it. Thus driver and test can drift together. [test_mpas_qv_smoothing.py:49](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:49)-[57](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:49)-[57)

- Conservation reductions sum all levels, allowing equal-and-opposite vertical errors to cancel. They should assert residual per level. [test_mpas_qv_smoothing.py:72](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:72)-[86](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:72)-[86)

- Tests force x64 globally, so neither the stated `1e-12`/`1e-13` conservation behavior nor the cast is tested in production fp32. [test_mpas_qv_smoothing.py:22](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:22)-[24](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_qv_smoothing.py:22)-[24) The global setting is not restored.

- `mixed_fp64_storage` has fp64 state storage but fp32 compute policy; this eager smoother computes in storage dtype and bypasses that policy. It is a performance/precision-policy inconsistency, not a sign error. [precision.py:136](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/precision.py:136)-[149](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/precision.py:136)-[149), [model_driver.py:6621](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6621)-[6627](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6621)-[6627)

One compatibility risk: this field was inserted mid-`ExperimentConfig`, changing positional `NamedTuple` ABI for every later field. No in-repo positional construction was found, but this is still a public-config break. [config.py:1100](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:1100)-[1111](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:1100)-[1111)

The required fix is to reject non-positive `dp` and guard the actual per-step `max_{c,k} S_{c,k}` (or subcycle/limit the update). A setup-only geometry guard cannot establish a DMP for an evolving pressure-weighted operator.
