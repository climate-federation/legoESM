  1208	    Phi = compute_geopotential(T, p_s, sigma_coord, phis)
  1209	    _ke_intg = 0.5 * (u**2 + v**2)
  1210	    _ie_intg = c_v * T
  1211	    _pe_intg = Phi
  1212	    _col_triple = jnp.sum(
  1213	        jnp.stack([_ke_intg, _ie_intg, _pe_intg], axis=-1)
  1214	        * mass_weight[..., None],
  1215	        axis=-2,
  1216	    )
  1217	    ke_col = _col_triple[..., 0]
  1218	    ie_col = _col_triple[..., 1]
  1219	    pe_col = _col_triple[..., 2]
  1220	
  1221	    ke, ie, pe = batch_global_area_sums([ke_col, ie_col, pe_col], grid)
  1222	
  1223	    total = ke + ie + pe
  1224	    return {
  1225	        'kinetic_energy': ke,
  1226	        'internal_energy': ie,
  1227	        'potential_energy': pe,
  1228	        'total_energy': total,
  1229	    }
  1230	
  1231	
  1232	def compute_nh_energy(
  1233	    state,
  1234	    grid: CubedSphereGrid,
  1235	    height_coord,

codex
The main serial selector is keyed correctly (`q_v`, `q_c`, `q_r`, and, for full microphysics, `q_i/q_s/q_g`), but the production MPI wrapper bypasses that selector entirely. I also found that the floor’s own `eps` path loses finite positive columns below the threshold, and its float32 reverse-mode claim is unsafe just above it because the quotient VJP squares a ~1e-30 denominator.
codex
1. **CONFIRMED DEFECT.** Conservation has more exceptions than documented net-negative columns.

   - Finite positive `q=[[5e-31]]`, `w=[1]` yields `after <= eps`, `factor=0`, and output zero—not the positive input integral. This contradicts the advertised invariant and monotone no-op claim. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:125) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:147)
   - `NaN` propagates through `maximum`; `+Inf/+Inf` makes `factor=NaN`. There is no finite-input contract.
   - The advertised generic `axis` API is broken for non-trailing axes: `q * w` broadcasts `w` on the trailing dimension before reducing `axis`. E.g. `q.shape==(2,3)`, `axis=0`, `w.shape==(2,)` errors. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:136)
   - Tests only exercise `axis=-1`, large f64 values, and describe net-negative as the sole limitation. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:94)

2. **REFUTED for real MPAS weights.** With positive weights,
   `after - before = sum(max(-q,0) * w) >= 0`; hence a positive-integral column has `0 <= before/after <= 1`. The factor cannot legitimately exceed one, so clipping it to one does not under-restore MPAS columns.

   MPAS uses positive `dsigma`: pure-sigma is an increasing-interface difference, and hybrid exposes `dA+dB`, whose construction is monotone. [vertical.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:150) [vertical.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:745)

   For arbitrary negative weights, the generic helper can indeed get `before > after` and silently fail conservation; that is outside valid MPAS use but should be documented.

3. **CONFIRMED DEFECT.** The reverse-mode “finite everywhere” claim is false in float32 near the active threshold. `1e-30` itself does not underflow float32, but its square does.

   For finite float32 `q=[2e-30, -5e-31]`, `w=[1,1]`: `after=2e-30 > eps`, `before=1.5e-30`, factor=`0.75`. The VJP of `before / after` contains `/ after**2`; `after**2≈4e-60` underflows to zero, creating the same `0/0` / `inf` VJP hazard the repository’s AD-safe-divide documentation explicitly warns about. The inner `where` only protects `after <= eps`; it leaves this active branch unchanged. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:147) [\_shared.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/_shared.py:35)

   The test forces x64 globally and has no float32, near-threshold gradient case. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:18) [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:138)

4. **REFUTED.** Rescaling a water species while leaving `T` unchanged is not, by itself, an energy-conservation defect. This operation redistributes and conserves that species’ weighted mass, so its column-integrated latent term is also conserved. A temperature correction would be required for the older case where a vapour sink was truncated after its latent heating was already applied—not for a same-species conservative redistribution. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:43) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:136)

5. **CONFIRMED DEFECT.** The subsequent surface-pressure mass fix breaks end-of-step column-water conservation whenever its correction is nonzero.

   Let `B=sum(q*dsigma)`. The clip preserves `B`, but then `_fix_mass_mpas_hydro` applies `p_s_final = p_s + c`; therefore the diagnosed water becomes `W_final=(p_s+c)B/g`, changing by `cB/g`. The production diagnostic uses exactly this `p_s*dsigma/g` formula. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1008) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1022) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1100) [column_integrals.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/column_integrals.py:14)

   This is normally active: both driver-level mass-fixer defaults are true. [config.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:127)

6. **CONFIRMED DEFECT for MPI reachability; serial key mismatch REFUTED.** Serial MPAS keys are correct: registries create `q_v/q_c/q_r` and full-microphysics adds `q_i/q_s/q_g/N_c/N_r/N_i`; the MPAS state builder preserves those exact keys. [tracers.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/tracers.py:65) [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:1539) Checkpoints use `trc_` only for array names and restore unprefixed dictionary keys. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5025)

   But distributed MPAS does not execute `_step_jit`: the driver dispatches to `_mpi_step`, whose separate floors block still unconditionally uses `jnp.maximum` and never reads `conservative_tracer_clamp`. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:7286) [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1098) The test merely inspects `_step_jit`, so it ratchets the wrong execution path for MPI. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:174)

**VERDICT: FIX-FIRST** — blocking issues are MPI silently ignoring the flag, loss of end-of-step water conservation after `p_s` repair, and the false float32 reverse-mode safety claim. Add direct float32 threshold tests and a behavioral MPI-path test; either narrow the helper to `axis=-1` or implement correct axis-aware weight broadcasting.
tokens used
256,851
1. **CONFIRMED DEFECT.** Conservation has more exceptions than documented net-negative columns.

   - Finite positive `q=[[5e-31]]`, `w=[1]` yields `after <= eps`, `factor=0`, and output zero—not the positive input integral. This contradicts the advertised invariant and monotone no-op claim. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:125) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:147)
   - `NaN` propagates through `maximum`; `+Inf/+Inf` makes `factor=NaN`. There is no finite-input contract.
   - The advertised generic `axis` API is broken for non-trailing axes: `q * w` broadcasts `w` on the trailing dimension before reducing `axis`. E.g. `q.shape==(2,3)`, `axis=0`, `w.shape==(2,)` errors. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:136)
   - Tests only exercise `axis=-1`, large f64 values, and describe net-negative as the sole limitation. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:94)

2. **REFUTED for real MPAS weights.** With positive weights,
   `after - before = sum(max(-q,0) * w) >= 0`; hence a positive-integral column has `0 <= before/after <= 1`. The factor cannot legitimately exceed one, so clipping it to one does not under-restore MPAS columns.

   MPAS uses positive `dsigma`: pure-sigma is an increasing-interface difference, and hybrid exposes `dA+dB`, whose construction is monotone. [vertical.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:150) [vertical.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/grids/vertical.py:745)

   For arbitrary negative weights, the generic helper can indeed get `before > after` and silently fail conservation; that is outside valid MPAS use but should be documented.

3. **CONFIRMED DEFECT.** The reverse-mode “finite everywhere” claim is false in float32 near the active threshold. `1e-30` itself does not underflow float32, but its square does.

   For finite float32 `q=[2e-30, -5e-31]`, `w=[1,1]`: `after=2e-30 > eps`, `before=1.5e-30`, factor=`0.75`. The VJP of `before / after` contains `/ after**2`; `after**2≈4e-60` underflows to zero, creating the same `0/0` / `inf` VJP hazard the repository’s AD-safe-divide documentation explicitly warns about. The inner `where` only protects `after <= eps`; it leaves this active branch unchanged. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:147) [\_shared.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/_shared.py:35)

   The test forces x64 globally and has no float32, near-threshold gradient case. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:18) [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:138)

4. **REFUTED.** Rescaling a water species while leaving `T` unchanged is not, by itself, an energy-conservation defect. This operation redistributes and conserves that species’ weighted mass, so its column-integrated latent term is also conserved. A temperature correction would be required for the older case where a vapour sink was truncated after its latent heating was already applied—not for a same-species conservative redistribution. [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:43) [conservation.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/conservation.py:136)

5. **CONFIRMED DEFECT.** The subsequent surface-pressure mass fix breaks end-of-step column-water conservation whenever its correction is nonzero.

   Let `B=sum(q*dsigma)`. The clip preserves `B`, but then `_fix_mass_mpas_hydro` applies `p_s_final = p_s + c`; therefore the diagnosed water becomes `W_final=(p_s+c)B/g`, changing by `cB/g`. The production diagnostic uses exactly this `p_s*dsigma/g` formula. [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1008) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1022) [primitive_eq_mpas.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1100) [column_integrals.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/column_integrals.py:14)

   This is normally active: both driver-level mass-fixer defaults are true. [config.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/config.py:127)

6. **CONFIRMED DEFECT for MPI reachability; serial key mismatch REFUTED.** Serial MPAS keys are correct: registries create `q_v/q_c/q_r` and full-microphysics adds `q_i/q_s/q_g/N_c/N_r/N_i`; the MPAS state builder preserves those exact keys. [tracers.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/core/tracers.py:65) [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:1539) Checkpoints use `trc_` only for array names and restore unprefixed dictionary keys. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5025)

   But distributed MPAS does not execute `_step_jit`: the driver dispatches to `_mpi_step`, whose separate floors block still unconditionally uses `jnp.maximum` and never reads `conservative_tracer_clamp`. [model_driver.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:7286) [voronoi_mpi.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/voronoi_mpi.py:1098) The test merely inspects `_step_jit`, so it ratchets the wrong execution path for MPI. [test_conservative_positive_clip.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_conservative_positive_clip.py:174)

**VERDICT: FIX-FIRST** — blocking issues are MPI silently ignoring the flag, loss of end-of-step water conservation after `p_s` repair, and the false float32 reverse-mode safety claim. Add direct float32 threshold tests and a behavioral MPI-path test; either narrow the helper to `axis=-1` or implement correct axis-aware weight broadcasting.
