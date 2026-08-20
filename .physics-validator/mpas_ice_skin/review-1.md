## Verdict

This is not ready for the scorecard. The helper’s interior algebra and the turbulent-flux sign are correct, but the driver integration has high-impact temporal, restart, and diagnostic defects.

## Confirmed bugs

1. **The skin integrates one instantaneous flux sample as a whole day, not a daily mean.**  
   [`model_driver.py:6682`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6682 ) reads the last retained `_sfc_diag` and advances with `dt_s=86400` at [6701–6705]. Radiation is explicitly diurnal ([radiation/integration.py:1027–1036](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1027)), and the exported SW/LW values are instantaneous (`down - up`) at [1192–1193](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1192). Held-radiation steps retain an older radiative value while turbulence is fresh, so the assembled flux can even be time-incoherent.

   At 2 m, `dT_new/dF = 0.042786 K/(W m⁻²)`. A 100 W m⁻² snapshot-vs-mean error moves the skin **4.28 K per “daily” update**. A half-sine SW flux with 200 W m⁻² noon peak has a 63.66 W m⁻² daily mean but a zero midnight sample: **2.72 K/day** skin error. This is a major seasonal/longitude artifact.

   Fix: accumulate `F_net * dt` every atmospheric step, retain the accumulator and elapsed seconds, then use the interval mean. Persist that pending accumulator in checkpoints.

2. **Restart continuity is broken; each restart drops the pending daily skin advance.**  
   Checkpoints are written after the last atmospheric step ([model_driver.py:6939–6960](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6939)), but the skin update for that completed day happens only at the *next* loop entry ([6674–6705](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6674)). The checkpoint stores only the pre-update `ice_T_skin` ([4264–4266](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:4264)), not `_sfc_diag` or a flux integral. On a fresh restarted model `_sfc_diag` is absent, so the first boundary skips and marks the day consumed.

   With constant −25 W m⁻², the omitted update is **1.070 K** at h=2 m (`271.35 → 270.280 K`). A 12-hour restart is worse: only the latter half-day’s diagnostics exist, yet they are applied for a full 86400 s at the next boundary.

   Fix: make the skin state advance at the end of the represented interval, or checkpoint the flux integral, elapsed time, and daily-bucket state. Add straight-vs-restarted bitwise tests with the feature enabled at integer and mid-day splits.

3. **CMOR `tas` still uses constant `cfg.T_ice`, not the prognostic skin.**  
   [`model_driver.py:5562–5565`](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5562) passes `config.T_ice` to the MOST diagnostic. `_tas_2m` then reconstructs the surface temperature from that value at [diagnostics.py:634–640](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:634). Thus the reported scorecard variable uses a 271.35 K ice surface even while the actual radiation and turbulence use the skin.

   This does not make `tas` wholly unchanged—the low model level responds—but it makes the 2 m extrapolation physically inconsistent and biases the exact metric this change targets.

   Fix: pass `self._ice_T_skin` when the feature is enabled; update the diagnostic type contract to accept an array.

4. **The melt cap uses the basal seawater freezing point as the surface melting point.**  
   The helper uses `T_freeze_ocean=271.35 K` both in the conductive basal term and in the upper clip ([surface_utils.py:354,403–405](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/forcing/surface_utils.py:354)). Those are not the same boundary condition. The repository’s prognostic sea-ice implementation explicitly uses `T_base = T_freeze_ocean` but clips the surface to `T_melt_surface` ([sea_ice.py:1062–1084](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/ice/legoesm/ice/sea_ice.py:1062)); `constants.T_freeze` is 273.15 K while seawater freezing is 271.35 K ([constants.py:55–60](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/constants.py:55)).

   This imposes a **1.8 K cold summer cap** and starts discarding energy too early. Fix: separate `T_base_K=271.35` from `T_melt_surface_K=273.15`; prescribed ice may still discard cap excess, but at the correct temperature.

5. **SIC transition is applied on the wrong side of the time interval.**  
   The flux is from the preceding interval, but the helper receives SIC at the new canonical day ([model_driver.py:6699–6705](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6699)). A cell going `sic: 0 → positive` evolves its newly created ice with a day of *open-water* flux. For prior open water and −25 W m⁻², it begins at 270.280 K rather than the documented 271.35 K seed.

   Fix: advance using previous-interval SIC, then apply the new SIC transition/reset.

6. **Save-after-load can silently strip a valid skin restart.**  
   Load clears `self._ice_T_skin` and stages the value only in `_carry_aux` ([4842–4845](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:4842)). Save only writes `self._ice_T_skin` ([4264–4266](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:4264)). Therefore load → save without running creates a plausible checkpoint missing `ice_T_skin`; unlike `physstate_*`, there is no refusal guard.

   The specific stale-prior-run concern is refuted **when `load_checkpoint()` is called**, because it clears the field. But a reused driver switched from feature-on to feature-off without a load also retains `_ice_T_skin`, and an off-feature save will write it. Gate save on the config and refuse a save while a skin is staged but unadopted.

7. **The claimed coupled “stable integration” is conditional, not guaranteed.**  
   The stated amplification is correct:
   \[
   a=(1-r\lambda)/(1+rg).
   \]
   But only conduction is implicit. At h=2 m, stability requires `lambda < 45.72 W m⁻² K⁻¹`; monotone convergence requires `<22.35`. At allowed h=0.1 m, the stability limit falls to about **22.6 W m⁻² K⁻¹**. Strong-wind sensible transfer plus LW can exceed this. The cap bounds divergence but produces an unphysical cap/floor oscillation; it is not “safe old behavior.”

   Fix: subcycle the skin/flux coupling, or use a local implicit/linearized surface-flux solve.

## Important limitations / minor contract defects

- **Partial SIC is not tile-resolved.** `F_net` is computed for the blended cell temperature, then treated as an ice-skin per-ice-area flux. That is a closure error at marginal ice, where an ice-tile LW/turbulent flux can differ sharply from the blended-cell flux. A global single-tile AMIP model cannot solve this exactly, but it should be documented and sensitivity-tested; it is most consequential during retreat/advance.

- `blend_surface_temperature` is now called with an array ice temperature, but advertises `T_ice: float` and documents a float ([surface_utils.py:16–30](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/forcing/surface_utils.py:16)). Runtime broadcasting is correct for `(nCells,)`; the public type/documentation contract is not.

- The helper is only differentiable in its interior. `clip` has zero derivatives at both caps and `where(sic > 0)` has zero flux gradient over open water and is non-differentiable at zero. That is acceptable for the physical cap/mask, but the “no dead gradient” conclusion is overbroad. End-to-end driver differentiation is unavailable anyway: `_sfc_diag` and `_ice_T_skin` are eager Python-side state, not a JAX carry.

## Refuted concerns

- **Flux sign and tuple slots are correct.** Radiation exports down-minus-up ([radiation/integration.py:1192–1205](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1192)); turbulence defines sensible and latent positive upward ([surface_layer.py:44–50](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py:44)). Therefore `sw + lw - sh - lh` is correct.

- **The helper’s equation, units, conduction sign, and interior Jacobian are correct.** At h=2 m, `C=1,931,202 J m⁻² K⁻¹`, `g=1.02 W m⁻² K⁻¹`, and the pure conductive timescale is 21.91 days.

- **The closure does not suffer JIT staleness.** It is evaluated eagerly and passed as dynamic `forcing["T_sfc"]`; mutations are outside the compiled MPAS step.

- **Fresh-run spin-up is expected, not a code bug.** Under constant −25 W m⁻² the residual pure-conduction warm error is 6.43 K after 30 days, 1.68 K after 60, and 0.44 K after 90. Month-3 onward is plausible for this idealized initialization, though it should be declared in experiment metadata.

- **No melt bookkeeping is acceptable for prescribed SIC/SST.** It is an open surface-energy reservoir by design. The incorrect 271.35 K cap is separate and is a real bug.

- **A global thickness is a documented model simplification, not a software failure.** It should be named explicitly as global/climatological; it cannot represent Antarctic thinner ice.

The current tests cover helper math, validation, and CLI only ([test_mpas_ice_skin.py:39–196](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_ice_skin.py:39)). They do not test any of the driver integration failures above.