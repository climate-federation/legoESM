# Preregistration — DINO FE faithful-default stability bisection

**Frozen before the controlled arms.**  This lane dispositions the regression
discovered by transfer experiment T2: the `nemo_dino_kamm` forward-Euler (FE)
card reaches an invalid raw `e3w_int` at outer step 36 on the developed-state
twin, whereas PR #1696 promoted the literal fidelity selectors on both DINO
cards after climate-running only `nemo_dino_kamm_mlf`.

This is a run-admission repair, not an FE fidelity claim.  The NEMO oracle is
MLF, and no literal selector is called an FE-faithful equivalent without a
separate oracle-grounded preregistration.

## Frozen instrument and admission

Use the committed CPU-only `fe_stability_repro.py` with `JAX_DISABLE_JIT=1`,
fp64, ladder `both`, the same `RUN_TRAJ`/`RUN_STEPDUMP` developed-state bridge,
surface forcing, seasonal clock, and production `model.step` path as T2.  Every
arm records its recipe overrides and first exception.  Unknown, malformed, or
duplicate overrides hard-fail.

The discriminating window is outer steps 1–40: the unmodified promoted card
must reproduce the registered step-36 checked error, while an admitted rescue
must complete all 40 steps with finite T/S/SSH/U/V.  The final scoped default
must then complete 64 steps under the same gate.  These windows prove removal
of the observed regression only; they do not establish climate stability.

## Frozen pre-sweep control and groups

The rescue control restores the selector evaluation choices that predated the
#1696 sweep while retaining unrelated standing card choices, including FE
`barotropic_diffusion_alpha=0.01`.  The restoration groups are:

| group | fields restored in the FE control |
|---|---|
| `BARO` | `barotropic_continuity_evaluation=generic`; `barotropic_een_coefficient_evaluation=generic`; `barotropic_pgf_evaluation=generic`; `barotropic_seed_evaluation=generic`; `barotropic_transport_accumulation_evaluation=generic` |
| `TKE_ZDF` | `tke_preclosure_coeff_source=current_subiteration`; `tke_matrix_evaluation=factored`; `tke_solver_evaluation=shared_thomas`; `tke_etau_exponential_evaluation=jax_expression`; `tke_htau_evaluation=jax_expression`; `tke_mxl_raw_evaluation=factored`; `tke_langmuir_evaluation=vectorized`; `tke_shear_evaluation_stage=implicit_solve_state`; `tke_shear_metric_source=tpoint_jacobian`; `tke_n2_evaluation_stage=implicit_solve_state`; `zdf_implicit_solver_evaluation=shared_thomas` |
| `REDI` | `gm_redi_slope_n2_evaluation=recompute`; `gm_redi_slope_prd_geometry_stage=current_step`; `gm_redi_slope_prd_evaluation=density_roundtrip`; `gm_redi_slope_metric_evaluation=division`; `gm_redi_slope_face_thickness_evaluation=static_face`; `gm_redi_flux_face_thickness_evaluation=tpoint_jacobian`; `gm_redi_horizontal_evaluation=cosine_scaled`; `gm_redi_vertical_skew_evaluation=normalized_sums`; `gm_redi_a33_evaluation=normalized_square`; `gm_redi_w_slope_stage_evaluation=redi_tuple`; `gm_redi_slope_depth_evaluation=legacy_jacobian_t_surface`; `gm_treguier_vertical_reduction_evaluation=tree`; `gm_treguier_sqrt_evaluation=guarded_floor` |
| `FORCING_METRIC` | `dino_wind_profile_evaluation=factored_smoothstep`; `vface_zonal_metric_evaluation=legacy_tracer_midpoint` |

First run `ALL_RESTORED`.  If it does not pass 40 steps, the sweep selectors
are not sufficient to explain the registered failure and this lane stops
without changing defaults.  If it passes, activate each group in turn on top
of the restored control.  A group is causal only if its activation fails by
step 40 and restoring that group rescues the otherwise-current card.  Bisect a
causal group field-by-field using the same paired activation/restoration rule;
if an interaction rather than an individual field owns the failure, scope the
minimal interacting set and record that fact.

## Frozen disposition

Only the causally admitted selector or minimal interacting set may be removed
from FE defaults.  The MLF recipe remains unchanged.  Each removed literal
selector is registered as `FE_EQUIVALENT_FUTURE_WORK`; FE falls back to its
pre-sweep implementation for that operation.  A non-vacuous regression test
must fail under the formerly promoted FE default and pass after scoping, and
the resolved recipe test must prove the literal value remains selected by
`nemo_dino_kamm_mlf`.

The strongest allowed result is:

`FE_PRE_SWEEP_RUNNABILITY_RESTORED_IN_64_STEP_DEVELOPED_STATE_WINDOW`

It is forbidden to print `FE_FAITHFUL`, `FE_STABLE_CLIMATE`, or a T2 fidelity
verdict from this lane.
