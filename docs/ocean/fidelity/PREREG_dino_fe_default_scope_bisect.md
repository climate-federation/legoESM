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

## User-authorized continuation: perpetual-Euler split coupling

Frozen 2026-08-30 before adding operand dumps or changing the FE step.  The
original selector bisection stopped correctly: `ALL_RESTORED` and the clean
pre-#1696 control both fail, so #1696 did not cause the instability.  The user
has now explicitly opened the pre-existing FE defect as a work item.  This
continuation does not reuse the refuted selector premise.

### Executed oracle branch

The reference is the built DINO override, not generic NEMO prose:

- `RUN_TRAJ/namelist_cfg:94,101` selects `nn_it000=1` and a from-rest start;
  `:351-353` selects split-explicit, `ln_bt_fw=.false.`, with the in-file
  warning that the forward barotropic branch crashes.
- `src/OCE/DOM/istate.F90:107-110,121-137` sets `l_1st_euler=.true.`, starts
  velocity from rest, and makes `Kbb == Kmm`.
- `cfgs/DINO/MY_SRC/stpmlf.F90:134-137` uses `rDt=rn_Dt` on that first step;
  `:332` calls `dyn_spg`; `:578` calls `mlf_baro_corr` without excluding the
  Euler step; only `:685-688` clears `l_1st_euler` and restores `2*rn_Dt`.
- `cfgs/DINO/MY_SRC/dynspg_ts.F90:241-268` gives the first step forward
  averaging weights and resets them to centred weights on step two;
  `:561-579` still takes the `ln_bt_fw=.false.` seed branch (the initial
  `Kbb==Kmm` makes the first seed degenerate); `:1123-1127` couples the
  barotropic increment back into the 3-D RHS.

Therefore NEMO provides exactly one Euler outer step, and that step still runs
the after-vertical-solve barotropic corrector.  It provides no perpetual-Euler
oracle trajectory.  A perpetual-Euler legoESM card can at most repeat the
executed Euler-step composition as a stabilized approximation; it cannot be
called NEMO-faithful.

### Frozen diagnosis

Extend the committed CPU reproducer, without duplicating model numerics, to
wrap the production barotropic call and record outer steps 25--35.  For each
step record maxima and locations for the step-entry and barotropic-output
`eta/u/v`, the frozen `F_slow_eta/u/v`, and the column-mean velocity immediately
before and after the implicit vertical solve.  Record the after-solve
column-mean deposit and its work proxy `max_abs(after_mean-target_mean)`.  A
planted one-cell perturbation of the captured target must make the deposit
detector fire.

The baseline is the current `nemo_dino_kamm` card and must reproduce failure in
steps 32--37.  The sole causal arm repeats NEMO's executed first-Euler-step
composition on every perpetual-FE step: apply the existing shared
`mlf_baro_corr` kernel after the implicit vertical solve, targeting the
barotropic solver's primary velocity average.  No coefficient, substep count,
forcing, bridge, ladder, or other selector changes.  This arm confirms the
missing-corrector mechanism only if:

1. baseline growth is already present in the after-solve column-mean deposit;
2. the arm removes that deposit at the same recorded steps;
3. the arm completes 40 steps finite while baseline fails; and
4. reverting only the call site makes the new N-step regression test fail.

If any condition fails, do not ship the corrector as an FE stabilization.
Instead record the observed leading operand and open a new one-variable arm
before another physics edit.

### Frozen disposition and bars

On confirmation, the `nemo_dino_kamm` recipe selects the same existing
`nemo_mlf_baro_corr` policy and the forward-Euler dispatch executes it every
step at NEMO's source position.  The MLF card remains unchanged.  The committed
regression gate runs 40 developed-state CPU steps in fp64/no-JIT and requires
all prognostic arrays finite; its non-vacuity control disables only the FE
corrector and must reproduce the registered failure by step 37.  The final
card then runs 64 steps under the existing admission gate.

The strongest allowed statement is
`PERPETUAL_FE_EULER_COMPOSITION_RUNS_64_STEP_DEVELOPED_STATE_WINDOW`.  It is
still forbidden to print `FE_FAITHFUL`, `FE_STABLE_CLIMATE`, or a T2 science
verdict.  At this preregistration point, T2 climate arms remain withdrawn
pending this admission and a later GPU rerun.  T1 GPU work is outside this
round and must not be inspected or modified.

### Corrector arm result and frozen filter discrimination

The corrector arm is refuted before any default change.  The baseline fails at
step 36; repeating `mlf_baro_corr` suppresses the measured vertical-solve
deposit but still reaches hundreds of metres of SSH and fails at step 37 with
the same raw-`e3w_int` check.  It therefore misses conditions 2--3 above and
must not be promoted as the FE stabilization.

The remaining source difference inside NEMO's *executed first Euler step* is
now frozen as the next one-variable arm.  `dynspg_ts.F90:202-208` sets
`ll_init=ll_bt_av=.true.` for DINO's boxcar filter; `:241-251` gives the
first-Euler window its forward centre; `:546-553` resets the AB3/AM4 histories;
`:766-780` executes `ts_bck_interp` before every surface-pressure-gradient
evaluation.  Thus NEMO's Euler step is not legoESM's current plain
forward-backward `nemo_boxcar_centred` loop.  It includes the AB3 velocity
predictor and the four-level SSH interpolation, with the startup ramp reset for
that window.  legoESM already implements exactly that reset/window combination
as `nemo_boxcar_ab3`; the current construction guard that calls its
forward-Euler use a non-NEMO hybrid is contradicted by the executed source.

Before the arm, add diagnostic-only callbacks to the existing probe (no copied
numerics) and record, over outer steps 25--35, the per-window maxima and
locations of the live continuity divergence, interpolated SSH pressure
gradient, EEN Coriolis tendency, and frozen slow forcing.  The callbacks must
have a planted nonzero control.  The causal arm changes exactly one resolved
field:

| field | baseline | Euler-filter arm |
|---|---|---|
| `barotropic_time_filter` | `nemo_boxcar_centred` | `nemo_boxcar_ab3` |

The after-level corrector stays `off`, alpha stays `0.01`, and every other
resolved field, bridge, forcing, ladder, timestep, and substep count is held.
The arm confirms the missing-AB3/AM4 mechanism only if baseline growth is
owned by the live continuity/PGF fast pair while slow forcing and Coriolis are
subleading through onset, and the arm completes 40 steps finite.  On
confirmation, promote only this filter on `nemo_dino_kamm`, retain the explicit
scope statement that perpetual Euler has no NEMO trajectory analogue, show the
plain-filter plant fail by step 37, and require 64 finite steps on the promoted
card.  Otherwise stop without another stabilization edit.

### Registered result (post-measurement binding)

The one-variable arm **CONFIRMED the admission mechanism**.  This is an
engineering admission result, not an oracle-fidelity or climate result.

- The plain-filter control fails at outer step 36.  At step 25 its live
  pressure-gradient tendency is `0.90269 m s-2`, versus frozen slow forcing
  `3.3084e-5 m s-2` and EEN Coriolis `2.308e-4 m s-2`; its continuity
  divergence is `22.1698 m s-1`.  By step 29 the live pressure-gradient
  tendency is `3.478 m s-2` and SSH is `684.48 m`.  The continuity/surface-PGF
  fast pair therefore owns the onset; slow forcing and Coriolis are subleading.
- Repeating `mlf_baro_corr` is refuted: that arm still fails at step 37.  Its
  experimental FE wiring was removed.
- Changing only `barotropic_time_filter` to `nemo_boxcar_ab3` completes 40
  steps finite.  The shipped `nemo_dino_kamm` card then completes the frozen
  64-step gate with no override; at step 64, `|eta|max=0.852169 m`,
  `|u|max=0.898959 m s-1`, and `|v|max=1.092863 m s-1`.

The admitted default is therefore `nemo_boxcar_ab3` on the perpetual-FE card.
The result authorizes re-running T2; it does not authorize `FE_FAITHFUL`,
`FE_STABLE_CLIMATE`, or any T2 statistical verdict.  Because NEMO executes
Euler only once and then switches to MLF, the permanent-FE card remains a
NEMO-sourced stabilized approximation with no oracle trajectory analogue.
