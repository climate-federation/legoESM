# DINO MLF composition A/B vs NEMO — result

Preregistration: `dino_mlf_composition_ab_preregister.md` (commit `e0862a8f9`,
committed BEFORE this measurement ran). Follows M-01
(`nemo_branch_isomorphism_map.md`, commit `80eaf9b33`), which measured
`_leapfrog_step` vs `_nemo_mlf_step` against EACH OTHER only.

## Rule 0 recap (full citations in the preregister doc)

`stpmlf.F90:196-199/250/368`: ONE traversal of `stp_MLF` per step;
`dyn_ldf`/`tra_ldf` (and the slope precompute feeding `tra_ldf`) read `Nbb` as
their OWN call argument, every other term in the SAME traversal reads `Nnn`.
`_nemo_mlf_step` (`ocean_model_latlon_cgrid.py:10636`, `_ldf_state=` at
`ocean_pe_latlon_cgrid.py:4624-4631`/`4612-4626`) reproduces this per-term
substitution inside ONE `_step_impl` call. `_leapfrog_step` (`:9885`) instead
runs the WHOLE pipeline twice (full Nnn pass + full Nbb pass) and combines
additively — a mechanism substitution `stpmlf.F90` does not require.

**Prediction (preregistered): `_nemo_mlf_step` sits closer to NEMO on T/S.**

## Method

`kamm_twin_90d._build_twin_state("nemo_dino_kamm_mlf", ...)`, fp64
(`PrecisionPolicy.fp64()` set explicitly), CPU, `LEGOESM_NEMO_E3T=both`.
Certified `mc` UNCHANGED for both arms (`outer_integrator="leapfrog"`,
`implicit_vmix_e3t_now_divisor=False`) — `_leapfrog_step`/`_nemo_mlf_step`
called directly (bypassing `model.step()`'s dispatch), avoiding the
`outer_integrator="nemo_mlf"` construction-time hard-require of
`implicit_vmix_e3t_now_divisor=True`. Two INDEPENDENT 160-step (dt=2700s, 5
days) trajectories from the SAME bridged day-180 state. vs-NEMO ground truth:
`RUN_D180_STEP1_1R` per-step restarts at kt=5761..5764 (steps 1-4 — the only
exact NEMO reference past day 180 anywhere in this oracle tree; nothing
exists at day 185/step 160 itself, confirmed by listing every DINO run
directory before measuring). Script:
`scripts/validate/ocean_fidelity/dino_1226/mlf_composition_ab_vs_nemo.py`.

Cross-check: this run's own step-1 arm-vs-arm T (4.9631e-4 degC) reproduces
`80eaf9b33`'s recorded 4.963e-4 exactly — two independent probes agreeing
(skill Rule 1e).

Determinism control: `_leapfrog_step` on the identical input state, twice —
**0.0 on every field** (log: "determinism control: leapfrog vs leapfrog =
0.0 on all fields").

## Table: worst |diff|, wet cells

| step | comparison | T [degC] | S [PSU] | u [m/s] | v [m/s] | eta [m] |
|---|---|---|---|---|---|---|
| 1 | leapfrog vs NEMO | 1.8352e-3 | 1.6828e-4 | 2.2673e-5 | 2.5608e-5 | 9.0337e-6 |
| 1 | nemo_mlf vs NEMO | 1.8352e-3 | 1.6828e-4 | 2.2692e-5 | 1.7253e-5 | 9.0439e-6 |
| 1 | leapfrog vs nemo_mlf | 4.9631e-4 | 4.1920e-5 | 5.1962e-6 | 9.5676e-6 | 5.7247e-6 |
| 4 | leapfrog vs NEMO | 1.04537e-2 | 2.01272e-3 | 1.10656e-2 | 4.81471e-3 | 1.48668e-5 |
| 4 | nemo_mlf vs NEMO | 1.04541e-2 | 2.01278e-3 | 1.10656e-2 | 4.81470e-3 | 1.48627e-5 |
| 4 | leapfrog vs nemo_mlf | 8.3369e-4 | 7.2532e-5 | 2.0436e-5 | 1.3935e-5 | 1.0769e-6 |
| 160 (day 5) | leapfrog vs nemo_mlf — **NO NEMO ANCHOR AT THIS STEP** | 1.33781e-2 | 1.53198e-3 | 2.74933e-2 | 9.89987e-3 | 3.57349e-5 |

`ratio_farther_over_closer` at step 4 (closer arm named): T 1.0000386
(leapfrog closer), S 1.0000255 (leapfrog closer), eta 1.0002723 (nemo_mlf
closer), u 1.0000005 (nemo_mlf closer), v 1.0000024 (nemo_mlf closer). Every
field's ratio is within 0.03% of 1.0.

## RETRACTION — prediction REFUTED

Preregistered: `_nemo_mlf_step` closer to NEMO on T/S. **Measured: it is not.**
At every one of steps 1-4, `_leapfrog_step` is marginally CLOSER to NEMO on T
and S (the two fields where the arms are known to disagree via the GM/Redi
channel) — the opposite of the prediction, though the margin (~4e-7 out of a
~1e-2 total, step 4) is three-to-four orders of magnitude smaller than the
arms' own disagreement with each other (8.3e-4 at step 4). Reason the Rule-0
structural read did not predict this correctly: matching `stpmlf.F90`'s
CALL-ARGUMENT routing (which time level `dyn_ldf`/`tra_ldf` read as their own
operand — the fact the citations establish) is not the same claim as matching
NEMO's numeric trajectory, because `dynldf_lev_rot_scheme.h90`'s Kbb
differencing also carries Kbb-level METRICS (`e3t/e3u/e3v` at Kbb, only the
final normalisation at Kmm) that neither legoESM arm was checked against
term-by-term here — PLAUSIBLE candidate, not measured in this run. Both arms
turn out to be equidistant from NEMO to 4-5 significant figures despite
disagreeing substantially with each other; the worst-cell location for
arm-vs-arm need not coincide with the worst-cell location for either arm's
NEMO distance (max-abs-diff reduction, independently maximised per
comparison — this is stated, not resolved).

## Verdict: NOT SCALE-COMPATIBLE

Every field's step-4 ratio (farther-arm-distance / closer-arm-distance) is
between 1.0000005 and 1.0002723 — nowhere near the factor->2 bar. **The MLF
composition choice is NOT a scale-compatible candidate mechanism for the
20-year DINO climate divergence.** Whatever dominates the ~1e-2 degC (T) /
~1e-2 m/s (u) distance from NEMO already present by step 4 is common to BOTH
compositions and roughly 3-4 orders of magnitude larger than the effect of
switching between them.

## UNVERIFIED

- The metric-routing mechanism offered above for the near-tie (Kbb-level
  `e3t/e3u/e3v` vs Kmm-level in `_leapfrog_step`'s second full pass) is
  PLAUSIBLE, not measured in this run — no term-by-term metric probe was run.
- `read_nemo_restart`'s own docstring labels the `tn/sn/un/vn` fields it
  reads as "(Kbb)" while this comparison treats them as NEMO's "now" state
  (matching legoESM's post-step `state.T`, and matching the existing,
  repeatedly-passing `verify_day0_matches_restart` day-0 gate's own
  pairing) — read as a stale/imprecise docstring label, not re-verified
  against `istate.F90`/`stpmlf.F90`'s swap here.
- Whether the ~1e-2 common-mode distance from NEMO by step 4 is itself a
  known, already-attributed residual (e.g. from `d180_step_walk.py`'s Phase
  1) was not cross-checked in this task.

## Choices — ASKED / UNASKED

- ASKED: 160 steps = 5 days at dt=2700s; the factor->2 SCALE-COMPATIBLE
  threshold — both specified verbatim in the task.
- ASKED: calling `_leapfrog_step`/`_nemo_mlf_step` directly with `mc` held at
  the certified values (avoiding the divisor confound) — the task's own
  explicit instruction for exactly this situation.
- **UNASKED**: substituting step 4 (kt=5764, the nearest EXACT NEMO anchor)
  for the literal "day 5" vs-NEMO comparison, because no NEMO reference
  exists at day 185/step 160 anywhere in this oracle tree. Preregistered in
  writing before measuring (not discovered after), but the substitution
  itself was my call, not put to the user first. Offered for revert: the
  alternative is running actual NEMO for 160 more steps from the day-180
  restart to manufacture a day-185 reference (cheap, single serial DINO
  config, not attempted here since it is a new-data-source choice).

## Provenance

Producer git SHA `e0862a8f9` (this branch, after the preregister commit).
Command: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
python scripts/validate/ocean_fidelity/dino_1226/mlf_composition_ab_vs_nemo.py --steps 160`.
No model code changed; measurement only.
