# Adversarial review ROUND 2 — `gwd_config_for` resolver (UNCOMMITTED)

You reviewed this change in round 1 and returned 7 CONFIRMED findings + 3 FALSE-POSITIVES
+ 1 test gap. Every confirmed finding has now been fixed or explicitly rebutted below.
Your job now: (a) verify each fix is CORRECT and COMPLETE (not a patch that moves the bug),
(b) find anything the fixes newly broke, (c) find anything BOTH of us still missed.
Read the real files; cite file:line. Repo root = cwd, working tree DIRTY.

## Round-1 findings and what was done

**R1-1 CONFIRMED (`--params` unreachable) — FIXED.**
`run_config_yaml.py` `_ATM_SCALAR_PARAM_MAP` gained 4 GWD entries
(`atm.gwd.HinesConfig.total_rms_wind` -> `hines_total_rms_wind`,
`atm.gwd.HinesConfig.Fmax` -> `hines_Fmax`,
`atm.gwd.McFarlaneConfig.directional_spread` -> `mcfarlane_directional_spread`,
`atm.gwd.McFarlaneConfig.tau_max` -> `mcfarlane_tau_max`) plus a note on why
`mcfarlane_k_wave` is NOT mapped. The stale `_resolve_gwd` rationale in the map preamble
was corrected. `tests/unit/_params_reachability_baseline.py` shrank by the 3 tier-1/2
entries. `tests/unit/test_params_config_loader.py` gained the GWD selector + `gwd_config`
readback in BOTH the over-claim (`sel`/`resolved_attr`) and the under-claim (`schemes`)
tables. VERIFIED: `test_params_reachability_audit.py` + `test_params_config_loader.py`
+ `test_param_specs.py` + `test_no_hardcoded_constants.py` + `test_tuning_trainable_bounds_sync.py`
= 3506 passed, 2 skipped.

**R1-2 CONFIRMED (seed ignores nested override) — FIXED.**
`model_driver.py` seed now uses `gwd_config_for(cfg)`; the now-dead local
`GravityWaveDragConfig` import was removed there and at the spectral lane (both were
orphaned by the migration; ruff F401 clean for GWD now). New regression
`test_physics_state_seed_honours_a_nested_override` builds an override with
`n_azimuths=8, n_wavenumbers=6, launch_flux=2.5e-3` and asserts the seeded carry is
`(3, 8, 6)` filled at `2.5e-3`. It FAILS on the old bare-config seed.

**R1-3 CONFIRMED (no validation) — FIXED, partially by design.**
`validate_strict` now rejects non-finite or non-positive values for all five overlaid
scalars (`mcfarlane_k_wave`, `mcfarlane_directional_spread`, `mcfarlane_tau_max`,
`hines_total_rms_wind`, `hines_Fmax`), with the failure modes documented at the guard.
DELIBERATELY positivity+finiteness only, NOT the `__param_spec__` ranges: duplicating the
calibratable range in two places is the drift bug this repo keeps paying for, and the
`--params` loader already range-checks against `__param_spec__`. 6 parametrized rejection
tests added. Your round-1 correction to my sign claim is ACCEPTED and the comment now says
what actually happens (`jnp.clip(drag, 0.0, Fmax)` with `Fmax < 0` returns the negative cap
at every level = constant spurious drag; the final `-minimum(abs(accel), cap)` keeps the
sign a deceleration).

**R1-4 (k_wave outside the param-spec contract) — REBUTTED as out of scope, documented.**
Adding a `__param_spec__` entry requires choosing bounds where the repo's two existing
catalogs disagree (`ml/tuning.py` 1e-5..2e-4 vs `aimip_params.py` 1e-5..5e-4). That is a
physics/calibration decision, not a wiring fix, and it would ADD a tier-1/2 param that then
needs its own map entry + audit churn. The concrete hazard you named (an unbounded live
knob) is closed by the R1-3 positive/finite guard; the residual is only that `--params`
cannot set it (`--config`/CLI can). Recorded as a NOTE at the map. Do you accept, or is
there a bug I am shipping by deferring?

**R1-5 CONFIRMED (`mcfarlane_N_ref` dead) — FIXED your way.**
The `TuningParameter` entry was DELETED from `packages/ml/legoesm/tuning.py` (it advertised
a nonexistent closure). The `ExperimentConfig` field is KEPT (positional ABI + serialized
configs + goldens) and re-commented `# INERT (no leaf field)` with the reason. No
validate_strict rejection of a non-default value was added — that would break existing
serialized configs that carry the historical default-valued field, for zero physics gain.

**R1-6 CONFIRMED (stale truncated fallback) — FIXED.**
Both converters now use `ExperimentConfig._field_defaults['mcfarlane_k_wave']` (the LIVE
field default, which cannot drift) instead of the truncated literal:
`config.py` `from_amip_config` and `to_amip_config`. New regression
`test_legacy_amip_upconvert_keeps_the_exact_k_wave_default` upconverts a real
`AMIPExperimentConfig()` and asserts the resolved leaf `== McFarlaneConfig()`.

**R1-7 CONFIRMED (positional ABI) — FIXED.**
`hines_total_rms_wind` / `hines_Fmax` moved from mid-tuple to the END of
`ExperimentConfig`, right after `mpas_ice_thickness_m`, with the file's own
"appended at the tuple END to preserve the positional ABI" note.

**R1-8/9/10 FALSE-POSITIVE — accepted, no action.** Byte-identical defaults, override
identity, composite dispatch, JIT/AIMIP all clean.

**R1-11 test gap — FIXED (partially).** `tests/unit/test_gwd_config_for.py` is now 28 tests:
- `test_defaults_match_bare_config` is parametrized over ALL 9 schemes and asserts
  WHOLE-TUPLE equality `gwd_config_for(cfg) == GravityWaveDragConfig(scheme=s)`.
- seed-with-override regression (R1-2).
- 6 validate_strict rejection cases (R1-3).
- legacy AMIP upconvert default (R1-6).
NOT added, with reasons: MPAS/spectral construction-path tests (those call sites are inside
1000-line driver methods with no seam short of a full driver build — the resolver itself and
`_resolve_gwd` are both directly tested, and the call sites are a one-token substitution);
an execution-level composite kernel test (`_combined_gwd` fan-out is already covered by the
composite suite; this change does not touch it); positional-compat test (the ABI is a
convention over field ORDER, and `test_yaml_to_experiment_config_golden` already pins the
full field set). Push back if you think any of these three is a real gap.

## Test evidence after the fixes

```
tests/unit/test_gwd_config_for.py tests/unit/test_gwd_override.py      34 passed
tests/unit/test_params_reachability_audit.py tests/unit/test_params_config_loader.py
  tests/unit/test_tuning_trainable_bounds_sync.py tests/test_param_specs.py
  tests/test_no_hardcoded_constants.py                                 3506 passed, 2 skipped
tests/unit/test_run_amip_cli.py test_yaml_to_experiment_config_golden.py
  test_config_roundtrip.py test_convection_config_for.py
  test_advertised_buildability.py test_gwd_landfrac_wiring.py          260 passed, 4 failed
tests/unit/test_mpas_qv_smoothing.py                                   17 passed
```
The 4 failures are PRE-EXISTING and confirmed unrelated:
- `test_run_amip_cli.py::test_config_yaml_round_trips_authoritative_values` and
  `::test_latlon24_production_variant_pins_polar_filter` both fail on
  `assert 'mcfarlane+hines' == 'mcfarlane'` (the YAML's GWD default vs the test's pin) —
  no `hines_*`/`mcfarlane_*` scalar involved.
- the 2 golden tests fail on 26 missing fields + `bechtold_downdraft_entrain_rate`
  0.0005 vs 0.0003 (upstream drift, confirmed failing at clean HEAD). `mcfarlane_k_wave`
  is NO LONGER in that diff.

## Current full diff of the change

Run `git diff` yourself. Files touched:
`packages/coupler/legoesm/driver/{config,model_driver,physics_pipeline,run_config_yaml}.py`,
`packages/ml/legoesm/tuning.py`, `scripts/run/run_amip.py`,
`tests/unit/{test_gwd_config_for,test_params_config_loader,_params_reachability_baseline}.py`,
`tests/unit/golden/yaml_to_experiment/*.json`.

## What I want attacked in round 2

1. Is the `_ATM_SCALAR_PARAM_MAP` extension CORRECT — do the 4 qualified names exist in
   `build_registry()`, does the tier-3 `tau_max` entry break the loader contract the way the
   removed `cloud_inhomogeneity_factor` did, and is the under-claim `schemes` reader
   (`pipe.gwd_config`) right given `get_gwd_fn` returns the LEAF for a single scheme but the
   FULL config for a composite?
2. Did shrinking the reachability baseline by exactly 3 leave it consistent (is
   `McFarlaneConfig.tau_max` supposed to be there too, or is tier-3 correctly excluded)?
3. Does the seed fix at `model_driver.py` now resolve the override for EVERY lane that seeds
   a carry, or is there a 4th seeding site I still missed? Grep `init_physics_state`.
4. Is the `validate_strict` positivity guard placed where every production entry point hits
   it, and does `math.isfinite` on a value that could legitimately be an int/None crash?
5. Does `ExperimentConfig._field_defaults[...]` inside `from_amip_config` / `to_amip_config`
   resolve correctly (staticmethod/classmethod scoping, class not yet bound at def time)?
6. Did moving the two `hines_*` fields to the tuple END break any keyword-construction,
   serialization, or golden-ordering assumption?
7. Is deleting the `mcfarlane_N_ref` tuning entry safe (any caller, YAML, or count-based
   test that indexes the catalog)?
8. Anything else, including whether any round-1 fix is wrong.

Answer as a numbered list tagged CONFIRMED-BUG / FALSE-POSITIVE / AMBIGUOUS with file:line.
If you find no substantive remaining issues, say so explicitly and explain why each residual
concern is not one.
