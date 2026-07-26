# Adversarial review ROUND 3 (final) — `gwd_config_for` resolver

Round 2 you returned 3 CONFIRMED bugs + 5 FALSE-POSITIVES. Two are fixed, one is rebutted
with grep evidence. Verify the fixes, judge the rebuttal, and state whether any substantive
issue remains. Repo root = cwd; run `git diff` for the real change.

## R2-1 CONFIRMED (`mcfarlane_k_wave` not settable) — FIXED, and you were right.

Verified your mechanism: `load_yaml_config` (`run_config_yaml.py:81-86`) raises SystemExit
on any `--config` key that is not an argparse dest, so with no flag the knob was
unreachable by BOTH routes. Added `--mcfarlane-k-wave` (`scripts/run/run_amip.py:1274-1282`)
+ `build_config_from_args` wiring, so `--config mcfarlane_k_wave:` is now legal too. The map
NOTE was corrected to say the flag is the gate for both routes (it previously claimed
`--config` worked without one — that was my error, not a comment nit).

New guard test `test_every_overlaid_scalar_has_a_run_amip_route` asserts EVERY scalar
`gwd_config_for` overlays is settable via a run_amip flag OR a scalar-map entry, so the next
overlaid scalar cannot ship unreachable.

## R2-3 CONFIRMED (goldens stale because of MY fields) — FIXED, and you were right.

I wrongly lumped the 2 new `hines_*` keys into the pre-existing golden drift. Measured with
the golden test's own `_canonical_from_config`:

```
BEFORE fix: williamson_test{2,5} produced-only keys = 26, including
            ['hines_Fmax', 'hines_total_rms_wind']
AFTER  fix: produced-only keys = 24, gwd-related = []
```

Both goldens gained `"hines_Fmax": 0.1` and `"hines_total_rms_wind": 2.0` (plus the earlier
`mcfarlane_k_wave` precision edit). This change's contribution to the golden diff is now
EXACTLY ZERO; the residual 24 keys + `bechtold_downdraft_entrain_rate` 0.0005/0.0003 +
`dycore`/`output` sub-dict drift are the documented upstream drift that fails at clean HEAD.

## R2-2 (legacy AMIP round-trip drops GWD scalars) — REBUTTED with evidence.

Claim: this is not a production drop-path, and it is not made worse by this change.

Evidence:
1. `to_amip_config()` has ZERO production callers.
   `grep -rn "to_amip_config()" --include=*.py packages src scripts | grep -v tests` = empty.
   Only `tests/unit/test_config_roundtrip.py` and `test_run_amip_cli.py` call it.
2. Checkpoint SAVE does not use it: `checkpoint.py:173-174` writes
   `json.dumps(config_to_dict(config))`, and `config_to_dict` (`config.py:2789-2805`) is a
   plain `_asdict()` over the NATIVE ExperimentConfig — it emits `hines_total_rms_wind`,
   `hines_Fmax` and every `mcfarlane_*` scalar. Restore goes through
   `experiment_config_from_dict`, which restores them by name.
3. The legacy flat schema is lossy for ~24 other ExperimentConfig knobs by EXPLICIT design
   (`config.py:2705-2712`: "ExperimentConfig has accreted many newer knobs the flat
   AMIPExperimentConfig never mirrored; drop those instead of raising (drift-proof
   round-trip)"). Extending it for GWD alone would be inconsistent, and would not close the
   general case.
4. `from_amip_config` is only reached for a LEGACY-FORMAT checkpoint
   (`checkpoint.py:42-63` dispatches on the absence of a `"grid"` dict). Such a checkpoint
   predates both fields, so there is no non-default value to lose — and the k_wave fallback
   it uses is now the live field default (R1-6 fix), which the new
   `test_legacy_amip_upconvert_keeps_the_exact_k_wave_default` pins.

If you still call this a bug, name the concrete production call sequence that loses a value.

## Related pre-existing issue I found while checking R2-2 (NOT fixed, flagging)

`config_to_dict` drops `turbulence_override` before serializing but NOT
`gravity_wave_drag_override` (`config.py:2799-2801`). A run with an override therefore
serializes it, and `experiment_config_from_dict` restores it as a plain DICT (only
grid/dycore/output are reconstructed as NamedTuples, `_SUB_CONFIGS`). The failure is LOUD —
`validate_strict` raises "gravity_wave_drag_override must be a GravityWaveDragConfig" — not
silent, and it predates this change. Do you agree it is out of scope here, or does the
resolver's `return override` verbatim make it newly dangerous?

## Test evidence (all on a compute node, JAX_ENABLE_X64=1)

```
test_gwd_config_for.py test_gwd_override.py test_run_amip_cli.py
  test_yaml_to_experiment_config_golden.py test_params_reachability_audit.py
  test_params_config_loader.py test_config_roundtrip.py     253 passed, 4 failed
test_params_reachability_audit test_params_config_loader
  test_tuning_trainable_bounds_sync test_param_specs
  test_no_hardcoded_constants                               3506 passed, 2 skipped
test_mpas_qv_smoothing.py                                   17 passed
```
The 4 failures are the documented pre-existing ones (2x `'mcfarlane+hines' == 'mcfarlane'`
default pin, 2x golden upstream drift with my contribution now zero).

## Final question

Is there any remaining substantive finding — a silently-inert parameter path, a wrong
value reaching a kernel, a broken conservation/JIT property, or a test that would not catch
a regression of what we fixed? If not, say "no substantive findings" and explain why each
residual concern (R1-4 k_wave param-spec deferral, R2-2, the override-serialization note)
is not a blocker.
