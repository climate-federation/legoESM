You are an independent adversarial reviewer for LegoESM (JAX-native differentiable ESM).
ROUND-5 (final) review of the day-137 drain-capacity lever wiring (hard-saturation-adjustment
trigger + per-step heating-cap overrides, threaded to the MPAS post-step drain). This wiring was
partly authored concurrently by two people and deduplicated; the key risk is duplicate/residue
definitions. FINDINGS ONLY — do NOT modify any file. Read the actual source (repo cwd, working
tree); cite file:line; label CONFIRMED/PLAUSIBLE + severity.

## What the change does
A pilot MPAS AMIP run died day 137 with the post-step hard-saturation drain PINNED at its 5 K/step
cap. `--params` can't reach the MPAS micro sub-config (built in `_run_mpas`, not the flattened
scalar map), so two ExperimentConfig flat scalars were added and threaded:
- `ExperimentConfig.hard_sat_adjust_threshold / hard_sat_max_heating_K` (None defaults, config.py
  ~621); validate_strict gate (requires `hard_saturation_adjustment=True`) + bounds ((1,2)/(0.5,50))
  loop (config.py ~1787).
- run_amip `--hard-sat-adjust-threshold` / `--hard-sat-max-heating-k` (~955) + build_config kwargs
  (~1631).
- `_run_mpas` threads the overrides through `apply_microphysics_experiment_flags`
  (model_driver ~5722-5746) so the post-step drain reads them from `_hsub.hard_sat_*` (~5770);
  the drain call is at ~6538 via `_mpas_hard_saturation_poststep`.
- Coupled path already consumed the scalars (physics_pipeline ~3093).
- `apply_microphysics_experiment_flags` (microphysics/config.py ~1000) is the SHARED fail-loud
  applier. Scheme sub-config defaults are real (1.1 / 5.0 K); ExperimentConfig defaults are None.
- Tests: tests/unit/test_run_amip_cli.py has a combined test (~446) AND three user tests
  (~3076-3122). The `--params` route uses `_ATM_SCALAR_PARAM_MAP` (run_config_yaml.py:177).

## Specific asks (rank CONFIRMED/PLAUSIBLE)
1. Deduplication residue: is there EXACTLY ONE of each — argparse `add_argument`, ExperimentConfig
   NamedTuple field, validate loop/block, build_config kwarg? Any dangling/orphaned reference to a
   removed copy? (Duplicate NamedTuple fields silently keep the last; duplicate add_argument raises.)
2. Do the MPAS post-step reads `_hsub.hard_sat_adjust_threshold / _hsub.hard_sat_max_heating_K`
   pick up the OVERRIDE values (applier `_replace` semantics), and fall back to the scheme
   __param_spec__ defaults (1.1 / 5.0) when the override is None?
3. Does the applier's fail-loud contract still hold with None defaults (None override on a scheme
   lacking the field = no-op; a non-None override on a scheme lacking the field = raise)? Is the
   `hard_saturation_adjustment=True` bool fail-loud covered on the MPAS path (which does NOT thread
   the bool through the applier — model_driver ~5757)?
4. Is the coupled path (physics_pipeline ~3077-3103) unaffected — bool applied in-scheme once, float
   overrides threaded via the applier only when present, no double-application?
5. Do the combined test (~446) and the three user tests (~3076-3122) have any CONTRADICTING
   assertion? Any duplicate/name-shadowing test?
6. Any NEW issue. Anything you would BLOCK the day-137 tuning A/B (single-process MPAS) on.

## Key files to read
- packages/coupler/legoesm/driver/config.py (fields ~621; validate ~1787)
- scripts/run/run_amip.py (argparse ~955; build ~1631)
- packages/coupler/legoesm/driver/model_driver.py (MPAS applier call ~5722-5773; drain ~6538;
  helper `_mpas_hard_saturation_poststep` ~217)
- packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py (applier ~1000; scheme
  field defaults ~342/397/767/857/940)
- packages/coupler/legoesm/driver/physics_pipeline.py (coupled path ~3077-3103)
- packages/coupler/legoesm/driver/run_config_yaml.py (_ATM_SCALAR_PARAM_MAP ~177)
- tests/unit/test_run_amip_cli.py (~64, ~446, ~3076-3122)
