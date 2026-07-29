# Adversarial physics/config review — `gwd_config_for` resolver (UNCOMMITTED)

You are an independent adversarial reviewer for legoESM (JAX differentiable ESM).
Repo root = cwd. Branch `mpas-stability-campaign`, HEAD `260447d0e`, working tree DIRTY.
Find every bug, sign error, unit inconsistency, silently-inert-parameter path,
conservation/JIT/retrace hazard, and test gap. CITE FILE:LINE. Read the real files —
do not trust this summary. If a candidate concern is NOT a bug, say so and explain why.

## Confirmed pre-existing defect this change fixes

Every production lane built `GravityWaveDragConfig(scheme=cfg.gravity_wave_drag)` from
the scheme STRING alone — MPAS (`packages/coupler/legoesm/driver/model_driver.py:6067`),
spectral (`:7578`), and the FV pipeline `_resolve_gwd`
(`packages/coupler/legoesm/driver/physics_pipeline.py:3439`). So the tuned
`ExperimentConfig` scalars `mcfarlane_k_wave` / `mcfarlane_directional_spread` /
`mcfarlane_tau_max` were SILENTLY INERT on every AMIP path, and Hines had no knob at all.
Same gap class as the `convection_config_for` bug fixed earlier this campaign.

Campaign motivation: the ua800 extratropical westerly bias traces to non-orographic drag
(controlled single-delta arm, day-30 matched: hines OFF gives SH +1.61 -> +2.55,
NH +2.08 -> +2.71 m/s at 800 hPa), so the Hines launch amplitude must be tunable.

## The change (uncommitted, `git diff`)

### 1. NEW resolver `physics_pipeline.py:3385-3435`

```python
def gwd_config_for(config):
    """... (docstring omitted, read the file) ..."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )

    scheme = getattr(config, "gravity_wave_drag", "none")
    override = getattr(config, "gravity_wave_drag_override", None)
    if override is not None:
        return override
    gc = GravityWaveDragConfig(scheme=scheme)
    if scheme == "none":
        return gc
    # McFarlane (orographic) tunables. ``mcfarlane_N_ref`` is deliberately
    # NOT wired: no McFarlaneConfig field of that name exists (dangling
    # ExperimentConfig scalar, tracked separately).
    mc = gc.mcfarlane._replace(
        k_wave=float(getattr(config, "mcfarlane_k_wave", gc.mcfarlane.k_wave)),
        directional_spread=float(getattr(
            config, "mcfarlane_directional_spread",
            gc.mcfarlane.directional_spread)),
        tau_max=float(getattr(config, "mcfarlane_tau_max",
                              gc.mcfarlane.tau_max)),
    )
    # Hines (non-orographic) launch amplitude + saturation flux cap.
    hn = gc.hines._replace(
        total_rms_wind=float(getattr(config, "hines_total_rms_wind",
                                     gc.hines.total_rms_wind)),
        Fmax=float(getattr(config, "hines_Fmax", gc.hines.Fmax)),
    )
    return gc._replace(mcfarlane=mc, hines=hn)
```

`_resolve_gwd` now delegates: `get_gwd_fn(gwd_config_for(config))`; its inline
override branch was removed.

### 2. `driver/config.py` ExperimentConfig (lines ~983-1000)

- NEW `hines_total_rms_wind: float = 2.0`, `hines_Fmax: float = 0.1`
  (match `HinesConfig` defaults at
  `packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/config.py:443-444`).
- `mcfarlane_k_wave` default literal CORRECTED from truncated `6.283185307e-5`
  to exact `2.0 * math.pi / 100e3` (== `McFarlaneConfig.k_wave`, config.py:400).
  Without this the newly-live overlay would have perturbed the kernel default by
  ~3e-11 relative.

### 3. `model_driver.py` — MPAS (:6067) and spectral (:7578) call sites use the resolver.

### 4. `scripts/run/run_amip.py` — `--hines-total-rms-wind`, `--hines-fmax`,
`--mcfarlane-tau-max` + `build_config_from_args` wiring.

### 5. `tests/unit/test_gwd_config_for.py` — 12 tests (PASSING, see below).

### 6. `tests/unit/golden/yaml_to_experiment/*.json` — ONLY the `mcfarlane_k_wave`
value updated to the exact constant. (Those goldens have SEPARATE pre-existing
upstream drift — 26 missing fields + `bechtold_downdraft_entrain_rate` 0.0005->0.0003 —
CONFIRMED failing at clean HEAD by stash-test. NOT in scope; do not propose fixing it.)

## Test evidence

```
srun ... JAX_ENABLE_X64=1 pytest tests/unit/test_gwd_config_for.py \
    tests/unit/test_gwd_override.py tests/unit/test_convection_config_for.py -q
24 passed in 2.98s
```

## My own preliminary findings — ATTACK THESE, confirm or refute each

**P1 (candidate CONFIRMED, mine).** `_ATM_SCALAR_PARAM_MAP`
(`packages/coupler/legoesm/driver/run_config_yaml.py:177-239`) is the allowlist that makes
a flat ExperimentConfig scalar reachable from a `--params` calibration YAML. Its preamble
(:169-171) explicitly names `_resolve_gwd` as a resolver that "build[s] default configs and
patch[es] only a few fields", so "a name-convention map would silently claim dead
overrides" — that rationale is now STALE for GWD. Meanwhile
`tests/unit/_params_reachability_baseline.py:184-186,258` still baselines
`atm.gwd.HinesConfig.Fmax`, `atm.gwd.HinesConfig.total_rms_wind` (tier 1) and
`atm.gwd.McFarlaneConfig.directional_spread` (tier 2) as UNREACHABLE. They are now
reachable. Doctrine (CLAUDE.md) is that the baseline is shrink-only and a newly-covered
param must shrink it. Is adding those 3 map entries + shrinking the baseline REQUIRED
here, or out of scope? Note `test_atm_scalar_map_is_pipeline_threaded` builds a pipeline
per entry and asserts the resolved scheme config carries the value — will it pass for a
GWD entry (the pipeline needs `gravity_wave_drag` set to a scheme that builds those
leaves)? `McFarlaneConfig.tau_max` is tier 3 — does the audit/loader accept a map entry
for a tier-3 param?

**P2 (candidate CONFIRMED, mine).** Third GWD-config construction site
`model_driver.py:9418` — the physics-state SEED — was left as bare
`GravityWaveDragConfig(scheme=cfg.gravity_wave_drag)`. It feeds
`init_physics_state`, which at
`packages/atmosphere/legoesm/atmosphere/physics/physics_state.py:296-311` reads
`gwd_cfg.prognostic_spectral.n_azimuths / .n_wavenumbers / .launch_flux` to size and fill
the `gwd_spectrum` carry. `gwd_config_for`'s override branch returns
`gravity_wave_drag_override` VERBATIM, and that override may carry a non-default
`prognostic_spectral` (n_azimuths / n_wavenumbers / launch_flux) — the override is the
documented ONLY coupled-path route to nested GWD options (config.py:1235-1242). So a run
with `gravity_wave_drag_override=GravityWaveDragConfig(scheme="prognostic_spectral",
prognostic_spectral=PrognosticSpectralConfig(n_azimuths=8))` seeds the carry at
(ncol,4,20) while the kernel expects (ncol,8,20) — shape mismatch / wrong launch_flux.
Pre-existing (the override branch predates this change) but one line from being fixed now
that the resolver exists. Is that a real bug on a reachable path, or is the seed
consistent by some route I missed? Check the shape-mismatch raise at model_driver:9433.

**P3 (candidate, mine).** `ExperimentConfig.from_amip_config` (config.py:2456) keeps the
now-stale TRUNCATED literal `getattr(amip_cfg, 'mcfarlane_k_wave', 6.283185307e-5)` as its
default fallback, and `to_amip_config` (:2633) the same. Since the overlay is now LIVE, a
legacy-AMIP-format checkpoint upconverted through `checkpoint.py:63,287` gets a k_wave that
differs from the ExperimentConfig default by ~3e-11 relative. Negligible physically, but it
is a silent default divergence + a hardcoded constant. Also: the new `hines_*` fields are
absent from both enumerated converters — is that a real drop-path (I believe not:
`AMIPExperimentConfig` (`packages/tools/legoesm/forcing/amip_config.py`) has NO
mcfarlane/hines fields at all, `to_amip_config` filters to `_fields`, so nothing to carry),
or did I miss a lane?

**P4 (candidate, mine).** `mcfarlane_N_ref`: I found NO `McFarlaneConfig` field of that name
(fields at gwd config.py:399-415). It IS registered as a tunable in
`packages/ml/legoesm/tuning.py:505` with bounds and a "Reference stratification used in the
orographic launch-stress closure" note. So it appears to be a fully dead knob that a user
can set and that silently does nothing — the exact bug class this PR fixes. Confirm it is
dead everywhere (including `mcfarlane.py` reading N from the column state rather than a
config), and say whether it should be removed or left (removing an ExperimentConfig field
breaks the positional ABI / goldens).

**P5 (candidate, mine).** `mcfarlane_k_wave` has NO `__param_spec__` entry in the GWD config
(neither `params` nor `excluded`) because its default is a computed expression, not a float
literal, so `all_float_field_names`/`required` never forces it (tests/test_param_specs.py:
235-283). It is now LIVE on every lane with no registered bounds and no CLI flag. Is that a
gap worth closing, or correctly out of scope?

## Attack surface I want independently checked

1. **Byte-identical defaults.** Verify EVERY overlaid field's ExperimentConfig default
   equals its leaf default — `mcfarlane_k_wave`, `mcfarlane_directional_spread`,
   `mcfarlane_tau_max`, `hines_total_rms_wind`, `hines_Fmax` — so no lane that relied on
   bare defaults changes. Include the `float()` coercion and NamedTuple `_replace` equality.
2. **Override precedence / identity.** The resolver returns the override OBJECT (not a
   copy). Any caller that mutates or expects a copy? Any caller that then `_replace`s it?
3. **Composite parsing.** Does `get_gwd_fn` (`.../gravity_wave_drag/integration.py:68-101`)
   plus `_combined_gwd` (:186+) actually deliver BOTH overlaid leaves to BOTH kernels for
   `"mcfarlane+hines"`, not just the first part?
4. **Third call site** (P2 above).
5. **JIT / retrace.** Overlaid values are static Python floats read at trace time — confirm
   no traced leakage, no new retrace trigger, no pytree-structure change.
6. **AIMIP interaction.** `packages/ml/legoesm/training/aimip_params.py:920-927` builds its
   OWN `GravityWaveDragConfig` from its own trainable dict. Does the new resolver
   double-apply or conflict on any shared path? Also
   `scripts/run/run_aimip_amip_{inference,finetune}.py:282/252` still build bare configs —
   in scope or not?
7. **validate_strict coverage.** The new `hines_*` fields have no bounds check; neither do
   the `mcfarlane_*` ones. Is a non-negative/finite guard warranted for a launch amplitude
   (a negative `total_rms_wind` or `Fmax` would flip the sign of a momentum-flux cap)? Trace
   what a negative value does in `hines.py`.
8. **Test gaps.** What does `tests/unit/test_gwd_config_for.py` fail to pin? In particular:
   is `got == GravityWaveDragConfig(scheme=s)` asserted for the WHOLE config at defaults
   (not just the two leaves), and for every scheme?
9. Anything else — including anything I got wrong above.

Answer as a numbered finding list, each tagged CONFIRMED-BUG / FALSE-POSITIVE / AMBIGUOUS,
with file:line and the smallest correct fix. Be adversarial: I would rather get 10 findings
of which 6 are refutable than miss one silently-inert-parameter path.
