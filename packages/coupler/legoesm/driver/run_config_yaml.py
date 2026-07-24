"""Shared YAML run-config loader for the production run drivers.

A ``--config FILE`` supplies argument DEFAULTS (applied via
``parser.set_defaults``), so any explicit CLI flag still overrides the file
(precedence: CLI > config file > parser default).  An optional ``include:`` base
is merged FIRST so tuned physics can live in one shared file
(``config/cmip/cmip_tuned_physics.yaml``) and be reused across run configs.

The loader is keyed purely on ``parser._actions`` dests, so it is
driver-agnostic: ``run_coupled.py`` and ``run_amip.py`` each pass their OWN
parser and the file is validated against that driver's CLI surface.  Every
merged key MUST be a known argument dest of the calling parser — an unknown key
is a hard error (dispatch-hardening: no silent typo'd / dropped override).

This is the single source of truth for the ``--config`` mechanism; the drivers
must not re-implement it (CLAUDE.md: no duplicate utilities).
"""

from __future__ import annotations

from pathlib import Path


def read_yaml_with_includes(path, _seen=None) -> dict:
    """Read a run YAML config, recursively merging an optional ``include:`` base
    FIRST so the tuned physics can live in one shared file and be reused across
    run configs.

    Precedence: the including file's keys override the base it includes
    (base < file).  An ``include:`` path is resolved relative to the including
    file.  Cycles and missing/non-mapping files raise.  Returns the merged raw
    dict; ``load_yaml_config`` then validates + type-coerces it.
    """
    import yaml
    p = Path(path).resolve()
    _seen = set() if _seen is None else _seen
    if p in _seen:
        raise SystemExit(f"--config: 'include' cycle detected at {p}.")
    _seen.add(p)
    if not p.exists():
        raise SystemExit(f"--config: file not found: {p}.")
    doc = yaml.safe_load(p.read_text())
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise SystemExit(
            f"--config {path}: expected a YAML mapping of argument=value, "
            f"got {type(doc).__name__}.")
    base_ref = doc.pop("include", None)
    merged: dict = {}
    if base_ref is not None:
        if not isinstance(base_ref, str):
            raise SystemExit(
                f"--config {path}: 'include' must be a single path string, "
                f"got {type(base_ref).__name__}.")
        merged.update(read_yaml_with_includes(p.parent / base_ref, _seen))
    merged.update(doc)  # the including file overrides its base
    return merged


def load_yaml_config(path, parser, *, example_keys: str | None = None) -> dict:
    """Load a run YAML config file into a dict of argument defaults.

    Used by ``--config`` to make a canonical run (e.g. the tuned
    ``config/cmip/cmip_ocean_{slab,3D}.yaml`` or the AMIP production config)
    reproducible from one file.  Supports an optional ``include:`` base merged
    first (see ``read_yaml_with_includes``).  Every (merged) key MUST be a known
    argument dest of ``parser``; an unknown key raises (no silent typo'd /
    dropped override — dispatch-hardening).

    Each scalar is coerced through that argument's ``type=`` callable, because
    ``parser.set_defaults`` (how the caller applies this) BYPASSES argparse's own
    type conversion: a value written as a quoted string (e.g. ``dt: "300"``)
    would otherwise reach the run as a str.  Returns the mapping so the caller
    can feed it to ``parser.set_defaults`` (an explicit CLI flag still wins).

    ``example_keys`` is an optional human hint (driver-specific example dests)
    appended to the unknown-key error to help operators fix typos.
    """
    doc = read_yaml_with_includes(path)
    actions = {a.dest: a for a in parser._actions}
    unknown = sorted(set(doc) - set(actions))
    if unknown:
        hint = f" (e.g. {example_keys})" if example_keys else ""
        raise SystemExit(
            f"--config {path}: unknown key(s) {unknown}. Keys must be valid "
            f"argument dests of this run script{hint}.")
    out: dict = {}
    for key, value in doc.items():
        argtype = getattr(actions[key], "type", None)
        # Coerce only string scalars through the arg's type (a YAML native
        # float/int/bool is already the right Python type; type=None args are
        # str/bool flags that need no conversion).
        if argtype is not None and isinstance(value, str):
            try:
                value = argtype(value)
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    f"--config {path}: key '{key}' value {value!r} is not a "
                    f"valid {getattr(argtype, '__name__', argtype)}: {exc}")
        # ``parser.set_defaults`` BYPASSES argparse's own ``choices`` check, so
        # a scheme literal (e.g. ``vertical_mixing_scheme: garbage``) supplied
        # via --config would otherwise reach the factory unvalidated.  Enforce
        # it here so a typo'd scheme fails loudly at load (dispatch-hardening),
        # exactly as an explicit CLI flag would.
        choices = getattr(actions[key], "choices", None)
        if choices is not None and value not in choices:
            try:
                allowed = sorted(choices)
            except TypeError:
                allowed = list(choices)
            raise SystemExit(
                f"--config {path}: key '{key}' value {value!r} is not one of "
                f"the allowed choices {allowed}.")
        out[key] = value
    return out


def require_config(config_value, *, driver: str = "run") -> None:
    """Enforce ``--require-config``: a run must be driven by a committed config.

    Raises ``SystemExit`` when ``--require-config`` is set but no ``--config``
    file was supplied, so a production/test run cannot silently fall back to
    parser defaults (issue #691).  A no-op when strict mode is off.
    """
    if config_value is None:
        raise SystemExit(
            f"{driver}: --require-config was set but no --config file was "
            "given.  Pass --config <yaml> so the run is fully specified by a "
            "committed configuration (no hidden parser defaults)."
        )


# ---------------------------------------------------------------------------
# --params : calibration (tuned parameter) loader (issue #691, format #690)
# ---------------------------------------------------------------------------
# A ``--params FILE`` supplies the CALIBRATION layer: converged scheme-parameter
# values, keyed by the ``param_collector`` qualified name ``scheme_key.field``
# (the SAME keying the registry and the SCM-RCE / AIMIP training output use, e.g.
# ``atm.clouds.CloudConfig.q_c_diagnostic: 3.0e-4``).  Applied to the built
# config AFTER ``--config``/CLI so a trained ``recommended_defaults`` drops in
# unchanged.  Every key is validated against the scheme's ``__param_spec__``
# (existence + bounds) and routed to the matching nested ``*Config`` NamedTuple
# by class name — a typo'd / out-of-bounds / absent-in-this-run parameter is a
# hard error, never a silent mis-set of physics.


def load_params_config(path) -> dict:
    """Load a calibration params YAML (``{qualified_name: value}``) → dict.

    Supports the same ``include:`` base merge as ``--config``.  Returns the raw
    ``{qualified_name: value}`` mapping; validation + routing happen in
    :func:`apply_params_to_config` (which needs the run's config object).
    """
    doc = read_yaml_with_includes(path)
    if not isinstance(doc, dict):
        raise SystemExit(f"--params {path}: top level must be a mapping of "
                         "'scheme_key.field: value' entries.")
    return dict(doc)


# The atmosphere ExperimentConfig FLATTENS a curated set of tunable scheme
# parameters to scalar fields (e.g. ``cloud_q_c_diagnostic`` <-
# CloudConfig.q_c_diagnostic) instead of nesting the scheme ``*Config``
# NamedTuples, so the class-router cannot reach them.  This maps each registry
# qualified name to its ExperimentConfig scalar — but ONLY for parameters the
# physics pipeline ACTUALLY threads from that scalar into the resolved scheme
# config (``build_cloud_config`` for clouds; ``_resolve_convection`` for
# sbm/bechtold).  Many ``<prefix>_<field>`` scalars EXIST on ExperimentConfig
# yet are never read (``_resolve_turbulence``/``_resolve_microphysics`` build
# default configs and patch only a few fields; ``_resolve_gwd`` did too until
# ``gwd_config_for`` landed — see its GWD block below), so a
# name-convention map would silently claim dead overrides.  The membership here
# is machine-verified end-to-end by ``test_atm_scalar_map_is_pipeline_threaded``
# (builds a pipeline per entry, asserts the resolved scheme config carries the
# value) — extend this dict only when the pipeline threads a new scalar
# (issue #691, codex audit).
_ATM_SCALAR_PARAM_MAP: dict[str, str] = {
    # clouds -> build_cloud_config (physics_pipeline)
    "atm.clouds.CloudConfig.rh_crit": "cloud_rh_crit",
    "atm.clouds.CloudConfig.q_c_diagnostic": "cloud_q_c_diagnostic",
    "atm.clouds.CloudConfig.conv_cloud_max": "cloud_conv_cloud_max",
    "atm.clouds.CloudConfig.conv_cloud_condensate": "cloud_conv_cloud_condensate",
    "atm.clouds.CloudConfig.p_xr": "cloud_p_xr",
    "atm.clouds.CloudConfig.alpha_xr": "cloud_alpha_xr",
    "atm.clouds.CloudConfig.adiabatic_lwc_rate": "cloud_adiabatic_lwc_rate",
    # convection -> _resolve_convection (physics_pipeline)
    "atm.conv.SBMConfig.tau_c": "sbm_tau_c",
    "atm.conv.SBMConfig.rh_ref": "sbm_RH_ref",
    "atm.conv.SBMConfig.cape_threshold": "sbm_cape_threshold",
    "atm.conv.BechtoldConfig.cape_threshold": "bechtold_cape_threshold",
    # cloud inhomogeneity (Cahalan) + convective autoconversion split (Sundqvist)
    # -> build_cloud_config / _resolve_convection (physics_pipeline)
    # NOTE: cloud_inhomogeneity_factor was REMOVED from this map 2026-07-23:
    # upstream #1280 moved it to CloudConfig's __param_spec__ EXCLUDED
    # partition (default 1.0 sits ON its physical bound — not a well-posed
    # sigmoid tunable), which drops it from the param registry, and a map key
    # absent from the registry breaks the --params loader contract (the
    # semantic conflict this branch inherited on merge).  The flat
    # ExperimentConfig scalar remains settable via --config / its CLI flag.
    "atm.clouds.CloudConfig.cloud_fsd": "cloud_fsd",
    "atm.conv.BechtoldConfig.autoconv_pe_max": "autoconv_pe_max",
    "atm.conv.BechtoldConfig.autoconv_q_c_crit": "autoconv_q_c_crit",
    # bechtold penetrative-downdraft closure knobs -> the dedicated
    # _bechtold_kwargs threading in _resolve_convection (unconditional), so
    # they are --params-reachable (2026-07-23; previously baselined CLI-only).
    "atm.conv.BechtoldConfig.downdraft_alpha": "bechtold_downdraft_alpha",
    "atm.conv.BechtoldConfig.downdraft_entrain_rate": "bechtold_downdraft_entrain_rate",
    "atm.conv.TiedtkeConfig.autoconv_pe_max": "autoconv_pe_max",
    "atm.conv.TiedtkeConfig.autoconv_q_c_crit": "autoconv_q_c_crit",
    # hard saturation-adjustment trigger + heating cap -> _resolve_microphysics
    # (physics_pipeline, via apply_microphysics_experiment_flags; the MPAS
    # post-step drain reads the same threaded sub-config in model_driver).
    # One flat scalar serves all five warm-rain schemes (only the active
    # scheme's sub-config is built).
    "atm.micro.KesslerConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.KesslerConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.MorrisonConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.MorrisonConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.P3Config.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.P3Config.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.SeifertBehengConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.SeifertBehengConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    "atm.micro.ThompsonConfig.hard_sat_adjust_threshold": "hard_sat_adjust_threshold",
    "atm.micro.ThompsonConfig.hard_sat_max_heating_K": "hard_sat_max_heating_K",
    # NOTE: LouisConfig.cloudtop_entrainment_efficiency was REMOVED 2026-07-23
    # for the same #1280 semantic conflict as cloud_inhomogeneity_factor above:
    # upstream excluded it from the __param_spec__ registry (default 0.0 = off
    # sits ON its bound), so the qualified name no longer exists for the
    # --params loader.  The flat scalar louis_cloudtop_entrainment_efficiency
    # remains settable via --config / CLI.  FOLLOW-UP: if --params reachability
    # is wanted back, re-spec the param with an activation-aware transform
    # instead of re-adding a dangling map key.
    # gravity wave drag -> gwd_config_for (physics_pipeline), which every lane
    # (FV pipeline / MPAS / spectral) now routes through.  Before it, these
    # scalars existed on ExperimentConfig but NO production path read them.
    # ``tau_max`` is tier 3 (a numerics clip), so the tier-1/2 reachability
    # audit does not require it — it is mapped anyway because the same resolver
    # threads it and the map's contract is "what the pipeline actually threads".
    "atm.gwd.HinesConfig.total_rms_wind": "hines_total_rms_wind",
    "atm.gwd.HinesConfig.Fmax": "hines_Fmax",
    "atm.gwd.McFarlaneConfig.directional_spread": "mcfarlane_directional_spread",
    "atm.gwd.McFarlaneConfig.tau_max": "mcfarlane_tau_max",
    # NOTE: ``mcfarlane_k_wave`` is threaded too but has NO ``__param_spec__``
    # entry (its computed 2*pi/100e3 default is not a float literal, so the
    # AST-based spec gate never required one) — there is no qualified name to
    # map.  Speccing it needs a bounds decision (ml/tuning.py says 1e-5..2e-4,
    # aimip_params says 1e-5..5e-4).  Until then its ONLY route is the
    # ``--mcfarlane-k-wave`` flag (which is also what makes the key legal in a
    # ``--config`` YAML — load_yaml_config rejects any key that is not a parser
    # dest), and validate_strict guards it positive+finite.
    # NOTE: the idealized GRAY radiation scheme threads a few of its params
    # (tau_equator, tau_pole via same-named scalars; sfc_albedo via the shared
    # `albedo_ocean` scalar) — deliberately NOT in this map.  Gray is not the
    # production radiation (rrtmgp is), and its scalars are set via `--config`
    # directly; the reachability audit baselines them under a documented
    # "idealized / --config-only" reason rather than the qualified-name loader.
}


def build_atm_scalar_param_map() -> dict[str, str]:
    """Return ``{registry qualified_name: ExperimentConfig scalar field}`` for
    the atmosphere tunable parameters that ExperimentConfig exposes as a flat
    scalar AND the physics pipeline actually threads into the scheme config
    (issue #691).  See :data:`_ATM_SCALAR_PARAM_MAP` for why this is a verified
    allowlist rather than a name-convention derivation."""
    return dict(_ATM_SCALAR_PARAM_MAP)


def _route_overrides_by_class(node, by_key: dict, *, applied: set):
    """Recursively splice ``{(module, class_name): {field: value}}`` into a
    config NamedTuple tree (depth-first; child configs updated before parent).

    Keyed on the FULL ``(defining module, class name)`` — not the bare class
    name — so same-named configs in different components (e.g. the ocean
    vertical-mixing ``TKEConfig`` vs the atmosphere turbulence ``TKEConfig``)
    never cross-route.  Records each applied key in ``applied`` so the caller can
    detect a target config that is ABSENT (never applied) or AMBIGUOUS (two
    instances of the same class in the tree)."""
    fields = getattr(node, "_fields", None)
    if fields is None or not isinstance(node, tuple):
        return node  # not a NamedTuple leaf
    replacements = {}
    for f in fields:
        child = getattr(node, f)
        new_child = _route_overrides_by_class(child, by_key, applied=applied)
        if new_child is not child:
            replacements[f] = new_child
    if replacements:
        node = node._replace(**replacements)
    key = (type(node).__module__, type(node).__name__)
    if key in by_key:
        if key in applied:
            raise SystemExit(
                f"--params: config {key[1]!r} ({key[0]}) appears more than once "
                "in the run config tree — cannot route the override unambiguously."
            )
        applied.add(key)
        # apply_param_overrides validates every field is on the NamedTuple.
        from legoesm.training.param_collector import apply_param_overrides
        node = apply_param_overrides(node, by_key[key])
    return node


def apply_params_to_config(config, params: dict, *, driver: str = "run",
                           scalar_param_map: dict | None = None):
    """Return ``config`` with calibration ``params`` (qualified_name: value)
    spliced into the matching nested ``*Config`` NamedTuples.

    ``scalar_param_map`` (``{qualified_name: config_field}``) handles configs
    that FLATTEN their tunable scheme params to top-level scalar fields instead
    of nesting the ``*Config`` — the atmosphere ExperimentConfig (see
    :func:`build_atm_scalar_param_map`).  A mapped parameter is applied to the
    top ``config`` via ``_replace`` (validated + bounds-checked identically);
    unmapped parameters use the nested class-router below.

    Each key is validated against ``param_collector.build_registry`` (must be a
    known parameter and, for scalar params, within its ``__param_spec__``
    bounds) and routed to the config of its declared ``(module, config_class)``.
    Raises ``SystemExit`` on an unknown parameter, a non-numeric or out-of-bounds
    value, or a target config that is absent from / ambiguous in this run's
    config tree — a ``--params`` file can never silently mis-set physics.  A
    no-op for empty ``params``.

    Note (soft limitation): a union config that holds ALL of a family's scheme
    sub-configs simultaneously (``VerticalMixingConfig`` carries kpp/tke/catke;
    ``MultiLayerLandConfig`` carries carbon + stomata) is always "present", so an
    override for a scheme that is not the *selected* one is applied to that
    (inert) sub-config rather than raising — it simply has no effect on the run.
    The strict absent-raise still catches wrong-component params (e.g. an
    atmosphere param in an ocean-only run).
    """
    if not params:
        return config
    from legoesm.training.param_collector import build_registry
    registry = {m.qualified_name: m for m in build_registry()}
    scalar_param_map = scalar_param_map or {}
    by_key: dict[tuple, dict[str, float]] = {}
    key_to_qname: dict[tuple, str] = {}
    flat: dict[str, float] = {}
    for qname, value in params.items():
        meta = registry.get(qname)
        if meta is None:
            raise SystemExit(
                f"{driver} --params: unknown parameter {qname!r} (not in the "
                "parameter registry).  Keys must be a param_collector qualified "
                "name 'scheme_key.field' (see config/cmip/params_tuned.yaml)."
            )
        # Scalar params (shape_key None) are numeric — coerce (YAML may quote
        # the value) and range-check against __param_spec__ bounds.  Array params
        # keep their list value (per-element tuple bounds are not range-checked).
        if meta.shape_key is None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"{driver} --params: {qname} value {value!r} is not numeric."
                )
            lo, hi = meta.bounds
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                if not (lo <= value <= hi):
                    raise SystemExit(
                        f"{driver} --params: {qname}={value} is outside its "
                        f"__param_spec__ bounds [{lo}, {hi}]."
                    )
        ec_field = scalar_param_map.get(qname)
        if ec_field is not None:
            flat[ec_field] = value  # flattened scalar on the top config
        else:
            key = (meta.module, meta.config_class)
            by_key.setdefault(key, {})[meta.field] = value
            key_to_qname[key] = qname
    if flat:
        top_fields = getattr(config, "_fields", ())
        bad = [f for f in flat if f not in top_fields]
        if bad:
            raise SystemExit(
                f"{driver} --params: mapped scalar field(s) {bad} are not on "
                f"{type(config).__name__} — the atmosphere scalar-param map does "
                "not match this driver's config."
            )
        config = config._replace(**flat)
    applied: set = set()
    config = _route_overrides_by_class(config, by_key, applied=applied)
    missing = set(by_key) - applied
    if missing:
        examples = ", ".join(sorted(key_to_qname[k] for k in missing))
        raise SystemExit(
            f"{driver} --params: parameter(s) {examples} target config "
            f"class(es) {sorted(k[1] for k in missing)} that are not present in "
            "this run's config (the scheme is not built into this driver's "
            "config object).  Remove them or enable the scheme."
        )
    return config

