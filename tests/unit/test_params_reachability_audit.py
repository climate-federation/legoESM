"""--params reachability audit — the #691 default-fallback ratchet.

Machine-checks, for every tier-1/2 ``__param_spec__`` parameter, whether it is
settable from a committed calibration file (``--params``, or the atm
``--config`` scalar it maps to) in at least one of its component's production
run drivers:

    atm     -> run_amip / run_coupled  (flattened ExperimentConfig scalar map)
    ocean   -> run_omip / run_coupled  (nested *Config class-router)
    land    -> run_lmip / run_coupled  (nested *Config class-router)
    ice     -> run_coupled             (SeaIceConfig bundle)
    coupler -> run_coupled             (CouplerConfig / LakeConfig bundle)

Reachability is probed through the PRODUCTION router itself
(``run_config_yaml._route_overrides_by_class`` — keyed on the defining
``(module, class name)``; a class instantiated twice in one tree is AMBIGUOUS
and therefore unroutable), so audit and loader semantics cannot drift.  The
computed uncovered set must EQUAL the
shrink-only baseline in ``_params_reachability_baseline.py``: a newly-spec'd
parameter that is unreachable goes red (wire it through the driver config or
consciously extend the baseline in review), and a newly-covered parameter goes
red until the baseline shrinks.  This is the machine-enforced form of #691's
"no production run depends on a hidden default for a tunable parameter".
"""
from __future__ import annotations

from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
from legoesm.training.param_collector import build_registry

from tests.unit._params_reachability_baseline import UNREACHABLE_PARAMS

_TUNABLE_TIERS = (1, 2)  # build_trainable_params "core" + "extended"


def _candidate_keys() -> set:
    """Distinct (module, class) keys of all tier-1/2 registry parameters —
    the only keys ``_compute_uncovered`` ever tests membership on."""
    return {(m.module, m.config_class) for m in build_registry()
            if m.tunable_tier in _TUNABLE_TIERS}


def _routable_keys(config_obj, candidate_keys: set) -> set:
    """Candidate keys the PRODUCTION router actually routes into ``config_obj``.

    Probes ``run_config_yaml._route_overrides_by_class`` (the exact function
    ``apply_params_to_config`` uses) with an empty per-key override: routing
    records the key in ``applied`` iff the class is PRESENT, and raises
    ``SystemExit`` iff it is AMBIGUOUS (instantiated more than once) — so the
    audit delegates presence/ambiguity semantics to production instead of
    re-implementing the tree walk (a mirror could drift)."""
    from legoesm.driver.run_config_yaml import _route_overrides_by_class

    routable: set = set()
    for key in candidate_keys:
        applied: set = set()
        try:
            _route_overrides_by_class(config_obj, {key: {}}, applied=applied)
        except SystemExit:
            continue  # AMBIGUOUS in this tree -> production refuses to route
        if key in applied:
            routable.add(key)
    return routable


def _driver_key_sets():
    """(module, class) reachability per nested-config driver + the atm map."""
    from scripts.run.run_lmip import _parse_args as lmip_args
    from scripts.run.run_lmip import build_config_from_args as lmip_build
    from scripts.run.run_omip import build_config_from_args as omip_build
    from scripts.run.run_omip import parse_args as omip_args

    candidates = _candidate_keys()
    omip_keys = _routable_keys(omip_build(omip_args(["--grid", "latlon"])),
                               candidates)
    # run_lmip coverage is the UNION over the user-selectable land-surface
    # schemes: --land-surface-scheme {simple_seb (default), two_leaf, clm_ml}
    # each constructs a DIFFERENT nested surface *Config (SimpleSEBConfig /
    # TwoLeafCanopyConfig / CLMMLConfig) BEFORE --params is spliced, so a param
    # in TwoLeafCanopyConfig/CLMMLConfig is reachable under its scheme even
    # though the default simple_seb config does not carry it.
    lmip_keys: set = set()
    for _scheme in ("simple_seb", "two_leaf", "clm_ml"):
        lmip_keys |= _routable_keys(
            lmip_build(lmip_args(
                ["--lat", "0.0", "--land-surface-scheme", _scheme])),
            candidates)

    # run_coupled --params bundle: the PRODUCTION routing bundle main()
    # applies (scripts/run/run_coupled.py::build_params_bundle — the coupled
    # config + explicitly-passed coupler / sea-ice / lake configs), so audit
    # and driver cannot drift.  Coverage is the union over the user-selectable
    # land variants (slab LandConfig default; MultiLayerLandConfig via
    # --land-scheme multilayer).
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.land.config import MultiLayerLandConfig

    from scripts.run.run_coupled import build_params_bundle

    coupled_keys: set = set()
    for cc in (CoupledConfig(),
               CoupledConfig(land_config=MultiLayerLandConfig())):
        coupled_keys |= _routable_keys(build_params_bundle(cc), candidates)

    return omip_keys, lmip_keys, coupled_keys


def _compute_uncovered() -> set[str]:
    """Uncovered tunables, honoring run_coupled's clobber-refusal contract.

    ``run_coupled._check_params_clobber`` hard-refuses routes whose overrides
    ``CoupledESMDriver.setup()`` would rebuild-and-discard, so the coupled
    bundle must NOT count as coverage for them:

    * ``ocean.*`` — refused unconditionally (calibrate via run_omip);
    * ``land.carbon.*`` — refused unconditionally (calibrate via run_lmip);
    * other ``land.*`` — refused under the DEFAULT ``--land-params clm`` but
      settable with ``--land-params analytical``, so the coupled route still
      counts (conditional reachability, documented in the guard's error).
    """
    omip_keys, lmip_keys, coupled_keys = _driver_key_sets()
    amap = set(build_atm_scalar_param_map())
    uncovered: set[str] = set()
    for m in build_registry():
        if m.tunable_tier not in _TUNABLE_TIERS:
            continue
        key = (m.module, m.config_class)
        comp = m.qualified_name.split(".")[0]
        if comp == "atm":
            covered = m.qualified_name in amap
        elif comp == "ocean":
            # run_coupled refuses ALL ocean.* (--params clobber guard), so
            # only run_omip counts.
            covered = key in omip_keys
        elif comp == "land":
            # run_coupled refuses land.carbon.* unconditionally; non-carbon
            # land.* is reachable there only via --land-params analytical.
            coupled_ok = (key in coupled_keys
                          and not m.qualified_name.startswith("land.carbon."))
            covered = key in lmip_keys or coupled_ok
        else:  # ice, coupler
            covered = key in coupled_keys
        if not covered:
            uncovered.add(m.qualified_name)
    return uncovered


def test_params_reachability_matches_baseline():
    uncovered = _compute_uncovered()
    newly_unreachable = uncovered - UNREACHABLE_PARAMS
    newly_covered = UNREACHABLE_PARAMS - uncovered
    assert not newly_unreachable, (
        "NEW tunable parameter(s) not settable via --params/--config in any of "
        f"their component's run drivers: {sorted(newly_unreachable)}.  Wire the "
        "scheme *Config onto the driver's config object (or expose an "
        "ExperimentConfig scalar for atm), or consciously add them to "
        "tests/unit/_params_reachability_baseline.py in review."
    )
    assert not newly_covered, (
        "Parameter(s) became reachable — shrink the baseline "
        f"(remove from _params_reachability_baseline.py): {sorted(newly_covered)}"
    )


def test_audit_is_not_vacuous():
    """Synthetic-violation guard: the audit really distinguishes covered from
    uncovered (guardrail doctrine: every tripwire proves it can fire)."""
    all_qnames = {m.qualified_name for m in build_registry()
                  if m.tunable_tier in _TUNABLE_TIERS}
    uncovered = _compute_uncovered()
    # Known-covered examples on each route: omip nested router, lmip nested
    # router, atm scalar map.
    for qname in (
        "ocean.vm.kpp.Ri_crit",
        "land.multilayer.Cd_land",
        "atm.clouds.CloudConfig.q_c_diagnostic",
    ):
        assert qname in all_qnames, f"{qname} vanished from the registry"
        assert qname not in uncovered, f"{qname} should be reachable"
    # Known-uncovered: pipeline-internal atm scheme configs.
    assert any(q.startswith("atm.") for q in uncovered), (
        "expected pipeline-internal atm params to be uncovered")


def test_clobber_refused_routes_have_covering_alternative():
    """Mechanical form of ``run_coupled._check_params_clobber``'s contract
    prose: every route the guard REFUSES must leave the parameter settable in
    the alternative driver its error message points at — otherwise the refusal
    would orphan a tunable and the guard-faithful branches in
    ``_compute_uncovered`` would silently hide it:

    * ``ocean.*`` refused -> run_omip must cover every ocean param the coupled
      bundle could otherwise route ("0 ocean params are coupled-only");
    * ``land.carbon.*`` refused -> run_lmip must route ``CarbonConfig``.

    (The stronger form — assert ``CoupledESMDriver.setup()`` literally rebuilds
    each refused config — needs a full setup() run with mesh build + CLM
    surfdata download, not unit-test-cheap; the guard refusals themselves are
    pinned by test_run_coupled_config_yaml.py::test_params_clobber_guard.)"""
    omip_keys, lmip_keys, coupled_keys = _driver_key_sets()
    for m in build_registry():
        if m.tunable_tier not in _TUNABLE_TIERS:
            continue
        key = (m.module, m.config_class)
        if m.qualified_name.startswith("ocean.") and key in coupled_keys:
            assert key in omip_keys, (
                f"{m.qualified_name}: run_coupled refuses ocean.* --params and "
                "points at run_omip, which does not route this config — the "
                "refusal orphans an otherwise coupled-routable ocean tunable."
            )
        if m.qualified_name.startswith("land.carbon."):
            assert key in lmip_keys, (
                f"{m.qualified_name}: run_coupled refuses land.carbon.* and "
                "points at run_lmip, which does not route CarbonConfig."
            )


def test_every_ice_and_coupler_param_reachable():
    """run_coupled's --params bundle makes ALL ice.* and coupler.* tunables
    settable — the concrete gap this audit closed (#691)."""
    uncovered = _compute_uncovered()
    leaked = {q for q in uncovered if q.split(".")[0] in ("ice", "coupler")}
    assert not leaked, f"ice/coupler tunables regressed to unreachable: {sorted(leaked)}"
