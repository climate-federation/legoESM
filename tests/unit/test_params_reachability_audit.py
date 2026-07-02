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

The reachability walk mirrors ``apply_params_to_config`` exactly (keyed on the
defining ``(module, class name)``; a class instantiated twice in one tree is
AMBIGUOUS and therefore unroutable).  The computed uncovered set must EQUAL the
shrink-only baseline in ``_params_reachability_baseline.py``: a newly-spec'd
parameter that is unreachable goes red (wire it through the driver config or
consciously extend the baseline in review), and a newly-covered parameter goes
red until the baseline shrinks.  This is the machine-enforced form of #691's
"no production run depends on a hidden default for a tunable parameter".
"""
from __future__ import annotations

from collections import Counter

from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
from legoesm.training.param_collector import build_registry

from tests.unit._params_reachability_baseline import UNREACHABLE_PARAMS

_TUNABLE_TIERS = (1, 2)  # build_trainable_params "core" + "extended"


def _walk_keys(node, counts: Counter) -> None:
    fields = getattr(node, "_fields", None)
    if fields is None or not isinstance(node, tuple):
        return
    counts[(type(node).__module__, type(node).__name__)] += 1
    for f in fields:
        _walk_keys(getattr(node, f), counts)


def _routable_keys(config_obj) -> set:
    """(module, class) keys reachable by ``apply_params_to_config``'s router.

    Mirrors ``_route_overrides_by_class``: a class that appears more than once
    in the tree raises AMBIGUOUS there, so it is NOT counted as reachable.
    """
    counts: Counter = Counter()
    _walk_keys(config_obj, counts)
    return {k for k, n in counts.items() if n == 1}


def _driver_key_sets():
    """(module, class) reachability per nested-config driver + the atm map."""
    from scripts.run.run_lmip import _parse_args as lmip_args
    from scripts.run.run_lmip import build_config_from_args as lmip_build
    from scripts.run.run_omip import build_config_from_args as omip_build
    from scripts.run.run_omip import parse_args as omip_args

    omip_keys = _routable_keys(omip_build(omip_args(["--grid", "latlon"])))
    lmip_keys = _routable_keys(lmip_build(lmip_args(["--lat", "0.0"])))

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
        coupled_keys |= _routable_keys(build_params_bundle(cc))

    return omip_keys, lmip_keys, coupled_keys


def _compute_uncovered() -> set[str]:
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
            covered = key in omip_keys or key in coupled_keys
        elif comp == "land":
            covered = key in lmip_keys or key in coupled_keys
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


def test_every_ice_and_coupler_param_reachable():
    """run_coupled's --params bundle makes ALL ice.* and coupler.* tunables
    settable — the concrete gap this audit closed (#691)."""
    uncovered = _compute_uncovered()
    leaked = {q for q in uncovered if q.split(".")[0] in ("ice", "coupler")}
    assert not leaked, f"ice/coupler tunables regressed to unreachable: {sorted(leaked)}"
