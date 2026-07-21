"""A scheme that RUNS in one driver must be selectable in the other.

The coupled atmosphere IS the AMIP atmosphere: ``CoupledESMDriver`` builds
``self._atm = ModelDriver(atm_config)`` -- the same driver and physics pipeline
``run_amip`` uses. So when two drivers offer different scheme sets on the same
axis and both resolve through the same factory, the narrower list is DRIFT, and
a user cannot select physics the model implements.

Each driver kept its own hardcoded ``choices=`` copy, and each axis happened to
be guarded by a point-fix test in exactly ONE driver -- so every axis drifted in
the other. Concretely: ``run_coupled`` blocked bechtold/tiedtke/emanuel/
kain_fritsch/zhang_mcfarlane (bechtold is run_amip's DEFAULT and the scheme in
config/amip/amip_production.yaml) and clubb_lite/ysu; ``run_amip`` blocked
mynn25.

NOT everything narrower is drift, and this file deliberately does NOT demand the
two drivers agree everywhere:
  * ``--microphysics ml_emulator`` returns an UNTRAINED network's ~zero
    tendencies ("suitable for testing" per its docstring) and run_amip exposes
    no checkpoint flag, while _require_full_physics_for_amip would count it
    ACTIVE. run_amip excluding it is PROTECTIVE.
  * ``--clouds resolved`` is SAM's CRM convention (cf=1 wherever condensate>0)
    and raises without explicit q_cloud/q_ice from microphysics. Excluding it
    from a C48 GCM driver is protective too.
A narrower CLI list can encode "this scheme does not function here" -- so a
mismatch is only a bug once the scheme is shown to resolve AND to be offered by
a sibling driver on the same pipeline.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.convection.integration import _get_convection_fn
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
from legoesm.driver.config import VALID_CONVECTION_SCHEMES, ExperimentConfig

_TURBULENCE_SCHEMES = (
    "smagorinsky", "louis", "tke", "mynn25", "clubb_lite", "clubb",
    "holtslag_boville", "ysu", "edmf", "none",
)


def _actions(path: str, name: str, builder: str) -> dict:
    sys.argv = [name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {a.dest: a for a in getattr(mod, builder)()._actions}


@pytest.fixture(scope="module")
def amip():
    return _actions("scripts/run/run_amip.py", "_ra_test", "build_arg_parser")


@pytest.fixture(scope="module")
def coupled():
    return _actions("scripts/run/run_coupled.py", "_rc_test", "build_parser")


# --- the implemented set is real (no phantoms in what we offer) -------------

@pytest.mark.parametrize("scheme", VALID_CONVECTION_SCHEMES)
def test_every_offered_convection_scheme_resolves(scheme):
    """Guard against the inverse error: offering a scheme that cannot run."""
    _get_convection_fn(ConvectionConfig(scheme=scheme))


@pytest.mark.parametrize("scheme", _TURBULENCE_SCHEMES)
def test_every_offered_turbulence_scheme_resolves(scheme):
    get_turbulence_fn(TurbulenceConfig(scheme=scheme))


# --- reachability ----------------------------------------------------------

def test_coupled_convection_offers_every_valid_scheme(coupled):
    assert set(coupled["convection"].choices) == set(VALID_CONVECTION_SCHEMES)


@pytest.mark.parametrize("scheme", ["bechtold", "tiedtke", "emanuel",
                                    "kain_fritsch", "zhang_mcfarlane"])
def test_coupled_can_select_the_schemes_it_used_to_block(coupled, scheme):
    """tiedtke is run_amip's --convection default and bechtold is the scheme
    pinned by config/amip/amip_production.yaml -- so a coupled run could select
    NEITHER of the two the atmosphere is actually run with."""
    assert scheme in coupled["convection"].choices


def test_both_drivers_offer_the_same_turbulence_set(amip, coupled):
    """Same ModelDriver, same factory -> the two lists must not disagree.

    They previously did, in BOTH directions: run_amip lacked mynn25, run_coupled
    lacked clubb_lite and ysu.
    """
    assert set(amip["turbulence"].choices) == set(coupled["turbulence"].choices)


def test_turbulence_lists_cover_every_resolvable_scheme(amip):
    assert set(amip["turbulence"].choices) == set(_TURBULENCE_SCHEMES)


# --- the phantom -----------------------------------------------------------

def test_coupled_no_longer_offers_the_rejected_bulk_scheme(coupled):
    """`--surface-bulk-scheme most` parsed fine and then died at driver
    construction, contradicting the flag's own NOTE ("deliberately NOT offered
    here")."""
    assert "most" not in coupled["surface_bulk_scheme"].choices


def test_most_is_still_rejected_by_validate_strict():
    """Pins the fix DIRECTION. `most` must not be made valid instead:
    turbulence/surface_layer.py gates the MOST path on ("coare3",
    "large_yeager") only, so an accepted "most" would silently fall through to
    the constant-coefficient branch -- trading a loud crash for wrong physics.
    """
    with pytest.raises(ValueError, match="surface_bulk_scheme"):
        ExperimentConfig(surface_bulk_scheme="most").validate_strict()


@pytest.mark.parametrize("scheme", ["constant", "coare3", "large_yeager"])
def test_every_offered_bulk_scheme_passes_validate_strict(coupled, scheme):
    """No offered value may be a phantom.

    The MOST schemes carry a documented CROSS-FIELD contract -- they require
    turbulence != 'none', so the atmosphere surface layer runs the same
    bulk-flux algorithm as the ocean tile -- so satisfy it here. That contract
    is the point: unlike 'most', these are rejected only in a combination the
    config explains, never on their own.
    """
    assert scheme in coupled["surface_bulk_scheme"].choices
    ExperimentConfig(
        surface_bulk_scheme=scheme, turbulence="louis"
    ).validate_strict()


def test_most_is_rejected_even_with_a_turbulence_scheme():
    """Distinguishes 'most' from the MOST schemes above: it is rejected on the
    VALUE, not on a cross-field combination, so no configuration rescues it."""
    with pytest.raises(ValueError, match="surface_bulk_scheme"):
        ExperimentConfig(
            surface_bulk_scheme="most", turbulence="louis"
        ).validate_strict()


# --- protective exclusions must STAY excluded ------------------------------

def test_amip_still_excludes_the_untrained_ml_emulator(amip):
    """ml_emulator returns an untrained network's ~zero tendencies while the
    AMIP full-physics guard would count it ACTIVE -- i.e. silently disabled
    microphysics that passes the guard. Excluding it is protective, NOT drift.
    """
    assert "ml_emulator" not in amip["microphysics"].choices


def test_amip_still_excludes_the_crm_only_cloud_scheme(amip):
    """'resolved' is SAM's CRM convention and raises without explicit
    q_cloud/q_ice; it is not a C48 GCM scheme."""
    assert "resolved" not in amip["clouds"].choices
