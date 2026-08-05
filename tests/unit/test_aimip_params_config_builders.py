"""Every ``AIMIPClassicalParams.to_*_config`` builder must run on default params.

Regression guard for the #1417 half-completed removal: ``tiedtke_cape_threshold``
and ``sbm_CAPE_threshold`` were dropped from ``_TIEDTKE_TRAINABLE`` /
``_SBM_TRAINABLE`` (their triggers are AD-unreachable, so they were being
optimised with exactly zero gradient), but ``to_tiedtke_config`` and
``to_sbm_config`` kept reading them out of ``as_dict()``. Every caller of those
two builders therefore died with ``KeyError``, which took out four
``test_aimip_spatial`` tests on main.

The general form of that defect — a builder reading a key the trainable list no
longer publishes — is what these tests pin: each builder is invoked, and every
``d["..."]`` any builder reads is asserted to exist in ``as_dict()``.
"""

import inspect
import re

import pytest

from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
from legoesm.training.aimip_params import AIMIPClassicalParams

# Builders needing an argument that has no usable default from reflection.
_BUILDER_KWARGS: dict[str, dict] = {}


def _default_params() -> AIMIPClassicalParams:
    return AIMIPClassicalParams.from_defaults()


def _builder_names() -> list[str]:
    return sorted(
        name
        for name, _ in inspect.getmembers(AIMIPClassicalParams, inspect.isfunction)
        if name.startswith("to_") and name.endswith("_config")
    )


def test_there_are_config_builders_to_check():
    """Guard against the reflection silently finding nothing."""
    names = _builder_names()
    assert len(names) >= 8, names
    assert "to_tiedtke_config" in names
    assert "to_sbm_config" in names


@pytest.mark.parametrize("name", _builder_names())
def test_every_config_builder_runs_on_default_params(name):
    """No builder may read a knob the trainable list does not publish."""
    params = _default_params()
    builder = getattr(params, name)
    cfg = builder(**_BUILDER_KWARGS.get(name, {}))
    assert cfg is not None


def test_tiedtke_and_sbm_keep_their_published_cape_threshold_defaults():
    """#1417: the dropped knobs fall back to the scheme default, not to a guess.

    Reading the value (rather than only building the config) is what makes this
    fail if someone "fixes" the KeyError by injecting some other number.
    """
    params = _default_params()
    assert (
        float(params.to_tiedtke_config().cape_threshold)
        == float(TiedtkeConfig().cape_threshold)
    )
    published = params.as_dict()
    assert "tiedtke_cape_threshold" not in published
    assert "sbm_CAPE_threshold" not in published


def test_no_builder_reads_a_key_absent_from_as_dict():
    """Static form of the same guard: every ``d["..."]`` must be a live key."""
    published = set(_default_params().as_dict())
    missing: dict[str, list[str]] = {}
    for name in _builder_names():
        src = inspect.getsource(getattr(AIMIPClassicalParams, name))
        absent = [
            key
            for key in re.findall(r'd\[\s*"([^"]+)"\s*\]', src)
            if key not in published
        ]
        if absent:
            missing[name] = absent
    assert not missing, f"builders read keys absent from as_dict(): {missing}"
