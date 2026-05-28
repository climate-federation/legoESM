"""Tests for the recipe-level constants override mechanism.

Verifies that ``override_constants`` (Phase G.0a item #2) propagates
through:

1. ``legoesm.constants.X`` itself.
2. Module-level snapshots that captured ``constants.X`` at import time
   (``legoesm.ocean.eos.rho_0``).
3. Fan-out aliased imports of ``eos.X`` in downstream ocean modules
   (e.g. ``surface_forcing.bulk_formulas.rho_0_ref``).
4. Restoration of original values on context exit, even when the body
   raises.

These are the load-bearing properties for Phase G tier-2 tendency
comparison: if any snapshot site fails to update, the comparison is
contaminated by 0.02-0.1% constant drift.
"""

from __future__ import annotations

import pytest

from legoesm import constants
from legoesm.ocean.eos import rho_0 as _eos_rho_0_at_import  # noqa: F401
from legoesm.ocean.fidelity.recipe_constants import (
    VEROS_CONSTANTS,
    UnknownConstantError,
    assert_registry_consistent,
    override_constants,
)


def test_registry_consistent_with_actual_modules():
    """``_CONSTANT_SHADOWS`` must reference real modules / attributes.
    Catches silent drift between the registry and the codebase."""
    errors = assert_registry_consistent()
    assert errors == [], "\n".join(errors)


def test_veros_constants_dict_keys_in_registry():
    """``VEROS_CONSTANTS`` must use only names known to the override
    mechanism; otherwise ``with override_constants(**VEROS_CONSTANTS):``
    would raise."""
    from legoesm.ocean.fidelity.recipe_constants import _CONSTANT_SHADOWS
    for name in VEROS_CONSTANTS:
        assert name in _CONSTANT_SHADOWS, (
            f"VEROS_CONSTANTS includes {name!r} but it is not in "
            f"_CONSTANT_SHADOWS — recipe load would fail."
        )


def test_override_updates_constants_module():
    """``constants.X`` is updated inside the with block and restored
    on exit."""
    original = constants.g
    with override_constants(g=9.81):
        assert constants.g == 9.81
    assert constants.g == original


def test_override_updates_eos_snapshot():
    """``legoesm.ocean.eos.rho_0`` was snapshotted at import time. The
    override must reach it."""
    import legoesm.ocean.eos as eos
    original_rho_0 = eos.rho_0
    with override_constants(rho_ocean=1024.0):
        assert eos.rho_0 == 1024.0
    assert eos.rho_0 == original_rho_0


def test_override_updates_fanout_aliases():
    """Modules that did ``from legoesm.ocean.eos import rho_0 as _RHO_0``
    captured a second-level snapshot. Override must reach those too."""
    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gm
    import legoesm.ocean.physics.surface_forcing.bulk_formulas as bf
    original_gm = gm._RHO_0
    original_bf = bf.rho_0_ref
    with override_constants(rho_ocean=1024.0):
        assert gm._RHO_0 == 1024.0
        assert bf.rho_0_ref == 1024.0
    assert gm._RHO_0 == original_gm
    assert bf.rho_0_ref == original_bf


def test_override_updates_g_shadow_in_coupler():
    """``legoesm.coupler.bulk_flux.G`` was snapshotted from
    ``constants.g``. Override must reach it."""
    import legoesm.coupler.bulk_flux as bf
    original_G = bf.G
    with override_constants(g=9.81):
        assert bf.G == 9.81
    assert bf.G == original_G


def test_unknown_constant_raises():
    """Overriding an unregistered constant raises before any mutation."""
    original_g = constants.g
    with pytest.raises(UnknownConstantError, match="not in the override registry"):
        with override_constants(definitely_not_a_real_constant=42.0):
            pass
    # No partial mutation.
    assert constants.g == original_g


def test_override_restores_on_exception():
    """When the body raises, originals are still restored."""
    import legoesm.ocean.eos as eos
    original_g = constants.g
    original_rho_0 = eos.rho_0
    with pytest.raises(RuntimeError):
        with override_constants(g=9.81, rho_ocean=1024.0):
            assert constants.g == 9.81
            assert eos.rho_0 == 1024.0
            raise RuntimeError("intentional")
    assert constants.g == original_g
    assert eos.rho_0 == original_rho_0


def test_veros_constants_full_override():
    """The ``with override_constants(**VEROS_CONSTANTS):`` pattern from
    the Phase G recipe loader workflow."""
    original_values = {
        "g": constants.g,
        "Omega": constants.Omega,
        "R_earth": constants.R_earth,
        "rho_ocean": constants.rho_ocean,
        "c_sw": constants.c_sw,
    }
    with override_constants(**VEROS_CONSTANTS):
        assert constants.g == VEROS_CONSTANTS["g"]
        assert constants.Omega == VEROS_CONSTANTS["Omega"]
        assert constants.R_earth == VEROS_CONSTANTS["R_earth"]
        assert constants.rho_ocean == VEROS_CONSTANTS["rho_ocean"]
        assert constants.c_sw == VEROS_CONSTANTS["c_sw"]
    for name, original in original_values.items():
        assert getattr(constants, name) == original


def test_empty_overrides_is_no_op():
    """``override_constants()`` with no kwargs is a no-op (used by
    recipes that don't override anything)."""
    original_g = constants.g
    with override_constants():
        assert constants.g == original_g
    assert constants.g == original_g
