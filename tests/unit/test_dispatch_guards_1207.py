"""Runtime guard tests for the dispatch hardening added in PR #1207.

`test_dispatch_hardening` statically proves the unknown-value ``raise``
exists; these tests EXERCISE each guard so a future refactor that drops the
raise (or lets the two LE_module/turbulence allowlists drift) goes red.

Covered:
- slab ``step_land`` rejects a surface_scheme that is neither SimpleSEB nor
  TwoLeafCanopy (CLMMLCanopyConfig would otherwise silently run SimpleSEB).
- ``validate_plane_config`` rejects an unknown turbulence_closure and accepts
  the valid set (incl. ``none``, the inviscid mode #1207 restored).
- LE_module rejected at BOTH ends (CanopyConfig.validate + the
  solve_canopy_closure entry guard) and the two allowlists share one source.
"""
from __future__ import annotations

import pytest


# --- slab surface_scheme dispatch -----------------------------------------

def test_slab_step_land_rejects_unknown_surface_scheme():
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land

    # CLMMLCanopyConfig is multilayer-only; the slab step must NOT silently
    # fall through to SimpleSEB. state/forcing are unused before the guard.
    cfg = LandConfig(surface_scheme=CLMMLCanopyConfig())
    with pytest.raises(ValueError, match="surface_scheme"):
        step_land(None, None, cfg, U_min=1.0, dt=1.0)


# --- plane turbulence_closure dispatch ------------------------------------

def _plane_cfg(closure):
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    # smagorinsky_cs=0 skips the Prandtl>0 check, isolating the closure guard.
    return CompressibleEulerConfig(
        smagorinsky_cs=0.0, smagorinsky_prandtl=1.0,
        turbulence_closure=closure,
    )


def test_turbulence_closure_rejects_unknown():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        validate_plane_config,
    )
    with pytest.raises(ValueError, match="turbulence_closure"):
        validate_plane_config(_plane_cfg("bogus"))


# "none"/"smagorinsky" need no extra coeffs; molecular/vreman/amd have their own
# downstream validator checks (viscosity/coeff > 0), so they are covered by the
# set-pin below rather than a full validate pass here.
@pytest.mark.parametrize("closure", ["none", "smagorinsky"])
def test_turbulence_closure_accepts_valid(closure):
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        validate_plane_config,
    )
    # Must NOT raise. "none" is the intentional inviscid mode restored in #1207.
    validate_plane_config(_plane_cfg(closure))


def test_turbulence_closure_valid_set_shared_and_pinned():
    """The validator and the slow_tendencies hot-path guard read ONE module
    constant (no drift); pin it so dropping a valid closure — e.g. re-losing the
    ``none`` inviscid mode — goes red."""
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as cep
    assert cep._VALID_TURBULENCE_CLOSURES == (
        "smagorinsky", "molecular", "none", "vreman", "amd")


# --- canopy LE_module dispatch (both ends + no drift) ---------------------

def test_le_module_config_validate_rejects_unknown():
    from legoesm.land.canopy.config import CanopyConfig
    with pytest.raises(ValueError, match="LE_module"):
        CanopyConfig(LE_module="bogus").validate()


def test_le_module_solver_entry_rejects_unknown():
    from legoesm.land.canopy.config import CanopyConfig
    from legoesm.land.canopy.solver import solve_canopy_closure

    # The guard is the first statement in solve_canopy_closure, before the
    # initial_state/bundle are touched — dummies reach it fine.
    with pytest.raises(ValueError, match="LE_module"):
        solve_canopy_closure(None, None, CanopyConfig(LE_module="bogus"))


def test_le_module_allowlist_is_single_source():
    """Solver imports the config allowlist (no hard-coded literal copy that can
    drift from the fail-early validator)."""
    from legoesm.land.canopy import solver
    from legoesm.land.canopy.config import VALID_LE_MODULES

    assert solver.VALID_LE_MODULES is VALID_LE_MODULES
    assert set(VALID_LE_MODULES) == {"BT", "PM"}


def test_le_module_solver_guard_references_shared_constant():
    """The binding-identity check above passes even if the guard *body* used a
    re-hardcoded literal — pin the membership test to the shared constant so
    reintroducing ``not in ("BT", "PM")`` in the solver goes red."""
    import inspect

    from legoesm.land.canopy.solver import solve_canopy_closure
    assert "not in VALID_LE_MODULES" in inspect.getsource(solve_canopy_closure)


def test_turbulence_guards_reference_shared_constant():
    """Both the validator and the slow_tendencies hot-path guard must test
    membership against the shared _VALID_TURBULENCE_CLOSURES (not duplicated
    literals that can drift)."""
    import inspect

    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as cep
    for fn in (cep.validate_plane_config,
               cep.plane_compressible_euler_slow_tendencies):
        assert "not in _VALID_TURBULENCE_CLOSURES" in inspect.getsource(fn), (
            f"{fn.__name__} turbulence guard must use the shared constant")


def test_le_module_accepts_every_valid_value():
    from legoesm.land.canopy.config import VALID_LE_MODULES, CanopyConfig
    for m in VALID_LE_MODULES:
        CanopyConfig(LE_module=m).validate()  # must not raise
