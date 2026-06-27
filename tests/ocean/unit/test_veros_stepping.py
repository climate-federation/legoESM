"""Unit tests for the shared Veros-faithful stepping composition (#433).

Pins the exact bundle so the 5 Veros fidelity recipes can splat it instead of
each re-spelling it (the duplication that let acc_basic silently diverge and
leak ~+1.8 GW of spurious KE before it was caught).
"""

from __future__ import annotations

from legoesm.ocean.fidelity.veros_stepping import veros_faithful_stepping
from legoesm.ocean.state import LatLonCGridOceanConfig

# The seven fields the bundle owns.
_FIELDS = (
    "outer_integrator", "dt_mom_ratio", "barotropic_solver",
    "coriolis_scheme", "ab2_scope", "momentum_friction_additive",
    "implicit_vmix_dzw_slot",
)


def test_faithful_branch_exact():
    """Free-run path → the Veros AB2 / rigid-lid / explicit-AB2 bundle."""
    assert veros_faithful_stepping(with_surface_forcing=True, dt_mom_ratio=9.0) == {
        "outer_integrator": "ab2",
        "dt_mom_ratio": 9.0,
        "barotropic_solver": "rigid_lid",
        "coriolis_scheme": "explicit_ab2",
        "ab2_scope": "advective",
        "momentum_friction_additive": True,
        "implicit_vmix_dzw_slot": True,
    }


def test_probe_branch_is_legoesm_defaults():
    """Frozen-state tendency-probe path → legoESM defaults (bit-identical),
    and dt_mom_ratio is forced to 1.0 regardless of the argument."""
    d = veros_faithful_stepping(with_surface_forcing=False, dt_mom_ratio=48.0)
    assert d == {
        "outer_integrator": "forward_euler",
        "dt_mom_ratio": 1.0,
        "barotropic_solver": "explicit_substep",
        "coriolis_scheme": "matsuno_split",
        "ab2_scope": "total",
        "momentum_friction_additive": False,
        "implicit_vmix_dzw_slot": False,
    }


def test_probe_branch_matches_model_defaults():
    """The probe branch must equal the config's own defaults — that is what
    keeps the frozen-state probe bit-identical to a no-stepping-bundle build."""
    default = LatLonCGridOceanConfig.from_flat()
    probe = veros_faithful_stepping(with_surface_forcing=False, dt_mom_ratio=9.0)
    for f in _FIELDS:
        assert probe[f] == getattr(default, f), f


def test_dt_mom_ratio_passthrough():
    for ratio in (48.0, 1.0, 8.0):
        assert veros_faithful_stepping(
            with_surface_forcing=True, dt_mom_ratio=ratio)["dt_mom_ratio"] == ratio


def test_splats_into_config_without_collision():
    """The bundle must be splattable into the real config constructor."""
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0,
        **veros_faithful_stepping(with_surface_forcing=True, dt_mom_ratio=9.0),
    )
    assert cfg.outer_integrator == "ab2"
    assert cfg.barotropic.barotropic_solver == "rigid_lid"
    assert cfg.dt_mom_ratio == 9.0
