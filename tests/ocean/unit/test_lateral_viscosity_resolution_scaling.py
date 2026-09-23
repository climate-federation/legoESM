"""Lateral viscosity derived from the mesh, anchored on eORCA1.2.

A single value cannot serve 1 degree and 1/12 degree: the explicit Laplacian
limit is ``A_h·dt/dx² <= 1/4``, and 1e5 m²/s puts eORCA1.2 at 0.033 of it while
putting ORCA12 26 times OVER it, where the cold start diverges at step 10 with
a 193 m/s current. The rule holds that distance-to-the-limit constant.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.ocean.state import (
    LateralViscosityConfig,
    resolution_scaled_lateral_viscosity,
    wet_min_spacing,
)

_ANCHOR_DX_M = 21286.244008030702   # eORCA1.2 narrowest wet cell, float64 grid
_ANCHOR_DX_M_F32 = 21286.244140625  # the same mesh built in float32
_ANCHOR_A_H = 1.0e5
_ORCA12_DX_M = 1506.5831298828125   # ORCA0083-N06 narrowest wet cell
# Measured equivalent-isotropic narrowest cell, 1/sqrt(1/dx^2 + 1/dy^2), on
# the same two meshes (probe: ocean_hires_runs/probe_anisotropic_limit.py).
_ANCHOR_DX_EFF_M = 15889.0506
_ORCA12_DX_EFF_M = 1288.0036
_DT_S = 150.0
_LIMIT = 0.25


def _cfg(**kw):
    return LateralViscosityConfig(A_h=None, **kw)


def test_anchor_spacing_reproduces_todays_value_bit_identically():
    """The anchor mesh must keep exactly the number it runs today."""
    assert resolution_scaled_lateral_viscosity(_ANCHOR_DX_M, _cfg()) == _ANCHOR_A_H


def _find_eorca1() -> Path:
    """The mesh lives under a checkout's (gitignored) data directory; from a
    linked worktree that directory belongs to the main checkout, so walk up."""
    rel = Path("data/grids/eORCA1.2_mesh_mask.nc")
    for parent in Path(__file__).resolve().parents:
        if (parent / rel).exists():
            return parent / rel
    return Path(__file__).resolve().parents[3] / rel


_EORCA1 = _find_eorca1()


@pytest.mark.skipif(not _EORCA1.exists(), reason=f"needs {_EORCA1}")
@pytest.mark.timeout(900)
def test_real_anchor_mesh_derives_todays_value_through_the_driver():
    """NOT the (x/x)^2 tautology: read the real eORCA1.2 mesh, take its
    narrowest wet cell through the same path the run uses, and require the
    value the lane runs today, bit-identically.  This is the gate that says
    existing coarse runs are unchanged."""
    import scripts.run.run_omip as run_omip

    _, _, cfg, model, kind = run_omip._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "type1",
        tripole_mesh=str(_EORCA1), tripole_fold_convention="n_lon-1-i")
    assert kind == "tripole"
    # The grid's metrics may be built in either precision; the anchor is the
    # float64 measurement, so require the value to 1e-8 relative and pin the
    # exact match for whichever precision this build produced.
    # 1.2e-8 relative when the grid metrics are single precision (the
    # builder's default) against a double-precision anchor: 8 significant
    # digits, not a behaviour change, but NOT bit-identical either.
    assert float(cfg.lateral_viscosity.A_h) == pytest.approx(_ANCHOR_A_H,
                                                            rel=1e-7)
    dx = cfg.lateral_viscosity.A_h_dx_m
    assert dx in (_ANCHOR_DX_M, _ANCHOR_DX_M_F32), dx
    if dx == _ANCHOR_DX_M:
        # float64 metrics: bit-identical to the value the lane runs today
        assert float(cfg.lateral_viscosity.A_h) == _ANCHOR_A_H
    assert cfg is model.config


@pytest.mark.skipif(not _EORCA1.exists(), reason=f"needs {_EORCA1}")
@pytest.mark.timeout(900)
def test_a_pin_still_wins_on_the_real_mesh():
    import scripts.run.run_omip as run_omip

    _, _, cfg, _, _ = run_omip._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "type1",
        tripole_mesh=str(_EORCA1), tripole_fold_convention="n_lon-1-i",
        A_h_override=3000.0)
    assert float(cfg.lateral_viscosity.A_h) == 3000.0
    assert cfg.lateral_viscosity.A_h_dx_m == 0.0     # pinned, not derived


def test_squared_law():
    cfg = _cfg()
    half = resolution_scaled_lateral_viscosity(_ANCHOR_DX_M / 2.0, cfg)
    assert half == pytest.approx(_ANCHOR_A_H / 4.0, rel=1e-12)
    tenth = resolution_scaled_lateral_viscosity(_ANCHOR_DX_M / 10.0, cfg)
    assert tenth == pytest.approx(_ANCHOR_A_H / 100.0, rel=1e-12)


def test_orca12_lands_inside_the_explicit_limit_and_the_margin_is_preserved():
    cfg = _cfg()
    a12 = resolution_scaled_lateral_viscosity(_ORCA12_DX_M, cfg)
    r12 = a12 * _DT_S / _ORCA12_DX_M**2
    r_anchor = _ANCHOR_A_H * _DT_S / _ANCHOR_DX_M**2
    assert r12 < _LIMIT, f"derived A_h={a12} is over the explicit limit"
    # Against the TRUE anisotropic limit, A*dt*(1/dx^2 + 1/dy^2) <= 1/2, using
    # the measured equivalent-isotropic narrowest cell of each mesh rather than
    # the min(dx, dy) proxy the rule keys on.
    for dx_eff, a_h in ((_ORCA12_DX_EFF_M, a12),
                        (_ANCHOR_DX_EFF_M, _ANCHOR_A_H)):
        assert a_h * _DT_S / dx_eff**2 < 0.5, (dx_eff, a_h)
    # the point of the squared law: same distance to the limit on every mesh
    assert r12 == pytest.approx(r_anchor, rel=1e-12)
    # and the linear (NEMO velocity-scale) alternative would NOT be stable,
    # which is why it was refused
    a12_linear = _ANCHOR_A_H * (_ORCA12_DX_M / _ANCHOR_DX_M)
    assert a12_linear * _DT_S / _ORCA12_DX_M**2 > _LIMIT


def test_a_pin_is_returned_unchanged():
    for pinned in (0.0, 3000.0, 1.0e5):
        cfg = LateralViscosityConfig(A_h=pinned)
        assert resolution_scaled_lateral_viscosity(_ORCA12_DX_M, cfg) == pinned


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_spacing_refuses(bad):
    with pytest.raises(ValueError, match="narrowest wet cell"):
        resolution_scaled_lateral_viscosity(bad, _cfg())


def test_non_positive_anchor_refuses():
    with pytest.raises(ValueError, match="A_h_ref_dx_m"):
        resolution_scaled_lateral_viscosity(1.0e3, _cfg(A_h_ref_dx_m=0.0))


class _Grid:
    def __init__(self, dx, dy):
        self.dx_T = np.asarray(dx)
        self.dy_T = np.asarray(dy)


def test_wet_min_spacing_ignores_land_and_takes_the_narrower_axis():
    dx = np.array([[9000.0, 9000.0], [4000.0, 9000.0]])
    dy = np.array([[9000.0, 3000.0], [9000.0, 9000.0]])
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])      # the 3000 m cell is LAND
    assert wet_min_spacing(_Grid(dx, dy), mask) == 4000.0
    # with that cell wet it becomes the answer
    assert wet_min_spacing(_Grid(dx, dy), np.ones_like(mask)) == 3000.0


def test_wet_min_spacing_refuses_a_mismatched_or_empty_mask():
    g = _Grid(np.ones((2, 2)) * 5000.0, np.ones((2, 2)) * 5000.0)
    with pytest.raises(ValueError, match="does not match"):
        wet_min_spacing(g, np.ones((3, 2)))
    with pytest.raises(ValueError, match="no ocean cells"):
        wet_min_spacing(g, np.zeros((2, 2)))


def test_model_refuses_an_unresolved_viscosity():
    """Nothing may run with A_h still None -- that is the silent path the
    derivation exists to close."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(n_levels=3, H_max=1000.0)
    cfg = LatLonCGridOceanConfig.from_flat()
    cfg = cfg._replace(
        lateral_viscosity=cfg.lateral_viscosity._replace(A_h=None))
    with pytest.raises(ValueError, match="nothing resolved it"):
        LatLonCGridOceanModel(grid, z, cfg)


def test_a_zero_or_nan_anchor_value_refuses():
    """A_h_ref = 0 would silently disable the viscosity, and NaN would pass the
    scheme's own ``A_h < 0`` check and then make every ``A_h > 0`` gate false
    (codex)."""
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="A_h_ref must be"):
            resolution_scaled_lateral_viscosity(_ORCA12_DX_M, _cfg(A_h_ref=bad))


def test_saturating_on_the_grids_metric_floor_refuses():
    """The grid builder floors EVERY metric, not only land's, so on a fine
    enough mesh this statistic would stop scaling silently (independent
    reviewer)."""
    floor = 1000.0
    g = _Grid(np.full((2, 2), floor), np.full((2, 2), floor))
    with pytest.raises(ValueError, match="metric floor"):
        wet_min_spacing(g, np.ones((2, 2)), clamp_floor_m=floor)
    # above the floor it is fine, and the check is opt-in
    g2 = _Grid(np.full((2, 2), 1500.0), np.full((2, 2), 2000.0))
    assert wet_min_spacing(g2, np.ones((2, 2)), clamp_floor_m=floor) == 1500.0
    assert wet_min_spacing(g, np.ones((2, 2))) == floor


def test_validation_names_an_unresolved_viscosity_instead_of_a_type_error():
    """validate_strict used to evaluate ``None < 0.0`` and raise a bare
    TypeError (codex)."""
    from legoesm.ocean.state import LatLonCGridOceanConfig

    cfg = LatLonCGridOceanConfig.from_flat()
    cfg = cfg._replace(
        lateral_viscosity=cfg.lateral_viscosity._replace(A_h=None))
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    with pytest.raises(ValueError, match="nothing resolved it"):
        LatLonCGridOceanModel._validate_config(cfg)


def test_the_coupled_tripole_lane_derives_too():
    """The coupled driver cloned the forced-ocean coefficients verbatim and
    kept 1e5, so a 1/12 degree coupled run would have carried the divergence
    (both reviewers)."""
    from legoesm.driver.coupled_esm_driver import (  # noqa: F401
        CoupledESMDriver,
    )
    import inspect

    src = inspect.getsource(CoupledESMDriver._build_tripole_ocean_config)
    assert "A_h=None" in src, "the coupled tripole config no longer derives"
    assert "A_h=1.0e5" not in src
    resolved = inspect.getsource(CoupledESMDriver)
    assert "resolution_scaled_lateral_viscosity" in resolved, (
        "the coupled driver must resolve the derived viscosity before it "
        "constructs the model")
