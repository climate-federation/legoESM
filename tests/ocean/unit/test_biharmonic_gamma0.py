"""FESOM2's resolution-scaled biharmonic B = gamma0*h^3 on the tripole C-grid
(parity batch 2, item 4)."""
import sys
import types
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.state import LateralViscosityConfig  # noqa: E402
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    _validate_biharmonic_gamma0, biharmonic_gamma0_coefficients)

GAMMA0 = 0.003
N_LAT, N_LON = 5, 7


def _grid(scale=1.0, with_metrics=True):
    rng = np.random.default_rng(0)
    m = {}
    if with_metrics:
        m = dict(dx_u=scale * (2000.0 + 1000.0 * rng.random((N_LAT, N_LON + 1))),
                 dy_u=scale * (2000.0 + 1000.0 * rng.random((N_LAT, N_LON + 1))),
                 dx_v=scale * (2000.0 + 1000.0 * rng.random((N_LAT + 1, N_LON))),
                 dy_v=scale * (2000.0 + 1000.0 * rng.random((N_LAT + 1, N_LON))))
    return types.SimpleNamespace(**m)


def test_coefficient_formula_and_cubic_scaling():
    g = _grid()
    cu, cv = biharmonic_gamma0_coefficients(g, GAMMA0)
    np.testing.assert_allclose(cu, GAMMA0 * (g.dx_u * g.dy_u) ** 1.5, rtol=1e-12)
    np.testing.assert_allclose(cv, GAMMA0 * (g.dx_v * g.dy_v) ** 1.5, rtol=1e-12)
    cu2, cv2 = biharmonic_gamma0_coefficients(_grid(scale=2.0), GAMMA0)
    np.testing.assert_allclose(cu2, 8.0 * cu, rtol=1e-12)
    np.testing.assert_allclose(cv2, 8.0 * cv, rtol=1e-12)


@pytest.mark.parametrize("bad", [
    dict(A_h=0.0, B_h=1.0e3, B_h_gamma0=GAMMA0, B_h_lat_scaling=False),
    dict(A_h=0.0, B_h=0.0, B_h_gamma0=GAMMA0, B_h_lat_scaling=True),
    dict(A_h=1.0e4, B_h=0.0, B_h_gamma0=GAMMA0, B_h_lat_scaling=False),
])
def test_refusals(bad):
    with pytest.raises(ValueError):
        _validate_biharmonic_gamma0(LateralViscosityConfig(**bad), _grid())
    with pytest.raises(ValueError, match="dx_u"):
        _validate_biharmonic_gamma0(
            LateralViscosityConfig(A_h=0.0, B_h=0.0, B_h_gamma0=GAMMA0, B_h_lat_scaling=False),
            _grid(with_metrics=False))
    assert _validate_biharmonic_gamma0(
        LateralViscosityConfig(A_h=0.0, B_h=0.0, B_h_gamma0=GAMMA0, B_h_lat_scaling=False),
        _grid()) is None
    assert _validate_biharmonic_gamma0(LateralViscosityConfig(A_h=0.0, B_h=1e3), _grid(with_metrics=False)) is None


def test_tripole_lane_routes_gamma0_and_the_tendency_damps_a_checkerboard(tmp_path):
    """End to end on a synthetic tripole mesh: --B-h-gamma0 reaches the recipe
    (A_h 0, lat scaling off), and the momentum tendency of a grid-scale
    checkerboard is a damping whose magnitude scales as gamma0*h^3/h^4 = gamma0/h
    per unit amplitude; gamma0 = 0 gives no biharmonic term."""
    from scripts.run import run_omip as R
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import biharmonic_gamma0_coefficients as coefs
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh
    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)
    common = dict(tripole_mesh=str(mesh), tripole_fold_convention="(n_lon-i)%n_lon",
                  forcing_mode="jra55_do_tropical")
    grid, z, cfg_on, model_on, _ = R._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "II", A_h_override=0.0, C_smag_lap=0.0,
        B_h_gamma0=GAMMA0, **common)
    lv = cfg_on.lateral_viscosity
    assert (lv.A_h, lv.B_h, lv.B_h_gamma0, lv.B_h_lat_scaling, lv.C_smag_lap) == (0.0, 0.0, GAMMA0, False, 0.0)
    assert hasattr(grid, "dx_u")
    _, _, cfg_off, model_off, _ = R._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "II", A_h_override=0.0, C_smag_lap=0.0, **common)
    assert cfg_off.lateral_viscosity.B_h_gamma0 == 0.0 and cfg_off.lateral_viscosity.B_h_lat_scaling is True
    with pytest.raises(ValueError, match="requires A_h = 0"):
        R._create_setup("tripole", "eorca1", 3, 1000.0, "none", "II",
                        A_h_override=500.0, C_smag_lap=0.0, B_h_gamma0=GAMMA0, **common)
    # a lateral-viscosity-only tendency difference: checkerboard u, rest otherwise
    state = R._init_rest_state("tripole", grid, z, H_max=1000.0)
    u = state.u.data
    j = jnp.arange(u.shape[0])[:, None, None]; i = jnp.arange(u.shape[1])[None, :, None]
    cb = 0.1 * ((-1.0) ** (j + i)) * jnp.ones_like(u)
    state = state._replace(u=state.u.replace(data=cb * state.u_mask.data[..., None]))
    t_on = model_on.tendencies(state); t_off = model_off.tendencies(state)
    du = np.asarray(t_on.du_dt.data) - np.asarray(t_off.du_dt.data)
    interior = np.broadcast_to(np.asarray(state.u_mask.data)[..., None] > 0.5, du.shape)
    assert np.abs(du[interior]).max() > 0.0
    # damping: tendency opposes the checkerboard on the interior faces (sign check)
    corr = float(np.sum(du[interior] * np.asarray(state.u.data)[interior]))
    assert corr < 0.0
    cu, _ = coefs(grid, GAMMA0)
    assert float(np.asarray(cu).min()) > 0.0


def test_gamma0_term_is_dissipative_on_a_random_field(tmp_path):
    """Conservative two-stage form: sum over faces of u * tendency <= 0 for a
    random interior velocity, not only for a checkerboard (codex batch-2 P1)."""
    from scripts.run import run_omip as R
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh
    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)
    common = dict(tripole_mesh=str(mesh), tripole_fold_convention="(n_lon-i)%n_lon",
                  forcing_mode="jra55_do_tropical", A_h_override=0.0, C_smag_lap=0.0)
    grid, z, _, m_on, _ = R._create_setup("tripole", "eorca1", 3, 1000.0, "none", "II",
                                          B_h_gamma0=GAMMA0, **common)
    _, _, _, m_off, _ = R._create_setup("tripole", "eorca1", 3, 1000.0, "none", "II", **common)
    state = R._init_rest_state("tripole", grid, z, H_max=1000.0)
    rng = np.random.default_rng(7)
    u = jnp.asarray(0.1 * rng.standard_normal(state.u.data.shape)) * state.u_mask.data[..., None]
    v = jnp.asarray(0.1 * rng.standard_normal(state.v.data.shape)) * state.v_mask.data[..., None]
    state = state._replace(u=state.u.replace(data=u), v=state.v.replace(data=v))
    t_on = m_on.tendencies(state); t_off = m_off.tendencies(state)
    du = np.asarray(t_on.du_dt.data) - np.asarray(t_off.du_dt.data)
    dv = np.asarray(t_on.dv_dt.data) - np.asarray(t_off.dv_dt.data)
    ke_tend = float((np.asarray(u) * du).sum() + (np.asarray(v) * dv).sum())
    assert ke_tend < 0.0
    assert np.abs(du).max() > 0.0


def test_nemo_drag_law_is_zero_and_differentiable_at_rest():
    """FESOM2's C_d|u| with ke0 = 0: zero drag at rest and a finite gradient."""
    from legoesm.ocean.dynamics.ocean_tendency_common import nemo_drag_r_from_speed_sq
    f = lambda s2: nemo_drag_r_from_speed_sq(s2, jnp.asarray(50.0), scheme="nemo_quadratic",
                                             cd0=0.0025, cd_max=0.1, z0=3e-3, ke0=0.0, von_karman=0.4)
    assert float(f(jnp.asarray(0.0))) == 0.0
    g = jax.grad(lambda s2: jnp.sum(f(s2)))(jnp.asarray(0.0))
    assert np.isfinite(float(g))
    assert float(f(jnp.asarray(4.0))) == pytest.approx(0.0025 * 2.0)
