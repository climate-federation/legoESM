"""The MPAS physics bridges accept a COLUMN model that carries cell winds.

A state with ``v`` present (the FV3 duo column view; MPAS carries edge-
normal ``u`` and ``v=None``) hands its geographic cell winds straight to
the turbulence and gravity-wave-drag bridges and gets BOTH cell tendency
components back, instead of the Perot edge reconstruction / edge-normal
projection pair.

Identity (non-vacuous): on a real Voronoi mesh, hand the column path
exactly the Perot reconstruction of the MPAS state's edge winds; its cell
tendencies, projected with the SAME edge projection the MPAS path uses,
equal the MPAS path's edge tendency bitwise, and its T tendency equals
the MPAS path's bitwise. The MPAS path itself is byte-identical to
before (the new branch is an ``if state.v is not None``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.gravity_wave_drag import (  # noqa: E402
    integration as gwd_int,
)
from legoesm.atmosphere.physics.gravity_wave_drag.config import (  # noqa: E402
    GravityWaveDragConfig,
)
from legoesm.atmosphere.physics.turbulence import integration as turb_int  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.grids.vertical import make_cam6_l32_levels  # noqa: E402
from legoesm.grids.voronoi import (  # noqa: E402
    cell_vector_to_edge_normal,
    reconstruct_cell_velocity,
)

DT = 300.0


@pytest.fixture(scope="module")
def setup():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    coord = make_cam6_l32_levels()
    state = held_suarez_init_mpas(mesh, coord, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    rng = np.random.default_rng(0)
    # a sheared, non-trivial wind so the bridges have work to do
    u_edge = jnp.asarray(rng.standard_normal(state.u.data.shape) * 8.0)
    T = jnp.asarray(290.0 - 40.0 * np.linspace(1.0, 0.0, nlev)[None, :]
                    + rng.standard_normal((ncell, nlev)))
    state = state._replace(
        u=state.u.replace(data=u_edge),
        T=state.T.replace(data=T),
        # Field-wrapped tracers, as the MPAS driver carries them
        tracers={nm: state.T.replace(data=val, name=nm, units="kg/kg")
                 for nm, val in (("q_v", jnp.full((ncell, nlev), 0.004)),
                                 ("q_c", jnp.zeros((ncell, nlev))),
                                 ("q_r", jnp.zeros((ncell, nlev))))})
    u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)
    col_state = state._replace(
        u=state.u.replace(data=u_cell, name="u"),
        v=state.u.replace(data=v_cell, name="v"))
    return mesh, coord, state, col_state


def _forcing(state):
    ncell = state.T.data.shape[0]
    return {"T_sfc": jnp.full((ncell,), 292.0)}


def test_turbulence_column_winds_match_the_edge_path(setup):
    mesh, coord, state, col_state = setup
    fn = turb_int._make_mpas_turbulence(TurbulenceConfig(scheme="louis"), DT)
    edge = fn(state, mesh, coord, forcing=_forcing(state))
    col = fn(col_state, mesh, coord, forcing=_forcing(col_state))
    edge_t = edge[0] if isinstance(edge, tuple) else edge
    col_t = col[0] if isinstance(col, tuple) else col
    assert edge_t.dv_dt is None                       # MPAS path unchanged
    assert col_t.dv_dt is not None
    assert col_t.du_dt.data.shape == state.T.data.shape
    assert col_t.dv_dt.data.shape == state.T.data.shape
    assert float(jnp.abs(col_t.dv_dt.data).max()) > 0.0, "vacuous: no wind tendency"
    projected = cell_vector_to_edge_normal(col_t.du_dt.data, col_t.dv_dt.data, mesh)
    np.testing.assert_array_equal(np.asarray(projected),
                                  np.asarray(edge_t.du_dt.data))
    np.testing.assert_array_equal(np.asarray(col_t.dT_dt.data),
                                  np.asarray(edge_t.dT_dt.data))


def test_gwd_column_winds_match_the_edge_path(setup):
    mesh, coord, state, col_state = setup
    fn = gwd_int._make_mpas_gwd(GravityWaveDragConfig(scheme="mcfarlane"), DT)
    edge = fn(state, mesh, coord)
    col = fn(col_state, mesh, coord)
    edge_t = edge[0] if isinstance(edge, tuple) else edge
    col_t = col[0] if isinstance(col, tuple) else col
    assert edge_t.dv_dt is None
    assert col_t.dv_dt is not None
    assert float(jnp.abs(col_t.du_dt.data).max()) > 0.0, "vacuous: no drag"
    # the GWD MPAS bridge projects with a plain two-cell average (its own
    # code, not cell_vector_to_edge_normal): reproduce that projection
    c0, c1 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    du, dv = col_t.du_dt.data, col_t.dv_dt.data
    angle = mesh.angleEdge[:, None]
    projected = (0.5 * (du[c0] + du[c1]) * jnp.cos(angle)
                 + 0.5 * (dv[c0] + dv[c1]) * jnp.sin(angle))
    np.testing.assert_array_equal(np.asarray(projected),
                                  np.asarray(edge_t.du_dt.data))
    np.testing.assert_array_equal(np.asarray(col_t.dT_dt.data),
                                  np.asarray(edge_t.dT_dt.data))


def test_frontal_source_refused_without_mesh_topology(setup):
    from legoesm.atmosphere.physics.gravity_wave_drag.config import E3SMCAMConfig
    mesh, coord, state, col_state = setup

    class _Columns:   # lat/lon only, no edge topology (the duo view)
        latCell = mesh.latCell
        lonCell = mesh.lonCell
        areaCell = mesh.areaCell
        nCells = mesh.nCells

    ncell, nlev = state.T.data.shape
    cfg = GravityWaveDragConfig(
        scheme="e3sm_cam", e3sm_cam=E3SMCAMConfig(source="frontal"))
    with pytest.raises(ValueError, match="frontal source"):
        gwd_int._e3sm_source_kwargs(  # noqa: SLF001
            cfg, _Columns(), ncell, nlev, None, None,
            col_state.u.data, col_state.v.data, state.T.data,
            coord.pressure_at_full(state.p_s.data))


def test_combined_keeps_the_meridional_tendency_behind_radiation(setup):
    """The combined accumulator carries the turbulence dv_dt of a column
    model through to the combined tendency (radiation runs first and
    supplies the zero dv_dt seed when the state carries v), and still
    returns dv_dt=None for the MPAS edge-wind state."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    mesh, coord, state, col_state = setup
    cfg = PhysicsConfig(turbulence=TurbulenceConfig(scheme="louis"))
    fn = make_physics(cfg, model_type="mpas", dt=DT)
    out = fn(col_state, mesh, coord, forcing=_forcing(col_state))
    comb = out[0] if isinstance(out, tuple) else out
    _d = lambda x: getattr(x, "data", x)  # noqa: E731  (Field or raw array)
    assert comb.dv_dt is not None, "meridional tendency dropped"
    assert _d(comb.dv_dt).shape == state.T.data.shape
    assert float(jnp.abs(_d(comb.dv_dt)).max()) > 0.0
    turb = turb_int._make_mpas_turbulence(TurbulenceConfig(scheme="louis"), DT)(
        col_state, mesh, coord, forcing=_forcing(col_state))
    turb_t = turb[0] if isinstance(turb, tuple) else turb
    np.testing.assert_array_equal(np.asarray(_d(comb.dv_dt)),
                                  np.asarray(turb_t.dv_dt.data))
    # the MPAS state itself is untouched by the seeding rule
    out_e = fn(state, mesh, coord, forcing=_forcing(state))
    comb_e = out_e[0] if isinstance(out_e, tuple) else out_e
    assert comb_e.dv_dt is None


def _rotated(col_state, theta):
    """Cell winds rotated by theta (east/north components)."""
    u, v = col_state.u.data, col_state.v.data
    c, s_ = np.cos(theta), np.sin(theta)
    return col_state._replace(u=col_state.u.replace(data=c * u - s_ * v),
                              v=col_state.v.replace(data=s_ * u + c * v))


def test_column_turbulence_dissipates_and_is_rotation_covariant(setup):
    """The meridional component is live for the first time on this path
    (MPAS discarded it): Louis drag must remove kinetic energy from the
    column (u*du + v*dv <= 0 summed over the column) and rotating the
    input winds by theta must rotate the tendency vector by theta
    (isotropic closure)."""
    mesh, coord, state, col_state = setup
    fn = turb_int._make_mpas_turbulence(TurbulenceConfig(scheme="louis"), DT)
    t = fn(col_state, mesh, coord, forcing=_forcing(col_state))
    t = t[0] if isinstance(t, tuple) else t
    u, v = np.asarray(col_state.u.data), np.asarray(col_state.v.data)
    du, dv = np.asarray(t.du_dt.data), np.asarray(t.dv_dt.data)
    work = (u * du + v * dv).sum(axis=1)                     # per column
    assert work.max() <= 1e-12 * np.abs(u * du).sum(axis=1).max()
    assert np.abs(dv).max() > 0.0
    theta = 0.7
    rot = _rotated(col_state, theta)
    tr = fn(rot, mesh, coord, forcing=_forcing(rot))
    tr = tr[0] if isinstance(tr, tuple) else tr
    c, s_ = np.cos(theta), np.sin(theta)
    np.testing.assert_allclose(np.asarray(tr.du_dt.data), c * du - s_ * dv,
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.asarray(tr.dv_dt.data), s_ * du + c * dv,
                               rtol=1e-10, atol=1e-14)


def test_column_gwd_dissipates(setup):
    """McFarlane orographic drag on the column path removes kinetic
    energy from every column with nonzero drag (both components)."""
    mesh, coord, state, col_state = setup
    fn = gwd_int._make_mpas_gwd(GravityWaveDragConfig(scheme="mcfarlane"), DT)
    t = fn(col_state, mesh, coord)
    t = t[0] if isinstance(t, tuple) else t
    u, v = np.asarray(col_state.u.data), np.asarray(col_state.v.data)
    du, dv = np.asarray(t.du_dt.data), np.asarray(t.dv_dt.data)
    work = (u * du + v * dv).sum(axis=1)
    assert np.abs(dv).max() > 0.0
    assert work.max() <= 1e-12 * np.abs(u * du).sum(axis=1).max()
