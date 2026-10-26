"""Tier-1 follow-ups to the CAM6 ``gw_oro`` port (``mcfarlane``), rows
59/60/62/63/65 of the oracle-audit asks (user decisions 2026-10-20).

Oracle: ESCOMP/CAM cam_cesm2_1_rel_60 (a03b84b7c4e34f965b115686f22a043b85739e56).

* 59 critical level: ``(ubmc > 0) .eqv. (ubi(k+1) > c)`` with ``c = 0``
  (gw_common.F90:485) -> ``crit_level_floor`` default 0.
* 60 every carried constituent is diffused (gw_common.F90:673-678 loops over
  all of ``q``; gw_drag.F90:1364 ``lq = .true.``) on the FV pipeline and the
  spectral lane too.
* 62 a zero model top raises (``log(pint)``, gw_common.F90:508).
* 63 a missing land fraction raises (CAM6 ``gw_lndscl_sgh = .true.``,
  gw_drag.F90:188/1757-1771).
* 65 the diffusion matrix uses the unfloored midpoint spacing
  (Coords1D ``rdst = 1/dst``, src/utils/coords_1d.F90:81-83).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
    gw_diff_tridiag_coeffs,
)
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd

FIXTURE = Path(__file__).resolve().parents[2] / "oracle_fixtures" / "gw_oro_cam6.npz"
_KEYS = ("u", "v", "T", "p_full", "p_half", "z_full", "z_half", "rho", "lat",
         "sgh", "lf", "q", "p_ref_half")


def _fx(dtype=np.float64):
    f = np.load(FIXTURE)
    return {k: f[k].astype(dtype) for k in _KEYS}, float(f["dt"])


def _call(f, dt, cfg=None, **over):
    kw = dict(h_topo_col=f["sgh"], p_ref_half=f["p_ref_half"],
              land_frac_col=f["lf"], tracers={"q_v": f["q"]})
    kw.update(over)
    return mcfarlane_gwd(
        f["u"], f["v"], f["T"], f["p_full"], f["p_half"], f["z_full"],
        f["z_half"], f["rho"], f["lat"], dt, cfg or McFarlaneConfig(), **kw)


def test_crit_level_floor_default_is_cam_phase_speed():
    assert McFarlaneConfig().crit_level_floor == 0.0


def test_missing_land_fraction_raises():
    f, dt = _fx()
    with pytest.raises(ValueError, match="land_frac_col"):
        _call(f, dt, land_frac_col=None)


@pytest.mark.parametrize("which", ["p_ref_half", "p_half"])
def test_zero_model_top_raises_eager(which):
    f, dt = _fx()
    f[which] = f[which].copy()
    f[which][..., 0] = 0.0
    with pytest.raises(ValueError, match="model-top"):
        _call(f, dt)


def test_zero_model_top_raises_under_jit():
    """The factories close over the coordinate, so p_ref_half is concrete
    inside the traced step and the check still fires."""
    f, dt = _fx()
    p_ref = f["p_ref_half"].copy()
    p_ref[0] = 0.0
    f = {k: jnp.asarray(v) for k, v in f.items()}
    p_ref = jnp.asarray(p_ref)          # closed over: concrete in the trace

    @jax.jit
    def step(u):
        return _call({**f, "u": u}, dt, p_ref_half=p_ref).du_dt

    with pytest.raises(ValueError, match="model-top p_ref_half"):
        step(jnp.asarray(f["u"]))


def test_mcfarlane_inputs_reference_interfaces_concrete_under_jit():
    """``mcfarlane_inputs`` (hydrostatic / MPAS / spectral factories) builds
    p_ref_half at trace time from a closed-over coordinate."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        mcfarlane_inputs,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(8)
    seen = {}

    @jax.jit
    def go(x):
        kw = mcfarlane_inputs("mcfarlane", sigma, None, 1, 8, None, x.dtype)
        seen["tracer"] = isinstance(kw["p_ref_half"], jax.core.Tracer)
        return x

    go(jnp.zeros(()))
    assert seen["tracer"] is False


def test_diffusion_matrix_uses_unfloored_midpoint_spacing():
    """A 0.5 Pa midpoint spacing enters as 1/0.5 (CAM rdst), not 1/1."""
    pmid = jnp.asarray([[100.0, 100.5, 300.0]])
    rhoi = jnp.ones((1, 4))
    kv = jnp.asarray([[0.0, 1.0, 1.0, 0.0]])
    rdpm = jnp.ones((1, 3))
    dt, g = 600.0, constants.g
    a, b, c = gw_diff_tridiag_coeffs(kv, rhoi, pmid, rdpm, dt, g, 0, 3)
    # c[0] = -kv[1]*tmpi[1]*rdpm[0], tmpi[1] = dt*(g*rhoi)^2/(pmid1-pmid0)
    np.testing.assert_allclose(float(c[0, 0]), -dt * g * g / 0.5, rtol=1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_gradients_finite_at_production_smoothing(dtype):
    if dtype == np.float64 and not jax.config.jax_enable_x64:
        pytest.skip("fp64 gradient needs JAX_ENABLE_X64=1")
    f, dt = _fx(dtype)
    f = {k: jnp.asarray(v) for k, v in f.items()}

    def loss(u, T):
        out = _call({**f, "u": u, "T": T}, dt)
        return (jnp.sum(out.du_dt ** 2) + jnp.sum(out.dT_dt ** 2)
                + jnp.sum(out.dq_dt["q_v"] ** 2))

    gu, gT = jax.grad(loss, argnums=(0, 1))(f["u"], f["T"])
    assert gu.dtype == dtype
    assert bool(jnp.all(jnp.isfinite(gu))) and bool(jnp.all(jnp.isfinite(gT)))
    assert float(jnp.abs(gu).max()) > 0.0


# --------------------------------------------------------------------------
# 60: constituent diffusion on the FV pipeline and the spectral lane
# --------------------------------------------------------------------------

def test_fv_pipeline_diffuses_every_carried_constituent():
    """physics_step_no_rad hands the kernel every carried constituent and
    adds the returned diffusion tendencies to the constituent totals."""
    from legoesm.atmosphere.physics.gravity_wave_drag import GWDOutput
    from tests.unit.test_gwd_landfrac_wiring import NLAT, NLON, _pipe, _run_step

    grid, pipe = _pipe("mcfarlane")
    pipe.f_land = jnp.ones(grid.grid_lat.shape)
    rec = {}

    def stub(dq):
        def kernel(**kw):
            rec.update(kw)
            z = jnp.zeros(kw["u"].shape)
            return GWDOutput(
                du_dt=z, dv_dt=z, dT_dt=z, eps_gwd=jnp.zeros(z.shape[0]),
                dq_dt=None if dq is None else {
                    k: jnp.full(z.shape, dq) for k in kw["tracers"]})
        return kernel

    pipe.gwd_fn = stub(None)
    base = _run_step(grid, pipe)
    pipe.gwd_fn = stub(1.0e-3)
    out = _run_step(grid, pipe)
    assert set(rec["tracers"]) == {"q_v", "q_c", "q_r"}
    assert rec["tracers"]["q_v"].shape == (NLAT * NLON, rec["u"].shape[1])
    for k in ("dq_v_dt", "dq_c_dt", "dq_r_dt"):
        np.testing.assert_allclose(
            np.asarray(getattr(out, k)) - np.asarray(getattr(base, k)),
            1.0e-3, rtol=0, atol=1e-15)


def test_spectral_lane_returns_constituent_diffusion():
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        make_gwd_physics,
    )
    from legoesm.core.field import Field
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    nlev = 8
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(nlev)
    shp = (grid.n_lat, grid.n_lon)
    grid = grid._replace(subgrid_topo_stddev=jnp.full(shp, 400.0),
                         land_frac=jnp.ones(shp[0] * shp[1]))
    q = jnp.broadcast_to(jnp.linspace(1e-5, 1.5e-2, nlev), (*shp, nlev))
    state = isothermal_rest_state_spectral(
        grid, sigma, tracers={"q_v": Field(data=q, name="q_v"),
                              "q_c": Field(data=2.0 * q, name="q_c")})
    T = jnp.broadcast_to(jnp.linspace(210.0, 290.0, nlev), (*shp, nlev))
    fields = {"u": jnp.full((*shp, nlev), 25.0), "v": jnp.zeros((*shp, nlev)),
              "T": T, "p_s": jnp.full(shp, constants.p_ref)}
    fn = make_gwd_physics(GravityWaveDragConfig(scheme="mcfarlane"),
                          model_type="spectral_pe", dt=900.0)
    tend, _ = fn(state, grid, sigma, grid_fields=fields)
    dq = np.asarray(tend.tracers["q_v"].data)
    assert dq.shape == q.shape and np.abs(dq).max() > 0.0
    # one diffusion operator for every constituent (CAM gw_diff_tend per m):
    # a tracer twice q_v gets exactly twice the tendency
    np.testing.assert_array_equal(np.asarray(tend.tracers["q_c"].data), 2.0 * dq)


# --------------------------------------------------------------------------
# lane proof: the model_type="mpas" factory is what the MPAS _run_column lane
# (primitive_eq_mpas._step_jit, static self) and the FV3-duo column lane
# (make_physics(model_type="mpas"), jit closing over self) both call.  The
# mesh and coordinate are closed over, as in those steps.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mpas_lane():
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    coord = make_cam6_l32_levels()
    state = held_suarez_init_mpas(mesh, coord, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    rng = np.random.default_rng(0)
    q = jnp.broadcast_to(jnp.linspace(1e-5, 1.5e-2, nlev), (ncell, nlev))
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape) * 8.0)),
        T=state.T.replace(data=jnp.asarray(
            290.0 - 70.0 * np.linspace(1.0, 0.0, nlev)[None, :]
            + rng.standard_normal((ncell, nlev)))),
        tracers={nm: state.T.replace(data=val, name=nm, units="kg/kg")
                 for nm, val in (("q_v", q), ("q_c", 2.0 * q))})
    mesh = mesh._replace(land_frac=jnp.ones(ncell),
                         subgrid_topo_stddev=jnp.full(ncell, 400.0))
    return mesh, coord, state


def test_mpas_column_lane_diffuses_tracers_under_jit(mpas_lane):
    from legoesm.atmosphere.physics.gravity_wave_drag import integration as gi
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    mesh, coord, state = mpas_lane
    fn = gi._make_mpas_gwd(GravityWaveDragConfig(scheme="mcfarlane"), 300.0)
    out = jax.jit(lambda st: fn(st, mesh, coord))(state)
    tend = out[0] if isinstance(out, tuple) else out
    tt = tend.tracer_tendencies
    dq = np.asarray(tt["q_v"].data)
    assert np.abs(dq).max() > 0.0
    np.testing.assert_array_equal(np.asarray(tt["q_c"].data), 2.0 * dq)


def test_mpas_column_lane_zero_top_raises_under_jit(mpas_lane):
    from legoesm.atmosphere.physics.gravity_wave_drag import integration as gi
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    mesh, coord, state = mpas_lane
    coord0 = coord._replace(A_half=coord.A_half.at[0].set(0.0))
    fn = gi._make_mpas_gwd(GravityWaveDragConfig(scheme="mcfarlane"), 300.0)
    with pytest.raises(ValueError, match="model-top"):
        jax.jit(lambda st: fn(st, mesh, coord0))(state)
