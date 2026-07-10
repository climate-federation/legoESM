"""QG-Leith harmonic viscosity (Bachman et al. 2017; Silvestri et al. 2024 "QG2").

Exercises ``qg_leith_viscosity_tendency_cgrid`` (the barotropic ν=(C·Δ/π)³·
√(|∇(ζ+f)|²+|∇δ|²) Laplacian closure) and the ``bound_qg_pv_gradient`` helper
(the Bu/Ro min bound, ready for the future buoyancy-stretching wiring):
energy dissipation, rest-zero, absolute-vs-relative vorticity distinction,
finiteness/AD, the bound math, and the config dispatch through a model step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _grid(n_lat=12, n_lon=24):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)


def _fields(grid, nlev=2, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, nlev)))
    v = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, nlev)))
    v = v.at[0].set(0.0).at[-1].set(0.0)
    return u, v


def test_rest_state_zero_tendency():
    """u=v=0 → ζ=0, δ=0; |∇(ζ+f)| from f alone is constant in lon so its h-point
    gradient magnitude is the β term only. With zero velocity the stress operator
    sees zero strain → zero tendency."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u = jnp.zeros((grid.n_lat, grid.n_lon + 1, 2))
    v = jnp.zeros((grid.n_lat + 1, grid.n_lon, 2))
    tu, tv = qg_leith_viscosity_tendency_cgrid(u, v, grid)
    assert float(jnp.max(jnp.abs(tu))) == 0.0
    assert float(jnp.max(jnp.abs(tv))) == 0.0


def test_finite_and_shapes():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid)
    tu, tv = qg_leith_viscosity_tendency_cgrid(u, v, grid)
    assert tu.shape == u.shape and tv.shape == v.shape
    assert bool(jnp.all(jnp.isfinite(tu))) and bool(jnp.all(jnp.isfinite(tv)))


def test_energy_dissipation():
    """QG-Leith removes kinetic energy (energy-stable stress-tensor operator)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=3)
    tu, tv = qg_leith_viscosity_tendency_cgrid(u, v, grid)
    area = np.asarray(grid.area)[..., None]
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)
    dE = float(np.sum(np.asarray(u) * np.asarray(tu) * A_u)
               + np.sum(np.asarray(v) * np.asarray(tv) * A_v))
    assert dE <= 1e-12, f"QG-Leith injected energy: dE/dt = {dE:.3e}"


def test_absolute_vorticity_floor():
    """QG-Leith uses Q=ζ+f, so for WEAK flow the planetary gradient |∇f|=β sets a
    velocity-INDEPENDENT viscosity floor ν≈(C·Δ/π)³·β. The tendency then scales
    ~LINEARLY with velocity (ν floored, strain ∝ u), unlike a pure-relative-
    vorticity Leith where ν∝|∇ζ|∝u gives a QUADRATIC tendency. Halving a weak u
    halves the tendency (ratio→2), not quarters it (ratio→4)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=5)
    eps = 1e-6                                   # weak flow → β-floor dominates ν
    tu1, _ = qg_leith_viscosity_tendency_cgrid(eps * u, eps * v, grid)
    tu2, _ = qg_leith_viscosity_tendency_cgrid(2 * eps * u, 2 * eps * v, grid)
    ratio = float(jnp.max(jnp.abs(tu2))) / (float(jnp.max(jnp.abs(tu1))) + 1e-300)
    assert 1.8 < ratio < 2.2, f"expected ~linear (f-floor), got ratio {ratio:.3f}"


def test_differentiable():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=7)

    def loss(u_):
        tu, _ = qg_leith_viscosity_tendency_cgrid(u_, v, grid)
        return jnp.sum(tu ** 2)

    g = jax.grad(loss)(u)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0


class TestBoundQGPVGradient:
    """The Bachman Bu/Ro min-bound helper (ready for the stretching wiring)."""

    def test_min_of_three(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import bound_qg_pv_gradient
        grad_q = jnp.array(1.0)
        # stretching makes the full gradient large; Bu/Ro bounds cap it.
        grad_q_stretch = jnp.array(10.0)
        Bu = jnp.array(1.0)        # gq2 = 1·(1+1) = 2
        Ro = jnp.array(1.0)        # gq3 = 1·(1+1) = 2
        out = float(bound_qg_pv_gradient(grad_q, grad_q_stretch, Bu, Ro))
        assert abs(out - 2.0) < 1e-12, out   # min(10, 2, 2) = 2 (bound active)

    def test_no_bound_when_stretch_small(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import bound_qg_pv_gradient
        grad_q = jnp.array(1.0)
        grad_q_stretch = jnp.array(1.0)        # no enhancement
        Bu = jnp.array(1e6)                     # gq2 ≈ 1
        Ro = jnp.array(1e6)                     # gq3 ≈ 1
        out = float(bound_qg_pv_gradient(grad_q, grad_q_stretch, Bu, Ro))
        assert abs(out - 1.0) < 1e-6, out       # min ≈ 1 (no bound)

    def test_array_elementwise(self):
        """The bound is elementwise over fields (the wired form takes 2D/3D
        gradient + Bu/Ro arrays)."""
        from legoesm.ocean.dynamics.latlon_cgrid_operators import bound_qg_pv_gradient
        grad_q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        stretch = jnp.array([[10.0, 2.0], [3.0, 100.0]])   # large at [0,0] and [1,1]
        Bu = jnp.array([[1.0, 1e6], [1e6, 1.0]])
        Ro = jnp.array([[1.0, 1e6], [1e6, 1.0]])
        out = np.asarray(bound_qg_pv_gradient(grad_q, stretch, Bu, Ro))
        # [0,0]: min(10, 1·2, 1·2)=2 ; [0,1]: min(2,2,2)=2 ; [1,0]: min(3,3,3)=3 ;
        # [1,1]: min(100, 4·2, 4·2)=8.
        assert np.allclose(out, [[2.0, 2.0], [3.0, 8.0]]), out


class TestB5bStretching:
    """B5b: the full QG2 baroclinic stretching term ∂_z(f/N²∇b) and its bound."""

    def test_ddz_centre_linear_and_constant(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import _ddz_centre
        nlev = 8
        h = jnp.full((3, 4, nlev), 20.0)                 # uniform 20 m
        # z (up) at centres: z_k = -(20*k + 10). A field linear in z: X = 2*z + 5.
        zc = -(20.0 * jnp.arange(nlev) + 10.0)
        X = 2.0 * zc[None, None, :] + 5.0 + jnp.zeros((3, 4, nlev))
        dXdz = _ddz_centre(X, h)
        assert jnp.allclose(dXdz, 2.0, atol=1e-9)         # ∂X/∂z = 2 everywhere
        # Constant field → zero derivative.
        assert jnp.allclose(_ddz_centre(jnp.full((3, 4, nlev), 7.0), h), 0.0, atol=1e-12)

    def test_stretching_zero_for_barotropic_buoyancy(self):
        """b depending only on z (no horizontal structure) ⇒ ∇b=0 ⇒ stretching=0."""
        from legoesm.ocean.dynamics.latlon_cgrid_operators import qg_pv_stretching_vec
        grid = _grid()
        nlev = 6
        zc = -(jnp.arange(nlev) * 20.0 + 10.0)
        b = (4e-6 * zc)[None, None, :] * jnp.ones((grid.n_lat, grid.n_lon, nlev))
        h = jnp.full((grid.n_lat, grid.n_lon, nlev), 20.0)
        f_h = (2 * constants.Omega * jnp.sin(jnp.asarray(np.radians(-50.0))))
        f_h = jnp.full((grid.n_lat, 1), float(f_h))
        sx, sy = qg_pv_stretching_vec(b, h, f_h, grid)
        assert float(jnp.max(jnp.abs(sx))) < 1e-15
        assert float(jnp.max(jnp.abs(sy))) < 1e-15

    def test_stretching_nonzero_for_a_front(self):
        """A horizontal buoyancy front (b varies in lat) gives a nonzero, finite
        stretching vector."""
        from legoesm.ocean.dynamics.latlon_cgrid_operators import qg_pv_stretching_vec
        grid = _grid()
        nlev = 6
        zc = -(np.arange(nlev) * 20.0 + 10.0)
        lat = np.asarray(grid.lat)
        # b = N² z + front(lat) · taper(z) — surface-intensified meridional front.
        front = np.tanh((lat - lat.mean()) / np.radians(5.0))[:, None, None]
        b = (4e-6 * zc)[None, None, :] + 5e-3 * front * np.exp(zc / 500.0)[None, None, :]
        b = jnp.asarray(np.broadcast_to(b, (grid.n_lat, grid.n_lon, nlev)).copy())
        h = jnp.full((grid.n_lat, grid.n_lon, nlev), 20.0)
        f_h = jnp.full((grid.n_lat, 1), float(2 * constants.Omega * np.sin(np.radians(-50.0))))
        sx, sy = qg_pv_stretching_vec(b, h, f_h, grid)
        assert bool(jnp.all(jnp.isfinite(sx))) and bool(jnp.all(jnp.isfinite(sy)))
        assert float(jnp.max(jnp.abs(sy))) > 0.0          # meridional front → ∂ᵧ stretching

    def test_unstable_column_no_spurious_spike(self):
        """A statically UNSTABLE column (N²<0) → stretching set to 0 there, NOT a
        huge floored f/N² spike (review SHOULD-FIX)."""
        from legoesm.ocean.dynamics.latlon_cgrid_operators import qg_pv_stretching_vec
        grid = _grid()
        nlev = 6
        zc = -(np.arange(nlev) * 20.0 + 10.0)
        lat = np.asarray(grid.lat)
        front = np.tanh((lat - lat.mean()) / np.radians(5.0))[:, None, None]
        # UNSTABLE base stratification (b DECREASES with height → N²<0).
        b = (-4e-6 * zc)[None, None, :] + 5e-3 * front + 0.0 * np.exp(zc)[None, None, :]
        b = jnp.asarray(np.broadcast_to(b, (grid.n_lat, grid.n_lon, nlev)).copy())
        h = jnp.full((grid.n_lat, grid.n_lon, nlev), 20.0)
        f_h = jnp.full((grid.n_lat, 1), float(2 * constants.Omega * np.sin(np.radians(-50.0))))
        sx, sy = qg_pv_stretching_vec(b, h, f_h, grid)
        # No blow-up: the stretching stays O(1e-9) (PV-gradient scale), not ~1e-2.
        assert float(jnp.max(jnp.abs(sy))) < 1e-6, float(jnp.max(jnp.abs(sy)))

    def test_operator_with_stretching_differs_from_barotropic(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            qg_leith_viscosity_tendency_cgrid,
        )
        grid = _grid()
        nlev = 6
        u, v = _fields(grid, nlev=nlev, seed=3)
        zc = -(np.arange(nlev) * 20.0 + 10.0)
        lat = np.asarray(grid.lat)
        front = np.tanh((lat - lat.mean()) / np.radians(5.0))[:, None, None]
        b = jnp.asarray(np.broadcast_to(
            (4e-6 * zc)[None, None, :] + 5e-3 * front * np.exp(zc / 500.0)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev)).copy())
        h = jnp.full((grid.n_lat, grid.n_lon, nlev), 20.0)
        bt_u, _ = qg_leith_viscosity_tendency_cgrid(u, v, grid)
        st_u, _ = qg_leith_viscosity_tendency_cgrid(
            u, v, grid, buoyancy=b, h_k=h, deformation_radius=6.75e3)
        assert bool(jnp.all(jnp.isfinite(st_u)))
        assert not jnp.allclose(bt_u, st_u, atol=1e-14)   # stretching changes ν


def test_config_dispatch_and_full_step():
    """lateral_friction_scheme='qg_leith' builds, validates, full step finite."""
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    r = build_eady_uniform_setup(n_lat=16, n_lon=16,
                                 momentum_advection="vector_invariant")
    cfg = r.model_config._replace(
        lateral_friction_scheme="qg_leith", qg_leith_coeff=2.0,
        lateral_viscosity=r.model_config.lateral_viscosity._replace(A_h=0.0, B_h=0.0, C_smag=0.0, C_leith=0.0))
    model = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 300.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))


def test_full_qg2_stretching_step_finite():
    """B5b live path: lateral_friction_scheme='qg_leith' + qg_leith_stretching=True
    threads buoyancy (from ρ') + thicknesses into the viscosity stage; a full
    model step stays finite (the faithful QG2)."""
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    r = build_eady_uniform_setup(n_lat=16, n_lon=16,
                                 momentum_advection="vector_invariant", nlev=8)
    cfg = r.model_config._replace(
        lateral_friction_scheme="qg_leith", qg_leith_coeff=2.0,
        qg_leith_stretching=True, qg_leith_deformation_radius_m=6.75e3,
        lateral_viscosity=r.model_config.lateral_viscosity._replace(A_h=0.0, B_h=0.0, C_smag=0.0, C_leith=0.0))
    model = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 300.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))
    assert bool(jnp.all(jnp.isfinite(nxt.T.data)))


def test_qg_leith_with_nonzero_other_friction_rejected():
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = build_eady_uniform_setup(n_lat=12, n_lon=12,
                                 momentum_advection="vector_invariant")
    bad = r.model_config._replace(lateral_friction_scheme="qg_leith", lateral_viscosity=r.model_config.lateral_viscosity._replace(A_h=1.0e4))
    with pytest.raises(ValueError, match="sole lateral friction"):
        LatLonCGridOceanModel(r.grid, r.z_coord, bad)


def test_omega_zero_grid_drops_planetary_vorticity():
    """The QG-Leith absolute vorticity must take f from the GRID's stored
    omega (#521): on an omega=0 grid the beta contribution vanishes, so
    the viscosity (and tendency) built from |grad(zeta+f)| differs from
    the Earth-rotation grid for identical velocities.  The pre-fix code
    hardcoded constants.Omega and made these bit-identical."""
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )

    g_earth = create_latlon_grid(16, 32)
    g0 = create_latlon_grid(16, 32, omega=0.0)
    rng = np.random.default_rng(3)
    u = jnp.asarray(0.2 * rng.standard_normal((16, 33, 2)))
    v = jnp.asarray(0.2 * rng.standard_normal((17, 32, 2)))
    v = v.at[0].set(0.0).at[-1].set(0.0)

    tu_e, tv_e = qg_leith_viscosity_tendency_cgrid(u, v, g_earth)
    tu_0, tv_0 = qg_leith_viscosity_tendency_cgrid(u, v, g0)
    assert not np.allclose(np.asarray(tu_e), np.asarray(tu_0))
    assert not np.allclose(np.asarray(tv_e), np.asarray(tv_0))
