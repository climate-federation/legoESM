"""Kinetic energy of one baroclinic Coriolis rotation.

The forward-backward rotation (u' with the old v', then v' with the new u')
adds energy of order (f dt)^2 to a zonal flow every step (Sielecki 1968),
1.2 percent per step here (20N to 32N, dt = 1800 s), which feeds zonal
geostrophic jets.  ``coriolis_time_scheme="trapezoidal"`` (Crank-Nicolson)
with ``coriolis_baroclinic_operator="energy_conserving"`` (equal and opposite
pair coefficients) leaves the energy unchanged to round-off for any flow,
also over partial cells, and still rotates the flow.  The trapezoidal scheme
with the four point average keeps a residual that does not vanish as dt -> 0
(f and the face weights differ between u and v faces).

``coriolis_time_scheme="trapezoidal_step"`` centres the rotation on the whole step,
u1 = u0 + dt T + dt C((u0 + u1)/2), with u0 the velocity before the explicit
tendency T.  Its energy change is then exactly dt times the work of T on the centred
velocity (u0 + u1)/2, so the forward pressure gradient stage no longer adds
(1/2) dt^2 |T|^2, about (f dt)^2 of the energy of balanced flow per step.
"""
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as oml  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    min_cell_to_uface, min_cell_to_vface)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    make_partial_cell_latlon, rest_state_latlon_cgrid_ocean)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
DT = 1800.0


def _setup(time_scheme, operator, flow):
    grid, wall = create_regional_grid("latlon", n_lat=12, n_lon=16, lat_south=20.0,
                                      lat_north=32.0, lon_west=140.0, lon_east=156.0)
    ocean = np.asarray(wall) > 0.5
    dz = np.full(6, 500.0)
    zc = create_z_star_from_thicknesses(dz)
    if flow == "zonal":                                   # flat bottom, full cells
        zc_pc, H, lm = zc, np.where(ocean, 3000.0, 0.0), ocean.astype(float)
    else:                                                 # a ridge of partial cells
        lon = np.broadcast_to(np.linspace(139.5, 156.5, 18)[None, :], (14, 18))
        H = np.where(ocean, 3000.0 - 1700.0 * np.exp(-((lon - 148.0) / 2.0) ** 2), 0.0)
        zc_pc, H, lm = make_partial_cell_latlon(zc, H, ocean.astype(float))
    cfg = LatLonCGridOceanConfig.from_flat(coriolis_time_scheme=time_scheme,
                                           coriolis_baroclinic_operator=operator)
    model = LatLonCGridOceanModel(grid, zc_pc, cfg)
    st = rest_state_latlon_cgrid_ocean(grid, zc_pc, land_mask_override=jnp.asarray(lm, dtype=jnp.float64),
                                       H_bathy_override=jnp.asarray(H, dtype=jnp.float64),
                                       H_max=float(zc.H_max))
    h = compute_layer_thickness(st.eta.data, st.H_bathy.data, zc_pc)
    hu = np.asarray(min_cell_to_uface(h)) * np.asarray(st.u_mask.data)[..., None]
    hv = np.asarray(min_cell_to_vface(h, model.grid)) * np.asarray(st.v_mask.data)[..., None]
    if flow == "zonal":
        zmid = np.cumsum(dz) - dz / 2
        u0 = 0.5 * np.cos(np.pi * zmid / 3000.0)[None, None, :] * (hu > 0)
        v0 = np.zeros(hv.shape)
    else:                                                 # any flow: random u and v
        rng = np.random.default_rng(0)
        u0, v0 = rng.normal(0.0, 0.3, hu.shape) * (hu > 0), rng.normal(0.0, 0.3, hv.shape) * (hv > 0)
    # zero depth mean, so the output is the rotated perturbation itself
    u0 = u0 - ((u0 * hu).sum(-1) / np.maximum(hu.sum(-1), 1e-10))[..., None] * (hu > 0)
    v0 = v0 - ((v0 * hv).sum(-1) / np.maximum(hv.sum(-1), 1e-10))[..., None] * (hv > 0)
    u0[:, -1] = u0[:, 0]                                  # periodic duplicate u face, as the model
    Au = (np.asarray(model.grid.dx_u) * np.asarray(model.grid.dy_u))[..., None]
    Av = (np.asarray(model.grid.dx_v) * np.asarray(model.grid.dy_v))[..., None]
    nx = Au.shape[1] - 1                                  # count the duplicate u face once

    def rot(u, v, **kw):
        return oml._forward_backward_coriolis_3d(jnp.asarray(u), jnp.asarray(v), DT, model.grid, model.z_coord,
                                                 model.config, st.u_mask.data, st.v_mask.data, st.land_mask.data,
                                                 st.eta.data, st.H_bathy.data, **kw)

    def dot(a, b, c, d):                                  # sum of h A (a c + b d) over faces
        return float(np.sum((hu * Au * np.asarray(a) * np.asarray(c))[:, :nx])
                     + np.sum(hv * Av * np.asarray(b) * np.asarray(d)))

    def zero_mean(u, v):
        u = u - ((u * hu).sum(-1) / np.maximum(hu.sum(-1), 1e-10))[..., None] * (hu > 0)
        v = v - ((v * hv).sum(-1) / np.maximum(hv.sum(-1), 1e-10))[..., None] * (hv > 0)
        u[:, -1] = u[:, 0]
        return u, v

    return model, st, u0, v0, hu, hv, rot, dot, zero_mean


def _ke_change(time_scheme, operator="four_point", flow="zonal"):
    _, _, u0, v0, _, _, rot, dot, _ = _setup(time_scheme, operator, flow)
    u1, v1 = rot(u0, v0)
    return (dot(u1, v1, u1, v1) - dot(u0, v0, u0, v0)) / dot(u0, v0, u0, v0), float(np.abs(np.asarray(v1) - v0).max())


def test_forward_backward_adds_energy_to_zonal_flow():
    dke, dv = _ke_change("forward_backward")
    assert dke > 4e-3, dke                    # measured 1.2e-2
    assert dv > 0.03, dv


def test_trapezoidal_energy_conserving_is_neutral_to_round_off_for_any_flow():
    for flow in ("zonal", "random_partial_cells"):
        dke, dv = _ke_change("trapezoidal", "energy_conserving", flow)
        assert abs(dke) < 1e-13, (flow, dke)
        assert dv > 0.03, (flow, dv)          # dt f u is about 0.06 m/s: the flow still turns


def test_trapezoidal_four_point_keeps_a_spatial_residual():
    dke, _ = _ke_change("trapezoidal", "four_point", "random_partial_cells")
    assert abs(dke) > 1e-9, dke


def test_unknown_coriolis_options_raise():
    grid, _ = create_regional_grid("latlon", n_lat=4, n_lon=4, lat_south=20.0, lat_north=24.0,
                                   lon_west=140.0, lon_east=144.0)
    zc = create_z_star_from_thicknesses(np.full(2, 500.0))
    for field, value in (("coriolis_time_scheme", "leapfrog"), ("coriolis_baroclinic_operator", "een")):
        with pytest.raises(ValueError):
            LatLonCGridOceanModel(grid, zc, LatLonCGridOceanConfig.from_flat(**{field: value}))


def test_step_centred_energy_change_is_the_work_on_the_centred_velocity():
    """Any flow, any tendency, partial cells: KE(u1) - KE(u0) = 2 dt (T, (u0 + u1)/2) to round-off."""
    for operator in ("energy_conserving",):
        _, _, u0, v0, hu, hv, rot, dot, zero_mean = _setup("trapezoidal_step", operator, "random_partial_cells")
        rng = np.random.default_rng(1)
        Tu, Tv = zero_mean(rng.normal(0.0, 2e-5, hu.shape) * (hu > 0), rng.normal(0.0, 2e-5, hv.shape) * (hv > 0))
        u1, v1 = (np.asarray(x) for x in rot(u0 + DT * Tu, v0 + DT * Tv, u_old=jnp.asarray(u0), v_old=jnp.asarray(v0)))
        dke = dot(u1, v1, u1, v1) - dot(u0, v0, u0, v0)
        work = 2.0 * DT * dot(Tu, Tv, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
        assert abs(dke - work) < 1e-12 * dot(u0, v0, u0, v0), (dke, work)
        assert abs(work) > 1e-6 * dot(u0, v0, u0, v0)    # the tendency does work: the check is not vacuous


def test_step_centred_removes_the_gain_of_a_balanced_forward_stage():
    """T balancing the Coriolis force of u0 (geostrophy): the step-centred rotation keeps the energy, the
    rotation of the forward-stage velocity gains about (f dt)^2 of it."""
    model, st, u0, v0, hu, hv, rot, dot, zero_mean = _setup("trapezoidal_step", "energy_conserving", "zonal")
    cor_u, cor_v = oml._coriolis_pair_operator(model.grid, jnp.asarray(hu), jnp.asarray(hv), jnp.float64)
    # the Coriolis acceleration is (cor_u(v), cor_v(u)) (the rotation adds dt cor_u(v') to u' and subtracts
    # dv = -dt cor_v(u') from v'); T is minus that
    Tu, Tv = zero_mean(-np.asarray(cor_u(jnp.asarray(v0))) * (hu > 0), -np.asarray(cor_v(jnp.asarray(u0))) * (hv > 0))
    E0 = dot(u0, v0, u0, v0)
    us, vs = u0 + DT * Tu, v0 + DT * Tv
    u1, v1 = rot(us, vs, u_old=jnp.asarray(u0), v_old=jnp.asarray(v0))
    gain_centred = (dot(u1, v1, u1, v1) - E0) / E0
    _, _, _, _, _, _, rot_trap, _, _ = _setup("trapezoidal", "energy_conserving", "zonal")
    u2, v2 = rot_trap(us, vs)
    gain_trap = (dot(u2, v2, u2, v2) - E0) / E0
    assert gain_trap > 4e-3, gain_trap                   # about (f dt)^2 at 20N to 32N
    assert abs(gain_centred) < 1e-2 * gain_trap, (gain_centred, gain_trap)


def test_step_centred_needs_the_old_velocity():
    _, _, u0, v0, _, _, rot, _, _ = _setup("trapezoidal_step", "energy_conserving", "zonal")
    with pytest.raises(ValueError):
        rot(u0, v0)
