"""#1100 — band-local Held-Suarez IC construction parity gate.

``build_sharded_held_suarez_state_atm_latlon`` must be BIT-IDENTICAL, shard
by shard, to the reference path (global ``held_suarez_init_latlon`` →
``hydrostatic_to_cgrid`` → ``shard_state_atm_latlon``) on a virtual-device
lat mesh — this is the correctness contract that lets the route-B bench skip
the per-process global build (the OOM mechanism under full-node CPU packing).

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
skips when fewer devices are available.  The multi-process
(``make_array_from_callback`` per-process rows) leg of #1100 needs a real
multi-controller launch and is validated by the bench itself on cluster —
this gate pins the single-process semantics both legs share.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV = 4
N_LAT = 16   # divisible by N_DEV
N_LON = 12
NLEV = 5


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _grid_sigma():
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    return grid, sigma


def _reference_sharded(grid, sigma, mesh):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        shard_state_atm_latlon)
    hs0 = held_suarez_init_latlon(grid, sigma)
    c0 = hydrostatic_to_cgrid(hs0, grid)
    return shard_state_atm_latlon(c0, mesh)


def test_bandlocal_build_bit_identical_to_global_shard():
    mesh = _mesh()
    grid, sigma = _grid_sigma()
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        build_sharded_held_suarez_state_atm_latlon)

    ref = _reference_sharded(grid, sigma, mesh)
    got = build_sharded_held_suarez_state_atm_latlon(grid, sigma, mesh)

    for name in ("u", "v", "T", "p_s", "phis"):
        a = getattr(ref, name)
        b = getattr(got, name)
        assert a.shape == b.shape, f"{name}: shape {b.shape} != {a.shape}"
        assert a.dtype == b.dtype, f"{name}: dtype {b.dtype} != {a.dtype}"
        assert a.sharding.is_equivalent_to(b.sharding, a.ndim), (
            f"{name}: sharding differs")
        # bit-identical, shard by shard (device_get concatenates shards)
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b)),
            err_msg=f"{name}: band-local build differs from global+shard")
    assert got.tracers == {}
    assert ref.tracers == {}


def test_bandlocal_build_matches_ic_kwargs():
    """Non-default IC kwargs flow through identically (seed, amplitude, T)."""
    mesh = _mesh()
    grid, sigma = _grid_sigma()
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        build_sharded_held_suarez_state_atm_latlon, shard_state_atm_latlon)

    kwargs = dict(T_init=285.0, perturbation_amplitude=0.25, seed=7)
    hs0 = held_suarez_init_latlon(grid, sigma, **kwargs)
    ref = shard_state_atm_latlon(hydrostatic_to_cgrid(hs0, grid), mesh)
    got = build_sharded_held_suarez_state_atm_latlon(
        grid, sigma, mesh, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(ref.T)), np.asarray(jax.device_get(got.T)))
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(ref.p_s)), np.asarray(jax.device_get(got.p_s)))


def test_bandlocal_build_steps_like_reference():
    """One sharded step from each construction is bit-identical."""
    mesh = _mesh()
    grid, sigma = _grid_sigma()
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        build_sharded_held_suarez_state_atm_latlon,
        make_sharded_atm_latlon_step)

    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    step = make_sharded_atm_latlon_step(model, mesh)

    ref = _reference_sharded(grid, sigma, mesh)
    got = build_sharded_held_suarez_state_atm_latlon(grid, sigma, mesh)
    s_ref = step(ref, 60.0)
    s_got = step(got, 60.0)
    for name in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(getattr(s_ref, name))),
            np.asarray(jax.device_get(getattr(s_got, name))),
            err_msg=f"{name}: stepped states diverge")
