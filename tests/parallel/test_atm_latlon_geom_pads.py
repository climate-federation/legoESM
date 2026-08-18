"""Precomputed stage-invariant geometry pads for the atm lat-band SPMD step.

Packing-plan bucket C: the SPMD band step precomputes the three 1-D
geometry pads (``grid.lat`` wall-0, ``grid.lat`` pole-clamped, and
``grid.cos_lat`` wall-0) ONCE at build time — sliced per band from the
globally padded arrays — instead of re-emitting a scalar-row ppermute
per RK stage inside the operators.  Two gates:

1. ``test_precomputed_band_pads_match_inbody_pad`` — for nd in {1, 2, 4},
   the build-time band slices are BIT-IDENTICAL to what the in-body
   backend-dispatched ``pad_with_pole_bc_lat`` (armed SPMD lat-band
   backend, real ppermute path) produces on every band.
2. ``test_step_identical_with_and_without_precompute`` — the sharded step
   output with the precomputed pads equals the output of the same step
   built WITHOUT them (in-body pads), bit-for-bit.
3. ``test_precomputed_pads_are_consumed`` — non-vacuity (codex 2026-08-11):
   tracing the with-pads step performs ZERO 1-D ``pad_with_pole_bc_lat``
   calls (the geometry pads really are skipped, not silently fallen back
   to), while the without-pads build restores them.

Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4`` or more).
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_spmd import (
    activate_latlon_spmd_halo,
    deactivate_latlon_spmd_halo,
)
from legoesm.parallel.shard_map_compat import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.atmosphere.dynamics.gcm import sharded_atm_latlon_step as sas
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)

N_LAT = 16
N_LON = 16
NLEV = 4

_PAD_SPECS = (
    # (attr, south_value, north_value) — must mirror _build_geometry_stacks.
    ("lat", 0.0, 0.0),
    ("lat", -math.pi / 2.0, math.pi / 2.0),
    ("cos_lat", 0.0, 0.0),
)


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_spmd_mesh(None)
    set_halo_backend("local")


def _mesh(n_dev):
    if len(jax.devices()) < n_dev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={n_dev}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:n_dev]), axis_names=("lat",))


def _grid():
    return create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)


@pytest.mark.parametrize("n_dev", [1, 2, 4])
def test_precomputed_band_pads_match_inbody_pad(n_dev):
    """Build-time band slices == the armed-SPMD in-body pad, every band."""
    mesh = _mesh(n_dev)
    grid = _grid()
    nl = N_LAT // n_dev
    for attr, sv, nv in _PAD_SPECS:
        field = jnp.asarray(getattr(grid, attr))
        # Build-time global pad (halo backend un-armed -> local jnp.pad),
        # sliced per band — exactly what _build_geometry_stacks stacks.
        ext_global = pad_with_pole_bc_lat(
            field, halo=1, south_value=sv, north_value=nv)
        band_slices = jnp.stack(
            [ext_global[r * nl:r * nl + nl + 2] for r in range(n_dev)],
            axis=0)  # (n_dev, nl+2)

        # In-body pad: the REAL armed-SPMD ppermute path inside shard_map.
        bands = jnp.stack(
            [field[r * nl:(r + 1) * nl] for r in range(n_dev)], axis=0)
        bands = jax.device_put(
            bands, NamedSharding(mesh, P("lat", None)))

        def body(tile):
            return pad_with_pole_bc_lat(
                tile[0], halo=1, south_value=sv, north_value=nv)[None]

        fn = jax.jit(shard_map(
            body, mesh=mesh, in_specs=P("lat", None),
            out_specs=P("lat", None), check_vma=False))
        activate_latlon_spmd_halo(mesh)
        try:
            inbody = fn(bands)
        finally:
            deactivate_latlon_spmd_halo()

        np.testing.assert_array_equal(
            np.asarray(inbody), np.asarray(band_slices),
            err_msg=f"{attr} pad ({sv}, {nv}) nd={n_dev}")


def _model_and_state():
    grid = _grid()
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_ppm_transport=True, time_integrator="ssp_rk3")
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(20260811)
    eps = 1.0e-3
    v0 = eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    v0[0] = 0.0
    v0[-1] = 0.0
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(v0),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    return model, state


@pytest.mark.parametrize("n_dev", [2, 4])
def test_step_identical_with_and_without_precompute(n_dev, monkeypatch):
    """Sharded step output is bit-identical with and without the pads."""
    mesh = _mesh(n_dev)
    model, state = _model_and_state()
    dt = 60.0

    step_with = sas.make_sharded_atm_latlon_step(model, mesh)
    assert "__lat_pad" in step_with._geom_stacks  # precompute actually wired

    orig = sas._build_geometry_stacks

    def _no_pads(*args, **kwargs):
        template, names, stacks, spec = orig(*args, **kwargs)
        for key in ("__lat_pad", "__lat_pad_pole", "__cos_lat_pad"):
            stacks.pop(key)
            spec.pop(key)
        return template, names, stacks, spec

    monkeypatch.setattr(sas, "_build_geometry_stacks", _no_pads)
    step_without = sas.make_sharded_atm_latlon_step(model, mesh)
    assert "__lat_pad" not in step_without._geom_stacks

    c0 = sas.shard_state_atm_latlon(state, mesh)
    out_with = c0
    out_without = c0
    for _ in range(2):
        out_with = step_with(out_with, dt)
        out_without = step_without(out_without, dt)

    leaves_w, treedef_w = jax.tree.flatten(out_with)
    leaves_wo, treedef_wo = jax.tree.flatten(out_without)
    assert treedef_w == treedef_wo
    for i, (a, b) in enumerate(zip(leaves_w, leaves_wo)):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b), err_msg=f"leaf {i}")


def test_precomputed_pads_are_consumed(monkeypatch):
    """Non-vacuity gate: the with-pads step body traces ZERO 1-D geometry
    pads; dropping the precompute entries restores the in-body pads.

    Counts trace-time calls of ``pad_with_pole_bc_lat`` on 1-D arrays (the
    only 1-D pads in the step are the three stage-invariant geometry pads;
    all state pads are >= 2-D).  Fails if any ``geom_pads`` hand-off in
    the chain _make_band_step_body -> _step_cgrid_impl ->
    cgrid_latlon_hydrostatic_tendencies -> operators is removed."""
    n_dev = 2
    mesh = _mesh(n_dev)

    import legoesm.grids.halo_latlon as hl
    calls = {"geom_1d": 0}
    orig_pad = hl.pad_with_pole_bc_lat

    def counting(arr, *args, **kwargs):
        if getattr(arr, "ndim", None) == 1:
            calls["geom_1d"] += 1
        return orig_pad(arr, *args, **kwargs)

    monkeypatch.setattr(hl, "pad_with_pole_bc_lat", counting)

    model, state = _model_and_state()
    c0 = sas.shard_state_atm_latlon(state, mesh)
    dt = 60.0

    step_with = sas.make_sharded_atm_latlon_step(model, mesh)
    calls["geom_1d"] = 0
    step_with(c0, dt)          # first call traces the band body
    n_with = calls["geom_1d"]

    orig_build = sas._build_geometry_stacks

    def _no_pads(*args, **kwargs):
        template, names, stacks, spec = orig_build(*args, **kwargs)
        for key in ("__lat_pad", "__lat_pad_pole", "__cos_lat_pad"):
            stacks.pop(key)
            spec.pop(key)
        return template, names, stacks, spec

    monkeypatch.setattr(sas, "_build_geometry_stacks", _no_pads)
    step_without = sas.make_sharded_atm_latlon_step(model, mesh)
    calls["geom_1d"] = 0
    step_without(c0, dt)
    n_without = calls["geom_1d"]

    assert n_with == 0, f"with-pads trace still pads 1-D geometry: {n_with}"
    assert n_without >= 3, f"in-body fallback should pad >= 3x: {n_without}"
