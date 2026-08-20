"""Packed per-stage exchange epoch for the atm lat-band SPMD step (bucket A).

Opt-in ``LEGOESM_LATLON_PACKED_EXCHANGE=1``: per RK stage the three
per-stage exchanges whose operands are live at the epoch site — the
``[B|ln p]`` gradient-stack fold pad (P1), the PPM T-advection halo-2
fold pad (P4) and the fused wall-BC entry pad (T, u, dp[, p_s]) (P2) —
merge into ONE north+south ppermute pair
(:func:`legoesm.parallel.latlon_spmd.make_latlon_band_packed_pad_body`).
Raw un-lon-padded edge rows ride one buffer per dtype group; wall
constants / pole folds / lon wrap are applied locally after receipt, so
the design is BIT-exact vs the per-exchange bodies.

Kept separate (recorded): the sigma_dot / mass-flux v-interp exchange
(operand depends on div(dp*v), i.e. on THIS epoch's padded dp) and the
step-entry ``reconstruct_vface_lower`` north row (bucket D refused: its
output feeds the KE inside B upstream of the epoch — merging would be
cyclic).  Census: 25 -> 13 collective-permutes/step (12 merged + 1 R).

Gates:
1. ``test_packed_body_matches_single_field_bodies`` — the packed body is
   bit-equal to the per-field fold/wall bodies (mixed halos, negate,
   wall constants; nd in {1, 2, 4} — polar bands are where it bites).
2. ``test_census_and_bit_exactness`` — optimized-HLO census env-off == 25
   EXACTLY (no regression) and env-on == 13; 2-step x64 trajectories
   bit-identical (non-vacuity: the census assert fails if the packing is
   silently disabled).
3. ``test_grad_parity`` — grad of a scalar loss through 1 step, env-on vs
   env-off, allclose 1e-12 (x64).  No new custom_vjp: the comm envelope
   is linear (ppermute self-transposing).
4. ``test_env_gate_dispatch_hardening`` — unknown env value RAISES.

Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=8`` or more).
"""
from __future__ import annotations

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_spmd import (
    activate_latlon_spmd_halo,
    deactivate_latlon_spmd_halo,
    make_latlon_band_packed_pad_body,
    make_latlon_band_pad_body,
    make_latlon_band_wall_pad_body,
    packed_exchange_mesh,
)
from legoesm.parallel.shard_map_compat import shard_map

from legoesm.atmosphere.dynamics.gcm import sharded_atm_latlon_step as sas
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)

N_LAT = 16
N_LON = 16
NLEV = 4

CENSUS_OFF = 25   # the bucket-C receipt (geom pads landed)
CENSUS_ON = 13    # 3 stages x (2 merged + 2 sigma_dot) + 1 reconstruct-v


@pytest.fixture(autouse=True)
def _restore_backend_and_env(monkeypatch):
    monkeypatch.delenv("LEGOESM_LATLON_PACKED_EXCHANGE", raising=False)
    yield
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_spmd_mesh(None)
    set_halo_backend("local")


def _mesh(n_dev):
    if len(jax.devices()) < n_dev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={n_dev}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:n_dev]), axis_names=("lat",))


def _model_and_state(vertical="sigma"):
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    if vertical == "hybrid":
        from legoesm.grids.vertical import create_hybrid_coordinate
        half = np.linspace(0.0, 1.0, NLEV + 1)
        B_half = half ** 2
        A_half = 0.1 * constants.p_ref * (half - B_half)
        sigma = create_hybrid_coordinate(
            NLEV, jnp.asarray(A_half), jnp.asarray(B_half))
    else:
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


# ==============================================================================
# (1) packed body == per-field fold/wall bodies, bit-for-bit
# ==============================================================================

@pytest.mark.parametrize("n_dev", [1, 2, 4])
def test_packed_body_matches_single_field_bodies(n_dev):
    """Mixed-halo, mixed-semantics, MIXED-DTYPE packed exchange vs the
    single-field bodies (fold h1 f64, fold h2 negate f32, wall h1 f64 with
    nonzero constants, wall h1 f32) — every band including the polar folds
    must be BIT-equal.  The f32 members exercise the per-dtype-group
    packing/split path (codex r1 MINOR 2)."""
    mesh = _mesh(n_dev)
    rng = np.random.default_rng(3)
    nl = N_LAT // n_dev
    f_a = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))     # fold h1
    f_b = jnp.asarray(                                               # fold h2 neg
        rng.standard_normal((N_LAT, N_LON, NLEV)), dtype=jnp.float32)
    f_c = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, NLEV)))  # wall h1
    f_d = jnp.asarray(                                               # wall h1 f32
        rng.standard_normal((N_LAT, N_LON, 1)), dtype=jnp.float32)
    specs = (("fold", 1, False), ("fold", 2, True), ("wall", 1, 1.5, -2.5),
             ("wall", 1, 0.0, 0.0))

    packed = make_latlon_band_packed_pad_body(mesh, specs)
    ref_a = make_latlon_band_pad_body(mesh, halo=1, negate=False)
    ref_b = make_latlon_band_pad_body(mesh, halo=2, negate=True)
    ref_c = make_latlon_band_wall_pad_body(
        mesh, halo=1, south_value=1.5, north_value=-2.5)
    ref_d = make_latlon_band_wall_pad_body(mesh, halo=1)

    def body(a, b, c, d):
        pa, pb, pc, pd = packed(a, b, c, d)
        return pa, pb, pc, pd, ref_a(a), ref_b(b), ref_c(c), ref_d(d)

    sp3 = P("lat", None, None)
    fn = jax.jit(shard_map(
        body, mesh=mesh, in_specs=(sp3,) * 4,
        out_specs=(sp3,) * 8, check_vma=False))
    args = tuple(jax.device_put(f, NamedSharding(mesh, sp3))
                 for f in (f_a, f_b, f_c, f_d))
    pa, pb, pc, pd, ra, rb, rc, rd = fn(*args)
    for got, ref, name in ((pa, ra, "fold h1 f64"), (pb, rb, "fold h2 neg f32"),
                           (pc, rc, "wall h1 f64"), (pd, rd, "wall h1 f32")):
        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(ref),
            err_msg=f"packed {name} != single-field body, nd={n_dev}")
        assert got.dtype == ref.dtype, name
    # non-vacuity of the reference: padded shapes actually grew
    assert pa.shape[0] == n_dev * (nl + 2)
    assert pb.shape[0] == n_dev * (nl + 4)


def test_packed_body_rejects_unknown_kind_and_2d_mesh():
    mesh = _mesh(1)
    with pytest.raises(ValueError, match="unknown spec kind"):
        make_latlon_band_packed_pad_body(mesh, (("magic", 1, False),))
    if len(jax.devices()) >= 2:
        mesh2d = jax.sharding.Mesh(
            np.array(jax.devices()[:2]).reshape(1, 2),
            axis_names=("lat", "lon"))
        with pytest.raises(ValueError, match="1-D .'lat',. band mesh"):
            make_latlon_band_packed_pad_body(
                mesh2d, (("fold", 1, False),))


def test_gradient_y_f_padded_shape_guard():
    """A wrong-shape pre-pad must raise, not silently mis-slice."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.operators_latlon_cgrid import gradient_y_cgrid
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    f = jnp.zeros((N_LAT, N_LON, NLEV))
    with pytest.raises(ValueError, match="halo-1 fold pad"):
        gradient_y_cgrid(f, grid, f_padded=jnp.zeros((N_LAT, N_LON, NLEV)))


# ==============================================================================
# (2) census + bit-exactness through the production sharded step
# ==============================================================================

def _census_and_run(mesh, env_val, monkeypatch, n_steps=2,
                    vertical="sigma"):
    if env_val is None:
        monkeypatch.delenv("LEGOESM_LATLON_PACKED_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("LEGOESM_LATLON_PACKED_EXCHANGE", env_val)
    model, state = _model_and_state(vertical)
    step = sas.make_sharded_atm_latlon_step(model, mesh)
    c0 = sas.shard_state_atm_latlon(state, mesh)
    hlo = jax.jit(lambda c, dt: step(c, dt)).lower(c0, 60.0).compile().as_text()
    n_cp = len(re.findall(r"collective-permute(?:-start)?\(", hlo))
    out = c0
    for _ in range(n_steps):
        out = step(out, 60.0)
    return n_cp, [np.asarray(x) for x in jax.tree.leaves(out)]


@pytest.mark.parametrize("n_dev", [2, 4, 8])
def test_census_and_bit_exactness(n_dev, monkeypatch):
    """env-off census == 25 exactly (no regression), env-on == 13, and the
    2-step x64 trajectories are BIT-identical.  nd=2/4 put every band on a
    pole — the case the local-fold/wall design must get right."""
    mesh = _mesh(n_dev)
    cp_off, leaves_off = _census_and_run(mesh, None, monkeypatch)
    cp_on, leaves_on = _census_and_run(mesh, "1", monkeypatch)
    assert cp_off == CENSUS_OFF, f"env-off census regressed: {cp_off}"
    assert cp_on == CENSUS_ON, f"env-on census: {cp_on} != {CENSUS_ON}"
    for i, (a, b) in enumerate(zip(leaves_off, leaves_on)):
        np.testing.assert_array_equal(a, b, err_msg=f"leaf {i}, nd={n_dev}")


def test_bit_exactness_hybrid(monkeypatch):
    """Hybrid vertical coordinate: the packed epoch carries the extra p_s
    wall field (4-field unpack lane) — env-on vs env-off bit-identical,
    and both censuses match the sigma lane's (same exchange structure)."""
    mesh = _mesh(2)
    cp_off, leaves_off = _census_and_run(
        mesh, None, monkeypatch, vertical="hybrid")
    cp_on, leaves_on = _census_and_run(
        mesh, "1", monkeypatch, vertical="hybrid")
    assert cp_off == CENSUS_OFF, f"hybrid env-off census: {cp_off}"
    assert cp_on == CENSUS_ON, f"hybrid env-on census: {cp_on}"
    for i, (a, b) in enumerate(zip(leaves_off, leaves_on)):
        np.testing.assert_array_equal(a, b, err_msg=f"hybrid leaf {i}")


# ==============================================================================
# (3) AD parity
# ==============================================================================

def test_grad_parity(monkeypatch):
    """grad of a scalar loss through 1 sharded step: env-on vs env-off,
    allclose 1e-12 (x64)."""
    n_dev = 2
    mesh = _mesh(n_dev)

    def grads(env_val):
        if env_val is None:
            monkeypatch.delenv("LEGOESM_LATLON_PACKED_EXCHANGE",
                               raising=False)
        else:
            monkeypatch.setenv("LEGOESM_LATLON_PACKED_EXCHANGE", env_val)
        model, state = _model_and_state()
        step = sas.make_sharded_atm_latlon_step(model, mesh)
        c0 = sas.shard_state_atm_latlon(state, mesh)

        def loss(c):
            out = step(c, 60.0)
            tot = jnp.zeros((), dtype=jnp.float64)
            for leaf in jax.tree.leaves(out):
                tot = tot + jnp.sum(jnp.asarray(leaf) ** 2)
            return tot

        g = jax.grad(loss)(c0)
        return [np.asarray(x) for x in jax.tree.leaves(g)]

    g_off = grads(None)
    g_on = grads("1")
    for i, (a, b) in enumerate(zip(g_off, g_on)):
        np.testing.assert_allclose(
            a, b, rtol=1e-12, atol=1e-12, err_msg=f"grad leaf {i}")


# ==============================================================================
# (4) env-gate dispatch hardening
# ==============================================================================

def test_env_gate_dispatch_hardening(monkeypatch):
    monkeypatch.setenv("LEGOESM_LATLON_PACKED_EXCHANGE", "yes")
    with pytest.raises(ValueError, match="LEGOESM_LATLON_PACKED_EXCHANGE"):
        packed_exchange_mesh()
    # '0' / '' are OFF; '1' without an armed band mesh is still None.
    monkeypatch.setenv("LEGOESM_LATLON_PACKED_EXCHANGE", "0")
    assert packed_exchange_mesh() is None
    monkeypatch.setenv("LEGOESM_LATLON_PACKED_EXCHANGE", "1")
    assert packed_exchange_mesh() is None    # backend un-armed
    mesh = _mesh(1)
    activate_latlon_spmd_halo(mesh)
    try:
        assert packed_exchange_mesh() is mesh
    finally:
        deactivate_latlon_spmd_halo()
