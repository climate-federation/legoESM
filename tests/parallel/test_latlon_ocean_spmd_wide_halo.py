"""SPMD equivalence + collective-count gate for the WIDE-HALO barotropic.

Twin of ``test_latlon_ocean_spmd_step.py`` with
``barotropic_wide_halo=True``: N steps of ``make_sharded_ocean_step`` across
4 (CPU) devices must match the single-device wide-halo step at the sharded
split-explicit re-association floor.  A staggered off-by-one in the wide
v-exchange, a reach under-budget at a band cut, or a pole-flag error in the
extended geometry shows up here as an O(1e-3+) band-cut mismatch.

Also gates the POINT of the wide path mechanically: the compiled sharded
step's WHILE-BODY ``collective-permute`` count must DROP vs the standard
config — per-iteration subcycle communication is exactly what the wide
exchange removes (static op totals are NOT runtime message counts: loop-body
ops run n_substeps times, the wide path's unrolled exchanges run once).

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      JAX_ENABLE_X64=1 pytest tests/parallel/test_latlon_ocean_spmd_wide_halo.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

pytestmark = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs >=4 devices (XLA_FLAGS host device count)")

# The sharded split-explicit re-association floor of the existing SPMD gate
# (test_latlon_ocean_spmd_step.py) — NOT a bug margin; real halo regressions
# are O(1e-3+) at the band cuts.
_ATOL, _RTOL = 2.0e-4, 1.0e-3


def _perturbed_state(grid, z_coord):
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


def _cfg(**flat):
    # local clamp so the serial reference runs the exact wide-path scheme
    # (the wide subcycle forces local per-substep clamping by contract).
    flat.setdefault("barotropic_local_subcycle_clamp", True)
    return LatLonCGridOceanConfig.from_flat(**flat)


def _run_pair(cfg, n_lat=48, n_lon=96, nlev=10, dt=600.0, n_steps=3):
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
        gather_state_latlon,
    )

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)

    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)
    model._ensure_vertex_mask(state0)

    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)
    return s, ss


@pytest.mark.parametrize("chunk", [0, 2])
def test_wide_halo_spmd_matches_single_device(chunk):
    cfg = _cfg(barotropic_wide_halo=True, barotropic_wide_halo_chunk=chunk)
    s, ss = _run_pair(cfg)
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(
            b, a, atol=_ATOL, rtol=_RTOL,
            err_msg=f"wide-halo SPMD {nm} mismatch (chunk={chunk})")


def _sharded_step_hlo(cfg, n_lat=48, n_lon=96, nlev=10):
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
    )

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    lowered = jax.jit(step).lower(ss, 600.0)
    return lowered.compile().as_text()


def _while_body_collective_count(hlo_text: str) -> int:
    """Collective-permute ops inside while-LOOP BODY computations.

    A static HLO op count is NOT a runtime message count: ops inside a
    while body execute once PER ITERATION (the standard subcycle's ~4 pads
    compile to a handful of ops but run n_substeps times), while the wide
    path's unrolled chunk exchanges each run once.  The structural claim to
    gate is therefore: the wide path leaves NO barotropic halo collective
    inside a loop body — its substeps are communication-free.

    Parses the HLO text into computation blocks (``%name (args) -> ... {``
    to the closing ``}``), counts collective-permutes per block, and sums
    the counts over blocks referenced as ``body=%name`` by while ops.
    """
    import re

    blocks: dict[str, int] = {}
    name = None
    count = 0
    for line in hlo_text.splitlines():
        m = re.match(r"^%?([\w\.\-]+)\s*\(.*\)\s*->.*{", line.strip())
        if name is None and m:
            name, count = m.group(1), 0
            continue
        if name is not None:
            if line.startswith("}"):
                blocks[name] = count
                name = None
            elif "collective-permute" in line and "done" not in line:
                count += 1
    body_names = set(re.findall(r"body=%?([\w\.\-]+)", hlo_text))
    return sum(blocks.get(b, 0) for b in body_names)


def test_wide_halo_empties_loop_body_collectives():
    """The halo-count gate, mechanically: with wide-halo ON, the barotropic
    subcycle's while body must contain FEWER halo collectives than the
    standard config — per-iteration communication is exactly what the wide
    exchange removes.  Loops shared by both configs (vertical solves etc.)
    contribute equally, so the strict inequality isolates the subcycle."""
    n_std = _while_body_collective_count(_sharded_step_hlo(_cfg()))
    n_wide = _while_body_collective_count(
        _sharded_step_hlo(_cfg(barotropic_wide_halo=True)))
    assert n_wide < n_std, (
        f"wide-halo left per-iteration halo collectives in the subcycle "
        f"body: standard while-body collectives={n_std}, wide={n_wide}")
