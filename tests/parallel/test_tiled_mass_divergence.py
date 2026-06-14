"""np24 gate: tiled PPM mass-flux divergence (dh_dt) — the design-doc HARDEST op.

The production height tendency `cgrid_mass_flux_divergence` (PPM upwind C-grid
mass transport) tiled via the U3 DEEP-GLOBAL-PRE-PAD pattern: h is a STAGE
INPUT, pre-padded one ring deeper than the production halo=2 (so a tile's PPM
reconstruction of its boundary cell carries the real diagonal/-3 neighbour at
interior cuts, while a FACE-edge tile reproduces the global's mode='edge'
ghost); the per-tile reconstruction is LOCAL (no in-stage ppermute).  The cc
winds u_c/v_c are staggered stage inputs (flux divergence reads only a cell's
own bounding faces -> no halo).  The shared `cgrid_ppm_fluxes_core(halo_in=3)`
reuses the production numerics verbatim (the same core, halo_in=2, drives the
global op).

Gated by BIT-IDENTITY: each tile's dh_dt block vs the global
cgrid_mass_flux_divergence (cc cells partition exactly -> direct compare, no
shared face) to a tight FMA-robust relative tolerance.

Base case: apply_fortran_xppm_boundary=False, non-duogrid.

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.core.operators_cdgrid import cgrid_mass_flux_divergence
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_cgrid_mass_divergence_stage_2d,
)

N = 24  # divisible by kt=2 (nl=12) and kt=3 (nl=8)


@pytest.fixture(scope="module")
def setup():
    set_halo_backend("local")
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert cdg.base.duogrid is None, "base cut is non-duogrid"
    rng = np.random.default_rng(515)
    # h positive (height); C-grid winds O(1) with BOTH signs so the upwind
    # where(u_c>0,...) branch is exercised on every face.
    h = jnp.asarray(1.0e3 + rng.standard_normal((6, N, N)))
    u_c = jnp.asarray(rng.standard_normal((6, N + 1, N)))
    v_c = jnp.asarray(rng.standard_normal((6, N, N + 1)))
    # Both upwind branches (where(u_c>0,...)) must be exercised — assert, not
    # just comment (codex audit LOW): standard-normal winds straddle 0.
    assert bool((u_c > 0).any()) and bool((u_c < 0).any()), "u_c needs both signs"
    assert bool((v_c > 0).any()) and bool((v_c < 0).any()), "v_c needs both signs"
    dh_g = cgrid_mass_flux_divergence(h, u_c, v_c, cdg)  # base case (defaults)
    return cdg, h, u_c, v_c, np.asarray(dh_g)


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_mass_divergence_matches_global(setup, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    NL = N // KT
    cdg, h, u_c, v_c, dh_g = setup

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_cgrid_mass_divergence_stage_2d(mesh, cdg, N, KT)

    fo = P("face", None, None)
    sh_fo = NamedSharding(mesh, fo)
    h_sh = jax.device_put(h, sh_fo)
    u_sh = jax.device_put(u_c, sh_fo)
    v_sh = jax.device_put(v_c, sh_fo)

    dh_t = np.asarray(stage(h_sh, u_sh, v_sh))   # (6, n, n), cc exact partition

    worst = 0.0
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                g = dh_g[f, ti * NL:(ti + 1) * NL, tj * NL:(tj + 1) * NL]
                t = dh_t[f, ti * NL:(ti + 1) * NL, tj * NL:(tj + 1) * NL]
                worst = max(worst, float(np.max(np.abs(t - g))))

    # FMA-robust relative tolerance: the deep-pad + per-tile reconstruction
    # reorders the global's contiguous PPM arithmetic -> O(1e-13) ULP drift,
    # not algorithmic.  Tight enough to catch a halo-depth / index / upwind /
    # stagger bug.
    scale = float(np.max(np.abs(dh_g))) + 1e-300
    rel = worst / scale
    assert rel < 1e-10, (
        f"kt={KT} tiled dh_dt vs global: abs={worst:.3e} rel={rel:.3e}")
