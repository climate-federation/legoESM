"""Gate: tiled area-weighted ``zero_mean_tendency`` (psum global reduction) — the
FIRST tiled GLOBAL collective in the cube stages (all prior tendency stages are
halo-only), the primitive the full tiled STEP's mass-fixer + the capstone's
``_apply_zero_mean_per_stage=True`` config path need.

Reference = the global ``core.conservation.zero_mean_tendency``.  Two checks:
(a) tiled == global up to psum reduction-ORDER reordering (the correction is a
global scalar; the per-cell output differs only by the psum-vs-jnp.sum ULP), and
(b) the CONSERVED property ``sum(out*area)==0`` to machine precision — the whole
point of the op.  Coord-free (pure area geometry).  NOT a wall-clock measurement.

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
from legoesm.core.conservation import zero_mean_tendency
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_zero_mean_tendency_stage_2d,
)

N = 24


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    return create_cubed_sphere_cdgrid(create_cubed_sphere(N))


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_zero_mean_matches_global(cdg, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    grid = cdg.base
    rng = np.random.default_rng(100 + KT)
    # dp_s/dt-like 2D cc tendency with a non-zero mean (so the correction bites).
    tend = jnp.asarray(0.5 * rng.standard_normal((6, N, N)) + 0.1)
    g = np.asarray(zero_mean_tendency(tend, grid))

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_zero_mean_tendency_stage_2d(mesh, grid, N, KT)
    t = np.asarray(stage(jax.device_put(
        tend, NamedSharding(mesh, P("face", None, None)))))

    # (a) match the global op (psum reorders the global scalar -> ULP drift).
    rel = float(np.max(np.abs(t - g))) / (float(np.max(np.abs(g))) + 1e-300)
    assert rel < 1e-10, f"tiled vs global rel {rel:.3e} (kt={KT})"

    # (b) the conserved property: area-weighted global sum == 0.
    area = np.asarray(grid.area).astype(np.float64)
    out_wsum = abs(float(np.sum(t.astype(np.float64) * area)))
    scale = float(np.sum(np.abs(np.asarray(tend)).astype(np.float64) * area)) + 1e-300
    assert out_wsum / scale < 1e-12, \
        f"sum(out*area)/scale = {out_wsum / scale:.3e} (kt={KT}) — not conserving"


def test_tiled_zero_mean_rejects_bad_n(cdg):
    """Fail-loud factory guard."""
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    with pytest.raises(ValueError, match="divisible"):
        make_tiled_zero_mean_tendency_stage_2d(mesh, cdg.base, N, 5)  # 24%5!=0
