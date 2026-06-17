"""Gate: tiled ``fix_ps_mass`` (dry-mass fixer via psum) — the cube tiled STEP's
post-RK3 mass conservation (default use_conservation_fixer+fix_mass), the second
tiled GLOBAL reduction (after zero_mean) and a TIER-0 truth (conservation) op.

Reference = the global ``core.conservation.fix_ps_mass(p_s_new, p_s_old, grid)``.
Two checks: (a) tiled == global up to psum reduction-ORDER reordering, and (b) the
CONSERVED property ``sum(out*area) == sum(p_s_old*area)`` to machine precision (the
fixer restores p_s_new's dry mass to p_s_old's).  Coord-free.  NOT a wall-clock
measurement.

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
from legoesm.core.conservation import fix_ps_mass
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fix_ps_mass_stage_2d,
)

N = 24


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    return create_cubed_sphere_cdgrid(create_cubed_sphere(N))


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_fix_ps_mass_matches_global(cdg, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    grid = cdg.base
    rng = np.random.default_rng(110 + KT)
    p_s_old = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, N, N)))
    # p_s_new = old + a mass-drifting perturbation (the fixer removes the drift).
    p_s_new = p_s_old + jnp.asarray(50.0 * rng.standard_normal((6, N, N)))
    g = np.asarray(fix_ps_mass(p_s_new, p_s_old, grid))

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fix_ps_mass_stage_2d(mesh, grid, N, KT)
    fo = NamedSharding(mesh, P("face", None, None))
    t = np.asarray(stage(jax.device_put(p_s_new, fo), jax.device_put(p_s_old, fo)))

    # (a) match the global op (psum reorders the global scalars -> ULP).
    rel = float(np.max(np.abs(t - g))) / (float(np.max(np.abs(g))) + 1e-300)
    assert rel < 1e-10, f"tiled vs global rel {rel:.3e} (kt={KT})"

    # (b) the conserved property: out's dry mass == p_s_old's dry mass.
    area = np.asarray(grid.area).astype(np.float64)
    mass_out = float(np.sum(t.astype(np.float64) * area))
    mass_target = float(np.sum(np.asarray(p_s_old).astype(np.float64) * area))
    assert abs(mass_out - mass_target) / abs(mass_target) < 1e-12, \
        f"mass drift {abs(mass_out - mass_target) / abs(mass_target):.3e} (kt={KT})"


def test_tiled_fix_ps_mass_rejects_bad_n(cdg):
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    with pytest.raises(ValueError, match="divisible"):
        make_tiled_fix_ps_mass_stage_2d(mesh, cdg.base, N, 5)
