"""Is a 3-D decomposition (horizontal x LEVEL) viable for the
finite-volume lanes?  Offline collective census, CPU virtual devices.

The spectral dycore anti-scales under level sharding (measured: 0.74x at
4 devices, T85) because its semi-implicit solve couples all levels. The
MPAS and lat-lon lanes are finite-volume: their vertical structure is
per-column (hydrostatic cumsum, vertical advection, implicit diffusion
solves), which couples levels too — but nobody has counted what GSPMD
actually emits when the LEVEL axis is sharded.

Method: jit the SERIAL step with the level axis sharded across N virtual
CPU devices (GSPMD constraint, no code changes), compile, and count the
collectives. The count is the verdict:

    O(1)-O(10) collectives/step  -> level sharding pays for its comm at
                                    some device count; a 3-D mesh is
                                    worth a GPU measurement.
    O(nlev) or more              -> every column operator serialises
                                    across the level shards; 3-D is dead
                                    for this lane, as for spectral.

Run:  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      JAX_PLATFORMS=cpu python scripts/tmp/_probe_level_shard_census.py
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def census(txt: str) -> dict:
    out = {}
    for op in ("all-gather", "all-reduce", "collective-permute",
               "all-to-all", "reduce-scatter"):
        out[op] = len(re.findall(rf"{op}(?!-done)", txt))
    out["total"] = sum(out.values())
    return out


def latlon(n_dev: int):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

    nlev = 24  # divisible by 2,4,8
    grid = create_latlon_grid(n_lat=48, n_lon=96)
    sigma = create_sigma_coordinate(nlev)
    cfg = CGridLatLonPrimitiveEquationConfig()
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True)

    mesh = Mesh(np.array(jax.devices()[:n_dev]), ("level",))

    def spec(leaf):
        # shard the trailing LEVEL axis where one exists
        if leaf.ndim >= 1 and leaf.shape[-1] == nlev:
            return NamedSharding(mesh, P(*([None] * (leaf.ndim - 1)
                                           + ["level"])))
        return NamedSharding(mesh, P())

    sharded = jax.tree.map(
        lambda x: jax.device_put(x, spec(x)) if hasattr(x, "ndim") else x,
        state)
    f = jax.jit(lambda s: model.step(s, 120.0))
    txt = f.lower(sharded).compile().as_text()
    return census(txt)


def mpas(n_dev: int):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    nlev = 24
    vmesh = create_voronoi_mesh(subdivision_level=4)
    sigma = create_sigma_coordinate(nlev)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3")
    model = MPASPrimitiveEquationModel(vmesh, sigma, cfg)
    state = baroclinic_wave_init_mpas(vmesh, sigma, perturbed=True)

    mesh = Mesh(np.array(jax.devices()[:n_dev]), ("level",))

    def spec(leaf):
        if hasattr(leaf, "ndim") and leaf.ndim >= 1 \
                and leaf.shape[-1] == nlev:
            return NamedSharding(mesh, P(*([None] * (leaf.ndim - 1)
                                           + ["level"])))
        return NamedSharding(mesh, P())

    sharded = jax.tree.map(
        lambda x: jax.device_put(x, spec(x)) if hasattr(x, "ndim") else x,
        state)
    f = jax.jit(lambda s: model.step(s, 120.0))
    txt = f.lower(sharded).compile().as_text()
    return census(txt)


def main() -> int:
    n = jax.device_count()
    print(f"# devices={n} (virtual CPU), level axis sharded, nlev=24")
    for name, fn in (("latlon LL48x96", latlon), ("mpas s4", mpas)):
        for nd in (2, 4, 8):
            if nd > n:
                continue
            try:
                c = fn(nd)
                print(f"{name:>16} level x{nd}: total={c['total']:5d}  {c}")
            except Exception as e:  # noqa: BLE001 — report, keep sweeping
                print(f"{name:>16} level x{nd}: FAILED {type(e).__name__}: "
                      f"{str(e)[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
