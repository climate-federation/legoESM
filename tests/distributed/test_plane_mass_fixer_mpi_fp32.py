"""float32 multi-rank dry-mass fixer holds the anchored dry mass to round-off.

The MPI fixer used to compute (target - current) from two ~1e14 float32
totals: a bounded per-step mass jitter of ~2e-7 relative. Anchored on the
global rho' integral it closes to round-off. Spawns ``mpiexec -n 2`` with x64
OFF (the production float32 configuration); run with plain ``pytest``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

MPIEXEC = os.environ.get("MPIEXEC") or shutil.which("mpiexec") or shutil.which("mpirun")

_SCRIPT = r"""
import jax, jax.numpy as jnp, numpy as np
assert not jax.config.jax_enable_x64
from mpi4py import MPI
from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric, make_rest_state)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import (
    gather_plane_field, make_plane_pencil_grid, make_plane_pencil_layout)
comm = MPI.COMM_WORLD
N, NL = 32, 16
lay = make_plane_pencil_layout(rank=comm.Get_rank(), n_ranks=2, n_ranks_y=1,
                               n_ranks_x=2, ny_global=N, nx_global=N, halo=1)
gg = create_plane_grid(nx=N, ny=N, nlev=NL, dx=4.0e3, dy=4.0e3, dtype=jnp.float32)
gl = make_plane_pencil_grid(lay, dx=4.0e3, dy=4.0e3, nlev=NL, dtype=jnp.float32)
hc = create_height_coordinate(NL, H=20.0e3)
rs = np.random.RandomState(7)
rest = make_rest_state(gg, hc, dtype=jnp.float32)
u = 0.5 * rs.standard_normal(rest.u.data.shape)
rho = 1e-3 * rs.standard_normal(rest.rho_prime.data.shape)
sl = lambda a: jnp.asarray(a[lay.iy_start:lay.iy_end, lay.ix_start:lay.ix_end], jnp.float32)
loc = make_rest_state(gl, hc, dtype=jnp.float32)
s = loc._replace(u=loc.u.replace(data=sl(u)), rho_prime=loc.rho_prime.replace(data=sl(rho)))
cfg = CompressibleEulerConfig(fix_mass=True, anchor_mass_to_initial=True,
                              hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
                              hyperdiff_w_coeff=0.0)
m = PlaneCompressibleEulerModel(gl, hc, make_flat_plane_terrain_metric(gl, hc), cfg)
own = jnp.ones((lay.ny_local, lay.nx_local), jnp.float32)
tm = make_flat_plane_terrain_metric(gg, hc)
w = (np.asarray(gg.area_T, np.float64) * np.asarray(tm.jacobian, np.float64))[:, :, None] \
    * np.asarray(hc.dz, np.float64)
def mass(st):
    g = gather_plane_field(st.rho_prime.data, lay)
    if g is None:
        return None
    assert g.dtype == np.float32
    return float(np.sum((np.asarray(hc.rho_ref, np.float64) + np.asarray(g, np.float64)) * w))
m0 = mass(s)
drift = 0.0
for _ in range(3):
    s = m.step_halo(s, 6.0, lay, owned_mask=own)
    mn = mass(s)
    if mn is not None:
        drift = max(drift, abs(mn - m0) / m0)
if comm.Get_rank() == 0:
    print("REL_DRIFT", drift)
"""


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_fp32_mpi_mass_fixer_holds_anchor_to_roundoff():
    """Max over 3 steps of |M(t) - M(0)| / M(0), 2 ranks, masses in fp64."""
    pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4jax")
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "0"
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([MPIEXEC, "-n", "2", sys.executable, "-c", _SCRIPT],
                       env=env, capture_output=True, text=True, timeout=900,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-3000:]
    rel = float(r.stdout.split("REL_DRIFT")[1].split()[0])
    print("fp32 2-rank 3-step dry-mass drift (relative):", rel)
    assert rel <= 1e-9, rel


_CALLER_TARGET_SCRIPT = r"""
import jax, jax.numpy as jnp, numpy as np
assert jax.config.jax_enable_x64
from mpi4py import MPI
from legoesm.atmosphere.dynamics.crm.rce_mpi import compute_dry_mass_plane_mpi
from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric, make_rest_state)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_grid, make_plane_pencil_layout
comm = MPI.COMM_WORLD
N, NL = 16, 8
lay = make_plane_pencil_layout(rank=comm.Get_rank(), n_ranks=2, n_ranks_y=1,
                               n_ranks_x=2, ny_global=N, nx_global=N, halo=1)
gg = create_plane_grid(nx=N, ny=N, nlev=NL, dx=4.0e3, dy=4.0e3, dtype=jnp.float64)
gl = make_plane_pencil_grid(lay, dx=4.0e3, dy=4.0e3, nlev=NL, dtype=jnp.float64)
hc = create_height_coordinate(NL, H=20.0e3)
rho = 1e-3 * np.random.RandomState(7).standard_normal((N, N, NL))
loc = make_rest_state(gl, hc, dtype=jnp.float64)
s = loc._replace(rho_prime=loc.rho_prime.replace(
    data=jnp.asarray(rho[lay.iy_start:lay.iy_end, lay.ix_start:lay.ix_end])))
cfg = CompressibleEulerConfig(fix_mass=True, anchor_mass_to_initial=False,
                              hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
                              hyperdiff_w_coeff=0.0)
tm = make_flat_plane_terrain_metric(gl, hc)
m = PlaneCompressibleEulerModel(gl, hc, tm, cfg)
own = jnp.ones((lay.ny_local, lay.nx_local), jnp.float64)
dry0 = compute_dry_mass_plane_mpi(s, gl, hc, tm, lay, own)
m._target_mass = dry0 * (1.0 + 1e-6)          # caller-set total, no anchor
out = m.step_halo(s, 2.0, lay, owned_mask=own)
dry1 = compute_dry_mass_plane_mpi(out, gl, hc, tm, lay, own)
if comm.Get_rank() == 0:
    print("REL_OFF", float(dry1 / dry0) - 1.0 - 1e-6)
"""


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_mpi_step_halo_honours_caller_target_without_anchor():
    """2 ranks, anchor_mass_to_initial=False: a caller-set ``_target_mass``
    drives the MPI fixer as it does the serial one (it used to be discarded
    for the pre-step mass)."""
    pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4jax")
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([MPIEXEC, "-n", "2", sys.executable, "-c",
                        _CALLER_TARGET_SCRIPT],
                       env=env, capture_output=True, text=True, timeout=900,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-3000:]
    off = float(r.stdout.split("REL_OFF")[1].split()[0])
    print("2-rank caller-target dry-mass offset:", off)
    assert abs(off) < 1e-12, off
