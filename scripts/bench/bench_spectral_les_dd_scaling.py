"""Distributed spectral-LES (y-slab) weak/strong scaling bench, fp32 + fp64.

Benchmarks the pseudo-spectral incompressible plane LES (``spectral_les_plane``)
under the y-slab MPI decomposition (``SpectralLESLayout``, ``parallel/
distributed_fft.py``) — the transpose-based distributed 2-D FFT pressure solve,
the LASD dynamic SGS test filters, the box3 ∂y halo and the global-mean wall
model. Runs the FULL oracle closure (``smagorinsky_dynamic=True``).

Modes:
  strong : fixed GLOBAL grid, ny split across ranks (wall-time should drop)
  weak   : fixed PER-RANK grid (ny_local fixed; global ny = n_ranks·ny_local)

Run (MUST use the tested MPI stack ``.venv-mpi`` = JAX 0.9.2 + mpi4jax 0.8.1)::

    export PATH="$HOME/.local/mpich/bin:$PATH" \
           LD_LIBRARY_PATH="$HOME/.local/mpich/lib:$LD_LIBRARY_PATH" \
           PYTHONPATH="$PWD/packages/core:$PWD/packages/atmosphere:$PWD" \
           JAX_PLATFORMS=cpu MPI4JAX_NO_WARN_JAX_VERSION=1 \
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1
    mpirun -np 4 .venv-mpi/bin/python scripts/bench/bench_spectral_les_dd_scaling.py \
        --mode strong --precision float64 --nx 64 --ny 64 --nz 32 --steps 30

CSV columns: mode,precision,n_ranks,ny_global,nx,nz,ny_local,wall_s,ms_per_step.
"""
from __future__ import annotations

import argparse
import sys
import time

# Precision must be set BEFORE jax imports config-sensitive state.
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--precision", default="float64", choices=["float32", "float64"])
_known, _ = _p.parse_known_args()
_X64 = _known.precision == "float64"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", _X64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    SpectralLESConfig, SpectralLESLayout, SpectralLESState, make_grid, step)

_DTYPE = jnp.float64 if _X64 else jnp.float32


def _init(ny_local, nx, nz, seed):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(1.0 + 0.1 * rng.standard_normal((ny_local, nx, nz)), _DTYPE)
    v = jnp.asarray(0.1 * rng.standard_normal((ny_local, nx, nz)), _DTYPE)
    w = np.zeros((ny_local, nx, nz + 1))
    w[..., 1:nz] = 0.05 * rng.standard_normal((ny_local, nx, nz - 1))
    z = jnp.zeros((ny_local, nx, nz), _DTYPE)
    return SpectralLESState(u=u, v=v, w=jnp.asarray(w, _DTYPE), rhs_u_prev=z,
                            rhs_v_prev=z,
                            rhs_w_prev=jnp.zeros((ny_local, nx, nz + 1), _DTYPE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strong", "weak"], default="strong")
    ap.add_argument("--precision", default="float64",
                    choices=["float32", "float64"])
    ap.add_argument("--ny", type=int, default=64, help="global ny (strong) "
                    "or per-rank ny (weak)")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()

    if args.mode == "strong":
        ny_global = args.ny
        if ny_global % n:
            if rank == 0:
                print(f"skip: ny_global {ny_global} % n_ranks {n} != 0")
            return
        ny_local = ny_global // n
    else:  # weak: per-rank ny fixed
        ny_local = args.ny
        ny_global = ny_local * n

    nx, nz = args.nx, args.nz
    cfg = SpectralLESConfig(nx=nx, ny=ny_global, nz=nz, Lx=600.0, Ly=600.0,
                            Lz=300.0, dealias=False, smagorinsky_dynamic=True,
                            spectral_filter=True, time_scheme="ab2")
    layout = SpectralLESLayout(rank=rank, n_ranks=n, ny_global=ny_global, nx=nx,
                               comm=comm)
    g = make_grid(cfg, dtype=_DTYPE, layout=layout)
    st = _init(ny_local, nx, nz, seed=1 + rank)

    dt = _DTYPE(2.0e-3)
    fn = jax.jit(lambda s: step(s, g, dt, (1.0, 0.0), 1.0e-4, first=False,
                                force=(1.0e-3, 0.0))[0])
    st = fn(st)
    st.u.block_until_ready()
    comm.Barrier()

    t0 = time.time()
    for _ in range(args.steps):
        st = fn(st)
    st.u.block_until_ready()
    comm.Barrier()
    wall = time.time() - t0
    ms = wall / args.steps * 1e3

    if rank == 0:
        line = (f"{args.mode},{args.precision},{n},{ny_global},{nx},{nz},"
                f"{ny_local},{wall:.4f},{ms:.2f}")
        print(f"mode={args.mode} prec={args.precision} np={n} "
              f"global={ny_global}x{nx}x{nz} ny_local={ny_local}: "
              f"{ms:.1f} ms/step")
        if args.csv:
            hdr = ("mode,precision,n_ranks,ny_global,nx,nz,ny_local,"
                   "wall_s,ms_per_step\n")
            import os
            new = not os.path.exists(args.csv)
            with open(args.csv, "a") as f:
                if new:
                    f.write(hdr)
                f.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
