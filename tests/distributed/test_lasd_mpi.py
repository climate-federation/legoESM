"""Distributed LASD dynamic Smagorinsky under y-slab MPI == serial.

``lasd_cs2`` (Bou-Zeid scale-dependent dynamic SGS, the oracle-faithful closure)
uses three horizontally-coupled operators the y-slab decomposition must
reproduce: the sharp spectral TEST filters (distributed FFT + local kx cutoff),
the planar means of the Germano contractions (global SUM), and the 3×3 box
average ``imfilter_box3`` (∂y rolls cross ranks → a 1-row y-halo).

Validation strategy. The three operators are checked for EXACT equivalence
(below). The composed ``C_s²`` field is NOT bit-matched on purpose: the
distributed FFT is not bit-identical to the serial ``rfft2`` (transpose vs
monolithic summation order, ~1e-10), and the dynamic procedure's quintic β-root
plus the ``LM/MM`` division amplify that into O(1e-2) on the raw C_s² — the
dynamic SGS is intrinsically noisy, which is exactly why it planar/box-averages
and clips. The physically meaningful invariant is the integrated STATE: the ν_t
noise is damped by the projection, the spectral filter and time integration, so a
full dynamic-SGS ``step()`` trajectory matches single-rank to ~1e-4 (checked
last). Same non-bit-identical class as the CRM fix_mass / moist-mean reductions.

Run::

    mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/test_lasd_mpi.py
    mpirun -np 4 .venv-mpi/bin/python -m pytest tests/distributed/test_lasd_mpi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    SpectralLESConfig, SpectralLESLayout, SpectralLESState, make_grid, step)
from legoesm.atmosphere.physics.turbulence.lasd_core import (  # noqa: E402
    lasd_cs2, spectral_test_filter, imfilter_box3)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NPROC = COMM.Get_size()

NY, NX, NZ = 12, 16, 8


def _rand(seed, shape=(NY, NX, NZ)):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _y_slab(f):
    nyl = NY // NPROC
    return f[RANK * nyl:(RANK + 1) * nyl]


def _gather_y(slab):
    return jnp.asarray(np.concatenate(COMM.allgather(np.asarray(slab)), axis=0))


def _layout():
    return SpectralLESLayout(rank=RANK, n_ranks=NPROC, ny_global=NY, nx=NX,
                             comm=COMM)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_spectral_test_filter_matches_serial():
    """Distributed sharp spectral test filter (FFT + local kx cutoff) == serial."""
    f = _rand(1)
    ser = spectral_test_filter(f, 4, 5)
    dist = spectral_test_filter(_y_slab(f), 4, 5, _layout())
    np.testing.assert_allclose(np.asarray(_gather_y(dist)), np.asarray(ser),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_imfilter_box3_matches_serial():
    """3×3 box average with the cross-rank ∂y 1-row halo == serial (exact)."""
    f = _rand(2)
    ser = imfilter_box3(f)
    dist = imfilter_box3(_y_slab(f), _layout())
    np.testing.assert_allclose(np.asarray(_gather_y(dist)), np.asarray(ser),
                               rtol=0, atol=1e-13)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_imfilter_box3_grad_matches_serial():
    """Grad through the box3 y-halo == serial (the AD-safe sendrecv VJP)."""
    f = _rand(3)

    def loss(x, layout):
        return jnp.sum(imfilter_box3(x, layout) ** 2)

    gs = jax.grad(lambda x: loss(x, None))(f)
    gd = jax.grad(lambda x: loss(x, _layout()))(_y_slab(f))
    np.testing.assert_allclose(np.asarray(_gather_y(gd)), np.asarray(gs),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_lasd_cs2_matches_serial():
    """Distributed lasd_cs2 == serial on a WELL-CONDITIONED smooth multi-mode
    field. This directly exercises the internal test-filter cutoff selection
    (cut1y/cut2y from the GLOBAL ny — a regression guard for the bug where
    uc.shape[0] gave ny_local and shrank the cutoffs, codex-caught 2026-06-09).

    Random unphysical strain is intentionally NOT used: the dynamic procedure's
    quintic β-root + LM/MM division make raw C_s² hypersensitive there, so the
    non-bit-identical distributed FFT (~1e-10) blows up to ~1e-2 — masking real
    cutoff errors. On a smooth field the Germano system is well-conditioned and
    C_s² matches to ~1e-12, so a wrong cutoff (a STRUCTURAL change) is caught."""
    yy = np.linspace(0, 2 * np.pi, NY, endpoint=False)
    xx = np.linspace(0, 2 * np.pi, NX, endpoint=False)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    base = (np.sin(X) + 0.7 * np.cos(2 * Y) + 0.5 * np.sin(3 * Y + X)
            + 0.3 * np.cos(4 * Y))   # modes spanning the 2Δ/4Δ test-filter band

    def fld(s):
        n = np.random.default_rng(s).standard_normal((NY, NX, NZ))
        return jnp.asarray(base[:, :, None] + 0.1 * n)

    args = [fld(i) for i in range(10)]
    args[-1] = jnp.sqrt(args[-1] ** 2 + 1e-2)          # |S| > 0
    delta = jnp.linspace(1.0, 2.0, NZ)

    cs2_ser = lasd_cs2(*args, delta, cs_max=1.0)
    cs2_dist = lasd_cs2(*[_y_slab(a) for a in args], delta, cs_max=1.0,
                        layout=_layout())
    np.testing.assert_allclose(np.asarray(_gather_y(cs2_dist)),
                               np.asarray(cs2_ser), rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(NY % NPROC != 0, reason="NY must divide by n_ranks")
def test_dynamic_sgs_step_matches_serial():
    """Full dynamic-LASD step() trajectory == single-rank (the integrated state,
    the physically meaningful invariant — see module docstring)."""
    cfg = SpectralLESConfig(nx=NX, ny=NY, nz=NZ, Lx=300.0, Ly=225.0, Lz=100.0,
                            dealias=False, smagorinsky_dynamic=True,
                            spectral_filter=True, time_scheme="ab2")
    dt, u_geo, f_cor, force = 2.0e-3, (1.0, 0.0), 1.0e-4, (1.0e-3, 0.0)

    def init():
        rng = np.random.default_rng(101)
        u = jnp.asarray(1.0 + 0.1 * rng.standard_normal((NY, NX, NZ)))
        v = jnp.asarray(0.1 * rng.standard_normal((NY, NX, NZ)))
        w = np.zeros((NY, NX, NZ + 1))
        w[..., 1:NZ] = 0.05 * rng.standard_normal((NY, NX, NZ - 1))
        z = jnp.zeros((NY, NX, NZ))
        return SpectralLESState(u=u, v=v, w=jnp.asarray(w), rhs_u_prev=z,
                                rhs_v_prev=z,
                                rhs_w_prev=jnp.zeros((NY, NX, NZ + 1)))

    def run(st, g):
        for i in range(3):
            st, _ = step(st, g, dt, u_geo, f_cor, first=(i == 0), force=force)
        return st

    st_ser = run(init(), make_grid(cfg))

    gi = init()
    st0 = SpectralLESState(
        u=_y_slab(gi.u), v=_y_slab(gi.v), w=_y_slab(gi.w),
        rhs_u_prev=_y_slab(gi.rhs_u_prev), rhs_v_prev=_y_slab(gi.rhs_v_prev),
        rhs_w_prev=_y_slab(gi.rhs_w_prev))
    st_dist = run(st0, make_grid(cfg, layout=_layout()))

    for name, dist, ser in (("u", st_dist.u, st_ser.u),
                            ("v", st_dist.v, st_ser.v),
                            ("w", st_dist.w, st_ser.w)):
        np.testing.assert_allclose(
            np.asarray(_gather_y(dist)), np.asarray(ser),
            rtol=1e-4, atol=1e-4, err_msg=f"{name} mismatch")
