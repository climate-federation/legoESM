"""MPI (mpirun -np 2) tests for the fused lat-lon halo exchange (O4 lever).

Run:  mpirun -np 2 python -m pytest tests/distributed/test_fused_halo_mpi.py

Covers what the serial suite cannot:
  * fused multi-field exchange == per-field single exchanges at a REAL
    interior partition cut (values from the neighbour rank);
  * AD parity through the fused sendrecv (``_sendrecv_vjp`` path);
  * static-metric trace-time folding: a concrete (non-Tracer) pad
    inside ``jax.jit`` produces NO ``mpi_sendrecv`` custom call in the
    compiled HLO (tripwire) while matching the traced-path values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

mpi4py = pytest.importorskip("mpi4py")
pytest.importorskip("mpi4jax")

from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK, SIZE = comm.Get_rank(), comm.Get_size()

pytestmark = pytest.mark.skipif(
    SIZE < 2, reason="needs mpirun -np 2",
)

N_LAT_GLOBAL, N_LON = 8, 6


def _layout():
    from legoesm.parallel.latlon_mpi import make_latlon_band_layout

    return make_latlon_band_layout(
        rank=RANK, n_ranks=SIZE,
        n_lat=N_LAT_GLOBAL, n_lon=N_LON,
    )


def _band_fields(layout, nlev=3, key=0):
    """Rank-local bands of deterministic global fields."""
    rng = np.random.default_rng(key)  # same stream on every rank
    g2d = rng.standard_normal((N_LAT_GLOBAL, N_LON))
    g3d = rng.standard_normal((N_LAT_GLOBAL, N_LON, nlev))
    s, e = layout.lat_start, layout.lat_end
    return jnp.asarray(g2d[s:e]), jnp.asarray(g3d[s:e])


def test_fused_matches_singles_at_cut():
    from legoesm.parallel.latlon_mpi import (
        pad_with_pole_bc_lat_mpi,
        pad_with_pole_bc_lat_multi_mpi,
    )

    layout = _layout()
    f2d, f3d = _band_fields(layout)
    sv, nv = (1.0e30, 0.0), (1.0e30, -3.0)

    fused = pad_with_pole_bc_lat_multi_mpi(
        (f2d, f3d), layout, halo=1, south_values=sv, north_values=nv,
    )
    singles = tuple(
        pad_with_pole_bc_lat_mpi(
            f, layout, halo=1, south_value=s, north_value=n,
        )
        for f, s, n in zip((f2d, f3d), sv, nv)
    )
    for got, want in zip(fused, singles):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_fused_halo2_matches_singles():
    from legoesm.parallel.latlon_mpi import (
        pad_with_pole_bc_lat_mpi,
        pad_with_pole_bc_lat_multi_mpi,
    )

    layout = _layout()
    f2d, f3d = _band_fields(layout, key=1)
    fused = pad_with_pole_bc_lat_multi_mpi((f2d, f3d), layout, halo=2)
    for got, f in zip(fused, (f2d, f3d)):
        want = pad_with_pole_bc_lat_mpi(f, layout, halo=2)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_fused_grad_matches_singles_grad():
    from legoesm.parallel.latlon_mpi import (
        pad_with_pole_bc_lat_mpi,
        pad_with_pole_bc_lat_multi_mpi,
    )

    layout = _layout()
    f2d, f3d = _band_fields(layout, key=2)

    def loss_fused(a, b):
        pa, pb = pad_with_pole_bc_lat_multi_mpi((a, b), layout, halo=1)
        return jnp.sum(pa**2) + jnp.sum(pb**2)

    def loss_singles(a, b):
        pa = pad_with_pole_bc_lat_mpi(a, layout, halo=1)
        pb = pad_with_pole_bc_lat_mpi(b, layout, halo=1)
        return jnp.sum(pa**2) + jnp.sum(pb**2)

    ga = jax.grad(loss_fused, argnums=(0, 1))(f2d, f3d)
    gb = jax.grad(loss_singles, argnums=(0, 1))(f2d, f3d)
    for x, y in zip(ga, gb):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y))


def test_static_metric_pad_folds_to_constant():
    """Concrete input -> trace-time host exchange -> NO per-step sendrecv."""
    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi

    layout = _layout()
    # Static 1D "metric" (deterministic, same global array every rank).
    metric_global = np.linspace(-1.0, 1.0, N_LAT_GLOBAL)
    metric_local = jnp.asarray(
        metric_global[layout.lat_start:layout.lat_end],
    )

    @jax.jit
    def f(x):
        pad = pad_with_pole_bc_lat_mpi(
            metric_local, layout, halo=1,
            south_value=-1.0, north_value=1.0,
        )
        return x * jnp.sum(pad)

    lowered = f.lower(jnp.float64(1.0))
    hlo = lowered.compile().as_text()
    assert "mpi_sendrecv" not in hlo, (
        "static-metric pad was traced into a per-step sendrecv — "
        "constant folding regressed"
    )

    # Values match the traced path exactly (neighbour rows at the cut,
    # constants at the poles).
    got = np.asarray(f(jnp.float64(1.0)))
    s, e = layout.lat_start, layout.lat_end
    south = -1.0 if layout.south_rank is None else metric_global[s - 1]
    north = 1.0 if layout.north_rank is None else metric_global[e]
    want = south + metric_global[s:e].sum() + north
    np.testing.assert_allclose(got, want)


def test_static_path_rejects_oversized_halo():
    """Codex MAJOR: static path must raise like the traced path, not
    silently short-send."""
    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi

    layout = _layout()
    tiny = jnp.asarray(np.arange(2.0))  # n_lat_local = 2 < halo = 3
    with pytest.raises(ValueError, match="halo=3 exceeds"):
        pad_with_pole_bc_lat_mpi(tiny, layout, halo=3)


def test_traced_boundary_constant_keeps_traced_path():
    """Codex MAJOR: a Tracer south/north value must NOT be constant-
    folded — the pad stays traced and the constant stays differentiable."""
    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi

    layout = _layout()
    metric_local = jnp.asarray(
        np.linspace(0.0, 1.0, N_LAT_GLOBAL)[
            layout.lat_start:layout.lat_end
        ],
    )

    def f(c):
        pad = pad_with_pole_bc_lat_mpi(
            metric_local, layout, halo=1, south_value=c, north_value=c,
        )
        return jnp.sum(pad)

    g = jax.grad(f)(jnp.float64(2.0))
    # The boundary constant appears once per pole-touching side on this
    # rank; interior-cut sides receive neighbour data (zero grad in c).
    n_boundary_sides = (layout.south_rank is None) + (
        layout.north_rank is None
    )
    np.testing.assert_allclose(np.asarray(g), float(n_boundary_sides))


def test_static_fold_is_1d_only():
    """ndim >= 2 concrete fields keep the traced sendrecv path
    (codex CRITICAL hardening: narrow static surface to 1-D metrics)."""
    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi

    layout = _layout()
    rng = np.random.default_rng(7)
    g2d = rng.standard_normal((N_LAT_GLOBAL, N_LON))
    field2d = jnp.asarray(g2d[layout.lat_start:layout.lat_end])

    @jax.jit
    def f(x):
        return x * jnp.sum(
            pad_with_pole_bc_lat_mpi(field2d, layout, halo=1),
        )

    hlo = f.lower(jnp.float64(1.0)).compile().as_text()
    assert "mpi_sendrecv" in hlo  # 2-D concrete stays traced
    f(jnp.float64(1.0)).block_until_ready()


def test_static_fold_env_kill_switch(monkeypatch):
    from legoesm.parallel.latlon_mpi import pad_with_pole_bc_lat_mpi

    monkeypatch.setenv("LEGOESM_LATLON_STATIC_METRIC_PAD", "0")
    layout = _layout()
    metric_local = jnp.asarray(
        np.linspace(0.0, 1.0, N_LAT_GLOBAL)[
            layout.lat_start:layout.lat_end
        ],
    )

    @jax.jit
    def f(x):
        pad = pad_with_pole_bc_lat_mpi(metric_local, layout, halo=1)
        return x * jnp.sum(pad)

    hlo = f.lower(jnp.float64(1.0)).compile().as_text()
    assert "mpi_sendrecv" in hlo  # traced path restored
    f(jnp.float64(1.0)).block_until_ready()
