"""Lat-lon band SPMD fixed-M PCG (multi-GPU, no mpi4jax) — parity vs serial.

The next increment of the ocean lat-lon 2-GPU SPMD path (audit lever #1):
the barotropic implicit-Helmholtz solve runs a hand-rolled fixed-iteration
PCG (``ocean.dynamics.barotropic_common._fixed_iteration_pcg``) whose inner
products go through ``_global_dot_batch``.  Under the single-controller SPMD
backend each device owns one LATITUDE BAND, so every ``jnp.sum`` in the PCG
is a PARTIAL sum over a band and must be combined with ``jax.lax.psum`` over
the ``"lat"`` mesh axis — ``is_multi_process()`` is FALSE under one process,
so the MPI allreduce never fires and a missing psum would silently let each
band solve its OWN sub-system (the MPAS analogue was job 8460616).

This validates the psum routing end-to-end: the SAME backend-oblivious
``A_op`` (built on ``pad_halo_latlon``, which routes to the band halo body
under the spmd backend) run by

  * SERIAL ``_fixed_iteration_pcg`` (local dots, one process), vs
  * SHARDED ``_fixed_iteration_pcg`` inside a ``shard_map`` over ``"lat"``
    (``_global_dot_batch`` -> ``batch_psum_spmd`` -> ``jax.lax.psum``),

must agree to solver tolerance (the only difference is summation order:
one global sum vs psum of per-band partials).  A genuinely band-COUPLING
operator (the 5-point ``I + c(-Delta)`` Helmholtz, lon-periodic + lat
ppermute + pole fold) makes the cross-band reduction load-bearing — a
band-local dot would converge to the wrong field.

Runs on host CPU devices (``--xla_force_host_platform_device_count=4``);
the production target is 2 GPUs but the shard_map/ppermute/psum logic is
device-agnostic.  Mirrors ``tests/parallel/test_latlon_spmd_halo.py``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.halo_latlon import pad_halo_latlon
from legoesm.ocean.dynamics.barotropic_common import _fixed_iteration_pcg
from legoesm.parallel.latlon_spmd import (
    activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
)
from legoesm.parallel.reductions import batch_psum_spmd

N_DEV = 4
N_LAT = 16
NL = N_LAT // N_DEV
N_LON = 8
COEFF = 0.35          # I + c(-Delta); c>0 keeps A diagonally dominant SPD
MAX_ITER = 40         # fixed-M; ample for this tiny well-conditioned system


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:N_DEV])
    return Mesh(dev, axis_names=("lat",))


def _make_helmholtz_ops(c=COEFF):
    """Backend-oblivious ``(A_op, M_inv)`` for ``A = I + c(-Delta)``.

    ``A_op`` reads neighbours through :func:`pad_halo_latlon` (halo=1): under
    the local backend it pads the full field; under the armed spmd backend,
    called inside a shard_map, it routes to the lat-band halo body.  So the
    SAME closure drives both the serial and the sharded solve — the only
    moving part under test is how the PCG's dot products are reduced.
    ``M_inv`` is the diagonal Jacobi inverse (communication-free, identical
    on every band)."""
    diag = 1.0 + 4.0 * c

    def A_op(x):
        p = pad_halo_latlon(x, halo=1)                     # (nl+2, n_lon+2)
        interior = p[1:-1, 1:-1]
        neigh = (p[2:, 1:-1] + p[:-2, 1:-1]
                 + p[1:-1, 2:] + p[1:-1, :-2])
        return interior + c * (4.0 * interior - neigh)     # x + c(-Delta x)

    def M_inv(x):
        return x / diag

    return A_op, M_inv


def test_batch_psum_spmd_matches_serial_sum():
    """Unit: ``batch_psum_spmd`` over a lat-sharded field == the global sum."""
    mesh = _mesh()
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    rng = np.random.default_rng(5)
    a = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    b = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    ref_dot = float(jnp.sum(a * b))
    ref_sq = float(jnp.sum(a * a))

    isp = P("lat", None)
    a_sh = jax.device_put(a, NamedSharding(mesh, isp))
    b_sh = jax.device_put(b, NamedSharding(mesh, isp))

    @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
             out_specs=P(), check_vma=False)
    def _dots(aa, bb):
        local = [jnp.sum(aa * bb), jnp.sum(aa * aa)]
        d, s = batch_psum_spmd(local, "lat")
        return jnp.stack([d, s])

    out = np.asarray(_dots(a_sh, b_sh))
    assert abs(out[0] - ref_dot) < 1e-10, f"dot: {out[0]} vs {ref_dot}"
    assert abs(out[1] - ref_sq) < 1e-10, f"sq: {out[1]} vs {ref_sq}"


def test_spmd_pcg_matches_serial_pcg():
    """Sharded fixed-M PCG (psum dots) == serial fixed-M PCG (local dots).

    Same A_op, same rhs, same iteration count — they differ ONLY in how the
    global inner products are formed (one sum vs psum of per-band partials),
    so they must agree to f64 reassociation tolerance.  Proves the
    ``_global_dot_batch`` -> ``batch_psum_spmd`` routing makes the ocean
    barotropic PCG correct under the lat-band SPMD backend."""
    mesh = _mesh()
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    A_op, M_inv = _make_helmholtz_ops()
    rng = np.random.default_rng(13)
    b = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    x0 = jnp.zeros((N_LAT, N_LON))

    # Serial reference (local backend: _global_dot_batch returns the local
    # sum, which already IS the global sum in one process).
    x_serial, rr_serial = _fixed_iteration_pcg(
        A_op, b, M_inv, x0, max_iter=MAX_ITER)
    x_serial = np.asarray(x_serial)

    # Sharded solve: arm the spmd backend so (a) A_op's pad routes to the
    # band halo body and (b) _global_dot_batch routes to psum.
    isp = P("lat", None)
    b_sh = jax.device_put(b, NamedSharding(mesh, isp))
    x0_sh = jax.device_put(x0, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
                 out_specs=isp, check_vma=False)
        def _solve(bb, xx0):
            # The body must see ONLY this device's latitude band (codex:
            # a replicated in_spec would make the psum multiply dots by
            # the device count; the shape pins that the field is sharded).
            assert bb.shape[0] == NL, (bb.shape, NL)
            x, _rr = _fixed_iteration_pcg(
                A_op, bb, M_inv, xx0, max_iter=MAX_ITER)
            return x
        x_sharded = np.asarray(_solve(b_sh, x0_sh))
    finally:
        deactivate_latlon_spmd_halo()

    worst = float(np.max(np.abs(x_sharded - x_serial)))
    assert worst < 1e-10, (
        f"SPMD PCG vs serial PCG: {worst:.3e} (rr_serial={float(rr_serial):.2e})")

    # Sanity: the operator genuinely couples bands — the solution is not a
    # trivial per-band-diagonal field (else the cross-band psum would be
    # untested).  Residual of the converged serial solve must be small.
    resid = float(np.max(np.abs(np.asarray(A_op(jnp.asarray(x_serial))) - np.asarray(b))))
    assert resid < 1e-6, f"serial PCG did not converge: resid={resid:.3e}"


def test_spmd_pcg_negative_control(monkeypatch):
    """NON-VACUITY: if the cross-band reduction is a no-op the sharded PCG
    must DIVERGE from serial.  Monkeypatch ``batch_psum_spmd`` to return the
    per-band PARTIAL sums unchanged (the bug the psum routing prevents) and
    assert the gathered solution misses serial by orders of magnitude — so
    the parity test above is genuinely testing the cross-band sum (codex
    LOW: add a negative control)."""
    mesh = _mesh()
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    A_op, M_inv = _make_helmholtz_ops()
    rng = np.random.default_rng(13)
    b = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    x0 = jnp.zeros((N_LAT, N_LON))
    x_serial = np.asarray(_fixed_iteration_pcg(
        A_op, b, M_inv, x0, max_iter=MAX_ITER)[0])

    # Break the reduction: _global_dot_batch imports batch_psum_spmd at
    # function scope from this module, so patching the attribute here is
    # picked up on the next call.
    import legoesm.parallel.reductions as _red
    monkeypatch.setattr(_red, "batch_psum_spmd",
                        lambda local, axis_name: local)

    isp = P("lat", None)
    b_sh = jax.device_put(b, NamedSharding(mesh, isp))
    x0_sh = jax.device_put(x0, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
                 out_specs=isp, check_vma=False)
        def _solve(bb, xx0):
            x, _rr = _fixed_iteration_pcg(
                A_op, bb, M_inv, xx0, max_iter=MAX_ITER)
            return x
        x_bad = np.asarray(_solve(b_sh, x0_sh))
    finally:
        deactivate_latlon_spmd_halo()

    worst = float(np.max(np.abs(x_bad - x_serial)))
    assert worst > 1e-4, (
        f"negative control FAILED to diverge ({worst:.3e}) — the parity "
        f"test may be vacuous (psum not actually exercised)")
