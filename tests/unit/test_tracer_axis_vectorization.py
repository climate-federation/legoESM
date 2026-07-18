"""Per-slot parity for the vmapped tracer paths (loop → trailing-axis vmap).

The tracer loops in the pseudo-incompressible and spectral plane LES were
trace-time Python loops (one kernel set per slot). They are now batched
with ``jax.vmap`` over the trailing tracer axis; the invariant is that
slot ``k`` of the batched result equals the scalar operator applied to
slot ``k`` alone — checked here against the SAME shared operators (no
re-derived numerics).
"""

from __future__ import annotations

import jax
import numpy as np
from legoesm.atmosphere.dynamics.les import pseudo_incompressible_plane as pip
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

jax.config.update("jax_enable_x64", True)

_NY, _NX, _NZ, _NT = 6, 8, 6, 3


def _rand(key, shape, amp=1.0):
    return amp * jax.random.normal(jax.random.PRNGKey(key), shape)


def _pi_grid():
    cfg = pip.PseudoIncompressibleConfig(
        nx=_NX, ny=_NY, nz=_NZ, Lx=8_000.0, Ly=6_000.0, Lz=3_000.0,
    )
    return pip.make_grid(cfg)


def test_tracer_sgs_matches_per_slot():
    g = _pi_grid()
    tracers = 1e-3 * (1.0 + _rand(0, (_NY, _NX, _NZ, _NT), 0.1))
    nu_t = 0.5 * (1.0 + 0.1 * _rand(1, (_NY, _NX, _NZ), 1.0)) ** 2
    sfc_qv_flux = 3.0e-5
    out = pip._tracer_sgs(tracers, nu_t, g, sfc_qv_flux)
    kh = nu_t / g.cfg.pr_sgs
    for k in range(_NT):
        q = tracers[..., k]
        ref = (pip._ddx_c(kh * pip._ddx_c(q, g.dx), g.dx)
               + pip._ddy_c(kh * pip._ddy_c(q, g.dy), g.dy)
               + pip._ddz_c(kh * pip._ddz_c(q, g.dz), g.dz))
        if k == 0:
            ref = ref.at[..., 0].add(sfc_qv_flux / g.dz)
        np.testing.assert_allclose(
            np.asarray(out[..., k]), np.asarray(ref),
            rtol=0.0, atol=1e-14, err_msg=f"slot {k}",
        )


def test_tracer_advection_matches_per_slot():
    g = _pi_grid()
    u = _rand(2, (_NY, _NX, _NZ), 1.0)
    v = _rand(3, (_NY, _NX, _NZ), 1.0)
    w = _rand(4, (_NY, _NX, _NZ + 1), 0.1).at[..., 0].set(0.0).at[..., -1].set(0.0)
    theta = 300.0 + _rand(5, (_NY, _NX, _NZ), 0.5)
    tracers = 1e-3 * (1.0 + _rand(6, (_NY, _NX, _NZ, _NT), 0.1))
    _au, _av, _aw, _ath, atr = pip.tendencies(u, v, w, theta, tracers, g)
    # Slot k transported alone must equal slot k of the batch (the same
    # shared scalar-advection operator per slot — no re-derivation).
    for k in range(_NT):
        _, _, _, _, atr_k = pip.tendencies(
            u, v, w, theta, tracers[..., k:k + 1], g,
        )
        np.testing.assert_allclose(
            np.asarray(atr[..., k]), np.asarray(atr_k[..., 0]),
            rtol=0.0, atol=1e-14, err_msg=f"slot {k}",
        )


def test_spectral_rhs_tracers_match_per_slot():
    cfg = sl.SpectralLESConfig(
        nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0,
        buoyancy=True, theta_ref0=300.0, moist=True, n_tracers=_NT,
        spectral_filter=False,
    )
    g = sl.make_grid(cfg)
    u = _rand(7, (8, 8, 8), 1.0)
    v = _rand(8, (8, 8, 8), 1.0)
    w = _rand(9, (8, 8, 9), 0.1).at[..., 0].set(0.0).at[..., -1].set(0.0)
    theta = 300.0 + _rand(10, (8, 8, 8), 0.5)
    tracers = 1e-3 * (1.0 + 0.1 * _rand(11, (8, 8, 8, _NT), 1.0))
    sfc_qv_flux = 3.0e-5
    *_, r_tracers = sl.rhs(
        u, v, w, g, (0.0, 0.0), 0.0, theta=theta,
        tracers=tracers, sfc_qv_flux=sfc_qv_flux,
    )
    nu_t = sl.eddy_viscosity(u, v, w, g)
    scalar_fn = (sl.scalar_rhs_monotone if g.cfg.monotone_scalars
                 else sl.scalar_rhs)
    for k in range(_NT):
        flx = sfc_qv_flux if k == 0 else 0.0
        ref = scalar_fn(tracers[..., k], u, v, w, nu_t, g, flx)
        np.testing.assert_allclose(
            np.asarray(r_tracers[..., k]), np.asarray(ref),
            rtol=0.0, atol=1e-13, err_msg=f"slot {k}",
        )


def test_filt_state_qv_only_and_all():
    cfg = sl.SpectralLESConfig(
        nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0,
        n_tracers=2, spectral_filter=True,
    )
    g = sl.make_grid(cfg)
    u = _rand(12, (8, 8, 8), 1.0)
    v = _rand(13, (8, 8, 8), 1.0)
    w = _rand(14, (8, 8, 9), 0.1)
    th = 300.0 + _rand(15, (8, 8, 8), 0.5)
    tr = 1e-3 * (1.0 + 0.1 * _rand(16, (8, 8, 8, 2), 1.0))
    # all-slot filtering (monotone_scalars=False)
    _, _, _, _, tr_f = sl._filt_state(u, v, w, th, tr, g)
    for k in range(2):
        np.testing.assert_allclose(
            np.asarray(tr_f[..., k]),
            np.asarray(sl._apply_filter(tr[..., k], g)),
            rtol=0.0, atol=1e-14, err_msg=f"slot {k}",
        )
    # q_v-only filtering (monotone scalars + filter_monotone_qv)
    g2 = g._replace(cfg=g.cfg._replace(
        monotone_scalars=True, filter_monotone_qv=True,
        filter_monotone_scalars=False,
    ))
    _, _, _, _, tr_q = sl._filt_state(u, v, w, th, tr, g2)
    np.testing.assert_allclose(
        np.asarray(tr_q[..., 0]),
        np.asarray(sl._apply_filter(tr[..., 0], g2)),
        rtol=0.0, atol=1e-14,
    )
    np.testing.assert_array_equal(
        np.asarray(tr_q[..., 1]), np.asarray(tr[..., 1]),
    )
