"""CATKE realism gate: Kato-Phillips (1969) wind-driven entrainment.

The classic 1-D mixed-layer benchmark: a constant surface wind stress on a
linearly stratified, initially-quiescent ocean drives a surface mixed layer
that DEEPENS monotonically and at a DECELERATING rate (the entrainment depth
grows ~ sqrt(t) as the deepening front works against stronger stratification).

This isolates CATKE's vertical-mixing physics in a single column:

    each step:  rho <- linear EOS(T);
                K_c <- catke_vertical_mixing(e, ..., u_star, Jb=0);
                T   <- implicit backward-Euler vertical diffusion of T by K_c;
                e   <- updated TKE.

Asserts the robust qualitative Kato-Phillips signature (not a brittle exact
coefficient): the mixed layer deepens monotonically, the deepening decelerates,
the surface layer homogenizes, and (no surface/bottom heat flux) column heat is
conserved.  This is the new-scheme realism gate for the CATKE import.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig
from legoesm.ocean.physics.vertical_mixing import catke as C
from legoesm.ocean.physics.vertical_mixing._shared import tridiag_thomas

_NLEV = 40
_DZ = 5.0          # uniform layer thickness [m] -> H = 200 m
_RHO0 = 1025.0
_ALPHA_RHO = 0.2   # d(rho)/d(T) magnitude [kg/m^3/K] -> linear stratification
_DT = 900.0        # [s]
_NSTEPS = 48       # 12 h
_USTAR = 0.012     # friction velocity [m/s] (tau ~ 0.15 Pa)


def _implicit_diffuse(T, K_iface, dz, dt):
    """Backward-Euler vertical diffusion of a cell-centred column T by an
    interface diffusivity K_iface (length nlev-1). Zero-flux top/bottom."""
    n = T.shape[-1]
    coef = K_iface / dz            # (nlev-1,) flux coefficient at interfaces
    # Sub/super-diagonals: each interface couples cells k and k+1.
    lower = -dt * coef / dz        # affects cell k+1 from interface k
    upper = -dt * coef / dz        # affects cell k from interface k
    a = jnp.concatenate([jnp.zeros((1,)), lower])          # (nlev,)
    c = jnp.concatenate([upper, jnp.zeros((1,))])          # (nlev,)
    b = 1.0 - a - c
    return tridiag_thomas(a, b, c, T)


def _mixed_layer_depth(T, dz, dT=0.05):
    """Depth [m] where T first drops dT below the surface value."""
    mixed = T[0] - T <= dT                  # True within the mixed layer
    n_mixed = jnp.sum(mixed.astype(jnp.int32))
    return float(n_mixed) * dz


def _run_kato_phillips(cfg):
    z_centre = (jnp.arange(_NLEV) + 0.5) * _DZ
    T = 15.0 - 0.04 * z_centre              # linear: warm top, cool deep (stable)
    S = jnp.full((_NLEV,), 35.0)
    u = jnp.zeros((_NLEV,))
    v = jnp.zeros((_NLEV,))
    dz_half = jnp.full((_NLEV - 1,), _DZ)
    depth_iface = (jnp.arange(_NLEV - 1) + 1.0) * _DZ      # interface depths >0
    H = _NLEV * _DZ
    hab = jnp.maximum(H - depth_iface, 0.0)
    e = jnp.full((_NLEV - 1,), cfg.minimum_tke)
    mld = []
    for _ in range(_NSTEPS):
        rho = _RHO0 - _ALPHA_RHO * (T - 10.0)
        _, K_c, e = C.catke_vertical_mixing(
            u, v, T, S, rho, dz_half, depth_iface, hab, H,
            tke_old=e, Jb=0.0, u_star=_USTAR, dt=_DT, cfg=cfg,
            rho_0=_RHO0,
        )
        T = _implicit_diffuse(T, K_c, _DZ, _DT)
        mld.append(_mixed_layer_depth(T, _DZ))
    return T, jnp.asarray(mld)


def test_kato_phillips_deepens_monotonically():
    T, mld = _run_kato_phillips(CATKEConfig())
    assert jnp.all(jnp.isfinite(mld))
    # Mixed layer forms and deepens by the end.
    assert mld[-1] > mld[0]
    assert mld[-1] >= 2.0 * _DZ                      # at least a few cells deep
    # Monotonic non-decreasing (entrainment only deepens under steady wind).
    assert jnp.all(jnp.diff(mld) >= -1e-9)


def test_kato_phillips_decelerates():
    """sqrt(t)-like: the mixed layer deepens MORE in the first half than the
    second (the deepening front works against stronger stratification)."""
    _, mld = _run_kato_phillips(CATKEConfig())
    half = _NSTEPS // 2
    growth_first = float(mld[half - 1] - mld[0])
    growth_second = float(mld[-1] - mld[half - 1])
    assert growth_first > 0.0
    assert growth_second <= growth_first + 1e-9


def test_kato_phillips_homogenizes_and_conserves_heat():
    T0_col = 15.0 - 0.04 * ((jnp.arange(_NLEV) + 0.5) * _DZ)
    T, _ = _run_kato_phillips(CATKEConfig())
    # Column heat conserved (no surface/bottom flux; mixing only redistributes).
    h0 = float(jnp.sum(T0_col) * _DZ)
    h1 = float(jnp.sum(T) * _DZ)
    assert abs(h1 - h0) / abs(h0) < 1e-6
    # Surface layer is well mixed: top few cells nearly uniform.
    assert float(jnp.max(T[:3]) - jnp.min(T[:3])) < 0.05
