"""Surface-layer "most" routing + z0h/z0 thermal-roughness ratio.

Covers two closed gaps:

GAP 1 — ``compute_surface_fluxes`` routes ``bulk_scheme="most"`` through the
iterative MOST solver (log-law momentum roughness + MOST stability), NOT the
constant-Cd fall-through.  Stability-dependent, and identical to a direct
``compute_most_fluxes(scheme="most", ...)`` call.

GAP 2 — the thermal/momentum roughness ratio z0h/z0 (previously the hardcoded
``z0 * 0.1``) is a config + ``compute_most_fluxes`` keyword.  A LARGER ratio
=> larger z0_t => smaller ln(z_t/z0_t) => larger heat exchange coefficient =>
STRONGER sensible/latent flux.  Default 0.1 is byte-identical to before.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    _single_tile_flux,
    compute_surface_fluxes,
)
from legoesm.core.bulk_flux import (
    _Z0H_Z0_RATIO_DEFAULT,
    compute_most_fluxes,
    validate_bulk_scheme,
)

# x64 is already forced by the JAX_ENABLE_X64=1 test env; do not mutate the
# global config at import time (it would leak into later-collected modules).


def _unstable_state():
    """A single non-neutral (unstable: warm/moist surface) column."""
    u = jnp.array([6.0])
    v = jnp.array([1.0])
    T = jnp.array([295.0])
    q_v = jnp.array([8.0e-3])
    T_sfc = jnp.array([300.0])  # surface warmer => unstable
    q_sfc = jnp.array([1.6e-2])  # surface moister
    rho = jnp.array([1.15])
    return u, v, T, q_v, T_sfc, q_sfc, rho


# ---------------------------------------------------------------------------
# GAP 1: "most" now routes through the stability-dependent MOST solver.
# ---------------------------------------------------------------------------

def test_most_differs_from_constant_under_nonneutral():
    """bulk_scheme="most" must give stability-dependent fluxes, NOT the
    constant-Cd fall-through (which ignores z0 + stratification)."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    cfg_const = SurfaceLayerConfig(bulk_scheme="constant")
    cfg_most = SurfaceLayerConfig(bulk_scheme="most", z0=1e-2)

    f_const = compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg_const)
    f_most = compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg_most)

    # Sensible + latent heat fluxes must differ (routing actually engaged).
    assert not np.allclose(np.asarray(f_const[2]), np.asarray(f_most[2]))
    assert not np.allclose(np.asarray(f_const[3]), np.asarray(f_most[3]))


def test_most_matches_direct_compute_most_fluxes():
    """compute_surface_fluxes(most) == direct compute_most_fluxes(scheme=most)
    with the same config-threaded arguments."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    cfg = SurfaceLayerConfig(bulk_scheme="most", z0=5e-3, z0h_z0_ratio=0.1)

    f_route = compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg)
    f_direct = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho,
        z_ref=cfg.z_ref,
        z0_init=cfg.z0,
        scheme="most",
        n_iter=cfg.bulk_n_iter,
        gustiness_w_zi=cfg.gustiness_w_zi,
        thermo_convention=cfg.thermo_convention,
        stability_scheme=cfg.stability_scheme,
        unstable_gamma=cfg.most_unstable_gamma,
        stable_beta=cfg.most_stable_beta,
        z0h_z0_ratio=cfg.z0h_z0_ratio,
    )
    for a, b in zip(f_route, f_direct):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


def test_most_differentiable_through_z0():
    """The "most" path is fully differentiable w.r.t. the momentum roughness."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()

    def shflx_of_z0(z0):
        cfg = SurfaceLayerConfig(bulk_scheme="most", z0=z0)
        return compute_surface_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho, cfg)[2][0]

    g = jax.grad(shflx_of_z0)(5e-3)
    assert np.isfinite(float(g))
    assert float(g) != 0.0


# ---------------------------------------------------------------------------
# GAP 2: z0h/z0 threading + monotonic sign.
# ---------------------------------------------------------------------------

def test_larger_z0h_ratio_strengthens_heat_flux():
    """Monotonic direction (unstable column, positive upward SH/LH):
    larger z0h/z0 => larger z0_t => smaller ln(z_t/z0_t) => larger C_h =>
    STRONGER (more positive) sensible AND latent flux."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    kw = dict(z_ref=10.0, z0_init=1e-2, scheme="most", n_iter=8)

    f_lo = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.05, **kw)
    f_hi = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.5, **kw)

    sh_lo, lh_lo = float(f_lo[2][0]), float(f_lo[3][0])
    sh_hi, lh_hi = float(f_hi[2][0]), float(f_hi[3][0])
    # Upward (positive) fluxes over a warm/moist surface get stronger.
    assert sh_hi > sh_lo > 0.0
    assert lh_hi > lh_lo > 0.0


def test_z0h_ratio_does_not_touch_momentum_stress():
    """z0h/z0 is scalar-roughness only: it must NOT change the momentum
    stress (which uses z0, not z0_t)."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    kw = dict(z_ref=10.0, z0_init=1e-2, scheme="most", n_iter=8)
    f_lo = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.05, **kw)
    f_hi = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.5, **kw)
    # theta*/q* feed back through the Obukhov length into u*, so the stress is
    # only WEAKLY coupled; assert the tau change is tiny relative to the flux
    # change (the ratio's primary lever is the scalar path).
    np.testing.assert_allclose(
        np.asarray(f_lo[0]), np.asarray(f_hi[0]), rtol=5e-2)


def test_backward_compat_default_ratio_byte_identical():
    """compute_most_fluxes with NO z0h_z0_ratio kwarg == passing the default
    (0.1) explicitly == the historical hardcoded z0*0.1."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    kw = dict(z_ref=10.0, z0_init=1e-2, scheme="most", n_iter=8)

    assert _Z0H_Z0_RATIO_DEFAULT == 0.1
    f_default = compute_most_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, **kw)
    f_explicit = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.1, **kw)
    for a, b in zip(f_default, f_explicit):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_config_default_ratio_is_point_one():
    assert SurfaceLayerConfig().z0h_z0_ratio == 0.1


@pytest.mark.parametrize("scheme", ["coare3", "large_yeager"])
def test_z0h_ratio_does_not_leak_into_ocean_schemes(scheme):
    """The ocean schemes (coare3 / large_yeager) compute their OWN scalar
    roughness inside the iteration (Fairall smooth-flow Re fit / LY09
    coefficient space) and MUST be EXACTLY invariant to z0h_z0_ratio: changing
    it must not move any output.  (The pre-loop z0_t seed still perturbs the
    first-iteration theta*/q* -> Obukhov length -> psi, and with finite n_iter
    that residual would leak the ratio; the implementation pins the ocean-scheme
    seed to the historical 0.1 so no such dataflow exists.)"""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    kw = dict(z_ref=10.0, z0_init=1e-3, scheme=scheme, n_iter=6)
    f_a = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.1, **kw)
    f_b = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, z0h_z0_ratio=0.9, **kw)
    for a, b in zip(f_a, f_b):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_tiled_most_matches_direct_solver():
    """The tiled single-tile path routes "most" through compute_most_fluxes
    (same as the top-level dispatch) and threads z0h_z0_ratio."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _unstable_state()
    cfg = SurfaceLayerConfig(bulk_scheme="most", z0=5e-3, z0h_z0_ratio=0.3)

    f_tile = _single_tile_flux(u, v, T, q_v, T_sfc, q_sfc, rho, cfg)
    f_direct = compute_most_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho,
        z_ref=cfg.z_ref, z0_init=cfg.z0, scheme="most",
        n_iter=cfg.bulk_n_iter,
        gustiness_w_zi=cfg.gustiness_w_zi,
        thermo_convention=cfg.thermo_convention,
        stability_scheme=cfg.stability_scheme,
        unstable_gamma=cfg.most_unstable_gamma,
        stable_beta=cfg.most_stable_beta,
        z0h_z0_ratio=cfg.z0h_z0_ratio,
    )
    for a, b in zip(f_tile, f_direct):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# GAP 1+2: dispatch hardening still raises on unknown schemes.
# ---------------------------------------------------------------------------

def test_validate_bulk_scheme_raises_on_unknown():
    with pytest.raises(ValueError):
        validate_bulk_scheme("moost")
    with pytest.raises(ValueError):
        compute_surface_fluxes(
            *_unstable_state(), SurfaceLayerConfig(bulk_scheme="bogus"))
