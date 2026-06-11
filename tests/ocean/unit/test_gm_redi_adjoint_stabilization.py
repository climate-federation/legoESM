"""Tests for ``GMRediConfig.adjoint_stabilization`` (#22 — long-horizon
reverse-mode gradient stabilization of the GM/Redi isoneutral operator).

Mechanism (probe-verified, .physics-validator/gm_adjoint_stab/RESULTS.md):
the slope saturation bounds the PRIMAL fluxes but not the LINEARIZED
operator — in the DM95 taper transition band d(taper*S)/dS exceeds the
primal coefficient bound, so the one-step tangent/adjoint propagator of the
explicit triad cross-terms amplifies (|G| > 1) where the primal is stable,
and long-horizon parameter adjoints explode (~x2-5/step on the ACC recipe).
``stop_gradient`` on the slope coefficients (Picard-frozen linearization) is
primal-invisible by construction and removes the amplification.

Pins here:
  1. PRIMAL BIT-IDENTITY: every mode produces bit-identical tendencies and
     K33 (stop_gradient is the identity in the primal).
  2. NON-VACUITY / MUTATION PIN: a manufactured taper-transition-band case
     where the EXACT-AD one-step Jacobian amplifies (spectral radius > 1)
     and the STABILIZED one does not. mode="none" IS the
     removed-stop_gradient mutant: if someone deletes the stop_gradient,
     the stabilized rows re-explode and the test goes red.
  3. Gradients still FLOW under stabilization (kappa and tracer-diffusion
     sensitivity nonzero/finite) while the slope/taper feedback gradient is
     exactly zero.
  4. Dispatch discipline: unknown literal raises ValueError at every entry.

Run: JAX_ENABLE_X64=1 python -m pytest
     tests/ocean/unit/test_gm_redi_adjoint_stabilization.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    dm95_taper,
    dm95_taper_scalar,
    validate_adjoint_stabilization,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_isoneutral_K33_latlon,
    compute_isopycnal_slopes_latlon_cgrid,
    gm_redi_tracer_tendency_latlon,
)

MODES = ("none", "stop_gradient_taper", "stop_gradient_slopes")
SG_MODES = ("stop_gradient_taper", "stop_gradient_slopes")

DT = 4800.0      # ACC recipe dt
S_MAX = 0.01     # recipe iso_slopec
WIDTH = 0.5      # recipe taper_width_frac
GAMMA = 8.0e-3   # vertical T gradient [K/m]
EOS_LIN = LinearEOSConfig()


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# Manufactured taper-transition-band channel (tiny; CPU-friendly)
# ---------------------------------------------------------------------------


def _setup(n_lat=4, n_lon=6, nlev=10):
    """Small flat-bottom channel with stretched-ish uniform vertical grid."""
    from legoesm.ocean.vertical import create_ocean_z_star

    grid, _wall = create_regional_latlon_grid(
        n_lat=n_lat, n_lon=n_lon,
        lat_south=-40.0, lat_north=-40.0 + 2.0 * n_lat,
        lon_west=0.0, lon_east=2.0 * n_lon, periodic_x=True,
    )
    # surface dz ~ 20 m like the ACC vertical (the conditional-stability
    # margin kappa*S^2*dt/dz^2 ~ 1 lives in the thin surface layers).
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=1900.0, dz_surface=20.0, dz_deep=400.0,
    )
    nlat, nlon = grid.n_lat, grid.n_lon          # builder adds 2 wall rows
    H_bathy = jnp.full((nlat, nlon), float(jnp.sum(z_coord.dz_ref)))
    eta = jnp.zeros((nlat, nlon))
    mask = jnp.ones((nlat, nlon)).at[0, :].set(0.0).at[-1, :].set(0.0)
    return grid, z_coord, H_bathy, eta, mask


def _manufactured_T(grid, z_coord, alpha):
    """T with isoneutral slope |S_y| = alpha*S_MAX (linear EOS: S=-T_y/T_z)."""
    z_c = jnp.asarray(z_coord.z_full_ref)
    y = jnp.asarray(grid.lat) * float(grid.radius)
    y = y - y[0]
    Ty = -alpha * S_MAX * GAMMA
    T_of_y = Ty * y
    T = (20.0 + GAMMA * z_c)[None, None, :] + T_of_y[:, None, None]
    return jnp.broadcast_to(T, (grid.n_lat, grid.n_lon, z_c.shape[0]))


def _cfg(mode, kappa=1000.0, implicit_K33=False):
    return GMRediConfig(
        kappa_GM=kappa, kappa_Redi=kappa, S_max=S_MAX,
        taper_width_frac=WIDTH, slope_scheme="triads",
        slope_density="neutral", implicit_K33=implicit_K33,
        K_iso_steep=0.0, adjoint_stabilization=mode,
    )


def _gm_step(mode, T, S, grid, z_coord, H_bathy, eta, mask, kappa=1000.0):
    """One explicit GM/Redi Euler step of T (the manufactured-band map)."""
    cfg = _cfg(mode, kappa=kappa, implicit_K33=False)
    dT, _dS = gm_redi_tracer_tendency_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg,
        eos="linear", eos_linear=EOS_LIN, mask=mask,
    )
    return T + DT * dT * mask[:, :, None]


# ---------------------------------------------------------------------------
# 1. Primal bit-identity
# ---------------------------------------------------------------------------


def test_primal_bit_identity_tendency_and_k33():
    grid, z_coord, H_bathy, eta, mask = _setup()
    T = _manufactured_T(grid, z_coord, alpha=1.3)
    S = jnp.full_like(T, 35.0)
    ref_dT = ref_dS = ref_K33 = None
    for mode in MODES:
        cfg = _cfg(mode, implicit_K33=True)
        dT, dS = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg,
            eos="linear", eos_linear=EOS_LIN, mask=mask)
        K33 = compute_isoneutral_K33_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg,
            eos="linear", eos_linear=EOS_LIN, mask=mask)
        if ref_dT is None:
            ref_dT, ref_dS, ref_K33 = dT, dS, K33
        else:
            # BIT identity, not allclose: stop_gradient is the identity.
            assert jnp.array_equal(dT, ref_dT), mode
            assert jnp.array_equal(dS, ref_dS), mode
            assert jnp.array_equal(K33, ref_K33), mode


def test_primal_bit_identity_centered_slopes():
    grid, z_coord, H_bathy, eta, mask = _setup()
    T = _manufactured_T(grid, z_coord, alpha=1.0)
    S = jnp.full_like(T, 35.0)
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.vertical import compute_ocean_jacobian

    eos_fn = make_eos_fn("linear", EOS_LIN)
    jac = compute_ocean_jacobian(eta, H_bathy, z_coord)
    rho = eos_fn(T, S, jnp.zeros_like(T))
    outs = []
    for mode in MODES:
        cfg = _cfg(mode)
        outs.append(compute_isopycnal_slopes_latlon_cgrid(
            rho, mask, z_coord, jac, grid, cfg,
            T=T, S=S, eos_fn=eos_fn))
    for sx, sy, tp in outs[1:]:
        assert jnp.array_equal(sx, outs[0][0])
        assert jnp.array_equal(sy, outs[0][1])
        assert jnp.array_equal(tp, outs[0][2])


def test_dm95_helpers_primal_identity_and_zero_grad():
    S = jnp.linspace(-3 * S_MAX, 3 * S_MAX, 41)
    t0 = dm95_taper_scalar(S, S_MAX, transition_width_frac=WIDTH)
    t1 = dm95_taper_scalar(S, S_MAX, transition_width_frac=WIDTH,
                           stop_gradient_taper=True)
    assert jnp.array_equal(t0[0], t1[0]) and jnp.array_equal(t0[1], t1[1])
    g0 = jax.grad(lambda s: jnp.sum(dm95_taper_scalar(
        s, S_MAX, transition_width_frac=WIDTH)[1]))(S)
    g1 = jax.grad(lambda s: jnp.sum(dm95_taper_scalar(
        s, S_MAX, transition_width_frac=WIDTH,
        stop_gradient_taper=True)[1]))(S)
    assert jnp.any(g0 != 0.0)
    assert jnp.all(g1 == 0.0)
    # vector variant
    a0 = dm95_taper(S, S, S_MAX, transition_width_frac=WIDTH)
    a1 = dm95_taper(S, S, S_MAX, transition_width_frac=WIDTH,
                    stop_gradient_taper=True)
    for x, y in zip(a0, a1):
        assert jnp.array_equal(x, y)


# ---------------------------------------------------------------------------
# 2. Non-vacuity / mutation pin: transition-band tangent amplification
# ---------------------------------------------------------------------------


def test_taper_band_tangent_amplification_pin():
    """Manufactured taper-transition-band case: exact AD amplifies, the
    stabilized modes do not.

    With slopes at 1.3*S_max (inside the DM95 transition band) and the ACC
    kappa*dt/dz^2 margin, the one-step Jacobian's spectral radius under
    exact AD is >> 1 (measured ~2 on this tiny channel) while both
    stop_gradient modes sit at ~1 (the frozen-coefficient operator is the
    primal-stable diffusion).  MUTATION PIN: mode="none" *is* the
    deleted-stop_gradient mutant — if the stop_gradient is removed from the
    stabilized paths they become "none" and the <= 1.05 assertions fail.
    """
    grid, z_coord, H_bathy, eta, mask = _setup(n_lat=4, n_lon=6, nlev=10)
    T0 = _manufactured_T(grid, z_coord, alpha=1.3)
    S = jnp.full_like(T0, 35.0)

    def rho_of(mode):
        f = lambda x: _gm_step(mode, x.reshape(T0.shape), S, grid, z_coord,
                               H_bathy, eta, mask).ravel()
        J = jax.jacrev(f)(T0.ravel())
        return float(np.max(np.abs(np.linalg.eigvals(np.asarray(J)))))

    rho_none = rho_of("none")
    rho_taper = rho_of("stop_gradient_taper")
    rho_slopes = rho_of("stop_gradient_slopes")
    # exact AD amplifies in the band (the adjoint instability)
    assert rho_none > 1.2, rho_none
    # the stabilized linearizations are bounded by the (stable) primal
    assert rho_taper <= 1.05, rho_taper
    assert rho_slopes <= 1.05, rho_slopes


def test_no_amplification_well_inside_slope_bound():
    """Control: slopes at 0.2*S_max (taper' ~ 0) — exact AD is stable too,
    i.e. the pin above is the *band* mechanism, not generic AD noise."""
    grid, z_coord, H_bathy, eta, mask = _setup(n_lat=4, n_lon=6, nlev=10)
    T0 = _manufactured_T(grid, z_coord, alpha=0.2)
    S = jnp.full_like(T0, 35.0)
    f = lambda x: _gm_step("none", x.reshape(T0.shape), S, grid, z_coord,
                           H_bathy, eta, mask).ravel()
    J = jax.jacrev(f)(T0.ravel())
    rho = float(np.max(np.abs(np.linalg.eigvals(np.asarray(J)))))
    assert rho <= 1.05, rho


# ---------------------------------------------------------------------------
# 3. Gradients still flow under stabilization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", SG_MODES)
def test_gradients_flow_through_stabilized_operator(mode):
    grid, z_coord, H_bathy, eta, mask = _setup()
    T = _manufactured_T(grid, z_coord, alpha=1.0)
    S = jnp.full_like(T, 35.0)

    def loss_kappa(kappa):
        cfg = _cfg(mode, kappa=kappa)
        dT, _ = gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, cfg,
            eos="linear", eos_linear=EOS_LIN, mask=mask)
        return jnp.sum((T + DT * dT) ** 2)

    g = jax.grad(loss_kappa)(1000.0)
    assert jnp.isfinite(g) and g != 0.0

    def loss_T(T_in):
        dT, _ = gm_redi_tracer_tendency_latlon(
            T_in, S, eta, H_bathy, grid, z_coord, _cfg(mode),
            eos="linear", eos_linear=EOS_LIN, mask=mask)
        return jnp.sum((T_in + DT * dT) ** 2)

    gT = jax.grad(loss_T)(T)
    assert bool(jnp.all(jnp.isfinite(gT))) and bool(jnp.any(gT != 0.0))


# ---------------------------------------------------------------------------
# 4. Dispatch discipline
# ---------------------------------------------------------------------------


def test_unknown_literal_raises():
    with pytest.raises(ValueError, match="adjoint_stabilization"):
        validate_adjoint_stabilization("sg_taper")  # typo'd literal

    grid, z_coord, H_bathy, eta, mask = _setup()
    T = _manufactured_T(grid, z_coord, alpha=1.0)
    S = jnp.full_like(T, 35.0)
    bad = _cfg("none")._replace(adjoint_stabilization="typo")
    with pytest.raises(ValueError, match="adjoint_stabilization"):
        gm_redi_tracer_tendency_latlon(
            T, S, eta, H_bathy, grid, z_coord, bad,
            eos="linear", eos_linear=EOS_LIN, mask=mask)


def test_default_config_is_none():
    assert GMRediConfig().adjoint_stabilization == "none"
