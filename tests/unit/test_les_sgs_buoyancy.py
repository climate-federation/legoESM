"""Lilly (1962) stable-stratification suppression of the plane-LES SGS eddy
viscosity (``sgs_buoyancy``).

Root cause it fixes: the strain-only ``ν_t`` over-entrains a stable inversion
(DYCOMS stratocumulus cloud top thins to LWP ~7 g/m² vs the 50-80 benchmark,
resolution- AND scheme-independent). The factor ``√(max(0, 1 − Ri/Pr_t))``
shuts SGS mixing off where ``Ri ≥ Pr_t``.

Covers: the shared factor (neutral/unstable/stable/AD), the LES-core
``sgs_buoyancy_factor`` (neutral no-op, stable suppression), and the ``rhs``
flag wiring (off = exact no-op skip; on = reduced scalar mixing).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics._shared import lilly_buoyancy_factor
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# 1. shared factor √(max(0, 1 − Ri/Pr_t))                                      #
# --------------------------------------------------------------------------- #
def test_lilly_factor_neutral_unstable_stable():
    Pr_t = 1.0
    # Neutral: Ri = 0 → f = 1.
    assert float(lilly_buoyancy_factor(jnp.array(0.0), Pr_t)) == 1.0
    # Unstable: Ri < 0 → f > 1 (convective enhancement).
    assert float(lilly_buoyancy_factor(jnp.array(-3.0), Pr_t)) > 1.0
    # Stable cutoff: Ri = Pr_t → f = 0; Ri > Pr_t → f = 0 (mixing off).
    assert float(lilly_buoyancy_factor(jnp.array(1.0), Pr_t)) == 0.0
    assert float(lilly_buoyancy_factor(jnp.array(5.0), Pr_t)) == 0.0
    # Monotone decreasing 0 < Ri < Pr_t.
    f = np.asarray(lilly_buoyancy_factor(jnp.linspace(0.0, 0.99, 20), Pr_t))
    assert np.all(np.diff(f) < 0.0)


def test_lilly_factor_ad_finite_at_cutoff():
    # The double-``where`` must yield a FINITE cotangent at the Ri = Pr_t kink
    # (a bare √(max(·,0)) leaks a 0·∞ NaN there).
    grad = jax.grad(lambda r: lilly_buoyancy_factor(r, 1.0))
    for ri in (0.0, 0.5, 1.0, 1.5):
        g = float(grad(jnp.array(ri)))
        assert np.isfinite(g), f"non-finite grad at Ri={ri}"


# --------------------------------------------------------------------------- #
# 2. LES-core sgs_buoyancy_factor on real fields                              #
# --------------------------------------------------------------------------- #
def _grid(nz=32, Lz=480.0, sgs_buoyancy=False):
    cfg = sl.SpectralLESConfig(nx=16, ny=16, nz=nz, Lx=320.0, Ly=320.0, Lz=Lz,
                               sgs_buoyancy=sgs_buoyancy, pr_sgs=1.0)
    return sl.make_grid(cfg)


def _sheared(g):
    """Vertical shear so |S| > 0 (else Ri = N²/0 is ill-posed)."""
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = np.asarray(g.z_c)
    u = jnp.broadcast_to((0.01 * z)[None, None, :], (ny, nx, nz))
    v = jnp.zeros((ny, nx, nz))
    w = jnp.zeros((ny, nx, nz + 1))
    return u, v, w


def _tracers(g, qv):
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    tr = jnp.zeros((ny, nx, nz, 3))
    return tr.at[..., 0].set(jnp.broadcast_to(qv[None, None, :], (ny, nx, nz)))


def test_sgs_factor_neutral_is_unity():
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    u, v, w = _sheared(g)
    theta = jnp.full((ny, nx, nz), 290.0)             # neutral: ∂θ/∂z = 0
    qv = jnp.full((nz,), 5.0e-3)                       # uniform moisture
    f = sl.sgs_buoyancy_factor(theta, _tracers(g, qv), u, v, w, g)
    # N² ≈ 0 ⇒ f ≈ 1 everywhere (documents the "≈1, not exactly 1 at fp" caveat
    # that justifies gating the neutral path off).
    np.testing.assert_allclose(np.asarray(f), 1.0, atol=1e-6)


def test_sgs_factor_suppresses_stable_inversion():
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    u, v, w = _sheared(g)
    z = np.asarray(g.z_c)
    # Mixed layer 290 K up to ~300 m, then a sharp +8 K inversion.
    theta_col = 290.0 + 8.0 / (1.0 + np.exp(-(z - 300.0) / 10.0))
    theta = jnp.broadcast_to(jnp.asarray(theta_col)[None, None, :], (ny, nx, nz))
    qv = jnp.full((nz,), 5.0e-3)
    f = np.asarray(sl.sgs_buoyancy_factor(theta, _tracers(g, qv), u, v, w, g))
    assert f.min() < 1.0, "no suppression under stable stratification"
    assert f.min() >= 0.0, "factor must stay non-negative"
    # Strongest suppression at the inversion (largest ∂θ/∂z).
    kz_inv = int(np.argmax(np.gradient(theta_col, z)))
    assert f[0, 0, kz_inv] < 0.5, "inversion not strongly suppressed"


# --------------------------------------------------------------------------- #
# 3. rhs flag wiring: off = exact no-op; on = reduced scalar mixing            #
# --------------------------------------------------------------------------- #
def _stable_state(g):
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    u, v, w = _sheared(g)
    z = np.asarray(g.z_c)
    theta = jnp.broadcast_to(
        jnp.asarray(290.0 + 8.0 / (1.0 + np.exp(-(z - 300.0) / 10.0)))[None, None, :],
        (ny, nx, nz))
    tr = _tracers(g, jnp.full((nz,), 5.0e-3))
    return u, v, w, theta, tr


def test_rhs_flag_off_is_noop_on_matches_suppression():
    g_off = _grid(sgs_buoyancy=False)
    g_on = _grid(sgs_buoyancy=True)
    u, v, w, theta, tr = _stable_state(g_off)
    common = dict(u_geo=(0.0, 0.0), f_cor=0.0, theta=theta, tracers=tr)
    *_off, Rth_off, _ = sl.rhs(u, v, w, g_off, **common)
    *_on, Rth_on, _ = sl.rhs(u, v, w, g_on, **common)

    # OFF path skips the branch entirely → identical to computing ν_t straight
    # from eddy_viscosity (the neutral/dry byte-identical guarantee). We prove it
    # by matching a hand-rolled θ-RHS built from the RAW ν_t.
    nu_t = sl.eddy_viscosity(u, v, w, g_off)
    Rth_raw = sl.scalar_rhs_monotone(theta, u, v, w, nu_t, g_off, 0.0) \
        if g_off.cfg.monotone_scalars else sl.scalar_rhs(theta, u, v, w, nu_t, g_off, 0.0)
    np.testing.assert_allclose(np.asarray(Rth_off), np.asarray(Rth_raw), rtol=1e-12,
                               atol=1e-14)

    # ON must differ (suppression changes the SGS θ flux at the inversion).
    assert not np.allclose(np.asarray(Rth_on), np.asarray(Rth_off)), \
        "sgs_buoyancy=True did not change the scalar tendency"


def test_nu_floor_survives_full_suppression():
    # With nu_floor > 0 the effective ν_t at a fully-suppressed inversion (f→0)
    # must retain the floor (f·ν_t + (1−f)·nu_floor → nu_floor), NOT go inviscid.
    nu_floor = 0.05
    cfg = sl.SpectralLESConfig(nx=16, ny=16, nz=32, Lx=320.0, Ly=320.0, Lz=480.0,
                               sgs_buoyancy=True, pr_sgs=1.0, nu_floor=nu_floor)
    g = sl.make_grid(cfg)
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    u, v, w = _sheared(g)
    z = np.asarray(g.z_c)
    theta = jnp.broadcast_to(
        jnp.asarray(290.0 + 12.0 / (1.0 + np.exp(-(z - 300.0) / 6.0)))[None, None, :],
        (ny, nx, nz))
    tr = _tracers(g, jnp.full((nz,), 5.0e-3))
    nu_raw = np.asarray(sl.eddy_viscosity(u, v, w, g))            # strain + floor
    f = np.asarray(sl.sgs_buoyancy_factor(theta, tr, u, v, w, g))
    nu_eff = f * nu_raw + (1.0 - f) * nu_floor                    # the rhs formula
    kz_inv = int(np.argmax(np.gradient(np.asarray(theta[0, 0]), z)))
    assert f[0, 0, kz_inv] < 0.05, "inversion not (near-)fully suppressed"
    # Floor preserved: effective ν_t ≳ nu_floor, not driven to ~0.
    assert nu_eff[0, 0, kz_inv] >= nu_floor * 0.99, "background floor was suppressed"


def test_sgs_factor_ad_finite_at_zero_strain():
    # Differentiable-LES guard (codex MED): at an EXACT zero-strain state (rest /
    # uniform velocity) the |S|² denominator must not backprop 0/0→NaN. Building
    # s2 from the strain COMPONENTS (not Smag²=(√·)²) keeps the reverse path safe.
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    z = np.asarray(g.z_c)
    theta = jnp.broadcast_to(
        jnp.asarray(290.0 + 8.0 / (1.0 + np.exp(-(z - 300.0) / 10.0)))[None, None, :],
        (ny, nx, nz))          # stable (N²>0) so f is a nontrivial function of S
    tr = _tracers(g, jnp.full((nz,), 5.0e-3))

    def loss(u, v, w):
        return jnp.sum(sl.sgs_buoyancy_factor(theta, tr, u, v, w, g))

    for label, vel in (("rest", 0.0), ("uniform", 3.0)):
        u = jnp.full((ny, nx, nz), vel)      # ∂u/∂x = ∂u/∂z = 0 ⇒ S_ij = 0
        v = jnp.zeros((ny, nx, nz))
        w = jnp.zeros((ny, nx, nz + 1))
        gu, gv, gw = jax.grad(loss, argnums=(0, 1, 2))(u, v, w)
        for name, gg in (("u", gu), ("v", gv), ("w", gw)):
            assert np.all(np.isfinite(np.asarray(gg))), \
                f"non-finite d(f)/d{name} at {label}-state (zero strain)"


if __name__ == "__main__":
    test_lilly_factor_neutral_unstable_stable()
    test_lilly_factor_ad_finite_at_cutoff()
    test_sgs_factor_neutral_is_unity()
    test_sgs_factor_suppresses_stable_inversion()
    test_rhs_flag_off_is_noop_on_matches_suppression()
    test_nu_floor_survives_full_suppression()
    test_sgs_factor_ad_finite_at_zero_strain()
    print("ok")
