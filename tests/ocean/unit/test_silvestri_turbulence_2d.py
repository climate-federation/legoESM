"""Silvestri §4 2D decaying-turbulence harness: Ishiko IC, velocity-from-
vorticity, WENO/centered vorticity advection, SSP-RK3, energy/enstrophy."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import legoesm.ocean.experiments.silvestri_turbulence_2d as T2


def test_ishiko_ic_real_zero_mean_peaks_at_kp():
    """The Ishiko IC is real, zero-mean, and its energy spectrum peaks near k_p."""
    N = 128
    zeta = T2.ishiko_initial_vorticity(N, k_p=12.0, seed=1)
    assert zeta.shape == (N, N)
    assert np.isrealobj(zeta)
    assert abs(zeta.mean()) < 1e-12
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    u, v = T2.velocity_from_vorticity(jnp.asarray(zeta), kx, ky, k2_inv)
    uh = np.fft.fft2(np.asarray(u)); vh = np.fft.fft2(np.asarray(v))
    E2 = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kbin = np.round(np.sqrt(np.asarray(k2))).astype(int)
    Ek = np.bincount(kbin.ravel(), weights=E2.ravel(), minlength=N)
    kpeak = int(np.argmax(Ek[1:40])) + 1
    assert 8 <= kpeak <= 16, kpeak       # peak near k_p=12


def test_ishiko_ic_enstrophy_resolution_independent():
    """The IC carries the physical enstrophy ½⟨ζ²⟩=∫k²E dk≈8.77 (T_e≈0.33),
    resolution-INDEPENDENT — the realised amplitude is NOT tiny/N-dependent (the
    bug that would reduce §4 to linear viscous decay). rms ζ ≈ 4.2."""
    target = T2.target_enstrophy(12.0, 16.0 / 3.0)
    assert 8.0 < target < 9.5, target          # continuum value ≈ 8.77
    T_e = target ** -0.5
    assert 0.30 < T_e < 0.37, T_e               # eddy turnover ≈ 0.33
    for N in (64, 128, 256):
        zeta = T2.ishiko_initial_vorticity(N, seed=1)
        ens = 0.5 * float(np.mean(zeta ** 2))
        assert np.isclose(ens, target, rtol=1e-6), (N, ens, target)
        assert 3.5 < float(np.sqrt(np.mean(zeta ** 2))) < 5.0   # rms ζ ≈ 4.2


def test_velocity_is_fv_divergence_free():
    """The FD/flux-operator centered divergence ∂ₓu+∂ᵧv == 0 EXACTLY (the
    velocity is centered-FD of ψ, so mixed differences commute → div=0). This is
    what makes flux-form advection of a constant ζ vanish."""
    N = 64
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=2))
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    u, v = T2.velocity_from_vorticity(zeta, kx, ky, k2_inv)
    dx = 2 * np.pi / N
    div = ((jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx)
           + (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dx))
    assert float(jnp.max(jnp.abs(div))) < 1e-11


@pytest.mark.parametrize("scheme_name", ["DNS", "W5V", "W9V", "W9D"])
def test_constant_vorticity_zero_advective_tendency(scheme_name):
    """Flux-form advection of a CONSTANT ζ is zero (the FV velocity is discretely
    divergence-free) — no spurious ζ·∇·u source. (Viscosity of a constant is 0
    too, so the whole rhs vanishes for a constant field.)"""
    N = 48
    scheme = T2.SILVESTRI_TURB2D_SCHEMES[scheme_name]
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    zeta_bg = jnp.asarray(np.random.default_rng(0).standard_normal((N, N)))
    u, v = T2.velocity_from_vorticity(zeta_bg, kx, ky, k2_inv)
    const = jnp.full((N, N), 3.7)
    if scheme.advection == "weno":
        adv = T2._flux_div_weno(const, u, v, scheme.weno_order, scheme.weno_smoothness)
    else:
        adv = T2._flux_div_centered(const, u, v)
    assert float(jnp.max(jnp.abs(adv))) < 1e-9, (scheme_name, float(jnp.max(jnp.abs(adv))))


@pytest.mark.parametrize("scheme_name", ["DNS", "Leith2", "W5V", "W9V", "W9D"])
def test_step_runs_energy_conserving(scheme_name):
    """SSP-RK3 steps on the PHYSICAL IC stay finite and APPROXIMATELY CONSERVE
    ENERGY (the paper's key claim — all schemes nearly conserve KE). The WENO
    schemes additionally DISSIPATE enstrophy; the 2nd-order centered (DNS) base
    is dispersive (enstrophy may grow), so only assert enstrophy decay for WENO."""
    N = 64
    scheme = T2.SILVESTRI_TURB2D_SCHEMES[scheme_name]
    rhs = T2.make_rhs(scheme, Re=3.3e4, N=N)
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=4))
    ke0, ens0 = T2.total_energy_enstrophy(zeta, N)
    assert ke0 > 1e-3        # PHYSICAL IC (not the inert near-zero-amplitude bug)
    dt = 2.0e-3
    for _ in range(30):
        zeta = T2.step_ssp_rk3(zeta, dt, rhs)
    assert bool(jnp.all(jnp.isfinite(zeta)))
    ke1, ens1 = T2.total_energy_enstrophy(zeta, N)
    if scheme.leith_C > 0.0:
        # Leith over-damps energy (the paper's finding for C=2) — only require it
        # stays bounded and positive, not conserved.
        assert 0.3 < ke1 / ke0 <= 1.02, (scheme_name, ke0, ke1)
    else:
        # WENO / DNS nearly CONSERVE energy — the paper's headline property.
        assert 0.9 < ke1 / ke0 < 1.02, (scheme_name, ke0, ke1)
    if scheme.advection == "weno":
        assert ens1 <= ens0 * 1.001, (scheme_name, ens0, ens1)   # WENO dissipates enstrophy


def test_weno_vorticity_distinction_split_vs_standard():
    """W*V ({ζ;u}) and W*D ({ζ;ζ}) give DIFFERENT vorticity tendencies on a
    structured field (the smoothness distinction the paper measures)."""
    N = 64
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=5))
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    u, v = T2.velocity_from_vorticity(zeta, kx, ky, k2_inv)
    adv_v = T2._flux_div_weno(zeta, u, v, 9, "split")
    adv_d = T2._flux_div_weno(zeta, u, v, 9, "standard")
    assert not jnp.allclose(adv_v, adv_d, atol=1e-10)
    # Both finite.
    assert bool(jnp.all(jnp.isfinite(adv_v))) and bool(jnp.all(jnp.isfinite(adv_d)))


def test_all_seven_schemes_present():
    assert set(T2.SILVESTRI_TURB2D_SCHEMES) == {
        "DNS", "Leith1", "Leith2", "W5D", "W9D", "W5V", "W9V"}
