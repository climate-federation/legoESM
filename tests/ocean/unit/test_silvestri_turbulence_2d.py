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
    # Radial energy spectrum from the IC velocity should peak near k_p=12.
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    u, v = T2.velocity_from_vorticity(jnp.asarray(zeta), kx, ky, k2_inv)
    uh = np.fft.fft2(np.asarray(u)); vh = np.fft.fft2(np.asarray(v))
    E2 = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    kmag = np.sqrt(np.asarray(k2))
    kbin = np.round(kmag).astype(int)
    Ek = np.bincount(kbin.ravel(), weights=E2.ravel(), minlength=N)
    kpeak = int(np.argmax(Ek[1:40])) + 1
    assert 8 <= kpeak <= 16, kpeak       # peak near k_p=12


def test_velocity_is_divergence_free():
    """∂ₓu+∂ᵧv == 0 SPECTRALLY (the velocity is constructed div-free; an FD
    divergence of a spectral field carries truncation error, so check spectral)."""
    N = 64
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=2))
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    u, v = T2.velocity_from_vorticity(zeta, kx, ky, k2_inv)
    div = jnp.real(jnp.fft.ifft2(1j * kx * jnp.fft.fft2(u)
                                 + 1j * ky * jnp.fft.fft2(v)))
    assert float(jnp.max(jnp.abs(div))) < 1e-12


def test_recovers_vorticity_roundtrip():
    """curl(velocity_from_vorticity(ζ)) == ζ to round-off — the sign-correct
    streamfunction inversion (ψ̂=−ζ̂/k²)."""
    N = 64
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=3))
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in T2._wavenumbers(N))
    u, v = T2.velocity_from_vorticity(zeta, kx, ky, k2_inv)
    zeta_back = jnp.real(jnp.fft.ifft2(1j * kx * jnp.fft.fft2(v)
                                       - 1j * ky * jnp.fft.fft2(u)))
    # ~1e-7 residual is the odd-derivative artifact at the (energy-free) Nyquist
    # mode of an even grid; the sign fix took the error from 5e-3 to 6e-7.
    assert float(jnp.max(jnp.abs(zeta_back - zeta))) < 1e-6


@pytest.mark.parametrize("scheme_name", ["DNS", "Leith2", "W5V", "W9V", "W9D"])
def test_step_runs_and_dissipates(scheme_name):
    """A few SSP-RK3 steps stay finite and do NOT increase energy/enstrophy
    (2D decaying turbulence: both decay; viscosity/WENO remove enstrophy)."""
    N = 64
    scheme = T2.SILVESTRI_TURB2D_SCHEMES[scheme_name]
    rhs = T2.make_rhs(scheme, Re=3.3e4, N=N)
    zeta = jnp.asarray(T2.ishiko_initial_vorticity(N, seed=4))
    ke0, ens0 = T2.total_energy_enstrophy(zeta, N)
    dt = 1.0e-3
    for _ in range(20):
        zeta = T2.step_ssp_rk3(zeta, dt, rhs)
    assert bool(jnp.all(jnp.isfinite(zeta)))
    ke1, ens1 = T2.total_energy_enstrophy(zeta, N)
    # Energy is nearly conserved (slightly decays); enstrophy decays.
    assert ke1 <= ke0 * 1.001, (scheme_name, ke0, ke1)
    assert ens1 <= ens0 * 1.001, (scheme_name, ens0, ens1)
    assert ke1 > 0.5 * ke0       # energy not destroyed (low dissipation)


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
