"""Adiabatic-parcel validation of the SDM condensation + coupling chain.

A closed parcel rising at constant updraft must: conserve total water
(q_v + q_l) to round-off; develop a supersaturation that peaks then relaxes as
droplets grow; grow its droplets and liquid water content; and cool along the
moist adiabat (warmer than the dry adiabat because of latent heating). This is
the classic warm-cloud activation parcel and exercises condensation.py +
coupling.py + box_model.py together.

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/microphysics/unit/test_sdm_parcel.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import (
    relative_humidity,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
)
from legoesm.atmosphere.physics.microphysics.sdm import (
    ParcelState,
    SDMConfig,
    SuperDropletState,
    liquid_mixing_ratio,
    run_parcel,
    saturation_ratio,
)


def _initial_parcel(T0=283.15, p0=9.0e4, rh0=0.98, n_sd=200, r0=1.0e-6,
                    N_per_m3=5.0e7):
    """Saturated-ish parcel seeded with monodisperse cloud droplets."""
    q_sat0 = float(saturation_mixing_ratio(jnp.asarray(T0), jnp.asarray(p0)))
    q_v0 = rh0 * q_sat0
    rho_air = p0 / (constants.R_d * T0)
    xi_per_kg = N_per_m3 / rho_air / n_sd   # real droplets per kg air per super-droplet
    ones = jnp.ones((n_sd,))
    droplets = SuperDropletState(
        multiplicity=ones * xi_per_kg,
        radius=ones * r0,
        solute_mass=ones * 0.0,
        active=ones,
    )
    return ParcelState(droplets=droplets, T=jnp.asarray(T0), p=jnp.asarray(p0),
                       q_v=jnp.asarray(q_v0), z=jnp.asarray(0.0))


def test_saturation_ratio_is_vapor_pressure_based():
    """S must be e/e_sat (what droplet growth needs), not the mixing-ratio
    ratio q_v/q_sat — the two differ by ~1%, a large error in the S-1 signal."""
    T, p, q_v = 283.15, 9.0e4, 8.0e-3
    parcel = ParcelState(
        droplets=None, T=jnp.asarray(T), p=jnp.asarray(p),
        q_v=jnp.asarray(q_v), z=jnp.asarray(0.0))
    e = p * q_v / (constants.epsilon + q_v)
    S_exact = e / float(saturation_vapor_pressure(jnp.asarray(T)))
    assert float(saturation_ratio(parcel)) == pytest.approx(S_exact, rel=1e-12)
    assert float(relative_humidity(jnp.asarray(T), jnp.asarray(p),
                                   jnp.asarray(q_v))) == pytest.approx(S_exact, rel=1e-12)
    # and it is measurably different from the naive mixing-ratio ratio q_v/q_sat
    # (the two coincide exactly at saturation and diverge away from it; the
    # supersaturation *slope* dS/dq differs by ~1%, biasing droplet growth).
    S_naive = q_v / float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p)))
    assert abs(S_exact - S_naive) / S_exact > 5e-4


def test_parcel_conserves_total_water():
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    q_t0 = float(parcel0.q_v + liquid_mixing_ratio(parcel0.droplets, 1.0))
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    q_t = np.asarray(hist["q_t"])
    assert np.max(np.abs(q_t - q_t0)) / q_t0 < 1e-8   # total water conserved


def test_parcel_supersaturation_peaks_then_relaxes():
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    S = np.asarray(hist["S"])
    S_max = S.max()
    assert S_max > 1.0                     # supersaturation develops
    i_max = int(S.argmax())
    assert i_max < len(S) - 1              # peak is interior
    assert S[-1] < S_max                   # relaxes after the peak
    # Quantitative: peak supersaturation is small and physical (a few %),
    # and the parcel equilibrates back toward saturation after the peak.
    assert 0.0 < (S_max - 1.0) < 0.1
    assert abs(S[-1] - 1.0) < 0.05


def test_parcel_droplets_and_lwc_grow():
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    r0 = float(parcel0.droplets.radius[0])
    q_l0 = float(liquid_mixing_ratio(parcel0.droplets, 1.0))
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    assert float(final.droplets.radius[0]) > r0          # droplets grew
    q_l_final = float(liquid_mixing_ratio(final.droplets, 1.0))
    assert q_l_final > q_l0                               # LWC increased
    # Condensed liquid equals the vapor lost (total-water closure), and the
    # parcel ends near the adiabatic LWC q_t - r_sat(T,p) (within the residual
    # supersaturation): q_l_final is a sizeable fraction of q_t - r_sat.
    q_t = float(parcel0.q_v) + q_l0
    r_sat_final = float(saturation_mixing_ratio(final.T, final.p))
    q_l_adiabatic = q_t - r_sat_final
    assert q_l_adiabatic > 0.0
    assert 0.5 < q_l_final / q_l_adiabatic < 1.5


def test_parcel_follows_moist_adiabat():
    """Final T is below the start (ascent cools) but above the dry-adiabatic
    value (latent heating warms the parcel)."""
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=2)
    parcel0 = _initial_parcel()
    T0 = float(parcel0.T)
    final, hist = run_parcel(parcel0, w=3.0, dt=0.05, n_steps=1000, cfg=cfg)
    z = float(final.z)
    T_dry = T0 - constants.g / constants.c_pd * z   # dry-adiabatic descent of T
    assert float(final.T) < T0
    assert float(final.T) > T_dry


def test_parcel_jit_static_argnames():
    """The advertised run_parcel(..., n_steps, cfg, M_air) API works under
    jax.jit with n_steps and cfg marked static."""
    import functools
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    parcel0 = _initial_parcel()
    run = jax.jit(run_parcel, static_argnames=("n_steps", "cfg"))
    final, hist = run(parcel0, 2.0, 0.05, 200, cfg, 1.0)
    assert jnp.isfinite(final.T) and jnp.isfinite(final.q_v)
    assert np.asarray(hist["S"]).shape == (200,)
