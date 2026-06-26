"""Morrison sub-grid in-cloud warm-rain closure (Morrison & Gettelman 2008).

``MorrisonConfig.subgrid_autoconversion`` evaluates autoconversion + accretion
on the IN-CLOUD water ``q_c/cf`` and scales the tendency back by the cloud
fraction ``cf`` (Sundqvist √-form from local RH).  Because the warm-rain rates
are strongly non-linear in cloud water (KK2000 PRC ∝ q_c^2.47), this ENHANCES
drizzle in partly-filled boxes relative to the grid-mean evaluation.

Pins:
  * default OFF is bit-identical (cf_eff ≡ 1 ⇒ grid-mean) — byte-reproducible;
  * ON changes the rain tendency in a partly-saturated column (cf < 1) and the
    sign is an ENHANCEMENT of autoconversion (more rain produced);
  * outputs stay finite.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    make_zero_hydrometeors,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.thermo import saturation_mixing_ratio
from legoesm import constants

jax.config.update("jax_enable_x64", True)


def _warm_partial_cloud_columns(rh=0.85, ncol=4, nlev=10):
    """Warm columns with cloud + rain water and RH in (rh_crit, 1) so the
    sub-grid cloud fraction cf < 1 (the regime the closure modifies)."""
    T = jnp.full((ncol, nlev), 290.0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = dp / (rho * constants.g)
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = q_sat * rh                       # RH in (rh_crit, 1) -> cf < 1
    h = make_zero_hydrometeors(ncol, nlev)
    h = h._replace(
        q_c=jnp.full_like(T, 1.0e-3),
        q_r=jnp.full_like(T, 1.0e-4),
        N_c=jnp.full_like(T, 1.0e8),
        N_r=jnp.full_like(T, 1.0e3),
    )
    return T, q_v, h, p_full, p_half, rho, dz


def test_subgrid_off_is_identity_default():
    """Default config (subgrid off) == explicit subgrid_autoconversion=False."""
    args = _warm_partial_cloud_columns()
    out_default = morrison_microphysics(*args, dt=10.0)
    out_off = morrison_microphysics(
        *args, dt=10.0, config=MorrisonConfig(subgrid_autoconversion=False))
    assert np.array_equal(np.asarray(out_default.dq_r_dt),
                          np.asarray(out_off.dq_r_dt))


def test_subgrid_on_changes_and_enhances_rain():
    """In a cf<1 column the in-cloud closure enhances autoconversion: the
    column-integrated rain production rises vs the grid-mean evaluation."""
    args = _warm_partial_cloud_columns()
    out_off = morrison_microphysics(
        *args, dt=10.0, config=MorrisonConfig(subgrid_autoconversion=False))
    out_on = morrison_microphysics(
        *args, dt=10.0, config=MorrisonConfig(subgrid_autoconversion=True))
    dqr_off = np.asarray(out_off.dq_r_dt)
    dqr_on = np.asarray(out_on.dq_r_dt)
    assert np.all(np.isfinite(dqr_on))
    # Must actually respond to the switch.
    assert not np.allclose(dqr_off, dqr_on)
    # Enhancement: more cloud water converted to rain in the cloudy fraction.
    assert float(np.sum(dqr_on)) > float(np.sum(dqr_off))


def test_subgrid_cf_floor_caps_enhancement():
    """A higher cf floor caps the in-cloud enhancement, so its rain production
    sits between the grid-mean (off) and the low-floor (full) closure."""
    args = _warm_partial_cloud_columns()
    off = float(np.sum(np.asarray(morrison_microphysics(
        *args, dt=10.0,
        config=MorrisonConfig(subgrid_autoconversion=False)).dq_r_dt)))
    low_floor = float(np.sum(np.asarray(morrison_microphysics(
        *args, dt=10.0,
        config=MorrisonConfig(subgrid_autoconversion=True,
                              subgrid_cf_min=0.1)).dq_r_dt)))
    high_floor = float(np.sum(np.asarray(morrison_microphysics(
        *args, dt=10.0,
        config=MorrisonConfig(subgrid_autoconversion=True,
                              subgrid_cf_min=0.9)).dq_r_dt)))
    assert off <= high_floor <= low_floor
