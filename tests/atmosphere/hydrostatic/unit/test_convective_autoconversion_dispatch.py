"""Scheme-level dispatch + physical-autoconversion split for the mass-flux
schemes that expose the plume updraft cloud water (Bechtold, Tiedtke).

Locks:
  1. ``precip_split_scheme='autoconversion'`` produces a finite, mass-conserving
     rain source derived from the plume ``q_c_u`` (the Sundqvist-1978 physical
     split replacing the constant ``precip_efficiency``);
  2. an unknown ``precip_split_scheme`` raises at scheme entry
     (dispatch-hardening) rather than silently running different physics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection import config as C
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection
from legoesm.thermo import saturation_specific_humidity

_NCOL, _NLEV = 2, 16


def _column():
    """A moist, conditionally-unstable column (upper levels near saturation) so
    the plume produces condensate the split can act on."""
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, _NLEV + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    ph = jnp.broadcast_to((sh * p_s)[None, :], (_NCOL, _NLEV + 1))
    pf = jnp.broadcast_to((sf * p_s)[None, :], (_NCOL, _NLEV))
    T = jnp.maximum(302.0 * jnp.clip(sf, 0.01, None) ** 0.19, 200.0)
    T = jnp.broadcast_to(T[None, :], (_NCOL, _NLEV))
    q_sat = saturation_specific_humidity(T, pf)
    q_v = jnp.where(sf[None, :] > 0.6, 0.98, 0.6) * q_sat
    u = jnp.broadcast_to(jnp.linspace(10.0, 2.0, _NLEV)[None, :], (_NCOL, _NLEV))
    v = jnp.full((_NCOL, _NLEV), 1.0)
    return T, q_v, pf, ph, u, v


def _bechtold(**cfg_kw):
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    out, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
        None, 1800.0, C.BechtoldConfig(subsidence_solve="implicit_flux", **cfg_kw),
        moisture_convergence=jnp.zeros_like(T))
    return out


def _tiedtke(**cfg_kw):
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    out, _ = tiedtke_convection(
        T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), 1800.0,
        C.TiedtkeConfig(**cfg_kw), moisture_convergence=jnp.zeros_like(T))
    return out


@pytest.mark.parametrize("runner", [_bechtold, _tiedtke])
def test_autoconversion_emits_finite_conserving_rain(runner):
    base = runner(precip_split_scheme="constant", precip_efficiency=0.0)
    auto = runner(precip_split_scheme="autoconversion",
                  autoconv_q_c_crit=5.0e-4, autoconv_pe_max=0.9)
    assert auto.dq_r_conv_dt is not None
    assert bool(jnp.all(jnp.isfinite(auto.dq_r_conv_dt)))
    assert bool(jnp.all(auto.dq_r_conv_dt >= 0.0))
    # MASS: anvil + rain == the positive detrained condensate the plume made
    # (the split introduces no source/sink; heat/vapor tendencies untouched).
    base_pos = jnp.maximum(base.dq_c_conv_dt, 0.0)
    total = auto.dq_c_conv_dt + auto.dq_r_conv_dt
    np.testing.assert_allclose(np.asarray(total), np.asarray(base_pos),
                               rtol=1e-5, atol=1e-20)
    # a moist column exceeds the critical loading somewhere -> some rain forms
    assert float(jnp.sum(auto.dq_r_conv_dt)) > 0.0


@pytest.mark.parametrize("runner", [_bechtold, _tiedtke])
def test_unknown_split_scheme_raises(runner):
    with pytest.raises(ValueError, match="precip_split_scheme"):
        runner(precip_split_scheme="garbage")


@pytest.mark.parametrize("runner", [_bechtold, _tiedtke])
def test_constant_default_is_legacy_noop(runner):
    # default precip_split_scheme='constant' + precip_efficiency=0 -> no rain
    out = runner()
    assert out.dq_r_conv_dt is None


def test_higher_pe_max_rains_more():
    """The emergent precip efficiency scales with pe_max: a larger ceiling sends
    a larger fraction of the same detrained condensate to rain."""
    lo = _bechtold(precip_split_scheme="autoconversion",
                   autoconv_q_c_crit=5.0e-4, autoconv_pe_max=0.5)
    hi = _bechtold(precip_split_scheme="autoconversion",
                   autoconv_q_c_crit=5.0e-4, autoconv_pe_max=0.95)
    assert float(jnp.sum(hi.dq_r_conv_dt)) > float(jnp.sum(lo.dq_r_conv_dt))
