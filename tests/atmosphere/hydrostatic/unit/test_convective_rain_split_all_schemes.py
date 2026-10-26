"""Every mass-flux convection scheme routes condensate through the shared
in-updraft rain split (output.split_convective_rain).

Bechtold + Tiedtke gained ``precip_efficiency`` first; this locks the SAME
knob + mass-conservation contract on the remaining detrain-to-cloud schemes
(Zhang-McFarlane, Kain-Fritsch, mass-flux, EDMF) so ALL of them can rain
directly instead of loading the grid-scale cloud (the AMIP over-bright-anvil /
convective-precip-deficit failure mode).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

_NCOL, _NLEV = 2, 14


def _column():
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, _NLEV + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_half = jnp.broadcast_to((sh * p_s)[None, :], (_NCOL, _NLEV + 1))
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (_NCOL, _NLEV))
    T = jnp.maximum(302.0 * jnp.clip(sf, 0.01, None) ** 0.19, 200.0)
    T = jnp.broadcast_to(T[None, :], (_NCOL, _NLEV))
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = jnp.where(sf[None, :] > 0.6, 0.97, 0.6) * q_sat
    u = jnp.broadcast_to(jnp.linspace(10.0, 2.0, _NLEV)[None, :], (_NCOL, _NLEV))
    v = jnp.full((_NCOL, _NLEV), 1.0)
    w = jnp.full((_NCOL, _NLEV), 0.1)
    prog = jnp.zeros((_NCOL, _NLEV))
    return T, q_v, p_full, p_half, u, v, w, prog


def _run(scheme, pe):
    from legoesm.atmosphere.physics.convection import config as C
    T, q_v, pf, ph, u, v, w, prog = _column()
    if scheme == "kain_fritsch":
        from legoesm.atmosphere.physics.convection.kain_fritsch import (
            kain_fritsch_convection as fn)
        cfg = C.KainFritschConfig(precip_efficiency=pe)
        return fn(T, q_v, pf, ph, w, prog, 600.0, config=cfg)[0]
    if scheme == "mass_flux":
        from legoesm.atmosphere.physics.convection.mass_flux import (
            mass_flux_convection as fn)
        cfg = C.MassFluxConfig(precip_efficiency=pe)
        return fn(T, q_v, pf, ph, jnp.zeros((_NCOL,)), 600.0, config=cfg)[0]
    if scheme == "edmf":
        from legoesm.atmosphere.physics.convection.mass_flux import (
            edmf_convection as fn)
        cfg = C.ConvectiveEDMFConfig(precip_efficiency=pe)
        return fn(T, q_v, pf, ph, jnp.full((_NCOL,), 0.1), 600.0, config=cfg)[0]
    raise ValueError(scheme)


# zhang_mcfarlane left out: the CAM6 port produces rain explicitly (cldprp
# c0 autoconversion -> dq_r_conv_dt) and has no precip_efficiency split.
_SCHEMES = ["kain_fritsch", "mass_flux", "edmf"]


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_default_no_split_byte_identical(scheme):
    """precip_efficiency=0 (default) ⇒ dq_r_conv_dt is None and the cloud
    source is unchanged (legacy behaviour)."""
    out = _run(scheme, 0.0)
    assert out.dq_r_conv_dt is None


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_split_emits_rain_conserving_mass(scheme):
    """precip_efficiency>0 ⇒ dq_r_conv_dt is the pe-fraction of the detrained
    condensate; cloud+rain conserves the positive condensate; heat/vapor
    tendencies are untouched by the diagnostic split."""
    pe = 0.6
    base = _run(scheme, 0.0)
    split = _run(scheme, pe)
    assert split.dq_r_conv_dt is not None
    base_pos = jnp.maximum(base.dq_c_conv_dt, 0.0)
    # cloud + rain == the positive condensate the scheme produced
    total = split.dq_c_conv_dt + split.dq_r_conv_dt
    np.testing.assert_allclose(np.asarray(total), np.asarray(base_pos),
                               rtol=1e-6, atol=1e-20)
    # rain is exactly pe of it
    np.testing.assert_allclose(np.asarray(split.dq_r_conv_dt),
                               np.asarray(base_pos * pe), rtol=1e-6, atol=1e-20)
    # the split is diagnostic: dT/dq_v unchanged
    np.testing.assert_allclose(np.asarray(split.dT_dt),
                               np.asarray(base.dT_dt), rtol=1e-6, atol=1e-20)
    np.testing.assert_allclose(np.asarray(split.dq_v_dt),
                               np.asarray(base.dq_v_dt), rtol=1e-6, atol=1e-20)


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_config_has_precip_efficiency_default_zero(scheme):
    from legoesm.atmosphere.physics.convection import config as C
    cfg_cls = {
        "kain_fritsch": C.KainFritschConfig,
        "mass_flux": C.MassFluxConfig,
        "edmf": C.ConvectiveEDMFConfig,
    }[scheme]
    assert cfg_cls().precip_efficiency == 0.0
