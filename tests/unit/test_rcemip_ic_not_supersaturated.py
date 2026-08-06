"""The RCEMIP initial sounding must not be supersaturated.

REGRESSION. The RCE300 initial column shipped at ~140 % relative humidity for
an unknown period: ``WING_T_V0`` was pinned to a FIXED 295.0 K for every SST
case while ``q_sfc`` stayed at the RCE300 value 0.01865 kg/kg. Those two are
not independent — pairing SST-300 moisture with an SST-295-like virtual
temperature puts ~40 % more vapour in the column than it can hold.

Measured before the fix:

    z (km)    T (K)     RH
     0.0      291.7    1.396
     1.1      285.0    1.389
     2.3      278.1    1.361
     4.6      263.8    1.271
     6.8      249.0    1.163
     9.1      233.9    1.068

A CRM started from that state condenses violently on the first step. The CRM
notes attribute the resulting blow-up to a "qc buoyancy cascade" and to Kessler
being a bulk scheme that switches abruptly at saturation; the microphysics is
doing the right thing with an impossible initial state.

This test is deliberately written against the PHYSICAL invariant rather than
against a particular constant: it recomputes RH from the sounding the module
actually produces, so it fails for ANY (T_v0, q_sfc) pairing that supersaturates
— whichever value a future editor decides is correct. That matters here because
this constant has already been "corrected" once in the wrong direction, with a
confident docstring justifying it.
"""

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.idealized import rcemip_initial_conditions as ic
from legoesm.thermo import saturation_mixing_ratio


# Physical tropical soundings sit well below saturation in the mean; RCEMIP's
# analytic profile is ~0.8 at the surface. Allow a small margin for the
# saturation-curve formulation, but nothing that could hide a 1.4.
_RH_CEILING = 1.02


def _sounding(n=60, z_top=30_000.0):
    z = jnp.linspace(0.0, z_top, n)
    q_v = ic.wing2018_qv_profile(z)
    T_v = ic.wing2018_virtual_temperature_profile(z)
    T = T_v / (1.0 + (1.0 / constants.epsilon - 1.0) * q_v)
    p = ic.wing2018_pressure_profile(z)
    return z, T, p, q_v


def test_initial_column_is_not_supersaturated():
    z, T, p, q_v = _sounding()
    rh = q_v / saturation_mixing_ratio(T, p)
    worst = float(jnp.max(rh))
    k = int(jnp.argmax(rh))
    assert worst <= _RH_CEILING, (
        f"RCEMIP initial sounding is SUPERSATURATED: max RH={worst:.3f} at "
        f"z={float(z[k]):.0f} m (T={float(T[k]):.2f} K, "
        f"p={float(p[k])/100:.1f} hPa, q_v={float(q_v[k]):.5f}). A CRM started "
        f"here condenses on step 1 and never reaches RCE. Check that "
        f"WING_T_V0 and WING_Q_SFC_DEFAULT describe the SAME case: "
        f"T_v0 = T_sfc*(1 + 0.608*q_sfc)."
    )


def test_surface_rh_is_physical():
    """Not merely <=1 — a tropical surface should be sub-saturated, not at 100 %."""
    _, T, p, q_v = _sounding()
    rh0 = float((q_v / saturation_mixing_ratio(T, p))[0])
    assert 0.6 <= rh0 <= 1.0, (
        f"surface RH={rh0:.3f} is not a physical tropical value (~0.8)")


def test_T_v0_is_consistent_with_q_sfc():
    """The invariant the regression violated, stated directly."""
    expected = ic.WING_T_SFC_DEFAULT * (
        1.0 + (1.0 / constants.epsilon - 1.0) * ic.WING_Q_SFC_DEFAULT)
    assert ic.WING_T_V0 == pytest.approx(expected, rel=1e-9), (
        f"WING_T_V0={ic.WING_T_V0} is inconsistent with T_sfc="
        f"{ic.WING_T_SFC_DEFAULT} and q_sfc={ic.WING_Q_SFC_DEFAULT} "
        f"(expected {expected:.4f})")


def test_the_legacy_value_would_fail_this_gate():
    """NON-VACUITY. Prove the gate catches the shipped regression.

    Without this, a future edit could satisfy the tests above trivially. Here
    the old fixed 295.0 K is fed back in and MUST supersaturate — if it does
    not, the gate is not measuring what it claims.
    """
    z = jnp.linspace(0.0, 30_000.0, 60)
    q_v = ic.wing2018_qv_profile(z)
    T_v = ic.wing2018_virtual_temperature_profile(
        z, T_v0=ic.WING_T_V0_FIXED_LEGACY)
    T = T_v / (1.0 + (1.0 / constants.epsilon - 1.0) * q_v)
    p = ic.wing2018_pressure_profile(z, T_v0=ic.WING_T_V0_FIXED_LEGACY)
    rh_legacy = float(jnp.max(q_v / saturation_mixing_ratio(T, p)))
    assert rh_legacy > 1.2, (
        f"the legacy T_v0=295 K sounding gives max RH={rh_legacy:.3f}; it was "
        f"measured at ~1.40, so this gate is no longer reproducing the "
        f"regression it exists to catch")
