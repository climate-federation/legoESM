"""RCEMIP1 (Wing et al. 2018) idealised tropical initial-condition profiles.

Reference: Wing et al. (2018), "Radiative-Convective Equilibrium
Model Intercomparison Project", Geosci. Model Dev., 11, 793–813,
doi:10.5194/gmd-11-793-2018, Table A1.

Wing 2018 analytical profiles
-----------------------------
* Surface state: ``p_sfc = 1014.8 hPa``, ``T_sfc`` set per
  RCE300/RCE295/RCE305 case (default 300 K).
* Tropopause height: ``z_t = 15 km``.
* Lapse rate: ``Γ = 0.0067 K/m`` below the tropopause; isothermal
  above.
* Surface specific humidity: ``q_sfc = 0.01865 kg/kg`` (RCE300).
* Moisture profile (z ≤ z_t)::

      q_v(z) = q_sfc · exp(-z/z_q1) · exp(-(z/z_q2)²)

  with ``z_q1 = 4 km``, ``z_q2 = 7.5 km``. Above z_t: ``q_v = q_t =
  10⁻¹¹`` (essentially dry stratosphere).
* Virtual surface temperature ``T_v0 = T_sfc · (1 + 0.608 · q_sfc)``
  defines the virtual-temperature reference for the hydrostatic
  pressure integral.
* Pressure: integrated hydrostatically from ``p_sfc`` using the
  virtual temperature profile; potential temperature follows
  ``θ = T · (p_ref/p)^κ`` (Poisson).

Returned types
--------------
:func:`wing2018_temperature_profile` and :func:`wing2018_qv_profile`
return 1D ``(nlev,)`` jax arrays at the supplied ``z`` heights.
:func:`wing2018_theta_profile` runs the hydrostatic integral
column-locally and returns θ at the same heights.
:func:`wing2018_initial_state_plane` is the column→3D broadcast
helper that produces a ``PlaneNonHydrostaticState`` with the Wing
profile imprinted on every horizontal column.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp

from legoesm import constants


# Wing 2018 Tab A1 canonical constants.
WING_GAMMA = 0.0067         # K/m, lapse rate below tropopause
WING_Z_T = 15_000.0         # m, tropopause height
# K, surface VIRTUAL temperature.  MUST be consistent with the case's SST and
# q_sfc: T_v0 = T_sfc * (1 + 0.608 * q_sfc).
#
# The previous value was a FIXED 295.0 for every SST case, on the reading that
# RCEMIP prescribes one T_v0 for all of 295/300/305 K while only q_v0 varies.
# That is internally inconsistent and made the RCE300 initial column
# SUPERSATURATED:
#
#     z (km)    T (K)     RH
#      0.0      291.7    1.396      <- 140 % relative humidity at the surface
#      1.1      285.0    1.389
#      2.3      278.1    1.361
#      4.6      263.8    1.271
#      6.8      249.0    1.163
#      9.1      233.9    1.068
#
# RCEMIP's q_0 DOES vary by case (12.00 / 18.65 / 24.00 g/kg at 295 / 300 /
# 305 K).  Pairing the RCE300 moisture (q_sfc = 0.01865) with a 295 K virtual
# temperature puts ~40 % more vapour in the column than it can hold, so the
# sounding condenses violently from the first step.  That is almost certainly
# the "qc buoyancy cascade" the CRM notes attribute to Kessler being bulk: the
# trigger is the initial state, not the microphysics.
#
# Deriving T_v0 from the case gives T_v0 = 300 * (1 + 0.608 * 0.01865)
# = 303.4 K and a surface RH of ~0.83 — a physical tropical sounding.
#
# NOT independently verified against the Wing 2018 text: the GMD PDF exceeds
# the fetch limit and the equations did not surface in search.  The argument
# here is internal consistency plus the RH test, and `test_rcemip_ic_not_
# supersaturated` is the real guard — it fails for ANY (T_v0, q_sfc) pairing
# that produces RH > 1, whichever constant a future editor prefers.
WING_T_V0_FIXED_LEGACY = 295.0   # retained for regression comparison only


WING_Z_Q1 = 4_000.0         # m, q_v lower-troposphere e-folding scale
WING_Z_Q2 = 7_500.0         # m, q_v upper-troposphere Gaussian scale
WING_Q_T = 1.0e-11          # kg/kg, stratospheric humidity floor
WING_P_SFC = 101_480.0      # Pa (1014.8 hPa per Wing Tab A1)
WING_T_SFC_DEFAULT = 300.0  # K (RCE300 case)
WING_Q_SFC_DEFAULT = 0.01865  # kg/kg (RCE300 case)

# Virtual-temperature factor: T_v = T · (1 + VIRTUAL_FACTOR · q_v).
# Derived from the molecular-weight ratio so the value stays in sync
# with `constants.R_v` / `constants.R_d`. Codex review iter-1: the
# raw 0.608 is the standard ε-derived value (1/ε - 1 = R_v/R_d - 1 ≈
# 0.608 for ε = R_d/R_v ≈ 0.622) — anchor it to the central constant.
_VIRTUAL_T_FACTOR = 1.0 / constants.epsilon - 1.0


def wing2018_T_v0(T_sfc: float = WING_T_SFC_DEFAULT,
                  q_sfc: float = WING_Q_SFC_DEFAULT) -> float:
    """Surface VIRTUAL temperature for the case, ``T_sfc*(1 + 0.608*q_sfc)``.

    Derived rather than hardcoded so T_v0 and q_sfc cannot drift apart. A
    mismatched pair is exactly what made the RCE300 initial column 140 %
    saturated; see the note on ``WING_T_V0_FIXED_LEGACY`` above.
    """
    return float(T_sfc) * (1.0 + _VIRTUAL_T_FACTOR * float(q_sfc))


# Module default for the RCE300 case, DERIVED so it cannot drift from q_sfc.
WING_T_V0 = wing2018_T_v0()



def wing2018_virtual_temperature_profile(
    z: jax.Array,
    T_v0: float = WING_T_V0,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
) -> jax.Array:
    """Wing 2018 *virtual* temperature ``T_v(z)``.

    Below the tropopause: linear lapse from the surface virtual temperature
    ``T_v0`` (RCEMIP-prescribed FIXED 295 K, Wing 2018 Tab 1 — NOT derived from
    the SST: the earlier ``T_v0 = T_sfc·(1+0.608·q_sfc)`` made the profile, incl.
    the tropopause cold point, ~8 K too warm at SST=300). Above: isothermal cap
    at ``T_v0 - Γ · z_t``. Wing 2018 Tab A1 prescribes
    this analytic profile on **virtual** T so the hydrostatic
    integral remains closed-form (the moist-air gas constant
    ``R = R_d · (1 + 0.608 q_v)`` absorbs into the virtual T,
    leaving dry-air ``R_d`` in the integrand).

    Returned by itself when callers need the virtual profile (the
    pressure integral uses it directly). Actual temperature
    ``T(z) = T_v(z) / (1 + 0.608 q_v(z))`` is the related
    :func:`wing2018_temperature_profile`.

    Parameters
    ----------
    z : jax.Array
    T_sfc, q_sfc : float
        Surface dry-air temperature [K] and specific humidity
        [kg/kg]. Defaults are the RCE300 case.
    z_t : float
        Tropopause height [m]. Default 15 km.
    Gamma : float
        Tropospheric virtual-T lapse rate [K/m]. Default 0.0067 K/m.
    """
    T_v_below = T_v0 - Gamma * z
    T_v_top = T_v0 - Gamma * z_t
    return jnp.where(z < z_t, T_v_below, T_v_top)


def wing2018_temperature_profile(
    z: jax.Array,
    T_v0: float = WING_T_V0,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
    z_q1: float = WING_Z_Q1,
    z_q2: float = WING_Z_Q2,
    q_t: float = WING_Q_T,
) -> jax.Array:
    """Actual (dry-bulb) temperature ``T(z) = T_v(z) / (1 + ε⁻¹·q_v(z))``.

    Codex review iter-1: the previous version returned T_v but
    called it T, which then propagated a virtual-T contamination
    into the Poisson θ. Split into separate virtual + actual
    helpers so callers cannot mix them up.

    Parameters mirror :func:`wing2018_virtual_temperature_profile`
    plus the q_v profile knobs ``z_q1``, ``z_q2``, ``q_t`` (needed
    to evaluate q_v(z) for the virtual→actual conversion).
    """
    T_v = wing2018_virtual_temperature_profile(
        z, T_v0=T_v0, z_t=z_t, Gamma=Gamma,
    )
    q_v = wing2018_qv_profile(
        z, q_sfc=q_sfc, z_t=z_t, z_q1=z_q1, z_q2=z_q2, q_t=q_t,
    )
    return T_v / (1.0 + _VIRTUAL_T_FACTOR * q_v)


def wing2018_qv_profile(
    z: jax.Array,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    z_q1: float = WING_Z_Q1,
    z_q2: float = WING_Z_Q2,
    q_t: float = WING_Q_T,
) -> jax.Array:
    """Wing 2018 specific humidity ``q_v(z)``.

    Below the tropopause::

        q_v(z) = q_sfc · exp(-z/z_q1) · exp(-(z/z_q2)²)

    Above the tropopause: ``q_t`` (10⁻¹¹). The two scales ``z_q1``
    and ``z_q2`` give a roughly exponential low-tropospheric decay
    that steepens upward, matching the observed tropical mean.
    """
    q_below = q_sfc * jnp.exp(-z / z_q1) * jnp.exp(-((z / z_q2) ** 2))
    return jnp.where(z < z_t, q_below, q_t)


def wing2018_pressure_profile(
    z: jax.Array,
    T_v0: float = WING_T_V0,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
    p_sfc: float = WING_P_SFC,
) -> jax.Array:
    """Pressure ``p(z)`` from hydrostatic balance with virtual T.

    Below the tropopause (linear lapse), the integral is analytical::

        p(z) = p_sfc · (1 - Γ z / T_v0)^(g / (R_d · Γ))

    Above the tropopause (isothermal), continue from ``p(z_t)``
    with the scale-height integral::

        p(z) = p(z_t) · exp(-(z - z_t) · g / (R_d · T_t))

    Uses ``R_d`` (dry-air gas constant). The virtual temperature
    absorbs the moisture correction so the troposphere integral
    stays analytical; above the tropopause moisture is negligible
    (q_v ≈ q_t = 10⁻¹¹) so dry-air ``R_d`` with ``T_t = T_v0 - Γ·z_t``
    is exact.
    """
    exp_trop = constants.g / (constants.R_d * Gamma)
    p_below = p_sfc * (1.0 - Gamma * z / T_v0) ** exp_trop
    p_at_z_t = p_sfc * (1.0 - Gamma * z_t / T_v0) ** exp_trop
    T_t = T_v0 - Gamma * z_t
    p_above = p_at_z_t * jnp.exp(
        -(z - z_t) * constants.g / (constants.R_d * T_t)
    )
    return jnp.where(z < z_t, p_below, p_above)


def wing2018_theta_profile(
    z: jax.Array,
    T_v0: float = WING_T_V0,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
    z_q1: float = WING_Z_Q1,
    z_q2: float = WING_Z_Q2,
    q_t: float = WING_Q_T,
    p_sfc: float = WING_P_SFC,
) -> jax.Array:
    """Potential temperature ``θ(z) = T(z) · (p_ref/p(z))^κ``.

    Uses ACTUAL temperature (dry-bulb) per the Poisson definition,
    NOT the virtual-T profile. The pressure integral still uses
    virtual T (analytical), but the Poisson exponent acts on the
    dry-bulb T. Codex iter-1 fix.

    Suitable as ``theta_ref_fn`` for the HeightCoordinate factories.
    """
    T = wing2018_temperature_profile(
        z, T_v0=T_v0, q_sfc=q_sfc, z_t=z_t, Gamma=Gamma,
        z_q1=z_q1, z_q2=z_q2, q_t=q_t,
    )
    p = wing2018_pressure_profile(
        z, T_v0=T_v0, q_sfc=q_sfc, z_t=z_t, Gamma=Gamma, p_sfc=p_sfc,
    )
    return T * (constants.p_ref / p) ** constants.kappa


def _make_closure(fn, **defaults):
    """Internal helper: bind defaults to a 1D-profile factory."""
    def _f(z):
        return fn(z, **defaults)
    return _f


def make_wing2018_theta_ref_fn(
    T_v0: float = WING_T_V0,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
    z_q1: float = WING_Z_Q1,
    z_q2: float = WING_Z_Q2,
    q_t: float = WING_Q_T,
    p_sfc: float = WING_P_SFC,
):
    """``z -> θ(z)`` closure for HeightCoordinate factories."""
    return _make_closure(
        wing2018_theta_profile,
        T_v0=T_v0, q_sfc=q_sfc, z_t=z_t, Gamma=Gamma,
        z_q1=z_q1, z_q2=z_q2, q_t=q_t, p_sfc=p_sfc,
    )


def make_wing2018_qv_ref_fn(
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    z_q1: float = WING_Z_Q1,
    z_q2: float = WING_Z_Q2,
    q_t: float = WING_Q_T,
):
    """``z -> q_v(z)`` closure for IC builders. Codex iter-1: matches
    :func:`make_wing2018_theta_ref_fn` so callers wire q_v and θ
    consistently from the same Wing 2018 parameter set."""
    return _make_closure(
        wing2018_qv_profile,
        q_sfc=q_sfc, z_t=z_t, z_q1=z_q1, z_q2=z_q2, q_t=q_t,
    )


def make_wing2018_pressure_ref_fn(
    T_v0: float = WING_T_V0,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
    p_sfc: float = WING_P_SFC,
):
    """``z -> p(z)`` closure for IC builders (diagnostics + reference
    pressure profile). Hydrostatic integral with virtual-T base."""
    return _make_closure(
        wing2018_pressure_profile,
        T_v0=T_v0, q_sfc=q_sfc, z_t=z_t, Gamma=Gamma, p_sfc=p_sfc,
    )


def make_wing2018_temperature_ref_fn(
    T_v0: float = WING_T_V0,
    q_sfc: float = WING_Q_SFC_DEFAULT,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
    z_q1: float = WING_Z_Q1,
    z_q2: float = WING_Z_Q2,
    q_t: float = WING_Q_T,
):
    """``z -> T(z)`` (actual, dry-bulb) closure for IC builders."""
    return _make_closure(
        wing2018_temperature_profile,
        T_v0=T_v0, q_sfc=q_sfc, z_t=z_t, Gamma=Gamma,
        z_q1=z_q1, z_q2=z_q2, q_t=q_t,
    )


def build_smooth_k1_pattern(ny: int, nx: int) -> jnp.ndarray:
    """Build the iter-203 smooth_k1 theta'-noise IC pattern.

    Returns a ``(ny, nx)`` array of ``0.5 * (cos(2π x/nx) + cos(2π y/ny))``
    with the horizontal mean explicitly subtracted to GUARANTEE zero
    mean on degenerate grids (nx=1 → cos=1 everywhere, mean=1, not
    zero-mean per the F11 fix-path-3 contract; iter-207 Codex MEDIUM
    #1 fix).

    Peak amplitude is 1.0 on healthy grids (nx, ny >= 2); the driver
    multiplies by ``theta_noise_amp`` [K] to scale.

    iter-208 promoted this helper from scripts/run_rce_mpi_long.py
    to the rcemip_initial_conditions module so the unit test can
    import it via the standard package path without loading the
    heavy driver module (mpi4jax / jax-MPI / argparse / etc.).
    """
    jj = jnp.arange(ny, dtype=jnp.float64)
    ii = jnp.arange(nx, dtype=jnp.float64)
    yy, xx = jnp.meshgrid(jj, ii, indexing="ij")
    two_pi = 2.0 * jnp.pi
    pattern = 0.5 * (
        jnp.cos(two_pi * xx / nx) + jnp.cos(two_pi * yy / ny)
    )
    return pattern - jnp.mean(pattern)
