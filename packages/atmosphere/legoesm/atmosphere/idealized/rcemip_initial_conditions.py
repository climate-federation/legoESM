"""RCEMIP1 (Wing et al. 2018) idealised tropical initial-condition profiles.

Reference: Wing et al. (2018), "Radiative-Convective Equilibrium
Model Intercomparison Project", Geosci. Model Dev., 11, 793–813,
doi:10.5194/gmd-11-793-2018, Table A1.

Wing 2018 analytical profiles
-----------------------------
* Surface state: ``p_sfc = 1014.8 hPa``, ``T_sfc`` set per
  RCE300/RCE295/RCE305 case (default 300 K).
* Tropopause height: ``z_t = 15 km``.
* Lapse rate: ``Γ`` below the tropopause; isothermal above.
* Moisture profile (z ≤ z_t)::

      q_v(z) = q_sfc · exp(-z/z_q1) · exp(-(z/z_q2)²)

  with ``z_q1 = 4 km``, ``z_q2 = 7.5 km``. Above z_t: ``q_v = q_t =
  10⁻¹¹`` (essentially dry stratosphere).
* Virtual surface temperature ``T_v0`` sets the virtual-temperature
  reference for the hydrostatic pressure integral.
* Pressure: integrated hydrostatically from ``p_sfc`` using the
  virtual temperature profile; potential temperature follows
  ``θ = T · (p_ref/p)^κ`` (Poisson).

Which numbers come from where
-----------------------------
The *functional form* above is Wing 2018 Tab A1.  The three free
constants ``(T_v0, Γ, q_sfc)`` are **calibrated against the gSAM
oracle** — gSAM 1.8.8 ``CASES/RCEMIP1/snd_rcemip_300s6.11.2``, the
sounding SAM itself starts RCEMIP1 from.  Each constant records its
estimator and the spread of an independent second estimator below.

The oracle also shows the analytic form's **structural limit**: gSAM's
stratosphere WARMS with height (194.4 K at the 14.5 km cold point →
207.5 K at 20 km → 231.2 K at 33 km), which a two-piece
lapse-then-isothermal profile cannot represent at any ``(T_v0, Γ)``.
For a faithful RCEMIP1 column use the tabulated sounding —
``scripts/run/run_rcemip_plane.py --sounding <gSAM snd>`` — which goes
through the same ``read_sam_snd`` + ``sam_case_setup`` path as
BOMEX/RICO/DYCOMS/GATE.  These analytic profiles remain the default and
are the right tool for a cheap, smooth, differentiable idealised column.

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


# --- Wing 2018 Tab A1 analytic-FORM constants (not case calibration) ---
# These define the shape of the two-piece profile and are unchanged.
WING_Z_T = 15_000.0         # m, tropopause height
WING_Z_Q1 = 4_000.0         # m, q_v lower-troposphere e-folding scale
WING_Z_Q2 = 7_500.0         # m, q_v upper-troposphere Gaussian scale
WING_Q_T = 1.0e-11          # kg/kg, stratospheric humidity floor
WING_P_SFC = 101_480.0      # Pa (1014.8 hPa per Wing Tab A1)
WING_T_SFC_DEFAULT = 300.0  # K, prescribed RCE300 SEA-SURFACE temperature. This
                            # is the surface BOUNDARY CONDITION for the flux
                            # code, NOT the initial surface AIR temperature —
                            # gSAM's RCE300 sounding starts the air at 296.92 K
                            # at z=37 m, ~3 K below the SST (air-sea
                            # disequilibrium).  Do NOT re-derive T_v0 from it;
                            # see the note on WING_T_V0.

# --- RCE300 case calibration, measured against the gSAM oracle ---------------
# Oracle: gSAM 1.8.8 CASES/RCEMIP1/snd_rcemip_300s6.11.2
#   sha256 973d50a113b18f2da6f3eeba51494af771ce868d68af035c2b3899eb3734341b
#   (`snd` in that deck is a byte-identical copy; the deck also ships
#    snd_rcemip_295s6.11.2 and snd_rcemip_305s6.11.2, so gSAM uses a DIFFERENT
#    sounding per SST rather than one profile across cases).
# Measured 2026-08-06, SLURM job 9331613, 74 native levels, x64.  Every value
# below carries the estimator that produced it AND the spread of an independent
# second estimator, so no constant rests on a single unchecked fit.
#
# Oracle surface state (z = 37 m): T = 296.917 K, q_v = 14.0703 g/kg,
# RH = 0.7545; max RH over the column = 0.8892 at z = 667 m; cold point
# 194.42 K at 14.5 km.
#
# K/m, tropospheric lapse rate.  Least squares on T_v(z) over the 38 oracle
# levels with z <= z_t (R^2 = 0.99567).  Independent 2-point secant
# (37 m -> 10 km) gives 0.0073219 — a 1.1 % spread.
# Was 0.0067 (Wing analytic-form value): 10 % too weak against the oracle.
WING_GAMMA = 0.0074034
# K, surface VIRTUAL temperature: the INTERCEPT of the same least-squares fit
# (independent 2-point estimator 299.727 K — a 0.24 % spread).
#
# Two previous values, both now disproven directly against gSAM's sounding
# rather than by argument:
#   295.0  — on the reading that RCEMIP prescribes one fixed T_v0 for all SSTs.
#            Paired with q_sfc it put the initial column at RH = 1.396 at the
#            surface; gSAM's is 0.7545.  The three per-SST soundings gSAM ships
#            also contradict the "one profile for all cases" premise.
#   303.4  — from T_v0 = T_sfc*(1 + 0.608*q_sfc), i.e. virtualising the SST.
#            That conflates the SEA-surface temperature with the surface AIR
#            virtual temperature; the oracle's air is ~3 K cooler than the SST,
#            so this overshoots by ~2.7 K.  T_v0 is therefore NOT derived from
#            T_sfc here — deriving it is precisely the error.
WING_T_V0 = 300.444
# kg/kg, surface (z=0) water-vapour mixing ratio.  The oracle's 37 m value
# 14.0703 g/kg extrapolated to z=0 through the Wing shape function
# exp(-z/z_q1)*exp(-(z/z_q2)^2), which reproduces the 37 m level EXACTLY.
# A least-squares fit of q_0 over the whole troposphere gives 14.7729 g/kg
# (a 4.0 % spread) — larger than the other two constants' spreads because
# gSAM's moisture profile is not exactly the Wing shape; the surface-anchored
# estimator is used since the surface state is what the gate asserts.
# Was 0.01865: 31 % too moist, and ~= q_sat(300 K) — the SATURATION value had
# been used where the case calls for a subsaturated ~14.2 g/kg.
WING_Q_SFC_DEFAULT = 0.0142014

# Virtual-temperature factor: T_v = T · (1 + VIRTUAL_FACTOR · q_v).
# Derived from the molecular-weight ratio so the value stays in sync
# with `constants.R_v` / `constants.R_d`. Codex review iter-1: the
# raw 0.608 is the standard ε-derived value (1/ε - 1 = R_v/R_d - 1 ≈
# 0.608 for ε = R_d/R_v ≈ 0.622) — anchor it to the central constant.
_VIRTUAL_T_FACTOR = 1.0 / constants.epsilon - 1.0


def wing2018_virtual_temperature_profile(
    z: jax.Array,
    T_v0: float = WING_T_V0,
    z_t: float = WING_Z_T,
    Gamma: float = WING_GAMMA,
) -> jax.Array:
    """Wing 2018 *virtual* temperature ``T_v(z)``.

    Below the tropopause: linear lapse from the surface virtual temperature
    ``T_v0`` (the gSAM-oracle least-squares intercept — NOT derived from the
    SST; see the ``WING_T_V0`` note, where both the fixed-295 K and the
    SST-virtualised readings are disproven against gSAM's own sounding).
    Above: isothermal cap at ``T_v0 - Γ · z_t``.  gSAM's stratosphere in fact
    WARMS with height, which this two-piece form cannot represent — use
    ``--sounding`` when stratospheric structure matters.  Wing 2018 Tab A1
    prescribes
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
        Tropospheric virtual-T lapse rate [K/m]. Default ``WING_GAMMA``
        (oracle-fitted; see the constant's note).
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
