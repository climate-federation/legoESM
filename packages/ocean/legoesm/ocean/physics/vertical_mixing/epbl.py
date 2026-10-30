"""Energetic Planetary Boundary Layer (ePBL) vertical mixing.

A JAX port of MOM6's ``MOM_energetic_PBL.F90`` (Reichl & Hallberg, 2018,
*Ocean Modelling* 132, 112-129), selectable as
``VerticalMixingConfig(scheme="epbl")`` alongside ``kpp`` and ``catke``.

Credits
-------
ePBL was created by Dr. Brandon G. Reichl and Dr. Robert Hallberg (NOAA GFDL)
as the ``MOM_energetic_PBL`` module of MOM6.  This module is a JAX port of
that Fortran code, modified for legoESM (level convention, fixed-trip-count
iterations for differentiability, legoESM's operator splitting):

    https://github.com/mom-ocean/MOM6/blob/main/src/parameterizations/vertical/MOM_energetic_PBL.F90

The machine-learned diffusivity options (the ``eqdisc_*`` equation-discovery
shape function and velocity scales) are due to Dr. Aakash Sane (Sane et al.
2023, 2026); their equation numbers below follow the MOM6 source comments.

Single-column model and the MOM6 step order
-------------------------------------------
By default legoESM's ocean single-column model, like KPP, evaluates ePBL on
the column BEFORE the step's surface flux is applied.  MOM6 applies the
surface flux to the top layer first and then evaluates ePBL on the updated
column.  ``EPBLConfig.k_after_surface_flux=True`` selects MOM6's order in the
single-column model (every other driver raises), so the legoESM
single-column model can be compared directly with the MOM6 single-column
model.  The single-column test case against MOM6 (50 m column, 1 m layers,
40N, 0.2 N/m^2 wind, -200/0/+200 W/m^2, 8 h) agrees to ~1e-3 degC with the
MOM6 order; see ``docs/ocean/specs/epbl_energetic_pbl.md`` and the ordering
tests in ``tests/ocean/unit/test_epbl.py``.

References
----------
- Reichl, B. G., & Hallberg, R. (2018). A simplified energetics based
  planetary boundary layer (ePBL) approach for ocean climate simulations.
  *Ocean Modelling*, 132, 112-129. https://doi.org/10.1016/j.ocemod.2018.10.004
- Reichl, B. G., & Li, Q. (2019). A parameterization with a constrained
  potential energy conversion rate of vertical mixing due to Langmuir
  turbulence. *Journal of Physical Oceanography*, 49(11), 2935-2959.
  https://doi.org/10.1175/JPO-D-18-0258.1
- Sane, A., Reichl, B. G., Adcroft, A., & Zanna, L. (2023). Parameterizing
  vertical mixing coefficients in the ocean surface boundary layer using
  neural networks. *Journal of Advances in Modeling Earth Systems*, 15(10),
  e2023MS003890. https://doi.org/10.1029/2023MS003890
- Sane, A., Reichl, B. G., Adcroft, A., & Zanna, L. (2026). Machine learned
  equations for vertical mixing coefficients in the ocean surface boundary
  layer. *Geophysical Research Letters*, 53(10), e2026GL122106.
  https://doi.org/10.1029/2026GL122106

What the scheme does
--------------------
ePBL sets the boundary-layer diffusivity from an energy budget rather than
from a prescribed shape.  Each timestep it

1. converts the surface wind stress into a mechanical TKE input
   ``mech_TKE = mstar * rho_0 * u_star^3 * dt`` and the surface buoyancy
   forcing into a convectively released potential energy ``conv_PErel``;
2. walks DOWN the column.  At every interface it proposes a diffusivity
   ``Kd = vonKar * vstar * mixlen``, prices the potential-energy change that
   applying it implicitly would cost, and pays for that change out of the
   remaining TKE.  Where the energy runs out, mixing stops -- and that depth
   IS the diagnosed boundary-layer depth;
3. iterates on the boundary-layer depth, because the mixing-length shape
   depends on the depth that step 2 produces (``use_mld_iteration``).

The energy accounting makes the scheme unconditionally stable in the sense
that it can never mix more than the forcing supplies.

Level convention
----------------
Surface-to-bottom, matching the rest of ``legoesm.ocean``: layer ``k=0`` is
the surface layer, interface ``K`` lies directly above layer ``K``.  MOM6's
1-based interface ``K`` maps to this module's ``K-1``.

Differentiability
-----------------
Every control-flow branch that depends on model state is a ``jnp.where``, and
both of MOM6's iterations are fixed-trip-count ``lax.fori_loop``s with a
"frozen once converged" carry, which reproduces MOM6's early ``exit``
numerically while staying reverse-differentiable.  Consequently
``max_mld_its`` and ``max_kd_its`` are WORK, not just caps: unlike MOM6, where
``EPBL_MLD_MAX_ITS=500`` costs nothing once the iteration converges after ~10
passes, here every trip is executed.  Size them from the tolerance
(``ceil(log2(depth / mld_tol))`` plus a margin for bisection), not from the
MOM6 cap.

KNOWN DIFFERENCES: RULE 5 smooth switches (issue #2180)
-------------------------------------------------------
MOM6's switches on traced state at the five places the MOM6 single-column
fixture exercises (each site is marked ``# RULE5:<id>``) are smooth by
default, using ``legoesm.core.smooth`` (softplus / logsumexp / sigmoid).  Each
unit class has one sharpness field on ``EPBLConfig``; ``math.inf`` gives
MOM6's exact switch, and at that limit the port reproduces its exact-port gap
to MOM6 (``tests/ocean/unit/test_epbl_mom6_oracle.py``).  Unlike MOM6, the
smooth forms are not exact at their kinks: ``max(0, 0)`` is ``ln 2 / sharpness``.

- ``smooth_bflux_sharpness_s3m2`` (default 1e10 s^3/m^2, width 1e-10 m^2/s^3):
  ``max(0, B)**2`` in the REICHL_H18 ``mstar_S`` (C1r) and ``min(0, B)`` in
  the MSTAR_CONV_ADJ reduction (C2).  At ``B = 0`` the stabilising mstar term
  is ``cs1 * ((ln2 * width)**2 * BLD / (u*^5 f))**cs2`` instead of 0 (the
  ``cs2 = 0.4`` power amplifies the width), and the convective reduction is
  slightly below 1, so a neutral surface gets slightly MORE mechanical TKE
  (deeper mixing) than in MOM6.  This is the largest gap: the fixture's Q = 0 case moves
  from 2.6e-13 / 1.0e-12 degC (dt 1200 / 300 s) to 3.5e-3 / 4.0e-3 degC at 8 h.
- ``smooth_ratio_sharpness`` (default 1e3, width 1e-3): the
  ``sigma_max`` clip to [0.1, 0.7] of the equation-discovery shape (C3);
  largest fixture change 5.7e-5 relative (Q = -200, dt 300 s).
- ``smooth_colht_sharpness_per_m`` (default 1e10 1/m, width 1e-10 m):
  ``min(ColHt_chg, 0)`` in ``_find_pe_chg_orig``'s PE change (C5).  A zero
  diffusivity now prices a PE change of ``pres_z * ln2 * width`` instead of 0,
  and every ``ColHt_chg`` within a few widths of 0 (either sign) is charged a
  little more, so mixing there is slightly MORE expensive than in MOM6.  The
  kernel's other three outputs (dPEc_dKd, dPE_max, dPEc_dKd_0) keep MOM6's
  exact ``min(., 0)`` (smoothing them moves no fixture gap beyond the test
  tolerance), so within a width of ``ColHt_chg = 0`` the returned slope is not
  the derivative of the smoothed PE change.  The energy-limited Kd solve uses
  that slope for its Newton steps and for its first guess, so there it can
  take different iterates and, if it has not converged within
  ``max_kd_its``, return a different Kd; a converged solve still lands on a
  root of the smoothed energy balance.
  The fixture gap at Q = -200 / dt 300 s goes from 2.6e-5 to 4.7e-6 degC; Q = 0
  goes to 6.9e-6 / 2.2e-5 degC.
- ``smooth_energy_sharpness_m2j`` (default 1e6 m^2/J, width 1e-6 J/m^2): the
  top-layer ``conv_PErel = max(TKE_forced, 0)`` (C6a) and the matching
  ``mech_TKE`` spend ``where(TKE_forced <= 0, max(0, mech + TKE_forced), mech)``
  (C6b/c, same kink; smoothing C6b/c alone leaves every fixture gap unchanged
  (measured), it is smoothed so the gradient at the kink exists).  At
  ``TKE_forced = 0`` the column starts with ``ln2 * width`` of convective PE.
  The two smooth forms are not complementary near the kink: for
  ``0 < TKE_forced <~ width`` the convective PE gets ``softplus >= TKE_forced``
  AND ``mech_TKE`` still gets ``sigmoid(-s TKE_forced) * TKE_forced``, so the
  column starts with up to ~``(ln2 + 0.3) * width`` [J/m^2] more energy than
  MOM6 gives it.  Fixture Q = 0 goes to 1.9e-8 / 8.6e-8 degC.

With every switch smooth the six fixture gaps are 2.47e-4, 3.49e-3, 1.10e-3
(dt 1200 s; Q = -200, 0, +200) and 4.70e-6, 3.99e-3, 1.34e-4 (dt 300 s).  No
smoothed site changes the column's conservation (ePBL returns diffusivities;
the implicit solve conserves T and S).  Still exact, pending MOM6 fixture
cases that exercise them: the OM4 mstar branch, ``_eqdisc_v0``, the linear
mixing-length shape, the ``eh`` / ``lh`` caps, the v0h buoyancy clip,
heat/cool select and v0 cap, the three derivative/limit ``min(., 0)`` in
``_find_pe_chg_orig``, the interior ``max(0, TKE_forced)`` and TKE depletion,
and the velocity-scale / mixing-length caps (which with ``eqdisc_shape`` feed
only diagnostics); the column descent's branch selection and the two
iterations (issue #2180 groups D and E) are unchanged.

Faithfulness
------------
``tests/ocean/unit/test_epbl.py`` pins the kernels against independent
reimplementations and checks the energy budget, the forcing-ordering of the
diagnosed depth, and reverse-mode differentiability.

FAITHFUL (ported expression-for-expression from MOM6):
- ``_find_pe_chg_orig`` -- ``find_PE_chg_orig``, the ``EPBL_ORIGINAL_PE_CALC``
  potential-energy kernel and its three derivative/limit outputs.
- ``_find_pe_chg`` -- ``find_PE_chg``, the newer algebraically-condensed form.
- ``_find_mstar`` -- ``find_mstar`` for all three ``EPBL_MSTAR_SCHEME`` values
  (``CONSTANT`` / ``OM4`` / ``REICHL_H18``) plus the ``MSTAR_CONV_ADJ``
  convective reduction, at ``answer_date >= 20190101``.
- ``_eqdisc_shape`` / ``_eqdisc_v0`` / ``_eqdisc_v0h`` -- ``kappa_eqdisc``,
  ``get_eqdisc_v0``, ``get_eqdisc_v0h``, the Sane et al. (2026) equation-
  discovery shape function and velocity scales, with MOM6's caps.
- ``_epbl_column`` -- ``ePBL_column``: the TKE decay, the ``nstar`` Rossby
  adjustment, the four diffusivity branches (convectively unstable / energy
  available / no energy / energy-limited Newton-and-false-position solve),
  the running mixed-layer-depth accumulation including its fractional last
  layer, and the MLD iteration with both the bisection and false-position
  updates and MOM6's ``EPBL_MLD_ITER_BUG`` bound ordering.

DEPARTURES (deliberate, each raises or is documented):
- Boussinesq only: thickness is taken in metres with ``H_to_RZ = rho_0``, so
  the non-Boussinesq ``SpV_avg`` arithmetic is not ported.  This matches how
  the rest of legoESM's ocean column physics is written.
- ``DIRECT_EPBL_MIXING_CALC`` (``find_Kd_from_PE_chg``) is NOT ported; the
  config has no field for it, so the iterative solver always runs.  MOM6's
  default is also the iterative solver.
- Langmuir turbulence (``EPBL_LT``, ``USE_LA_LI2016``) and the bottom
  boundary layer (``ePBL_BBL_column``, ``EPBL_BBL_EFFIC``) are not ported.
- ``EPBL_ANSWER_DATE`` is fixed at the modern (>= 20240101) arithmetic; the
  pre-2019 and pre-2024 expression variants are not selectable.
- MOM6 applies the surface fluxes BEFORE ePBL and hands it the resulting
  per-layer energy ``TKE_forced``.  Here ``tke_forcing`` is computed for the
  top layer only, from the net surface heat and salt fluxes, matching
  ``applyBoundaryFluxesInOut`` for a non-penetrating flux.  Penetrating
  shortwave does not contribute a sub-surface ``TKE_forced`` profile.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.core.smooth import (
    sigmoid_switch, smooth_clamp, smooth_min, smooth_relu)
from legoesm.ocean.physics.vertical_mixing.config import EPBLConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput


__physics_contract__ = {
    "summary": (
        "Energetic PBL (Reichl & Hallberg 2018): the boundary-layer "
        "diffusivity Kd = vonKar * vstar * mixlen is limited at every "
        "interface by the turbulent kinetic energy remaining after paying "
        "the potential-energy cost of mixing at all shallower interfaces; "
        "the depth at which the energy is exhausted is the diagnosed "
        "boundary-layer depth, iterated to self-consistency with the "
        "mixing-length shape."
    ),
    "inputs": {
        "h": "m", "u": "m/s", "v": "m/s", "T": "degC", "S": "psu",
        "dSV_dT": "m^3/kg/degC", "dSV_dS": "m^3/kg/psu",
        "tke_forcing": "J/m^2", "B_flux": "m^2/s^3",
        "u_star": "m/s", "absf": "1/s", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "K_v": "m^2/s", "A_v": "m^2/s", "mld": "m",
    },
    "sign_convention": (
        "Depths and thicknesses are positive downward, surface-to-bottom "
        "(k=0 at the surface). B_flux is positive for a STABILIZING "
        "(warming) surface buoyancy flux, matching MOM6's buoyancyFlux. "
        "tke_forcing is positive when the surface forcing RELEASES potential "
        "energy (surface cooling / brine rejection) and negative when mixing "
        "the forcing through the top layer COSTS energy (surface warming)."
    ),
    # "none" because ePBL emits no tendencies of its own -- it returns
    # diffusivities, and the zero-flux implicit solve that consumes them is
    # what conserves the column integrals of T and S. Same declaration as
    # CATKE, the other K-profile-only closure. The scheme's own invariant is
    # that the potential-energy change it authorises never exceeds the
    # turbulent kinetic energy the surface forcing supplies.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Reichl & Hallberg (2018), Ocean Modelling 132, 112-129, "
        "doi:10.1016/j.ocemod.2018.10.004; Reichl & Li (2019), JPO 49, "
        "2935-2959, doi:10.1175/JPO-D-18-0258.1. Machine-learned diffusivity "
        "options: Sane et al. (2023), JAMES 15, e2023MS003890, "
        "doi:10.1029/2023MS003890; Sane et al. (2026), GRL 53, e2026GL122106, "
        "doi:10.1029/2026GL122106. Ported from MOM6 MOM_energetic_PBL.F90 "
        "(B. G. Reichl and R. Hallberg), https://github.com/mom-ocean/MOM6/"
        "blob/main/src/parameterizations/vertical/MOM_energetic_PBL.F90."
    ),
    "idealized_test": (
        "tests/ocean/unit/test_epbl.py: a 50 m / 1 m single column at 40 N "
        "under 0.2 Pa of wind stress deepens monotonically with surface "
        "cooling and shoals with surface heating; the kernels match "
        "independent reimplementations to round-off."
    ),
}



# --- Sane et al. (2026) equation-discovery coefficients -----------------
# MOM6 EPBL_EQD_DIFFUSIVITY_COEFFS defaults (c1..c18, 1-based in the paper
# and in MOM6; index 0 below is a placeholder so c[n] reads as in the paper).
# c1-c6   set sigma_max (the depth of the diffusivity maximum);
# c7-c14  the v0 velocity scale; c14-c18 the v0h velocity scale.
_EQD_ML_C = (
    0.0,
    1.7908, 0.6904, 0.0712, 0.4380, 2.6821, 1.5845,
    0.1550, 1.1120, 0.8616, 0.0984, 45.0, 2.8570,
    3.290, 0.0785, 0.650, 0.0944, 6.0277, 15.7292,
)

# --- Fixed coefficients of the published ePBL formulation ---------------
# Reichl & Hallberg (2018) / MOM6 MOM_energetic_PBL.F90. These are part of
# the scheme's definition rather than tunable knobs, so they live here with
# their provenance instead of in EPBLConfig.
#: Floor on the REICHL_H18 surface decay factor (MOM_energetic_PBL.F90:1538).
_RH18_SURFACE_SCALE_FLOOR = 0.05
#: Obukhov limit on the rotation-limited mstar in the OM4 scheme; about
#: 0.5 / von Karman (MOM_energetic_PBL.F90:3578).
_OM4_MSTAR_N_CAP = 1.25
#: Coefficient of the natural-Rossby-number adjustment to nstar, from the
#: curve fit to Wang (GRL, 2003) (MOM_energetic_PBL.F90:1390).
_NSTAR_ROSSBY_COEF = 0.2

_EPBL_MSTAR_SCHEMES = ("constant", "om4", "reichl_h18")
_EPBL_WT_SCHEMES = ("cube_root_tke", "reichl_h18")
_SMOOTH_SHARPNESS_FIELDS = tuple(
    f for f in EPBLConfig._fields if f.startswith("smooth_"))

# A denominator floor. MOM6 relies on Fortran's `if (x > 0)` guards; the
# traced equivalent needs a finite epsilon so the unselected branch cannot
# produce a NaN that `jnp.where` would propagate through reverse-mode AD.
_TINY = 1.0e-30


def _safe_div(num, den):
    """``num / den`` with the zero-denominator branch returning zero.

    The double ``where`` keeps the gradient finite: the inner one stops the
    division itself from ever seeing a zero, which would otherwise put a NaN
    into the cotangent even though the outer ``where`` discards the value.
    """
    ok = jnp.abs(den) > _TINY
    safe_den = jnp.where(ok, den, 1.0)
    return jnp.where(ok, num / safe_den, 0.0)


def _safe_sqrt(x):
    """``sqrt(max(x, 0))`` with a zero cotangent at the origin.

    ``d/dx sqrt(x)`` is infinite at zero, and an infinite cotangent
    multiplied by the zero from a surrounding ``where`` yields NaN, which
    would poison every gradient through the scheme. ePBL takes square roots
    only of energies and fluxes that are legitimately zero in the unforced
    limit, so pinning the derivative to zero there is the useful choice.
    """
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _safe_pow(x, p):
    """``x**p`` for ``x >= 0`` with a zero cotangent at the origin.

    Same reasoning as :func:`_safe_sqrt`: ``0**p`` for fractional ``p`` has
    an infinite derivative.
    """
    pos = x > 0.0
    return jnp.where(pos, jnp.where(pos, x, 1.0) ** p, 0.0)


def _cbrt(x):
    """Sign-preserving cube root (MOM6's ``cuberoot``).

    The derivative is infinite at the origin; as in :func:`_safe_sqrt` it is
    pinned to zero there so reverse-mode AD stays finite in the no-energy
    limit, where ePBL cube-roots an exactly zero TKE.
    """
    ax = jnp.abs(x)
    pos = ax > 0.0
    return jnp.where(pos, jnp.sign(x) * jnp.where(pos, ax, 1.0) ** (1.0 / 3.0),
                     0.0)


def validate_epbl_config(cfg: EPBLConfig) -> None:
    """Fail loudly on an ePBL configuration this port cannot honour.

    Dispatch discipline: an unknown scheme name must raise rather than fall
    through to a default, and an option that is not ported must raise rather
    than silently run without it.
    """
    if cfg.mstar_scheme not in _EPBL_MSTAR_SCHEMES:
        raise ValueError(
            f"Unknown EPBLConfig.mstar_scheme={cfg.mstar_scheme!r}; expected "
            f"one of {_EPBL_MSTAR_SCHEMES} (MOM6 EPBL_MSTAR_SCHEME "
            f"CONSTANT / OM4 / REICHL_H18)."
        )
    if cfg.wt_scheme not in _EPBL_WT_SCHEMES:
        raise ValueError(
            f"Unknown EPBLConfig.wt_scheme={cfg.wt_scheme!r}; expected one of "
            f"{_EPBL_WT_SCHEMES} (MOM6 EPBL_VEL_SCALE_SCHEME "
            f"CUBE_ROOT_TKE / REICHL_H18)."
        )
    if cfg.eqdisc_v0 and cfg.eqdisc_v0h:
        raise ValueError(
            "EPBLConfig.eqdisc_v0 and eqdisc_v0h are mutually exclusive: both "
            "set the equation-discovery velocity scale, so enabling both "
            "would silently use only one of them."
        )
    if not (cfg.dt_s > 0.0):
        raise ValueError(
            f"EPBLConfig.dt_s must be positive, got {cfg.dt_s!r}. ePBL is an "
            "energy-per-timestep closure, so it needs the model timestep; set "
            "dt_s to the same value the driver steps with."
        )
    if cfg.max_mld_its < 1:
        raise ValueError(
            f"EPBLConfig.max_mld_its must be >= 1, got {cfg.max_mld_its!r}.")
    if cfg.max_kd_its < 1:
        raise ValueError(
            f"EPBLConfig.max_kd_its must be >= 1, got {cfg.max_kd_its!r}.")
    if cfg.mld_tol <= 0.0:
        raise ValueError(
            f"EPBLConfig.mld_tol must be positive, got {cfg.mld_tol!r}.")
    for name in _SMOOTH_SHARPNESS_FIELDS:
        value = getattr(cfg, name)
        if not value > 0.0:  # also rejects NaN; math.inf (exact MOM6) passes
            raise ValueError(
                f"EPBLConfig.{name} must be > 0 (math.inf = MOM6's exact "
                f"switch), got {value!r}.")


# ---------------------------------------------------------------------------
# mstar
# ---------------------------------------------------------------------------


def _find_mstar(cfg: EPBLConfig, b_flux, u_star, bld, absf):
    """MOM6 ``find_mstar`` at answer dates >= 20190101.

    Returns the ratio of the TKE input to ``u_star^3`` [nondim].
    """
    if cfg.mstar_scheme == "constant":
        mstar = jnp.full_like(jnp.asarray(u_star), cfg.fixed_mstar)
    elif cfg.mstar_scheme == "om4":
        absf_safe = jnp.maximum(absf, 1.0e-20)
        mstar_s = cfg.mstar_coef * _safe_sqrt(
            _safe_div(jnp.maximum(0.0, b_flux), u_star**2 * absf_safe))
        # Rotation (Ekman length) limited mixing; zero unless u_star exceeds
        # the rotational scale, exactly as MOM6's guarded log.
        # At absf * bld == 0 MOM6's log is +inf, so min() below returns the cap.
        ekman = absf * bld
        ratio = _safe_div(u_star, ekman)
        mstar_n = jnp.where(
            u_star > ekman,
            jnp.where(ekman > 0.0,
                      cfg.c_ek * jnp.log(jnp.maximum(ratio, 1.0)),
                      _OM4_MSTAR_N_CAP),
            0.0)
        # 1.25 is about 0.5/von Karman, which gives the Obukhov limit.
        mstar = jnp.maximum(mstar_s, jnp.minimum(_OM4_MSTAR_N_CAP, mstar_n))
        if cfg.mstar_cap > 0.0:
            mstar = jnp.minimum(cfg.mstar_cap, mstar)
    else:  # reichl_h18
        msn_term = cfg.rh18_mstar_cn2 * jnp.exp(
            cfg.rh18_mstar_cn3 * _safe_div(bld * absf, u_star))
        mstar_n = (cfg.rh18_mstar_cn1 * msn_term) / (1.0 + msn_term)
        absf_safe = jnp.maximum(absf, 1.0e-20)
        mstar_s = cfg.rh18_mstar_cs1 * _safe_pow(
            _safe_div(smooth_relu(b_flux, cfg.smooth_bflux_sharpness_s3m2) ** 2  # RULE5:C1r
                      * bld, u_star**5 * absf_safe), cfg.rh18_mstar_cs2)
        mstar = mstar_n + mstar_s

    # Reduce mstar to account for convective turbulence eroding the stable
    # density gradient that mechanical mixing would otherwise work against.
    mscr_term1 = -bld * smooth_min(b_flux, 0.0, cfg.smooth_bflux_sharpness_s3m2)  # RULE5:C2
    mscr_term2 = 2.0 * mstar * u_star**3
    mstar_conv_red = jnp.where(
        jnp.abs(mscr_term2) > 0.0,
        _safe_div((1.0 - cfg.mstar_convect_coef) * mscr_term1 + mscr_term2,
                  mscr_term1 + mscr_term2),
        1.0 - cfg.mstar_convect_coef,
    )
    return mstar * mstar_conv_red


# ---------------------------------------------------------------------------
# Equation-discovery (machine-learned) shape function and velocity scales
# ---------------------------------------------------------------------------


def _eqdisc_shape(cfg: EPBLConfig, z_int, absf, b_flux, u_star, mld_guess):
    """MOM6 ``kappa_eqdisc`` -- Sane et al. (2026) Eq. 3, 5, 6.

    ``z_int`` is the depth of each interface [m], positive downward, with
    ``z_int[0] = 0``.  Returns the nondimensional shape function.
    """
    c = _EQD_ML_C
    hbl = mld_guess
    u_star_i = _safe_div(1.0, u_star)
    # Boundary-layer depth measured in Monin-Obukhov depths and in Ekman
    # depths.  The sign convention follows Sane et al.: Lh > 0 is cooling.
    lh = (-b_flux * hbl) * (u_star_i**3)
    eh = (hbl * absf) * u_star_i
    eh = jnp.minimum(eh, cfg.eqd_eh_upper_cap)
    lh = jnp.clip(lh, -cfg.eqd_lh_cap, cfg.eqd_lh_cap)

    f_fn = 1.0 / (c[3] + c[4] * jnp.exp(-c[5] * lh)) + c[6]
    f_eh = f_fn * eh
    sm = _safe_div(f_eh, c[1] * f_eh + c[2])
    sm = smooth_clamp(sm, cfg.eqd_sigma_max_lower_cap,
                      cfg.eqd_sigma_max_upper_cap,
                      cfg.smooth_ratio_sharpness)  # RULE5:C3sm

    sm_h = sm * hbl
    eps = cfg.eqd_shape_function_epsilon
    h_minus_smh_i = _safe_div(1.0, hbl - sm_h)
    coef_c3 = (2.0 * (1.0 - eps)) * h_minus_smh_i**3
    coef_c2 = (3.0 * (eps - 1.0)) * h_minus_smh_i**2

    # Quadratic from the surface down to the maximum at sm_h, cubic from
    # there to the boundary-layer base, then a small constant below it.
    z_sm_h_i = _safe_div(z_int, sm_h)
    quad = -z_sm_h_i**2 + 2.0 * z_sm_h_i
    dz_from_max = z_int - sm_h
    cubic = coef_c3 * dz_from_max**3 + coef_c2 * dz_from_max**2 + 1.0

    shape = jnp.where(z_int <= sm_h, quad, jnp.where(z_int <= hbl, cubic, eps))
    # MOM6 sets shape_func(1) = 0 before the loop and never revisits it.
    return shape.at[0].set(0.0)


def _eqdisc_v0(cfg: EPBLConfig, absf, b_flux, u_star):
    """MOM6 ``get_eqdisc_v0`` -- Sane et al. (2026) Eq. 8 (no depth input)."""
    c = _EQD_ML_C
    bflux_c = jnp.clip(b_flux, cfg.eqd_bflux_lower_cap, cfg.eqd_bflux_upper_cap)
    f_capped = jnp.maximum(absf, cfg.eqd_f_lower)
    f_u2 = f_capped * u_star**2
    root_b_f = _safe_sqrt(jnp.abs(bflux_c) * f_capped)

    # Heating / neutral branch.
    den_heat = (bflux_c + (c[8] + c[9]) * u_star * root_b_f
                + (c[8] * c[9] + c[9] ** 2) * f_u2)
    v0_heat = _safe_div(
        (c[7] * ((u_star * root_b_f) + (c[9] * f_u2))) * u_star, den_heat)

    # Cooling branch.
    f_prime = f_capped / constants.Omega
    root_b_by_omega = _safe_sqrt(-bflux_c / constants.Omega)
    den_cool = ((-bflux_c + c[11] * f_u2 * jnp.exp(-f_prime * c[12]))
                + c[13] * f_u2)
    v0_cool = (_safe_div(c[10] * (-bflux_c * root_b_by_omega), den_cool)
               + c[14] * u_star)

    v0 = jnp.where(bflux_c >= 0.0, v0_heat, v0_cool)
    return jnp.clip(v0, cfg.eqd_v0_lower_cap, cfg.eqd_v0_upper_cap)


def _eqdisc_v0h(cfg: EPBLConfig, b_flux, u_star, mld_guess):
    """MOM6 ``get_eqdisc_v0h`` -- Sane et al. (2026) Eq. 9 and 10."""
    c = _EQD_ML_C
    bflux_c = jnp.clip(b_flux, cfg.eqd_bflux_lower_cap, cfg.eqd_bflux_upper_cap)
    u_star_2 = u_star * u_star
    u_star_3 = u_star_2 * u_star
    b_h = jnp.abs(bflux_c) * mld_guess
    b_h_13 = _cbrt(b_h)

    den_heat = (c[15] * b_h + c[16] * u_star * (b_h_13 * b_h_13)) + u_star_3
    v0_heat = _safe_div(c[14] * (u_star_2 * u_star_2), den_heat)

    den_cool = c[17] * (b_h_13 * b_h_13) + c[18] * u_star_2
    v0_cool = _safe_div(b_h, den_cool) + c[14] * u_star

    v0 = jnp.where(bflux_c >= 0.0, v0_heat, v0_cool)
    return jnp.clip(v0, cfg.eqd_v0_lower_cap, cfg.eqd_v0_upper_cap)


# ---------------------------------------------------------------------------
# Potential-energy change kernels
# ---------------------------------------------------------------------------


def _find_pe_chg_orig(kddt_h, h_k, b_den_1, dte_term, dse_term,
                      dt_km1_t2, ds_km1_t2,
                      dt_to_dpe_k, ds_to_dpe_k, dt_to_dpe_a, ds_to_dpe_a,
                      pres_z,
                      dt_to_dcolht_k, ds_to_dcolht_k,
                      dt_to_dcolht_a, ds_to_dcolht_a, *, colht_sharpness):
    """MOM6 ``find_PE_chg_orig``.

    Returns ``(PE_chg, dPEc_dKd, dPE_max, dPEc_dKd_0)``: the column potential-
    energy change from applying ``kddt_h`` at this interface, its derivative,
    its limit for infinite diffusivity, and its derivative at zero.
    """
    b1 = _safe_div(1.0, b_den_1 + kddt_h)
    b1kd = kddt_h * b1
    i_kr_denom = _safe_div(1.0, h_k * b_den_1 + (b_den_1 + h_k) * kddt_h)

    dt_k = (kddt_h * i_kr_denom) * dte_term
    ds_k = (kddt_h * i_kr_denom) * dse_term
    dt_km1 = b1kd * (dt_k + dt_km1_t2)
    ds_km1 = b1kd * (ds_k + ds_km1_t2)

    pe_chg = ((dt_to_dpe_k * dt_k + dt_to_dpe_a * dt_km1)
              + (ds_to_dpe_k * ds_k + ds_to_dpe_a * ds_km1))
    colht_chg = ((dt_to_dcolht_k * dt_k + dt_to_dcolht_a * dt_km1)
                 + (ds_to_dcolht_k * ds_k + ds_to_dcolht_a * ds_km1))
    pe_chg = pe_chg - pres_z * smooth_min(colht_chg, 0.0, colht_sharpness)  # RULE5:C5a

    dkr_dkd = (h_k * b_den_1) * i_kr_denom**2
    ddt_k_dkd = dkr_dkd * dte_term
    dds_k_dkd = dkr_dkd * dse_term
    ddt_km1_dkd = (b1**2 * b_den_1) * (dt_k + dt_km1_t2) + b1kd * ddt_k_dkd
    dds_km1_dkd = (b1**2 * b_den_1) * (ds_k + ds_km1_t2) + b1kd * dds_k_dkd
    dpec_dkd = ((dt_to_dpe_k * ddt_k_dkd + dt_to_dpe_a * ddt_km1_dkd)
                + (ds_to_dpe_k * dds_k_dkd + ds_to_dpe_a * dds_km1_dkd))
    dcolht_dkd = ((dt_to_dcolht_k * ddt_k_dkd + dt_to_dcolht_a * ddt_km1_dkd)
                  + (ds_to_dcolht_k * dds_k_dkd + ds_to_dcolht_a * dds_km1_dkd))
    dpec_dkd = dpec_dkd - pres_z * jnp.minimum(dcolht_dkd, 0.0)

    inv_sum = _safe_div(1.0, b_den_1 + h_k)
    dpe_max = ((dt_to_dpe_a * dt_km1_t2 + ds_to_dpe_a * ds_km1_t2)
               + ((dt_to_dpe_k + dt_to_dpe_a) * dte_term
                  + (ds_to_dpe_k + ds_to_dpe_a) * dse_term) * inv_sum)
    dcolht_max = ((dt_to_dcolht_a * dt_km1_t2 + ds_to_dcolht_a * ds_km1_t2)
                  + ((dt_to_dcolht_k + dt_to_dcolht_a) * dte_term
                     + (ds_to_dcolht_k + ds_to_dcolht_a) * dse_term) * inv_sum)
    dpe_max = dpe_max - pres_z * jnp.minimum(dcolht_max, 0.0)

    inv_b = _safe_div(1.0, b_den_1)
    inv_hb = _safe_div(1.0, h_k * b_den_1)
    dpec_dkd_0 = ((dt_to_dpe_a * dt_km1_t2 + ds_to_dpe_a * ds_km1_t2) * inv_b
                  + (dt_to_dpe_k * dte_term + ds_to_dpe_k * dse_term) * inv_hb)
    dcolht_dkd_0 = (
        (dt_to_dcolht_a * dt_km1_t2 + ds_to_dcolht_a * ds_km1_t2) * inv_b
        + (dt_to_dcolht_k * dte_term + ds_to_dcolht_k * dse_term) * inv_hb)
    dpec_dkd_0 = dpec_dkd_0 - pres_z * jnp.minimum(dcolht_dkd_0, 0.0)

    return pe_chg, dpec_dkd, dpe_max, dpec_dkd_0


def _find_pe_chg(kddt_h0, dkddt_h, hp_a, hp_b, th_a, sh_a, th_b, sh_b,
                 dt_to_dpe_a, ds_to_dpe_a, dt_to_dpe_b, ds_to_dpe_b,
                 pres_z, dt_to_dcolht_a, ds_to_dcolht_a,
                 dt_to_dcolht_b, ds_to_dcolht_b):
    """MOM6 ``find_PE_chg`` (the condensed, non-``orig`` kernel).

    Returns ``(PE_chg, dPEc_dKd, dPE_max, dPEc_dKd_0)``.
    """
    hps = hp_a + hp_b
    bdt1 = hp_a * hp_b + kddt_h0 * hps
    dt_c = hp_a * th_b - hp_b * th_a
    ds_c = hp_a * sh_b - hp_b * sh_a
    pec_core = (hp_b * (dt_to_dpe_a * dt_c + ds_to_dpe_a * ds_c)
                - hp_a * (dt_to_dpe_b * dt_c + ds_to_dpe_b * ds_c))
    colht_core = (hp_b * (dt_to_dcolht_a * dt_c + ds_to_dcolht_a * ds_c)
                  - hp_a * (dt_to_dcolht_b * dt_c + ds_to_dcolht_b * ds_c))

    y1_3 = _safe_div(dkddt_h, bdt1 * (bdt1 + dkddt_h * hps))
    pe_chg = pec_core * y1_3 - pres_z * jnp.minimum(colht_core * y1_3, 0.0)

    y1_4 = _safe_div(1.0, (bdt1 + dkddt_h * hps) ** 2)
    dpec_dkd = pec_core * y1_4 - pres_z * jnp.minimum(colht_core * y1_4, 0.0)

    y_max = _safe_div(1.0, bdt1 * hps)
    dpe_max = pec_core * y_max - pres_z * jnp.minimum(colht_core * y_max, 0.0)

    y_0 = _safe_div(1.0, bdt1**2)
    dpec_dkd_0 = pec_core * y_0 - pres_z * jnp.minimum(colht_core * y_0, 0.0)

    return pe_chg, dpec_dkd, dpe_max, dpec_dkd_0


# ---------------------------------------------------------------------------
# The column solver
# ---------------------------------------------------------------------------


def _top_layer_energy(cfg: EPBLConfig, tkef0, mech_tke_0):
    """``(conv_PErel, mech_TKE)`` above the first interior interface.

    The top-layer forcing ``tkef0`` is spent (or released) before the first
    interior interface is reached: MOM6 ``conv_PErel = max(TKE_forced, 0)``
    and ``mech_TKE = max(0, mech_TKE + TKE_forced)`` where the forcing costs
    energy.  Both switches kink at ``tkef0 = 0`` and are smooth by default.
    """
    s_e = cfg.smooth_energy_sharpness_m2j
    if s_e == math.inf:
        # MOM6's switches verbatim (a 0/1 blend would turn an overflowed
        # branch into NaN where MOM6 selects the other one).
        return (jnp.where(tkef0 > 0.0, tkef0, 0.0),
                jnp.where(tkef0 <= 0.0,
                          jnp.maximum(0.0, mech_tke_0 + tkef0), mech_tke_0))
    conv_perel_0 = smooth_relu(tkef0, s_e)  # RULE5:C6a
    spent = sigmoid_switch(-tkef0, s_e)  # RULE5:C6b
    mech_tke_0 = (spent * smooth_relu(mech_tke_0 + tkef0, s_e)  # RULE5:C6c
                  + (1.0 - spent) * mech_tke_0)
    return conv_perel_0, mech_tke_0


def _epbl_column(h, u, v, t0, s0, dsv_dt, dsv_ds, tke_forcing,
                 b_flux, absf, u_star, mech_tke_in, mld_guess_in,
                 cfg: EPBLConfig, g: float, rho_0: float):
    """One column of ePBL.  Returns ``(Kd_interfaces, mld, mixvel, mixlen)``.

    ``Kd`` has length ``nz+1`` with zeros at the surface and bottom, matching
    MOM6's interface array.  All inputs are single-column 1-D arrays ordered
    surface to bottom.
    """
    nz = h.shape[0]
    dt = cfg.dt_s

    # --- column geometry and the PE sensitivity coefficients -------------
    d_mass = rho_0 * h                       # mass per unit area [kg/m^2]
    d_pres = g * d_mass                      # hydrostatic pressure step [Pa]
    pres_z = jnp.concatenate([jnp.zeros((1,), h.dtype), jnp.cumsum(d_pres)])
    pres_lay = pres_z[:-1] + 0.5 * d_pres    # mid-layer pressure

    dt_to_dpe = (d_mass * pres_lay) * dsv_dt
    ds_to_dpe = (d_mass * pres_lay) * dsv_ds
    dt_to_dcolht = d_mass * dsv_dt
    ds_to_dcolht = d_mass * dsv_ds

    dz_sum = jnp.sum(h)
    # Distance from the bottom to each interface, over the column depth.
    below = jnp.concatenate([jnp.cumsum(h[::-1])[::-1], jnp.zeros((1,), h.dtype)])
    hb_hs = _safe_div(below, dz_sum)
    z_int = jnp.concatenate([jnp.zeros((1,), h.dtype), jnp.cumsum(h)])

    spv_dt = 1.0 / (dt * rho_0)              # Boussinesq specific volume / dt
    h_dz_int = 1.0                           # Boussinesq: thickness == distance

    # dt divided by the mean thickness around each interior interface.
    h_mean = 0.5 * (h[:-1] + h[1:])
    dt_h_int = dt / jnp.maximum(h_mean, 1e-15 * dz_sum)

    column_depth = dz_sum

    # --- initial energy state above the first interior interface ---------
    tkef0 = tke_forcing[0]

    def one_pass(mld_guess):
        """A single descent through the column for a given depth guess."""
        mstar = _find_mstar(cfg, b_flux, u_star, mld_guess, absf)
        conv_perel_0, mech_tke_0 = _top_layer_energy(
            cfg, tkef0, mstar * mech_tke_in)

        # --- mixing-length shape ---------------------------------------
        if (not cfg.use_mld_iteration) or cfg.translay_scale >= 1.0 \
                or cfg.translay_scale < 0.0:
            mixlen_shape = jnp.ones((nz + 1,), h.dtype)
        elif cfg.eqdisc_shape:
            mixlen_shape = _eqdisc_shape(cfg, z_int, absf, b_flux, u_star,
                                         mld_guess)
        else:
            frac = jnp.maximum(0.0, _safe_div(mld_guess - z_int, mld_guess))
            mixlen_shape = (cfg.translay_scale
                            + (1.0 - cfg.translay_scale)
                            * frac ** cfg.mixlen_exponent)
            mixlen_shape = mixlen_shape.at[0].set(1.0)
            # A zero guess degenerates to the transition-layer value, as in
            # MOM6's `MLD_guess <= 0` branch.
            mixlen_shape = jnp.where(
                mld_guess > 0.0, mixlen_shape,
                jnp.where(cfg.translay_scale > 0.0, cfg.translay_scale, 1.0))

        # --- equation-discovery velocity scale --------------------------
        if cfg.eqdisc_v0:
            v0_ml = _eqdisc_v0(cfg, absf, b_flux, u_star)
        elif cfg.eqdisc_v0h:
            v0_ml = _eqdisc_v0h(cfg, b_flux, u_star, mld_guess)
        else:
            v0_ml = jnp.zeros((), h.dtype)

        # The TKE decay length is rotational; MOM6 applies it per layer.
        idecay = _safe_div(cfg.tke_decay * absf, u_star)

        def body(carry, xs):
            (mech_tke, conv_perel, htot, dztot, uhtot, vhtot,
             sfc_connected, mld_out, hp_a_prev,
             dt_to_dpe_a_prev, ds_to_dpe_a_prev,
             dt_to_dcolht_a_prev, ds_to_dcolht_a_prev,
             kddt_h_prev, prev_term_t, prev_term_s) = carry
            (h_above, h_below, t_above, t_below, s_above, s_below,
             u_below, v_below, tkef_k, dt_h, hbs_raw, mls_k, pres_k,
             dt_to_dpe_k, ds_to_dpe_k,
             dt_to_dcolht_k, ds_to_dcolht_k,
             dt_to_dcolht_km1, ds_to_dcolht_km1) = xs

            # 1. Mechanical TKE decays across the layer above.
            exp_kh = jnp.where(idecay > 0.0, jnp.exp(-h_above * idecay), 1.0)
            mech_tke = mech_tke * exp_kh

            # 2. Convectively released PE accumulates.
            conv_perel = conv_perel + jnp.maximum(0.0, tkef_k)

            # 3. nstar follows a natural Rossby number (Wang, GRL 2003 fit).
            rossby_den = conv_perel + _NSTAR_ROSSBY_COEF * _safe_sqrt(
                0.5 * dt * rho_0 * (absf * dztot) ** 3 * conv_perel)
            nstar_fc = jnp.where(
                cfg.nstar * conv_perel > 0.0,
                cfg.nstar * _safe_div(conv_perel, rossby_den),
                cfg.nstar)
            tot_tke = mech_tke + nstar_fc * conv_perel

            # 4. Spend TKE on mixing any penetrating forcing through this layer.
            neg = tkef_k < 0.0
            deplete = neg & (tkef_k + tot_tke < 0.0)
            reduc = jnp.where(tot_tke > 0.0,
                              _safe_div(tot_tke + tkef_k, tot_tke), 0.0)
            scale = jnp.where(deplete, 0.0, jnp.where(neg, reduc, 1.0))
            tot_tke = tot_tke * scale
            mech_tke = mech_tke * scale
            conv_perel = conv_perel * scale

            # 5. Terms that do not depend on this interface's diffusivity.
            dte_t2 = kddt_h_prev * prev_term_t
            dse_t2 = kddt_h_prev * prev_term_s
            dt_km1_t2 = ((t_below - t_above)
                         - _safe_div(kddt_h_prev, hp_a_prev) * prev_term_t)
            ds_km1_t2 = ((s_below - s_above)
                         - _safe_div(kddt_h_prev, hp_a_prev) * prev_term_s)
            dte_term = dte_t2 + hp_a_prev * (t_above - t_below)
            dse_term = dse_t2 + hp_a_prev * (s_above - s_below)

            convectively_stable = (
                0.0 <= ((dt_to_dcolht_k + dt_to_dcolht_km1)
                        * (t_above - t_below)
                        + (ds_to_dcolht_k + ds_to_dcolht_km1)
                        * (s_above - s_below)))
            exhausted = ((mech_tke + conv_perel) <= 0.0) & convectively_stable

            # 6. Mean kinetic energy available to be converted into TKE.
            if cfg.mke_to_tke_effic > 0.0:
                dmke_max = jnp.where(
                    htot * h_below > 0.0,
                    (rho_0 * cfg.mke_to_tke_effic) * 0.5
                    * _safe_div(h_below, (htot + h_below) * htot)
                    * (((uhtot - u_below * htot) ** 2)
                       + ((vhtot - v_below * htot) ** 2)),
                    0.0)
                mke2_hharm = jnp.where(
                    htot * h_below > 0.0,
                    _safe_div(htot + h_below, htot * h_below), 0.0)
            else:
                dmke_max = jnp.zeros((), h.dtype)
                mke2_hharm = jnp.zeros((), h.dtype)

            # 7. The turbulent velocity scale and the proposed diffusivity.
            dz_tt = dztot
            hbs_here = jnp.minimum(hbs_raw, mls_k)

            def velocity_scale(conv_pe, extra_tke):
                tke_here = mech_tke + cfg.wstar_ustar_coef * extra_tke
                if cfg.wt_scheme == "cube_root_tke":
                    vs = cfg.vstar_scale_fac * _cbrt(spv_dt * tke_here)
                else:  # reichl_h18
                    surface_scale = jnp.maximum(
                        _RH18_SURFACE_SCALE_FLOOR,
                        1.0 - _safe_div(dztot, mld_guess))
                    vs = (cfg.vstar_scale_fac * surface_scale) * (
                        cfg.vstar_surf_fac * u_star
                        + _cbrt((cfg.wstar_ustar_coef * conv_pe) * spv_dt))
                return jnp.where(tke_here > 0.0, vs, 0.0), tke_here > 0.0

            def diffusivity(vstar, has_tke):
                ml = jnp.maximum(
                    cfg.min_mix_len,
                    _safe_div((dz_tt * hbs_here) * vstar,
                              (cfg.ekman_scale_coef * absf) * (dz_tt * hbs_here)
                              + vstar))
                if not cfg.use_mld_iteration:
                    kd = ((h_dz_int * vstar) * cfg.vonkar
                          * _safe_div((dz_tt * hbs_here) * vstar,
                                      (cfg.ekman_scale_coef * absf)
                                      * (dz_tt * hbs_here) + vstar))
                elif cfg.eqdisc_shape:
                    kd = mls_k * v0_ml * mld_guess
                else:
                    kd = (h_dz_int * vstar) * cfg.vonkar * ml
                return (jnp.where(has_tke, ml, 0.0),
                        jnp.where(has_tke, kd, 0.0))

            vstar, has_tke = velocity_scale(conv_perel, conv_perel)
            mixlen_k, kd_guess0 = diffusivity(vstar, has_tke)
            kddt_h_g0 = kd_guess0 * dt_h

            pe_chg_g0, _, pe_chg_max, dpec_dkd_kd0 = _find_pe_chg_orig(
                kddt_h_g0, h_below, hp_a_prev, dte_term, dse_term,
                dt_km1_t2, ds_km1_t2, dt_to_dpe_k, ds_to_dpe_k,
                dt_to_dpe_a_prev, ds_to_dpe_a_prev, pres_k,
                dt_to_dcolht_k, ds_to_dcolht_k,
                dt_to_dcolht_a_prev, ds_to_dcolht_a_prev,
                colht_sharpness=cfg.smooth_colht_sharpness_per_m)
            mke_src_g0 = dmke_max * (1.0 - jnp.exp(-kddt_h_g0 * mke2_hharm))

            convectively_unstable = (
                (pe_chg_g0 < 0.0) | ((~has_tke) & (dpec_dkd_kd0 < 0.0)))

            # --- branch A: convectively unstable -------------------------
            # Re-price the velocity scale against the energy that the
            # instability itself releases.
            vstar_c, has_tke_c = velocity_scale(
                conv_perel, conv_perel - pe_chg_max)
            mixlen_c, kd_c = diffusivity(vstar_c, has_tke_c)
            dpe_conv_c, _, _, _ = _find_pe_chg_orig(
                kd_c * dt_h, h_below, hp_a_prev, dte_term, dse_term,
                dt_km1_t2, ds_km1_t2, dt_to_dpe_k, ds_to_dpe_k,
                dt_to_dpe_a_prev, ds_to_dpe_a_prev, pres_k,
                dt_to_dcolht_k, ds_to_dcolht_k,
                dt_to_dcolht_a_prev, ds_to_dcolht_a_prev,
                colht_sharpness=cfg.smooth_colht_sharpness_per_m)
            # If that costs energy after all, fall back to the first guess.
            revert = dpe_conv_c > 0.0
            kd_cu = jnp.where(pe_chg_max <= 0.0,
                              jnp.where(revert, kd_guess0, kd_c), kd_guess0)
            vstar_cu = jnp.where(pe_chg_max <= 0.0,
                                 jnp.where(revert, vstar, vstar_c), vstar)
            mixlen_cu = jnp.where(pe_chg_max <= 0.0,
                                  jnp.where(revert, mixlen_k, mixlen_c),
                                  mixlen_k)
            dpe_conv = jnp.where(pe_chg_max <= 0.0,
                                 jnp.where(revert, pe_chg_g0, dpe_conv_c),
                                 pe_chg_g0)
            mke_src_cu = jnp.where(
                (pe_chg_max <= 0.0) & (~revert),
                dmke_max * (1.0 - jnp.exp(-(kd_c * dt_h) * mke2_hharm)),
                mke_src_g0)
            conv_perel_cu = conv_perel - dpe_conv
            mech_tke_cu = mech_tke + mke_src_cu
            tot_tke_cu = tot_tke
            mld_add_cu = h_below

            # --- branch B: enough energy for the proposed mixing ---------
            enough = (tot_tke + (mke_src_g0 - pe_chg_g0)) >= 0.0
            tot_b = tot_tke + mke_src_g0
            reduc_b = jnp.where(tot_b > 0.0,
                                _safe_div(tot_b - pe_chg_g0, tot_b), 0.0)
            tot_tke_b = reduc_b * tot_b
            mech_tke_b = reduc_b * (mech_tke + mke_src_g0)
            conv_perel_b = reduc_b * conv_perel
            mld_add_b = h_below

            # --- branch C: no energy at all ------------------------------
            no_energy = tot_tke == 0.0

            # --- branch D: energy-limited; solve for the affordable Kd ---
            kddt_h_max0 = kddt_h_g0
            tke_left_max0 = tot_tke + (mke_src_g0 - pe_chg_g0)
            tke_left_min0 = tot_tke
            guess0 = tot_tke * _safe_div(
                kddt_h_max0,
                jnp.maximum(pe_chg_g0 - mke_src_g0,
                            kddt_h_max0 * (dpec_dkd_kd0
                                           - dmke_max * mke2_hharm)))

            def newton_body(_, st):
                (guess, kd_min, kd_max, tl_min, tl_max, pe_last, done) = st
                pe_chg, dpec_dkd, _, _ = _find_pe_chg_orig(
                    guess, h_below, hp_a_prev, dte_term, dse_term,
                    dt_km1_t2, ds_km1_t2, dt_to_dpe_k, ds_to_dpe_k,
                    dt_to_dpe_a_prev, ds_to_dpe_a_prev, pres_k,
                    dt_to_dcolht_k, ds_to_dcolht_k,
                    dt_to_dcolht_a_prev, ds_to_dcolht_a_prev,
                    colht_sharpness=cfg.smooth_colht_sharpness_per_m)
                mke_src = dmke_max * (1.0 - jnp.exp(-mke2_hharm * guess))
                dmke_src_dk = dmke_max * mke2_hharm * jnp.exp(
                    -mke2_hharm * guess)
                tke_left = tot_tke + (mke_src - pe_chg)

                pos = tke_left >= 0.0
                kd_min_n = jnp.where(pos, guess, kd_min)
                tl_min_n = jnp.where(pos, tke_left, tl_min)
                kd_max_n = jnp.where(pos, kd_max, guess)
                tl_max_n = jnp.where(pos, tl_max, tke_left)

                slope = dpec_dkd - dmke_src_dk
                dk_newton = _safe_div(tke_left, slope)
                kd_newton = guess + dk_newton
                use_newton = ((slope > 0.0)
                              & (kd_newton <= kd_max_n)
                              & (kd_newton >= kd_min_n))
                kd_false = _safe_div(
                    tl_max_n * kd_min_n - kd_max_n * tl_min_n,
                    tl_max_n - tl_min_n)
                nxt = jnp.where(use_newton, kd_newton, kd_false)
                dk = nxt - guess

                converged = jnp.abs(dk) < 1e-9 * guess
                # MOM6 keeps the last guess on convergence so the energy
                # calculation need not be repeated.
                guess_n = jnp.where(done | converged, guess, nxt)
                return (guess_n, kd_min_n, kd_max_n, tl_min_n, tl_max_n,
                        jnp.where(done, pe_last, pe_chg), done | converged)

            init_newton = (guess0, jnp.zeros((), h.dtype), kddt_h_max0,
                           tke_left_min0, tke_left_max0,
                           jnp.zeros((), h.dtype), jnp.array(False))
            (kddt_h_d, _, _, _, _, pe_chg_d, _) = lax.fori_loop(
                0, cfg.max_kd_its, newton_body, init_newton)
            kd_d = _safe_div(kddt_h_d, dt_h)
            mld_add_d = _safe_div(pe_chg_d, pe_chg_g0) * h_below

            # --- select the branch ---------------------------------------
            # Order follows MOM6: unstable, then enough energy, then none,
            # then energy-limited.
            kd_k = jnp.where(
                convectively_unstable, kd_cu,
                jnp.where(enough, kd_guess0,
                          jnp.where(no_energy, 0.0, kd_d)))
            vstar_out = jnp.where(convectively_unstable, vstar_cu, vstar)
            mixlen_out = jnp.where(convectively_unstable, mixlen_cu, mixlen_k)
            mech_tke_n = jnp.where(
                convectively_unstable, mech_tke_cu,
                jnp.where(enough, mech_tke_b, 0.0))
            conv_perel_n = jnp.where(
                convectively_unstable, conv_perel_cu,
                jnp.where(enough, conv_perel_b, 0.0))
            mld_add = jnp.where(
                convectively_unstable, mld_add_cu,
                jnp.where(enough, mld_add_b,
                          jnp.where(no_energy, 0.0, mld_add_d)))
            disconnect = jnp.where(
                convectively_unstable, False,
                jnp.where(enough, False, True))

            # The exhausted-and-stable case overrides everything above.
            kd_k = jnp.where(exhausted, 0.0, kd_k)
            vstar_out = jnp.where(exhausted, 0.0, vstar_out)
            mixlen_out = jnp.where(exhausted, 0.0, mixlen_out)
            mech_tke_n = jnp.where(exhausted, 0.0, mech_tke_n)
            conv_perel_n = jnp.where(exhausted, 0.0, conv_perel_n)
            mld_add = jnp.where(exhausted, 0.0, mld_add)
            disconnect = jnp.where(exhausted, True, disconnect)

            mld_out_n = mld_out + jnp.where(sfc_connected > 0.0, mld_add, 0.0)

            # --- advance the downward tridiagonal factorisation ----------
            kddt_h_k = kd_k * dt_h
            b1 = _safe_div(1.0, hp_a_prev + kddt_h_k)
            c1 = kddt_h_k * b1
            dte_km1 = b1 * (kddt_h_k * (t_below - t_above) + dte_t2)
            dse_km1 = b1 * (kddt_h_k * (s_below - s_above) + dse_t2)
            hp_a_new = h_below + (hp_a_prev * b1) * kddt_h_k
            dt_to_dpe_a_new = dt_to_dpe_k + c1 * dt_to_dpe_a_prev
            ds_to_dpe_a_new = ds_to_dpe_k + c1 * ds_to_dpe_a_prev
            dt_to_dcolht_a_new = dt_to_dcolht_k + c1 * dt_to_dcolht_a_prev
            ds_to_dcolht_a_new = ds_to_dcolht_k + c1 * ds_to_dcolht_a_prev

            # --- running sums for the next interface ---------------------
            sfc_connected_n = jnp.where(disconnect, 0.0, sfc_connected)
            uhtot_n = jnp.where(disconnect, u_below * h_below,
                                uhtot + u_below * h_below)
            vhtot_n = jnp.where(disconnect, v_below * h_below,
                                vhtot + v_below * h_below)
            htot_n = jnp.where(disconnect, h_below, htot + h_below)
            dztot_n = jnp.where(disconnect, h_below, dztot + h_below)

            new_carry = (mech_tke_n, conv_perel_n, htot_n, dztot_n,
                         uhtot_n, vhtot_n, sfc_connected_n, mld_out_n,
                         hp_a_new, dt_to_dpe_a_new, ds_to_dpe_a_new,
                         dt_to_dcolht_a_new, ds_to_dcolht_a_new,
                         kddt_h_k,
                         (t_above - t_below) + dte_km1,
                         (s_above - s_below) + dse_km1)
            return new_carry, (kd_k, vstar_out, mixlen_out)

        init = (mech_tke_0, conv_perel_0, h[0], h[0], u[0] * h[0], v[0] * h[0],
                jnp.ones((), h.dtype), h[0], h[0],
                dt_to_dpe[0], ds_to_dpe[0], dt_to_dcolht[0], ds_to_dcolht[0],
                jnp.zeros((), h.dtype), jnp.zeros((), h.dtype),
                jnp.zeros((), h.dtype))
        xs = (h[:-1], h[1:], t0[:-1], t0[1:], s0[:-1], s0[1:],
              u[1:], v[1:], tke_forcing[1:], dt_h_int, hb_hs[1:nz],
              mixlen_shape[1:nz], pres_z[1:nz],
              dt_to_dpe[1:], ds_to_dpe[1:],
              dt_to_dcolht[1:], ds_to_dcolht[1:],
              dt_to_dcolht[:-1], ds_to_dcolht[:-1])
        carry, (kd_int, vstar_int, mixlen_int) = lax.scan(body, init, xs)
        mld_found = carry[7]

        # Pad to the full interface array: no mixing at the surface or bottom.
        zero = jnp.zeros((1,), h.dtype)
        kd_full = jnp.concatenate([zero, kd_int, zero])
        vstar_full = jnp.concatenate([zero, vstar_int, zero])
        mixlen_full = jnp.concatenate([zero, mixlen_int, zero])
        return kd_full, mld_found, vstar_full, mixlen_full

    # --- boundary-layer depth iteration ---------------------------------
    # MOM6 replaces a missing guess with mid-column before any pass,
    # iterating or not (MOM_energetic_PBL.F90:1231).
    mld_guess_0 = jnp.where(mld_guess_in > 0.0, mld_guess_in,
                            0.5 * column_depth)
    if not cfg.use_mld_iteration:
        kd, mld, mixvel, mixlen = one_pass(mld_guess_0)
        return kd, mld, mixvel, mixlen

    def mld_body(i, st):
        (guess, min_mld, max_mld, dmld_min, dmld_max, done,
         kd_keep, mld_keep, mixvel_keep, mixlen_keep) = st

        kd, mld_found, mixvel, mixlen = one_pass(guess)

        # Keep the first converged pass's answer, as MOM6's `exit` does.
        keep = done
        kd_keep = jnp.where(keep, kd_keep, kd)
        mixvel_keep = jnp.where(keep, mixvel_keep, mixvel)
        mixlen_keep = jnp.where(keep, mixlen_keep, mixlen)
        mld_keep = jnp.where(keep, mld_keep, mld_found)

        diff = mld_found - guess
        converged = jnp.abs(diff) < cfg.mld_tol
        if cfg.mld_iter_bug:
            # MOM6's default ordering: a guess that is too shallow is tested
            # before convergence, so diff == +mld_tol exactly updates the
            # lower bound instead of exiting.
            too_shallow = diff > cfg.mld_tol
            converged = (~too_shallow) & converged
        else:
            too_shallow = (~converged) & (diff > 0.0)

        min_mld_n = jnp.where(too_shallow, guess, min_mld)
        dmld_min_n = jnp.where(too_shallow, diff, dmld_min)
        too_deep = (~too_shallow) & (~converged)
        max_mld_n = jnp.where(too_deep, guess, max_mld)
        dmld_max_n = jnp.where(too_deep, diff, dmld_max)

        if cfg.mld_bisection:
            next_guess = 0.5 * (min_mld_n + max_mld_n)
        else:
            # MOM6 takes the false-position step only from its third
            # iteration on, and skips it every fourth (OBL_it = i + 1).
            obl_it = i + 1
            bracketed = ((dmld_min_n > 0.0) & (dmld_max_n < 0.0)
                         & (obl_it > 2) & ((obl_it - 1) % 4 > 0))
            false_pos = _safe_div(
                dmld_min_n * max_mld_n - dmld_max_n * min_mld_n,
                dmld_min_n - dmld_max_n)
            in_bounds = (mld_found > min_mld_n) & (mld_found < max_mld_n)
            next_guess = jnp.where(
                bracketed, false_pos,
                jnp.where(in_bounds, mld_found,
                          0.5 * (min_mld_n + max_mld_n)))

        done_n = done | converged
        guess_n = jnp.where(done_n, guess, next_guess)
        return (guess_n, min_mld_n, max_mld_n, dmld_min_n, dmld_max_n, done_n,
                kd_keep, mld_keep, mixvel_keep, mixlen_keep)

    zero_int = jnp.zeros((nz + 1,), h.dtype)
    init_mld = (mld_guess_0, jnp.zeros((), h.dtype), column_depth,
                jnp.asarray(-1.0, h.dtype), jnp.asarray(1.0, h.dtype),
                jnp.array(False), zero_int, h[0], zero_int, zero_int)
    out = lax.fori_loop(0, cfg.max_mld_its, mld_body, init_mld)
    return out[6], out[7], out[8], out[9]


@partial(jax.jit, static_argnames=("cfg", "g", "rho_0"))
def _epbl_columns(h, u, v, t0, s0, dsv_dt, dsv_ds, tke_forcing,
                  b_flux, absf, u_star, mech_tke_in, mld_guess,
                  cfg: EPBLConfig, g: float, rho_0: float):
    """``_epbl_column`` vmapped over a flat stack of columns, compiled once.

    The jit is not an optimisation detail: driven step-by-step from Python,
    the ePBL graph is large enough that re-dispatching it every timestep
    exhausts the process address space during XLA codegen. Compiling once
    per (shape, config) keeps a long integration bounded, and ``EPBLConfig``
    is a NamedTuple of scalars so it is hashable and safe as a static arg.
    """
    return jax.vmap(_epbl_column, in_axes=(0,) * 8 + (0, 0, 0, 0, 0)
                    + (None, None, None))(
        h, u, v, t0, s0, dsv_dt, dsv_ds, tke_forcing,
        b_flux, absf, u_star, mech_tke_in, mld_guess, cfg, g, rho_0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _require_forcing(**fields):
    """Refuse missing surface forcing, as MOM6 does: a zero default would run
    the column unforced with no downstream guard able to see it."""
    missing = [k for k, v in fields.items() if v is None]
    if missing:
        raise ValueError(
            f"ePBL needs its surface forcing; {', '.join(missing)} is None. "
            "Pass explicit zeros for an unforced column.")


def epbl_vertical_mixing(
    u, v, T, S, dSV_dT, dSV_dS, h,
    cfg: EPBLConfig,
    tau_x=None, tau_y=None, B_flux=None, tke_forcing=None,
    absf=None, mld_guess=None,
    g: float = constants.g,
    rho_0: float = constants.rho_ocean,
    apply_diffusion: bool = False,
) -> VerticalMixingOutput:
    """Apply ePBL vertical mixing to an arbitrarily shaped field of columns.

    Parameters
    ----------
    u, v, T, S : array ``(..., nlev)``
        Velocities [m/s], potential temperature [degC] and salinity [psu],
        ordered surface to bottom.
    dSV_dT, dSV_dS : array ``(..., nlev)``
        Partial derivatives of specific volume with respect to temperature
        [m^3/kg/degC] and salinity [m^3/kg/psu].
    h : array ``(..., nlev)`` or ``(nlev,)``
        Layer thicknesses [m].
    tau_x, tau_y : array ``(...)`` or None
        Surface wind stress [Pa]. Required (``None`` raises).
    B_flux : array ``(...)`` or None
        Surface buoyancy flux [m^2/s^3], POSITIVE when stabilizing (warming),
        matching MOM6's ``buoyancyFlux``. Required (``None`` raises).
    tke_forcing : array ``(..., nlev)`` or None
        Energy per unit area [J/m^2] required (negative) or released
        (positive) by mixing the surface forcing through each layer. Only the
        top layer is non-zero for a non-penetrating flux; see
        :func:`surface_tke_forcing`. Required (``None`` raises).
    absf : array ``(...)`` or None
        Absolute Coriolis parameter [1/s]. Defaults to zero.
    mld_guess : array ``(...)`` or None
        First guess at the boundary-layer depth [m]. ``None`` or a
        non-positive value starts from half the column depth, which is
        MOM6's ``MLD_ITERATION_GUESS = False`` behaviour.

    Returns
    -------
    VerticalMixingOutput
        ``K_v`` and ``A_v`` at the ``nlev-1`` interior interfaces, with the
        configured backgrounds added. ePBL sets one diffusivity, so the
        viscosity is the same profile with its own background floor
        (Prandtl number one, as in MOM6).
    """
    validate_epbl_config(cfg)
    _require_forcing(tau_x=tau_x, tau_y=tau_y, B_flux=B_flux,
                     tke_forcing=tke_forcing)
    if apply_diffusion:
        raise NotImplementedError(
            "epbl_vertical_mixing does not apply the diffusion itself: ePBL "
            "is an implicit-mixing closure, so it returns K_v/A_v for the "
            "backward-Euler solver. Set implicit_vertical_mixing=True on the "
            "driver (apply_diffusion=False here)."
        )

    nlev = T.shape[-1]
    horiz = T.shape[:-1]
    dtype = T.dtype

    def _b(x, default=0.0):
        if x is None:
            return jnp.full(horiz, default, dtype=dtype)
        return jnp.broadcast_to(jnp.asarray(x, dtype), horiz)

    tau_x_b, tau_y_b = _b(tau_x), _b(tau_y)
    b_flux_b = _b(B_flux)
    absf_b = _b(absf)
    mld_b = _b(mld_guess, -1.0)

    if h.ndim == 1:
        h_b = jnp.broadcast_to(jnp.asarray(h, dtype), horiz + (nlev,))
    else:
        h_b = jnp.asarray(h, dtype)

    if tke_forcing is None:
        tke_f = jnp.zeros(horiz + (nlev,), dtype)
    else:
        tke_f = jnp.broadcast_to(jnp.asarray(tke_forcing, dtype),
                                 horiz + (nlev,))

    # Friction velocity and the mechanical TKE input before mstar.
    tau_mag = _safe_sqrt(tau_x_b**2 + tau_y_b**2)
    # MOM6 prices mech_TKE from the raw u_star and floors u_star after.
    u_star_raw = _safe_sqrt(tau_mag / rho_0)
    mech_tke_in = cfg.dt_s * rho_0 * u_star_raw**3
    u_star = jnp.maximum(u_star_raw, cfg.ustar_min)

    # Blend the local f with the planetary rotation rate (MOM6 ML_OMEGA_FRAC).
    if cfg.omega_frac >= 1.0:
        absf_eff = jnp.full_like(absf_b, 2.0 * constants.Omega)
    elif cfg.omega_frac > 0.0:
        absf_eff = _safe_sqrt(cfg.omega_frac * 4.0 * constants.Omega**2
                              + (1.0 - cfg.omega_frac) * absf_b**2)
    else:
        absf_eff = absf_b

    flat = math.prod(horiz) if horiz else 1

    r = (-1, nlev)
    kd, mld, mixvel, mixlen = _epbl_columns(
        h_b.reshape(r), u.reshape(r), v.reshape(r), T.reshape(r), S.reshape(r),
        dSV_dT.reshape(r), dSV_dS.reshape(r), tke_f.reshape(r),
        b_flux_b.reshape(flat), absf_eff.reshape(flat),
        u_star.reshape(flat), mech_tke_in.reshape(flat), mld_b.reshape(flat),
        cfg, g, rho_0,
    )

    # Interior interfaces only, matching VerticalMixingOutput's convention.
    k_interior = kd[:, 1:nlev].reshape(horiz + (nlev - 1,))
    K_v = k_interior + cfg.K_bg
    A_v = k_interior + cfg.A_bg

    # ePBL returns diffusivities only; the implicit solver applies them, so
    # every explicit tendency channel is zero.  The boundary-layer depth and
    # the mixing length/velocity are available from ``epbl_diagnostics``.
    zeros = jnp.zeros_like(T)
    del mld, mixvel, mixlen
    return VerticalMixingOutput(
        du_dt=zeros, dv_dt=zeros, dT_dt=zeros, dS_dt=jnp.zeros_like(S),
        K_v=K_v, A_v=A_v,
    )


def epbl_diagnostics(
    u, v, T, S, dSV_dT, dSV_dS, h, cfg: EPBLConfig,
    tau_x=None, tau_y=None, B_flux=None, tke_forcing=None,
    absf=None, mld_guess=None,
    g: float = constants.g, rho_0: float = constants.rho_ocean,
):
    """Return ``(K_v_raw, mld, mixvel, mixlen)`` for diagnosis and testing.

    ``K_v_raw`` is the full ``nlev+1`` interface array WITHOUT the background
    floors, exactly as MOM6's ``Kd_ePBL`` diagnostic; ``mld`` is the
    diagnosed boundary-layer depth [m] (MOM6 ``ePBL_h_ML``).
    """
    validate_epbl_config(cfg)
    _require_forcing(tau_x=tau_x, tau_y=tau_y, B_flux=B_flux,
                     tke_forcing=tke_forcing)
    nlev = T.shape[-1]
    horiz = T.shape[:-1]
    dtype = T.dtype

    def _b(x, default=0.0):
        if x is None:
            return jnp.full(horiz, default, dtype=dtype)
        return jnp.broadcast_to(jnp.asarray(x, dtype), horiz)

    tau_mag = _safe_sqrt(_b(tau_x) ** 2 + _b(tau_y) ** 2)
    # MOM6 prices mech_TKE from the raw u_star and floors u_star after.
    u_star_raw = _safe_sqrt(tau_mag / rho_0)
    mech_tke_in = cfg.dt_s * rho_0 * u_star_raw**3
    u_star = jnp.maximum(u_star_raw, cfg.ustar_min)
    absf_b = _b(absf)
    if cfg.omega_frac >= 1.0:
        absf_eff = jnp.full_like(absf_b, 2.0 * constants.Omega)
    elif cfg.omega_frac > 0.0:
        absf_eff = _safe_sqrt(cfg.omega_frac * 4.0 * constants.Omega**2
                              + (1.0 - cfg.omega_frac) * absf_b**2)
    else:
        absf_eff = absf_b

    h_b = (jnp.broadcast_to(jnp.asarray(h, dtype), horiz + (nlev,))
           if h.ndim == 1 else jnp.asarray(h, dtype))
    tke_f = (jnp.zeros(horiz + (nlev,), dtype) if tke_forcing is None
             else jnp.broadcast_to(jnp.asarray(tke_forcing, dtype),
                                   horiz + (nlev,)))
    flat = math.prod(horiz) if horiz else 1
    r = (-1, nlev)

    kd, mld, mixvel, mixlen = _epbl_columns(
        h_b.reshape(r), u.reshape(r), v.reshape(r), T.reshape(r), S.reshape(r),
        dSV_dT.reshape(r), dSV_dS.reshape(r), tke_f.reshape(r),
        _b(B_flux).reshape(flat), absf_eff.reshape(flat),
        u_star.reshape(flat), mech_tke_in.reshape(flat),
        _b(mld_guess, -1.0).reshape(flat), cfg, g, rho_0,
    )
    return (kd.reshape(horiz + (nlev + 1,)),
            mld.reshape(horiz),
            mixvel.reshape(horiz + (nlev + 1,)),
            mixlen.reshape(horiz + (nlev + 1,)))


def surface_tke_forcing(q_net, salt_flux, h_top, dSV_dT_top, dSV_dS_top,
                        *, c_sw: float, dt: float,
                        g: float = constants.g,
                        rho_0: float = constants.rho_ocean):
    """Energy released by mixing the surface fluxes through the top layer.

    This is MOM6's ``cTKE`` for the top layer (``MOM_diabatic_aux``
    ``applyBoundaryFluxesInOut``) restricted to a non-penetrating heat flux
    and no net water flux:

        cTKE = -(1/2) h g rho_0^2 (dT dSV_dT + dS dSV_dS)

    with ``dT = q_net dt / (rho_0 c_sw h)``.  Surface cooling makes ``dT``
    negative and ``cTKE`` positive, i.e. convectively released energy.

    Parameters
    ----------
    q_net : array
        Net surface heat flux [W/m^2], POSITIVE INTO the ocean.
    c_sw, dt : float
        Seawater heat capacity [J/kg/K] and the model timestep [s]. Both are
        REQUIRED: the run's own heat capacity must be used (a defaulted one
        silently rescales the released energy), and the energy depends on the
        timestep through the temperature increment.
    salt_flux : array or None
        Net surface salt flux [psu*kg/m^2/s], positive = saltier surface.
    h_top : array
        Top layer thickness [m].
    """
    d_temp = q_net * dt / (rho_0 * c_sw * jnp.maximum(h_top, _TINY))
    d_salt = (0.0 if salt_flux is None
              else salt_flux * dt / (rho_0 * jnp.maximum(h_top, _TINY)))
    g_hconv2 = g * rho_0**2
    return -(0.5 * h_top * g_hconv2) * (d_temp * dSV_dT_top
                                        + d_salt * dSV_dS_top)
