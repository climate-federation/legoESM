"""Plant hydraulics shared by the three arms: soil-to-leaf water supply.

Root-xylem-leaf water transport of one leaf class, used by every
``transpiration_stress`` mode that replaces the empirical theta multiplier:

* ``hydraulic`` (baseline / least-cost arms): the Medlyn demand must equal
  the supply the xylem delivers at the class's draw-down, ``LAI_c E_c(dpsi_c)
  / D = gs_w,c`` (one Newton row per class in ``canopy/solver.py``), and the
  Medlyn slope is down-regulated by the leaf water potential through a
  Weibull factor ``W_g1(psi_L) = 2^(-(psi_L/psi50_g1)^ck_g1)``.
* ``phydro``: the P-hydro profit optimum (``land/phydro.py``) on the same
  supply curve.

Water supply (soil-to-root resistance in series with a vulnerable xylem):

    P(psi)   = 0.5 ** ((psi / psi50) ** ck)                   (Weibull)
    lsc      = 1 / (rsoil_leaf + 1 / (gplant * Pbar))
    Pbar     = mean of P over [psi_leaf, psi_soil]            (Joshi eq. 2)
    E        = lsc * dpsi,  psi_leaf = psi_soil - dpsi

The soil-side state (``soil_root_supply``) is the conductance-weighted soil
water potential of the rooted layers (Bonan et al. 2014 eqs. A23-A25); the
soil-to-root resistance itself is masked to zero (user decision 2026-10-28).
The xylem constants (``gplant_mol``, ``psi50_xylem_mpa``, ``ck_xylem``,
``root_biomass_gm2``) live in ``PHydroConfig`` and are shared by both modes;
``PlantHydraulicsConfig`` carries the parameters only the ``hydraulic`` mode
reads (the slope's vulnerability curve).

References: Joshi et al. (2022) Nat. Plants 8, 1304-1316; Bonan et al.
(2014) GMD 7, 2193-2222; Kennedy et al. (2019) JAMES 11, 485-513.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Soil-to-leaf water supply of a leaf class: conductance-weighted "
        "soil potential of the rooted layers (Bonan 2014 A23-A25, resistance "
        "masked to zero), Weibull xylem vulnerability, E = lsc(psi_s, dpsi) "
        "dpsi with Pbar the path mean of the vulnerability curve, the class "
        "conductance gs_w = LAI_c E / D, and the Weibull down-regulation of "
        "the Medlyn slope by the leaf water potential."
    ),
    "inputs": {
        "psi_soil": "m", "theta_soil": "1", "dz": "m", "root_frac": "1",
        "lai": "1", "dpsi": "MPa", "psi_s": "MPa", "D/P": "mol/mol",
    },
    "outputs": {
        "psi_s_mpa": "MPa", "rsoil_leaf": "MPa m^2 s/mol",
        "E_leaf": "mol/m^2 leaf/s", "gs_w": "mol/m^2/s", "W": "1",
    },
    "sign_convention": (
        "psi <= 0 (matric, MPa); dpsi >= 0 is the soil-minus-leaf draw-down, "
        "psi_leaf = psi_s - dpsi; E >= 0 for dpsi >= 0; 0 < W <= 1."
    ),
    "conserves": [],  # a conductance, not a flux budget
    "differentiable": True,
    "reference": (
        "Joshi et al. (2022) Nature Plants 8 1304-1316; Bonan et al. (2014) "
        "GMD 7 2193-2222 eqs. A23-A25; Kennedy et al. (2019) JAMES 11 485-513."
    ),
    "idealized_test": (
        "tests/land/unit/test_hydraulic_stress.py: with the vulnerability "
        "curves switched off (psi50 -> -inf) the hydraulic baseline "
        "reproduces the unstressed Medlyn column; the supply-demand row is "
        "zero at convergence; sunlit leaves draw down more than shaded; gs "
        "falls monotonically with drying soil."
    ),
}

__param_spec__ = {
    "PlantHydraulicsConfig": {
        "scheme_key": "land.plant_hydraulics",
        "params": {
            "psi50_g1_mpa": {
                "units": "MPa", "bounds": (-8.0, -0.5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": (
                    "Kennedy et al. 2019 (CLM5 PHS: leaf psi50 of the stomatal "
                    "down-regulation); design docs/land/hydraulic_stress_design.md"),
                "shape": None,
            },
            "ck_g1": {
                "units": "1", "bounds": (1.0, 6.0), "tunable_tier": 0,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": "shared Weibull shape, not tuned (design decision 6)",
                "shape": None,
            },
        },
    },
}

# --- SPA fine-root geometry (Bonan 2014 / CLM-ML MLpftcon table; fixed) ---
_ROOT_RADIUS_M = 0.29e-3   # [m] fine-root radius (root_radius_SPA)
_ROOT_DENSITY_GM3 = 0.31e6  # [g biomass / m3 root] (root_density_SPA)
_ROOT_RESIST = 25.0        # [MPa s g mmol-1 H2O] root resistivity (root_resist_SPA)
_MMOL_PER_MOL = 1.0e3      # SPA's mmol convention -> this module's mol convention
_RBD_FLOOR = 1e-10  # coeff-ok: Bonan A23 root-biomass-density floor [g/m3]
_HK_FLOOR = 1e-30   # [mol m-1 s-1 MPa-1] floor of the soil conductivity in the radial resistance
_G_SOIL_FLOOR = 1e-30  # [mol m-2 s-1 MPa-1] floor of the summed layer conductance
_MPA_PER_PA = 1e-6
# Wettest soil matric potential a root can see: field capacity.  The Richards
# lane can overshoot to POSITIVE potentials for a step when rain lands on dry
# soil (a water-balance artefact); fed to the optimum that reverses the
# soil-to-leaf gradient and yields negative conductance and capacities.
_PSI_SOIL_CAP_MPA = -0.033
_PSI_SOIL_CAP_WIDTH_MPA = 0.005  # softplus width of the cap (AD safety)
_LN2 = float(np.log(2.0))
# Weibull argument guards: the ratio psi/psi50 is clipped so (ratio ** ck) and
# its derivatives stay finite for every admissible ck.
_WEIBULL_RATIO_MIN = 1.0e-12
_WEIBULL_RATIO_MAX = 1.0e3
# Gauss-Legendre nodes/weights on [0, 1] for the xylem-path mean of P.
_GL_NODES, _GL_WEIGHTS = (
    0.5 * (np.polynomial.legendre.leggauss(8)[0] + 1.0),
    0.5 * np.polynomial.legendre.leggauss(8)[1],
)
_D_FRAC_FLOOR = 1e-9  # coeff-ok: div-by-zero guard on VPD/P in gs = E / D


class PlantHydraulicsConfig(NamedTuple):
    """Parameters read only by ``transpiration_stress='hydraulic'``.

    The xylem supply constants are ``PHydroConfig``'s (shared with the
    P-hydro arm); these are the Weibull curve that closes the Medlyn slope
    with the leaf water potential, ``W_g1 = 2^(-(psi_L/psi50_g1)^ck_g1)``.
    """

    psi50_g1_mpa: float = -2.0  # [MPa] leaf potential at which the slope is halved
    ck_g1: float = 1.5          # [-] Weibull shape of the slope curve


class PhydroSupply(NamedTuple):
    """Soil-side supply state, all ``(ncol,)``."""

    rsoil_leaf: jax.Array  # [MPa m2 leaf s/mol] soil-to-root resistance per leaf area
    psi_s_mpa: jax.Array   # [MPa] conductance-weighted soil water potential


class LeafHydraulics(NamedTuple):
    """Supply state of ONE leaf class for the instantaneous step, all ``(ncol,)``.

    ``dpsi_mpa`` is the class's Newton unknown (soil-to-leaf draw-down);
    the soil fields are THIS step's (not the acclimated) supply; the
    hydraulic constants travel as per-column arrays so the canopy solver's
    implicit backward pass can differentiate with respect to them.
    ``gamma_cost`` is read by the P-hydro stationarity row only (None under
    ``hydraulic``).
    """

    dpsi_mpa: jax.Array        # [MPa] draw-down (Newton unknown)
    psi_s_mpa: jax.Array       # [MPa] conductance-weighted soil potential
    rsoil_leaf: jax.Array      # [MPa m2 leaf s/mol] soil-to-root resistance
    lai_class: jax.Array       # [m2 leaf/m2 ground] leaf area of the class
    gplant_mol: jax.Array      # [mol/m2 leaf/s/MPa]
    psi50_xylem_mpa: jax.Array  # [MPa]
    ck_xylem: jax.Array        # [-]
    gamma_cost: jax.Array | None  # [umol/m2/s/MPa^2] (phydro only)


def soil_root_supply(
    psi_soil_m: jax.Array,
    theta_soil: jax.Array,
    dz_m: jax.Array,
    root_frac: jax.Array,
    lai: jax.Array,
    hydraulics_cfg,
    cfg,
) -> PhydroSupply:
    """Soil-to-root resistance (Bonan 2014 eqs. A23-A25) and bulk soil potential.

    Layered soil-to-root resistances from root geometry + the Richards
    lane's own hydraulic conductivity; conductances sum in parallel across
    layers.  The bulk soil potential is the CONDUCTANCE-weighted mean — the
    exact effective potential of parallel layers feeding one root — so a
    dried-out layer drops out through its own collapsing conductance.
    ``rsoil_leaf`` is returned as ZERO (see the note at the return); the xylem
    resistance is added where the draw-down is known.  ``cfg`` is any config
    carrying ``root_biomass_gm2`` (``PHydroConfig``).

    Shapes: ``psi_soil_m``/``theta_soil``/``root_frac`` are
    ``(ncol, nlev)``; ``dz_m`` is ``(nlev,)``; ``lai`` is ``(ncol,)``.
    """
    from legoesm.land.soil_hydraulics import hydraulic_conductivity

    # Soil conductivity [m/s] -> [mol m-1 s-1 MPa-1]:
    # (m/s) * (mol m-3 water) / (MPa m-1 head) with head = rho*g*1e-6.
    hk_ms = hydraulic_conductivity(psi_soil_m, theta_soil, hydraulics_cfg)
    mol_per_m3 = constants.rho_water / constants.M_h2o
    mpa_per_m = constants.rho_water * constants.g * _MPA_PER_PA
    hk_mol = jnp.maximum(hk_ms, 0.0) * mol_per_m3 / mpa_per_m

    # Matric potential [MPa], smoothly capped at field capacity: ~psi well
    # below the cap, -> the cap from below above it (sign: psi <= 0, drier is
    # more negative).
    smp_mpa = _PSI_SOIL_CAP_MPA - _PSI_SOIL_CAP_WIDTH_MPA * jax.nn.softplus(
        (_PSI_SOIL_CAP_MPA - psi_soil_m * mpa_per_m) / _PSI_SOIL_CAP_WIDTH_MPA)

    dz = jnp.broadcast_to(dz_m, psi_soil_m.shape)
    rbd = jnp.maximum(cfg.root_biomass_gm2 * root_frac / dz, _RBD_FLOOR)
    rld = rbd / (_ROOT_DENSITY_GM3 * jnp.pi * _ROOT_RADIUS_M ** 2)
    root_dist = jnp.sqrt(1.0 / (rld * jnp.pi))
    # A23 (radial soil-to-root) + A24 (root tissue).  _ROOT_RESIST is in the
    # SPA mmol convention; _MMOL_PER_MOL converts the resulting resistance to
    # the mol convention used throughout this module.
    soilr1 = jnp.log(root_dist / _ROOT_RADIUS_M) / (
        2.0 * jnp.pi * rld * dz * jnp.maximum(hk_mol, _HK_FLOOR))
    soilr2 = (_ROOT_RESIST * _MMOL_PER_MOL) / (rbd * dz)
    soilr = soilr1 + soilr2  # [MPa m2 ground s mol-1]

    g_layers = 1.0 / soilr
    g_soil_ground = jnp.maximum(jnp.sum(g_layers, axis=-1), _G_SOIL_FLOOR)  # [mol/m2 ground/s/MPa]
    psi_s = jnp.sum(smp_mpa * g_layers, axis=-1) / g_soil_ground

    # The soil-to-root resistance is MASKED to zero (user decision 2026-10-28):
    # the plant sees the conductance-weighted soil potential at its root
    # surface, as in rphydro.  The layer conductances above still weight that
    # potential.  (The per-leaf-area resistance, Bonan A25, would be
    # ``max(lai, lai_floor) / g_soil_ground``; its running mean, needed by the
    # acclimated optimum, is dominated by dry-spell spikes of 1e7-1e8 and shut
    # the plant down for weeks after rain.)
    del lai
    rsoil_leaf = jnp.zeros_like(psi_s)
    return PhydroSupply(rsoil_leaf=rsoil_leaf, psi_s_mpa=psi_s)


def _weibull_exponent(psi_mpa: jax.Array, psi50_mpa, ck) -> jax.Array:
    """``ln(2) * (psi/psi50)**ck`` with the ratio clipped to a finite range.

    The clip keeps the power AND its derivatives (w.r.t. psi, psi50 and ck)
    finite: at psi = 0 the bare ``ratio**ck`` has a NaN ck-derivative
    (``0 * log 0``), and at very negative psi it overflows.
    """
    ratio = jnp.clip(psi_mpa / psi50_mpa, _WEIBULL_RATIO_MIN, _WEIBULL_RATIO_MAX)
    return _LN2 * ratio ** ck


def weibull_vulnerability(psi_mpa: jax.Array, psi50_mpa, ck) -> jax.Array:
    """Fraction of maximum conductance/capacity left at ``psi``: ``0.5**((psi/psi50)**ck)``."""
    return jnp.exp(-_weibull_exponent(psi_mpa, psi50_mpa, ck))


def weibull_stress(psi_leaf_mpa: jax.Array, psi50_mpa, ck) -> jax.Array:
    """Down-regulation factor ``W(psi_L) = 2^(-(psi_L/psi50)^ck)`` in (0, 1].

    The same Weibull form as the xylem vulnerability, applied to a leaf-level
    quantity (the Medlyn slope; the capacities in a later step): 1 for a
    wet leaf, 1/2 at ``psi50``, -> 0 as the leaf dries.
    """
    return weibull_vulnerability(psi_leaf_mpa, psi50_mpa, ck)


def leaf_psi(dpsi_mpa: jax.Array, psi_s_mpa: jax.Array) -> jax.Array:
    """Leaf water potential ``psi_L = psi_s - dpsi`` [MPa]."""
    return psi_s_mpa - dpsi_mpa


def supply_transpiration(dpsi, psi_s_mpa, rsoil_leaf, gplant_mol,
                         psi50_xylem_mpa, ck_xylem):
    """Transpiration the soil-to-leaf path supplies at a draw-down.

    ``E = lsc(psi_s, dpsi) * dpsi`` [mol H2O / m2 leaf / s] with
    ``lsc = 1/(rsoil + 1/(gplant * Pbar))`` and ``Pbar`` the mean of the
    Weibull curve over the xylem path ``[psi_s - dpsi, psi_s]`` (fixed-node
    quadrature: smooth in dpsi, psi50 and ck; no incomplete-gamma function).
    Shared by the acclimated and the instantaneous optimum and by the
    supply-demand closure.
    """
    pbar = sum(
        float(w) * weibull_vulnerability(
            psi_s_mpa - float(t) * dpsi, psi50_xylem_mpa, ck_xylem)
        for t, w in zip(_GL_NODES, _GL_WEIGHTS))
    kx = gplant_mol * pbar
    # lsc = 1/(rsoil + 1/kx), written without the 1/kx so kx -> 0 is finite.
    lsc = kx / (1.0 + rsoil_leaf * kx)
    return lsc * dpsi


# --- leaf VPD floor of the supply conductance gs = E / D ---
_VPD_FLOOR_PA = 50.0        # [Pa] floor on the leaf-to-air VPD in gs = E / D (as the rphydro kernel)
_VPD_FLOOR_WIDTH_PA = 0.0   # [Pa] width of the smooth floor; 0 = hard max


def floored_vpd(vpd_pa: jax.Array) -> jax.Array:
    """Leaf VPD with the kernel's 50 Pa floor, optionally smoothed.

    The hard ``max(VPD, 50)`` has a derivative jump at the floor; a shaded
    leaf colder than the canopy-air dew point sits exactly there and the
    canopy Newton stalls on the kink.  With a width ``w`` the floor is the
    smooth-max ``50 + 0.5 (v + sqrt(v^2 + w^2))``, ``v = VPD - 50``: same
    limits, continuous derivative, ``w/2`` above the hard floor at VPD = 50.
    """
    v = vpd_pa - _VPD_FLOOR_PA
    if _VPD_FLOOR_WIDTH_PA <= 0.0:
        return jnp.maximum(vpd_pa, _VPD_FLOOR_PA)
    return _VPD_FLOOR_PA + 0.5 * (v + jnp.sqrt(v * v + _VPD_FLOOR_WIDTH_PA ** 2))


def leaf_supply_gs_w(hyd: LeafHydraulics, d_frac: jax.Array) -> jax.Array:
    """Water-vapour conductance of the whole leaf CLASS at its draw-down.

    ``E_leaf = supply_transpiration(dpsi)`` per leaf area; the class
    transpires ``lai_class * E_leaf`` per ground area, so
    ``gs_w = lai_class * E_leaf / D`` [mol H2O / m2 ground / s].  The
    draw-down is taken as given (>= 0): the canopy solver projects its
    trial steps onto that bound, so no floor is applied here (a smooth floor
    put a corner at zero that the damped Newton stalled on).
    """
    e_leaf = supply_transpiration(
        hyd.dpsi_mpa, hyd.psi_s_mpa, hyd.rsoil_leaf,
        hyd.gplant_mol, hyd.psi50_xylem_mpa, hyd.ck_xylem)
    return hyd.lai_class * e_leaf / jnp.maximum(d_frac, _D_FRAC_FLOOR)


def supply_demand_residual(hyd: LeafHydraulics, d_frac: jax.Array,
                           gs_w_demand: jax.Array, b0: jax.Array) -> jax.Array:
    """Supply-equals-demand closure of a leaf class's draw-down [mol/m2/s].

    Zero where the water the xylem delivers at ``dpsi``, as the conductance
    ``leaf_supply_gs_w``, plus the cuticular intercept ``b0`` (not fed by the
    xylem, as in the P-hydro conductance), equals the stomatal model's
    demand ``gs_w_demand`` (mol H2O/m2/s, the same quantity the leaf energy
    balance uses).  Negative below the root (``dpsi`` too small), positive
    above it while the supply curve rises.  At ``dpsi = 0`` it is
    ``b0 - gs_w_demand <= 0`` for a Medlyn demand floored at ``g0 = b0``, so
    the root is at or above zero draw-down: zero at night, where the demand
    is the intercept alone.
    """
    return leaf_supply_gs_w(hyd, d_frac) + b0 - gs_w_demand


__all__ = [
    "PlantHydraulicsConfig", "PhydroSupply", "LeafHydraulics",
    "soil_root_supply", "weibull_vulnerability", "weibull_stress", "leaf_psi",
    "supply_transpiration", "floored_vpd", "leaf_supply_gs_w",
    "supply_demand_residual",
]
