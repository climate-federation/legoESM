"""P-hydro: profit-maximizing stomatal/capacity optimality on plant hydraulics.

Joshi et al. (2022, Nature Plants) profit maximization — assimilation net of
the cost of photosynthetic capacity and of hydraulic risk — over the
soil-to-leaf water-potential draw-down ``dpsi`` and the leaf-internal CO2
ratio ``chi``.  Like the least-cost P model (``land/p_model.py``), it supplies
ACCLIMATED PARAMETERS (leaf-top Vcmax25, rjv25, and a stomatal slope for the
active conductance model) — the host kernels compute fluxes.

Profit, at the P-model acclimation means and the CURRENT root-zone soil water
potential (transcribed from jaideep777/phydro @a59e2c6, ``phydro_analytical``):

    F(chi, dpsi) = A - alpha * Jmax - gamma * dpsi^2          [Joshi eq. 6]
    A      = gs * ca * (1 - chi)                              (CO2 supply)
    gs     = lsc(dpsi) * dpsi / (1.6 * D)                     (water supply)
    J      = 4 * A * (chi + 2 g) / (chi - g),  g = Gamma*/ca  (A = A_j)
    Jmax   = [Jlim * J / sqrt(Jlim^2 - J^2)] / f(psi_leaf),   Jlim = 4 phi0 Iabs

``Jmax`` is the INVESTED capacity whose water-stressed fraction
``f(psi_leaf) * Jmax`` delivers the electron transport ``J`` (Smith light
response, as in rpmodel / rphydro).  Vcmax follows from coordination
(A_c = A_j) at the optimum.  The two stationarity conditions are solved by
nested bisection (``F`` is concave in ``chi`` at fixed ``dpsi``); gradients
come from the implicit function theorem at the optimum, never from the
iterations.

Water supply (soil-to-root resistance in series with a vulnerable xylem):

    P(psi)   = 0.5 ** ((psi / psi50) ** ck)                   (Weibull)
    lsc      = 1 / (rsoil_leaf + 1 / (gplant * Pbar))
    Pbar     = mean of P over [psi_leaf, psi_soil]            (Joshi eq. 2)

DELIBERATE DEPARTURES from Joshi et al. (2022) / rphydro:

1. **Soil-to-root resistance in series** (Bonan et al. 2014, GMD 7,
   2193-2222, eqs. A23-A25) from root geometry + the Richards lane's own
   hydraulic conductivity; rphydro has a plant conductance only.  The xylem
   integral runs from the bulk soil potential, not the root-surface
   potential (exact as ``rsoil_leaf -> 0``).
2. **Non-stomatal limitation.**  ``f(psi_leaf)`` — a second Weibull curve —
   scales the effective Jmax; Joshi has no such term (``f = 1`` recovers
   their model).
3. **Slope, not gs.**  The optimum's chi maps to the ACTIVE stomatal
   model's slope at the acclimated means (valid for g0 ~ 0), see
   :func:`slope_for_model`.
4. **Kinetics and temperature responses from the shared repo constants**
   (as ``land/p_model.py`` departures 2-3), not rphydro's literals.

References: Joshi et al. (2022) Nat. Plants 8, 1304-1316; Bonan et al.
(2014) GMD 7, 2193-2222 (soil-to-root resistance); Kennedy et al. (2019)
JAMES 11, 485-513 (Weibull vulnerability form used in CLM5).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.canopy.photosynthesis import (
    jmax_temperature_response,
    vcmax_temperature_response,
)
from legoesm.land.leaf_biophysics import DIFFUSIVITY_RATIO_H2O_CO2
from legoesm.land.p_model import (
    PModelAcclimState,
    PModelConfig,
    optimal_chi,
    _phi0,
)

__physics_contract__ = {
    "summary": (
        "P-hydro (Joshi et al. 2022): maximize A - alpha*Jmax - gamma*dpsi^2 "
        "over (chi, dpsi) with gs = lsc(dpsi)*dpsi/(1.6 D), a Weibull xylem "
        "vulnerability curve in series with the Bonan-2014 soil-to-root "
        "resistance, and a Weibull non-stomatal limitation on Jmax, at the "
        "P-model acclimation means and the instantaneous conductance-weighted "
        "soil water potential. Supplies leaf-top Vcmax25, rjv25 and a "
        "stomatal slope for the active model; replaces the empirical theta "
        "stress."
    ),
    "inputs": {
        "psi_soil": "m", "theta_soil": "1", "dz": "m", "root_frac": "1",
        "lai": "1", "acclim means": "see land.p_model",
    },
    "outputs": {
        "vcmax25_leaf": "umol/m^2/s", "rjv25": "1", "chi": "1",
        "dpsi_mpa": "MPa", "psi_leaf_mpa": "MPa", "gs_mol": "mol/m^2/s",
        "converged": "bool",
        "rsoil_leaf": "MPa m^2 s/mol", "psi_s_mpa": "MPa",
    },
    "sign_convention": (
        "psi <= 0 (matric, MPa after conversion); dpsi >= 0 is the "
        "soil-minus-leaf draw-down, psi_leaf = psi_s - dpsi; capacities >= 0 "
        "and -> 0 smoothly as the soil dries."
    ),
    "conserves": [],  # parameter prediction, not a flux
    "differentiable": True,
    "reference": (
        "Joshi et al. (2022) Nature Plants 8 1304-1316; Bonan et al. (2014) "
        "GMD 7 2193-2222 eqs. A23-A25; Kennedy et al. (2019) JAMES 11 485-513."
    ),
    "idealized_test": (
        "tests/land/unit/test_phydro.py: reproduces the rphydro R package "
        "(reference table) with f = 1 and no soil resistance; the returned "
        "optimum dominates a brute-force (chi, dpsi) grid; capacities fall "
        "smoothly with drying soil; implicit gradients match central finite "
        "differences and stay finite on dry and wet columns."
    ),
}

__param_spec__ = {
    "PHydroConfig": {
        "scheme_key": "land.phydro",
        "excluded": {},
        "params": {
            "gplant_mol": {
                "units": "mol m-2 s-1 MPa-1", "bounds": (5e-4, 2e-2),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "hydraulics",
                "reference": "SPA / CLM-ML gplant table (4 mmol m-2 s-1 MPa-1)",
                "shape": None,
            },
            "alpha_cost": {
                "units": "1", "bounds": (0.05, 0.2),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "hydraulics",
                "reference": "Joshi et al. 2022 Table 1 (0.08-0.12)",
                "shape": None,
            },
            "gamma_cost": {
                "units": "umol m-2 s-1 MPa-2", "bounds": (0.0, 10.0),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "hydraulics",
                "reference": "Joshi et al. 2022 Table 1 (0.28-5 by plant type)",
                "shape": None, "legacy_name": "gamma_cost",
            },
            "psi50_xylem_mpa": {
                "units": "MPa", "bounds": (-8.0, -0.5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": "Joshi et al. 2022 (xylem psi50)",
                "shape": None,
            },
            "ck_xylem": {
                "units": "1", "bounds": (1.0, 6.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": "Joshi et al. 2022 (Weibull shape b)",
                "shape": None,
            },
            # Lower bound -1000 MPa is the "no non-stomatal limitation" setting
            # (f ~ 1); a calibration should search a narrower range.
            "psi50_jmax_mpa": {
                "units": "MPa", "bounds": (-1000.0, -0.3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": "non-stomatal limitation of Jmax (this module)",
                "shape": None,
            },
            "ck_jmax": {
                "units": "1", "bounds": (1.0, 6.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": "non-stomatal limitation of Jmax (this module)",
                "shape": None,
            },
            "root_biomass_gm2": {
                "units": "g m-2", "bounds": (50.0, 2000.0),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "hydraulics",
                "reference": "CLM-ML fine-root biomass (Bonan 2014 A23-A24)",
                "shape": None,
            },
        },
    },
}

# --- SPA fine-root geometry (Bonan 2014 / CLM-ML MLpftcon table; fixed) ---
_ROOT_RADIUS_M = 0.29e-3   # [m] fine-root radius (root_radius_SPA)
_ROOT_DENSITY_GM3 = 0.31e6  # [g biomass / m3 root] (root_density_SPA)
_ROOT_RESIST = 25.0        # [MPa s g mmol-1 H2O] root resistivity (root_resist_SPA)

# --- numerics (module constants; iteration counts are never config) ---
_N_CHI_ITER = 30    # inner bisection on dF/dchi (bracket width 2^-30)
_N_DPSI_ITER = 30   # outer bisection on dF/ddpsi
_DPSI_MAX_MPA = 10.0  # [MPa] upper bracket of the draw-down search
_RBD_FLOOR = 1e-10  # coeff-ok: Bonan A23 root-biomass-density floor [g/m3]
_LAI_FLOOR = 1.0e-3  # [m2/m2] guards the per-leaf-area resistance on bare columns
# Wettest soil matric potential a root can see: field capacity.  The Richards
# lane can overshoot to POSITIVE potentials for a step when rain lands on dry
# soil (a water-balance artefact); fed to the optimum that reverses the
# soil-to-leaf gradient and yields negative conductance and capacities.
_PSI_SOIL_CAP_MPA = -0.033
_PSI_SOIL_CAP_WIDTH_MPA = 0.005  # softplus width of the cap (AD safety)
_UMOL_PER_MOL = 1.0e6
_LN2 = float(np.log(2.0))
# Weibull argument guards: the ratio psi/psi50 is clipped so (ratio ** ck) and
# its derivatives stay finite for every admissible ck, and the exponent of the
# INVERSE curve is capped so 1/f cannot overflow.
_WEIBULL_RATIO_MIN = 1.0e-12
_WEIBULL_RATIO_MAX = 1.0e3
_WEIBULL_INV_EXP_MAX = 30.0
# Gauss-Legendre nodes/weights on [0, 1] for the xylem-path mean of P.
_GL_NODES, _GL_WEIGHTS = (
    0.5 * (np.polynomial.legendre.leggauss(8)[0] + 1.0),
    0.5 * np.polynomial.legendre.leggauss(8)[1],
)
# Implicit-gradient guards: the Newton-refinement step is applied only on
# columns with a live interior optimum and a well-conditioned Hessian.
_A_ACTIVE_UMOL = 1.0e-6    # [umol m-2 s-1] below this the plant is shut down
_HESS_DET_RTOL = 1.0e-8    # relative determinant floor of the 2x2 Hessian


class PHydroConfig(NamedTuple):
    """P-hydro tunables (Joshi profit strategy on a vulnerable xylem)."""

    # --- hydraulics ---
    gplant_mol: float = 4.0e-3     # [mol/m2 leaf/s/MPa] maximum xylem conductance
    psi50_xylem_mpa: float = -2.0  # [MPa] xylem potential at 50% conductance loss
    ck_xylem: float = 1.5          # [-] Weibull shape of the xylem curve
    root_biomass_gm2: float = 500.0  # [g/m2] fine-root biomass
    # --- costs ---
    alpha_cost: float = 0.1        # [-] cost per unit Jmax
    gamma_cost: float = 0.0        # [umol/m2/s/MPa^2] hydraulic risk cost
    # --- non-stomatal limitation of Jmax ---
    psi50_jmax_mpa: float = -3.0   # [MPa] leaf potential at 50% Jmax loss
    ck_jmax: float = 1.0           # [-] Weibull shape of the Jmax curve


class PhydroSupply(NamedTuple):
    """Soil-side supply state, all ``(ncol,)``."""

    rsoil_leaf: jax.Array  # [MPa m2 leaf s/mol] soil-to-root resistance per leaf area
    psi_s_mpa: jax.Array   # [MPa] conductance-weighted soil water potential


class PhydroCapacities(NamedTuple):
    """Acclimated parameter predictions from the profit optimum."""

    vcmax25_leaf: jax.Array  # [umol/m^2/s]
    rjv25: jax.Array         # [-]
    chi: jax.Array           # [-] optimal ci/ca (diagnostic + slope source)
    dpsi_mpa: jax.Array      # [MPa] optimal draw-down (diagnostic)
    psi_leaf_mpa: jax.Array  # [MPa] psi_s - dpsi (diagnostic)
    gs_mol: jax.Array        # [mol CO2/m^2/s] optimal stomatal conductance (diagnostic)
    converged: jax.Array     # [bool] False where dpsi hit the search bracket


class _Env(NamedTuple):
    """Per-column inputs of the profit function, all ``(ncol,)``."""

    psi_s: jax.Array    # [MPa]
    rsoil: jax.Array    # [MPa m2 s/mol]
    d_frac: jax.Array   # [mol/mol] VPD / pressure
    ca_umol: jax.Array  # [umol/mol] ambient CO2
    g: jax.Array        # [-] Gamma* / ca
    jlim: jax.Array     # [umol m-2 s-1] 4 * phi0 * Iabs


def soil_root_supply(
    psi_soil_m: jax.Array,
    theta_soil: jax.Array,
    dz_m: jax.Array,
    root_frac: jax.Array,
    lai: jax.Array,
    hydraulics_cfg,
    cfg: PHydroConfig,
) -> PhydroSupply:
    """Soil-to-root resistance (Bonan 2014 eqs. A23-A25) and bulk soil potential.

    Layered soil-to-root resistances from root geometry + the Richards
    lane's own hydraulic conductivity; conductances sum in parallel across
    layers.  The bulk soil potential is the CONDUCTANCE-weighted mean — the
    exact effective potential of parallel layers feeding one root — so a
    dried-out layer drops out through its own collapsing conductance.
    ``rsoil_leaf`` is the per-LEAF-area resistance (A25); the xylem resistance
    is added inside :func:`phydro_optimum`, where it depends on the draw-down.

    Shapes: ``psi_soil_m``/``theta_soil``/``root_frac`` are
    ``(ncol, nlev)``; ``dz_m`` is ``(nlev,)``; ``lai`` is ``(ncol,)``.
    """
    from legoesm.land.soil_hydraulics import hydraulic_conductivity

    # Soil conductivity [m/s] -> [mol m-1 s-1 MPa-1]:
    # (m/s) * (mol m-3 water) / (MPa m-1 head) with head = rho*g*1e-6.
    hk_ms = hydraulic_conductivity(psi_soil_m, theta_soil, hydraulics_cfg)
    mol_per_m3 = constants.rho_water / constants.M_h2o
    mpa_per_m = constants.rho_water * constants.g * 1e-6
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
    # SPA mmol convention; the *1e3 converts the resulting resistance to the
    # mol convention used throughout this module.
    soilr1 = jnp.log(root_dist / _ROOT_RADIUS_M) / (
        2.0 * jnp.pi * rld * dz * jnp.maximum(hk_mol, 1e-30))
    soilr2 = (_ROOT_RESIST * 1e3) / (rbd * dz)
    soilr = soilr1 + soilr2  # [MPa m2 ground s mol-1]

    g_layers = 1.0 / soilr
    g_soil_ground = jnp.maximum(jnp.sum(g_layers, axis=-1), 1e-30)  # [mol/m2 ground/s/MPa]
    psi_s = jnp.sum(smp_mpa * g_layers, axis=-1) / g_soil_ground

    # A25: per-leaf-area soil resistance.
    rsoil_leaf = jnp.maximum(lai, _LAI_FLOOR) / g_soil_ground
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


def _supply_side(dpsi: jax.Array, env: _Env, cfg: PHydroConfig):
    """``(gs, 1/f)`` at a draw-down: CO2 conductance and the Jmax stress inverse."""
    # Mean of P over the xylem path [psi_leaf, psi_s] (fixed-node quadrature:
    # smooth in dpsi, psi50 and ck; no incomplete-gamma function).
    pbar = sum(
        float(w) * weibull_vulnerability(
            env.psi_s - float(t) * dpsi, cfg.psi50_xylem_mpa, cfg.ck_xylem)
        for t, w in zip(_GL_NODES, _GL_WEIGHTS))
    kx = cfg.gplant_mol * pbar
    # lsc = 1/(rsoil + 1/kx), written without the 1/kx so kx -> 0 is finite.
    lsc = kx / (1.0 + env.rsoil * kx)
    gs = lsc * dpsi / (DIFFUSIVITY_RATIO_H2O_CO2 * env.d_frac)
    inv_f = jnp.exp(jnp.minimum(
        _weibull_exponent(env.psi_s - dpsi, cfg.psi50_jmax_mpa, cfg.ck_jmax),
        _WEIBULL_INV_EXP_MAX))
    return gs, inv_f


def _electron_transport(chi: jax.Array, gs: jax.Array, env: _Env):
    """``(A, J)``: CO2-supply assimilation and the J that sustains it (A = A_j)."""
    a = gs * env.ca_umol * (1.0 - chi)
    j = 4.0 * a * (chi + 2.0 * env.g) / (chi - env.g)
    return a, j


def _jmax_per_j(j: jax.Array, env: _Env) -> jax.Array:
    """``Jmax_eff / J`` of the Smith light response: ``Jlim / sqrt(Jlim^2 - J^2)``."""
    return env.jlim / jnp.sqrt(jnp.maximum(env.jlim ** 2 - j ** 2, 1e-30))


def _profit_chi(chi, gs, inv_f, dpsi, env: _Env, cfg: PHydroConfig):
    """Profit ``A - alpha*Jmax - gamma*dpsi^2`` at a given supply state."""
    a, j = _electron_transport(chi, gs, env)
    jmax = j * _jmax_per_j(j, env) * inv_f
    return a - cfg.alpha_cost * jmax - cfg.gamma_cost * dpsi * dpsi


def _profit(chi, dpsi, env: _Env, cfg: PHydroConfig):
    gs, inv_f = _supply_side(dpsi, env, cfg)
    return _profit_chi(chi, gs, inv_f, dpsi, env, cfg)


def _d(fn, x):
    """Element-wise derivative of an element-wise ``fn`` at ``x``."""
    return jax.jvp(fn, (x,), (jnp.ones_like(x),))[1]


def _chi_min(gs: jax.Array, env: _Env) -> jax.Array:
    """Lower end of the feasible chi interval: where ``J`` reaches ``Jlim``.

    Root of ``chi^2 + b*chi - g*(2 + q) = 0`` with ``q = Jlim/(4 gs ca)``,
    ``b = q - 1 + 2g`` (from ``J(chi) = Jlim``); the conjugate form avoids the
    cancellation at small gs (large q).
    """
    q = env.jlim / jnp.maximum(4.0 * gs * env.ca_umol, 1e-30)
    b = q - 1.0 + 2.0 * env.g
    c = env.g * (2.0 + q)
    root = jnp.sqrt(b * b + 4.0 * c)
    return jnp.where(b >= 0.0, 2.0 * c / (b + root), 0.5 * (root - b))


def _chi_opt(gs, inv_f, dpsi, env: _Env, cfg: PHydroConfig) -> jax.Array:
    """Profit-maximizing chi at a fixed supply state (bisection on dF/dchi).

    ``F`` is concave in chi on ``(chi_min, 1)``: A is linear and J (hence
    Jmax) is convex-decreasing in chi.  ``dF/dchi > 0`` up to chi = 1 means
    capacity never pays for itself — the bracket collapses to chi = 1 (A = 0).
    """
    def step(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        rising = _d(lambda c: _profit_chi(c, gs, inv_f, dpsi, env, cfg), mid) > 0.0
        return jnp.where(rising, mid, lo), jnp.where(rising, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0, _N_CHI_ITER, step, (_chi_min(gs, env), jnp.ones_like(gs)))
    return 0.5 * (lo + hi)


def _solve(env: _Env, cfg: PHydroConfig):
    """Nested-bisection optimum ``(chi, dpsi)`` (no gradients flow through it).

    Outer bisection on the envelope derivative ``dF/ddpsi`` at the inner chi
    optimum.  A still-positive derivative at the top of the bracket leaves
    dpsi at ``_DPSI_MAX_MPA`` (reported as not converged).
    """
    def step(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        gs, inv_f = _supply_side(mid, env, cfg)
        chi = _chi_opt(gs, inv_f, mid, env, cfg)
        rising = _d(lambda d: _profit(chi, d, env, cfg), mid) > 0.0
        return jnp.where(rising, mid, lo), jnp.where(rising, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0, _N_DPSI_ITER, step,
        (jnp.zeros_like(env.psi_s), jnp.full_like(env.psi_s, _DPSI_MAX_MPA)))
    dpsi = 0.5 * (lo + hi)
    gs, inv_f = _supply_side(dpsi, env, cfg)
    return _chi_opt(gs, inv_f, dpsi, env, cfg), dpsi


def _implicit_refine(chi0, dpsi0, env: _Env, cfg: PHydroConfig):
    """One Newton step on the stationarity conditions, for IFT gradients.

    ``x = x0 - H^-1 grad F(x0; theta)`` with ``x0`` and the Hessian ``H``
    gradient-stopped: the forward value moves only by the (tiny) bisection
    residual, and the backward pass is ``dx/dtheta = -H^-1 d(grad F)/dtheta``
    — the implicit function theorem (same identity as the CLM-ML
    stomatal-efficiency root, ``_bisect_gs_ift``).  Skipped (zero gradient)
    where the plant is shut down, the search hit its bracket, or the Hessian
    is not a well-conditioned maximum.
    """
    def grad_f(chi, dpsi, env_, cfg_):
        return (_d(lambda c: _profit(c, dpsi, env_, cfg_), chi),
                _d(lambda d: _profit(chi, d, env_, cfg_), dpsi))

    env_sg = jax.tree_util.tree_map(jax.lax.stop_gradient, env)
    cfg_sg = jax.tree_util.tree_map(jax.lax.stop_gradient, cfg)
    h11 = _d(lambda c: grad_f(c, dpsi0, env_sg, cfg_sg)[0], chi0)
    h12 = _d(lambda d: grad_f(chi0, d, env_sg, cfg_sg)[0], dpsi0)
    h22 = _d(lambda d: grad_f(chi0, d, env_sg, cfg_sg)[1], dpsi0)
    det = h11 * h22 - h12 * h12

    gs0, _ = _supply_side(dpsi0, env_sg, cfg_sg)
    a0, _ = _electron_transport(chi0, gs0, env_sg)
    converged = dpsi0 < _DPSI_MAX_MPA * (1.0 - 2.0 ** (2 - _N_DPSI_ITER))
    ok = (converged & (a0 > _A_ACTIVE_UMOL) & (h11 < 0.0)
          & (det > _HESS_DET_RTOL * jnp.abs(h11 * h22)))

    g1, g2 = grad_f(chi0, dpsi0, env, cfg)
    g1 = jnp.where(ok, g1, 0.0)
    g2 = jnp.where(ok, g2, 0.0)
    det = jnp.where(ok, det, 1.0)
    chi = chi0 - (h22 * g1 - h12 * g2) / det
    dpsi = dpsi0 - (h11 * g2 - h12 * g1) / det
    return chi, dpsi, converged


def phydro_optimum(
    acclim: PModelAcclimState | None,
    supply: PhydroSupply,
    cfg: PHydroConfig,
    pcfg: PModelConfig,
) -> PhydroCapacities:
    """Profit-maximizing acclimated capacities on the vulnerable-xylem supply.

    Maximizes ``A - alpha*Jmax - gamma*dpsi^2`` over ``(chi, dpsi)`` (module
    docstring) and returns the water-stressed capacities the host kernel
    should run with: Vcmax from coordination at the optimum's J, and the
    effective ``f(psi_leaf) * Jmax``, both normalised to 25 degC with the
    host's own Kattge & Knorr responses at the growth temperature.
    """
    if acclim is None:
        raise ValueError(
            "P-hydro capacities requested but no acclimation state was "
            "supplied (pmodel_acclim=None): this land lane does not carry "
            "PModelAcclimState. Initialise it (init_pmodel_acclim) or select "
            "transpiration_stress='beta_theta'."
        )
    tg_k = acclim.t_mean_K
    # Kinetics at the acclimated means (same conversions as the least-cost
    # lane; its beta-based chi is unused here).
    _chi_lc, _xi, _ci_lc, gamma_star, big_k = optimal_chi(
        tg_k, acclim.vpd_mean_pa, acclim.co2_mean_ppm, acclim.ps_ema, pcfg)
    del _chi_lc, _xi, _ci_lc
    ca_pa = acclim.co2_mean_ppm * acclim.ps_ema / _UMOL_PER_MOL
    env = _Env(
        psi_s=supply.psi_s_mpa,
        rsoil=supply.rsoil_leaf,
        d_frac=jnp.maximum(acclim.vpd_mean_pa, pcfg.vpd_min_pa) / acclim.ps_ema,
        ca_umol=acclim.co2_mean_ppm,
        g=gamma_star / ca_pa,
        jlim=4.0 * _phi0(tg_k, pcfg) * acclim.iabs_mean,
    )

    chi0, dpsi0 = jax.lax.stop_gradient(_solve(
        jax.tree_util.tree_map(jax.lax.stop_gradient, env),
        jax.tree_util.tree_map(jax.lax.stop_gradient, cfg)))
    chi, dpsi, converged = _implicit_refine(chi0, dpsi0, env, cfg)

    gs, _inv_f = _supply_side(dpsi, env, cfg)
    _a, j = _electron_transport(chi, gs, env)
    # Coordination (A_c = A_j): Vcmax = (J/4) * (ci + K)/(ci + 2 Gamma*).
    mj_over_mc = (chi + big_k / ca_pa) / (chi + 2.0 * env.g)
    tg_c = tg_k - constants.T_freeze
    f_v = vcmax_temperature_response(tg_k, tg_c)
    f_j = jmax_temperature_response(tg_k, tg_c)
    vcmax25 = 0.25 * j * mj_over_mc / f_v
    # Effective Jmax / Vcmax; J cancels, so the ratio stays finite as J -> 0.
    rjv25 = (4.0 * _jmax_per_j(j, env) / mj_over_mc) * (f_v / f_j)

    return PhydroCapacities(
        vcmax25_leaf=vcmax25, rjv25=rjv25, chi=chi, dpsi_mpa=dpsi,
        psi_leaf_mpa=supply.psi_s_mpa - dpsi, gs_mol=gs, converged=converged)


def slope_for_model(
    stomatal_model: str,
    chi: jax.Array,
    acclim: PModelAcclimState,
    gamma_star_pa: jax.Array,
    d0_leuning_kpa: float | None = None,
) -> jax.Array:
    """Stomatal SLOPE for the active model reproducing the optimum's gs.

    Each mapping makes its model's conductance equal ``1.6*A/(ca*(1-chi))``
    at the acclimated mean state (valid for g0 ~ 0; GLM-reviewed algebra):
    medlyn ``g1 = sqrt(D_kPa)*chi/(1-chi)``; ball_berry
    ``m = 1.6/(RH*(1-chi))``; leuning ``a1 = 1.6*(ca-Gamma*)*(1+D/D0)/
    (ca*(1-chi))``.
    """
    from legoesm.thermo import saturation_vapor_pressure

    one_minus = jnp.maximum(1.0 - chi, 0.02)  # coeff-ok: 1-chi floor of the slope mapping (numerics)
    if stomatal_model == "medlyn":
        d_kpa = jnp.maximum(acclim.vpd_mean_pa, 1.0) / 1000.0
        return jnp.sqrt(d_kpa) * chi / one_minus
    if stomatal_model == "ball_berry":
        es = saturation_vapor_pressure(acclim.t_mean_K)
        rh = jnp.clip(1.0 - acclim.vpd_mean_pa / jnp.maximum(es, 1.0),
                      0.05, 1.0)  # coeff-ok: RH floor of the Ball-Berry slope mapping (numerics)
        return DIFFUSIVITY_RATIO_H2O_CO2 / (rh * one_minus)
    if stomatal_model == "leuning":
        if d0_leuning_kpa is None:
            raise ValueError(
                "slope_for_model('leuning') needs d0_leuning_kpa (from the "
                "active scheme's config); got None")
        frac = 1e-6
        ca_pa = acclim.co2_mean_ppm * frac * acclim.ps_ema
        d_kpa = acclim.vpd_mean_pa / 1000.0
        return (DIFFUSIVITY_RATIO_H2O_CO2 * (ca_pa - gamma_star_pa)
                * (1.0 + d_kpa / d0_leuning_kpa)
                / (jnp.maximum(ca_pa, 1e-3) * one_minus))  # coeff-ok: div-by-zero guard on ca [Pa]
    raise ValueError(
        f"unknown stomatal_model {stomatal_model!r} for the P-hydro slope "
        "mapping; must be one of ('medlyn', 'ball_berry', 'leuning')")
