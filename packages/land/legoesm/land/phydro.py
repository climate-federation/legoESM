"""P-hydro: profit-maximizing stomatal/capacity optimality on plant hydraulics.

Joshi et al. (2022, Nature Plants) OPTIMIZATION STRATEGY — maximize
assimilation net of hydraulic risk over the soil-to-leaf water-potential
draw-down — on the SPA/CLM-ML PLANT-HYDRAULICS supply (user decision: one
hydraulics formulation shared across the big-leaf, two-leaf and CLM-ML
lanes): Bonan et al. (2014, GMD 7, 2193-2222) eqs. A23-A28 soil-to-root
resistance from root geometry + soil hydraulic conductivity, a constant
plant (xylem) conductance ``gplant``, and the SPA minimum leaf water
potential floor.  Like the least-cost P model (``land/p_model.py``), it
supplies ACCLIMATED PARAMETERS (leaf-top Vcmax25, rjv25, and a stomatal
slope for the active conductance model) — the host kernels compute fluxes.

Profit, at the P-model acclimation means and the CURRENT root-zone soil
water potential (instantaneous, like the empirical stress it replaces):

    Profit(dpsi) = A(gs(dpsi)) - gamma * dpsi^2         [Joshi eq. 6 shape]
    gs(dpsi)     = lsc * dpsi / (1.6 * D)               (supply = demand)
    lsc          = 1 / (rsoil_leaf + 1/gplant)          (SPA, per leaf area)

with A the Jmax-cost-discounted coordination rate ``phi0*Iabs*mj'(chi)`` and
``chi`` closed by supply-demand (A = gs*ca*(1-chi)).  The one-dimensional
optimum over ``dpsi`` is found on a smooth sigmoid map onto the admissible
interval ``(0, psi_s - minlwp)`` by unrolled damped Newton (differentiable).

DELIBERATE DEPARTURES (each reviewed; see the PR trail):

1. **SPA supply, not Joshi's Weibull vulnerability.**  User decision (one
   shared hydraulics).  The conductance-loss feedback of a vulnerability
   curve is absent; the hydraulic risk enters through the quadratic cost
   and the (smoothly capped) minlwp floor.  GLM review: profit remains
   strictly concave -> unique interior optimum.
2. **Jmax cost via the shared Wang-2017 c\\* closure.**  Joshi's explicit
   ``alpha * Jmax`` term is eliminated by its own first-order condition; in
   Stocker's normalisation that elimination IS the ``c\\* = 0.41`` limitation
   already inside ``mj'`` (``p_model.capacities_from_chi``), so no separate
   ``alpha`` knob exists here (alpha ~ 0.08-0.12 "treatable as a constant",
   Joshi Table 1).
3. **Slope, not gs.**  The optimum's chi maps to the ACTIVE stomatal
   model's slope at the acclimated means (valid for g0 ~ 0):
   medlyn ``g1 = sqrt(D_kPa)*chi/(1-chi)``; ball_berry
   ``m = 1.6/(RH*(1-chi))``; leuning ``a1 = 1.6*(ca-Gamma*)*(1+D/D0)/
   (ca*(1-chi))`` — each makes its model reproduce ``gs = 1.6*A/(ca*(1-chi))``
   at the mean state.

References: Joshi et al. (2022) Nat. Plants 8, 1304-1316; Bonan et al.
(2014) GMD 7, 2193-2222 (SPA hydraulics); Williams et al. (1996) (SPA).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.leaf_biophysics import DIFFUSIVITY_RATIO_H2O_CO2
from legoesm.land.p_model import (
    PModelAcclimState,
    PModelConfig,
    capacities_from_chi,
    optimal_chi,
    _phi0,
    _smooth_floor,
)

__physics_contract__ = {
    "summary": (
        "P-hydro (Joshi et al. 2022 profit strategy on the SPA/Bonan-2014 "
        "plant-hydraulics supply): maximize phi0*Iabs*mj'(chi) - "
        "gamma*dpsi^2 over the soil-to-leaf draw-down dpsi, with gs = "
        "lsc*dpsi/(1.6 D) and chi closed by supply-demand at the P-model "
        "acclimation means and the instantaneous extraction-weighted soil "
        "water potential. Supplies leaf-top Vcmax25, rjv25 and a stomatal "
        "slope for the active model; replaces the empirical theta stress."
    ),
    "inputs": {
        "psi_soil": "m", "theta_soil": "1", "dz": "m", "root_frac": "1",
        "lai": "1", "acclim means": "see land.p_model",
    },
    "outputs": {
        "vcmax25_leaf": "umol/m^2/s", "rjv25": "1", "chi": "1",
        "dpsi_mpa": "MPa", "lsc": "mol/m^2/s/MPa", "psi_s_mpa": "MPa",
    },
    "sign_convention": (
        "psi <= 0 (matric, MPa after conversion); dpsi >= 0 is the "
        "soil-minus-leaf draw-down; capacities >= 0 and -> 0 smoothly as "
        "psi_s -> minlwp."
    ),
    "conserves": [],  # parameter prediction, not a flux
    "differentiable": True,
    "reference": (
        "Joshi et al. (2022) Nature Plants 8 1304-1316; Bonan et al. (2014) "
        "GMD 7 2193-2222 eqs. A23-A28; Williams et al. (1996) SPA."
    ),
    "idealized_test": (
        "tests/land/unit/test_phydro.py: wet-soil optimum has dpsi* > 0 and "
        "finite capacities; capacities decrease monotonically as psi_s "
        "dries toward minlwp and -> 0 smoothly; gamma up -> dpsi* down; "
        "stationarity |dProfit/dz| ~ 0 at the returned optimum and profit "
        "beats +/- perturbations; gradients finite everywhere."
    ),
}

__param_spec__ = {
    "PHydroConfig": {
        "scheme_key": "land.phydro",
        "excluded": {
            "dpsi_cap_width_mpa": "numerics: softplus width of the dpsi cap",
        },
        "params": {
            "gplant_mol": {
                "units": "mol m-2 s-1 MPa-1", "bounds": (5e-4, 2e-2),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "hydraulics",
                "reference": "SPA / CLM-ML gplant table (4 mmol m-2 s-1 MPa-1)",
                "shape": None,
            },
            "gamma_cost": {
                "units": "umol m-2 s-1 MPa-2", "bounds": (0.1, 10.0),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "hydraulics",
                "reference": "Joshi et al. 2022 Table 1 (0.28-5 by plant type)",
                "shape": None,
            },
            "minlwp_mpa": {
                "units": "MPa", "bounds": (-5.0, -0.5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "hydraulics",
                "reference": "SPA minimum leaf water potential (-2 MPa)",
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
_N_CHI_ITER = 15    # inner supply-demand chi fixed point (damped)
_N_DPSI_ITER = 25   # outer damped Newton on the sigmoid-mapped dpsi
_RBD_FLOOR = 1e-10  # coeff-ok: Bonan A23 root-biomass-density floor [g/m3]


class PHydroConfig(NamedTuple):
    """P-hydro tunables (SPA supply + Joshi profit strategy)."""

    gplant_mol: float = 4.0e-3     # [mol/m2 leaf/s/MPa] xylem conductance
    gamma_cost: float = 1.0        # [umol/m2/s/MPa^2] hydraulic risk cost
    minlwp_mpa: float = -2.0       # [MPa] SPA minimum leaf water potential
    root_biomass_gm2: float = 500.0  # [g/m2] fine-root biomass
    # --- numerics ---
    dpsi_cap_width_mpa: float = 0.05  # softplus width of the dpsi ceiling


class PhydroSupply(NamedTuple):
    """Soil-to-leaf supply state, all ``(ncol,)``."""

    lsc_mol: jax.Array   # [mol/m2 leaf/s/MPa] whole-plant leaf-specific conductance
    psi_s_mpa: jax.Array  # [MPa] extraction-weighted soil water potential


class PhydroCapacities(NamedTuple):
    """Acclimated parameter predictions from the profit optimum."""

    vcmax25_leaf: jax.Array  # [umol/m^2/s]
    rjv25: jax.Array         # [-]
    chi: jax.Array           # [-] optimal ci/ca (diagnostic + slope source)
    dpsi_mpa: jax.Array      # [MPa] optimal draw-down (diagnostic)


def soil_root_supply(
    psi_soil_m: jax.Array,
    theta_soil: jax.Array,
    dz_m: jax.Array,
    root_frac: jax.Array,
    lai: jax.Array,
    hydraulics_cfg,
    cfg: PHydroConfig,
) -> PhydroSupply:
    """SPA soil-to-leaf supply (Bonan 2014 eqs. A23-A28), single big leaf.

    Layered soil-to-root resistances from root geometry + the Richards
    lane's own hydraulic conductivity; conductances sum in parallel across
    layers; the extraction-weighted soil potential uses the max-extractable
    weights ``(smp - minlwp)/soilr`` (A26-A28).  ``rsoil`` converts to a
    per-LEAF-area resistance with LAI (A25) and adds the xylem resistance
    ``1/gplant`` in series.

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

    smp_mpa = psi_soil_m * mpa_per_m  # matric potential [MPa], <= 0

    dz = jnp.broadcast_to(dz_m, psi_soil_m.shape)
    rbd = jnp.maximum(cfg.root_biomass_gm2 * root_frac / dz, _RBD_FLOOR)
    rld = rbd / (_ROOT_DENSITY_GM3 * jnp.pi * _ROOT_RADIUS_M ** 2)
    root_dist = jnp.sqrt(1.0 / (rld * jnp.pi))
    # A23 (radial soil-to-root) + A24 (root tissue).  _ROOT_RESIST is in the
    # SPA mmol convention; 1e-3 converts the resulting resistance to the
    # mol convention used throughout this module.
    soilr1 = jnp.log(root_dist / _ROOT_RADIUS_M) / (
        2.0 * jnp.pi * rld * dz * jnp.maximum(hk_mol, 1e-30))
    soilr2 = (_ROOT_RESIST * 1e3) / (rbd * dz)
    soilr = soilr1 + soilr2  # [MPa m2 ground s mol-1]

    g_layers = 1.0 / soilr
    g_soil_ground = jnp.sum(g_layers, axis=-1)  # [mol/m2 ground/s/MPa]

    # A26-A28: extraction-weighted soil potential (weights = max extractable
    # flow per layer, floored at 0; dry columns collapse to minlwp).
    w = jnp.maximum((smp_mpa - cfg.minlwp_mpa) / soilr, 0.0)
    w_tot = jnp.sum(w, axis=-1)
    psi_s = jnp.where(
        w_tot > 0.0,
        jnp.sum(smp_mpa * w, axis=-1) / jnp.maximum(w_tot, 1e-30),
        jnp.full_like(w_tot, cfg.minlwp_mpa))

    # A25: per-leaf-area soil resistance, + xylem in series.
    rsoil_leaf = jnp.maximum(lai, 1e-3) / jnp.maximum(g_soil_ground, 1e-30)
    lsc = 1.0 / (rsoil_leaf + 1.0 / cfg.gplant_mol)
    return PhydroSupply(lsc_mol=lsc, psi_s_mpa=psi_s)


def _chi_supply_demand(
    gs_mol: jax.Array,
    tg_k: jax.Array,
    ca_pa: jax.Array,
    gamma_star_pa: jax.Array,
    phi0: jax.Array,
    iabs: jax.Array,
    ps_pa: jax.Array,
    pcfg: PModelConfig,
) -> jax.Array:
    """chi closing A_demand = A_coordination for a GIVEN gs (damped fixed point).

    Demand: A = gs * ca_mol_frac * (1 - chi) * 1e6 [umol m-2 s-1].
    Coordination: A = phi0 * Iabs * mj'(chi) (Wang-limited light rate).
    """
    ca_frac = ca_pa / ps_pa  # mol/mol
    from legoesm.land.p_model import C_STAR

    chi = jnp.full_like(gs_mol, 0.7)
    for _ in range(_N_CHI_ITER):
        ci = chi * ca_pa
        mj = (ci - gamma_star_pa) / (ci + 2.0 * gamma_star_pa)
        mj_safe = _smooth_floor(
            mj, C_STAR + pcfg.mj_floor_eps, pcfg.mj_floor_width)
        k = (C_STAR / mj_safe) ** (2.0 / 3.0)
        a_coord = phi0 * iabs * mj_safe * jnp.sqrt(1.0 - k)  # [umol]
        denom = jnp.maximum(gs_mol * ca_frac * 1e6, 1e-9)
        chi_new = jnp.clip(1.0 - a_coord / denom, 0.05, 0.98)
        chi = 0.5 * (chi + chi_new)  # damped
    return chi


def phydro_optimum(
    acclim: PModelAcclimState | None,
    supply: PhydroSupply,
    cfg: PHydroConfig,
    pcfg: PModelConfig,
) -> PhydroCapacities:
    """Profit-maximizing acclimated capacities on the SPA supply.

    Maximizes ``phi0*Iabs*mj'(chi(dpsi)) - gamma*dpsi^2`` over the smoothly
    capped draw-down ``dpsi in (0, psi_s - minlwp)`` (sigmoid-mapped, so the
    corner at the cap is smooth), by unrolled damped Newton on the map
    variable — fully differentiable.  Capacities at the optimum come from
    the SHARED chi->capacity algebra (``p_model.capacities_from_chi``).
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
    # lane; the beta term of optimal_chi is unused here — chi comes from the
    # supply-demand closure — but the kinetic byproducts are shared).
    _chi_lc, _xi, _ci_lc, gamma_star, big_k = optimal_chi(
        tg_k, acclim.vpd_mean_pa, acclim.co2_mean_ppm, acclim.ps_ema, pcfg)
    del _chi_lc, _xi, _ci_lc
    frac = 1e-6  # exact conversion umol/mol -> mol/mol
    ca_pa = acclim.co2_mean_ppm * frac * acclim.ps_ema
    d_frac = jnp.maximum(acclim.vpd_mean_pa, pcfg.vpd_min_pa) / acclim.ps_ema
    phi0 = _phi0(tg_k, pcfg)
    iabs = acclim.iabs_mean

    # Smoothly-capped admissible draw-down (0, cap]; softplus keeps the cap
    # positive-and-smooth even as psi_s -> minlwp (GLM review: no kink).
    cap = cfg.dpsi_cap_width_mpa * jax.nn.softplus(
        (supply.psi_s_mpa - cfg.minlwp_mpa) / cfg.dpsi_cap_width_mpa)

    def profit(z):
        dpsi = cap * jax.nn.sigmoid(z)
        gs = supply.lsc_mol * dpsi / (DIFFUSIVITY_RATIO_H2O_CO2 * d_frac)
        chi = _chi_supply_demand(
            gs, tg_k, ca_pa, gamma_star, phi0, iabs, acclim.ps_ema, pcfg)
        ci = chi * ca_pa
        from legoesm.land.p_model import C_STAR
        mj = (ci - gamma_star) / (ci + 2.0 * gamma_star)
        mj_safe = _smooth_floor(
            mj, C_STAR + pcfg.mj_floor_eps, pcfg.mj_floor_width)
        k = (C_STAR / mj_safe) ** (2.0 / 3.0)
        a = phi0 * iabs * mj_safe * jnp.sqrt(1.0 - k)
        return jnp.sum(a - cfg.gamma_cost * dpsi * dpsi), (dpsi, chi)

    grad_fn = jax.grad(lambda z: profit(z)[0])
    hess_diag = jax.grad(lambda z: jnp.sum(grad_fn(z)))

    z = jnp.zeros_like(supply.psi_s_mpa)
    for _ in range(_N_DPSI_ITER):
        g = grad_fn(z)
        h = hess_diag(z)
        # Concave interior: h < 0. Damped Newton with a bounded step; the
        # tanh bound keeps early iterations stable far from the optimum.
        step = g / (jnp.abs(h) + 1e-6)
        z = z + jnp.tanh(step)

    _, (dpsi_star, chi_star) = profit(z)
    ci_star = chi_star * ca_pa
    vcmax25, rjv25 = capacities_from_chi(
        chi=chi_star, ci_pa=ci_star, gamma_star_pa=gamma_star,
        big_k_pa=big_k, tg_k=tg_k, iabs=iabs, phi0=phi0, cfg=pcfg)
    return PhydroCapacities(
        vcmax25_leaf=vcmax25, rjv25=rjv25, chi=chi_star, dpsi_mpa=dpsi_star)


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

    one_minus = jnp.maximum(1.0 - chi, 0.02)
    if stomatal_model == "medlyn":
        d_kpa = jnp.maximum(acclim.vpd_mean_pa, 1.0) / 1000.0
        return jnp.sqrt(d_kpa) * chi / one_minus
    if stomatal_model == "ball_berry":
        es = saturation_vapor_pressure(acclim.t_mean_K)
        rh = jnp.clip(1.0 - acclim.vpd_mean_pa / jnp.maximum(es, 1.0),
                      0.05, 1.0)
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
                / (jnp.maximum(ca_pa, 1e-3) * one_minus))
    raise ValueError(
        f"unknown stomatal_model {stomatal_model!r} for the P-hydro slope "
        "mapping; must be one of ('medlyn', 'ball_berry', 'leuning')")
