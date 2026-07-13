"""Multi-layer snowpack column (energy- and mass-conserving).

The CLM-faithful multi-layer snow *core* (Oleson et al. 2013, CLM5 Tech Note §6),
replacing the single-node bulk SWE budget (``snow_budget.update_snow``) for the
cold-bias / snow-tower fix.  Selected via ``MultiLayerLandConfig.snow_scheme ==
"multilayer"`` (default ``"single"`` keeps the bulk budget, bit-for-bit).  Wiring
plan + coupling design: ``docs/land/phase2b_snow_thermal_plan.md``.  Integration
stages: (1) DONE — module promoted from ``_future`` + ``snow_scheme`` config gate;
(2) prognostic :class:`SnowColumnState` on ``MultiLayerLandState``; (3) couple the
surface flux ``Q_top`` in and the snow<->soil conductive flux ``G_bottom`` to the
top soil layer, route ``drainage`` to the soil, expose pack-top T as skin T;
(4) snow albedo + explicit ``h2osno_max`` cap; (5) enable + validate.

Physics (fixed ``n_layers`` equal-SWE-mass layers; the total pack SWE is
remapped to ``n_layers`` equal-mass layers each step, a conservative 1-D
cumulative-mass remap so total ice, liquid, and enthalpy are preserved to
machine precision while shapes stay static and differentiable):

* accumulation — snowfall adds fresh-density ice at ``min(T_air, T_freeze)``
  (added as enthalpy so the top layer stays conservative even if it holds
  liquid);
* compaction — density relaxes toward ``rho_snow_max`` (destructive
  metamorphism + overburden, Anderson 1976 / CLM), setting layer thickness
  ``dz = swe_water_equiv / density`` and the snow conductivity
  ``k = k_snow_ref * (rho / rho_snow_ref)^2`` (Sturm 1997 style);
* thermal — implicit (backward-Euler) heat diffusion through the layers with a
  prescribed surface flux ``Q_top`` and a base flux ``G_bottom`` (to the soil),
  using the full sensible heat capacity ``ice*c_ice + liq*c_liq``;
* phase change (ENTHALPY method) — each layer's water mass ``w`` and enthalpy
  ``H`` (relative to all-ice-at-``T_freeze``) determine the equilibrium
  ice/liquid/T: ``H<=0`` all ice below freezing, ``H>=w*L_f`` all liquid above
  freezing, else a two-phase mix at ``T_freeze`` with ``liq=H/L_f``.  This
  captures melt AND refreeze in one enthalpy-conserving step;
* percolation — liquid beyond the irreducible holding capacity moves down (as
  water at ``T_freeze``, carrying its latent enthalpy ``L_f``), re-equilibrating
  each receiving layer; the liquid leaving the base is ``drainage``.

Conserved (see ``tests/land/test_snow_column.py``):
* WATER: ``sum(swe_ice + swe_liq)`` changes only by ``snowfall*dt - drainage``.
* ENERGY: column enthalpy ``H = sum[(ice*c_ice + liq*c_liq)*(T-T_freeze) +
  liq*L_f]`` changes only by ``(Q_top - G_bottom)*dt`` (boundary heat) plus the
  fresh-snow enthalpy in, minus the drainage enthalpy out (``drainage_heat``,
  the water's full enthalpy ``drainage*(c_liq*(T-T_freeze) + L_f)``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.timestepping.tridiagonal import thomas_solve

# --- reference snow material properties (CICE / Sturm 1997) ---
_RHO_SNOW_REF = constants.rho_snow      # kg/m^3, conductivity reference density
_K_SNOW_REF = constants.k_snow          # W/m/K at the reference density
_C_ICE = constants.c_pi                 # J/kg/K (snow grains ~ ice)
_C_LIQ = constants.c_pw                 # J/kg/K liquid water
_TF = constants.T_freeze                # freezing point [K]
_LF = constants.L_f                     # latent heat of fusion [J/kg]
_EPS = 1e-12                            # generic small floor
_DZ_HALF_MIN = 1e-4                     # [m] min interface distance (empty-pack conductance bound)


class SnowColumnConfig(NamedTuple):
    """Multi-layer snowpack configuration."""
    n_layers: int = 5                    # fixed number of snow layers
    rho_snow_fresh: float = 100.0        # fresh-snow density [kg/m^3]
    rho_snow_max: float = 450.0          # max densified snow density [kg/m^3]
    compaction_timescale_s: float = 8.64e5  # density-relaxation e-folding time [s] (~10 d)
    irreducible_liq_frac: float = 0.05   # liquid held per unit ice mass [-]
    k_conductivity_exponent: float = 2.0  # k ~ (rho/rho_ref)^exp (Sturm 1997)
    min_pack_swe: float = 1e-8           # [kg/m^2] below which the pack is empty


__param_spec__ = {
    "SnowColumnConfig": {
        "scheme_key": "land.snow_column",
        "excluded": {
            "compaction_timescale_s": "numerics: density-relaxation e-folding time",
            "k_conductivity_exponent": "material: Sturm (1997) conductivity exponent",
            "min_pack_swe": "numerics: empty-pack floor",
        },
        "params": {
            "rho_snow_fresh": {
                "units": "kg/m^3", "bounds": (50.0, 200.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "material",
                "reference": "fresh-snow density (Anderson 1976 / CLM5)", "shape": None,
            },
            "rho_snow_max": {
                "units": "kg/m^3", "bounds": (300.0, 550.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "material",
                "reference": "densified-snow ceiling (CLM5)", "shape": None,
            },
            "irreducible_liq_frac": {
                "units": "1", "bounds": (0.0, 0.15), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "irreducible liquid water holding capacity (CLM5)", "shape": None,
            },
        },
    },
}


class SnowColumnState(NamedTuple):
    """Prognostic multi-layer snow state, all shape ``(..., n_layers)``.

    ``swe_ice`` / ``swe_liq`` are the frozen / liquid water masses per layer
    [kg/m^2] (SWE, i.e. water-equivalent); ``T`` is layer temperature [K];
    ``density`` is the snow (ice-matrix) density [kg/m^3] (for thickness +
    conductivity).  Layer 0 is the TOP (surface).
    """
    swe_ice: jnp.ndarray
    swe_liq: jnp.ndarray
    T: jnp.ndarray
    density: jnp.ndarray


def initial_snow_state(shape, config: SnowColumnConfig = SnowColumnConfig()):
    """An empty snowpack (all masses zero) of the given leading ``shape``."""
    n = config.n_layers
    z = jnp.zeros(shape + (n,))
    return SnowColumnState(
        swe_ice=z, swe_liq=z,
        T=jnp.full(shape + (n,), _TF),
        density=jnp.full(shape + (n,), config.rho_snow_fresh),
    )


def _sensible_hc(swe_ice, swe_liq):
    """Sensible volumetric heat capacity per layer [J/m^2/K]."""
    return swe_ice * _C_ICE + swe_liq * _C_LIQ


def _enthalpy(swe_ice, swe_liq, T):
    """Per-layer enthalpy relative to all-ice-at-T_freeze [J/m^2]:
    ``H = (ice*c_ice + liq*c_liq)*(T - T_freeze) + liq*L_f``."""
    return _sensible_hc(swe_ice, swe_liq) * (T - _TF) + swe_liq * _LF


def _phase_from_w_H(w, H):
    """Equilibrium (ice, liq, T) from total water ``w`` [kg/m^2] and enthalpy
    ``H`` [J/m^2] (relative to all-ice-at-T_freeze).  Exactly conserves w and H.

    * ``H <= 0``   -> all ice, ``T = T_freeze + H/(w*c_ice) <= T_freeze``.
    * ``H >= w*Lf`` -> all liquid, ``T = T_freeze + (H-w*Lf)/(w*c_liq) >= T_freeze``.
    * else -> two-phase at ``T_freeze`` with ``liq = H/Lf``.
    """
    all_liquid = H >= w * _LF
    all_ice = H <= 0.0
    liq_mix = jnp.clip(H / _LF, 0.0, w)
    liq = jnp.where(all_liquid, w, jnp.where(all_ice, 0.0, liq_mix))
    ice = w - liq
    C = jnp.maximum(_sensible_hc(ice, liq), _EPS)
    T = _TF + (H - liq * _LF) / C
    # Empty layers (w ~ 0): no sensible mass -> pin to T_freeze.
    T = jnp.where(w > _EPS, T, _TF)
    return ice, liq, T


def _remap_equal_mass(swe_ice, swe_liq, T, density):
    """Conservative 1-D remap of the pack onto ``n`` EQUAL-SWE-mass layers
    (cumulative-mass coordinate).  Preserves total ice, total liquid, total
    enthalpy ``sum(H)`` and total ``mass*density`` exactly.  All arrays
    ``(..., n)``.
    """
    n = swe_ice.shape[-1]
    mass = swe_ice + swe_liq
    total = jnp.sum(mass, axis=-1, keepdims=True)
    enth = _enthalpy(swe_ice, swe_liq, T)                    # extensive, conserved
    dmass = density * mass

    cum = jnp.cumsum(mass, axis=-1)
    lo_in = cum - mass
    target = total / n
    idx = jnp.arange(n)
    out_lo = idx * target
    out_hi = (idx + 1) * target
    lo = jnp.maximum(out_lo[..., :, None], lo_in[..., None, :])
    hi = jnp.minimum(out_hi[..., :, None], cum[..., None, :])
    overlap = jnp.clip(hi - lo, 0.0, None)                   # (..., n_out, n_in)
    # True safe divide: contribute nothing from a zero-mass input layer.
    frac = jnp.where(mass[..., None, :] > _EPS,
                     overlap / jnp.where(mass[..., None, :] > _EPS,
                                         mass[..., None, :], 1.0),
                     0.0)

    def _g(x):
        return jnp.sum(frac * x[..., None, :], axis=-1)

    new_ice, new_liq = _g(swe_ice), _g(swe_liq)
    new_enth, new_dmass = _g(enth), _g(dmass)
    new_mass = new_ice + new_liq
    empty = new_mass <= _EPS
    # Re-derive the equilibrium phase from the conserved (water, enthalpy).
    # Total WATER and total ENTHALPY are conserved by the remap; the ice/liquid
    # split re-equilibrates (physical: mixing a cold-ice layer with a
    # warm-liquid layer partially refreezes), which _phase_from_w_H enforces.
    out_ice, out_liq, out_T = _phase_from_w_H(new_mass, new_enth)
    new_density = jnp.where(empty,
                            SnowColumnConfig().rho_snow_fresh,
                            new_dmass / jnp.maximum(new_mass, _EPS))
    return out_ice, out_liq, out_T, new_density


def _thickness_and_conductivity(swe_ice, swe_liq, density, config):
    """Layer thickness [m] and Sturm-1997 conductivity [W/m/K]."""
    mass = swe_ice + swe_liq
    dz = mass / jnp.maximum(density, _EPS)
    k = _K_SNOW_REF * (density / _RHO_SNOW_REF) ** config.k_conductivity_exponent
    return dz, k


def step_snow_column(
    state: SnowColumnState,
    precip_snow: jnp.ndarray,
    T_air: jnp.ndarray,
    Q_top: jnp.ndarray,
    G_bottom: jnp.ndarray,
    dt: float,
    config: SnowColumnConfig = SnowColumnConfig(),
):
    """Advance the multi-layer snowpack one step.

    Parameters
    ----------
    state : SnowColumnState
    precip_snow : array (...,)   snowfall rate [kg/m^2/s]
    T_air : array (...,)         air temperature [K] (fresh-snow temperature)
    Q_top : array (...,)         net surface energy flux INTO the pack top [W/m^2]
    G_bottom : array (...,)      conductive flux from the pack base INTO the
                                 soil [W/m^2, positive downward]
    dt : float
    config : SnowColumnConfig

    Returns
    -------
    new_state : SnowColumnState
    drainage : array (...,)        liquid water leaving the pack base [kg/m^2]
                                   this step (route to soil infiltration).
    drainage_heat : array (...,)   enthalpy of that drained water relative to
                                   T_freeze [J/m^2] (= drainage*(c_liq*(T-Tf) +
                                   L_f); route to the soil so energy closes).
    """
    swe_ice, swe_liq, T, density = state
    n = swe_ice.shape[-1]                                     # static from shape (JIT-safe)

    # --- 1. Accumulation: fresh snow (ice at min(T_air,T_freeze)) into TOP ---
    snowfall = precip_snow * dt                              # [kg/m^2]
    T_fresh = jnp.minimum(T_air, _TF)
    fresh_enth = snowfall * _C_ICE * (T_fresh - _TF)         # enthalpy added [J/m^2]
    # Add mass + enthalpy to the top layer, then re-derive (ice,liq,T).
    # Fresh snow is ICE at T_fresh, so it adds only sensible enthalpy
    # c_ice*(T_fresh-T_freeze) (no latent term — it is not liquid).
    top_w = swe_ice[..., 0] + swe_liq[..., 0] + snowfall
    top_H = _enthalpy(swe_ice[..., 0], swe_liq[..., 0], T[..., 0]) + fresh_enth
    ti, tl, tt = _phase_from_w_H(top_w, top_H)
    swe_ice = swe_ice.at[..., 0].set(ti)
    swe_liq = swe_liq.at[..., 0].set(tl)
    T = T.at[..., 0].set(tt)
    # Fresh-snow density mixing (mass-weighted) for the top layer.
    old_top_mass = state.swe_ice[..., 0] + state.swe_liq[..., 0]
    density = density.at[..., 0].set(
        jnp.where(snowfall > _EPS,
                  (old_top_mass * density[..., 0] + snowfall * config.rho_snow_fresh)
                  / jnp.maximum(old_top_mass + snowfall, _EPS),
                  density[..., 0])
    )

    # --- 2. Equal-mass remap (static N, conserves mass + enthalpy) ---
    swe_ice, swe_liq, T, density = _remap_equal_mass(swe_ice, swe_liq, T, density)

    # --- 3. Compaction: density relaxes toward rho_snow_max ---
    density = density + (config.rho_snow_max - density) * jnp.clip(
        dt / config.compaction_timescale_s, 0.0, 1.0)
    density = jnp.clip(density, config.rho_snow_fresh, config.rho_snow_max)

    # --- 4. Thermal diffusion (implicit backward-Euler) ---
    dz, k = _thickness_and_conductivity(swe_ice, swe_liq, density, config)
    C = jnp.maximum(_sensible_hc(swe_ice, swe_liq), _EPS)     # full ice+liquid HC
    dz_half = jnp.maximum(0.5 * (dz[..., :-1] + dz[..., 1:]), _DZ_HALF_MIN)
    k_half = 2.0 * k[..., :-1] * k[..., 1:] / (k[..., :-1] + k[..., 1:] + _EPS)
    coeff = k_half / dz_half
    diag = C / dt
    diag = diag.at[..., 1:].add(coeff)
    diag = diag.at[..., :-1].add(coeff)
    sub = -coeff
    sup = -coeff
    rhs = C / dt * T
    rhs = rhs.at[..., 0].add(Q_top)                          # surface flux INTO top
    rhs = rhs.at[..., -1].add(-G_bottom)                     # base loses G_bottom to soil
    a = jnp.pad(sub, [(0, 0)] * (sub.ndim - 1) + [(1, 0)])
    c = jnp.pad(sup, [(0, 0)] * (sup.ndim - 1) + [(0, 1)])
    T = thomas_solve(a, diag, c, rhs)

    # --- 5. Phase change (enthalpy method): melt + refreeze per layer ---
    w = swe_ice + swe_liq
    H = _enthalpy(swe_ice, swe_liq, T)
    swe_ice, swe_liq, T = _phase_from_w_H(w, H)

    # --- 6. Percolation + drainage (liquid carries its full enthalpy) ---
    # Scan top->bottom: excess liquid beyond the irreducible holding capacity
    # flows down and each receiving layer re-equilibrates.  Carry BOTH the
    # downward liquid MASS and its ENTHALPY, so above-freezing liquid transports
    # its sensible heat too (not just latent L_f) and the budget closes.
    def _perc(carry, i):
        in_m, in_H = carry
        ice_i = jnp.take(swe_ice, i, axis=-1)
        liq_i = jnp.take(swe_liq, i, axis=-1)
        T_i = jnp.take(T, i, axis=-1)
        # Receive the downward liquid flux (mass + enthalpy) and re-equilibrate.
        w_i = ice_i + liq_i + in_m
        H_i = _enthalpy(ice_i, liq_i, T_i) + in_H
        ice_i, liq_i, T_i = _phase_from_w_H(w_i, H_i)
        # Drain liquid beyond the irreducible holding capacity; it leaves at the
        # layer temperature T_i, carrying enthalpy c_liq*(T_i-Tf)+Lf per unit
        # mass (removing liquid at T_i leaves the remainder's T_i unchanged).
        hold = config.irreducible_liq_frac * ice_i
        drain = jnp.maximum(liq_i - hold, 0.0)
        drain_H = drain * (_C_LIQ * (T_i - _TF) + _LF)
        liq_i = liq_i - drain
        return (drain, drain_H), (ice_i, liq_i, T_i)

    z = jnp.zeros(swe_ice.shape[:-1])
    (drainage, drainage_heat), (ice_s, liq_s, T_s) = jax.lax.scan(
        _perc, (z, z), jnp.arange(n))
    swe_ice = jnp.moveaxis(ice_s, 0, -1)
    swe_liq = jnp.moveaxis(liq_s, 0, -1)
    T = jnp.moveaxis(T_s, 0, -1)
    # ``drainage`` (mass) and ``drainage_heat`` (enthalpy relative to T_freeze)
    # are the fluxes OUT of the base -> route the water AND its heat to the soil.

    new_state = SnowColumnState(swe_ice=swe_ice, swe_liq=swe_liq, T=T, density=density)
    return new_state, drainage, drainage_heat


def total_water(state: SnowColumnState) -> jnp.ndarray:
    """Total pack water (ice + liquid) [kg/m^2], summed over layers."""
    return jnp.sum(state.swe_ice + state.swe_liq, axis=-1)


def column_enthalpy(state: SnowColumnState) -> jnp.ndarray:
    """Column enthalpy relative to T_freeze [J/m^2] (sensible + fusion of the
    liquid fraction), summed over layers — for energy-conservation checks."""
    return jnp.sum(_enthalpy(state.swe_ice, state.swe_liq, state.T), axis=-1)
