"""Bin diffusional-growth coefficients (oracle ``JERRATE_KS``/``JERTIMESC_KS``).

Maxwellian vapor growth of one bin particle with Pruppacher–Klett
ventilation:

    dm_k/dt = B_k · s,   s = supersaturation (e/e_s − 1)
    B_k = 4π C_k f_vent(Re_k, Sc) / (F_D + F_K)
    F_D = R_v T / (D e_s(T)),   F_K = (L/(R_v T) − 1) · L / (k_air T)

(oracle ``JERRATE_KS``: ``FD1``/``FK1``/``B11_MY`` with capacitance
``RIEC``, 4π = ``CONST = 12.566372``). For liquid drops the capacitance is
the radius (the oracle reads it from ``capacity33.asc``; spheres
analytically). The supersaturation relaxation integral

    SFN = (1/ρ_air) Σ_k f_k B_k dm_k        (oracle ``JERTIMESC_KS``,
                                             ``CF = B8L = 1/ROR``)

feeds the analytic supersaturation integrator (next module in the chain).

Saturation vapor pressure comes from ``legoesm.thermo`` (repo-wide single
saturation curve) — NOT a port of the oracle's ``A·exp(−B/T)`` exponential
fit / ``POLYSVP``. Deviation vs the oracle curve (codex-verified): ~+0.1%
near 273 K but up to ≈3.4% at the 243 K / 310 K extremes, propagating
proportionally into ``F_D`` and ``B`` there. Model consistency wins
anyway (CLAUDE.md shared-utilities rule: a re-derived curve diverging from
the model's own saturation caused false supersaturation in CI before) —
oracle-fidelity comparisons at cold/hot extremes must account for this.

Other accepted constant deviations from the oracle's CGS values:
``constants.L_v`` vs the oracle's 2.5e10 erg/g (+0.04%, ~0.08% in F_K) and
``constants.p_atm_std`` vs PZERO = 101300 Pa (0.025% in D) — both kept on
repo-wide constants by design.

Differences from ``sdm/condensation.py`` (same Maxwell denominator, shared
deliberately at the formula level, not by import): SDM works per-particle
in dR²/dt with a Knudsen (Fukuta–Walter) correction for micron droplets;
the bin scheme works in dm/dt with ventilation for the precipitation-size
tail — the oracle has no Knudsen term and SDM has no ventilation. The
cross-check test pins both to the same Maxwell core in the overlap regime.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    bin_mass_widths,
    radius_from_mass,
)
from legoesm.thermo import saturation_vapor_pressure

__physics_contract__ = {
    "summary": (
        "Maxwellian bin growth coefficients dm/dt = B·s with "
        "Pruppacher-Klett ventilation (oracle JERRATE_KS) and the "
        "supersaturation relaxation integral SFN = Σ f B dm / ρ_air "
        "(oracle JERTIMESC_KS)."
    ),
    "inputs": {
        "masses": "kg (bin centres)",
        "T": "K",
        "p": "Pa",
        "v_term": "m s^-1 (bin terminal velocities, for ventilation)",
        "f": "m^-3 kg^-1 (size distribution, relaxation integral)",
        "rho_air": "kg m^-3",
    },
    "outputs": {
        "B": "kg s^-1 per unit supersaturation (per bin)",
        "SFN": "s^-1 (supersaturation relaxation rate integral)",
    },
    "sign_convention": (
        "B > 0 always; positive supersaturation s grows drops (dm/dt = B s "
        "> 0), subsaturation evaporates. SFN >= 0; the e-folding time of "
        "supersaturation against this spectrum is 1/(SFN·(A2-ish thermo "
        "factor)) in the JERSUPSAT integrator."
    ),
    # Coefficients only — no state update happens in this module.
    "conserves": ["none"],
    # Differentiable a.e.: the oracle's ventilation branch at Re = 2.5 is a
    # deliberate ~1% jump and the ventilation_max cap is kinked — gradients
    # are finite everywhere but not globally smooth across those points.
    "differentiable": True,
    "reference": (
        "Pruppacher & Klett (1997) ch. 13; Khain et al. (2004) JAS 61:2963; "
        "WRF module_mp_fast_sbm.F JERRATE_KS/JERTIMESC_KS"
    ),
    "idealized_test": (
        "Ventilation-off B reproduces the SDM dR²/dt Maxwell core "
        "(cross-implementation agreement < 1e-10 rel with matched D); "
        "f_vent → 1 as V→0 and is capped at ventilation_max; B > 0 and "
        "monotone in capacitance; SFN linear in f and equal to Σ f B dm/ρ."
    ),
}

# 4π (oracle CONST = 12.566372) — geometry, not tunable.
_FOUR_PI = 4.0 * jnp.pi


# Beard-Pruppacher ventilation factor constants (fixed).
_VENT_RE_THRESHOLD = 2.5
_VENT_LOW_C = 0.108
_VENT_HIGH_C0 = 0.78
_VENT_HIGH_C1 = 0.308

def vapor_diffusivity(
    T: jax.Array, p: jax.Array, config: FastSBMConfig = FastSBMConfig()
) -> jax.Array:
    """``D = D_ref (p₀/p)(T/T₀)^a`` [m²/s] (oracle ``D_MY``; p₀ = 1013.25
    hPa = ``constants.p_atm_std``, T₀ = ``constants.T_freeze``)."""
    return (
        config.d_vapor_ref_m2s
        * (constants.p_atm_std / p)
        * (T / constants.T_freeze) ** config.diffusivity_T_exponent
    )


def ventilation_factor(
    masses: jax.Array,
    v_term: jax.Array,
    T: jax.Array,
    p: jax.Array,
    rho_bulk: float = constants.rho_water,
    config: FastSBMConfig = FastSBMConfig(),
) -> jax.Array:
    """Pruppacher–Klett ventilation coefficient per bin (oracle ``VENTPLM``).

    ``Re = 2 r V / ν`` (diameter Reynolds number — the oracle's
    ``B·V·(m/ρ_b)^{1/3}`` with ``B = 2(3/4π)^{1/3}/ν``), ``X = √Re·Sc^{1/3}``:

        f = 1 + 0.108 X²        (Re < 2.5)
        f = 0.78 + 0.308 X      (Re ≥ 2.5)

    capped at ``config.ventilation_max``. The oracle branch point is
    deliberately discontinuous (≈1% jump at Re = 2.5) — kept faithful.
    """
    nu = config.nu_air_ref_m2s
    D = vapor_diffusivity(T, p, config)
    schmidt = nu / D
    r = radius_from_mass(masses, rho_bulk)
    reynolds = 2.0 * r * v_term / nu
    # AD guard: ``sqrt`` has an unbounded derivative at 0, and the shipped
    # condensation path calls this with v_term = 0 ⇒ reynolds ≡ 0.  A shared
    # ``x = sqrt(reynolds)`` fed to BOTH jnp.where branches routes a
    # 0·inf = NaN cotangent into ``reynolds`` in reverse mode (same masked-
    # branch trap as nucleation.critical_dry_radius).  Fix: the low-Re branch
    # uses ``x² = reynolds·Sc^(2/3)`` directly (identical primal, smooth in
    # reynolds), and the high-Re branch clamps the sqrt argument — inactive
    # for its selected domain (Re ≥ 2.5), finite gradient when masked.
    sc13 = schmidt ** (1.0 / 3.0)
    x_high = jnp.sqrt(jnp.maximum(reynolds, 1e-30)) * sc13
    f = jnp.where(
        reynolds < _VENT_RE_THRESHOLD,
        1.0 + _VENT_LOW_C * reynolds * sc13 * sc13,
        _VENT_HIGH_C0 + _VENT_HIGH_C1 * x_high,
    )
    return jnp.minimum(f, config.ventilation_max)


def drop_growth_coefficient(
    masses: jax.Array,
    T: jax.Array,
    p: jax.Array,
    v_term: jax.Array,
    capacitance: jax.Array | None = None,
    rho_bulk: float = constants.rho_water,
    config: FastSBMConfig = FastSBMConfig(),
) -> jax.Array:
    """Growth coefficient ``B_k`` [kg/s per unit supersaturation] per bin
    (oracle ``B11_MY``): ``B = 4π C f_vent / (F_D + F_K)``.

    ``capacitance`` defaults to the sphere value (drop radius); pass the
    oracle's tabulated capacities for ice habits later.
    """
    if capacitance is None:
        capacitance = radius_from_mass(masses, rho_bulk)
    e_s = saturation_vapor_pressure(T)
    D = vapor_diffusivity(T, p, config)
    f_vent = ventilation_factor(masses, v_term, T, p, rho_bulk, config)
    F_D = constants.R_v * T / (D * e_s)
    F_K = (
        (constants.L_v / (constants.R_v * T) - 1.0)
        * constants.L_v
        / (constants.k_air * T)
    )
    return _FOUR_PI * capacitance * f_vent / (F_D + F_K)


def supersat_relaxation_integral(
    f: jax.Array,
    masses: jax.Array,
    growth_coefficient: jax.Array,
    rho_air: jax.Array,
) -> jax.Array:
    """``SFN = (1/ρ_air) Σ_k f_k B_k dm_k`` [s⁻¹] (oracle ``JERTIMESC_KS``
    with ``CF = 1/ROR``; ``dm = 3 COL m``). Bins on the last axis."""
    return jnp.sum(f * growth_coefficient * bin_mass_widths(masses), axis=-1) / rho_air
