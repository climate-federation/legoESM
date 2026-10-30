"""Warm bin condensation/evaporation step (oracle ``ONECOND1``).

Assembles the ported chain for one microphysics step of one column cell:

    S = q p/(0.622+0.378 q)/e_s − 1     (FAST_SBM l. 4594-4597, e_s=POLYSVP)
    B_k  = JERRATE   (growth coefficients)
    SFN  = JERTIMESC (spectrum relaxation integral)
    R    = (OPER2(q) + (BB1/T²)·AL1) (1+S) SFN          (ONECOND1 ``RW``)
    S(t), ∫S dt = JERSUPSAT (exact linear-ODE step, zero in-step forcing —
                  the oracle calls it with ``DYN1 = 0``)
    m_k' = JERDFUN  (m^{2/3} growth update with B_k ∫S dt)
    f'   = JERNEWF  (KO + 3-point remap)
    Δq_c = max(Σ f' m dm/ρ, 0) − max(Σ f m dm/ρ, 0)
    q'   = q − Δq_c,   T' = T + AL1·Δq_c,  AL1 = 2500 K  (oracle literal)

Faithfulness note on substepping: this oracle version's ONECOND1 loop sets
``DTNEWL = min(DT, DT−TIMENEW)`` with no further limiter, so it executes
exactly ONE substep of the full ``dt`` and its final
``JERDFUN_NEW(SUPINTW)`` remap of the original spectrum coincides with the
in-loop remap — the port implements that single-pass form directly.
(The multi-substep machinery is vestigial there; if a future oracle
re-enables ``DT_WATER_COND`` limits, wrap this function in ``lax.scan``.)

Closure: ``q + q_c/ρ`` is invariant and ``ΔT + AL1·Δq_v = 0`` with the
oracle's ``AL1 = 2500 K`` (its rounding of L_v/c_pd, kept so the port
reproduces the running scheme; user decision 2026-10-19).

Scope: ONE ``ONECOND1`` call (the oracle's warm path for one condensation
substep). The FAST_SBM ``NCOND = 3`` substep loop, its ``DIFFU`` gate and
the ``wrf_error_fatal`` stops are NOT here (column assembly; see the
package status).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.diffusional_growth import (
    drop_growth_coefficient,
    saturation_vapor_pressure_fast_sbm,
    supersat_relaxation_integral,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    mass_density,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.remap import (
    condensation_new_masses,
    remap_spectrum,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.supersaturation import (
    integrate_supersaturation,
    supersat_relaxation_rate,
)

__physics_contract__ = {
    "summary": (
        "One warm condensation/evaporation step of the liquid bin spectrum "
        "with exact vapor/heat closure (oracle ONECOND1: JERRATE -> "
        "JERTIMESC -> JERSUPSAT -> JERDFUN/JERNEWF -> T,q update)."
    ),
    "inputs": {
        "f": "m^-3 kg^-1 (liquid size distribution)",
        "T": "K",
        "q_v": "kg/kg (vapor mixing ratio)",
        "p": "Pa",
        "rho_air": "kg m^-3",
        "dt": "s",
        "v_term": "m s^-1 (bin terminal velocities for ventilation)",
    },
    "outputs": {
        "f": "m^-3 kg^-1",
        "T": "K",
        "q_v": "kg/kg",
        "dq_c": "kg/kg (condensed this step; negative = net evaporation)",
    },
    "sign_convention": (
        "Supersaturated (S>0): droplets grow, dq_c>0, q_v decreases, T "
        "rises by AL1*dq_c (AL1 = 2500 K, oracle); subsaturated: "
        "evaporation, opposite signs. Total water q_v + q_c/rho is exactly "
        "invariant (when neither mass total is clamped); dT + AL1*dq_v = 0."
    ),
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": (
        "Khain et al. (2004) JAS 61:2963; Khain & Sednev (1996); WRF "
        "module_mp_fast_sbm.F ONECOND1"
    ),
    "idealized_test": (
        "Supersaturated parcel condenses (dq_c>0, T up, S decays toward "
        "equilibrium); subsaturated cloudy parcel evaporates; exactly "
        "saturated parcel with no spectrum is a fixed point; total water "
        "and moist enthalpy closures hold to roundoff; gradients flow to "
        "(T, q_v) and config-bearing inputs."
    ),
}


class WarmCondensationResult(NamedTuple):
    f: jax.Array         # updated spectrum (n_bins,)
    T: jax.Array         # updated temperature [K]
    q_v: jax.Array       # updated vapor mixing ratio [kg/kg]
    dq_c: jax.Array      # condensed mixing ratio this step [kg/kg]
    S_new: jax.Array     # end-of-step supersaturation (diagnostic)
    min_psi: jax.Array   # remap negativity diagnostic (oracle fatal cond.)


def saturation_ratio_fast_sbm(T: jax.Array, p: jax.Array, q_v: jax.Array) -> jax.Array:
    """FAST_SBM saturation ratio over water (l. 4594-4597):
    ``EW1N/ES1N`` with ``EW1N = q p/(0.622 + 0.378 q)``, ``ES1N`` = POLYSVP."""
    e_v = q_v * p / (constants.epsilon_fast_sbm
                     + constants.one_minus_epsilon_fast_sbm * q_v)
    return e_v / saturation_vapor_pressure_fast_sbm(T)


def warm_condensation_step(
    f: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p: jax.Array,
    rho_air: jax.Array,
    dt: float | jax.Array,
    masses: jax.Array,
    v_term: jax.Array | None = None,
    config: FastSBMConfig = FastSBMConfig(),
) -> WarmCondensationResult:
    """One ONECOND1 step for a single cell (1-D spectrum; ``vmap`` columns).

    ``v_term`` defaults to still air (ventilation factor 1) — pass bin
    terminal velocities once sedimentation lands for full fidelity.
    """
    if v_term is None:
        v_term = jnp.zeros_like(masses)

    S = saturation_ratio_fast_sbm(T, p, q_v) - 1.0     # DEL1IN = EW1N/ES1N-1.
    B = drop_growth_coefficient(masses, T, p, v_term, config=config)
    sfn = supersat_relaxation_integral(f, masses, B, rho_air)
    R = supersat_relaxation_rate(T, q_v, S, sfn)
    # Oracle calls JERSUPSAT with DYN1 = 0 — dynamics forcing enters
    # between microphysics calls, not inside the substep.
    step = integrate_supersaturation(S, R, jnp.zeros_like(S), dt)

    m_new = condensation_new_masses(masses, B, step.S_int)
    remap = remap_spectrum(f, masses, m_new)

    # Mass closure (oracle DELMASSL1 = RMASSLAA − RMASSLBB, each
    # Σ f x² ·COL3/ROR clamped at >= 0; QPN = QOLD − Δ, TPN = TOLD + AL1·Δ).
    m_after = jnp.maximum(mass_density(remap.f_new, masses) / rho_air, 0.0)
    m_before = jnp.maximum(mass_density(f, masses) / rho_air, 0.0)
    dq_c = m_after - m_before
    q_new = q_v - dq_c
    T_new = T + constants.lv_over_cp_fast_sbm * dq_c

    return WarmCondensationResult(
        f=remap.f_new, T=T_new, q_v=q_new, dq_c=dq_c,
        S_new=step.S_new, min_psi=remap.min_psi)
