"""Stateless column adapter for the Super-Droplet Method.

Exposes SDM as a switchable ``scheme="sdm"`` microphysics backend that matches
the legoESM column-physics interface
``micro_fn(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, cfg)
-> MicrophysicsOutput`` used by every dynamical core (see
``microphysics/integration.py``).

The column interface is **stateless** (no particle state is carried between
steps), while SDM is Lagrangian. This adapter therefore implements the
well-defined, differentiable part — **diffusional condensation/evaporation** —
on a *reconstructed* mean cloud droplet: each cell's existing cloud water
``q_c`` and a prescribed cloud-droplet number concentration ``cdnc`` define a
mean droplet radius ``R = (q_c ρ / (cdnc · (4/3)π ρ_w))^{1/3}`` (floored at
``r_min_reconstruct`` with a correspondingly reduced effective number for thin
cloud, so the closure never manufactures sub-micron Kelvin artifacts and the
tendency is continuous in ``q_c``); the SDM growth
law (``condensation.integrate_radius``) advances ``R`` for one step at the
cell's saturation ratio ``S = e/e_sat``; the regrown liquid is deposited back
as ``dq_c``, with the condensed/evaporated mass exchanged with vapor (total
water conserved) and released as latent heat. Given nonnegative inputs, the
donor clamps keep ``q_v`` and ``q_c`` nonnegative over an explicit step — the
operator never manufactures negatives (it clips its own inputs for the physics,
like the other column schemes), but it does not repair a pre-existing negative
tracer; that is the dynamical core's responsibility.

**Documented limitations** (the faithful Lagrangian model is ``box_model.py``,
and full coupling would need a particle-state-carrying interface):
collision-coalescence, sedimentation/precipitation, and aerosol activation /
nucleation are NOT done here — they require a persistent droplet population.
A supersaturated *clear* cell (no ``q_c``) produces no cloud (no activation).
So this adapter grows/evaporates pre-existing cloud water; it is a condensation
operator, not a complete precipitating microphysics. ``dq_r/dq_i/...`` are zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import relative_humidity
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.condensation import integrate_radius
from legoesm.atmosphere.physics.microphysics.sdm.particles import SuperDropletState

__physics_contract__ = {
    "summary": (
        "SDM column condensation operator: grow a reconstructed mean cloud "
        "droplet (from q_c and a prescribed CDNC) by vapor diffusion and "
        "deposit the cloud-water / vapor / temperature tendencies."
    ),
    "inputs": {
        "T": "K",
        "q_v": "kg/kg",
        "q_c": "kg/kg",
        "p_full": "Pa",
        "rho": "kg/m^3",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s",
        "dq_v_dt": "kg/kg/s",
        "dq_c_dt": "kg/kg/s",
    },
    "sign_convention": (
        "Supersaturated (S>1) grows q_c (dq_c_dt>0), removes vapor "
        "(dq_v_dt=-dq_c_dt) and warms (dT_dt=L_v/c_p·dq_c_dt); subsaturated "
        "evaporates q_c, moistens and cools. Total water q_v+q_c is conserved."
    ),
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF SuperDropletsMoist (phaseChange)",
    "idealized_test": (
        "Clear/subsaturated column -> zero tendency; a supersaturated cloudy "
        "cell grows q_c and removes the same q_v (dq_v=-dq_c), dT=L_v/c_p·dq_c; "
        "no-growth (S=1, no curvature) round-trip gives dq_c=0 exactly; donor "
        "clamps keep q_v, q_c >= 0 over one explicit step for nonnegative inputs."
    ),
}

# (4/3)π — sphere volume prefactor (pure geometry).
_FOUR_THIRDS_PI = 4.0 / 3.0 * jnp.pi


def sdm_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SDMConfig = SDMConfig(),
) -> MicrophysicsOutput:
    """Super-Droplet Method column condensation tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water-vapor mixing ratio [kg/kg], shape (ncol, nlev).
    hydrometeors : HydrometeorState
        Hydrometeor state (only ``q_c`` is used).
    p_full, p_half : jax.Array
        Full / half level pressure [Pa].
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m] (unused — no sedimentation here).
    dt : float
        Physics step [s].
    config : SDMConfig
        SDM configuration (uses ``cdnc``, ``qc_min``, and the condensation
        integrator settings). Static argument.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    _dtype = T.dtype
    rho_w = constants.rho_water

    q_c = jnp.clip(hydrometeors.q_c, 0.0, None)
    q_v_pos = jnp.clip(q_v, 0.0, None)

    # Reconstruct a mean cloud droplet from q_c and the prescribed CDNC:
    # mass per droplet m = q_c ρ / N ; R = (m / ((4/3)π ρ_w))^{1/3}.
    # THIN-CLOUD CLOSURE: when the fixed-cdnc inversion would give a droplet
    # smaller than r_min_reconstruct (an nm-scale Kelvin-barrier *artifact* of
    # the closure, not physics), hold the droplet at r_min_reconstruct and
    # reduce the effective number instead (N_eff = q_c·ρ/m_min ∝ q_c). The
    # round trip stays exact (N_eff·m/ρ = q_c in both branches) and the
    # tendency vanishes continuously as q_c -> 0 (no jump at the qc_min gate).
    cdnc = config.cdnc
    m_min = _FOUR_THIRDS_PI * rho_w * config.r_min_reconstruct**3
    m_drop = q_c * rho / cdnc                       # [kg] per droplet at full cdnc
    N_eff = jnp.where(m_drop >= m_min, cdnc, q_c * rho / m_min)   # [1/m^3]
    R = jnp.cbrt(jnp.maximum(m_drop, m_min) / (_FOUR_THIRDS_PI * rho_w))
    cloudy = q_c > config.qc_min                    # cells with cloud water
    R = jnp.where(cloudy, R, config.r_min_reconstruct)

    # Saturation ratio S = e/e_sat (vapor-pressure based — NOT q_v/q_sat).
    S = relative_humidity(T, p_full, q_v)

    # Grow the mean droplet by one step using the scheme config as-is. The
    # reconstruction carries no aerosol, so ``solute_mass = 0`` makes the Raoult
    # term identically zero regardless of ``cfg.include_solute``; the Kelvin
    # curvature term is honored per ``cfg.include_curvature`` and is harmless
    # here because the reconstructed droplet is never smaller than
    # ``r_min_reconstruct`` (1 um default), where the Kelvin barrier is
    # ~0.1 % supersaturation.
    droplets = SuperDropletState(
        multiplicity=jnp.ones_like(R),
        radius=R,
        solute_mass=jnp.zeros_like(R),
        active=jnp.where(cloudy, 1.0, 0.0).astype(_dtype),
    )
    droplets = integrate_radius(droplets, S, T, dt, config)
    R_new = droplets.radius

    # Regrown cloud water (same N_eff closure as the reconstruction, so the
    # no-growth round trip is exact); clear cells stay clear.
    q_c_new = N_eff * _FOUR_THIRDS_PI * rho_w * R_new**3 / rho
    q_c_new = jnp.where(cloudy, q_c_new, q_c)

    dq_c_dt = (q_c_new - q_c) / dt
    # Donor clamps: condensation cannot exceed available vapor, evaporation
    # cannot exceed available cloud water (positivity over one explicit step).
    dq_c_dt = jnp.clip(dq_c_dt, -q_c / dt, q_v_pos / dt)
    dq_v_dt = -dq_c_dt
    dT_dt = constants.L_v * dq_c_dt / constants.c_pd

    z = jnp.zeros((ncol, nlev), dtype=_dtype)
    z1 = jnp.zeros((ncol,), dtype=_dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=z,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=z1,
    )
