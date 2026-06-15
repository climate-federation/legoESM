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

By default this adapter keeps that legacy condensation-only path. An explicit
``SDMConfig(column_do_coalescence=True)`` opt-in runs a stateless
per-step-reconstructed box-SDM: reconstruct an ``n_sd`` population in each cell
from the Eulerian ``q_c, N_c, q_r, N_r``, run ``box_step`` (condensation then
Shima Monte-Carlo coalescence), then project the resulting particles back to
``dq_c, dq_r, dN_c, dN_r``. This is useful for warm-rain process experiments in
the existing Eulerian tracer framework, but it is NOT faithful advected
Lagrangian SDM: particles are not persistent, not transported by the resolved
flow, and the keyless dispatch forces a fixed config-seeded PRNG stream.

**Documented limitations** (the faithful Lagrangian model is ``box_model.py``,
and full coupling would need a particle-state-carrying interface):
sedimentation/precipitation and aerosol-Köhler activation/nucleation are NOT
done here. In the DEFAULT condensation-only path a supersaturated *clear* cell
(no ``q_c``) produces no cloud (no activation), and ``dq_r/dN_*`` are zero. The
opt-in ``column_do_coalescence`` path adds a MINIMAL saturation activation
(nucleate ``cdnc`` embryo droplets at the radius floor where a cell is
supersaturated and cloud-free, so condensation has something to grow — see
:func:`_sdm_reconstructed_box_microphysics`); it is still NOT aerosol-Köhler
activation, and the embryo liquid is negligible (a small documented leak).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from legoesm import constants
from legoesm.thermo import relative_humidity
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)
from legoesm.atmosphere.physics.microphysics.sdm.box_model import (
    BoxState,
    box_step,
)
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.condensation import integrate_radius
from legoesm.atmosphere.physics.microphysics.sdm.coupling import (
    cloud_rain_mixing_ratios,
)
from legoesm.atmosphere.physics.microphysics.sdm.init import sample_exponential_mass
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


def _sdm_condensation_only_microphysics(
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
    """Legacy Super-Droplet Method column condensation tendencies.

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
        SDM configuration (uses ``cdnc``, ``r_min_reconstruct``, and the condensation
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
    # round trip is exact in both branches (N_eff·m/ρ = q_c, including q_c=0
    # where N_eff=0), so no cloudy/clear gate is needed at all: the tendency is
    # exactly continuous, ∝ q_c for thin cloud and identically 0 in clear air.
    # A supersaturated clear cell still produces nothing (no activation —
    # documented above).
    cdnc = config.cdnc
    m_min = _FOUR_THIRDS_PI * rho_w * config.r_min_reconstruct**3
    m_drop = q_c * rho / cdnc                       # [kg] per droplet at full cdnc
    N_eff = jnp.where(m_drop >= m_min, cdnc, q_c * rho / m_min)   # [1/m^3]
    R = jnp.cbrt(jnp.maximum(m_drop, m_min) / (_FOUR_THIRDS_PI * rho_w))

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
        active=jnp.ones_like(R),
    )
    droplets = integrate_radius(droplets, S, T, dt, config)
    R_new = droplets.radius

    # Regrown cloud water (same N_eff closure as the reconstruction, so the
    # no-growth round trip is exact; clear cells have N_eff = 0 -> stay clear).
    q_c_new = N_eff * _FOUR_THIRDS_PI * rho_w * R_new**3 / rho

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


def _normalised_exponential_samples(
    keys: jax.Array,
    n_sd: int,
    dtype,
) -> jax.Array:
    """Unit-mean exponential samples, vmapped over cell keys."""
    raw = jax.vmap(
        lambda k: sample_exponential_mass(k, n_sd, 1.0, dtype=dtype)
    )(keys)
    mean = jnp.mean(raw, axis=-1, keepdims=True)
    return raw / jnp.maximum(mean, jnp.asarray(1.0e-30, dtype=dtype))


def _cloud_rain_number_concentrations(
    state: SuperDropletState,
    V_cell: jax.Array,
    r_rain: float,
) -> tuple[jax.Array, jax.Array]:
    """Project represented droplet counts to cloud/rain number [1/m^3]."""
    represented = state.active * state.multiplicity
    is_cloud = state.radius < r_rain
    inv_v = 1.0 / V_cell
    N_c = jnp.sum(jnp.where(is_cloud, represented, 0.0)) * inv_v
    N_r = jnp.sum(jnp.where(is_cloud, 0.0, represented)) * inv_v
    return N_c, N_r


def _sdm_reconstructed_box_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SDMConfig,
) -> MicrophysicsOutput:
    """Stateless per-cell reconstructed-box SDM with Shima coalescence.

    This is an Eulerian adapter closure, not a persistent Lagrangian SDM
    coupling. A unit horizontal area is assumed, so ``V_cell = dz`` and
    multiplicities are reconstructed from number concentration times volume.
    The arbitrary unit area cancels from the coalescence probability when the
    number concentrations and ``V_cell`` are used consistently.
    """
    del p_half
    n_sd = int(config.column_n_sd)
    if n_sd < 4:
        raise ValueError(
            f"column_n_sd must be >= 4 when column_do_coalescence=True, got {n_sd}"
        )

    n_cloud_sd = n_sd // 2
    n_rain_sd = n_sd - n_cloud_sd
    ncol, nlev = T.shape
    dtype = T.dtype
    ncell = ncol * nlev
    dt_arr = jnp.asarray(dt, dtype=dtype)
    pref = _FOUR_THIRDS_PI * constants.rho_water
    m_min_cloud = pref * config.r_min_reconstruct**3
    m_rain = pref * config.r_rain**3

    q_c = jnp.clip(hydrometeors.q_c, 0.0, None)
    q_r = jnp.clip(hydrometeors.q_r, 0.0, None)
    q_v_pos = jnp.clip(q_v, 0.0, None)
    V = jnp.maximum(dz, jnp.asarray(1.0e-12, dtype=dtype))

    flat = lambda a: jnp.reshape(a, (ncell,))  # noqa: E731
    T_f = flat(T)
    qv_f = flat(q_v_pos)
    qc_f = flat(q_c)
    qr_f = flat(q_r)
    Nc_f = flat(hydrometeors.N_c)
    Nr_f = flat(hydrometeors.N_r)
    p_f = flat(p_full)
    rho_f = flat(rho)
    V_f = flat(V)

    # Fold a cheap state-derived hash into the base key so the stochastic
    # collision pairing varies as the field evolves (codex 2026-06-13: a fixed
    # per-step key biases collisions — seed 0 gave ~no rain). Deterministic,
    # jit-safe (a pure function of the current state).
    state_hash = jnp.mod(
        jnp.sum(jnp.abs(qv_f) * 1.0e9 + jnp.abs(T_f) * 1.0e3),
        jnp.asarray(2.0 ** 31, dtype=dtype)).astype(jnp.uint32)
    base_key = random.fold_in(random.PRNGKey(int(config.column_seed)), state_hash)
    cell_ids = jnp.arange(ncell, dtype=jnp.uint32)
    keys = jax.vmap(lambda i: random.fold_in(base_key, i))(cell_ids)
    keys_cloud = jax.vmap(lambda k: random.fold_in(k, 0))(keys)
    keys_rain = jax.vmap(lambda k: random.fold_in(k, 1))(keys)
    raw_cloud = _normalised_exponential_samples(keys_cloud, n_cloud_sd, dtype)
    raw_rain = _normalised_exponential_samples(keys_rain, n_rain_sd, dtype)

    ones_cloud = jnp.ones((n_cloud_sd,), dtype=dtype)
    ones_rain = jnp.ones((n_rain_sd,), dtype=dtype)

    def cell(T_c, qv_c, qc_c, qr_c, Nc_c, Nr_c, p_c, rho_c, V_c,
             raw_c, raw_r, key):
        M_air = rho_c * V_c

        # Cloud: use prognostic N_c when present, otherwise the configured CDNC.
        # For ultra-thin cloud, reduce the effective number so the mean mass is
        # no smaller than the same radius floor used by the condensation-only
        # adapter, preserving total cloud mass exactly.
        # Minimal saturation activation (this SDM column has NO aerosol-Köhler
        # activation — codex 2026-06-13): where the cell is supersaturated and
        # carries no cloud, nucleate ``cdnc`` embryo droplets at the radius floor
        # so condensation can grow them. Without this a vapor-only IC (BOMEX)
        # never forms cloud. Embryo liquid is negligible (~r_min) so activation
        # itself adds ≈0 water (small documented leak); growth draws from vapor.
        S_c = relative_humidity(T_c, p_c, qv_c)
        Nc_target = jnp.where(Nc_c > 0.0, Nc_c, config.cdnc)
        activate = (S_c > 1.0) & (qc_c <= 0.0)
        Nc_eff = jnp.where(
            qc_c > 0.0,
            jnp.minimum(Nc_target, qc_c * rho_c / m_min_cloud),
            jnp.where(activate, config.cdnc, 0.0),
        )
        mean_m_c = jnp.where(qc_c > 0.0,
                             qc_c * rho_c / jnp.maximum(Nc_eff, 1.0e-30),
                             m_min_cloud)
        mass_c = mean_m_c * raw_c
        xi_c = ones_cloud * (Nc_eff * V_c / n_cloud_sd)

        # Rain: use prognostic N_r when present, otherwise a configurable floor.
        # If the requested rain number would imply sub-rain-threshold droplets,
        # reduce the effective number so every rain-mode sample is >= r_rain.
        Nr_target = jnp.where(Nr_c > 0.0, Nr_c, config.column_n_rain_floor)
        Nr_eff = jnp.where(
            qr_c > 0.0,
            jnp.minimum(Nr_target, qr_c * rho_c / m_rain),
            0.0,
        )
        mean_m_r = jnp.where(Nr_eff > 0.0, qr_c * rho_c / Nr_eff, m_rain)
        mass_r = m_rain + jnp.maximum(mean_m_r - m_rain, 0.0) * raw_r
        xi_r = ones_rain * (Nr_eff * V_c / n_rain_sd)

        mass = jnp.concatenate([mass_c, mass_r])
        xi = jnp.concatenate([xi_c, xi_r])
        radius = jnp.cbrt(jnp.maximum(mass, jnp.asarray(1.0e-300, dtype=dtype))
                          / pref)
        active = (xi > 0.0).astype(dtype)
        droplets0 = SuperDropletState(
            multiplicity=xi,
            radius=radius,
            solute_mass=jnp.zeros((n_sd,), dtype=dtype),
            active=active,
        )
        box0 = BoxState(droplets=droplets0, T=T_c, p=p_c, q_v=qv_c, key=key)
        box1 = box_step(
            box0, V_c, M_air, dt_arr, config,
            do_condensation=True, do_coalescence=True,
        )

        qc0, qr0 = cloud_rain_mixing_ratios(
            droplets0, V_c, rho_c, config.r_rain)
        qc1, qr1 = cloud_rain_mixing_ratios(
            box1.droplets, V_c, rho_c, config.r_rain)
        nc0, nr0 = _cloud_rain_number_concentrations(
            droplets0, V_c, config.r_rain)
        nc1, nr1 = _cloud_rain_number_concentrations(
            box1.droplets, V_c, config.r_rain)
        inv_dt = 1.0 / dt_arr
        dqv = (box1.q_v - qv_c) * inv_dt
        dqc = (qc1 - qc0) * inv_dt
        dqr = (qr1 - qr0) * inv_dt
        # (codex 1) condensation donor clamp (box_step lacks the legacy clamp):
        # vapor sink ≤ available vapor, evaporation source ≤ available liquid.
        # Move the correction into the liquid so q_v+q_c+q_r is conserved, and
        # recompute the latent heating from the clamped condensation.
        #
        # box_step conserves total water to machine precision, so the
        # pre-clamp dqv+dqc+dqr ≈ 0. Clamping vapor by Δ = dqv_cl - dqv must be
        # COMPENSATED with the OPPOSITE sign in liquid to keep the sum invariant
        # (the earlier ``dqc += Δ`` added +2Δ of spurious total water whenever
        # the clamp bound — codex round 1).
        dqv_cl = jnp.clip(dqv, -qv_c * inv_dt, (qc_c + qr_c) * inv_dt)
        delta = dqv_cl - dqv      # vapor change from the clamp
        # (codex 2 round 2) Apportion the −delta liquid correction so BOTH q_c
        # and q_r stay nonnegative over the explicit step. Putting all of −delta
        # into q_c drove q_c negative when coalescence/condensation had already
        # made dqc strongly negative (e.g. q_c → −0.036 at rh=1.2, dt=60). Since
        # the clamp removes at most the available liquid (qc_c+qr_c)/dt, the
        # total-liquid post-step is ≥ 0, so the correction can always be split
        # between the two species to keep each ≥ 0: take from q_c first, spill
        # the part that would underflow q_c into q_r (and vice-versa). The split
        # is conservation-neutral (the two corrections sum to −delta).
        dqc = dqc - delta
        qc_post = qc_c + dqc * dt_arr
        qr_post = qr_c + dqr * dt_arr
        # Amount by which q_c would go negative (≥ 0); move it from q_c to q_r.
        spill_c = jnp.maximum(-qc_post, 0.0) * inv_dt
        dqc = dqc + spill_c
        dqr = dqr - spill_c
        # Symmetric guard if the correction instead underflowed q_r.
        qr_post = qr_c + dqr * dt_arr
        spill_r = jnp.maximum(-qr_post, 0.0) * inv_dt
        dqr = dqr + spill_r
        dqc = dqc - spill_r
        dqv = dqv_cl
        dT = -constants.L_v / constants.c_pd * dqv
        # (codex 2) number tendencies cannot drive the Eulerian number tracers
        # negative (the reconstructed baseline ≠ the Eulerian N when N starts 0).
        dNc = jnp.maximum((nc1 - nc0) * inv_dt, -Nc_c * inv_dt)
        dNr = jnp.maximum((nr1 - nr0) * inv_dt, -Nr_c * inv_dt)
        return (dT, dqv, dqc, dqr, dNc, dNr)

    dT, dqv, dqc, dqr, dNc, dNr = jax.vmap(cell)(
        T_f, qv_f, qc_f, qr_f, Nc_f, Nr_f, p_f, rho_f, V_f,
        raw_cloud, raw_rain, keys,
    )

    shape = (ncol, nlev)
    z = jnp.zeros(shape, dtype=dtype)
    z1 = jnp.zeros((ncol,), dtype=dtype)
    return MicrophysicsOutput(
        dT_dt=jnp.reshape(dT, shape),
        dq_v_dt=jnp.reshape(dqv, shape),
        dq_c_dt=jnp.reshape(dqc, shape),
        dq_r_dt=jnp.reshape(dqr, shape),
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=jnp.reshape(dNc, shape),
        dN_r_dt=jnp.reshape(dNr, shape),
        dN_i_dt=z,
        precipitation=z1,
    )


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
    """Super-Droplet Method column tendencies for the Eulerian dispatch.

    Default behavior is the legacy condensation-only mean-droplet closure. With
    ``config.column_do_coalescence=True``, this runs a stateless
    reconstructed-box SDM step per cell and returns cloud/rain mass and number
    tendencies from Shima coalescence. The opt-in path is stochastic,
    deterministic for ``config.column_seed``, and not reverse-mode
    differentiable.
    """
    if config.column_do_coalescence:
        return _sdm_reconstructed_box_microphysics(
            T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config)
    return _sdm_condensation_only_microphysics(
        T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config)
