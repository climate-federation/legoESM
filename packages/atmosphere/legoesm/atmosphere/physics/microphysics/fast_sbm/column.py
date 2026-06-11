"""Stateless column adapter — fast-SBM as ``scheme="fast_sbm"``.

Matches the legoESM column-physics interface
``micro_fn(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, cfg)
-> MicrophysicsOutput`` (see ``microphysics/integration.py``), following
the SDM adapter's stateless-reconstruction pattern but resolving a FULL
33-bin liquid spectrum per cell:

1. **Reconstruct** the liquid spectrum from bulk state: cloud water
   ``q_c`` → lognormal mode (prescribed ``cdnc``, ``cloud_geom_std``),
   rain ``q_r`` → exponential (Marshall-Palmer-like) mode using the
   dycore's prognostic ``N_r`` (floored). Each mode is rescaled to carry
   its bulk mass EXACTLY, so the adapter's closure introduces no
   discretization mass error.
2. **Evolve** one step of the ported oracle warm physics on the combined
   spectrum: ONECOND1 condensation/evaporation (exact vapor/heat closure)
   then Bott collision-coalescence (Hall/Long computed kernels) — so
   autoconversion/accretion EMERGE from the resolved stochastic-collection
   equation across the oracle's ``KRDROP`` (~50 µm) cloud/rain boundary
   instead of being parameterized.
3. **Project** back to bulk tendencies: ``dq_c``, ``dq_r`` (mass below /
   above ``KRDROP``), ``dN_c``, ``dN_r``, with ``dT``/``dq_v`` from the
   condensation closure.

After the per-cell physics, per-bin **sedimentation** (oracle
``FALFLUXHUCM_Z``) settles the spectrum down the column and yields the
surface precipitation.

A supersaturated cell nucleates new droplets from a prescribed aerosol
reservoir by Köhler activation (``nucleation.activate_ccn``) — so a clear
supersaturated cell forms cloud, not nothing.

**Documented limitations**:
* No ice (``dq_i/dq_s/dq_g = 0``) — lands with the ice iteration.
* Spectrum shape is re-imposed each step by reconstruction; the
  bin-resolved physics acts within the step.
* CCN activation is **deficit-diagnostic**, not a depleting aerosol
  reservoir: the activated number relaxes toward the Köhler target
  ``N_CCN·frac_above(r_crit)`` (self-limiting per cell), but there is no
  persistent inter-step/inter-cell aerosol budget (the oracle carries
  ``FCCNR`` and depletes it). Activated droplets are seeded into the
  smallest bin (the oracle places them into bins 1–8 by aerosol size).
* Sedimentation fall speeds ARE level-dependent (oracle ``VR1(K,KR)``);
  the in-step **collision kernel** still uses a fixed warm-cloud
  reference state (the oracle pressure-interpolates 3 kernel tables —
  weak dependence; a later iteration can interpolate).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.fast_sbm.collision import (
    bott_coalescence,
    collision_ck_matrix,
    f_from_g,
    g_from_f,
    precompute_collision_tables,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.condensation_driver import (
    warm_condensation_step,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    KRDROP,
    bin_mass_widths,
    bin_mixing_ratios_from_f,
    discretize_exponential,
    discretize_lognormal,
    f_from_bin_mixing_ratios,
    mass_density,
    mass_doubling_grid,
    mass_doubling_grid_np,
    number_density,
    radius_from_mass,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.freezing import (
    freeze_step,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.nucleation import (
    activate_ccn,
)
from legoesm.thermo import relative_humidity
from legoesm.atmosphere.physics.microphysics.fast_sbm.sedimentation import (
    sediment_bins,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)
from legoesm.atmosphere.physics.microphysics.sdm.kernels import (
    golovin_kernel,
    hall_kernel,
    long_kernel,
    terminal_velocity_cloud_rain_shima,
)
from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Warm fast-SBM column operator: reconstruct the 33-bin liquid "
        "spectrum from (q_c, q_r, N_r), run oracle condensation "
        "(ONECOND1) + Bott collision-coalescence, project back to bulk "
        "tendencies — autoconversion emerges from the resolved spectrum."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "q_c": "kg/kg", "q_r": "kg/kg",
        "N_r": "1/m^3", "p_full": "Pa", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dN_c_dt": "1/m^3/s", "dN_r_dt": "1/m^3/s",
        "precipitation": "kg/m^2/s (surface flux from per-bin settling)",
    },
    "sign_convention": (
        "Condensation: dq_c+dq_r>0 with dq_v=-(dq_c+dq_r) and dT_dt="
        "(L_v/c_pd)(dq_c+dq_r)/dt>0; coalescence moves mass cloud->rain "
        "(dq_c<0, dq_r>0) at fixed total liquid; total water and moist "
        "enthalpy are exactly closed."
    ),
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": (
        "Khain et al. (2004) JAS 61:2963; Shpund et al. (2019) JGR "
        "124:9800; WRF module_mp_fast_sbm.F FAST_SBM/ONECOND1/coal_bott"
    ),
    "idealized_test": (
        "Supersaturated cloudy cell condenses with exact closures; "
        "drizzle-free thin cloud produces ~no rain while a dense cloud "
        "transfers mass to rain through KRDROP (emergent autoconversion); "
        "clear cell is a fixed point; total water invariant to roundoff "
        "per cell."
    ),
}


def _kernel_matrix(masses, config: FastSBMConfig):
    """Collision kernel K(m_i, m_j) [m^3/s] from computed formulations
    (the oracle's YW* tables are file-read; CLAUDE-spec'd substitution —
    see docs/specs/bin_microphysics.md kernel strategy)."""
    r = radius_from_mass(masses)
    ri, rj = r[:, None], r[None, :]
    if config.collision_kernel == "golovin":
        return golovin_kernel(ri, rj, config.golovin_b)
    # Terminal velocities at a reference warm-cloud state (the oracle's
    # pressure interpolation of its tables is the analogue; refine when
    # the sedimentation iteration lands per-level velocities).
    v = terminal_velocity_cloud_rain_shima(
        r, jnp.asarray(1.1), jnp.asarray(9.0e4), jnp.asarray(283.0))
    dv = jnp.abs(v[:, None] - v[None, :])
    if config.collision_kernel == "hall":
        return hall_kernel(ri, rj, dv)
    if config.collision_kernel == "long":
        return long_kernel(ri, rj, dv)
    raise ValueError(
        f"Unknown fast_sbm collision_kernel: {config.collision_kernel!r} "
        "(expected 'hall', 'long', or 'golovin')")


def _reconstruct_spectrum(q_c, q_r, N_c, N_r, rho, masses,
                          config: FastSBMConfig):
    """Liquid spectrum f [m^-3 kg^-1] carrying exactly q_c+q_r of mass.

    Cloud number uses the dycore's PROGNOSTIC ``N_c`` when the column
    carries one (``N_c > 0``); otherwise the prescribed ``config.cdnc``
    closes a single-moment column (codex review item 10 — keeps cloud
    number consistent with whatever the driver feeds back as ``dN_c_dt``).
    """
    # Cloud mode: lognormal at the cloud number, mass-rescaled to q_c·ρ.
    n_c = jnp.where(N_c > 0.0, N_c, config.cdnc)
    m_mean_c = q_c * rho / n_c
    r_med = radius_from_mass(jnp.maximum(m_mean_c, 1.0e-30)) \
        * jnp.exp(-1.5 * jnp.log(config.cloud_geom_std) ** 2)
    f_c = discretize_lognormal(masses, n_c, r_med,
                               config.cloud_geom_std)
    mass_c = mass_density(f_c, masses)
    f_c = f_c * jnp.where(mass_c > 0.0, q_c * rho / jnp.where(
        mass_c > 0.0, mass_c, 1.0), 0.0)
    # Rain mode: exponential with the dycore's N_r (floored).
    n_r = jnp.maximum(N_r, config.n_rain_floor)
    m_mean_r = q_r * rho / n_r
    f_r = discretize_exponential(masses, n_r,
                                 jnp.maximum(m_mean_r, 1.0e-30))
    mass_r = mass_density(f_r, masses)
    f_r = f_r * jnp.where(mass_r > 0.0, q_r * rho / jnp.where(
        mass_r > 0.0, mass_r, 1.0), 0.0)
    return f_c + f_r


def fast_sbm_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: FastSBMConfig = FastSBMConfig(),
) -> MicrophysicsOutput:
    """Fast-SBM warm microphysics over ``(ncol, nlev)`` fields."""
    masses = mass_doubling_grid(dtype=T.dtype)
    # Tables are static grid geometry — precomputed from the NumPy twin so
    # this works inside jit/grad traces (a traced `masses` cannot cross
    # into the host-side table build).
    tables = precompute_collision_tables(mass_doubling_grid_np())
    kernel = _kernel_matrix(masses, config)
    ck = collision_ck_matrix(kernel, dt)
    # Oracle diagnostic split is IF(KRR < KRDROP) with KRDROP=15 (1-based)
    # → 1-based bins 1..14 are cloud, 15.. rain. In 0-based that is bins
    # 0..13 (= KRDROP-1 of them) cloud, 14.. rain (codex review item 6).
    cloud_bins = KRDROP - 1

    q_c = jnp.maximum(hydrometeors.q_c, 0.0)
    q_r = jnp.maximum(hydrometeors.q_r, 0.0)
    q_v_pos = jnp.maximum(q_v, 0.0)

    def cell(T_c, qv_c, qc_c, qr_c, Nc_c, Nr_c, p_c, rho_c):
        # Pre-physics spectrum (the vapor/heat closure baseline).
        f_pre = _reconstruct_spectrum(qc_c, qr_c, Nc_c, Nr_c, rho_c, masses,
                                      config)
        # CCN activation, DEFICIT form (codex review iters 8-9): the
        # diagnosable activated number at this S is the TARGET cloud number
        # N_target = N_CCN·frac_above(r_crit); we seed only the deficit
        # max(0, N_target − N_existing). This is self-limiting in a
        # multi-step column WITHOUT a persistent aerosol reservoir — at a
        # steady cloud N_existing ≈ N_target so nothing re-activates,
        # avoiding the spurious re-activation a `ccn_number − N_existing`
        # reservoir would cause when reconstruction resets the count.
        S_c = relative_humidity(T_c, p_c, qv_c)
        n_existing = number_density(
            jnp.where(jnp.arange(masses.shape[0]) < cloud_bins, f_pre, 0.0),
            masses)
        n_target = activate_ccn(S_c, T_c, jnp.asarray(config.ccn_number),
                                masses, config).n_activated
        n_new = jnp.maximum(n_target - n_existing, 0.0)
        f_seed = f_pre.at[0].add(n_new / bin_mass_widths(masses)[0])
        cond = warm_condensation_step(
            f_seed, T_c, qv_c, p_c, rho_c, dt, masses, config=config)
        g1 = bott_coalescence(g_from_f(cond.f, masses), ck, masses, tables)
        f1 = f_from_g(g1, masses)
        # Immersion freezing (oracle FREEZ): supercooled drops freeze to
        # ice, releasing fusion heat. Liquid feeding sedimentation is the
        # UNFROZEN remainder; the frozen mass becomes q_i. (Ice-spectrum
        # carry-through / ice collision / melting / ice sedimentation are
        # the ice-subsystem iterations; here ice is a diagnostic sink.)
        frz = freeze_step(f1, masses, T_c, rho_c, dt, config)
        f1_liq = frz.f_liquid
        dq_i_dt_c = mass_density(frz.f_ice, masses) / rho_c / dt
        # Total-water closure: vapor change = −(in-cell CONDENSED gain),
        # where condensed = liquid + ice (freezing is liquid↔ice internal,
        # vapor-neutral). mass(f1)=mass(f1_liq)+mass(f_ice) so this is
        # exactly −(mass(f1)−mass(f_pre))/ρ.
        dq_v = -(mass_density(f1, masses) - mass_density(f_pre, masses)) \
            / rho_c
        dq_v_dt = dq_v / dt
        # Heating = condensation (L_v) + fusion (L_f).
        dT_dt_c = -(constants.L_v / constants.c_pd) * dq_v_dt + frz.dT / dt
        return (dT_dt_c, dq_v_dt, dq_i_dt_c, f_pre, f1_liq)

    cell_v = jax.vmap(jax.vmap(cell))
    dT_dt, dqv_dt, dqi_dt, f0, f1 = cell_v(
        T, q_v_pos, q_c, q_r, hydrometeors.N_c, hydrometeors.N_r, p_full, rho)

    # Sedimentation on the per-bin mixing ratios (oracle FALFLUXHUCM_Z;
    # couples levels, so it runs on the (ncol, nlev, n_bins) field).
    # Fall speeds are LEVEL-DEPENDENT (oracle VR1(K,KR)): density/pressure
    # at each level set the bin terminal velocity (codex review iter 8-9).
    r_bins = radius_from_mass(masses)
    v_cell = jax.vmap(jax.vmap(
        lambda rho_c, p_c, T_c: terminal_velocity_cloud_rain_shima(
            r_bins, rho_c, p_c, T_c)))(rho, p_full, T)   # (ncol, nlev, nkr)
    q_bins = bin_mixing_ratios_from_f(f1, masses, rho)
    dq_bins_dt, precip = sediment_bins(
        q_bins, rho, v_cell, dz, dt, config.n_fall_substeps)
    f_final = f_from_bin_mixing_ratios(
        q_bins + dt * dq_bins_dt, masses, rho)

    # Bulk projection across the oracle cloud/rain boundary.
    bin_is_cloud = jnp.arange(masses.shape[0]) < cloud_bins

    def split(f):
        f_c = jnp.where(bin_is_cloud, f, 0.0)
        f_r = jnp.where(~bin_is_cloud, f, 0.0)
        return (mass_density(f_c, masses) / rho,
                mass_density(f_r, masses) / rho,
                number_density(f_c, masses),
                number_density(f_r, masses))

    qc0, qr0, nc0, nr0 = split(f0)
    qc1, qr1, nc1, nr1 = split(f_final)
    inv_dt = 1.0 / dt

    # Freezing depletes the liquid that reaches `split` (f1→f1_liq feeds
    # sedimentation), so dq_c/dq_r already reflect the frozen loss; dq_i is
    # the matching ice gain (liquid+ice conserved, vapor untouched).
    zeros = jnp.zeros_like(T)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dqv_dt,
        dq_c_dt=(qc1 - qc0) * inv_dt,
        dq_r_dt=(qr1 - qr0) * inv_dt,
        dq_i_dt=dqi_dt,
        dq_s_dt=zeros,
        dq_g_dt=zeros,
        dN_c_dt=(nc1 - nc0) * inv_dt,
        dN_r_dt=(nr1 - nr0) * inv_dt,
        dN_i_dt=zeros,
        precipitation=precip,
    )
