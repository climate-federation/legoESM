"""Optional prognostic bulk-aerosol tracer (accumulation-mode number budget).

An OPT-IN (default OFF) prognostic aerosol representation that augments the
prescribed-AOD proxy: a single-mode accumulation aerosol NUMBER concentration
``N_a`` [1/m^3] carried in the atmospheric physics state, evolved by local
sources and sinks.  When enabled, the prognostic number feeds the ARG modal
activation (:mod:`arg_activation`) INSTEAD of the prescribed AOD, closing the
emission -> aerosol -> activation -> cloud-droplet chain prognostically.

Budget (per grid cell, number concentration ``N`` [1/m^3]):

    dN/dt = S - L * N

with the tendency split into a SOURCE ``S`` [1/m^3/s] (>= 0) and a linear SINK
RATE ``L`` [1/s] (>= 0):

  SOURCES (S, sign +):
    * surface emission of a prescribed number flux ``E`` [1/m^2/s] into the
      bottom layer:  S_emis = E / dz_surface  (bottom level only);
    * SO2 -> sulfate oxidation proxy: a prescribed precursor number field
      ``so2`` [1/m^3] converts to new particles with e-folding ``tau_ox``:
      S_ox = so2 / tau_ox  (all levels).
  SINKS (L, sign -):
    * dry deposition at the surface layer: L_dry = v_dep / dz_surface
      (bottom level only), ``v_dep`` [m/s];
    * wet scavenging proportional to the microphysics precipitation rate
      ``P`` [kg/m^2/s]:  L_wet = k_scav * P  (all levels), ``k_scav`` [m^2/kg].

**Vertical convention**: the last array axis is the vertical, index ``-1`` is
the surface-adjacent (bottom) layer, index ``0`` the model top — matching the
column physics elsewhere (``T_col[:, -1]`` is the surface).

**Positivity**: the sink is LINEAR in ``N``, so the integrator
(:func:`step_prognostic_aerosol`) treats it implicitly (backward Euler):

    N_new = (N + dt * S) / (1 + dt * L)  >= 0

The source ``S`` and sink rate ``L`` are CLAMPED to be non-negative inside
:func:`_sources_and_sink_rate` (emission/oxidation are physical sources, dry
deposition/scavenging physical sinks), so the update is unconditionally
non-negative for ``N >= 0`` and ``dt > 0`` REGARDLESS of the config values —
the sink can never drive the number negative in one step, at any step size, and
``1 + dt*L`` never vanishes.  The budget closes exactly:
``N_new - N = dt (S - L N_new)``.

**Static config (feature gate)**: ``config`` is a Python ``NamedTuple`` treated
as a compile-time constant (closed over / static), so the ``if config.enabled``
short-circuit in :func:`step_prognostic_aerosol` is a static feature gate (the
``fix_mass``/``fix_moisture`` pattern), NOT data-dependent control flow on a
traced array.  Do not pass ``config`` as a traced ``jit`` argument.

The full 3-D advective transport of the tracer through the dycore is OUT OF
SCOPE for this pass; :func:`step_prognostic_aerosol` is a pure, tested LOCAL
source/sink integrator meant to be called from the physics pipeline (see the
module wiring note in ``physics_state.py`` / the driver integration hook).

Reference
---------
Abdul-Razzak & Ghan (2000), JGR 105(D5), 6837-6844 (activation feed).
Bulk single-moment number budget after e.g. the sulfate-mass approach of
Liu et al. (2012, MAM3, GMD 5, 709-739) reduced to a number-only tracer.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Optional prognostic accumulation-mode aerosol NUMBER budget: local "
        "sources (surface emission, SO2->sulfate oxidation) minus linear sinks "
        "(surface dry deposition, precipitation wet scavenging)."
    ),
    "inputs": {
        "aerosol_number": "1/m^3 (prognostic aerosol number, >= 0)",
        "dz": "m (layer thickness)",
        "precip_rate": "kg/m^2/s (surface precipitation from microphysics)",
        "emission_flux": "1/m^2/s (surface number emission, optional field)",
        "so2_precursor": "1/m^3 (SO2 oxidation precursor number, optional)",
    },
    "outputs": {
        "aerosol_number_tendency": "1/m^3/s (dN/dt = source - sink_rate*N)",
    },
    "sign_convention": (
        "emission and oxidation are SOURCES (+, increase N); dry deposition "
        "and wet scavenging are SINKS (-, decrease N); N stays >= 0 and the "
        "implicit sink cannot drive N negative in one step. Bottom layer = "
        "index -1 (surface); emission/dry-dep act there only."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Abdul-Razzak & Ghan (2000), JGR 105(D5), 6837-6844 (activation feed); "
        "single-moment number budget after Liu et al. (2012), GMD 5, 709-739"
    ),
    "idealized_test": (
        "emission-only -> N increases; sink-only (dry dep + scavenging) -> N "
        "decreases but stays >= 0 for any dt; enabled=False leaves the state "
        "unchanged"
    ),
}

__param_spec__ = {
    "PrognosticAerosolConfig": {
        "scheme_key": "atm.aerosol.PrognosticAerosolConfig",
        "excluded": {},
        "params": {
            "emission_number_flux_m2_s": {
                "units": "1/(m^2 s)", "bounds": (1.0e6, 1.0e10),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "aerosol_source",
                "reference": "accumulation-mode surface number emission flux",
                "shape": None,
            },
            "dry_dep_velocity_m_s": {
                "units": "m/s", "bounds": (1.0e-4, 5.0e-2),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "aerosol_sink",
                "reference": "aerosol dry-deposition velocity (Slinn 1982)",
                "shape": None,
            },
            "so2_oxidation_timescale_s": {
                "units": "s", "bounds": (3.6e3, 8.64e5),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "aerosol_source",
                "reference": "SO2->sulfate oxidation e-folding (~1 day)",
                "shape": None,
            },
            "wet_scavenging_coeff_m2_kg": {
                "units": "m^2/kg", "bounds": (1.0e-2, 1.0e2),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "aerosol_sink",
                "reference": "precipitation wet-scavenging coefficient",
                "shape": None,
            },
            "mode_r_g_um": {
                "units": "um", "bounds": (1.0e-2, 5.0e-1),
                "tunable_tier": 3, "transform": "sigmoid",
                "category": "aerosol_activation",
                "reference": "prognostic accumulation-mode geometric radius",
                "shape": None,
            },
            "mode_sigma_g": {
                "units": "1", "bounds": (1.2, 2.5),
                "tunable_tier": 3, "transform": "sigmoid",
                "category": "aerosol_activation",
                "reference": "prognostic accumulation-mode geometric std",
                "shape": None,
            },
            "mode_kappa": {
                "units": "1", "bounds": (1.0e-2, 1.2),
                "tunable_tier": 3, "transform": "sigmoid",
                "category": "aerosol_activation",
                "reference": "prognostic accumulation-mode hygroscopicity",
                "shape": None,
            },
        },
    },
}

# --- AD / numerics floors -------------------------------------------------
_DZ_FLOOR_M = 1.0               # layer-thickness floor [m] (E/dz, v_dep/dz)
_DT_FLOOR_S = 1.0e-6            # timestep floor [s] guarding 1/(1+dt L)
_TAU_OX_FLOOR_S = 1.0e-3        # oxidation-timescale floor [s] guarding 1/tau_ox
_NUMBER_FLOOR_M3 = 0.0          # aerosol number is non-negative


class PrognosticAerosolConfig(NamedTuple):
    """Configuration for the optional prognostic aerosol-number tracer.

    Fields
    ------
    enabled : bool
        Master switch (default ``False`` = prescribed-AOD proxy path
        unchanged, existing runs byte-identical).
    emission_number_flux_m2_s : float
        Surface aerosol NUMBER emission flux [1/m^2/s] (SOURCE, bottom layer).
    dry_dep_velocity_m_s : float
        Aerosol dry-deposition velocity [m/s] (SINK, bottom layer).
    so2_oxidation_timescale_s : float
        SO2 -> sulfate oxidation e-folding [s] applied to a precursor field.
    wet_scavenging_coeff_m2_kg : float
        Wet-scavenging coefficient [m^2/kg] on the precip rate [kg/m^2/s]
        (SINK, all levels).
    mode_r_g_um, mode_sigma_g, mode_kappa : float
        Assumed lognormal shape of the prognostic accumulation mode used when
        the prognostic number drives ARG activation: geometric-mean dry radius
        [um], geometric std [-] and kappa-Koehler hygroscopicity [-].
    """

    enabled: bool = False
    emission_number_flux_m2_s: float = 1.0e8
    dry_dep_velocity_m_s: float = 1.0e-3
    so2_oxidation_timescale_s: float = 8.64e4
    wet_scavenging_coeff_m2_kg: float = 1.0
    mode_r_g_um: float = 0.05
    mode_sigma_g: float = 2.0
    mode_kappa: float = 0.6


def _sources_and_sink_rate(
    dz: jnp.ndarray,
    precip_rate: jnp.ndarray,
    config: PrognosticAerosolConfig,
    *,
    emission_flux: jnp.ndarray | None = None,
    so2_precursor: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(source, sink_rate)`` = ``(S [1/m^3/s], L [1/s])``.

    Shared numerics for the explicit tendency and the implicit integrator so
    the source/sink algebra is written once.  ``dz`` sets the spatial shape
    ``(..., nlev)``; ``precip_rate`` is the per-column surface rate ``(...,)``.
    """
    dz_safe = jnp.maximum(dz, _DZ_FLOOR_M)
    dz_surface = dz_safe[..., -1]                       # (...,) bottom layer

    # --- SOURCES (sign +) --------------------------------------------------
    # Surface emission: a number flux [1/m^2/s] deposited into the bottom
    # layer only -> volumetric rate E/dz_surface [1/m^3/s].
    e_flux = (config.emission_number_flux_m2_s if emission_flux is None
              else emission_flux)
    e_flux = jnp.broadcast_to(jnp.asarray(e_flux, dtype=dz.dtype),
                              dz_surface.shape)
    emission_rate_bottom = e_flux / dz_surface          # (...,)
    source = jnp.zeros_like(dz_safe)
    source = source.at[..., -1].add(emission_rate_bottom)

    # SO2 -> sulfate oxidation: a precursor number field decays to new
    # particles with e-folding tau_ox (all levels). Absent precursor -> 0.
    if so2_precursor is not None:
        so2 = jnp.clip(jnp.asarray(so2_precursor, dtype=dz.dtype), 0.0, None)
        # Floor tau_ox so a zero/garbage oxidation timescale cannot inject a
        # NaN/inf that the downstream non-negativity clip could not repair.
        tau_ox = jnp.maximum(config.so2_oxidation_timescale_s, _TAU_OX_FLOOR_S)
        source = source + so2 / tau_ox

    # --- SINK RATE (sign -, linear in N) -----------------------------------
    # Wet scavenging: proportional to the surface precip rate, applied at all
    # levels (rain sweeps the column). L_wet = k_scav * P [1/s].
    precip = jnp.clip(jnp.asarray(precip_rate, dtype=dz.dtype), 0.0, None)
    sink_rate = (config.wet_scavenging_coeff_m2_kg
                 * precip[..., None] * jnp.ones_like(dz_safe))
    # Dry deposition: surface layer only, L_dry = v_dep / dz_surface [1/s].
    dry_rate_bottom = config.dry_dep_velocity_m_s / dz_surface
    sink_rate = sink_rate.at[..., -1].add(dry_rate_bottom)

    # Non-negativity clamp: emission/oxidation are PHYSICAL sources (>= 0) and
    # deposition/scavenging PHYSICAL sinks (>= 0). Clamping both makes the
    # implicit update positivity-preserving for ANY config, precip or dt — a
    # negative/garbage emission flux or coefficient can never drive S < 0 (=> a
    # negative injection) nor L < 0 (=> ``1 + dt*L <= 0`` division/overshoot).
    source = jnp.clip(source, _NUMBER_FLOOR_M3, None)
    sink_rate = jnp.clip(sink_rate, 0.0, None)
    return source, sink_rate


def aerosol_number_tendency(
    aerosol_number: jnp.ndarray,
    dz: jnp.ndarray,
    precip_rate: jnp.ndarray,
    config: PrognosticAerosolConfig,
    *,
    emission_flux: jnp.ndarray | None = None,
    so2_precursor: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Explicit local tendency ``dN/dt = S - L*N`` [1/m^3/s].

    SOURCES minus SINKS, sign convention documented in the module docstring:
    emission + oxidation are positive; dry deposition + wet scavenging are
    negative (linear in ``N``).  For time stepping use
    :func:`step_prognostic_aerosol` (positivity-preserving); this explicit form
    is for diagnostics and coupling.
    """
    n = jnp.clip(aerosol_number, _NUMBER_FLOOR_M3, None)
    source, sink_rate = _sources_and_sink_rate(
        dz, precip_rate, config,
        emission_flux=emission_flux, so2_precursor=so2_precursor)
    return source - sink_rate * n


def step_prognostic_aerosol(
    aerosol_number: jnp.ndarray,
    dt: float | jnp.ndarray,
    dz: jnp.ndarray,
    precip_rate: jnp.ndarray,
    config: PrognosticAerosolConfig,
    *,
    emission_flux: jnp.ndarray | None = None,
    so2_precursor: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Advance the aerosol number one step, positivity-preserving [1/m^3].

    Backward-Euler on the LINEAR sink so the removal can never overshoot:

        N_new = (N + dt*S) / (1 + dt*L)   >= 0.

    When ``config.enabled`` is ``False`` the state is returned UNCHANGED (the
    prescribed-AOD proxy path is untouched — existing runs byte-identical).
    """
    if not config.enabled:
        return aerosol_number
    n = jnp.clip(aerosol_number, _NUMBER_FLOOR_M3, None)
    dt_safe = jnp.maximum(jnp.asarray(dt, dtype=n.dtype), _DT_FLOOR_S)
    source, sink_rate = _sources_and_sink_rate(
        dz, precip_rate, config,
        emission_flux=emission_flux, so2_precursor=so2_precursor)
    # Implicit linear-sink update: unconditionally non-negative.
    return (n + dt_safe * source) / (1.0 + dt_safe * sink_rate)
