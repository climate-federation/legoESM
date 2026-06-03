"""Single-layer gray slab atmosphere — the cheapest atmosphere complexity rung.

A 0-D energy-balance model (EBM): one bulk atmospheric temperature ``T_atm``
exchanging gray longwave + shortwave radiation with one slab surface temperature
``T_sfc``.  It is the classic single-layer-greenhouse model (e.g. Hartmann,
*Global Physical Climatology*, §3) — the atmosphere is transparent to a fraction
of the shortwave and a gray (emissivity ``ε``) absorber/emitter in the longwave.

This is the column-extent (``extent="column"``) atmosphere brick: it carries no
horizontal grid and no dynamics, just the two coupled column energy budgets.  It
sits *below* the dynamical-core complexity ladder (shallow_water → hydrostatic →
nonhydrostatic), which is why it is its own model rather than an
``AtmosphereComplexity`` rung.  The next rung up — a multi-layer gray
radiative–convective-equilibrium *column* — lives in
``atmosphere.idealized.radiative_convective_column`` and reuses the full
``physics.radiation.gray`` engine.

Energy budgets (per unit area, W m⁻²), with ``S`` the TOA insolation, ``α`` the
planetary albedo, ``a`` the atmospheric shortwave absorptivity, ``ε`` the
atmospheric longwave emissivity (surface emissivity taken as 1), and ``H`` an
optional sensible-heat exchange ``H = k·(T_sfc − T_atm)``::

    C_sfc dT_sfc/dt = (1−α)(1−a)S + ε σ T_atm⁴ − σ T_sfc⁴ − H
    C_atm dT_atm/dt = (1−α)  a  S + ε σ T_sfc⁴ − 2 ε σ T_atm⁴ + H

The two budgets sum to ``(1−α)S − OLR`` with ``OLR = (1−ε) σ T_sfc⁴ + ε σ T_atm⁴``
(the turbulent term ``H`` cancels), so global energy is conserved exactly and the
equilibrium satisfies the TOA balance ``absorbed SW = OLR``.  With ``H = 0`` the
radiative equilibrium has the closed form in :func:`slab_equilibrium`; in the
classic limit ``a = 0, ε = 1`` it reduces to ``σ T_sfc⁴ = 2 (1−α) S = 2 σ T_eff⁴``.

Everything is pure JAX (no control flow on traced values), so the tendencies, a
stepped run, and the analytic equilibrium are all ``jax.grad``-differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

__all__ = [
    "SlabAtmosphereConfig",
    "SlabAtmosphereState",
    "slab_tendency",
    "slab_step",
    "slab_equilibrium",
    "toa_imbalance",
    "outgoing_longwave",
    "SlabAtmosphereModel",
]


class SlabAtmosphereConfig(NamedTuple):
    """Parameters of the single-layer gray slab atmosphere.

    The heat capacities are *model* parameters (set by the chosen slab depths),
    not universal constants: ``c_atm`` defaults to a dry air column ``c_pd p_ref/g``
    and ``c_sfc`` to a 50 m ocean mixed layer (depth × water volumetric heat
    capacity 4.18×10⁶ J m⁻³ K⁻¹).
    """

    emissivity: float = 0.8          # atmospheric gray LW emissivity ε ∈ (0, 1]
    albedo: float = 0.30             # planetary shortwave albedo α ∈ [0, 1)
    sw_atm_absorption: float = 0.0   # fraction a ∈ [0, 1] of (1−α)S absorbed in the air
    # Column heat capacities [J m⁻² K⁻¹] (slab depths are the model choice):
    c_atm: float = constants.c_pd * constants.p_ref / constants.g   # dry air column
    c_sfc: float = 50.0 * 4.18e6                                    # 50 m ocean mixed layer
    # H = sensible_coeff·(T_sfc − T_atm) [W m⁻² K⁻¹]; 0 → radiative-only (analytic eq.)
    sensible_coeff: float = 0.0


class SlabAtmosphereState(NamedTuple):
    """Prognostic temperatures [K].  Scalars or ``(ncol,)`` arrays (vectorized)."""

    T_atm: jnp.ndarray
    T_sfc: jnp.ndarray


def _radiative_terms(state: SlabAtmosphereState, insolation, config: SlabAtmosphereConfig):
    sig = constants.sigma_sb
    sw_total = (1.0 - config.albedo) * insolation
    sw_sfc = (1.0 - config.sw_atm_absorption) * sw_total
    sw_atm = config.sw_atm_absorption * sw_total
    lw_atm_emit = config.emissivity * sig * state.T_atm ** 4   # emitted up AND down
    lw_sfc_emit = sig * state.T_sfc ** 4                       # surface emission (ε_sfc = 1)
    lw_sfc_absorbed_by_atm = config.emissivity * lw_sfc_emit   # fraction ε of upward beam
    return sw_sfc, sw_atm, lw_atm_emit, lw_sfc_emit, lw_sfc_absorbed_by_atm


def slab_tendency(state: SlabAtmosphereState, insolation, config: SlabAtmosphereConfig):
    """Return ``(dT_atm_dt, dT_sfc_dt)`` [K s⁻¹] from the two energy budgets."""
    sw_sfc, sw_atm, lw_atm_emit, lw_sfc_emit, lw_sfc_abs = _radiative_terms(
        state, insolation, config)
    sensible = config.sensible_coeff * (state.T_sfc - state.T_atm)  # surface → atmosphere
    net_sfc = sw_sfc + lw_atm_emit - lw_sfc_emit - sensible
    net_atm = sw_atm + lw_sfc_abs - 2.0 * lw_atm_emit + sensible
    return net_atm / config.c_atm, net_sfc / config.c_sfc


def slab_step(state: SlabAtmosphereState, dt: float, insolation,
              config: SlabAtmosphereConfig) -> SlabAtmosphereState:
    """One explicit-Euler step of the coupled budgets."""
    dT_atm_dt, dT_sfc_dt = slab_tendency(state, insolation, config)
    return SlabAtmosphereState(
        T_atm=state.T_atm + dt * dT_atm_dt,
        T_sfc=state.T_sfc + dt * dT_sfc_dt,
    )


def outgoing_longwave(state: SlabAtmosphereState, config: SlabAtmosphereConfig):
    """TOA outgoing longwave radiation [W m⁻²]: ``(1−ε) σ T_sfc⁴ + ε σ T_atm⁴``."""
    sig = constants.sigma_sb
    return ((1.0 - config.emissivity) * sig * state.T_sfc ** 4
            + config.emissivity * sig * state.T_atm ** 4)


def toa_imbalance(state: SlabAtmosphereState, insolation, config: SlabAtmosphereConfig):
    """Net top-of-atmosphere flux [W m⁻²]: ``(1−α) S − OLR`` (→ 0 at equilibrium)."""
    return (1.0 - config.albedo) * insolation - outgoing_longwave(state, config)


def slab_equilibrium(insolation, config: SlabAtmosphereConfig) -> SlabAtmosphereState:
    """Closed-form RADIATIVE equilibrium (``sensible_coeff`` taken as 0).

    Solving the two budgets with ``H = 0``::

        σ T_atm⁴ = (1−α) S [ (1−a) + a/ε ] / (2 − ε)
        σ T_sfc⁴ = (1−α)(1−a) S + ε σ T_atm⁴

    For ``sensible_coeff ≠ 0`` use :class:`SlabAtmosphereModel` / :func:`slab_step`
    and integrate to a numerical steady state.
    """
    sig = constants.sigma_sb
    a = config.sw_atm_absorption
    eps = config.emissivity
    sw = (1.0 - config.albedo) * insolation
    sigT_atm4 = sw * ((1.0 - a) + a / eps) / (2.0 - eps)
    sigT_sfc4 = (1.0 - a) * sw + eps * sigT_atm4
    return SlabAtmosphereState(
        T_atm=(sigT_atm4 / sig) ** 0.25,
        T_sfc=(sigT_sfc4 / sig) ** 0.25,
    )


class SlabAtmosphereModel:
    """Standalone single-column slab-atmosphere harness (no grid, no dynamics).

    Mirrors the other single-column bricks (``SingleColumnModel``,
    ``OceanColumnModel``, ``LandColumnModel``): ``step`` advances one ``dt`` and
    ``run`` loops, sharing one elapsed clock.  ``insolation`` is a constant
    [W m⁻²] or a callable ``elapsed_seconds -> insolation`` (so diurnal/seasonal
    forcing can be supplied by the caller without recompilation).
    """

    def __init__(self, config: SlabAtmosphereConfig | None = None, *, dt: float = 3600.0):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.config = config if config is not None else SlabAtmosphereConfig()
        self.dt = float(dt)

    def _insolation_at(self, insolation, elapsed: float):
        return insolation(elapsed) if callable(insolation) else insolation

    def step(self, state: SlabAtmosphereState, insolation, *,
             elapsed: float = 0.0) -> SlabAtmosphereState:
        return slab_step(state, self.dt, self._insolation_at(insolation, elapsed), self.config)

    def run(self, state: SlabAtmosphereState, insolation, *, nsteps: int,
            save_every: int = 1) -> tuple[SlabAtmosphereState, list[SlabAtmosphereState]]:
        """Integrate ``nsteps`` steps; return ``(final_state, history)``."""
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        history: list[SlabAtmosphereState] = []
        elapsed = 0.0
        for n in range(nsteps):
            state = self.step(state, insolation, elapsed=elapsed)
            elapsed += self.dt
            if (n + 1) % save_every == 0:
                history.append(state)
        return state, history

    def equilibrium(self, insolation) -> SlabAtmosphereState:
        """Closed-form radiative equilibrium (constant ``insolation``)."""
        return slab_equilibrium(self._insolation_at(insolation, 0.0), self.config)
