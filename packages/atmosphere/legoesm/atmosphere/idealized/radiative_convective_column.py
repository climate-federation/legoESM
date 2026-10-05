"""Multi-layer gray radiative–convective-equilibrium (RCE) column — slab rung 2.

The next atmosphere complexity rung above a 0-D slab (the deleted, never-wired
``atmosphere.slab``): a
single-column, no-dynamics harness that relaxes an ``nlev`` temperature profile to
radiative–convective equilibrium by composing the existing, separately-validated
substrate pieces — it introduces NO new radiation or convection numerics:

  * pressures from a sigma coordinate (``grids.vertical.SigmaCoordinate``);
  * radiative heating from the Frierson two-stream gray scheme
    (``physics.radiation.gray.gray_radiation``);
  * a slab-surface energy balance closing the column at the lower boundary.

This is exactly the harness ``idealized.radiative_equilibrium`` names as
out-of-scope for itself ("run the gray backend in a column-only no-dynamics loop
until heating rates fall below a tolerance").  Kept DRY (``q_v = 0``).

Convective adjustment (to make this a true radiative–*convective* column) is the
documented next step but is NOT yet wired: the existing
``physics.convection.dca`` scheme adjusts toward the *saturated moist* adiabat and
gates by moist CAPE, so it cannot serve as the dry adjustment a dry column needs
(``convective_adjustment`` therefore stays False and raises if set — a correct
dry-adiabatic mode is the follow-up).

The direct equilibrium condition is that the per-level radiative heating rate and
the net surface flux both vanish (``rce_surface_net_flux → 0``,
``heating_rate → 0``); the harness drives the column there.  The net
top-of-atmosphere flux (``rce_toa_imbalance`` = absorbed SW − OLR) then approaches
zero only up to the gray scheme's own flux-closure residual — the discretized
flux-divergence heating rate does not sum *exactly* to the TOA-minus-surface flux
(an O(1) W m⁻² inconsistency intrinsic to ``gray_radiation``, not this harness),
so use the per-level heating / surface flux as the convergence metric.

Everything is pure JAX, so the stepped column and its equilibrium are
``jax.grad``-differentiable (e.g. d(T_sfc_eq)/d(insolation) for sensitivity).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.grids.vertical import SigmaCoordinate, create_sigma_coordinate

from legoesm import constants

__all__ = [
    "RCEColumnConfig",
    "RCEColumnState",
    "make_rce_sigma_coordinate",
    "rce_column_step",
    "rce_surface_net_flux",
    "rce_toa_imbalance",
    "RadiativeConvectiveColumn",
]

_CONVECTION_UNSUPPORTED = (
    "convective_adjustment is not yet supported: the existing "
    "physics.convection.dca scheme adjusts toward the SATURATED MOIST adiabat "
    "(and gates by moist CAPE), so it cannot provide the dry adjustment a dry "
    "(q_v=0) column needs.  A correct dry-adiabatic mode (target γ_dry, dry-CAPE "
    "gate) is the documented follow-up; keep convective_adjustment=False for the "
    "radiative-equilibrium column."
)


class RCEColumnConfig(NamedTuple):
    """Configuration of the gray radiative–convective-equilibrium column."""

    nlev: int = 30
    p_s: float = constants.p_ref          # surface pressure [Pa]
    sigma_top: float = 0.01               # model-top sigma (≈ p_top/p_s)
    lat_deg: float = 0.0                  # column latitude [deg] (LW τ depends on lat)
    c_sfc: float = 50.0 * constants.rho_water * constants.c_pw  # slab-surface heat capacity [J m⁻² K⁻¹]
    # Convective adjustment is NOT yet wired (see rce_column_step): the existing
    # physics.convection.dca scheme targets the *saturated moist* adiabat and gates
    # by moist CAPE, so it cannot serve as the dry adjustment a dry (q_v=0) column
    # needs.  A correct dry-adiabatic mode (target γ_dry = R_d T / (c_pd p), dry-CAPE
    # gate) is the documented follow-up; until then this column is pure radiative
    # equilibrium and ``convective_adjustment`` must stay False.
    convective_adjustment: bool = False
    gray: GrayRadiationConfig = GrayRadiationConfig()


class RCEColumnState(NamedTuple):
    """Prognostic column state: ``T`` (ncol, nlev), ``T_sfc`` (ncol,), dry ``q_v``."""

    T: jnp.ndarray
    T_sfc: jnp.ndarray
    q_v: jnp.ndarray  # specific humidity [kg/kg]; kept 0 for the dry RCE contract


def make_rce_sigma_coordinate(config: RCEColumnConfig) -> SigmaCoordinate:
    return create_sigma_coordinate(config.nlev, sigma_top=config.sigma_top)


def _pressures(state: RCEColumnState, sigma: SigmaCoordinate, config: RCEColumnConfig):
    ncol = state.T.shape[0]
    p_s = jnp.full((ncol,), config.p_s, dtype=state.T.dtype)
    p_full = sigma.pressure_at_full(p_s)   # (ncol, nlev)
    p_half = sigma.pressure_at_half(p_s)   # (ncol, nlev+1)
    return p_full, p_half


def _radiation(state: RCEColumnState, insolation, sigma: SigmaCoordinate,
               config: RCEColumnConfig):
    p_full, p_half = _pressures(state, sigma, config)
    ncol = state.T.shape[0]
    lat = jnp.full((ncol,), jnp.deg2rad(config.lat_deg), dtype=state.T.dtype)
    insol = jnp.broadcast_to(jnp.asarray(insolation, dtype=state.T.dtype), (ncol,))
    rad = gray_radiation(state.T, p_full, p_half, state.T_sfc, lat,
                         None, insol, config.gray)
    return rad, p_full, p_half


def rce_surface_net_flux(rad, config: RCEColumnConfig) -> jnp.ndarray:
    """Net downward radiative flux into the surface [W m⁻²] (interface index -1)."""
    return (rad.sw_flux_down[:, -1] - rad.sw_flux_up[:, -1]
            + rad.lw_flux_down[:, -1] - rad.lw_flux_up[:, -1])


def rce_toa_imbalance(state: RCEColumnState, insolation, config: RCEColumnConfig,
                      sigma: SigmaCoordinate | None = None) -> jnp.ndarray:
    """Net downward flux at the top of atmosphere [W m⁻²] = absorbed SW − OLR."""
    sigma = sigma if sigma is not None else make_rce_sigma_coordinate(config)
    rad, _, _ = _radiation(state, insolation, sigma, config)
    return (rad.sw_flux_down[:, 0] - rad.sw_flux_up[:, 0]
            + rad.lw_flux_down[:, 0] - rad.lw_flux_up[:, 0])


def rce_column_step(state: RCEColumnState, dt: float, insolation,
                    sigma: SigmaCoordinate, config: RCEColumnConfig) -> RCEColumnState:
    """Advance one ``dt``: radiative heating + slab-surface energy balance.

    (Convective adjustment is deferred — see ``RCEColumnConfig`` — so this is the
    pure gray radiative-equilibrium step.)
    """
    if config.convective_adjustment:
        raise NotImplementedError(_CONVECTION_UNSUPPORTED)
    rad, p_full, p_half = _radiation(state, insolation, sigma, config)
    T = state.T + dt * rad.heating_rate
    net_sfc = rce_surface_net_flux(rad, config)
    T_sfc = state.T_sfc + dt * net_sfc / config.c_sfc
    return RCEColumnState(T=T, T_sfc=T_sfc, q_v=state.q_v)


class RadiativeConvectiveColumn:
    """Standalone gray-RCE column harness (no horizontal grid, no dynamics)."""

    def __init__(self, config: RCEColumnConfig | None = None, *, dt: float = 1800.0):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.config = config if config is not None else RCEColumnConfig()
        if self.config.nlev < 1:
            raise ValueError(f"nlev must be >= 1, got {self.config.nlev}")
        if self.config.c_sfc <= 0:
            raise ValueError(f"c_sfc must be > 0, got {self.config.c_sfc}")
        if self.config.convective_adjustment:
            raise NotImplementedError(_CONVECTION_UNSUPPORTED)  # fail fast
        self.dt = float(dt)
        self.sigma = make_rce_sigma_coordinate(self.config)

    def initial_state(self, *, T0: float = 250.0, T_sfc0: float = 288.0,
                      ncol: int = 1) -> RCEColumnState:
        """Isothermal dry initial column at ``T0`` with surface ``T_sfc0``."""
        nlev = self.config.nlev
        return RCEColumnState(
            T=jnp.full((ncol, nlev), T0),
            T_sfc=jnp.full((ncol,), T_sfc0),
            q_v=jnp.zeros((ncol, nlev)),
        )

    def step(self, state: RCEColumnState, insolation) -> RCEColumnState:
        return rce_column_step(state, self.dt, insolation, self.sigma, self.config)

    def run(self, state: RCEColumnState, insolation, *, nsteps: int,
            save_every: int = 1) -> tuple[RCEColumnState, list[RCEColumnState]]:
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        history: list[RCEColumnState] = []
        for n in range(nsteps):
            state = self.step(state, insolation)
            if (n + 1) % save_every == 0:
                history.append(state)
        return state, history

    def toa_imbalance(self, state: RCEColumnState, insolation) -> jnp.ndarray:
        return rce_toa_imbalance(state, insolation, self.config, self.sigma)
