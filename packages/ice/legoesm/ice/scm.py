"""Single-column (standalone) driver for the legoESM sea-ice thermodynamics.

The sea-ice counterpart of :class:`legoesm.land.scm.LandColumnModel` and the
atmosphere / ocean SCMs: a minimal driver that runs the ice *brick* on its own —
init state, prescribed atmospheric + ocean forcing, time loop, history — so the
ice model can be exercised **independently of the coupler** (thermodynamic
growth/melt studies, surface-flux and albedo sweeps), using the *same*
``ice.sea_ice.step_sea_ice`` the coupled driver calls.

Scope: ice *dynamics* (free-drift / EVP rheology, ridging, transport) act on a 2-D
grid of neighbouring cells, so they are not a single-column concept; this driver
runs the **thermodynamic** slab path (``SeaIceConfig.dynamics="none"``, ``grid=None``)
on ``ncol`` independent columns.

The atmospheric *forcing* (an ``AtmToSurface`` — now a core pytree in
``legoesm.core.coupling_fields`` — or a ``callable(elapsed_seconds) ->
AtmToSurface``) is duck-typed so a caller may pass a constant field or a
time-varying closure; the ocean coupling (SST + surface currents) is prescribed
as plain arrays.  This module imports no ``coupler`` (boundary-clean).

Example
-------
>>> from legoesm.ice.scm import IceColumnModel, cold_polar_forcing
>>> scm = IceColumnModel.create(ncol=1, dt=3600.0, forcing=cold_polar_forcing(ncol=1))
>>> final_state, history = scm.run(nsteps=240, save_every=24)  # 10-day growth
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface


def cold_polar_forcing(
    ncol: int = 1,
    *,
    sw_down: float = 0.0,
    lw_down: float = 200.0,  # coeff-ok: polar-night idealized forcing [W/m^2]
    T_lowest: float = 250.0,  # noqa: N803 (physical symbol) — coeff-ok: idealized forcing [K]
    u_lowest: float = 6.0,
) -> AtmToSurface:
    """A cold, dark (polar-night) :class:`AtmToSurface` over ``ncol`` ice cells.

    A convenience for thermodynamic ice growth/melt studies; pass any
    ``AtmToSurface`` (or a ``callable(elapsed) -> AtmToSurface``) to
    :meth:`IceColumnModel.create` for full control.  Lives here because
    ``AtmToSurface`` is now a core pytree any component may import.
    """

    def full(v: float) -> jax.Array:
        return jnp.full((ncol,), v)

    return AtmToSurface(
        sw_down=full(sw_down), lw_down=full(lw_down), precip_total=full(0.0),
        precip_snow=full(0.0), T_lowest=full(T_lowest), q_lowest=full(1e-3),  # coeff-ok: idealized polar-night forcing
        u_lowest=full(u_lowest), v_lowest=full(0.0), p_lowest=full(9.7e4),  # coeff-ok: idealized polar-night forcing
        p_surface=full(1.0e5), rho_lowest=full(1.3), cos_zenith=full(0.0),  # coeff-ok: idealized polar-night forcing
        co2_ppmv=jnp.asarray(400.0), has_radiation=jnp.asarray(1.0),  # coeff-ok: idealized forcing
        has_precipitation=jnp.asarray(1.0),
    )


class IceColumnHistory(NamedTuple):
    """Saved time series from a :meth:`IceColumnModel.run`."""

    time: jax.Array  # (nsaved,) elapsed seconds
    h_ice: jax.Array  # (nsaved, ncol) ice thickness [m]
    concentration: jax.Array  # (nsaved, ncol) ice areal fraction [0-1]
    lhflx: jax.Array  # (nsaved, ncol) latent heat flux [W/m2, +up]


class IceColumnModel:
    """Standalone single-column thermodynamic sea-ice driver."""

    def __init__(
        self, *, state, config, forcing, ocean_sst, ocean_u, ocean_v, dt, u_min,
    ):
        self.state = state
        self.config = config
        self._forcing = forcing
        self._ocean_sst = ocean_sst
        self._ocean_u = ocean_u
        self._ocean_v = ocean_v
        self.dt = dt
        self._u_min = u_min
        # Shared elapsed-time clock (step() and run() agree); see LandColumnModel.
        self.elapsed = 0.0

    @classmethod
    def create(
        cls,
        *,
        forcing,
        config=None,
        ncol: int = 1,
        dt: float = 3600.0,
        h_ice_init: float = 0.5,
        concentration_init: float = 0.9,  # coeff-ok: SCM initial condition
        T_ice_init: float = 260.0,  # noqa: N803 (physical symbol) — coeff-ok: SCM initial condition [K]
        ocean_sst: float = constants.T_freeze_ocean,
        u_min: float = 1.0,
        h_snow_init: float = 0.0,
        S_ice_init: float = constants.S_ice_bulk_default,
    ) -> IceColumnModel:
        """Build a thermodynamic ice column for ``ncol`` independent cells.

        ``ocean_sst`` is the prescribed under-ice ocean temperature (default the
        ocean freezing point); ocean currents are zero.  ``config`` must keep
        ``dynamics="none"`` (the column has no neighbours for rheology).
        """
        from legoesm.core.field import Field
        from legoesm.ice import SeaIceConfig
        from legoesm.ice.state import SeaIceState, SI3ColumnState
        from legoesm.ice.config import validate_si3_thermo_config
        from legoesm.ice.bitz_lipscomb import (
            ice_enthalpy_from_temperature,
            option2_salinity_profile,
            snow_enthalpy_from_temperature,
        )

        if config is None:
            config = SeaIceConfig()
        if config.dynamics != "none":
            raise ValueError(
                "IceColumnModel runs the thermodynamic slab path only "
                f"(dynamics='none'); got dynamics={config.dynamics!r}.  Ice "
                "dynamics/rheology act on a 2-D grid, not a single column."
            )

        dims = ("column",)
        if config.thermo_scheme == "si3_bl99":
            validate_si3_thermo_config(config)
            c = config.ice_constants
            bulk = jnp.full((ncol,), S_ice_init)
            sss = jnp.full((ncol,), c.S_ocean_ref)
            sal = option2_salinity_profile(bulk, sss)
            Ti = jnp.full((ncol, 3), T_ice_init)
            Ts = jnp.full((ncol, 3), T_ice_init)
            state = SI3ColumnState(
                concentration=Field(jnp.full((ncol,), concentration_init), name="ice_concentration", dims=dims, units="1"),
                h_ice=Field(jnp.full((ncol,), h_ice_init), name="h_ice", dims=dims, units="m"),
                h_snow=Field(jnp.full((ncol,), h_snow_init), name="h_snow", dims=dims, units="m"),
                T_surface=Field(jnp.full((ncol,), T_ice_init), name="ice_surface_temperature", dims=dims, units="K"),
                e_ice=Field(ice_enthalpy_from_temperature(Ti, sal, c), name="ice_layer_enthalpy", dims=dims + ("ice_layer",), units="J/m3"),
                e_snow=Field(snow_enthalpy_from_temperature(Ts, c), name="snow_layer_enthalpy", dims=dims + ("snow_layer",), units="J/m3"),
                S_bulk=Field(bulk, name="bulk_ice_salinity", dims=dims, units="PSU"),
                S_layers=Field(sal, name="ice_layer_salinity", dims=dims + ("ice_layer",), units="PSU"),
                age_volume=Field(jnp.zeros((ncol,)), name="ice_age_volume", dims=dims, units="s"),
            )
        else:
            state = SeaIceState(
                h_ice=Field(jnp.full((ncol,), h_ice_init), name="h_ice", dims=dims, units="m"),
                T_ice=Field(jnp.full((ncol,), T_ice_init), name="T_ice", dims=dims, units="K"),
                concentration=Field(
                    jnp.full((ncol,), concentration_init),
                    name="ice_concentration", dims=dims, units="1",
                ),
            )
        zeros = jnp.zeros((ncol,))
        return cls(
            state=state, config=config, forcing=forcing,
            ocean_sst=jnp.full((ncol,), ocean_sst), ocean_u=zeros, ocean_v=zeros,
            dt=dt, u_min=u_min,
        )

    def _forcing_at(self, t: float):
        """The atmospheric forcing (an AtmToSurface) at elapsed time *t*."""
        f = self._forcing
        return f(t) if callable(f) else f

    def step(self) -> object:
        """Advance one step under the forcing at the CURRENT elapsed time, then
        advance the clock; returns the per-step ``TileResponse``."""
        from legoesm.ice import step_sea_ice

        forcing = self._forcing_at(self.elapsed)
        new_state, response = step_sea_ice(
            self.state, forcing, self._ocean_sst, self._ocean_u, self._ocean_v,
            self.config, self._u_min, self.dt, grid=None,
        )
        self.state = new_state
        self.elapsed += self.dt
        return response

    def run(
        self, nsteps: int, *, save_every: int = 1
    ) -> tuple[object, IceColumnHistory]:
        """Integrate ``nsteps`` (looping :meth:`step`) -> ``(final_state, history)``."""
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        times, h_hist, c_hist, lhf = [], [], [], []
        for k in range(nsteps):
            response = self.step()
            if k % save_every == 0 or k == nsteps - 1:
                times.append(self.elapsed)
                h_hist.append(self.state.h_ice.data)
                c_hist.append(self.state.concentration.data)
                lhf.append(response.lhflx)

        return self.state, IceColumnHistory(
            time=jnp.asarray(times),
            h_ice=jnp.stack(h_hist),
            concentration=jnp.stack(c_hist),
            lhflx=jnp.stack(lhf),
        )
