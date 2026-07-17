"""Single-column (standalone) driver for the legoESM land surface.

The land counterpart of :class:`legoesm.atmosphere.forcing.scm.scm.SingleColumnModel` and
:class:`legoesm.ocean.scm.OceanColumnModel`: a minimal driver that runs the land
*brick* on its own — init state, prescribed atmospheric forcing, time loop,
history — so the land model can be exercised **independently of the coupler**
(land spin-up, soil-hydrology / surface-flux parameterization studies, stability
sweeps), using the *same* step functions the coupled driver calls.

It wraps the canonical land step — ``land.slab_land.step_land`` (slab bucket) or
``land.multilayer_land.step_multilayer_land`` (multi-layer Richards column),
selected by the config type, exactly as the complexity ladder
(``components.LandComplexity``) and ``driver.component_factory.create_land_component``
do — no physics is re-implemented here.

The *forcing* is the per-step atmospheric state the land sees — an
``AtmToSurface`` (or a ``callable(elapsed_seconds) -> AtmToSurface``).  It is
duck-typed so a caller may pass a constant field or a time-varying closure;
``AtmToSurface`` lives in ``legoesm.core.coupling_fields`` (a core pytree any
component may import), so this module stays boundary-clean (no ``coupler`` import).

Example
-------
>>> from legoesm.land.scm import LandColumnModel, constant_land_forcing
>>> from legoesm.land import LandConfig
>>> scm = LandColumnModel.create(
...     config=LandConfig(), ncol=1, dt=3600.0, forcing=constant_land_forcing(ncol=1))
>>> final_state, history = scm.run(nsteps=240, save_every=24)  # 10-day spin-up
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.coupling_fields import AtmToSurface


def constant_land_forcing(
    ncol: int = 1,
    *,
    sw_down: float = 200.0,  # coeff-ok: SCM idealized forcing [W/m^2]
    lw_down: float = 300.0,  # coeff-ok: SCM idealized forcing [W/m^2]
    precip_total: float = 1e-5,  # coeff-ok: SCM idealized forcing [kg/m^2/s]
    T_lowest: float = 280.0,  # noqa: N803 — coeff-ok: SCM forcing [K]
    q_lowest: float = 5e-3,  # coeff-ok: SCM idealized forcing [kg/kg]
    u_lowest: float = 3.0,
    v_lowest: float = 0.0,
    p_surface: float = 1.0e5,  # coeff-ok: SCM idealized forcing [Pa]
) -> AtmToSurface:
    """A spatially-uniform, time-constant :class:`AtmToSurface` over ``ncol`` columns.

    A convenience for land spin-up / process studies; pass any ``AtmToSurface``
    (or a ``callable(elapsed) -> AtmToSurface``) to :meth:`LandColumnModel.create`
    for full control.  Lives here (not the test) because ``AtmToSurface`` is now a
    core pytree any component may import — the SCM ships its own forcing.
    """

    def full(v: float) -> jax.Array:
        return jnp.full((ncol,), v)

    return AtmToSurface(
        sw_down=full(sw_down), lw_down=full(lw_down), precip_total=full(precip_total),
        precip_snow=full(0.0), T_lowest=full(T_lowest), q_lowest=full(q_lowest),
        u_lowest=full(u_lowest), v_lowest=full(v_lowest), p_lowest=full(0.95 * p_surface),  # coeff-ok: idealized forcing
        p_surface=full(p_surface), rho_lowest=full(1.15), cos_zenith=full(0.5),  # coeff-ok: idealized forcing
        co2_ppmv=jnp.asarray(400.0), has_radiation=jnp.asarray(1.0),  # coeff-ok: idealized forcing
        has_precipitation=jnp.asarray(1.0),
    )


class LandColumnHistory(NamedTuple):
    """Saved time series from a :meth:`LandColumnModel.run`."""

    time: jax.Array  # (nsaved,) elapsed seconds
    T_soil_surface: jax.Array  # (nsaved, ncol) surface/top-layer soil temperature [K]
    shflx: jax.Array  # (nsaved, ncol) sensible heat flux [W/m2, +up]
    lhflx: jax.Array  # (nsaved, ncol) latent heat flux [W/m2, +up]


class LandColumnModel:
    """Standalone single-column land driver (slab or multi-layer)."""

    def __init__(
        self, *, step_fn, state, config, forcing, dt, u_min, multilayer, lat,
        start_day_of_year=0.0,
    ):
        self._step_fn = step_fn
        self.state = state
        self.config = config
        self._forcing = forcing
        self.dt = dt
        self._u_min = u_min
        self._multilayer = multilayer
        self._lat = lat
        self._start_doy = start_day_of_year
        # Elapsed model time [s] since construction — advanced by every step() so
        # callable forcing and the seasonal day-of-year stay correct across BOTH
        # public paths (step() and run() share one clock).
        self.elapsed = 0.0

    @classmethod
    def create(
        cls,
        *,
        forcing,
        config=None,
        ncol: int = 1,
        dt: float = 3600.0,
        T_soil_init: float = 280.0,  # noqa: N803 — coeff-ok: SCM initial condition [K]
        W_bucket_init: float = 75.0,  # noqa: N803 — coeff-ok: SCM initial condition [kg/m^2]
        u_min: float = 1.0,
        lat=None,
        start_day_of_year: float = 0.0,
    ) -> LandColumnModel:
        """Build a column land model.  The *config* type selects the rung: a
        ``LandConfig`` runs the slab bucket, a ``MultiLayerLandConfig`` runs the
        multi-layer Richards column (same dispatch as ``create_land_component``).
        """
        from legoesm.land import (
            LandConfig,
            MultiLayerLandConfig,
            step_land,
            step_multilayer_land,
        )
        from legoesm.land.multilayer_land import init_multilayer_land_state

        if config is None:
            config = LandConfig()

        if isinstance(config, MultiLayerLandConfig):
            state = init_multilayer_land_state(ncol, config, T_init=T_soil_init)
            return cls(
                step_fn=step_multilayer_land, state=state, config=config,
                forcing=forcing, dt=dt, u_min=u_min, multilayer=True, lat=lat,
                start_day_of_year=start_day_of_year,
            )
        if isinstance(config, LandConfig):
            state = _init_slab_land_state(ncol, T_soil_init, W_bucket_init)
            return cls(
                step_fn=step_land, state=state, config=config,
                forcing=forcing, dt=dt, u_min=u_min, multilayer=False, lat=lat,
                start_day_of_year=start_day_of_year,
            )
        raise TypeError(
            f"config must be LandConfig or MultiLayerLandConfig, "
            f"got {type(config).__name__!r}"
        )

    def _forcing_at(self, t: float):
        """The forcing (an AtmToSurface) at elapsed time *t* — constant or callable."""
        f = self._forcing
        return f(t) if callable(f) else f

    def _t_soil_surface(self) -> jax.Array:
        # slab: T_soil is a Field (ncol,); multilayer: array (ncol, nlayers), top = 0
        return self.state.T_soil[:, 0] if self._multilayer else self.state.T_soil.data

    def step(self) -> object:
        """Advance one step under the forcing at the CURRENT elapsed model time,
        then advance the clock; returns the per-step ``TileResponse``.

        Tracks ``self.elapsed`` so repeated ``step()`` calls apply time-varying
        (callable) forcing and the seasonal day-of-year correctly — identical to
        what ``run()`` does (``run`` just loops this method).
        """
        doy = self._start_doy + self.elapsed / 86400.0
        forcing = self._forcing_at(self.elapsed)
        new_state, response, _carbon = self._step_fn(
            self.state, forcing, self.config, self._u_min, self.dt,
            lat=self._lat, doy=doy,
        )
        self.state = new_state
        self.elapsed += self.dt
        return response

    def run(
        self, nsteps: int, *, save_every: int = 1
    ) -> tuple[object, LandColumnHistory]:
        """Integrate ``nsteps`` (looping :meth:`step`) and return
        ``(final_state, history)``.  ``save_every`` controls the output cadence;
        the seasonal day-of-year advances from ``start_day_of_year`` (set at
        :meth:`create`) via the shared elapsed-time clock.
        """
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every}")
        times, t_soil_hist, shf, lhf = [], [], [], []
        for k in range(nsteps):
            response = self.step()
            if k % save_every == 0 or k == nsteps - 1:
                times.append(self.elapsed)  # elapsed AFTER this step = (k+1)*dt
                t_soil_hist.append(self._t_soil_surface())
                shf.append(response.shflx)
                lhf.append(response.lhflx)

        return self.state, LandColumnHistory(
            time=jnp.asarray(times),
            T_soil_surface=jnp.stack(t_soil_hist),
            shflx=jnp.stack(shf),
            lhflx=jnp.stack(lhf),
        )


def _init_slab_land_state(ncol, T_soil_init, W_bucket_init):  # noqa: N803
    """Build a slab :class:`~legoesm.land.state.LandState` for ``ncol`` columns."""
    from legoesm.core.field import Field
    from legoesm.land import LandState

    def field(name: str, v: float, units: str):
        return Field(jnp.full((ncol,), v), name=name, dims=("column",), units=units)

    return LandState(
        T_soil=field("T_soil", T_soil_init, "K"),
        W_bucket=field("W_bucket", W_bucket_init, "kg/m2"),
        snow_depth=field("snow_depth", 0.0, "kg/m2"),
        snow_age=field("snow_age", 0.0, "s"),
    )
