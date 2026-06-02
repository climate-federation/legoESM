"""The standalone land column driver runs the land brick independent of the coupler."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.scm import LandColumnModel

_NCOL = 4
_DT = 3600.0
_NSTEPS = 24


def constant_land_forcing(ncol: int = 1) -> AtmToSurface:
    """A uniform, time-constant AtmToSurface for ``ncol`` land columns.

    Lives in the test (not the component) because constructing it imports the
    coupler's coupling_fields, which the components layer must not depend on.
    """
    def full(v: float) -> jnp.ndarray:
        return jnp.full((ncol,), v)

    return AtmToSurface(
        sw_down=full(200.0), lw_down=full(300.0), precip_total=full(1e-5),
        precip_snow=full(0.0), T_lowest=full(280.0), q_lowest=full(5e-3),
        u_lowest=full(3.0), v_lowest=full(0.0), p_lowest=full(9.5e4),
        p_surface=full(1.0e5), rho_lowest=full(1.15), cos_zenith=full(0.5),
        co2_ppmv=jnp.asarray(400.0), has_radiation=jnp.asarray(1.0),
        has_precipitation=jnp.asarray(1.0),
    )


def test_constant_forcing_shape() -> None:
    f = constant_land_forcing(ncol=_NCOL)
    assert f.sw_down.shape == (_NCOL,)
    assert jnp.all(jnp.isfinite(f.T_lowest))


def test_slab_land_spins_up_standalone() -> None:
    """A slab-land column integrates on its own and advances soil state."""
    from legoesm.land import LandConfig

    scm = LandColumnModel.create(
        config=LandConfig(), ncol=_NCOL, dt=_DT,
        forcing=constant_land_forcing(ncol=_NCOL),
        T_soil_init=280.0,
    )
    final_state, history = scm.run(_NSTEPS, save_every=6)

    nsaved = history.T_soil_surface.shape[0]
    assert history.T_soil_surface.shape == (nsaved, _NCOL)
    assert history.shflx.shape == (nsaved, _NCOL)
    assert jnp.all(jnp.isfinite(history.T_soil_surface))
    assert jnp.all(jnp.isfinite(history.shflx)) and jnp.all(jnp.isfinite(history.lhflx))
    # The brick actually integrated: soil temperature moved from its init value.
    assert jnp.max(jnp.abs(final_state.T_soil.data - 280.0)) > 1e-6


def test_multilayer_land_spins_up_standalone() -> None:
    """A multi-layer Richards column integrates on its own."""
    from legoesm.land import MultiLayerLandConfig

    scm = LandColumnModel.create(
        config=MultiLayerLandConfig(), ncol=_NCOL, dt=_DT,
        forcing=constant_land_forcing(ncol=_NCOL),
        T_soil_init=280.0,
    )
    final_state, history = scm.run(_NSTEPS, save_every=6)

    nsaved = history.T_soil_surface.shape[0]
    assert history.T_soil_surface.shape == (nsaved, _NCOL)
    assert jnp.all(jnp.isfinite(history.T_soil_surface))
    assert final_state.T_soil.shape[0] == _NCOL
    # top-layer soil temperature advanced under the forcing
    assert jnp.max(jnp.abs(final_state.T_soil[:, 0] - 280.0)) > 1e-6


def test_callable_forcing_advances_with_time_and_step_matches_run() -> None:
    """Time-varying forcing is applied at the right elapsed time, and a manual
    step() loop equals run() exactly (shared clock — no t=0 freeze)."""
    from legoesm.land import LandConfig

    base = constant_land_forcing(ncol=_NCOL)

    def forcing(t: float):
        # shortwave ramps with elapsed time -> the trajectory depends on the clock
        return base._replace(sw_down=base.sw_down * (1.0 + t / 86400.0))

    n = 8
    scm_run = LandColumnModel.create(
        config=LandConfig(), ncol=_NCOL, dt=_DT, forcing=forcing
    )
    final_run, _ = scm_run.run(n)

    # Repeated step() must reproduce run() exactly (both advance the same clock).
    scm_step = LandColumnModel.create(
        config=LandConfig(), ncol=_NCOL, dt=_DT, forcing=forcing
    )
    for _ in range(n):
        scm_step.step()
    assert jnp.allclose(final_run.T_soil.data, scm_step.state.T_soil.data)

    # And the time-varying forcing actually changed the outcome vs a constant run
    # (proving the ramp is applied at the right t, not frozen at t=0).
    scm_const = LandColumnModel.create(
        config=LandConfig(), ncol=_NCOL, dt=_DT, forcing=base
    )
    final_const, _ = scm_const.run(n)
    assert not jnp.allclose(final_run.T_soil.data, final_const.T_soil.data)


@pytest.mark.parametrize("bad", [{"nsteps": 0}, {"nsteps": 4, "save_every": 0}])
def test_run_rejects_nonpositive_steps(bad) -> None:
    """run() validates nsteps/save_every rather than stacking empty history."""
    from legoesm.land import LandConfig

    scm = LandColumnModel.create(config=LandConfig(), forcing=constant_land_forcing())
    nsteps = bad.pop("nsteps")
    with pytest.raises(ValueError):
        scm.run(nsteps, **bad)


def test_unknown_config_type_raises() -> None:
    with pytest.raises(TypeError):
        LandColumnModel.create(config=object(), forcing=constant_land_forcing())
