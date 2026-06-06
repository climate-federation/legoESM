"""The standalone ice column driver runs the sea-ice brick independent of the coupler."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ice.scm import IceColumnModel, cold_polar_forcing

_NCOL = 4
_DT = 3600.0
_NSTEPS = 24


def test_ice_column_evolves_standalone() -> None:
    """A thermodynamic ice column integrates on its own and changes ice state."""
    scm = IceColumnModel.create(
        ncol=_NCOL, dt=_DT, forcing=cold_polar_forcing(ncol=_NCOL),
        h_ice_init=0.5,
    )
    final_state, history = scm.run(_NSTEPS, save_every=6)

    nsaved = history.h_ice.shape[0]
    assert history.h_ice.shape == (nsaved, _NCOL)
    assert jnp.all(jnp.isfinite(history.h_ice))
    assert jnp.all(jnp.isfinite(history.concentration))
    assert jnp.all(jnp.isfinite(history.lhflx))
    # Ice thickness stays physical (non-negative) and the column actually
    # integrated: thickness moved from its 0.5 m initial value under cold forcing.
    assert jnp.all(final_state.h_ice.data >= 0.0)
    assert jnp.max(jnp.abs(final_state.h_ice.data - 0.5)) > 1e-6


def test_step_matches_run_for_time_varying_forcing() -> None:
    """A manual step() loop reproduces run() exactly (shared elapsed clock)."""
    base = cold_polar_forcing(ncol=_NCOL)

    def forcing(t: float):
        # longwave drifts with elapsed time -> trajectory depends on the clock
        return base._replace(lw_down=base.lw_down + 10.0 * (t / 86400.0))

    n = 8
    a = IceColumnModel.create(ncol=_NCOL, dt=_DT, forcing=forcing)
    final_a, _ = a.run(n)
    b = IceColumnModel.create(ncol=_NCOL, dt=_DT, forcing=forcing)
    for _ in range(n):
        b.step()
    assert jnp.allclose(final_a.h_ice.data, b.state.h_ice.data)


@pytest.mark.parametrize("bad", [{"nsteps": 0}, {"nsteps": 4, "save_every": 0}])
def test_run_rejects_nonpositive_steps(bad) -> None:
    """run() validates nsteps/save_every rather than stacking empty history."""
    scm = IceColumnModel.create(forcing=cold_polar_forcing())
    nsteps = bad.pop("nsteps")
    with pytest.raises(ValueError):
        scm.run(nsteps, **bad)


def test_dynamics_rung_rejected_for_column() -> None:
    """Ice dynamics need a grid — the column driver refuses a dynamic config."""
    from legoesm.ice import SeaIceConfig

    with pytest.raises(ValueError, match="dynamics"):
        IceColumnModel.create(
            config=SeaIceConfig(dynamics="evp"), forcing=cold_polar_forcing()
        )
