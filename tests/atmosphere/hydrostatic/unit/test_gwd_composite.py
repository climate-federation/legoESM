"""Composite ('+'-joined) GWD: resolver, summation, and SSO plumbing.

ExperimentConfig has validated composite gravity_wave_drag strings (e.g.
``hines+mcfarlane``) since the composite comment landed, but until the
campaign wiring the resolver raised ``Unknown GWD scheme`` at runtime.
These tests lock: (a) get_gwd_fn resolves composites and rejects invalid
ones, (b) composite tendencies are exactly the sum of the members run on
the same state, (c) the per-column subgrid orography reaches the
orographic member through the hydrostatic factory path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    get_gwd_fn,
    gwd_scheme_is_orographic,
    make_gwd_physics,
)


def _make_state(n=4, nlev=10, wind_speed=20.0):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev)) * wind_speed,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev)) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    tracers = {
        "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    return state._replace(tracers=tracers), grid, sigma


def _tend(scheme, grid_sso=None):
    state, grid, sigma = _make_state()
    if grid_sso is not None:
        grid = grid._replace(subgrid_topo_stddev=grid_sso(grid))
    fn = make_gwd_physics(
        GravityWaveDragConfig(scheme=scheme), model_type="hydrostatic",
        dt=300.0,
    )
    tend, _ = fn(state, grid, sigma)
    return tend


def test_get_gwd_fn_resolves_composite():
    cfg = GravityWaveDragConfig(scheme="hines+mcfarlane")
    name, fn, sub = get_gwd_fn(cfg)
    assert name == "hines+mcfarlane"
    assert callable(fn)
    assert sub is cfg          # parent config back (members bound internally)


@pytest.mark.parametrize("bad", [
    "e3sm_cam+hines",              # e3sm_cam not composable (#834)
    "ml_emulator+hines",          # ml_emulator not composable
    "hines+",                     # empty part -> non-composable ""
    "prognostic_spectral+prognostic_spectral",  # >1 stateful carry
])
def test_get_gwd_fn_rejects_invalid_composites(bad):
    # main's #834 factory guard (_validate_gwd_composite): non-composable
    # parts, or more than one stateful (spectrum-carrying) source.
    with pytest.raises(ValueError):
        get_gwd_fn(GravityWaveDragConfig(scheme=bad))


@pytest.mark.parametrize("ok", [
    "hines+mcfarlane",                 # non-oro + orographic
    "hines+prognostic_spectral",       # non-oro + one stateful source (#834)
    "mcfarlane+prognostic_spectral",   # orographic + one stateful source
])
def test_get_gwd_fn_accepts_valid_composites(ok):
    name, fn, sub = get_gwd_fn(GravityWaveDragConfig(scheme=ok))
    assert name == ok and callable(fn)


def test_validate_strict_rejects_duplicate_composite():
    from legoesm.driver.config import ExperimentConfig

    with pytest.raises(ValueError, match="duplicate parts"):
        ExperimentConfig(gravity_wave_drag="mcfarlane+mcfarlane").validate_strict()
    # a valid composite still passes strict validation
    ExperimentConfig(gravity_wave_drag="hines+mcfarlane").validate_strict()


def test_orographic_helper_truth_table():
    assert gwd_scheme_is_orographic("mcfarlane")
    assert gwd_scheme_is_orographic("lindzen")
    assert gwd_scheme_is_orographic("hines+mcfarlane")
    assert not gwd_scheme_is_orographic("hines")
    assert not gwd_scheme_is_orographic("rayleigh+hines")
    assert not gwd_scheme_is_orographic("none")


def test_composite_tendencies_are_member_sum():
    t_h = _tend("hines")
    t_m = _tend("mcfarlane")
    t_c = _tend("hines+mcfarlane")
    for attr in ("du_dt", "dv_dt", "dT_dt"):
        np.testing.assert_allclose(
            np.asarray(getattr(t_c, attr).data),
            np.asarray(getattr(t_h, attr).data)
            + np.asarray(getattr(t_m, attr).data),
            rtol=1e-12, atol=1e-18,
        )
    # a composite of two active schemes must actually do something
    assert float(jnp.max(jnp.abs(t_c.du_dt.data))) > 0.0


def test_composite_forwards_h_topo_col_to_orographic_member_only():
    def zero_sso(grid):
        return jnp.zeros_like(grid.lat)

    # SSO == 0 kills the McFarlane launch (tau_0 ∝ h²) but must leave the
    # non-orographic Hines member untouched -> composite == hines alone.
    t_c0 = _tend("hines+mcfarlane", grid_sso=zero_sso)
    t_h = _tend("hines")
    for attr in ("du_dt", "dv_dt", "dT_dt"):
        np.testing.assert_allclose(
            np.asarray(getattr(t_c0, attr).data),
            np.asarray(getattr(t_h, attr).data),
            rtol=1e-12, atol=1e-18,
        )
    # and a NONZERO SSO must change the composite (mcfarlane active)
    t_c = _tend("hines+mcfarlane",
                grid_sso=lambda g: 500.0 * jnp.ones_like(g.lat))
    assert float(jnp.max(jnp.abs(
        t_c.du_dt.data - t_c0.du_dt.data))) > 0.0
