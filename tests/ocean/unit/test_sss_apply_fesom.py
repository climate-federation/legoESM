"""SSS restoring on the FESOM node mesh.

The point of this file is the TIME-LEVEL test.  The obvious way to write a
salinity back into the FESOM facade is ``with_fields(state, mesh, S=...)``,
and it is wrong per step: its ``S`` branch does ``repl.update(S=Sf,
S_old=Sf)``, so it rewrites the scheme's history as well as its salinity.
That defect raises nothing and produces a plausible run, which is why it gets
an explicit test here rather than a comment.

Requires ``fesom_jax`` and the packaged pi mesh; skipped otherwise, like the
rest of the FESOM arm.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step_fesom
from legoesm.ocean.dynamics.ocean_model_fesom import (
    FesomOceanGrid,
    build_flat_bottom_mesh,
    create_lock_exchange_state,
    with_surface_salinity,
)
from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
from legoesm.ocean.forcing.sss_restoring import SSSRestoringConfig

_S_UNIFORM = 35.0


@pytest.fixture(scope="module")
def _pi_mesh():
    pytest.importorskip("fesom_jax")
    from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
    return load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR)


@pytest.fixture(scope="module")
def _cfg():
    # land_lat_threshold=80 is the PRODUCTION value and leaves dry nodes on
    # the mesh, which the land-mask test below needs in order not to be
    # vacuous.
    return LockExchangeConfig(S_uniform=_S_UNIFORM)


@pytest.fixture(scope="module")
def _mesh(_pi_mesh, _cfg):
    return build_flat_bottom_mesh(
        _pi_mesh,
        H_max=_cfg.H_max,
        nlev=_cfg.nlev,
        land_lat_threshold=_cfg.land_lat_threshold,
    )


@pytest.fixture
def _state(_mesh, _cfg):
    return create_lock_exchange_state(_mesh, _cfg)


@pytest.fixture(scope="module")
def _grid(_mesh):
    return FesomOceanGrid(mesh=_mesh)


def _target(state, value):
    return np.full(state.S.data.shape[0], value, dtype=np.float64)


def _restore(state, grid, *, target, tau_days=30.0, dt=86400.0):
    return apply_sss_restoring_step_fesom(
        state,
        S_target=_target(state, target),
        ice_concentration=None,
        config=SSSRestoringConfig(
            enabled=True, tau_restore_days_default=tau_days, regions=(),
        ),
        grid=grid,
        dt=dt,
    )


class TestApplySSSRestoringFesom:

    def test_disabled_is_noop(self, _state, _grid):
        out = apply_sss_restoring_step_fesom(
            _state,
            S_target=_target(_state, 34.0),
            ice_concentration=None,
            config=SSSRestoringConfig(enabled=False),
            grid=_grid,
            dt=3600.0,
        )
        assert out is _state

    def test_old_time_level_is_untouched(self, _state, _grid):
        """THE test: the surface moves and the OLD time level does not.

        ``with_fields`` would fail this -- it sets ``S_old = S``.  A target
        far from the model salinity keeps the two assertions independent:
        the first would pass trivially if nothing moved at all, so the second
        is asserted as well.
        """
        S_old_before = np.asarray(_state.inner.S_old).copy()
        S_top_before = np.asarray(_state.S.data[..., 0]).copy()

        out = _restore(_state, _grid, target=_S_UNIFORM - 2.0)

        S_top_after = np.asarray(out.S.data[..., 0])
        assert not np.allclose(S_top_after, S_top_before), (
            "the surface salinity did not move, so the S_old check below "
            "would be vacuous")
        np.testing.assert_array_equal(
            np.asarray(out.inner.S_old), S_old_before)

    def test_deep_levels_are_bit_identical(self, _state, _grid):
        S_deep_before = np.asarray(_state.inner.S[:, 1:]).copy()
        out = _restore(_state, _grid, target=_S_UNIFORM - 2.0)
        np.testing.assert_array_equal(
            np.asarray(out.inner.S[:, 1:]), S_deep_before)

    def test_temperature_and_velocity_are_untouched(self, _state, _grid):
        T_before = np.asarray(_state.inner.T).copy()
        uv_before = np.asarray(_state.inner.uv).copy()
        out = _restore(_state, _grid, target=_S_UNIFORM - 2.0)
        np.testing.assert_array_equal(np.asarray(out.inner.T), T_before)
        np.testing.assert_array_equal(np.asarray(out.inner.uv), uv_before)

    def test_dry_nodes_are_untouched(self, _state, _grid):
        """Land nodes keep their salinity exactly.

        Guarded against vacuity: the production land threshold must actually
        leave dry nodes on this mesh, or the test proves nothing.
        """
        dry = np.asarray(_state.land_mask.data) < 0.5
        assert dry.any(), (
            "no dry nodes on this mesh -- the land-mask test is vacuous")
        S_top_before = np.asarray(_state.S.data[..., 0]).copy()
        out = _restore(_state, _grid, target=_S_UNIFORM - 2.0)
        S_top_after = np.asarray(out.S.data[..., 0])
        np.testing.assert_array_equal(S_top_after[dry], S_top_before[dry])
        wet = ~dry
        assert not np.allclose(S_top_after[wet], S_top_before[wet])

    def test_salty_bias_freshens_the_surface(self, _state, _grid):
        wet = np.asarray(_state.land_mask.data) > 0.5
        target = _S_UNIFORM - 0.8
        out = _restore(_state, _grid, target=target, tau_days=30.0,
                       dt=86400.0)
        d = (np.asarray(out.S.data[..., 0])
             - np.asarray(_state.S.data[..., 0]))[wet]
        assert np.all(d < 0.0)
        # tau=30 d, dt=1 d, bias 0.8 psu -> about -0.0267 psu.
        assert abs(float(np.mean(d)) - (-(0.8) / 30.0)) < 1.0e-3

    def test_fresh_bias_salinifies_the_surface(self, _state, _grid):
        wet = np.asarray(_state.land_mask.data) > 0.5
        out = _restore(_state, _grid, target=_S_UNIFORM + 0.8)
        d = (np.asarray(out.S.data[..., 0])
             - np.asarray(_state.S.data[..., 0]))[wet]
        assert np.all(d > 0.0)

    def test_zero_bias_does_not_move_the_surface(self, _state, _grid):
        out = _restore(_state, _grid, target=_S_UNIFORM)
        np.testing.assert_allclose(
            np.asarray(out.S.data[..., 0]),
            np.asarray(_state.S.data[..., 0]), atol=1e-12)


def test_with_fields_would_collapse_the_time_levels(_state, _mesh):
    """NON-VACUITY for ``test_old_time_level_is_untouched``.

    That test only means something if the rejected alternative would actually
    fail it.  ``with_fields`` is the obvious way to write a salinity into this
    facade; here it is shown to move ``S_old`` as well, which is precisely the
    defect ``with_surface_salinity`` exists to avoid.  If this ever stops
    holding, the time-level test above has stopped discriminating and both
    should be revisited.
    """
    from legoesm.ocean.dynamics.ocean_model_fesom import with_fields

    n = int(_mesh.nod2D)
    bumped = with_fields(
        _state, _mesh,
        S=jnp.full((n,), _S_UNIFORM - 3.0, dtype=jnp.float64))
    assert not np.allclose(
        np.asarray(bumped.inner.S_old), np.asarray(_state.inner.S_old)), (
        "with_fields no longer collapses S_old; the time-level test above is "
        "now vacuous")


class TestWithSurfaceSalinityGuards:

    def test_rejects_a_full_column(self, _state, _mesh):
        full = jnp.asarray(_state.inner.S)
        with pytest.raises(ValueError, match="S_top shape"):
            with_surface_salinity(_state, _mesh, full)

    def test_rejects_a_wrong_node_count(self, _state, _mesh):
        n = int(_mesh.nod2D)
        with pytest.raises(ValueError, match="S_top shape"):
            with_surface_salinity(
                _state, _mesh, jnp.zeros((n - 1,), dtype=jnp.float64))

    def test_writes_only_the_surface(self, _state, _mesh):
        n = int(_mesh.nod2D)
        new_top = jnp.full((n,), 12.5, dtype=jnp.float64)
        out = with_surface_salinity(_state, _mesh, new_top)
        np.testing.assert_allclose(np.asarray(out.inner.S[:, 0]), 12.5)
        np.testing.assert_array_equal(
            np.asarray(out.inner.S[:, 1:]),
            np.asarray(_state.inner.S[:, 1:]))
        np.testing.assert_array_equal(
            np.asarray(out.inner.S_old), np.asarray(_state.inner.S_old))
