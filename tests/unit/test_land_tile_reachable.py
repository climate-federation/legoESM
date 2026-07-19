"""Direct unit tests for the driver-level land-tile reachability invariant
``coupled_esm_driver.assert_land_tile_reachable``.

The CONFIG-level CLI guard (``run_coupled.apply_land_runoff_scheme``) can only
see the declared ``f_land_mode``/``land_mode`` fields; it documents that it
cannot catch the two RUNTIME paths where ``f_land`` materialises to zero while a
land model is configured -- ``from_ocean`` over an all-wet ocean mask, and the
``analytical`` branch falling back to zeros on a grid with no latitude. This
invariant runs on the MATERIALISED ``f_land`` at coupler-init and closes that
residue.

Pinned as a pure function so the check is exercised without building a full
dynamic-ocean / degenerate-grid driver.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.driver.coupled_esm_driver import assert_land_tile_reachable

_ALL_WET = jnp.zeros((4, 6))            # f_land = 0 everywhere (dead land tile)
_HAS_LAND = jnp.zeros((4, 6)).at[0].set(0.5)   # some land somewhere


class TestSilentlyDeadLandTileRaises:
    def test_slab_land_with_zero_f_land_from_ocean_raises(self):
        """land_mode='slab' + f_land_mode='from_ocean' (all-wet ocean) -> the
        land tile has zero area: the misconfiguration the invariant exists for."""
        with pytest.raises(ValueError, match="materialized land fraction is zero"):
            assert_land_tile_reachable("slab", "from_ocean", _ALL_WET)

    def test_multilayer_land_with_zero_f_land_analytical_raises(self):
        """The analytical degenerate-grid (no latitude) fall-to-zeros path with a
        real land model must also raise."""
        with pytest.raises(ValueError, match="silently dead"):
            assert_land_tile_reachable("multilayer", "analytical", _ALL_WET)


class TestLegitimateZeroLandDoesNotRaise:
    def test_aquaplanet_slab_land_with_f_land_mode_zero_is_allowed(self):
        """f_land_mode='zero' is the EXPLICIT aquaplanet-with-slab-land request
        (--preset aquaplanet --land-scheme slab); a zero land area is intended,
        not an accident -- it must NOT raise even though the tile has no area."""
        assert_land_tile_reachable("slab", "zero", _ALL_WET)  # no raise

    def test_land_mode_none_never_raises(self):
        """With no land model there is nothing to run on the (zero) tile."""
        assert_land_tile_reachable("none", "from_ocean", _ALL_WET)  # no raise
        assert_land_tile_reachable("none", "analytical", _ALL_WET)  # no raise

    def test_real_land_area_passes(self):
        """A land model with actual land area is the normal, valid case."""
        assert_land_tile_reachable("slab", "analytical", _HAS_LAND)
        assert_land_tile_reachable("multilayer", "from_ocean", _HAS_LAND)


class TestDriverActuallyExecutesTheInvariant:
    """A pure-helper test is silent if the driver forgets to CALL it (codex): a
    real ``setup()`` must EXECUTE the invariant on the materialised tile fraction.

    Execution-level, not AST: an AST/source check cannot tell a live call from
    dead code (``if False:``) or catch a reassignment between materialising
    ``f_land`` and the call (codex round 2). Here we monkeypatch the invariant to
    a recorder, run a real coupler ``setup()``, and assert (a) it was invoked and
    (b) the array it received IS the ``f_land`` the driver stored in
    ``TileConfig`` -- so neither dead-coding the call nor passing an unrelated
    array would pass."""

    def _minimal_slab_land_driver(self):
        from legoesm.driver.config import (
            DycoreConfig,
            ExperimentConfig,
            GridConfig,
            OutputConfig,
        )
        from legoesm.driver.coupled_config import preset_slab_simple
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        atm_config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
            dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
            output=OutputConfig(diag_days=1),
            radiation="gray",
            days=1,
        )
        # slab_simple: land_mode='slab' + f_land_mode='analytical' -- the NON-
        # exempt branch the invariant actually guards (unlike aquaplanet's
        # f_land_mode='zero', which early-returns, codex round 3). On this
        # cubed-sphere grid the analytical mask generates real land (|lat|>25),
        # so max(f_land) > 0 and setup() completes while the guard runs live.
        coupled_cfg = preset_slab_simple()
        return CoupledESMDriver(atm_config, coupled_cfg)

    def test_setup_runs_the_invariant_on_the_stored_tile_fraction(
        self, monkeypatch
    ):
        from legoesm.driver import coupled_esm_driver as mod

        real = mod.assert_land_tile_reachable
        calls = []

        def recorder(land_mode, f_land_mode, f_land):
            calls.append((land_mode, f_land_mode, f_land))
            return real(land_mode, f_land_mode, f_land)

        monkeypatch.setattr(mod, "assert_land_tile_reachable", recorder)

        driver = self._minimal_slab_land_driver()
        driver.setup()

        assert calls, (
            "coupler setup() never executed assert_land_tile_reachable -- the "
            "invariant is present in source but not on the live init path")
        land_mode, f_land_mode, recorded_f_land = calls[-1]
        # Pin the live call on the branch the invariant GUARDS (not the exempt
        # f_land_mode='zero' early return): analytical, with real land present.
        assert land_mode == "slab" and f_land_mode == "analytical"
        assert float(jnp.max(recorded_f_land)) > 0.0, (
            "the analytical mask produced no land on this grid -- pick a grid "
            "whose analytical branch yields land so the guard runs on a live "
            "non-zero fraction")
        # The array the invariant received IS the one the coupler runs on.
        assert jnp.array_equal(
            recorded_f_land, driver._tile_config.f_land), (
            "the invariant was called with an array other than the f_land the "
            "driver stored in TileConfig -- it is not guarding the live tile "
            "fraction")
