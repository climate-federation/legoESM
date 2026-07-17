"""Multi-category sea-ice reachability from the CORE-II OMIP runner
(``--ice-categories`` / ``--ice-ridging``, scheme-reachability audit #13).

Three layers, mirroring the audit's discipline:

1. the pure resolver ``_resolve_ice_categories`` — refuse-not-ignore, and the
   default byte-identical to pre-flag runs;
2. the argparse surface — the flags exist and parse to the resolver's inputs;
3. an ANTI-PHANTOM execution smoke: the state and config built exactly the
   way the driver builds them (``_ice_state_spatial_shape`` +
   ``init_dynamic_ice_state`` + ``distribute_dynamic_state_to_categories``,
   ``SeaIceConfig(n_categories=5, itd_remap='lipscomb2001', ridging=ON,
   brine=ON)``) must survive a real ``step_sea_ice`` on a lat-lon grid — a
   flag whose accepted values crash on the first step is worse than no flag
   (the --ice-categories/--ice-dynamics phantoms deleted from run_coupled).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from scripts.run.run_omip_core2 import (
    _build_arg_parser,
    _ice_global_stats,
    _ice_state_spatial_shape,
    _require_prognostic_ice_for_itd_flags,
    _resolve_ice_categories,
)

# ---------------------------------------------------------------------------
# 1. Resolver
# ---------------------------------------------------------------------------

class TestResolveIceCategories:
    def test_default_is_byte_identical_single_category(self):
        """n=1, no ridging -> the exact pre-flag SeaIceConfig fields (itd
        'simple' is the NamedTuple default and is never consulted at one
        category)."""
        assert _resolve_ice_categories(1, False, True, "VoronoiMesh") == (
            1, "simple", False)

    def test_multicat_forces_lipscomb(self):
        """This runner always enables brine, and step_sea_ice rejects
        multi-category tracers under 'simple' — so lipscomb2001 is set
        automatically, not offered as a choice that could never run."""
        n, itd, ridge = _resolve_ice_categories(5, False, True, "VoronoiMesh")
        assert (n, itd, ridge) == (5, "lipscomb2001", False)

    def test_ridging_with_multicat_and_strain_ops(self):
        assert _resolve_ice_categories(5, True, True, "VoronoiMesh") == (
            5, "lipscomb2001", True)

    def test_ridging_refused_at_one_category(self):
        """The step's ridging gate is multi-category-only: accepting
        --ice-ridging at n=1 would be a silent every-step no-op."""
        with pytest.raises(SystemExit, match="--ice-categories >= 2"):
            _resolve_ice_categories(1, True, True, "VoronoiMesh")

    def test_ridging_refused_without_strain_ops(self):
        """Tripole lacks the strain-rate operators; step_sea_ice would raise
        at entry, so the driver refuses up front with the actionable
        message."""
        with pytest.raises(SystemExit, match="strain-rate"):
            _resolve_ice_categories(5, True, False, "LatLonCGridGeometry")

    def test_zero_categories_refused(self):
        with pytest.raises(SystemExit, match="--ice-categories"):
            _resolve_ice_categories(0, False, True, "VoronoiMesh")


# ---------------------------------------------------------------------------
# 2. Argparse surface
# ---------------------------------------------------------------------------

class TestParserSurface:
    def test_flags_parse_and_default_off(self):
        args = _build_arg_parser().parse_args([])
        assert args.ice_categories == 1
        assert args.ice_ridging is False

    def test_flags_round_trip(self):
        args = _build_arg_parser().parse_args(
            ["--ice-categories", "5", "--ice-ridging"])
        assert args.ice_categories == 5
        assert args.ice_ridging is True


# ---------------------------------------------------------------------------
# 3. Anti-phantom execution smoke (lat-lon production shapes)
# ---------------------------------------------------------------------------

def test_multicat_config_and_state_survive_a_real_step():
    """Build the 5-category state + config the way main() builds them and run
    one real step_sea_ice on a small lat-lon grid: multi-cat shape must
    survive, fields stay finite, the trailing category axis stays trailing,
    and the velocity fields stay spatial-only."""
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice import (
        SeaIceConfig,
        distribute_dynamic_state_to_categories,
        init_dynamic_ice_state,
        step_sea_ice,
    )
    from legoesm.ice.config import BrineConfig, RidgingConfig

    n_cat = 5
    grid = create_latlon_grid(n_lat=8, n_lon=12)
    spatial = _ice_state_spatial_shape(grid, "latlon")
    assert spatial == (8, 12)

    # Driver order: single-category state first (where --ice-init would
    # apply), then the ITD lift.
    st = init_dynamic_ice_state(spatial, S_ice_init=0.0)
    h = jnp.zeros(spatial).at[:2, :].set(1.5)      # polar band of 1.5 m ice
    conc = jnp.zeros(spatial).at[:2, :].set(0.9)
    st = st._replace(
        h_ice=st.h_ice.replace(data=h),
        concentration=st.concentration.replace(data=conc),
        T_ice=st.T_ice.replace(data=jnp.full(spatial, 255.0)),
    )
    st = distribute_dynamic_state_to_categories(st, n_cat)

    n_res, itd, ridge_on = _resolve_ice_categories(
        n_cat, True, True, type(grid).__name__)
    config = SeaIceConfig(
        dynamics="evp", transport="advect",
        n_categories=n_res, itd_remap=itd,
        ridging=RidgingConfig(enabled=ridge_on),
        brine=BrineConfig(enabled=True),
    )

    forcing = AtmToSurface(
        sw_down=jnp.full(spatial, 50.0), lw_down=jnp.full(spatial, 200.0),
        T_lowest=jnp.full(spatial, 250.0), q_lowest=jnp.full(spatial, 1e-3),
        u_lowest=jnp.full(spatial, 5.0), v_lowest=jnp.zeros(spatial),
        p_lowest=jnp.full(spatial, 1.0e5), p_surface=jnp.full(spatial, 1.0e5),
        rho_lowest=jnp.full(spatial, 1.3),
        cos_zenith=jnp.full(spatial, 0.2), co2_ppmv=jnp.full(spatial, 400.0),
        precip_total=jnp.zeros(spatial), precip_snow=jnp.zeros(spatial),
        has_radiation=jnp.ones(spatial), has_precipitation=jnp.zeros(spatial),
    )
    new_state, resp = step_sea_ice(
        st, forcing, jnp.full(spatial, 271.35),
        jnp.zeros(spatial), jnp.zeros(spatial),
        config, U_min=0.0, dt=1800.0, grid=grid)

    assert new_state.h_ice.data.shape == spatial + (n_cat,)
    assert new_state.u_ice.data.shape == spatial
    for name in ("h_ice", "T_ice", "concentration", "h_snow", "S_ice"):
        arr = getattr(new_state, name).data
        assert bool(jnp.all(jnp.isfinite(arr))), f"{name} not finite"
    # The response feeds the ocean channels; it must be aggregate (spatial).
    assert bool(jnp.all(jnp.isfinite(resp.salt_flux)))
    # Ice survived one winter step over the seeded band.
    agg_conc = jnp.sum(new_state.concentration.data, axis=-1)
    assert float(jnp.max(agg_conc)) > 0.1


# ---------------------------------------------------------------------------
# 4. Codex round-1 regressions
# ---------------------------------------------------------------------------

class TestItdFlagsRequirePrognosticIce:
    """--ice-categories/--ice-ridging without --prognostic-sea-ice were
    accepted no-ops (their only consumer sits inside the prognostic-ice
    build): the exact accept-then-ignore shape the audit forbids (codex)."""

    def test_categories_without_prognostic_ice_refused(self):
        with pytest.raises(ValueError, match="--prognostic-sea-ice"):
            _require_prognostic_ice_for_itd_flags(5, False, False)

    def test_ridging_without_prognostic_ice_refused(self):
        with pytest.raises(ValueError, match="--prognostic-sea-ice"):
            _require_prognostic_ice_for_itd_flags(1, True, False)

    def test_defaults_pass_without_prognostic_ice(self):
        _require_prognostic_ice_for_itd_flags(1, False, False)  # no raise

    def test_flags_pass_with_prognostic_ice(self):
        _require_prognostic_ice_for_itd_flags(5, True, True)  # no raise


class TestIceGlobalStatsMulticat:
    """_ice_global_stats crashed on a real multi-category state: its old
    ``conc.ndim > h.ndim`` aggregation test was never true (both fields carry
    the category axis), so the spatial ocean-mask broadcast failed on the
    FIRST [ice] diag line (codex).  Aggregation now keys off the mask rank."""

    def _stats(self, ice_state):
        import types

        import numpy as np

        grid = types.SimpleNamespace(area=np.ones((4, 6)))
        mask = np.ones((4, 6))
        return _ice_global_stats(ice_state, grid, mask)

    def _single_cat(self):
        from legoesm.ice import init_dynamic_ice_state

        st = init_dynamic_ice_state((4, 6), S_ice_init=0.0)
        h = jnp.zeros((4, 6)).at[0, 0].set(2.0).at[1, 1].set(0.5)
        conc = jnp.zeros((4, 6)).at[0, 0].set(1.0).at[1, 1].set(0.4)
        return st._replace(
            h_ice=st.h_ice.replace(data=h),
            concentration=st.concentration.replace(data=conc))

    def test_multicat_state_no_longer_crashes_and_matches_single_cat(self):
        from legoesm.ice import distribute_dynamic_state_to_categories

        st1 = self._single_cat()
        area1, conc1, hmax1, _ = self._stats(st1)
        st5 = distribute_dynamic_state_to_categories(st1, 5)
        area5, conc5, hmax5, _ = self._stats(st5)
        # The delta ITD conserves area/concentration/thickness exactly, so
        # the aggregate diagnostics must agree with the single-category ones.
        assert area5 == pytest.approx(area1)
        assert conc5 == pytest.approx(conc1)
        assert hmax5 == pytest.approx(hmax1)


def test_dynamic_to_slab_aggregates_lower_rank_multicat():
    """dynamic_to_slab used the cubed-sphere-only ``h.ndim > 3`` heuristic:
    lat-lon multi-category (rank 3) and MPAS (rank 2) slipped through with
    the category axis still attached to a scalar SeaIceState (codex).  Rank
    vs the spatial-only u_ice detects every grid."""
    from legoesm.ice import distribute_dynamic_state_to_categories
    from legoesm.ice.state import dynamic_to_slab, init_dynamic_ice_state

    st1 = init_dynamic_ice_state((4, 6), S_ice_init=0.0)
    h = jnp.zeros((4, 6)).at[0, 0].set(2.0)
    conc = jnp.zeros((4, 6)).at[0, 0].set(0.8)
    st1 = st1._replace(
        h_ice=st1.h_ice.replace(data=h),
        concentration=st1.concentration.replace(data=conc))
    st5 = distribute_dynamic_state_to_categories(st1, 5)
    slab = dynamic_to_slab(st5)
    assert slab.h_ice.data.shape == (4, 6)
    assert slab.concentration.data.shape == (4, 6)
    assert float(slab.concentration.data[0, 0]) == pytest.approx(0.8)
    assert float(slab.h_ice.data[0, 0]) == pytest.approx(2.0)
