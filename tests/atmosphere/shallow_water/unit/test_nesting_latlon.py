"""One-way (parent -> child) lat-lon nesting: geometry, conservation, AD.

Covers the substrate grid construct (:mod:`legoesm.grids.nesting`), the
capability wiring (``instantiate(..., nesting=True)``), and the shallow-water
1-way nest stepping driver
(:mod:`legoesm.atmosphere.dynamics.shallow_water_nesting`).

The LOAD-BEARING numerical check is interior mass conservation through the nest
boundary on a Williamson-2 (steady geostrophic) flow: with the child boundary
prescribed from the exact steady parent, the child interior mass must be
conserved to ~round-off (target 1e-12 relative).

Run (x64, CPU):
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/atmosphere/shallow_water/unit/test_nesting_latlon.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.nesting import (
    NestedLatLonGrid,
    create_nested_latlon_grid,
    apply_boundary_interp,
)
from legoesm.grids.capability import instantiate, capability_matrix
from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
    williamson_test2_cgrid,
    CGridLatLonShallowWaterConfig,
)
from legoesm.atmosphere.dynamics.shallow_water_nesting import (
    NestedSWState,
    initial_nested_state,
    make_nested_stepper,
    nested_sw_step,
    step_child,
    interior_mass,
    interpolate_parent_to_child,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nest() -> NestedLatLonGrid:
    return create_nested_latlon_grid(
        parent_n_lat=32,
        refinement_ratio=3,
        lat_south_deg=10.0,
        lat_north_deg=50.0,
        lon_west_deg=40.0,
        lon_east_deg=120.0,
        n_halo=2,
        n_relax=4,
    )


@pytest.fixture
def fp64_policy():
    """Temporarily set fp64 STORAGE so mass is conserved to ~round-off.

    The mass-conservation guarantee is fundamentally storage-precision bound: the
    fixer's uniform-depth correction is rounded to the state dtype each step, so
    fp32 storage caps relative conservation at ~1e-7 (N*eps over the interior
    cells), while fp64 reaches ~1e-15.  The 1e-12 target therefore requires fp64
    storage (consistent with JAX_ENABLE_X64=1).  Restored after the test.
    """
    from legoesm.core.precision import get_policy, set_policy, PrecisionPolicy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


@pytest.fixture
def nest_fp64(fp64_policy) -> NestedLatLonGrid:
    return create_nested_latlon_grid(
        parent_n_lat=32, refinement_ratio=3,
        lat_south_deg=10.0, lat_north_deg=50.0,
        lon_west_deg=40.0, lon_east_deg=120.0, n_halo=2, n_relax=4,
    )


# ---------------------------------------------------------------------------
# Geometry: exact integer refinement, child nested inside parent
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_refinement_ratio(self, nest):
        r = nest.refinement_ratio
        assert r == 3
        # Child cell spacing is exactly parent/r in BOTH directions.
        assert nest.child.dlat == pytest.approx(nest.parent.dlat / r, rel=1e-12)
        assert nest.child.dlon == pytest.approx(nest.parent.dlon / r, rel=1e-12)

    def test_child_counts_are_multiples_of_r(self, nest):
        r = nest.refinement_ratio
        assert nest.child.n_lat % r == 0
        assert nest.child.n_lon % r == 0

    def test_child_inside_parent_lat_band(self, nest):
        # Child latitudes lie strictly inside the parent cell-centre band so the
        # bilinear boundary stencil never extrapolates past a pole.
        plat = np.asarray(nest.parent.lat)
        clat = np.asarray(nest.child.lat)
        assert clat.min() > plat.min()
        assert clat.max() < plat.max()

    def test_child_cell_centres_nest_in_parent(self, nest):
        # Each parent cell in the footprint splits into r child cells whose
        # centres average to the parent centre (exact centre nesting).
        r = nest.refinement_ratio
        clat = np.asarray(nest.child.lat)
        # The r-cell running means of the child centres reproduce a subset of
        # parent centres (to round-off).
        grouped = clat.reshape(-1, r).mean(axis=1)
        plat = np.asarray(nest.parent.lat)
        # every grouped value must coincide with SOME parent centre.  Tolerance
        # is float32-storage-bound (the default precision policy stores grid
        # axes as float32; the refinement itself is computed in float64).
        atol = 1e-9 if clat.dtype == np.float64 else 1e-6
        for g in grouped:
            assert np.min(np.abs(plat - g)) < atol

    def test_interior_boundary_masks_partition(self, nest):
        m_int = np.asarray(nest.interior_mask)
        m_bnd = np.asarray(nest.boundary_mask)
        assert np.allclose(m_int + m_bnd, 1.0)
        # interior is a contiguous block inset by n_halo
        nh = nest.n_halo
        assert m_int[nh:-nh, nh:-nh].all()
        assert m_int[:nh, :].sum() == 0
        assert m_int[:, :nh].sum() == 0

    def test_rejects_pole_adjacent_box(self):
        with pytest.raises(ValueError, match="margin north and south"):
            create_nested_latlon_grid(
                parent_n_lat=16, refinement_ratio=2,
                lat_south_deg=-89.9, lat_north_deg=89.9,
                lon_west_deg=0.0, lon_east_deg=90.0,
            )

    def test_rejects_bad_ratio(self):
        with pytest.raises(ValueError, match="refinement_ratio"):
            create_nested_latlon_grid(
                parent_n_lat=16, refinement_ratio=1,
                lat_south_deg=10.0, lat_north_deg=40.0,
                lon_west_deg=0.0, lon_east_deg=60.0,
            )


# ---------------------------------------------------------------------------
# Boundary interpolation operator
# ---------------------------------------------------------------------------


class TestInterp:
    def test_constant_field_exact(self, nest):
        # Bilinear interpolation of a constant parent field is exact everywhere.
        c = 7.25
        parent_field = jnp.full((nest.parent.n_lat, nest.parent.n_lon), c)
        out = apply_boundary_interp(parent_field, nest.interp_centers)
        assert out.shape == (nest.child.n_lat, nest.child.n_lon)
        assert jnp.allclose(out, c, atol=1e-10)

    def test_weights_sum_to_one(self, nest):
        for w in (nest.interp_centers, nest.interp_uface, nest.interp_vface):
            s = np.asarray(w.weights).sum(axis=-1)
            assert np.allclose(s, 1.0, atol=1e-10)

    def test_linear_lon_field_reproduced(self, nest):
        # A field LINEAR in longitude index is reproduced by bilinear interp at
        # child cell centres (bilinear is exact on bilinear data away from the
        # periodic seam, which this interior box avoids).
        plon = np.asarray(nest.parent.lon)
        field = jnp.asarray(np.broadcast_to(plon[None, :], (nest.parent.n_lat, nest.parent.n_lon)).copy())
        out = np.asarray(apply_boundary_interp(field, nest.interp_centers))
        expect = np.asarray(nest.child.lon2d)
        # interior of the child (away from any wrap) matches the child lon grid
        nh = nest.n_halo
        assert np.allclose(out[nh:-nh, nh:-nh], expect[nh:-nh, nh:-nh], atol=1e-6)


# ---------------------------------------------------------------------------
# Capability wiring
# ---------------------------------------------------------------------------


class TestCapability:
    def test_matrix_reports_latlon_nesting(self):
        mat = capability_matrix()
        assert "regional" in mat["latlon"]["nesting_extents"]
        # other grids still report no nesting
        assert mat["cubed_sphere"]["nesting_extents"] == []

    def test_instantiate_builds_nest(self):
        nest = instantiate(
            "latlon", extent="regional", nesting=True,
            parent_n_lat=24, refinement_ratio=2,
            lat_south_deg=0.0, lat_north_deg=40.0,
            lon_west_deg=0.0, lon_east_deg=80.0,
        )
        assert isinstance(nest, NestedLatLonGrid)
        assert nest.refinement_ratio == 2

    def test_instantiate_rejects_nesting_other_grid(self):
        with pytest.raises(NotImplementedError, match="nesting"):
            instantiate("cubed_sphere", extent="regional", nesting=True, n=12)

    def test_instantiate_rejects_operators_with_nesting(self):
        with pytest.raises(ValueError, match="pair"):
            instantiate(
                "latlon", extent="regional", nesting=True, operators=True,
                parent_n_lat=24, refinement_ratio=2,
                lat_south_deg=0.0, lat_north_deg=40.0,
                lon_west_deg=0.0, lon_east_deg=80.0,
            )


# ---------------------------------------------------------------------------
# Stepping: shapes, one-way boundary correctness
# ---------------------------------------------------------------------------


class TestStepping:
    def test_initial_state_shapes(self, nest):
        parent_ic = williamson_test2_cgrid(nest.parent)
        state = initial_nested_state(nest, parent_ic)
        assert isinstance(state, NestedSWState)
        assert state.child.h.shape == (nest.child.n_lat, nest.child.n_lon)
        assert state.child.u.shape == (nest.child.n_lat, nest.child.n_lon + 1)
        assert state.child.v.shape == (nest.child.n_lat + 1, nest.child.n_lon)

    def test_band_exactly_prescribed(self, nest):
        # Direct check of the prescription: step with a FIXED parent_bc and
        # confirm the band equals that bc's interpolation exactly.
        parent_ic = williamson_test2_cgrid(nest.parent)
        state = initial_nested_state(nest, parent_ic)
        cfg = CGridLatLonShallowWaterConfig(fix_mass=False)
        bc = interpolate_parent_to_child(parent_ic, nest)
        new = nested_sw_step(
            state, parent_ic, nest, 90.0, interior_mass(state.child, nest), cfg,
        )
        bmask = np.asarray(nest.boundary_mask, dtype=bool)
        assert np.allclose(
            np.asarray(new.child.h)[bmask], np.asarray(bc.h)[bmask], atol=1e-10,
        )

    def test_step_child_only_matches_combined_child(self, nest):
        # step_child (child-only, parent integrated separately) must produce the
        # SAME child as nested_sw_step's internal child update for the same
        # parent_bc — the one-way coupling decouples the parent integration.
        parent_ic = williamson_test2_cgrid(nest.parent)
        state = initial_nested_state(nest, parent_ic)
        cfg = CGridLatLonShallowWaterConfig(fix_mass=True, use_ppm_transport=True)
        tgt = interior_mass(state.child, nest)
        combined = nested_sw_step(state, state.parent, nest, 90.0, tgt, cfg)
        child_only = step_child(state.child, state.parent, nest, 90.0, tgt, cfg)
        assert jnp.allclose(combined.child.h, child_only.h, atol=1e-10)
        assert jnp.allclose(combined.child.u, child_only.u, atol=1e-10)
        assert jnp.allclose(combined.child.v, child_only.v, atol=1e-10)

    def test_relaxation_reduces_boundary_noise(self, nest):
        # The Davies relaxation zone must suppress the boundary-reflection v-wind
        # noise relative to a hard wall (n_relax=0) on the steady W2 flow.
        def max_vwind(n_relax):
            nst = create_nested_latlon_grid(
                parent_n_lat=32, refinement_ratio=3,
                lat_south_deg=10.0, lat_north_deg=50.0,
                lon_west_deg=40.0, lon_east_deg=120.0,
                n_halo=2, n_relax=n_relax,
            )
            cfg = CGridLatLonShallowWaterConfig(fix_mass=True, use_ppm_transport=True)
            st = initial_nested_state(nst, williamson_test2_cgrid(nst.parent))
            step = make_nested_stepper(nst, cfg)
            tgt = interior_mass(st.child, nst)
            for _ in range(120):
                st = step(st, st.parent, 120.0, tgt)
            vc = 0.5 * (st.child.v[:-1, :] + st.child.v[1:, :])
            return float(jnp.max(jnp.abs(vc)))

        assert max_vwind(8) < max_vwind(0)


# ---------------------------------------------------------------------------
# CONSERVATION (load-bearing): Williamson-2 interior mass to ~1e-12
# ---------------------------------------------------------------------------


class TestConservation:
    def _run(self, nest, fix_mass, nsteps=120, dt=120.0):
        parent_ic = williamson_test2_cgrid(nest.parent)
        state = initial_nested_state(nest, parent_ic)
        cfg = CGridLatLonShallowWaterConfig(
            fix_mass=fix_mass, use_ppm_transport=True,
        )
        step = make_nested_stepper(nest, cfg)
        target = interior_mass(state.child, nest)
        m0 = float(target)
        for _ in range(nsteps):
            state = step(state, state.parent, dt, target)
        m1 = float(interior_mass(state.child, nest))
        return abs(m1 - m0) / abs(m0), state

    def test_interior_mass_conserved_williamson2(self, nest_fp64):
        # LOAD-BEARING numerical check, run under fp64 storage (see fp64_policy):
        # the child free-interior mass is conserved to ~round-off on a steady
        # Williamson-2 flow through the nest boundary.
        rel_drift, _ = self._run(nest_fp64, fix_mass=True)
        assert rel_drift < 1e-12, f"interior mass rel drift {rel_drift:.3e} >= 1e-12"

    def test_fixer_measures_real_drift(self, nest):
        # With the fixer OFF the drift is small but nonzero (the boundary flux is
        # not flux-exact under bilinear prolongation); the fixer is what closes
        # it to round-off — confirming the fixer is correcting a real, bounded
        # imbalance, not masking a leak.
        rel_off, _ = self._run(nest, fix_mass=False)
        assert 0.0 < rel_off < 1e-3

    def test_parent_remains_steady(self, nest):
        # The parent integrates as a standalone global W2 (steady) flow.
        _, state = self._run(nest, fix_mass=True)
        acc = jnp.float64
        parent_ic = williamson_test2_cgrid(nest.parent)
        mp0 = float(jnp.sum(parent_ic.h.astype(acc) * nest.parent.area.astype(acc)))
        mp1 = float(jnp.sum(state.parent.h.astype(acc) * nest.parent.area.astype(acc)))
        assert abs(mp1 - mp0) / abs(mp0) < 1e-4


# ---------------------------------------------------------------------------
# Differentiability: jax.grad through one nested step is finite + nonzero
# ---------------------------------------------------------------------------


class TestDifferentiability:
    def test_grad_through_one_step_finite_nonzero(self, nest):
        parent_ic = williamson_test2_cgrid(nest.parent)
        state0 = initial_nested_state(nest, parent_ic)
        cfg = CGridLatLonShallowWaterConfig(fix_mass=True, use_ppm_transport=True)
        target = interior_mass(state0.child, nest)
        dt = 90.0

        def loss(child_h):
            child = state0.child._replace(h=child_h)
            st = NestedSWState(parent=state0.parent, child=child)
            st1 = nested_sw_step(st, state0.parent, nest, dt, target, cfg)
            # scalar functional of the child interior
            imask = nest.interior_mask.astype(st1.child.h.dtype)
            return jnp.sum((st1.child.h * imask) ** 2)

        g = jax.grad(loss)(state0.child.h)
        assert g.shape == state0.child.h.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_grad_wrt_parent_state(self, nest):
        # The one-way coupling means parent -> child boundary is differentiable:
        # d(child interior)/d(parent h) must be finite and nonzero.
        parent_ic = williamson_test2_cgrid(nest.parent)
        state0 = initial_nested_state(nest, parent_ic)
        cfg = CGridLatLonShallowWaterConfig(fix_mass=True)
        target = interior_mass(state0.child, nest)

        def loss(parent_h):
            parent = state0.parent._replace(h=parent_h)
            st = NestedSWState(parent=parent, child=state0.child)
            st1 = nested_sw_step(st, parent, nest, 90.0, target, cfg)
            return jnp.sum(st1.child.h ** 2)

        g = jax.grad(loss)(state0.parent.h)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0
