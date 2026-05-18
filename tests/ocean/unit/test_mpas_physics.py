"""Direct unit tests for the MPAS ocean physics factory.

Targets ``legoesm.ocean.physics.mpas_physics.make_mpas_ocean_physics``.
Until 2026-04-28 the factory only handled
``surface_forcing.scheme == "prescribed"``; the global_overturning
experiment configures ``scheme = "combined"`` (prescribed wind + SST
restoring), and that case used to fall through to a zero-tendency
no-op silently — see ``scripts/global_overturning/run_global_overturning_mpas_baseline.py``.

These tests pin the four schemes the factory is now contracted to
support (none, prescribed, restoring, combined) and assert that
unrecognized schemes raise ``NotImplementedError`` rather than no-op.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    RestoringConfig,
    SurfaceForcingConfig,
)


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_coord():
    return create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0
    )


@pytest.fixture(scope="module")
def state(mesh, z_coord):
    return rest_state_mpas_ocean(
        mesh, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=500.0, land_lat_threshold=85.0,
    )


def _physics_config(scheme: str) -> OceanPhysicsConfig:
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme=scheme,
            prescribed=PrescribedForcingConfig(
                wind_profile="two_belt", tau_max=0.1,
            ),
            restoring=RestoringConfig(
                tau_T=30.0 * 86400.0,
                tau_S=1.0e30,           # disabled
                T_star_eq=25.0,
                T_star_pole=0.0,
                S_star=35.0,
                T_profile="cosine",
            ),
        ),
    )


def _du_dT_max(tend):
    return (
        float(jnp.max(jnp.abs(tend.du_dt.data))),
        float(jnp.max(jnp.abs(tend.dT_dt.data))),
    )


class TestSurfaceForcingDispatch:
    """Each documented scheme must exercise the right tendency branches."""

    def test_none_yields_zero_tendencies(self, mesh, z_coord, state):
        fn = make_mpas_ocean_physics(_physics_config("none"))
        du_max, dT_max = _du_dT_max(fn(state, mesh, z_coord))
        assert du_max == 0.0
        assert dT_max == 0.0

    def test_prescribed_drives_wind_only(self, mesh, z_coord, state):
        fn = make_mpas_ocean_physics(_physics_config("prescribed"))
        du_max, dT_max = _du_dT_max(fn(state, mesh, z_coord))
        assert du_max > 0.0, "wind stress branch must produce du_dt"
        assert dT_max == 0.0, "prescribed (Q_net=0) must not move T"

    def test_restoring_drives_T_only(self, mesh, z_coord, state):
        fn = make_mpas_ocean_physics(_physics_config("restoring"))
        du_max, dT_max = _du_dT_max(fn(state, mesh, z_coord))
        assert du_max == 0.0, "restoring scheme must not produce du_dt"
        assert dT_max > 0.0, "SST restoring branch must produce dT_dt"

    def test_combined_drives_both_wind_and_restoring(self, mesh, z_coord, state):
        """Regression for the silent no-op bug fixed 2026-04-28.

        Prior to the fix, scheme='combined' fell past the prescribed
        branch and returned zero tendencies — the global_overturning
        MPAS baseline ran for a sim-year with bit-identical state.
        """
        fn = make_mpas_ocean_physics(_physics_config("combined"))
        du_max, dT_max = _du_dT_max(fn(state, mesh, z_coord))
        assert du_max > 0.0, "combined must apply wind stress"
        assert dT_max > 0.0, "combined must apply restoring"


class TestUnsupportedScheme:
    def test_unknown_scheme_raises_not_implemented(self):
        cfg = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="bulk_formulas")
        )
        with pytest.raises(NotImplementedError, match="bulk_formulas"):
            make_mpas_ocean_physics(cfg)


class TestRestoringMaskApplied:
    def test_dT_dt_zero_on_pure_land_cells(self, mesh, z_coord, state):
        """Restoring must respect land_mask — land cells get zero dT/dt
        even when the mesh's grid_lat would target a non-zero T_star."""
        fn = make_mpas_ocean_physics(_physics_config("restoring"))
        tend = fn(state, mesh, z_coord)
        mask = state.land_mask.data  # 1=ocean, 0=land
        # On any land cell (mask==0), the surface-layer dT/dt must be zero.
        land_idx = jnp.where(mask < 0.5, size=mask.shape[0], fill_value=-1)[0]
        # Only check actual land cells (fill_value=-1 entries are padding)
        land_idx = land_idx[land_idx >= 0]
        if land_idx.size > 0:
            land_dT = tend.dT_dt.data[land_idx, 0]
            assert float(jnp.max(jnp.abs(land_dT))) == 0.0


def _convection_physics_config(scheme: str) -> OceanPhysicsConfig:
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme=scheme,
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5),
        ),
    )


def _flip_state_to_unstable(state):
    """Invert the surface-layer T to make the column statically unstable.

    Default rest_state has T_water_init_C > T_deep (stable).  Setting the top
    layer cooler than the second layer creates an N² < 0 condition that
    the enhanced_diffusion branch must detect.
    """
    T = state.T.data.copy() if hasattr(state.T.data, "copy") else jnp.array(state.T.data)
    T = jnp.array(state.T.data).at[:, 0].set(0.0)  # 0 °C surface
    return state._replace(T=Field(data=T, name="T", dims=state.T.dims, units=state.T.units))


class TestConvectionDispatch:
    """Convective adjustment routing for ``scheme='enhanced_diffusion'``."""

    def test_none_yields_zero_T_tendency(self, mesh, z_coord, state):
        unstable = _flip_state_to_unstable(state)
        fn = make_mpas_ocean_physics(_convection_physics_config("none"))
        tend = fn(unstable, mesh, z_coord)
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0

    def test_enhanced_diffusion_fires_on_unstable_column(
        self, mesh, z_coord, state,
    ):
        """Surface=0°C, k=1=2°C → unstable; K_conv must mix surface+k=1."""
        unstable = _flip_state_to_unstable(state)
        fn = make_mpas_ocean_physics(
            _convection_physics_config("enhanced_diffusion"))
        tend = fn(unstable, mesh, z_coord)
        # Some ocean cell in the surface layer must have non-zero dT/dt.
        ocean = state.land_mask.data > 0.5
        dT_top = tend.dT_dt.data[:, 0]
        max_ocean_dT = float(jnp.max(jnp.abs(jnp.where(ocean, dT_top, 0.0))))
        assert max_ocean_dT > 0.0, (
            "enhanced_diffusion convection must produce non-zero dT/dt "
            "on an unstable column"
        )

    def test_stable_column_no_convection(self, mesh, z_coord, state):
        """Default rest_state is stable (T_water_init_C=20 > T_deep=2).  With
        K_bg=0 the only source of vertical mixing is the convection
        branch when N²<0 — so dT/dt must be zero (or below numerical
        noise) for a strictly stable column."""
        cfg = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=1.0, K_bg=0.0,
                ),
            ),
        )
        fn = make_mpas_ocean_physics(cfg)
        tend = fn(state, mesh, z_coord)
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) < 1e-12

    def test_unsupported_convection_scheme_raises(self):
        cfg = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            convection=OceanConvectionConfig(scheme="plume"),
        )
        with pytest.raises(NotImplementedError, match="plume"):
            make_mpas_ocean_physics(cfg)

    def test_convection_respects_land_mask(self, mesh, z_coord, state):
        unstable = _flip_state_to_unstable(state)
        fn = make_mpas_ocean_physics(
            _convection_physics_config("enhanced_diffusion"))
        tend = fn(unstable, mesh, z_coord)
        mask = state.land_mask.data
        land_idx = jnp.where(mask < 0.5, size=mask.shape[0], fill_value=-1)[0]
        land_idx = land_idx[land_idx >= 0]
        if land_idx.size > 0:
            assert float(jnp.max(jnp.abs(tend.dT_dt.data[land_idx]))) == 0.0
