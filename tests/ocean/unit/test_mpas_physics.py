"""Direct unit tests for the MPAS ocean physics factory.

Targets ``legoesm.ocean.physics.mpas_physics.make_mpas_ocean_physics``.
Until 2026-04-28 the factory only handled
``surface_forcing.scheme == "prescribed"``; the global_overturning
experiment configures ``scheme = "combined"`` (prescribed wind + SST
restoring), and that case used to fall through to a zero-tendency
no-op silently — see ``scripts/run/global_overturning/run_global_overturning_mpas_baseline.py``.

These tests pin the four schemes the factory is now contracted to
support (none, prescribed, restoring, combined) and assert that
unrecognized schemes raise ``NotImplementedError`` rather than no-op.
"""

from __future__ import annotations

import pytest
import numpy as np
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

    def test_unimplemented_vmix_scheme_raises_on_mpas(self):
        """Finding #4: a vertical_mixing scheme MPAS does not wire in
        (richardson, or a typo) must RAISE NotImplementedError, not just warn
        and silently drop the K-profile (which runs DIFFERENT physics than
        requested).  'none'/'kpp'/'tke' are supported ('tke' via the diagnostic
        quasi-steady closure + implicit solver, covered in test_mpas_tke.py);
        'catke' is rejected separately (ValueError)."""
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        # 'richardson' computes a scheme-specific K-profile MPAS would silently
        # drop; a typo must also raise.  ('constant' is the DEFAULT, handled by
        # the MPASOceanConfig background -> accepted, tested below; 'tke' is now
        # SUPPORTED via make_tke_profiles_mpas and is tested in test_mpas_tke.py,
        # incl. its implicit-required ValueError guard.)
        for bad in ("richardson", "kpp_typo"):
            cfg = OceanPhysicsConfig(
                surface_forcing=SurfaceForcingConfig(scheme="none"),
                vertical_mixing=VerticalMixingConfig(scheme=bad),
            )
            with pytest.raises(NotImplementedError, match="vertical_mixing"):
                make_mpas_ocean_physics(cfg)

    def test_supported_vmix_schemes_do_not_raise_on_mpas(self):
        """'none', 'kpp' and 'constant' (the default, via the MPAS background
        A_v/K_v + implicit solver) must construct without raising — regression
        guard so the finding-#4 raise does not over-reach onto the default path."""
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        for ok in ("none", "kpp", "constant"):
            cfg = OceanPhysicsConfig(
                surface_forcing=SurfaceForcingConfig(scheme="none"),
                vertical_mixing=VerticalMixingConfig(scheme=ok),
            )
            assert make_mpas_ocean_physics(cfg) is not None

    def test_default_physics_config_constructs_on_mpas(self):
        """The DEFAULT OceanPhysicsConfig (vertical_mixing.scheme='constant')
        MUST construct — finding #4 must not break the production MPAS path."""
        assert make_mpas_ocean_physics(OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="none"))) is not None

    def test_nondefault_constant_vmix_raises_on_mpas(self):
        """Codex review #3: a NON-DEFAULT VerticalMixingConfig.constant.A_v/K_v
        would be SILENTLY ignored on MPAS (which reads MPASOceanConfig.A_v/K_v),
        so it must RAISE.  The DEFAULT constant config (which matches the MPAS
        background) still constructs (asserted in the supported-schemes test)."""
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig, ConstantVerticalMixingConfig,
        )
        for over in (dict(A_v=5e-3), dict(K_v=9e-4), dict(A_v=2e-3, K_v=2e-4)):
            cfg = OceanPhysicsConfig(
                surface_forcing=SurfaceForcingConfig(scheme="none"),
                vertical_mixing=VerticalMixingConfig(
                    scheme="constant",
                    constant=ConstantVerticalMixingConfig(**over)),
            )
            with pytest.raises(NotImplementedError, match="MPASOceanConfig"):
                make_mpas_ocean_physics(cfg)

    def test_convective_momentum_viscosity_rejected_on_mpas(self):
        """MPAS convective adjustment is tracer-only: nonzero nu_conv/nu_bg
        must raise rather than being silently dropped (the edge-normal
        momentum would need a TRiSK cell->edge reconstruction)."""
        cfg = OceanPhysicsConfig(
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=1.0),
            ),
        )
        with pytest.raises(ValueError, match="unsupported on MPAS"):
            make_mpas_ocean_physics(cfg)

    def test_tracer_only_convection_accepted_on_mpas(self):
        """nu_conv = nu_bg = 0 (default) is accepted — tracer-only mixing."""
        cfg = OceanPhysicsConfig(
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=1.0, nu_conv=0.0, nu_bg=0.0,
                ),
            ),
        )
        # Construction must not raise.
        assert make_mpas_ocean_physics(cfg) is not None


class TestExternalSchemeTwoWay:
    """F11 phase 3: the MPAS 'external' scheme applies a coupler-provided
    OceanSurfaceForcing (tau / q_net / real salt_flux) — the MPAS analogue of
    the cubed-sphere 'external' scheme.  Freshwater (eta + virtual salt) is the
    step(freshwater=) arg's job, so surface_forcing.freshwater is IGNORED here
    (no double count)."""

    def _sf(self, state, **kw):
        from legoesm.ocean.state import OceanSurfaceForcing
        nCells = state.T.data.shape[0]
        z = jnp.zeros(nCells)
        return OceanSurfaceForcing(
            sw_down=kw.get("sw_down"), q_net=kw.get("q_net"),
            tau_x=kw.get("tau_x"), tau_y=kw.get("tau_y"),
            freshwater=kw.get("freshwater"), salt_flux=kw.get("salt_flux"))

    def test_external_scheme_is_accepted(self, mesh, z_coord, state):
        # Must not raise NotImplementedError (was unsupported before phase 3).
        fn = make_mpas_ocean_physics(_physics_config("external"))
        assert callable(fn)

    def test_external_without_surface_forcing_fails_closed(self, mesh, z_coord, state):
        """scheme='external' with no OceanSurfaceForcing must RAISE, not
        silently drop all coupling fluxes (whole-run coupling failure)."""
        fn = make_mpas_ocean_physics(_physics_config("external"))
        with pytest.raises(ValueError, match="external.*requires an"):
            fn(state, mesh, z_coord, surface_forcing=None)

    def test_kpp_freshwater_is_conservative_no_double_count(self, mesh, z_coord, state):
        """With explicit KPP, surface_forcing.freshwater drives KPP buoyancy +
        a NON-LOCAL salinity redistribution whose COLUMN INTEGRAL is zero — it
        injects no net surface salt, so it does NOT double-count the
        step(freshwater=) virtual-salt path.  Pins the no-double-count contract
        on the KPP path for DEEP columns (the sea-ice coupling regime); the
        shallow partial-seafloor case is a documented pre-existing caveat of the
        freshwater (virtual-salt) non-local term.  (Real salt_flux is kept out
        of the non-local term entirely — buoyancy only — so REAL salt mass stays
        exact; see test_kpp_real_salt_is_buoyancy_only_mass_conservative.)"""
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.vertical import compute_layer_thickness
        from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
        cfg = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="external"),
            vertical_mixing=VerticalMixingConfig(scheme="kpp"))
        fn = make_mpas_ocean_physics(cfg)
        mask = state.land_mask.data
        # Impose a vertical salinity gradient so the non-local term is active.
        S = state.S.data * 0.0 + 35.0 + jnp.linspace(0.0, 1.0, state.S.data.shape[1])[None, :]
        st = state._replace(S=state.S.replace(data=S))
        sf = OceanSurfaceForcing(
            sw_down=None, q_net=None, tau_x=None, tau_y=None,
            freshwater=mask * 2.0e-4, salt_flux=None)
        tend = fn(st, mesh, z_coord, surface_forcing=sf)
        h = compute_layer_thickness(st.eta.data, st.H_bathy.data, z_coord)
        col_int = jnp.sum(tend.dS_dt.data * h, axis=-1)
        # KPP salinity tendency conserves column salt (redistribution only).
        assert float(jnp.max(jnp.abs(jnp.where(mask > 0.5, col_int, 0.0)))) < 1e-12, (
            "KPP freshwater salinity tendency must have zero column integral "
            "(redistribution, not a net surface source) to avoid double count")

    def _kpp_external_model(self, mesh, z_coord):
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
        cfg = MPASOceanConfig(
            physics=OceanPhysicsConfig(
                surface_forcing=SurfaceForcingConfig(scheme="external"),
                vertical_mixing=VerticalMixingConfig(scheme="kpp")),
            implicit_vertical_mixing=False)
        return MPASOceanModel(mesh, z_coord, config=cfg), cfg

    def test_kpp_salt_only_does_not_trigger_nonlocal_salinity_fallback(
            self, mesh, z_coord, state):
        """Real salt_flux feeds KPP BUOYANCY only; with salt_flux but NO
        freshwater the KPP non-local salinity flux Q_sfc_S must be an explicit
        ZERO — NOT left None (which makes kpp_vertical_mixing DIAGNOSE a
        non-local salinity flux from the near-surface gradient, re-injecting a
        KPP salt term).  Discriminator on a NON-UNIFORM-S, convectively-active
        (strong brine -> B_f>0) DEEP column: salt-only tendencies must EQUAL the
        tendencies with an explicit ZERO freshwater field (which forces
        Q_sfc_S=0).  A None-fallback diagnoses a nonzero non-local salt term and
        the two diverge."""
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.vertical import compute_layer_thickness
        mask = state.land_mask.data
        # Non-uniform salinity so the diagnosed-fallback would be nonzero.
        S_grad = (jnp.full_like(state.S.data, 34.0)
                  + 1.5 * jnp.linspace(0.0, 1.0, state.S.data.shape[1])[None, :])
        st = state._replace(S=state.S.replace(data=S_grad))
        model, cfg = self._kpp_external_model(mesh, z_coord)
        salt_flux = jnp.where(mask > 0.5, 5.0e-3, 0.0)  # strong brine -> unstable
        base = dict(sw_down=None, q_net=None, tau_x=mask * 0.05,
                    tau_y=jnp.zeros_like(mask), salt_flux=salt_flux)
        dS_salt_only = model.tendencies(
            st, surface_forcing=OceanSurfaceForcing(freshwater=None, **base)).dS_dt.data
        dS_zero_fw = model.tendencies(
            st, surface_forcing=OceanSurfaceForcing(
                freshwater=jnp.zeros_like(mask), **base)).dS_dt.data
        # KPP is active: it redistributes the salinity gradient (interior dS≠0).
        assert float(jnp.max(jnp.abs(dS_salt_only[:, 1:]))) > 0.0, "KPP did not run"
        # salt-only must NOT diagnose a non-local salt flux: it must match the
        # explicit-zero-freshwater case (Q_sfc_S=0).
        assert float(jnp.max(jnp.abs(dS_salt_only - dS_zero_fw))) < 1e-12, (
            "salt-only KPP diagnosed a non-local salinity flux (Q_sfc_S left "
            "None) instead of using an explicit zero")
        # Real-salt mass conserved: column integral == explicit floored-h_k source.
        h = compute_layer_thickness(
            st.eta.data, st.H_bathy.data, z_coord,
            min_water_column_m=cfg.min_water_column_m)
        col_int = jnp.sum(dS_salt_only * h, axis=-1)
        expected = salt_flux * 1.0e3 / float(cfg.rho_0)
        rel_err = jnp.where(
            mask > 0.5, jnp.abs(col_int - expected) / jnp.maximum(expected, 1e-30), 0.0)
        assert float(jnp.max(rel_err)) < 1e-5, (
            "real-salt mass not conserved: net column salt tendency must equal "
            "the explicit floored-h_k salt source")

    def test_salt_flux_fails_closed_under_non_external_scheme(self, mesh, z_coord, state):
        """surface_forcing.salt_flux must only be consumed under the coupler-
        driven 'external' (or 'none') scheme.  Passing it under 'restoring'
        (the ocean has its own forcing) must RAISE, not silently inject salt."""
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        cfg = MPASOceanConfig(physics=_physics_config("restoring"))
        model = MPASOceanModel(mesh, z_coord, config=cfg)
        mask = state.land_mask.data
        sf = self._sf(state, salt_flux=mask * 1.0e-4)
        with pytest.raises(ValueError, match="salt_flux.*scheme"):
            model.tendencies(state, surface_forcing=sf)

    def test_kpp_sees_real_salt_buoyancy(self, mesh, z_coord, state):
        """The MPAS KPP adapter must feed the real salt_flux into its surface
        buoyancy (destabilizing), matching the lat-lon KPP path.  Adding brine
        salt deepens/strengthens boundary-layer mixing, so the KPP-driven
        INTERIOR temperature tendency (mixing the existing T gradient) changes
        vs no salt — proving KPP 'sees' the salt buoyancy, not just the
        surface salinity source."""
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
        cfg = OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="external"),
            vertical_mixing=VerticalMixingConfig(scheme="kpp"))
        fn = make_mpas_ocean_physics(cfg)
        mask = state.land_mask.data
        # Need some wind stress so KPP has a boundary layer to deepen.
        base = dict(tau_x=mask * 0.05, tau_y=jnp.zeros_like(mask))
        sf_no_salt = OceanSurfaceForcing(
            sw_down=None, q_net=None, freshwater=None, salt_flux=None, **base)
        sf_salt = OceanSurfaceForcing(
            sw_down=None, q_net=None, freshwater=None,
            salt_flux=mask * 5.0e-3, **base)  # strong brine -> destabilizing
        dT_no = fn(state, mesh, z_coord, surface_forcing=sf_no_salt).dT_dt.data[:, 1:]
        dT_salt = fn(state, mesh, z_coord, surface_forcing=sf_salt).dT_dt.data[:, 1:]
        assert float(jnp.max(jnp.abs(dT_salt - dT_no))) > 1e-12, (
            "MPAS KPP did not respond to real salt_flux buoyancy "
            "(interior mixing unchanged)")

    def _model(self, mesh, z_coord):
        """MPAS model with external surface forcing + no KPP (isolates the real
        salt SOURCE, which is applied in mpas_ocean_baroclinic_tendencies with
        the canonical floored h_k[:,0])."""
        from legoesm.ocean.mpas_config import MPASOceanConfig
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
        cfg = MPASOceanConfig(physics=OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="external"),
            vertical_mixing=VerticalMixingConfig(scheme="none")))
        return MPASOceanModel(mesh, z_coord, config=cfg)

    def test_real_salt_flux_salinifies_top_layer(self, mesh, z_coord, state):
        model = self._model(mesh, z_coord)
        mask = state.land_mask.data
        sf = self._sf(state, salt_flux=mask * 1.0e-4)  # +salt into ocean
        tend = model.tendencies(state, surface_forcing=sf)
        dS0 = tend.dS_dt.data[:, 0]
        # Real salt source raises SSS on ocean cells; land cells untouched.
        assert float(jnp.max(dS0)) > 1e-9, "salt_flux did not salinify SSS"
        assert float(jnp.max(jnp.abs(jnp.where(mask < 0.5, dS0, 0.0)))) == 0.0

    def test_real_salt_flux_is_mass_conservative_on_partial_top_cell(
            self, mesh, z_coord, state):
        """On a partial-cell column whose TOP cell is shallow (H_bathy <
        dz_ref[0]), the real salt source must inject exactly salt_flux of salt
        MASS — dS_dt[0]*rho_0*h_top/1e3 == salt_flux — using the SAME canonical
        floored top-layer thickness h_k[:,0] the tracer update integrates mass
        against (NOT dz_ref[0]*jacobian, which would mis-scale on partial/floored
        top cells)."""
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.ocean.vertical import (
            create_partial_cell_coordinate, compute_layer_thickness)
        mask = state.land_mask.data
        # Make a few OCEAN columns shallow so their top cell is partial
        # (5 m << dz_ref[0]=20 m); leave the rest deep.
        H = state.H_bathy.data
        shallow = (mask > 0.5) & (jnp.arange(H.shape[0]) % 7 == 0)
        H_new = jnp.where(shallow, 5.0, H)
        st = state._replace(
            H_bathy=state.H_bathy.replace(data=H_new),
            eta=state.eta.replace(data=jnp.zeros_like(state.eta.data)))
        pc = create_partial_cell_coordinate(z_coord, H_new)
        model = self._model(mesh, pc)
        salt_flux = jnp.where(shallow, 1.0e-4, 0.0)
        sf = OceanSurfaceForcing(
            sw_down=None, q_net=None, tau_x=None, tau_y=None,
            freshwater=None, salt_flux=salt_flux)
        tend = model.tendencies(st, surface_forcing=sf)
        # Use the SAME floored thickness the model integrates mass against.
        h0 = compute_layer_thickness(
            st.eta.data, H_new, pc,
            min_water_column_m=model.config.min_water_column_m)[:, 0]
        injected = tend.dS_dt.data[:, 0] * float(model.config.rho_0) * h0 / 1.0e3
        err = jnp.where(shallow, jnp.abs(injected - salt_flux), 0.0)
        assert float(jnp.max(err)) < 1e-12, (
            "partial-top-cell salt source not mass-conservative: injected salt "
            "mass must equal salt_flux (uses the canonical floored top thickness)")

    def test_tau_and_qnet_drive_momentum_and_heat(self, mesh, z_coord, state):
        fn = make_mpas_ocean_physics(_physics_config("external"))
        mask = state.land_mask.data
        sf = self._sf(state, tau_x=mask * 0.1, tau_y=jnp.zeros_like(mask),
                      q_net=mask * 50.0)
        tend = fn(state, mesh, z_coord, surface_forcing=sf)
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0, "tau drove no current"
        assert float(jnp.max(tend.dT_dt.data[:, 0])) > 1e-9, "q_net>0 did not warm"

    def test_surface_forcing_freshwater_is_not_a_salinity_source(self, mesh, z_coord, state):
        """MPAS delivers freshwater salinity via step(freshwater=); neither the
        physics external block NOR the real-salt source may treat
        surface_forcing.freshwater as a salinity source (else the virtual-salt
        dilution double-counts).  With ONLY surface_forcing.freshwater set (no
        freshwater= arg, no KPP), dS_dt must be exactly zero."""
        model = self._model(mesh, z_coord)
        mask = state.land_mask.data
        sf = self._sf(state, freshwater=mask * 2.0e-4)  # ONLY freshwater set
        tend = model.tendencies(state, surface_forcing=sf)
        assert float(jnp.max(jnp.abs(tend.dS_dt.data))) == 0.0, (
            "surface_forcing.freshwater must NOT be a salinity source "
            "(freshwater salinity is the step(freshwater=) arg's job)")


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


class TestExternalRGBChlPenetration:
    """--sw-rgb-chl on the MPAS lane: when the coupler attaches a surface
    chlorophyll field the 'external' scheme MUST route the penetrating SW
    through the SHARED NEMO RGB kernel (apply_shortwave_penetration,
    scheme='rgb_chl') instead of the fixed two-band Jerlov profile.  The
    decisive no-silent-no-op check: the SW temperature tendency DIFFERS
    between chl=None (Jerlov) and a nonzero chl (RGB), proving chl reaches
    the MPAS penetration kernel."""

    def _sw_sf(self, state, *, chl):
        from legoesm.ocean.state import OceanSurfaceForcing
        mask = state.land_mask.data
        return OceanSurfaceForcing(
            sw_down=mask * 200.0,        # W/m^2 incident SW
            q_net=mask * 100.0,          # W/m^2 net heat
            tau_x=None, tau_y=None,
            freshwater=None, salt_flux=None,
            chl=chl)

    def test_chl_changes_mpas_sw_penetration(self, mesh, z_coord, state):
        fn = make_mpas_ocean_physics(_physics_config("external"))
        mask = state.land_mask.data

        dT_jerlov = fn(state, mesh, z_coord,
                       surface_forcing=self._sw_sf(state, chl=None)).dT_dt.data
        dT_rgb = fn(state, mesh, z_coord,
                    surface_forcing=self._sw_sf(
                        state, chl=mask * 0.3)).dT_dt.data  # 0.3 mg/m^3

        assert bool(jnp.all(jnp.isfinite(dT_jerlov)))
        assert bool(jnp.all(jnp.isfinite(dT_rgb)))
        # Both schemes must actually heat the column (SW is nonzero).
        assert float(jnp.max(jnp.abs(dT_jerlov))) > 0.0, "Jerlov SW did not heat"
        assert float(jnp.max(jnp.abs(dT_rgb))) > 0.0, "RGB SW did not heat"
        # Non-vacuity: chl must reach the kernel -> the two profiles DIFFER.
        max_diff = float(jnp.max(jnp.abs(dT_rgb - dT_jerlov)))
        assert max_diff > 1e-9, (
            "chl did not change the MPAS SW penetration profile — the RGB "
            "kernel was not reached (silent no-op)")
