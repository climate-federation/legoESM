"""Combined ocean physics orchestrator.

Provides OceanPhysicsConfig and make_ocean_physics(), which create a
single physics function combining vertical mixing, lateral mixing,
surface forcing, bottom drag, and convection.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.state import OceanState, OceanSurfaceForcing, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.physics.tendencies import zero_ocean_tendencies

from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)

from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
from legoesm.ocean.physics.lateral_mixing.integration import make_lateral_mixing_physics
from legoesm.ocean.physics.surface_forcing.integration import make_surface_forcing_physics
from legoesm.ocean.physics.convection.integration import make_convection_physics


class OceanPhysicsConfig(NamedTuple):
    """Unified ocean physics configuration.

    Set the ``scheme`` field of any sub-config to ``"none"`` to disable
    that module entirely.
    """
    vertical_mixing: VerticalMixingConfig = VerticalMixingConfig()
    lateral_mixing: LateralMixingConfig = LateralMixingConfig()
    surface_forcing: SurfaceForcingConfig = SurfaceForcingConfig()
    bottom_drag: BottomDragConfig = BottomDragConfig()
    convection: OceanConvectionConfig = OceanConvectionConfig()
    shortwave_penetration: ShortwavePenetrationConfig | None = ShortwavePenetrationConfig()
    # Fox-Kemper mixed-layer-eddy (MLE) restratification (NEMO tramle nn_mle=1).
    # ``None`` = disabled (default).  When set, a bolus-tracer-tendency physics_fn
    # is appended; it is implemented for the lat-lon / tripole C-grid only and
    # raises ``NotImplementedError`` on a CubedSphereGrid (MPAS/cube MLE is a
    # separate follow-up).  See ``lateral_mixing.mle_latlon_cgrid``.
    mle: MLEConfig | None = None
    # Ocean-scoped physical constants (Phase G, G-C2). Defaults reference
    # legoesm.constants -> zero behaviour change. Gives the physics factories
    # access to recipe-pinned constants so the inline `constants.g` buoyancy
    # reads (KPP/TKE/k_profiles) can be de-mirrored to read config.constants.g.
    constants: ConstantsConfig = ConstantsConfig()


def _make_mle(cfg: MLEConfig) -> Callable:
    """Create the Fox-Kemper MLE bolus-tracer physics function.

    Lat-lon / tripole C-grid ONLY.  The returned ``physics_fn`` computes the
    in-situ density, N^2 and the C-grid face masks the MLE adapter needs, then
    calls :func:`mle_tracer_tendency_latlon_cgrid`.  A ``CubedSphereGrid`` (the
    cube / MPAS layout) raises ``NotImplementedError`` — MLE on those grids is a
    separate follow-up (the cube ocean is a 4-D ``(face, x, y, level)`` layout
    that the C-grid operators do not handle).
    """
    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing: OceanSurfaceForcing | None = None,
                   ) -> OceanTendencies:
        if isinstance(grid, CubedSphereGrid):
            raise NotImplementedError(
                "Fox-Kemper MLE (OceanPhysicsConfig.mle) is implemented for the "
                "lat-lon / tripole C-grid only; the CubedSphereGrid (cube / MPAS) "
                "path is a separate follow-up. Disable MLE (mle=None) for this grid."
            )
        # Deferred imports (avoid a module-load cycle through the C-grid
        # operators, and keep the cube/MPAS import path free of them).
        from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
        from legoesm.ocean.eos import (
            compute_buoyancy_frequency,
            compute_ocean_rho,
        )
        from legoesm.ocean.physics.lateral_mixing.mle_latlon_cgrid import (
            mle_tracer_tendency_latlon_cgrid,
        )
        from legoesm.ocean.physics.tendencies import wrap_ocean_tendencies
        from legoesm.ocean.vertical import (
            compute_layer_thickness,
            compute_ocean_jacobian,
        )

        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        rho = compute_ocean_rho(state, z_coord, J)
        N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, J)
        # Actual partial-cell-aware live thickness (IDENTICAL to compute_ocean_rho)
        # so the MLE MLD/buoyancy/volume are consistent over real bathymetry.
        h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
        mask = state.land_mask.data
        u_mask, v_mask = compute_face_masks(mask, grid)
        dT_dt, dS_dt = mle_tracer_tendency_latlon_cgrid(
            state.T.data, state.S.data, rho, N2,
            mask, u_mask, v_mask, z_coord, J, grid, cfg, h_k=h_k,
        )
        return wrap_ocean_tendencies(None, None, dT_dt, dS_dt, state)

    return physics_fn


def make_ocean_physics(
    config: OceanPhysicsConfig,
    apply_vertical_diffusion: bool = True,
) -> Callable:
    """Create a combined ocean physics function.

    The returned function calls each enabled physics module and sums
    their tendencies.

    Parameters
    ----------
    config : OceanPhysicsConfig
    apply_vertical_diffusion : bool
        If False, vertical mixing and the ``enhanced_diffusion``
        convection scheme return zero local-diffusion tendency; the
        dynamics step is responsible for applying their K_v/A_v
        profiles via an implicit backward-Euler solve.  Non-local
        terms (KPP counter-gradient flux) are still applied
        explicitly.  This is the mode required for
        ``LatLonCGridOceanConfig.from_flat(implicit_vertical_mixing=True)``.

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord, surface_forcing=None) -> OceanTendencies
    """
    # Tidal mixing rides on VerticalMixingConfig but is a SEPARATE caller-applied
    # additive step (ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step),
    # not part of this composition. Reject it here — before the scheme dispatch —
    # so it cannot silently no-op even when vertical_mixing.scheme == "none".
    if config.vertical_mixing.tidal.enabled:
        raise NotImplementedError(
            "VerticalMixingConfig.tidal.enabled=True is not consumed by "
            "make_ocean_physics. Tidal mixing is applied as a separate additive "
            "step via ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step "
            "(with a K_tidal field from "
            "vertical_mixing.tidal.compute_tidal_diffusivity); enable it there, "
            "not in the physics-composition config."
        )
    # zdfiwm rides the implicit K-profile path only (avm must enter the
    # backward-Euler momentum solve).  On the implicit route the host model
    # builds this composition with apply_vertical_diffusion=False and the
    # wave K is added in compute_vertical_K_profiles — legitimate.  On the
    # EXPLICIT route (apply_vertical_diffusion=True) nothing downstream
    # consumes iwm, so reject rather than silently no-op (tidal rule).
    if (apply_vertical_diffusion
            and getattr(config.vertical_mixing, "iwm", None) is not None
            and config.vertical_mixing.iwm.enabled):
        raise NotImplementedError(
            "VerticalMixingConfig.iwm.enabled=True is not consumed by the "
            "EXPLICIT physics composition.  Internal wave-driven mixing is "
            "applied inside compute_vertical_K_profiles and requires "
            "implicit_vertical_mixing=True on the host model config."
        )
    if (apply_vertical_diffusion
            and getattr(config.vertical_mixing, "ddm", None) is not None
            and config.vertical_mixing.ddm.enabled):
        raise NotImplementedError(
            "VerticalMixingConfig.ddm.enabled=True is not consumed by the "
            "EXPLICIT physics composition.  Double-diffusive mixing is applied "
            "inside compute_vertical_K_profiles (separate salt diffusivity) and "
            "requires implicit_vertical_mixing=True on the host model config."
        )

    fns = []

    if config.vertical_mixing.scheme != "none":
        fns.append(make_vertical_mixing_physics(
            config.vertical_mixing,
            apply_diffusion=apply_vertical_diffusion,
            constants_config=config.constants,
        ))
    if config.lateral_mixing.scheme != "none":
        fns.append(make_lateral_mixing_physics(config.lateral_mixing))
    if config.surface_forcing.scheme != "none":
        fns.append(make_surface_forcing_physics(config.surface_forcing))
    # Physics-level bottom drag is deprecated — use the dynamics-level
    # ``bottom_drag_r`` field on the model config instead.  The dynamics
    # path applies drag in both the baroclinic PE and the barotropic
    # substeps, which is physically correct (MOM6 convention).
    if config.bottom_drag.scheme != "none":
        raise ValueError(
            f"Physics-level bottom drag (scheme={config.bottom_drag.scheme!r}) "
            "is deprecated. Use bottom_drag_r on your model config "
            "(LatLonCGridOceanConfig or MPASOceanConfig) instead, which "
            "applies drag in both the baroclinic PE and the barotropic "
            "substeps (matching MOM6). Set BottomDragConfig(scheme='none') "
            "in your OceanPhysicsConfig."
        )
    if config.convection.scheme != "none":
        # Fail closed: a user-configured convective momentum viscosity cannot
        # be honoured under KPP (KPP owns interior momentum convection for
        # N²<0; adding enhanced_diffusion's A_v too would double-count, so it
        # is suppressed below).  Rather than silently ignore nu_conv/nu_bg,
        # reject the combination — the user must set nu_*=0 (let KPP own it)
        # or pick a non-KPP vertical_mixing scheme.
        if (config.vertical_mixing.scheme == "kpp"
                and config.convection.scheme == "enhanced_diffusion"):
            _ed = config.convection.enhanced_diffusion
            if _ed.nu_conv != 0.0 or _ed.nu_bg != 0.0:
                raise ValueError(
                    "EnhancedDiffusionConfig convective momentum viscosity "
                    "(nu_conv/nu_bg) cannot be combined with KPP vertical "
                    "mixing: KPP already enhances interior momentum where "
                    "N²<0, so applying nu_* on top would double-count and is "
                    "suppressed. Set EnhancedDiffusionConfig(nu_conv=0.0, "
                    "nu_bg=0.0) to let KPP own convective momentum, or choose "
                    "a non-KPP vertical_mixing scheme."
                )
        # Suppress convective momentum viscosity when KPP is the vertical-
        # mixing scheme: KPP already enhances interior momentum for N²<0,
        # so adding enhanced_diffusion's A_v here would double-count (the
        # A_v fields are summed below).  Mirrors the ``vmix.scheme != "kpp"``
        # gate in compute_vertical_K_profiles (the implicit fallback path).
        fns.append(make_convection_physics(
            config.convection,
            apply_diffusion=apply_vertical_diffusion,
            emit_momentum_viscosity=(config.vertical_mixing.scheme != "kpp"),
        ))

    if config.mle is not None:
        fns.append(_make_mle(config.mle))

    sw_config = config.shortwave_penetration

    def physics_fn(
        state: OceanState,
        grid: CubedSphereGrid,
        z_coord: OceanZStarCoordinate,
        surface_forcing: OceanSurfaceForcing | None = None,
    ) -> OceanTendencies:
        if not fns and sw_config is None:
            return zero_ocean_tendencies(state)

        # Sum tendencies from all enabled sub-physics modules.
        # When implicit vertical mixing is active, the vertical-mixing
        # and convection modules populate K_v / A_v on the returned
        # OceanTendencies; collect and sum them here so the dynamics
        # step can use the profiles without re-running KPP.
        K_v_sum = None
        A_v_sum = None
        if fns:
            first = fns[0](state, grid, z_coord, surface_forcing)
            du_dt = first.du_dt.data
            dv_dt = first.dv_dt.data
            dT_dt = first.dT_dt.data
            dS_dt = first.dS_dt.data
            deta_dt = first.deta_dt.data
            if first.K_v is not None:
                K_v_sum = first.K_v
            if first.A_v is not None:
                A_v_sum = first.A_v

            for fn in fns[1:]:
                t = fn(state, grid, z_coord, surface_forcing)
                du_dt = du_dt + t.du_dt.data
                dv_dt = dv_dt + t.dv_dt.data
                dT_dt = dT_dt + t.dT_dt.data
                dS_dt = dS_dt + t.dS_dt.data
                deta_dt = deta_dt + t.deta_dt.data
                if t.K_v is not None:
                    K_v_sum = t.K_v if K_v_sum is None else K_v_sum + t.K_v
                if t.A_v is not None:
                    A_v_sum = t.A_v if A_v_sum is None else A_v_sum + t.A_v
        else:
            z3 = jnp.zeros_like(state.u.data)
            z2 = jnp.zeros_like(state.eta.data)
            du_dt, dv_dt, dT_dt, dS_dt, deta_dt = z3, z3, z3, z3, z2

        # Shortwave penetration: distribute SW heating through water column.
        if (
            sw_config is not None
            and surface_forcing is not None
            and surface_forcing.sw_down is not None
        ):
            from legoesm.ocean.vertical import compute_ocean_jacobian
            J = compute_ocean_jacobian(
                state.eta.data, state.H_bathy.data, z_coord,
            )
            sw_tend = shortwave_penetration_tendency(
                surface_forcing.sw_down,
                z_coord.dz_ref,
                z_coord.z_half_ref,
                J,
                sw_config,
            )
            dT_dt = dT_dt + sw_tend

        dims_3d = state.T.dims if hasattr(state.T, 'dims') else ("face", "x", "y", "level")
        dims_2d = state.eta.dims if hasattr(state.eta, 'dims') else ("face", "x", "y")
        return OceanTendencies(
            du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
            dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
            deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
            dH_bathy_dt=Field(
                data=jnp.zeros_like(state.eta.data),
                name="dH_bathy_dt", dims=dims_2d, units="m/s",
            ),
            dland_mask_dt=Field(
                data=jnp.zeros_like(state.eta.data),
                name="dland_mask_dt", dims=dims_2d, units="1/s",
            ),
            K_v=K_v_sum,
            A_v=A_v_sum,
        )

    return physics_fn
