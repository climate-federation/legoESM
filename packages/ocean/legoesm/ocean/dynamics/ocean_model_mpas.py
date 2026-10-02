"""MPAS ocean model on Voronoi meshes.

Orchestrates split-explicit time stepping: baroclinic (slow) tendencies
computed by :func:`mpas_ocean_baroclinic_tendencies`, then barotropic
(fast) substeps update the free surface and reconcile velocities.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import cast_pytree
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.state import physics_with_constants
from legoesm.ocean.mpas_config import (
    K_ZETA_BIH_MAX_SPACING_RATIO,
    MPASOceanConfig,
    resolution_scaled_k_zeta_bih,
)
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    OceanZStarCoordinate,
    compute_layer_thickness,
    diagnose_w_from_flux_div,
    flux_form_vertical_tracer_advection,
    flux_form_vertical_tracer_advection_tvd,
)
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import min_cell_to_edge
from legoesm.ocean.dynamics.advection_mpas import (
    compute_upup_cells,
    tvd_tracer_to_edges,
)
from legoesm.core.operators_voronoi import divergence_cell_3d
from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.barotropic_mpas import (
    barotropic_substeps_mpas,
    reconcile_3d_velocity,
)
from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
    barotropic_implicit_mpas,
)
from legoesm.ocean.conservation_mpas import mpas_ocean_conservation_fixer
from legoesm.ocean.freshwater import FreshwaterForcing, freshwater_eta_tendency
from legoesm.core.operators_voronoi import tangential_velocity_3d
from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
    gm_redi_tracer_tendency_mpas,
)
from legoesm.ocean.physics.vertical_mixing import (
    implicit_vertical_diffusion_ocean,
    build_dz_half,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    masked_background_vmix_coefficient,
)


def _forward_backward_coriolis_mpas_3d(
    u_3d: jnp.ndarray,
    dt: float,
    mesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
    mask: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
) -> jnp.ndarray:
    """Apply semi-implicit Coriolis to baroclinic perturbation velocity.

    Uses the same trapezoidal predictor-corrector scheme as the barotropic
    solver (barotropic_mpas.py), but operates on the perturbation velocity
    u' = u - u_bar only.  This avoids double-counting with the barotropic
    solver's Coriolis treatment of the depth-mean flow.

    Matches the latlon C-grid pattern in _forward_backward_coriolis_3d
    (ocean_model_latlon_cgrid.py).

    Parameters
    ----------
    u_3d : (nEdges, nlev) full 3D velocity
    dt : time step [s]
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    mask : (nCells,) land mask
    eta, H_bathy : (nCells,) for layer thickness computation

    Returns
    -------
    u_3d_new : (nEdges, nlev) full velocity with Coriolis applied to perturbation
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = (mask[c1] * mask[c2])[:, jnp.newaxis]  # (nEdges, 1)

    # Layer thickness at edges for depth averaging.
    # On partial cells, MUST use min-rule so the u_bar computed here
    # matches the barotropic solver's u_bar — otherwise the depth-mean
    # we strip in u_prime = u - u_bar disagrees with what the barotropic
    # step adds back at the next iteration, and the inconsistency
    # accumulates as a phantom kick on every step edge.
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    if isinstance(z_coord, OceanPartialCellCoordinate):
        h_e = min_cell_to_edge(h_k, mesh)
    else:
        h_e = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    # ``H_e = sum(h_e)`` and ``u_bar`` numerator ``sum(u_3d * h_e)``
    # share the level axis and h_e weight — fuse into one stacked sum.
    _u_pair = jnp.sum(
        jnp.stack([h_e, u_3d * h_e], axis=-1), axis=1, keepdims=True,
    )
    H_e = jnp.maximum(_u_pair[..., 0], config.min_water_column_m)
    u_bar = _u_pair[..., 1] / H_e  # (nEdges, 1)
    u_bar = u_bar * edge_mask

    # Perturbation velocity
    u_prime = (u_3d - u_bar) * edge_mask  # (nEdges, nlev)

    # Coriolis parameter at edges
    f_e = mesh.fEdge[:, jnp.newaxis]  # (nEdges, 1)

    # Semi-implicit (trapezoidal predictor-corrector):
    # 1. Predict with old tangential velocity
    v_t_old = tangential_velocity_3d(u_prime, mesh)
    u_prime_star = (u_prime + dt * f_e * v_t_old) * edge_mask

    # 2. Correct with averaged tangential velocity
    v_t_star = tangential_velocity_3d(u_prime_star, mesh)
    u_prime_new = (u_prime + dt * f_e * 0.5 * (v_t_old + v_t_star)) * edge_mask

    # Reconstruct full velocity
    return u_prime_new + u_bar


class MPASOceanModel:
    """MPAS ocean model with split-explicit time stepping.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig or None
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        z_coord: OceanZStarCoordinate,
        config: MPASOceanConfig | None = None,
        *,
        iwm_forcing=None,
    ):
        self.mesh = mesh
        self.z_coord = z_coord
        self.config = config or MPASOceanConfig()
        self._cfl_checked = False
        self._vf_checked = False
        # Internal wave-driven mixing (zdfiwm) static forcing maps
        # (IWMForcing of de Lavergne power/decay fields on THIS mesh's
        # (nCells,) cell centres), captured as closure constants by the
        # jitted step.  Both-or-neither with the config switch — mirrors
        # the lat-lon C-grid model.
        self._iwm_forcing = iwm_forcing
        _vmix_cfg_init = (self.config.physics.vertical_mixing
                          if getattr(self.config, "physics", None) is not None
                          else None)
        _iwm_cfg_init = (getattr(_vmix_cfg_init, "iwm", None)
                         if _vmix_cfg_init is not None else None)
        self._iwm_cfg = (_iwm_cfg_init
                         if (_iwm_cfg_init is not None
                             and _iwm_cfg_init.enabled) else None)
        if iwm_forcing is not None and self._iwm_cfg is None:
            raise ValueError(
                "iwm_forcing was supplied but "
                "physics.vertical_mixing.iwm.enabled is not True — the maps "
                "would be silently ignored.")
        if (self._iwm_cfg is not None
                and not getattr(self.config, "implicit_vertical_mixing",
                                False)):
            raise ValueError(
                "vertical_mixing.iwm.enabled=True requires "
                "implicit_vertical_mixing=True on MPAS (zdfiwm contributes "
                "to the implicit avt/avm profiles).")

        # ONE MODEL, ONE SET OF CONSTANTS. This configuration carries its own
        # gravity and reference density, and its physics pipeline carries a
        # second set. Nothing reconciled them, so a card pinning the model's
        # gravity left every mixing and convection path on the library's --
        # the two halves of one model on different planets, and silently.
        #
        # The lat-lon model has refused that since it was found there, and
        # this delegates to the SAME routing helper rather than re-deriving
        # the rule. That matters for the case a first version of this check
        # got wrong: a physics pipeline still PROVABLY on the library
        # defaults is not a pin, it is an absence, and it must INHERIT the
        # model's values rather than be rejected as a conflict. Rejecting it
        # broke a real card -- an unstructured configuration pinning its
        # reference density beside an untouched physics block -- before its
        # first step.
        #
        # Traced constants stay undecidable and are left alone; that is why
        # the helper asks for a proven difference rather than using ``!=``.
        _model_cc = ConstantsConfig(g=self.config.g, rho_0=self.config.rho_0)
        _phys = getattr(self.config, "physics", None)
        if _phys is not None:
            _routed = physics_with_constants(_phys, _model_cc)
            if _routed is not _phys:
                self.config = self.config._replace(physics=_routed)
            # The pipeline's own pair is authoritative once routed: it carries
            # every constant, including the ones the model configuration has
            # no field for, so rebuilding a pair from gravity and density
            # alone would drop a pinned specific heat.
            _cc = getattr(self.config.physics, "constants", None)
            self._constants_config = _cc if _cc is not None else _model_cc
        else:
            self._constants_config = _model_cc

        # Resolution-scaled biharmonic vorticity damping.  Resolved HERE --
        # the one place every driver, experiment and test hands a mesh and a
        # configuration to the same constructor -- so no mesh can silently
        # inherit a coefficient tuned at another resolution.  Ghost edges from
        # the SPMD padding carry ``dvEdge = 0`` and ``dcEdge = 1`` (the latter
        # avoids 0/0 in the gradient), so REAL edges are selected by dvEdge --
        # selecting on dcEdge would keep every ghost at 1 m and both skew the
        # mean and trip the uniformity guard on every sharded run.
        if self.config.K_zeta_bih is None:
            _dc = np.asarray(self.mesh.dcEdge)
            _dv = np.asarray(self.mesh.dvEdge)
            _dc = _dc[(_dv > 0.0) & (_dc > 0.0)]
            if _dc.size == 0:
                raise ValueError(
                    "MPASOceanModel: K_zeta_bih=None requires a mesh with "
                    "positive dcEdge to derive the coefficient from.")
            _ratio = float(_dc.max()) / float(_dc.min())
            if _ratio > K_ZETA_BIH_MAX_SPACING_RATIO:
                raise ValueError(
                    "MPASOceanModel: K_zeta_bih=None derives ONE coefficient "
                    "from the mean cell spacing, which describes a "
                    f"quasi-uniform mesh; this mesh spans {_ratio:.1f}x in "
                    f"dcEdge (limit {K_ZETA_BIH_MAX_SPACING_RATIO}). Pin the "
                    "coefficient explicitly (MPASOceanConfig(K_zeta_bih=...) / "
                    "--mpas-k-zeta-bih) for a variable-resolution mesh.")
            # Mesh arrays are commonly stored as float32.  Accumulating their
            # mean in float32 made the configured ico6 anchor differ from the
            # model's own mean by 7.8 mm, defeating the promised exact anchor.
            _dx = float(_dc.mean(dtype=np.float64))
            self.config = self.config._replace(
                K_zeta_bih=resolution_scaled_k_zeta_bih(_dx, self.config),
                K_zeta_bih_dx_m=_dx)

        _valid_solvers = ("explicit_substep", "implicit_cn")
        if self.config.barotropic_solver not in _valid_solvers:
            raise ValueError(
                f"barotropic_solver must be one of {_valid_solvers}, "
                f"got {self.config.barotropic_solver!r}"
            )
        _valid_time_filters = ("box", "cosine")
        if self.config.barotropic_time_filter not in _valid_time_filters:
            raise ValueError(
                f"barotropic_time_filter must be one of {_valid_time_filters}, "
                f"got {self.config.barotropic_time_filter!r}"
            )
        # Loud no-op guard (mirrors the latlon C-grid model): the time filter is
        # consumed only by the explicit_substep substep (barotropic_substeps_mpas);
        # under implicit_cn it is silently inert. Warn so the no-op is visible.
        if (self.config.barotropic_time_filter != "cosine"
                and self.config.barotropic_solver != "explicit_substep"):
            import warnings
            warnings.warn(
                f"barotropic_time_filter={self.config.barotropic_time_filter!r} has "
                f"NO effect under barotropic_solver={self.config.barotropic_solver!r}"
                ": the time filter is consumed only by the explicit_substep "
                'barotropic substep. Use barotropic_solver="explicit_substep" to '
                'apply it, or leave the filter at its "cosine" default.',
                stacklevel=2)
        # #1484 codex HIGH: the conservation fixer's volume target is
        # V_new = V_old, i.e. it ASSUMES no volume source. Under
        # real_freshwater the entire freshwater signal IS a volume source, so
        # fix_volume would delete it and leave the full sum(A*F)/rho_0 as
        # residual. Refuse until the fixer takes a freshwater-aware target.
        if (self.config.freshwater_closure == "real_freshwater"
                and getattr(self.config, "use_conservation_fixer", False)
                and getattr(self.config, "fix_volume", True)):
            raise ValueError(
                'freshwater_closure="real_freshwater" is incompatible with '
                "use_conservation_fixer=True + fix_volume=True: the fixer "
                "drives V_new to V_old, which DELETES the freshwater volume "
                "source (residual = sum(A*F)/rho_0). Set fix_volume=False, or "
                "give the fixer a freshwater-aware volume target (#1484).")
        _valid_fw = ("none", "virtual_salt_flux", "real_freshwater")
        if self.config.freshwater_closure not in _valid_fw:
            raise ValueError(
                f"freshwater_closure must be one of {_valid_fw}, got "
                f"{self.config.freshwater_closure!r} (the MPAS path implements "
                "virtual_salt_flux and real_freshwater)"
            )
        # NEMO ln_rnf_depth_ini per-cell runoff spread-depth map [m]: fail fast
        # on a bad map (mirrors LatLonCGridOceanConfig validation).  A zero/
        # negative or non-finite cell would silently drop the runoff dilution
        # (the map divides the runoff by h_rnf inside the freshwater helper).
        _rdsm = getattr(self.config, "runoff_depth_spread_map", None)
        if _rdsm is not None:
            import numpy as _np
            _rdsm_np = _np.asarray(_rdsm)
            if (not _np.isfinite(_rdsm_np).all()) or (_rdsm_np <= 0).any():
                raise ValueError(
                    "runoff_depth_spread_map must be finite and > 0 everywhere "
                    "(build it with nemo_runoff_depth_map, which floors at 1 m) "
                    "— zero/negative cells would silently drop the runoff "
                    "dilution.")
        # Reserved distributed-PCG knobs (single-rank stock CG today; see
        # barotropic_implicit_mpas.py Step-4 TODO).  Validate so the
        # schema stays consistent with the lat-lon path.
        if self.config.barotropic_implicit_pcg_fixed_iters < 1:
            raise ValueError(
                "barotropic_implicit_pcg_fixed_iters must be >= 1; got "
                f"{self.config.barotropic_implicit_pcg_fixed_iters!r}"
            )
        if self.config.barotropic_implicit_pcg_residual_tol <= 0.0:
            raise ValueError(
                "barotropic_implicit_pcg_residual_tol must be > 0; got "
                f"{self.config.barotropic_implicit_pcg_residual_tol!r}"
            )

        # Precompute upwind-of-upwind cell indices for TVD advection.
        # This is a one-time mesh topology operation stored as static data.
        # "tvd" (Van Leer) and "superbee" (Sweby) share the upup stencil and
        # differ only in the flux limiter applied at the face.
        if self.config.tracer_advection in ("tvd", "superbee"):
            self._upup_pos, self._upup_neg = compute_upup_cells(mesh)
        elif self.config.tracer_advection == "upwind":
            self._upup_pos = None
            self._upup_neg = None
        else:
            raise ValueError(
                f"Unknown tracer_advection literal "
                f"{self.config.tracer_advection!r}; expected one of: "
                f"upwind, tvd, superbee."
            )

        # Model-selected EOS callable, built ONCE and threaded into every
        # density-consuming mixing path (KPP Ri/buoyancy, convective-adjustment
        # static stability) so a non-Wright EOS (e.g. nemo_seos for DINO) drives
        # the mixing decision consistently with the baroclinic dycore — matching
        # the lat-lon model's ``_vmix_eos_fn``.  ``eos_linear`` is read defensively
        # (None ⇒ make_eos_fn supplies LinearEOSConfig() defaults for eos="linear").
        # NOTE: only ``eos`` + ``eos_linear`` are model-config fields; the
        # nemo_seos / veros_* oracle coefficients use their NamedTuple DEFAULTS
        # here (DINO uses the default S-EOS set). A future custom-coefficient
        # field would need threading the matching ``eos_<scheme>`` config too.
        from legoesm.ocean.eos import make_eos_fn as _make_eos_fn
        self._eos_fn = _make_eos_fn(
            self.config.eos, getattr(self.config, "eos_linear", None))

        if self.config.physics is not None:
            self._physics_fn = make_mpas_ocean_physics(
                self.config.physics,
                implicit_vertical_mixing=self.config.implicit_vertical_mixing,
                eos_fn=self._eos_fn,
            )
        else:
            self._physics_fn = None

        # Build KPP profile function for implicit vertical mixing path.
        # When implicit_vertical_mixing=True and KPP is enabled, we need
        # the raw K profiles (not tendencies) to feed the implicit solver.
        self._kpp_profiles_fn = None
        if self.config.implicit_vertical_mixing and self.config.physics is not None:
            _vm_cfg = getattr(self.config.physics, "vertical_mixing", None)
            if _vm_cfg is not None and _vm_cfg.scheme == "kpp":
                from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
                    make_kpp_profiles_mpas,
                )
                self._kpp_profiles_fn = make_kpp_profiles_mpas(
                    _vm_cfg, eos_fn=self._eos_fn,
                    constants_config=self._constants_config,
                    iwm_applied_by_model=True)

        # Build TKE profile function for the implicit vertical mixing path.
        # Like KPP, TKE returns raw (A_v, K_v) cell profiles that feed the
        # backward-Euler implicit solver; UNLIKE KPP it has no explicit-
        # tendency path on MPAS (implicit-only — enforced in
        # make_mpas_ocean_physics).  Two modes (static on tke.prognostic):
        # diagnostic quasi-steady Mode-B (no carried field), or the NEMO
        # prognostic Mode-A carrying MPASOceanState.tke — one backward-Euler
        # en step per model step, seeded from the carry (mirrors the lat-lon
        # prognostic path).
        self._tke_profiles_fn = None
        self._tke_prognostic = False
        if self.config.implicit_vertical_mixing and self.config.physics is not None:
            _vm_cfg_tke = getattr(self.config.physics, "vertical_mixing", None)
            if _vm_cfg_tke is not None and _vm_cfg_tke.scheme == "tke":
                from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
                    make_tke_profiles_mpas,
                )
                self._tke_profiles_fn = make_tke_profiles_mpas(
                    _vm_cfg_tke, eos_fn=self._eos_fn,
                    constants_config=self._constants_config,
                    iwm_applied_by_model=True)
                self._tke_prognostic = bool(
                    getattr(_vm_cfg_tke.tke, "prognostic", False))

        # Cache convection config for implicit vertical mixing path.
        self._conv_config = None
        # The run's own constants, for the convective-adjustment density below.
        # Read from the physics config the model was built with rather than
        # left to the library default, which the comparison cards do not use.
        if self.config.implicit_vertical_mixing and self.config.physics is not None:
            _conv_cfg = getattr(self.config.physics, "convection", None)
            if _conv_cfg is not None and _conv_cfg.scheme == "enhanced_diffusion":
                self._conv_config = _conv_cfg.enhanced_diffusion
                # Implicit MPAS convection is tracer-only (the edge momentum
                # solve receives no convective viscosity).  Reject a nonzero
                # convective momentum viscosity instead of silently dropping
                # it — consistent with the explicit MPAS guard in
                # mpas_physics.make_mpas_ocean_physics.
                if (self._conv_config.nu_conv != 0.0
                        or self._conv_config.nu_bg != 0.0):
                    raise ValueError(
                        "EnhancedDiffusionConfig convective momentum "
                        "viscosity (nu_conv/nu_bg) is unsupported on MPAS: "
                        "the convective adjustment mixes tracers only (edge-"
                        "normal momentum would need a TRiSK cell->edge "
                        "reconstruction). Set EnhancedDiffusionConfig("
                        "nu_conv=0.0, nu_bg=0.0)."
                    )

    def check_barotropic_cfl(self, dt: float) -> float:
        """Check barotropic CFL and warn if marginal or unstable.

        Parameters
        ----------
        dt : float
            Baroclinic timestep [s].

        Returns
        -------
        cfl : float
            Barotropic CFL number.
        """
        import math
        import warnings

        g = self.config.g
        H_max = self.z_coord.H_max
        n_sub = self.config.n_barotropic_substeps
        # Padded ghost edges (SPMD / sharding pads) carry dcEdge=1, dvEdge=0;
        # real edges always have dvEdge>0 — exclude the pads from the minimum.
        _real = self.mesh.dvEdge > 0
        dx_min = float(jnp.min(jnp.where(_real, self.mesh.dcEdge, jnp.inf)))

        c_baro = math.sqrt(g * H_max)
        dt_baro = dt / n_sub
        cfl = c_baro * dt_baro / dx_min

        if cfl > 0.8:
            n_min = math.ceil(c_baro * dt / (0.8 * dx_min))
            warnings.warn(
                f"Barotropic CFL = {cfl:.2f} (> 0.8) — may be unstable. "
                f"c_baro={c_baro:.1f} m/s, dx_min={dx_min:.0f} m, "
                f"dt_baro={dt_baro:.1f} s. "
                f"Suggest n_barotropic_substeps >= {n_min} "
                f"(currently {n_sub}).",
                stacklevel=2,
            )
        self.check_vorticity_filter_stability(dt)
        return cfl

    def vorticity_filter_stability_number(self, dt: float) -> float:
        """``K_zeta_bih * dt * lambda_max**2`` for the EXPLICIT biharmonic
        vorticity filter (``ocean_pe_mpas`` visc term, forward-Euler in the
        momentum update).

        ``lambda_max ~ 8 / dv_min**2`` is the vertex-Laplacian spectral limit
        of a regular degree-3 dual mesh (codex review 2026-09-04 of the
        level-8 blowup), evaluated at the mesh's MINIMUM dvEdge; the number
        is 1.4 on the level-7 mesh (min dv 28.9 km, dt 150 s, K 1e14 -- ran
        180 days) and 11 on level 8 (min dv 14.4 km, dt 75 s, same K -- blew
        up in 10 steps with a per-step growth of ~5).  The classic explicit
        limit is 2.
        """
        K = float(self.config.K_zeta_bih)
        if K <= 0.0:
            return 0.0
        dv_min = float(jnp.min(jnp.where(self.mesh.dvEdge > 0.0,
                                         self.mesh.dvEdge, jnp.inf)))
        lam = 8.0 / (dv_min * dv_min)
        return K * dt * lam * lam

    def check_vorticity_filter_stability(self, dt: float) -> float:
        """Refuse an explicit vorticity filter beyond its stability limit.

        The level-8 blowup (2026-09-04) was exactly this: the NEMO-match
        recipe's fixed ``K_zeta_bih = 1e14`` crossed the explicit limit on
        the finer dual mesh and nothing checked it.  Raise at >= 2 (the
        forward-Euler limit), warn at >= 1 (level 7 sits at 1.4).
        """
        import warnings
        n = self.vorticity_filter_stability_number(dt)
        if n >= 2.0:
            raise ValueError(
                f"explicit biharmonic vorticity filter is unstable: "
                f"K_zeta_bih*dt*lambda_max^2 = {n:.2f} >= 2 (K_zeta_bih="
                f"{float(self.config.K_zeta_bih):.3g} m^4/s, dt={dt:g} s, "
                f"min dvEdge={float(jnp.min(jnp.where(self.mesh.dvEdge > 0.0, self.mesh.dvEdge, jnp.inf))):.0f} m). "
                f"Lower K_zeta_bih (scale ~dv^3 with the mesh) or set it to 0.")
        if n >= 1.0:
            warnings.warn(
                f"explicit biharmonic vorticity filter is marginal: "
                f"K_zeta_bih*dt*lambda_max^2 = {n:.2f} (limit 2).",
                stacklevel=2)
        return n

    def tendencies(
        self,
        state: MPASOceanState,
        freshwater: FreshwaterForcing | None = None,
        surface_forcing=None,
        sponge=None,
        halo_refresh=None,
    ) -> MPASOceanTendencies:
        """Compute baroclinic tendencies."""
        return mpas_ocean_baroclinic_tendencies(
            state, self.mesh, self.z_coord, self.config,
            freshwater=freshwater,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            sponge=sponge,
            halo_refresh=halo_refresh,
        )

    def _step_impl(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater: FreshwaterForcing | None = None,
        surface_forcing=None,
        sponge=None,
        halo_refresh=None,
    ) -> MPASOceanState:
        """Core step logic — no JIT wrapper.

        Use this directly inside an outer ``@jax.jit`` context (e.g.
        ``lax.scan``) to avoid nested JIT boundaries.  For standalone
        calls, use :meth:`step` which adds the ``@jax.jit`` decorator.

        Parameters
        ----------
        state : MPASOceanState
        dt : float
            Baroclinic timestep [s].
        freshwater : FreshwaterForcing or None
            Freshwater forcing (P, E, runoff, ice). If None, no freshwater.
        surface_forcing : optional
            External surface forcing passed to the physics pipeline.
        halo_refresh : MPASOceanHaloRefresh, optional
            Distributed IN-STEP packed halo refresh
            (``legoesm.parallel.voronoi_mpi.make_mpas_ocean_halo_refresh``
            — the stage-correctness lever).  The step's stencil chains
            consume more hops than the partition ``halo_depth=2`` between
            per-step entry refreshes (biharmonic two-pass operators, the
            forward-backward Coriolis on UPDATED u, every barotropic
            substep, the TVD tracer advection on UPDATED tracers),
            silently corrupting owned cells at partition boundaries.
            With this armed the halo is re-armed at each audited
            dependency frontier ([stage-halo R1/R2/R3] here, T1/T2 in
            the tendencies, B1/B2 in the barotropic substeps).  ``None``
            (the serial default) is byte-identical — every site is a
            static Python branch.

        Returns
        -------
        MPASOceanState
        """
        # Capture the INCOMING carry dtype before the compute cast: the
        # Mode-A tke store below must pin the carry back to the caller's
        # (storage) dtype, and after this line state.tke is already compute
        # dtype (codex r3 RED — pinning post-cast froze the carry at f64
        # under an f32-storage/f64-compute policy).
        _tke_in_dtype = (state.tke.data.dtype
                         if getattr(state, "tke", None) is not None else None)
        state = cast_pytree(state, None, "compute")

        config = self.config
        mesh = self.mesh
        z_coord = self.z_coord
        mask = state.land_mask.data

        # Per-level active mask.  On partial-cell coords, below-seafloor
        # cells have ``h_partial = 0`` (vertical.py:267).  The tracer
        # update divides by ``jnp.maximum(h_k_new, 1e-10)``; using only
        # the 2D land mask there lets a float-precision residual in
        # ``hT_new`` amplify to ~1e10 tracer values below the seafloor
        # — the same bug that caused step-1 blowup on lat-lon
        # (PR #231).  Even though the MPAS PGF stencils gate the
        # corrupted ρ from the active dynamics, the values still feed
        # the GM/Redi tendency and any non-active-aware diagnostic.
        # Audit 2026-05-04.
        if isinstance(z_coord, OceanPartialCellCoordinate):
            active_3d = z_coord.is_active.astype(state.T.data.dtype)
        else:
            active_3d = mask[:, jnp.newaxis]

        # 1. Compute baroclinic tendencies
        tend = self.tendencies(state, freshwater=freshwater,
                               surface_forcing=surface_forcing,
                               sponge=sponge, halo_refresh=halo_refresh)

        # 2. Update tracers (forward Euler)
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # Fill land cells with ocean-neighbor average (Neumann BC) so that
        # subsequent operators see smooth values at coastlines instead of
        # the sharp ocean-to-zero discontinuity that `* mask` would create.
        c1_m = mesh.cellsOnEdge[0]
        c2_m = mesh.cellsOnEdge[1]
        T_new = fill_land_cells_mpas(T_new, mask, c1_m, c2_m,
                                     mesh.edgesOnCell, mesh.nEdgesOnCell)
        S_new = fill_land_cells_mpas(S_new, mask, c1_m, c2_m,
                                     mesh.edgesOnCell, mesh.nEdgesOnCell)
        # 2a (coefficients). Vertical-mixing coefficients for the implicit
        # tracer solve (after the halo exchange below) and the momentum
        # solve (3a).  When KPP and/or convective adjustment are enabled,
        # their K profiles are added to the background K_v here so that ALL
        # vertical mixing goes through the unconditionally stable implicit
        # solver — no explicit CFL constraint on K_conv.  Built from the
        # start-of-step state only.
        A_v_kpp_cells = None  # KPP/TKE scheme viscosity at cells; shared with momentum solve
        if config.implicit_vertical_mixing:
            h_k_impl = compute_layer_thickness(
                state.eta.data, state.H_bathy.data, z_coord,
                min_water_column_m=config.min_water_column_m,
            )  # (nCells, nlev)
            dz_cell = jnp.maximum(h_k_impl, 1e-10)
            dz_half_cell = build_dz_half(dz_cell)
            # Build per-cell K_v that is zero at sub-seafloor interfaces.
            # On partial cells, the bottom_level gives the deepest active
            # full level; interfaces below that must have K=0 to prevent
            # the tridiagonal solver from seeing huge coefficients.
            if hasattr(z_coord, 'bottom_level'):
                # Background K_v on active interfaces, zeroed below the
                # seafloor (shared MPAS cell/edge helper — #517 item 6).
                K_v_cell, _active_half_c = masked_background_vmix_coefficient(
                    config.K_v, z_coord.bottom_level, T_new.shape[1] - 1,
                )  # (nCells, nlev-1); _active_half_c reused by convection
            else:
                K_v_cell = jnp.full(
                    (T_new.shape[0], T_new.shape[1] - 1),
                    config.K_v,
                    dtype=T_new.dtype,
                )
                _active_half_c = None

            # --- KPP K_v profile (tracer diffusivity at half-levels) ---
            if self._kpp_profiles_fn is not None:
                A_v_kpp_cells, K_v_kpp_cells = self._kpp_profiles_fn(
                    state, mesh, z_coord, surface_forcing,
                )
                K_v_cell = K_v_cell + K_v_kpp_cells

            # --- TKE K profile (diagnostic Gaspar/Burchard; implicit-only) ---
            # TKE and KPP are mutually exclusive (a single vertical_mixing.scheme),
            # so the shared ``A_v_kpp_cells`` momentum-viscosity carrier holds
            # whichever scheme is active — the momentum implicit solve (step 3a)
            # interpolates it to edges the same way for both.  A_v = K_M (TKE
            # momentum viscosity), K_v_tke = K_H (tracer diffusivity).
            if self._tke_profiles_fn is not None:
                if self._tke_prognostic:
                    # Mode A: the carry MUST be seeded before the first step
                    # (state.tke=None -> Field mid-run would change the scan
                    # carry pytree structure). Structural check — jit-safe.
                    if state.tke is None:
                        raise ValueError(
                            "prognostic TKE on MPAS requires a seeded "
                            "MPASOceanState.tke before the first step "
                            "(pytree-stable scan carry): call "
                            "model.seed_tke(state) on the initial state.")
                    A_v_kpp_cells, K_v_tke_cells, _tke_new = (
                        self._tke_profiles_fn(
                            state, mesh, z_coord, surface_forcing,
                            dt_tke=dt,
                        ))
                else:
                    A_v_kpp_cells, K_v_tke_cells = self._tke_profiles_fn(
                        state, mesh, z_coord, surface_forcing,
                    )
                    _tke_new = None
                K_v_cell = K_v_cell + K_v_tke_cells

            # --- Convective-adjustment K profile (where N²<0) ---
            if self._conv_config is not None:
                from legoesm.ocean.eos import compute_ocean_rho
                from legoesm.ocean.vertical import compute_ocean_jacobian
                _J_conv = compute_ocean_jacobian(
                    state.eta.data, state.H_bathy.data, z_coord,
                )
                _J_conv = jnp.where(mask > 0.5, _J_conv, 1.0)
                _rho_conv = compute_ocean_rho(
                    state, z_coord, _J_conv, eos_fn=self._eos_fn,
                    g=self._constants_config.g,
                    rho0=self._constants_config.rho_0)
                # Density difference at half-levels: drho > 0 ⇒ unstable
                # (denser water sits above lighter water).
                _drho = _rho_conv[:, :-1] - _rho_conv[:, 1:]  # (nCells, nlev-1)
                _cfg_c = self._conv_config
                if _cfg_c.smooth_transition:
                    _K_conv_profile = (
                        _cfg_c.K_bg
                        + (_cfg_c.K_conv - _cfg_c.K_bg)
                        * jax.nn.sigmoid(_drho * _cfg_c.sigmoid_sharpness)
                    )
                else:
                    _K_conv_profile = jnp.where(
                        _drho > 0, _cfg_c.K_conv, _cfg_c.K_bg,
                    )
                # Mask sub-seafloor interfaces
                if _active_half_c is not None:
                    _K_conv_profile = jnp.where(
                        _active_half_c, _K_conv_profile, 0.0,
                    )
                # Mask land cells
                _K_conv_profile = _K_conv_profile * mask[:, None]
                K_v_cell = K_v_cell + _K_conv_profile

            # --- NEMO zdfiwm: internal-wave-driven mixing, ADDITIVE on top
            # of the closure (zdfphy order: closure first, zdf_iwm adds onto
            # avt/avm) — same contribution and same iwm_K_profile call as
            # the lat-lon lane; K_iwm is recomputed from THIS grid's own
            # N²/geometry, only the static power maps were remapped. ---
            if self._iwm_cfg is not None:
                from legoesm.ocean.physics.vertical_mixing.k_profiles import (
                    iwm_K_profile,
                )
                _K_iwm = iwm_K_profile(
                    state, z_coord, self.config.physics, self._iwm_cfg,
                    eos_fn=self._eos_fn,
                    iwm_fields=self._iwm_forcing,
                ).astype(K_v_cell.dtype)
                if _active_half_c is not None:
                    _K_iwm = jnp.where(_active_half_c, _K_iwm, 0.0)
                _K_iwm = _K_iwm * mask[:, None]
                K_v_cell = K_v_cell + _K_iwm
                A_v_kpp_cells = (_K_iwm if A_v_kpp_cells is None
                                 else A_v_kpp_cells + _K_iwm)

        # 3. Update 3D velocity with baroclinic perturbation tendency.
        # tend.du_dt uses RELATIVE vorticity in the PV flux only (no
        # planetary Coriolis) — Coriolis on the 3D perturbation is
        # applied via forward-backward Matsuno below. Matches lat-lon
        # pattern (#160).
        u_star = state.u.data + dt * tend.du_dt.data

        # 3a. Implicit vertical viscosity (backward-Euler) on velocity.
        # Applied AFTER the explicit tendency Euler step but BEFORE the
        # Coriolis rotation — mimics the MOM6/MPAS-Ocean operator split.
        # Uses per-edge layer thickness from ``min_cell_to_edge`` on
        # partial cells so the tridiagonal system sees the actual
        # vertical grid at each edge.  Masked by ``edge_mask_3d`` to
        # zero sub-seafloor levels.
        #
        # When KPP is enabled under implicit vertical mixing, A_v_kpp
        # (at cells) is interpolated to edges and added to the background
        # A_v so the full viscosity profile goes through the implicit
        # solver.
        if config.implicit_vertical_mixing:
            c1_e = mesh.cellsOnEdge[0]
            c2_e = mesh.cellsOnEdge[1]
            edge_mask_2d = (mask[c1_e] * mask[c2_e])
            # Build per-edge, per-level mask matching tendency function
            if isinstance(z_coord, OceanPartialCellCoordinate):
                from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
                    compute_max_level_edge_bot,
                )
                bot_e = compute_max_level_edge_bot(z_coord.bottom_level, mesh)
                nlev_u = u_star.shape[1]
                k_idx = jnp.arange(nlev_u, dtype=bot_e.dtype)
                edge_mask_3d = (
                    (k_idx[None, :] <= bot_e[:, None]).astype(u_star.dtype)
                    * edge_mask_2d[:, None]
                )
            else:
                edge_mask_3d = jnp.broadcast_to(
                    edge_mask_2d[:, None], u_star.shape,
                ).astype(u_star.dtype)

            # Compute edge layer thickness for the implicit solve
            h_k_for_edge = compute_layer_thickness(
                state.eta.data, state.H_bathy.data, z_coord,
                min_water_column_m=config.min_water_column_m,
            )
            if isinstance(z_coord, OceanPartialCellCoordinate):
                h_e_impl = min_cell_to_edge(h_k_for_edge, mesh)
            else:
                h_e_impl = 0.5 * (h_k_for_edge[c1_e] + h_k_for_edge[c2_e])

            dz_edge = jnp.maximum(h_e_impl, 1e-10)
            dz_half_edge = build_dz_half(dz_edge)
            # Build per-edge A_v that is zero at sub-seafloor interfaces.
            # Without this, the tridiagonal system at sub-seafloor levels
            # (where dz=1e-10) gets coefficients ~dt*K/dz^2 ~ 3e20,
            # making the system singular and producing NaN.
            if isinstance(z_coord, OceanPartialCellCoordinate):
                # Background A_v on active interfaces, zeroed below the
                # seafloor (shared MPAS cell/edge helper — #517 item 6).
                # ``active_half_edge`` is cast to the velocity dtype to match
                # the float mask reused by the KPP-edge masking below.
                A_v_edge, _active_half_edge_b = masked_background_vmix_coefficient(
                    config.A_v, bot_e, u_star.shape[1] - 1,
                )  # (nEdges, nlev-1)
                active_half_edge = _active_half_edge_b.astype(u_star.dtype)
            else:
                A_v_edge = jnp.full(
                    (u_star.shape[0], u_star.shape[1] - 1),
                    config.A_v,
                    dtype=u_star.dtype,
                )
                active_half_edge = None

            # Add KPP viscosity at edges (interpolated from cells).
            # A_v_kpp_cells was computed in the coefficient build (2a) above.
            if A_v_kpp_cells is not None:
                A_v_kpp_edge = 0.5 * (
                    A_v_kpp_cells[c1_e] + A_v_kpp_cells[c2_e]
                )  # (nEdges, nlev-1)
                # Mask sub-seafloor interfaces
                if active_half_edge is not None:
                    A_v_kpp_edge = A_v_kpp_edge * active_half_edge
                A_v_edge = A_v_edge + A_v_kpp_edge

            u_star = implicit_vertical_diffusion_ocean(
                u_star * edge_mask_3d, A_v_edge, dz_edge, dz_half_edge, dt,
            ) * edge_mask_3d

        # [stage-halo R1+R2] ONE message re-arms both rings.  The updated
        # tracers' halo carries the NEIGHBOR rank's tendencies (masked-wrong
        # beyond the outer ring) and GM/Redi (1-2 hops) and the MLE bolus
        # consume it; u_star's ring likewise, consumed by the forward-backward
        # Coriolis (2 tangential hops).  The vertical-mixing coefficients and
        # u_star (3/3a) read only the start-of-step state and tendencies, so
        # they are built before the exchange; the tracer solve stays after it.
        if halo_refresh is not None:
            (u_star,), (T_new, S_new) = halo_refresh.both(
                (u_star,), (T_new, S_new))

        # 2a (solve). Implicit vertical tracer diffusion (backward-Euler),
        # BEFORE GM/Redi and advection, on per-cell layer thickness (partial
        # cells) so the tridiagonal system respects the seafloor.
        # Sub-seafloor levels are zeroed BEFORE the solve so it doesn't smooth
        # stale sub-seafloor values into active cells; land-fill values are
        # preserved via ``jnp.where`` so GM/Redi (which follows) sees smooth
        # coastline values.
        if config.implicit_vertical_mixing:
            # Zero sub-seafloor and land before solve (safe input)
            T_solve = T_new * active_3d
            S_solve = S_new * active_3d
            T_solved = implicit_vertical_diffusion_ocean(
                T_solve, K_v_cell, dz_cell, dz_half_cell, dt,
            )
            S_solved = implicit_vertical_diffusion_ocean(
                S_solve, K_v_cell, dz_cell, dz_half_cell, dt,
            )
            # Restore: use solved values on active cells, keep land-fill
            # values on inactive cells (needed by GM/Redi).
            T_new = jnp.where(active_3d > 0.5, T_solved, T_new)
            S_new = jnp.where(active_3d > 0.5, S_solved, S_new)

        # 2b. GM/Redi isopycnal mixing (forward Euler tendency on top of
        # the physics-stepped tracer, before advection).  Mirrors the
        # lat-lon pattern in ocean_model_latlon_cgrid.py.  Only the
        # centred scheme is implemented on MPAS (Phase 1-4 of the plan
        # at docs/ocean/experiments/gm_redi_mpas_plan.md); the triad
        # branch raises NotImplementedError.
        if config.gm_redi is not None:
            dT_gm, dS_gm = gm_redi_tracer_tendency_mpas(
                T_new, S_new, state.eta.data, state.H_bathy.data,
                mesh, z_coord, config.gm_redi,
                eos=config.eos, eos_linear=config.eos_linear,
                mask=mask,
            )
            T_new = T_new + dt * dT_gm * active_3d
            S_new = S_new + dt * dS_gm * active_3d

        # 2c. Fox-Kemper MLE submesoscale restratification (forward-Euler
        # bolus tracer tendency, same additive pattern as GM/Redi above).
        # Voronoi port of NEMO nn_mle=1 — see mle_mpas.py / the plan doc.
        if config.mle is not None:
            from legoesm.ocean.physics.lateral_mixing.mle_mpas import (
                mle_tracer_tendency_mpas,
            )
            dT_mle, dS_mle = mle_tracer_tendency_mpas(
                T_new, S_new, state.eta.data, state.H_bathy.data,
                mesh, z_coord, config.mle,
                eos=config.eos, eos_linear=config.eos_linear,
                mask=mask,
            )
            T_new = T_new + dt * dT_mle * active_3d
            S_new = S_new + dt * dS_mle * active_3d

        # 3b. Forward-backward (trapezoidal predictor-corrector) Coriolis
        # on the 3D perturbation velocity. Unconditionally stable for
        # inertial oscillations; mirrors the lat-lon
        # _forward_backward_coriolis_3d call in ocean_model_latlon_cgrid.py.
        u_baro = _forward_backward_coriolis_mpas_3d(
            u_star, dt, mesh, z_coord, config,
            mask, state.eta.data, state.H_bathy.data,
        )

        # 4. Barotropic substeps
        # The 3D baroclinic tendency has been applied to u_baro above.
        # F_slow_u (depth-mean of du_dt_full, with planetary Coriolis
        # subtracted) is passed to the barotropic solver so it can
        # apply *online evolving* f·v_t(u_bar) during each substep —
        # matching the lat-lon C-grid pattern. See ocean_pe_mpas.py and
        # barotropic_mpas.py for the split and its rationale.
        n_sub = config.n_barotropic_substeps
        dt_baro = dt / n_sub

        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]

        # Create intermediate state with updated velocity for barotropic
        state_for_baro = state._replace(
            u=state.u.replace(data=u_baro),
        )

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        if freshwater is not None and config.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(freshwater, config.rho_0) * mask

            # Global freshwater normalization: subtract the area-weighted
            # mean so the global integral of F_slow_eta is exactly zero.
            # This prevents global volume drift from unbalanced P-E+R
            # (standard OMIP practice for runs without sea ice).  Local
            # ``jnp.sum`` (exact single-rank, matching the salt virtual-salt
            # normalization which removes the same area-mean).
            #
            # MPI: this eta mean is rank-local (no owned-cell mask), as is the
            # sibling top-layer-salt mean in ``mpas_ocean_baroclinic_tendencies``
            # (ocean_pe_mpas).  A multi-rank MPAS run with normalize_freshwater is
            # refused by the SINGLE fail-fast guard at that tendency reduction
            # SOURCE (codex round-3) — which ``self.tendencies(...)`` above (step
            # 1) hits FIRST, so this eta block is never reached under MPI.  Keep
            # this local sum (single-rank-correct) until owned-mask plumbing
            # (``owned_mask`` + ``global_sum_if_distributed``) lands on both paths.
            if config.normalize_freshwater:
                # Distributed context from the refresh object (MPI layout or
                # SPMD lane): owned-masked partial sums + its cross-rank SUM.
                # Without it the historical rank-local mean stays, guarded by
                # the multi-rank refusal.
                _hr_owned_eta = getattr(halo_refresh, "owned_mask_cells", None)
                _hr_gsum_eta = getattr(halo_refresh, "global_sum", None)
                if (config.freshwater_closure == "real_freshwater"
                        and _hr_owned_eta is None):
                    # codex RED: the multi-rank refusal for freshwater
                    # normalization lives inside the virtual-salt block,
                    # which real_freshwater skips -- but this eta mean is
                    # still RANK-LOCAL, so guard it here too.
                    from legoesm.ocean.freshwater import (
                        refuse_multiprocess_eta_normalization,
                    )
                    refuse_multiprocess_eta_normalization("MPASOceanModel")
                area = mesh.areaCell
                _w_eta = area * mask
                if _hr_owned_eta is not None:
                    _w_eta = _w_eta * _hr_owned_eta.astype(_w_eta.dtype)
                # RESTORING IS EXCLUDED (codex round 2 RED; same fix as the
                # lat-lon path).  The flag removes the CORE-II P-E+R imbalance,
                # a forcing-dataset artifact; SSS restoring is not part of it,
                # and NEMO never normalizes its `erp`.  Normalizing the full
                # net would subtract the restoring's own global mean from every
                # cell -- a spurious uniform water flux AND a globally weakened
                # restoring, both silent.
                _fw_rest = getattr(freshwater, "restoring", None)
                _F_rest = (0.0 if _fw_rest is None
                           else (jnp.asarray(_fw_rest) / config.rho_0) * mask)
                _F_phys = F_slow_eta - _F_rest
                _num_l = jnp.sum(_F_phys * _w_eta)
                _den_l = jnp.sum(_w_eta)
                if _hr_gsum_eta is not None:
                    _num_l, _den_l = _hr_gsum_eta([_num_l, _den_l])
                F_mean = _num_l / jnp.maximum(_den_l, 1e-10)
                F_slow_eta = (_F_phys - F_mean * mask) + _F_rest

        F_slow_u_data = tend.F_slow_u.data if tend.F_slow_u is not None else None

        if config.barotropic_solver == "implicit_cn":
            # Single-step implicit CN free surface (no substepping, no
            # time filter).  See barotropic_implicit_mpas.py for the
            # scheme.  Eliminates the TRiSK rotational null branch
            # (Thuburn 2008; Ringler+ 2010 §6) that monotonically grows
            # in the explicit_substep run on global ico4 (#214).
            #
            # FAIL-FAST under MPI: MPAS implicit_cn is SINGLE-RANK only
            # (stock CG).  Unlike the lat-lon C-grid, the Voronoi
            # barotropic A_op does no halo exchange and its reductions
            # would double-count ghost cells, so a multi-rank run would
            # SILENTLY produce a stale-ghost / rank-local-mass solve.
            # Refuse it explicitly until the distributed Voronoi PCG lands
            # (barotropic_implicit_mpas.py Step-4 TODO).
            # ``is_multi_process()`` alone MISSES the Voronoi MPI path
            # (it builds a partition layout without arming the global
            # halo backend — codex 2026-06-11 CRITICAL); the world-size
            # check trips on any real ``mpirun -np N`` launch.
            from legoesm.parallel.reductions import (
                is_multi_process,
                mpi_world_size,
            )
            # Multi-rank now SUPPORTED when the Voronoi partition layout
            # is armed (distributed fixed-M PCG: halo-composed A_op +
            # owned-masked area-weighted dots — see
            # barotropic_implicit_mpas).  Refuse only the layout-LESS
            # multi-rank launch, where the stock-CG fallback would
            # silently run rank-local.
            from legoesm.parallel.voronoi_mpi import (
                get_active_voronoi_layout,
            )
            # The SPMD lane (voronoi_spmd_ocean) threads its owned mask +
            # psum reducer through halo_refresh instead of an MPI layout;
            # its distributed PCG is the same solver, so the layout-less
            # refusal applies only when neither context exists.
            _hr_spmd = getattr(halo_refresh, "owned_mask_cells", None) is not None
            if (get_active_voronoi_layout() is None and not _hr_spmd
                    and (is_multi_process() or mpi_world_size() > 1)):
                raise NotImplementedError(
                    "barotropic_solver='implicit_cn' under multi-rank MPAS "
                    "requires the Voronoi partition layout (call "
                    "initialize_voronoi_mpi and build the model on "
                    "layout.local_mesh); without it the solve would "
                    "silently run rank-local.  Use 'explicit_substep' "
                    "otherwise."
                )
            eta_new, u_bar_new, Hu_avg = barotropic_implicit_mpas(
                state_for_baro, mesh, z_coord, config, dt,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u_data,
                halo_refresh=halo_refresh,
            )
        else:
            eta_new, u_bar_new, Hu_avg = barotropic_substeps_mpas(
                state_for_baro, mesh, z_coord, config, dt_baro, n_sub,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u_data,
                halo_refresh=halo_refresh,
            )

        # 5. Layer thicknesses before and after barotropic
        h_k_old = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )  # (nCells, nlev)
        h_k_new = compute_layer_thickness(
            eta_new, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )  # (nCells, nlev)

        # 6. Reconcile 3D velocity
        # Compute u_bar_old from the UPDATED state (state_for_baro),
        # not the original. This ensures depth_avg(u_3d_new) = u_bar_new.
        # On partial cells, MUST use the same min-rule per-level edge
        # thickness as the implicit-CN solver (barotropic_implicit_mpas).
        # If we used a centered 0.5*(h[c1]+h[c2]) here, u_bar_old and
        # u_bar_new would be on different bases and reconcile_3d_velocity
        # would inject the difference into u_3d as a phantom barotropic
        # kick at every step edge — drove the seamount rest-state
        # explosion at step 2.
        partial_cells = isinstance(z_coord, OceanPartialCellCoordinate)
        if partial_cells:
            h_e_k = min_cell_to_edge(h_k_old, mesh)
            H_e = jnp.maximum(jnp.sum(h_e_k, axis=1),
                              config.min_water_column_m)
        else:
            h_e_k = 0.5 * (h_k_old[c1] + h_k_old[c2])  # (nEdges, nlev)
            H_total = jnp.maximum(state.eta.data + state.H_bathy.data,
                                  config.min_water_column_m)
            H_e = 0.5 * (H_total[c1] + H_total[c2])
        u_bar_old = jnp.sum(u_baro * h_e_k, axis=1) / jnp.maximum(H_e, 1e-10)

        u_3d_new = reconcile_3d_velocity(
            u_baro, u_bar_old, u_bar_new, mesh, mask,
        )
        # [stage-halo R3] ONE packed refresh of the post-solve
        # prognostics before the transport/advection block: the
        # w-diagnosis divergence (1 hop on mass_flux), the TVD tracer
        # advection (2 hops on T/S and the transport velocity), and the
        # delta_u correction's Hu_avg ring all read halos consumed or
        # updated since the last refresh.  eta_new rides along so the
        # h_k_new-derived thicknesses agree at the ring.
        if halo_refresh is not None:
            (u_3d_new, Hu_avg), (T_new, S_new, eta_new) = halo_refresh.both(
                (u_3d_new, Hu_avg), (T_new, S_new, eta_new))

        # 7. Barotropic correction for transport-consistent tracer advection
        #
        # Correct the reconciled 3D velocity so that depth-integrated
        # transport matches the time-averaged barotropic transport Hu_avg
        # exactly.  This ensures mass flux consistency between the
        # barotropic continuity equation (which produced eta_new) and
        # the tracer transport (Hallberg & Adcroft 2009, issue #145).
        #
        # The correction is a uniform (depth-independent) velocity shift:
        #   delta_u = (Hu_avg - sum_k(u_3d * h_e)) / H_e
        # This preserves baroclinic shear while matching Hu_avg.
        edge_mask = mask[c1] * mask[c2]
        # Fuse the two h_e_k-weighted column reductions into one stack.
        _hu_pair = jnp.sum(
            jnp.stack([h_e_k, u_3d_new * h_e_k], axis=-1), axis=1,
        )
        H_e_old = _hu_pair[..., 0]  # (nEdges,)
        Hu_3d = _hu_pair[..., 1]    # (nEdges,)
        delta_u = (Hu_avg - Hu_3d) / jnp.maximum(H_e_old, 1e-10)
        u_transport = u_3d_new + delta_u[:, jnp.newaxis]  # (nEdges, nlev)

        # Per-layer mass fluxes with full 3D velocity structure.
        # Preserves baroclinic shear and produces non-zero w from
        # Ekman pumping/suction (unlike uniform barotropic distribution
        # which gives w ≡ 0).
        mass_flux = h_e_k * u_transport * edge_mask[:, jnp.newaxis]

        # 8. Diagnose vertical velocity from per-layer flux divergence
        #
        # From continuity: dh_k/dt + div_h(h_k * u_k) + w_{k-1/2} - w_{k+1/2} = 0
        # We accumulate div_h(h_k * u_k) bottom-up to get w at interfaces.
        flux_div_3d = divergence_cell_3d(mass_flux, mesh)  # (nCells, nlev)

        w = diagnose_w_from_flux_div(
            flux_div_3d, z_coord, thickness_weighted=True,
        )  # (nCells, nlev+1)

        # 9. Flux-form tracer transport (horizontal + vertical)
        #
        # Both horizontal and vertical transport use the barotropic-averaged
        # per-layer mass fluxes for consistency:
        #   h_new * T_new = h_old * T_mid
        #     - dt * div_h(mass_flux * T_face_h)   [horizontal flux]
        #     - dt * (w * T_face_v)                 [vertical flux]
        #
        # T_mid contains diffusion+physics from the Euler step (step 2).
        # Advection (horizontal + vertical) is applied here.
        # This matches the latlon C-grid algorithm (ocean_model_latlon_cgrid.py).
        # ``active_3d`` (built above) is per-level on partial cells.

        use_tvd = config.tracer_advection in ("tvd", "superbee")
        if use_tvd:
            from legoesm.ocean.dynamics._flux_limiters import resolve_tvd_limiter
            limiter_fn = resolve_tvd_limiter(config.tracer_advection)

        for tr_name in ['T', 'S']:
            tr = T_new if tr_name == 'T' else S_new

            # Horizontal flux: reconstruct tracer at edges
            # MPAS convention: u > 0 means flow from c1 to c2 (edge normal).
            if use_tvd:
                tr_edge = tvd_tracer_to_edges(
                    tr, mass_flux, mesh,
                    self._upup_pos, self._upup_neg,
                    cell_active=active_3d,
                    limiter_fn=limiter_fn,
                )
            else:
                # First-order upwind
                tr_c1 = tr[c1]  # (nEdges, nlev)
                tr_c2 = tr[c2]  # (nEdges, nlev)
                tr_edge = jnp.where(mass_flux > 0, tr_c1, tr_c2)
            tracer_flux = mass_flux * tr_edge  # (nEdges, nlev)
            div_hut = divergence_cell_3d(tracer_flux, mesh)  # (nCells, nlev)

            # Vertical flux divergence
            if use_tvd:
                vert_flux_div = flux_form_vertical_tracer_advection_tvd(
                    tr, w, h_k_old, dt, cell_active=active_3d,
                    limiter_fn=limiter_fn,
                )
            else:
                vert_flux_div = flux_form_vertical_tracer_advection(tr, w)

            # Full flux-form tracer update:
            # h_new * T_new = h_old * T_mid - dt * vert - dt * horiz
            hT_new = h_k_old * tr - dt * vert_flux_div - dt * div_hut
            tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
            # Preserve pre-step land values instead of zeroing them.
            # Zeroing T, S on land each step and then averaging those
            # zeros into coastal cells via the Neumann fill produced a
            # cold/fresh front that propagated into the interior
            # one-cell-per-step. See issue #164. Matches the lat-lon
            # pattern in ocean_model_latlon_cgrid.py:493.
            # On partial-cell coordinates ``active_3d`` is per-level
            # and additionally preserves pre-step values in below-
            # seafloor cells (audit 2026-05-04).
            tr_new = jnp.where(active_3d > 0.5, tr_new, tr)

            if tr_name == 'T':
                T_corrected = tr_new
            else:
                S_corrected = tr_new

        if (freshwater is not None and F_slow_eta is not None
                and config.freshwater_closure == "real_freshwater"):
            # Surface dilution of the volume closure (2026-09-05): the eta
            # channel above spread the surface water uniformly over the
            # column; add the downward transport of the resident water so the
            # top cell dilutes by -S_1 F/(rho h_1) and the layers below keep
            # S and T (NEMO vvl).  Same normalised rate as eta; ice SALT flux
            # stays on its own channel.  Shared helper with the lat-lon core.
            from legoesm.ocean.freshwater import (
                real_freshwater_dilution_tendencies,
                real_freshwater_entry,
                resolve_runoff_spread_arg,
            )
            _F_entry, _entry_heat = real_freshwater_entry(
                freshwater, F_slow_eta, h_k_new, mask, config.rho_0,
                T_corrected, runoff_spread_m=resolve_runoff_spread_arg(config))
            _dS_dil, _dT_dil = real_freshwater_dilution_tendencies(
                F_slow_eta, S_corrected, T_corrected, h_k_new, mask,
                F_entry=_F_entry, entry_heat=_entry_heat)
            S_corrected = S_corrected + (dt * _dS_dil).astype(S_corrected.dtype)
            T_corrected = T_corrected + (dt * _dT_dil).astype(T_corrected.dtype)

        # Final state construction with explicit land masking
        T_final = T_corrected
        # Clamp salinity >= 0.  The virtual_salt_flux closure uses a
        # constant S_ref (not local S) so it can overshoot to negative
        # values in shallow cells with large freshwater input (e.g.
        # Hudson Bay).  The real_freshwater closure would avoid this
        # but virtual_salt_flux is the standard Boussinesq approach.
        S_final = jnp.maximum(S_corrected, 0.0)

        # Padding edges (dvEdge = 0) see wind stress and vertical mixing but
        # nothing that balances them; zero them so they cannot grow without
        # bound over a run (real edges: multiply by exactly 1).
        _real_edge = (self.mesh.dvEdge > 0).astype(u_3d_new.dtype)[:, jnp.newaxis]
        state_new = MPASOceanState(
            u=state.u.replace(data=u_3d_new * _real_edge),
            T=state.T.replace(data=T_final),
            S=state.S.replace(data=S_final),
            eta=state.eta.replace(data=eta_new * mask),
            w=state.w.replace(data=w),
            H_bathy=state.H_bathy,
            land_mask=state.land_mask,
            rho_ref_z=state.rho_ref_z,
            # Carry the prognostic TKE through unchanged (codex MED: omitting
            # it here silently defaulted a seeded Field back to None inside
            # jit — a pytree-structure change). The Mode-A store below
            # overwrites the DATA; this preserves the STRUCTURE on every
            # intermediate state (conservation fixer, freeze floor).
            tke=state.tke,
        )

        # 10. Conservation fixers (#166: pass expected forcing so fixer
        # only removes numerical drift, not the forcing itself)
        if config.use_conservation_fixer:
            _f64 = jnp.float64
            # Owned-cell mask: under MPI partitioning, halo cells are
            # also stored on neighbouring ranks; including them in the
            # local sums double-counts after the global allreduce.
            # ``self._owned_mask`` is set by the MPI-aware constructor
            # (``MPASPrimitiveEquationModel`` / ``MPASOceanModel``)
            # when running distributed; defaults to ``None`` (single-
            # rank, all cells owned) otherwise.
            owned_mask = getattr(self, "_owned_mask", None)
            _hr_owned = getattr(halo_refresh, "owned_mask_cells", None)
            if owned_mask is None and _hr_owned is not None:
                # Refresh-carried context (MPI layout or SPMD ppermute lane).
                owned_mask = _hr_owned
            if owned_mask is None:
                # Distributed Voronoi runs arm a partition layout
                # instead of setting ``self._owned_mask`` — pull the
                # owned-cell mask from it (size-checked: the model must
                # actually be built on that layout's local mesh).  With
                # ``is_multi_process()`` now layout-aware, an unmasked
                # local sum would double-count halo cells in the
                # allreduce.
                from legoesm.parallel.voronoi_mpi import (
                    get_matching_voronoi_layout,
                )
                _vl = get_matching_voronoi_layout(mesh)
                if _vl is not None:
                    owned_mask = _vl.owned_mask_cells
            if owned_mask is None:
                eff_mask = mask
            else:
                eff_mask = mask * owned_mask.astype(mask.dtype)
            wa = eff_mask.astype(_f64)[:, jnp.newaxis] * mesh.areaCell.astype(_f64)[:, jnp.newaxis]
            expected_dHeat_local = jnp.sum(
                tend.dT_dt.data.astype(_f64) * h_k_old.astype(_f64) * wa
            ) * dt
            expected_dSalt_local = jnp.sum(
                tend.dS_dt.data.astype(_f64) * h_k_old.astype(_f64) * wa
            ) * dt
            # Globally sum the locally-masked expected forcing so the
            # comparison against the globally-summed ``heat_old`` /
            # ``salt_old`` inside the fixer is consistent.
            # RESOLVED(voronoi-mpi-conservation, 2026-06-11):
            # ``is_multi_process()`` is now layout-aware (it returns
            # True when a Voronoi partition layout is armed — the
            # distributed-MPAS-PCG triangulation caught the allreduce
            # silently skipping), and the local sums above are
            # owned-masked from the same layout, so this dispatch is
            # globally correct on the partition path.
            from legoesm.parallel.reductions import is_multi_process
            _gsum = getattr(halo_refresh, "global_sum", None)
            if _gsum is not None:
                expected_dHeat, expected_dSalt = _gsum(
                    [expected_dHeat_local, expected_dSalt_local])
            elif is_multi_process():
                from legoesm.parallel.reductions import global_sum_mpi
                expected_dHeat = global_sum_mpi(expected_dHeat_local)
                expected_dSalt = global_sum_mpi(expected_dSalt_local)
            else:
                expected_dHeat = expected_dHeat_local
                expected_dSalt = expected_dSalt_local
            state_new = mpas_ocean_conservation_fixer(
                state_new, state, mesh, z_coord, config,
                expected_dHeat=expected_dHeat,
                expected_dSalt=expected_dSalt,
                owned_mask=owned_mask,
                reduce_fn=_gsum,
            )

        if config.freeze_floor:
            # Sea-ice thermodynamic surrogate (config.freeze_floor): an exposed
            # surface ocean cell cannot super-cool below the seawater freezing
            # point — the excess heat loss physically goes into ice latent heat,
            # which holds SST at freezing.  legoESM carries no prognostic ice, so
            # without this the Arctic surface over-cools ~4 C below NEMO (whose
            # LIM ice caps SST).  SURFACE-ONLY (k=0); an intentional bounded
            # non-conservative heat source applied AFTER the conservation fixer
            # (matches LatLonCGridOceanModel._apply_freeze_floor).
            T = state_new.T.data
            # Freeze-point floor [degC].  Default ("constant") keeps the scalar
            # freeze_floor_temp_c byte-identical; a liquidus scheme
            # (config.freezing.scheme) floors each surface cell at its own
            # freezing point from the local surface salinity.  freezing_point
            # returns KELVIN; state T is degC, so subtract constants.T_freeze.
            # scheme is static => feature-gating branch (matches the latlon
            # LatLonCGridOceanModel._apply_freeze_floor).  MED-1.
            if config.freezing.scheme == "constant":
                floor_c = config.freeze_floor_temp_c
            else:
                from legoesm import constants as _consts
                from legoesm.ocean.eos import freezing_point
                S_sfc = state_new.S.data[..., 0]
                floor_c = (
                    freezing_point(S_sfc, 0.0, scheme=config.freezing.scheme)
                    - _consts.T_freeze
                )
            T_floored = T.at[..., 0].set(jnp.maximum(T[..., 0], floor_c))
            state_new = state_new._replace(T=state_new.T.replace(data=T_floored))

        if self._tke_profiles_fn is not None and self._tke_prognostic:
            # Store the prognostic TKE carry (Mode A): profiles ran on the
            # OLD state's tke; the update advances one backward-Euler en step
            # (mirrors the lat-lon model step's tke_new store).  Reuse the
            # incoming Field wrapper so pytree structure is unchanged, and
            # PIN the carry dtype to the PRE-compute-cast dtype captured at
            # entry (codex r3: state.tke here is already compute dtype, so
            # pinning to it froze the carry at f64 under an f32-storage/
            # f64-compute policy — the scan-carry leaf must return in the
            # caller's dtype).
            state_new = state_new._replace(
                tke=state.tke.replace(
                    data=_tke_new.astype(_tke_in_dtype)))

        return cast_pytree(state_new, None, "storage")

    def seed_tke(self, state: MPASOceanState) -> MPASOceanState:
        """Seed the prognostic TKE carry (``MPASOceanState.tke``) if needed.

        Call ONCE on the initial state before stepping when the prognostic
        TKE closure is active — the scan/step carry must be pytree-stable, so
        the None->Field promotion cannot happen inside the step (mirrors the
        lat-lon model's pre-scan seeding). No-op when the carry is already
        seeded (e.g. loaded from a restart) or the closure is not prognostic
        TKE. Seeds ``tke_background`` on wet columns, 0 on land, at the
        interior interfaces (nCells, nlev-1).
        """
        if not (self._tke_profiles_fn is not None and self._tke_prognostic):
            return state
        if state.tke is not None:
            return state
        from legoesm.core.field import Field
        tke_cfg = self.config.physics.vertical_mixing.tke
        lm = state.land_mask.data
        nlev = state.T.data.shape[-1]
        dtype = state.T.data.dtype
        tke0 = jnp.where(
            lm[:, jnp.newaxis] > 0.5, tke_cfg.tke_background, 0.0,
        ).astype(dtype) * jnp.ones((1, nlev - 1), dtype=dtype)
        return state._replace(
            tke=Field(data=tke0, name="tke", dims=("nCells", "level"),
                      units="m^2/s^2"))

    def step(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater: FreshwaterForcing | None = None,
        surface_forcing=None,
        sponge=None,
        halo_refresh=None,
    ) -> MPASOceanState:
        """Host-side entry: the vorticity-filter stability gate runs once
        (the OMIP runner calls step(), not step_checked(); codex 2026-09-06),
        then the JIT-compiled :meth:`_step_jit`.  ``dt`` must be a Python
        float on the first call for the gate; later calls pass through.
        """
        if not self._vf_checked and not isinstance(dt, jax.core.Tracer):
            # (a traced dt -- step() called under an outer jit/grad -- cannot
            # be gated here; the checked entry and the runner pass floats)
            self.check_vorticity_filter_stability(float(dt))
            self._vf_checked = True
        return self._step_jit(state, dt, freshwater=freshwater,
                              surface_forcing=surface_forcing, sponge=sponge,
                              halo_refresh=halo_refresh)

    @partial(jax.jit, static_argnums=(0,), static_argnames=("halo_refresh",))
    def _step_jit(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater: FreshwaterForcing | None = None,
        surface_forcing=None,
        sponge=None,
        halo_refresh=None,
    ) -> MPASOceanState:
        """JIT-compiled wrapper around :meth:`_step_impl`.

        For use inside an outer JIT context (e.g. ``lax.scan``), call
        ``_step_impl`` directly to avoid nested JIT boundaries.

        ``halo_refresh`` (the distributed in-step stage-correctness
        refresh — see :meth:`_step_impl`) is a STATIC argument: build it
        ONCE per layout (``make_mpas_ocean_halo_refresh``) and pass the
        SAME object every call, or the jit cache re-traces.
        """
        return self._step_impl(
            state, dt, freshwater=freshwater,
            surface_forcing=surface_forcing, sponge=sponge,
            halo_refresh=halo_refresh,
        )

    def step_checked(
        self,
        state: MPASOceanState,
        dt: float,
        freshwater=None,
        surface_forcing=None,
        sponge=None,
        halo_refresh=None,
    ) -> MPASOceanState:
        """Advance one timestep with host-side runtime validation.

        Unlike the previous implementation which silently clipped tracers,
        this raises on out-of-bounds values so the caller sees the failure.
        """
        if not self._cfl_checked:
            self.check_barotropic_cfl(dt)
            self._cfl_checked = True
        state_new = self.step(state, dt, freshwater=freshwater,
                              surface_forcing=surface_forcing,
                              sponge=sponge, halo_refresh=halo_refresh)
        if self.config.enable_runtime_checks:
            self._assert_runtime_invariants(state_new)
        return state_new

    def _assert_runtime_invariants(self, state: MPASOceanState) -> None:
        """Host-side runtime checks (matching cubed-sphere ocean model).

        Fuses the finite-check, T-min/max, and S-min/max reductions
        into a single ``jnp.stack`` + ``np.asarray`` host transfer so
        the runtime checks cost one GPU stall per step instead of
        five.  Uses ``jnp.where``-masked ``nanmin``/``nanmax`` to
        avoid the boolean indexing path (``state.T.data[wet]`` allocates
        a dynamically-shaped array that cannot be JIT'd; the masked
        reductions are equivalent and stay on device).
        """
        mask = state.land_mask.data
        wet = mask > 0.5
        wet3 = wet[..., jnp.newaxis] if state.T.data.ndim > 1 else wet
        T_wet = jnp.where(wet3, state.T.data, jnp.nan)
        S_wet = jnp.where(wet3, state.S.data, jnp.nan)
        any_wet = jnp.any(wet)

        finite_ok = (
            jnp.all(jnp.isfinite(state.u.data))
            & jnp.all(jnp.isfinite(state.T.data))
            & jnp.all(jnp.isfinite(state.S.data))
            & jnp.all(jnp.isfinite(state.eta.data))
        )

        _stats = jnp.stack([
            finite_ok.astype(state.eta.data.dtype),
            any_wet.astype(state.eta.data.dtype),
            jnp.nanmin(T_wet).astype(state.eta.data.dtype),
            jnp.nanmax(T_wet).astype(state.eta.data.dtype),
            jnp.nanmin(S_wet).astype(state.eta.data.dtype),
            jnp.nanmax(S_wet).astype(state.eta.data.dtype),
        ])
        host = np.asarray(_stats)
        finite_ok_h = bool(host[0] > 0.5)
        any_wet_h = bool(host[1] > 0.5)

        if not finite_ok_h:
            raise FloatingPointError(
                "MPAS ocean runtime check failed: non-finite state detected"
            )

        config = self.config
        if any_wet_h:
            T_min_val = float(host[2])
            T_max_val = float(host[3])
            S_min_val = float(host[4])
            S_max_val = float(host[5])
            if T_min_val < config.temperature_min_c or T_max_val > config.temperature_max_c:
                raise ValueError(
                    f"MPAS ocean runtime check failed: T out of bounds "
                    f"[{T_min_val:.2f}, {T_max_val:.2f}] vs "
                    f"[{config.temperature_min_c}, {config.temperature_max_c}]"
                )
            if S_min_val < config.salinity_min_psu or S_max_val > config.salinity_max_psu:
                raise ValueError(
                    f"MPAS ocean runtime check failed: S out of bounds "
                    f"[{S_min_val:.2f}, {S_max_val:.2f}] vs "
                    f"[{config.salinity_min_psu}, {config.salinity_max_psu}]"
                )

    def integrate(
        self,
        state: MPASOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ):
        """Forward integration.

        Parameters
        ----------
        state : MPASOceanState
        duration : float
            Total integration time [s].
        dt : float
            Timestep [s].
        save_every : int
            Save trajectory every N steps.

        Returns
        -------
        (final_state, trajectory)
        """
        n_steps = int(duration / dt)
        trajectory = []

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: MPASOceanState,
        n_steps: int,
        dt: float,
    ):
        """Differentiable integration via jax.lax.scan.

        Parameters
        ----------
        state : MPASOceanState
        n_steps : int
        dt : float

        Returns
        -------
        (final_state, trajectory)
        """
        def scan_fn(carry, _):
            s = self.step(carry, dt)
            return s, s

        final, trajectory = jax.lax.scan(scan_fn, state, None, length=n_steps)
        return final, trajectory
