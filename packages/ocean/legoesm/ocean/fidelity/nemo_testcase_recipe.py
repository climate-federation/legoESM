"""Pure-config legoESM cards for the certified NEMO 5.0.2 test cases.

The domain formulas are transcribed from NEMO's ``tests/LOCK_EXCHANGE`` and
``tests/OVERFLOW`` user definitions.  Dynamics are selected exclusively from
the shared lat-lon C-grid model; this module contains no solver or oracle I/O.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.barotropic_common import nemo_auto_substeps
from legoesm.ocean.fidelity.nemo_recipe import NEMORecipe
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_full_step_coordinate,
    create_partial_cell_coordinate,
    create_z_star_from_thicknesses,
)


class NEMOTestcaseCard(NamedTuple):
    """A runnable recipe plus source-pinned test-case metadata."""

    case: str
    recipe: NEMORecipe
    dt_s: float
    n_steps: int
    dummy_bottom_records: int
    bbl_adv_option: int
    bbl_diffusive_option: int
    bbl_aht_m2_s: float
    bbl_gamma_s: float


class C1DSlabOceanCard(NamedTuple):
    """Production card for the one-column ocean half of rung 3.6.

    ``meridionally_periodic`` records the existing halo convention the runner
    must activate; the card contains no alternate solver.
    """

    recipe: NEMORecipe
    dt_s: float
    n_steps: int
    ice_cadence: int
    meridionally_periodic: bool
    precision_policy: PrecisionPolicy


def _model_config(
    *, barotropic_time_filter: str, n_barotropic_substeps: int,
    bbl_adv_option: int, bbl_gamma_s: float,
) -> LatLonCGridOceanConfig:
    """The selectors shared by both certified ``key_qco + key_RK3`` runs."""

    return LatLonCGridOceanConfig.from_flat(
        constants=NEMO_CONSTANTS_CONFIG,
        # Resolved ln_TEOS10=.true.; NEMO 5.0.2 eosbn2.F90:1920-2108
        # selects the Roquet TEOS-10 coefficient table and :260-288 evaluates it.
        eos="nemo_teos10",
        # eosbn2.F90:253-258 passes the live geometric gdept to the polynomial;
        # do not recover depth from legoESM's iterated in-situ pressure.
        eos_depth="geometric",
        tracer_advection="fct2",
        # NEMO key_RK3: stprk3_stg.F90:112-249,519-559 restarts tracer
        # stages from Kbb with dt/3, dt/2, and dt.
        tracer_time_integrator="rk3_ws",
        momentum_advection="flux_form",
        momentum_flux_scheme="upwind3",
        momentum_time_integrator="rk3_ws",
        # key_RK3 is a single scheme identity: Kmm transports + two-step FCT
        # + per-stage external-mode correction + distinct un_adv/hu transport.
        # NEMO exposes no switches for these internals (stprk3_stg.F90:
        # 257-274,433-446), so legoESM exposes none either.
        # ln_dynadv_up3 dispatches to dynadv_up3 (dynadv.F90:87-89); its
        # vertical UP3 flux is dynadv_up3.F90:239-365. dynzad is dead here.
        vertical_momentum_scheme="off",
        # NEMO 5.0.2 dynhpg.F90:117-123 dispatches ln_hpg_sco (the resolved
        # value on both cards) to hpg_sco, not hpg_djc.  The canonical
        # nemo_sco option transcribes its recurrence at :340-390.
        pgf_scheme="nemo_sco",
        pgf_quadrature="nemo_trapezoid",
        barotropic_solver="explicit_substep",
        barotropic_time_filter=barotropic_time_filter,
        n_barotropic_substeps=n_barotropic_substeps,
        barotropic_face_depth="nemo_ssh_avg",
        barotropic_continuity_evaluation="nemo_literal",
        barotropic_transport_accumulation_evaluation="nemo_literal",
        barotropic_seed_face_depth="nemo_ssh_avg",
        barotropic_seed_evaluation="nemo_literal",
        barotropic_pgf_evaluation="nemo_literal",
        # NEMO commits the primary velocity-weighted uu_b(Kaa) into the RK3
        # prognostic velocity (dynspg_ts.F90:845-847;
        # stprk3_stg.F90:433-446).  The distinct un_adv transport time mean
        # remains the tracer-flux operand at stprk3_stg.F90:257-274; it is not
        # a prognostic-velocity frame.
        barotropic_reconcile_target="velocity_avg",
        # This is numerical diffusion in the free-surface equation, not
        # NEMO's similarly named time-filter alpha.  The oracle has no such
        # stabilizer, so Rule 9 requires an exact zero on certified cards.
        barotropic_diffusion_alpha=0.0,
        bbl_adv_option=bbl_adv_option,
        bbl_gamma_s=bbl_gamma_s,
        adaptive_implicit_vertadv=True,
        implicit_vertical_mixing=True,
        zdf_implicit_solver_evaluation="nemo_literal",
        implicit_vmix_e3t_now_divisor=True,
        A_h=0.0,
        B_h=0.0,
        C_smag=0.0,
        C_smag_lap=0.0,
        C_leith=0.0,
        K_h=0.0,
        K_bih=0.0,
        A_v=1.0e-4,
        K_v=0.0,
        bottom_drag_r=0.0,
        gm_redi=None,
        physics=None,
        normalize_freshwater=False,
        use_conservation_fixer=False,
        fix_eta_drift=False,
        freeze_floor=False,
    )


def _resolved_auto_substeps(grid, bathymetry, dt_s: float) -> int:
    """Resolve NEMO ``ln_bt_auto`` on the actual T-cell depth and metric.

    NEMO computes ``zcu`` pointwise and takes its maximum before the ceiling
    (``dynspg_ts.F90:1223-1240``).  Keeping depth and metric paired avoids the
    global-max upper bound used by callers that lack collocated arrays.
    """

    wet = np.asarray(bathymetry) > 0.0
    inverse_metric = (
        np.asarray(grid.dx_T, dtype=np.float64) ** -2
        + np.asarray(grid.dy_T, dtype=np.float64) ** -2
    )
    if inverse_metric.shape != wet.shape:
        inverse_metric = np.broadcast_to(inverse_metric, wet.shape)
    courant_factor = np.asarray(bathymetry, dtype=np.float64) * inverse_metric
    return nemo_auto_substeps(
        dt_s,
        1.0,
        float(np.max(courant_factor[wet], initial=0.0)),
        float(NEMO_CONSTANTS_CONFIG.g),
        cmax=0.8,
    )


def _closed_box_mask(n_lat: int, n_lon: int) -> jnp.ndarray:
    mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
    mask = mask.at[(0, -1), :].set(0.0)
    return mask.at[:, (0, -1)].set(0.0)


def _cartesian_grid(n_lon: int, spacing_m: float):
    # NEMO usrdef_hgr: T positions are dx*(-0.5 + global_index-1), in km.
    return create_beta_plane_cgrid_geometry(
        3,
        n_lon,
        dx_m=spacing_m,
        dy_m=spacing_m,
        f0=0.0,
        beta=0.0,
        x_origin_m=-spacing_m,
        y_origin_m=-spacing_m,
        cartesian_pseudo_lat=False,
        dtype=jnp.float64,
    )


def _native_initial_state(
    grid, z_coord, bathymetry, x_km, cold, warm, front_km
):
    surface_wet = bathymetry > 0.0
    state = rest_state_latlon_cgrid_ocean(
        grid,
        z_coord,
        T_water_init_C=warm,
        T_deep=warm,
        S_uniform=35.0,
        H_max=float(z_coord.H_max),
        land_mask_override=surface_wet,
        H_bathy_override=bathymetry,
    )
    active = jnp.asarray(z_coord.is_active) & surface_wet[..., None]
    temperature = jnp.where(x_km[..., None] <= front_km, cold, warm)
    temperature = jnp.where(active, temperature, 0.0).astype(jnp.float64)
    salinity = jnp.where(active, 35.0, 0.0).astype(jnp.float64)
    return state._replace(
        T=state.T.replace(data=temperature),
        S=state.S.replace(data=salinity),
    )


def build_lock_exchange_zco_card() -> NEMOTestcaseCard:
    """Certified LOCK_EXCHANGE: 64 km x 20 m, flat full-step z coordinate."""

    grid = _cartesian_grid(130, 500.0)
    x_km = jnp.broadcast_to(
        jnp.asarray((np.arange(130, dtype=np.float64) - 0.5) * 0.5)[None, :],
        (3, 130),
    )
    wet = _closed_box_mask(3, 130)
    bathymetry = wet * 20.0
    z_ref = create_z_star_from_thicknesses(
        jnp.full((20,), 1.0),
        t_depth_ref_m=np.arange(20, dtype=np.float64) + 0.5,
    )
    bottom = jnp.where(wet > 0.0, 19, -1)
    z_coord = create_full_step_coordinate(z_ref, bottom)
    state = _native_initial_state(
        grid, z_coord, bathymetry, x_km, 5.0, 30.0, 32.0
    )
    # ln_bt_fw=T, nn_bt_flt=3, rn_bt_alpha=.07: dynspg_ts.F90:199-226,
    # 536-553,1676-1711. The canonical option carries substep history.
    model_config = _model_config(
        barotropic_time_filter="nemo_ab3am4",
        n_barotropic_substeps=_resolved_auto_substeps(grid, bathymetry, 1.0),
        bbl_adv_option=0,
        bbl_gamma_s=0.0,
    )
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=wet,
        initial_state=state,
    )
    card = NEMOTestcaseCard(
        "LOCK_EXCHANGE-zco", recipe, 1.0, 61200, 1, 0, 0, 0.0, 0.0
    )
    validate_nemo_testcase_card(card)
    return card


def build_overflow_zps_card() -> NEMOTestcaseCard:
    """Certified OVERFLOW: tanh bathymetry with z partial bottom cells."""

    grid = _cartesian_grid(202, 1000.0)
    wet = _closed_box_mask(3, 202)
    # Static source geometry: evaluate with host fp64 libm, matching the NEMO
    # Fortran initialization rather than XLA's target-specific tanh lowering.
    x_1d_km = np.arange(202, dtype=np.float64) - 0.5
    x_km = jnp.broadcast_to(jnp.asarray(x_1d_km)[None, :], (3, 202))
    # Scalar libm ``tanh`` matches the gfortran oracle's call bit-for-bit;
    # NumPy's vector tanh differs by one bathymetry ULP in one column.
    depth_1d = np.asarray(
        [500.0 + 0.5 * 1500.0 * (1.0 + math.tanh((x - 40.0) / 7.0))
         for x in x_1d_km],
        dtype=np.float64,
    )
    depth = jnp.broadcast_to(jnp.asarray(depth_1d)[None, :], (3, 202))
    bathymetry = jnp.where(wet > 0.0, depth, 0.0)
    z_ref = create_z_star_from_thicknesses(
        jnp.full((100,), 20.0),
        t_depth_ref_m=10.0 + 20.0 * np.arange(100, dtype=np.float64),
    )
    z_coord = create_partial_cell_coordinate(
        z_ref, bathymetry, bottom_index_rule="nemo_tpoint"
    )
    # NEMO clips the analytic input depth onto its zps bottom-cell geometry;
    # the model water-column depth is therefore the sum of the executed cells.
    effective_bathymetry = jnp.sum(z_coord.h_partial, axis=-1)
    state = _native_initial_state(
        grid, z_coord, effective_bathymetry, x_km, 10.0, 20.0, 20.0
    )
    # ln_bt_fw=T, nn_bt_flt=1, rn_bt_alpha=0: dynspg_ts.F90:1058-1080,
    # 1676-1711. The canonical option re-runs the cold-start ramp each step.
    model_config = _model_config(
        barotropic_time_filter="nemo_boxcar1_ab3",
        n_barotropic_substeps=_resolved_auto_substeps(
            grid, effective_bathymetry, 10.0
        ),
        bbl_adv_option=2,
        bbl_gamma_s=20.0,
    )
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=wet,
        initial_state=state,
    )
    # The shared canonical BBL implementation lives in ocean.physics.bbl_adv;
    # these pins let the phase-2 fidelity harness select it without solver glue.
    card = NEMOTestcaseCard(
        "OVERFLOW-zps", recipe, 10.0, 6120, 1, 2, 0, 1000.0, 20.0
    )
    validate_nemo_testcase_card(card)
    return card


def build_c1d_omip_l3_slab_ocean_card() -> C1DSlabOceanCard:
    """Build the reviewed 10 m, one-wet-layer C1D OCE+ICE ocean card.

    This composes the same shared ``key_qco + key_RK3`` dynamics used by the
    test-case recipes.  C1D resolves ``ln_dynadv_OFF``; horizontal advection is
    therefore structurally zero on its single doubly-periodic cell, while the
    shared flux-form update remains the active ``dynspg_ts`` statement.
    """

    f_84n = (
        2.0 * float(NEMO_CONSTANTS_CONFIG.Omega)
        * math.sin(math.radians(84.0))  # const-ok: C1D rn_lat1d case geometry
    )
    grid = create_beta_plane_cgrid_geometry(
        1, 1,
        dx_m=100.0,  # const-ok: C1D usrdef_hgr.F90:88-91 case metric
        dy_m=100.0,  # const-ok: C1D usrdef_hgr.F90:88-91 case metric
        f0=f_84n,
        beta=0.0,
        cartesian_pseudo_lat=False,
        dtype=jnp.float64,
    )
    wet = jnp.ones((1, 1), dtype=jnp.float64)
    # Two allocated levels reproduce jpk=2; bottom index zero leaves exactly
    # one wet layer.  The second thickness is never executed.
    z_ref = create_z_star_from_thicknesses(
        jnp.asarray((10.0, 10.0), dtype=jnp.float64),  # const-ok: Decision 6
        t_depth_ref_m=np.asarray((5.0, 15.0), dtype=np.float64),
    )
    z_coord = create_full_step_coordinate(
        z_ref, jnp.zeros((1, 1), dtype=np.int32))
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=-1.690032958984375,  # C1D init_oce_T.nc, exact fp64
        T_deep=-1.690032958984375,
        S_uniform=34.0,  # const-ok: C1D init_oce_S.nc case value
        H_max=10.0,  # const-ok: user Decision 6 free construction parameter
        land_mask_override=wet,
        H_bathy_override=wet * 10.0,
    )
    eta0 = jnp.full(
        (1, 1), -1.6666666666666667, dtype=jnp.float64)  # ice-load SSH
    # ``rest_state`` constructs closed-wall v masks before a runner activates
    # the existing periodic-halo context.  C1D is periodic in both horizontal
    # directions, so its two representations of the sole meridional face are
    # wet (NEMO usrdef_dom/cyclone domain masks); bind that case geometry in
    # the card rather than introducing a one-column operator.
    state = state._replace(
        eta=state.eta.replace(data=eta0),
        v_mask=state.v_mask.replace(data=jnp.ones_like(state.v_mask.data)),
    )
    cfg = _model_config(
        barotropic_time_filter="nemo_boxcar1_ab3",
        n_barotropic_substeps=631,
        bbl_adv_option=0,
        bbl_gamma_s=0.0,
    )
    cfg = cfg._replace(
        # C1D's resolved namelist prints ln_traadv_OFF=T; tra_adv therefore
        # leaves Krhs unchanged at stprk3_stg.F90:540.  Inheriting the GYRE
        # FCT selector fed the QCO vertical transport to a branch NEMO did not
        # execute, despite the one-column horizontal geometry.
        tracer_advection="off",
        momentum_advection="off",
        # C1D resolves NEMO's ENS Coriolis inside each WS-RK3 3-D momentum
        # tendency.  The shared ``explicit_ab2`` spelling selects that
        # tendency placement (the name predates the RK3 caller); the
        # ``matsuno_split`` alternative would move rotation after the stages
        # and makes their Krhs identically zero in this one-layer column.
        coriolis_scheme="explicit_ab2",
        # With the resolved linear-dynamics switch NEMO's ENE relative term
        # is structurally zero and only its planetary part remains.  legoESM's
        # shared ``ene`` + explicit face-Coriolis split is the same one-column
        # operator; ``ene_total`` would require the shared EEN live barotropic
        # stencil, a different NEMO branch.
        vorticity_scheme="ene",
        barotropic_coriolis_split="live",
        # C1D_OMIP_L3_COUPLED10M executes NEMO's implicit wind-stress
        # deposition in dynzdf.F90:328-330 and the mandatory post-solve
        # barotropic correction in stprk3_stg.F90:437-445.  These are the
        # shared NEMO selectors, not slab-only numerics.
        surface_stress_implicit=True,
        # sbcfwb.F90:292-295 changes e3t/SSH under nn_fwb_voltype=1; salt is
        # carried only by SI3's real ``sfx`` source in trasbc.F90:310-313.
        # The generic virtual-salt closure would apply emp a second time after
        # the certified RK3 source program.
        freshwater_closure="real_freshwater",
        bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_r=5.0e-5,  # namdrg_bot resolved linear coefficient
            bottom_drag_bbl_thickness=0.0,
            bottom_drag_scheme="nemo_linear",
        ),
        zdf_drag_in_matrix=True,
        zdf_baroclinic_only=True,
        barotropic_drag_substep=True,
        barotropic=cfg.barotropic._replace(
            nemo_stage_mean_imposition=True),
    )
    recipe = NEMORecipe(cfg, cfg.physics, grid, z_coord, wet, state)
    return C1DSlabOceanCard(
        recipe=recipe,
        dt_s=3600.0,  # C1D rn_Dt
        n_steps=8760,
        ice_cadence=4,
        meridionally_periodic=True,
        precision_policy=PrecisionPolicy.fp64(transcendentals="libm"),
    )


def validate_nemo_testcase_card(card: NEMOTestcaseCard) -> None:
    """Reject any card composition not exercised by its named oracle run."""
    expected = {
        "LOCK_EXCHANGE-zco": ("nemo_ab3am4", 1, 0, 0.0),
        "OVERFLOW-zps": ("nemo_boxcar1_ab3", 3, 2, 20.0),
    }
    if card.case not in expected:
        raise ValueError(f"unknown NEMO testcase card {card.case!r}")
    cfg = card.recipe.model_config
    filt, count, bbl_option, gamma = expected[card.case]
    actual = (
        cfg.barotropic.barotropic_time_filter,
        cfg.barotropic.n_barotropic_substeps,
        cfg.bbl_adv_option,
        cfg.bbl_gamma_s,
    )
    if actual != (filt, count, bbl_option, gamma):
        raise ValueError(
            f"{card.case} filter/substep/BBL composition {actual!r} does not "
            f"match the executed oracle {(filt, count, bbl_option, gamma)!r}")
    required = {
        "barotropic_face_depth": "nemo_ssh_avg",
        "barotropic_continuity_evaluation": "nemo_literal",
        "barotropic_transport_accumulation_evaluation": "nemo_literal",
        "barotropic_seed_face_depth": "nemo_ssh_avg",
        "barotropic_seed_evaluation": "nemo_literal",
        "barotropic_pgf_evaluation": "nemo_literal",
    }
    for field, value in required.items():
        got = getattr(cfg.barotropic, field)
        if got != value:
            raise ValueError(
                f"{card.case} requires {field}={value!r}, got {got!r}")
    if cfg.barotropic.barotropic_diffusion_alpha != 0.0:
        raise ValueError(
            f"{card.case} forbids unmatched live eta diffusion")
    if (cfg.zdf_implicit_solver_evaluation != "nemo_literal"
            or not cfg.implicit_vmix_e3t_now_divisor):
        raise ValueError(
            f"{card.case} requires the NEMO literal implicit-ZDF program")
    if not cfg.tracer_wall_neumann_fill:
        raise ValueError(
            f"{card.case} requires NEMO's closed-wall tracer halo fill")
    if cfg.barotropic.barotropic_reconcile_target != "velocity_avg":
        raise ValueError(
            f"{card.case} requires NEMO's prognostic uu_b(Kaa) velocity frame")
    # The namelists select ENS, while these Cartesian cases have f=0 and only
    # one wet y row.  Prove the inherited rotation operator is structurally
    # eliminated; otherwise reject rather than silently run an AL81/Matsuno
    # third model.
    f_t = np.asarray(card.recipe.grid.f_T)
    wet = np.asarray(card.recipe.land_mask) > 0.5
    if np.any(f_t != 0.0) or np.count_nonzero(np.any(wet, axis=1)) != 1:
        raise ValueError(
            f"{card.case} legacy rotation is permitted only when f=0 and "
            "the meridional operator is structurally absent")


def build_nemo_testcase_card(case: str) -> NEMOTestcaseCard:
    """Fail-closed dispatch for the two phase-2 cards."""

    builders = {
        "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card,
        "OVERFLOW-zps": build_overflow_zps_card,
    }
    if case not in builders:
        raise ValueError(
            f"unknown NEMO testcase {case!r}; expected one of {sorted(builders)}"
        )
    return builders[case]()


__all__ = (
    "C1DSlabOceanCard",
    "NEMOTestcaseCard",
    "build_c1d_omip_l3_slab_ocean_card",
    "build_lock_exchange_zco_card",
    "build_overflow_zps_card",
    "build_nemo_testcase_card",
    "validate_nemo_testcase_card",
)
