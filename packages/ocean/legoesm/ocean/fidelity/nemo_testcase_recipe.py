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
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.barotropic_common import nemo_auto_substeps
from legoesm.ocean.fidelity.nemo_recipe import (
    NEMOModelRecipeConfig,
    NEMORecipe,
    nemo_lat_lon_model_config,
    nemo_gyre_emp,
    nemo_gyre_qsr,
    nemo_gyre_t_star,
    nemo_gyre_wind,
)
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    NemoEENBarotropicOperands,
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
    surface_boundary_condition: str = "none"


class GYRESurfaceBoundaryCondition(NamedTuple):
    """Analytic fields emitted by GYRE ``usrdef_sbc`` on T points."""

    qsr_w_m2: jnp.ndarray
    t_star_c: jnp.ndarray
    emp_kg_m2_s: jnp.ndarray
    utau_pa: jnp.ndarray
    vtau_pa: jnp.ndarray
    taum_pa: jnp.ndarray
    wndm_m_s: jnp.ndarray


def _model_config(
    *, barotropic_time_filter: str, n_barotropic_substeps: int,
    bbl_adv_option: int, bbl_gamma_s: float, whole_step_identity: str,
) -> LatLonCGridOceanConfig:
    """The selectors shared by both certified ``key_qco + key_RK3`` runs."""

    if whole_step_identity not in {"lane1_flux_up3", "gyre_vector_ene_c2"}:
        raise ValueError(
            "unknown whole_step_identity; expected 'lane1_flux_up3' or "
            "'gyre_vector_ene_c2'"
        )

    if whole_step_identity == "gyre_vector_ene_c2":
        # One collapsed resolved-GYRE identity.  The numerical blocks come from
        # the canonical NEMO/DINO card; only this testcase's OMIP TEOS-10/SCO,
        # RK3, and resolved GYRE parameter selections are supplied here.
        config = nemo_lat_lon_model_config(
            NEMOModelRecipeConfig(
                momentum_core="vector_invariant_ene",
                eos="nemo_teos10",
                tracer_advection="fct2",
                pgf_scheme="nemo_sco",
                pgf_quadrature="nemo_trapezoid",
                barotropic_solver="explicit_substep",
                n_barotropic_substeps=n_barotropic_substeps,
                barotropic_time_filter=barotropic_time_filter,
                momentum_time_integrator="rk3_ws",
                adaptive_implicit_vertadv=False,
                implicit_vertical_mixing=True,
                A_h=1.0e5,
                A_h_lat_scaling=False,
                A_h_floor=0.0,
                C_smag_lap=0.0,
                B_h=0.0,
                K_h=0.0,
                K_bih=0.0,
                gm_redi=True,
                kappa_GM=0.0,
                kappa_Redi=1000.0,
                redi_S_max=0.01,
                lateral_operator="nemo_iso_lap",
                mle=False,
                rgb_shortwave=False,
                normalize_freshwater=False,
                freeze_floor=False,
                bottom_drag_scheme="nemo_quadratic",
            )
        )
        tke = config.physics.vertical_mixing.tke._replace(
            # OMIP-style TEOS-10 deviation: both zdftke and zdfevd consume
            # eosbn2, not the S-EOS identity used by shipped GYRE.
            n2_mode="nemo_bn2",
            n2_eos_form="teos10",
            buoyancy_timing="pre_mixing",
            shear_production="pre_solve",
            prandtl_mode="nemo_ri",
            positivity="floor",
            tke_surface_bc_level="nemo_z0",
            tke_buoyancy_sink="nemo_explicit",
            mxl_min=0.01,
            mxl0_min_m=0.04,
            tke_dry_wmask=True,
            kappaM_max=float("inf"),
            tke_preclosure_coeff_source="carried_previous_step",
            tke_matrix_evaluation="nemo_literal",
            tke_solver_evaluation="nemo_literal",
            tke_etau_exponential_evaluation="jax_expression",
            tke_htau_evaluation="jax_expression",
            tke_mxl_raw_evaluation="factored",
            tke_langmuir_evaluation="vectorized",
            tke_shear_evaluation_stage="step_entry",
            # RK3 has no leapfrog eta-before carrier; the QCO live metric is
            # reconstructed at the current stage from the same raw ladder.
            tke_shear_metric_source="tpoint_jacobian",
            tke_n2_evaluation_stage="step_entry",
        )
        physics = config.physics._replace(
            vertical_mixing=config.physics.vertical_mixing._replace(
                tke=tke, vmix_background_mode="nemo_max_floor"
            ),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=100.0,
                    nu_conv=100.0,
                    K_bg=0.0,
                    nu_bg=0.0,
                    smooth_transition=False,
                    n2_mode="nemo_bn2",
                    n2_eos_form="teos10",
                    n2_threshold=-1.0e-12,
                    two_level_trigger=True,
                ),
            ),
            shortwave_penetration=ShortwavePenetrationConfig(
                scheme="jerlov_2band", water_type="I"
            ),
            mle=None,
        )
        return config._replace(
            physics=physics,
            eos_depth="geometric",
            tracer_time_integrator="rk3_ws",
            vertical_momentum_scheme="nemo_advective",
            zad_bottom_face_mask="nemo_faithful",
            zad_qco_evaluation="nemo_literal",
            wzv_call2_evaluation="nemo_literal",
            vorticity_scheme="ene_total",
            coriolis_scheme="explicit_ab2",
            # GYRE's np_ENE dyn_cor_2D is subcycled live: subtract the Kmm
            # ENE Coriolis from zu_frc, then reapply it to every AB3 mid-step
            # velocity (dynspg_ts.F90:359,689; nn_bt_flt=3).
            barotropic_coriolis_split="live",
            lateral_viscosity_operator="nemo_div_curl",
            lateral_viscosity_e3_weighting="nemo_e3",
            surface_stress_implicit=True,
            bbl_adv_option=bbl_adv_option,
            bbl_gamma_s=bbl_gamma_s,
            zdf_implicit_solver_evaluation="nemo_literal",
            implicit_vmix_e3t_now_divisor=True,
            use_conservation_fixer=False,
            fix_eta_drift=False,
            barotropic=config.barotropic._replace(
                barotropic_diffusion_alpha=0.0,
                barotropic_face_depth="nemo_ssh_avg",
                barotropic_continuity_evaluation="nemo_literal",
                barotropic_transport_accumulation_evaluation="nemo_literal",
                barotropic_seed_face_depth="nemo_ssh_avg",
                barotropic_seed_evaluation="nemo_literal",
                barotropic_pgf_evaluation="nemo_literal",
                barotropic_coriolis="ene_metric",
                barotropic_een_seed="nemo_kmm",
                barotropic_een_coefficient_evaluation="nemo_literal",
                barotropic_reconcile_target="velocity_avg",
                nemo_stage_mean_imposition=True,
            ),
        )

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
        vertical_momentum_scheme="nemo_up3",
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


# The 31-value arrays are the fp64 records certified from the running
# ``usrdef_zgr.F90:93-175`` MI96 construction in lane 2 phase 1.  Re-evaluating
# LOG(COSH()) with NumPy libm misses the oracle's immutable 1e-15 pointwise bar;
# pinning the source-produced ladder preserves the NEMO arithmetic instead of
# weakening the bar.  The card executes the first 30 wet records; record 31 is
# NEMO's permanently dry dummy bottom level.
_GYRE_E3T_1D = np.asarray([
    10.003514801805068, 10.26472604769333, 10.653529090795246,
    11.231608712113939, 12.08969824829228, 13.360330096794087,
    15.23507363590079, 17.986499995084273, 21.993282036775668,
    27.762634353116596, 35.93635369855406, 47.254890225334634,
    62.44322808843981, 81.98917271094399, 105.83539858887082,
    133.10877001399524, 162.09543281938386, 190.599631250789,
    216.5654109595439, 238.6192469742948, 256.2608904106037,
    269.7067692950918, 279.5807802413228, 286.63508275979984,
    291.5762701420572, 294.9895332266615, 297.32472884315484,
    298.9118390394151, 299.98567369315606, 300.7100172156124,
    301.0986264392159,
], dtype=np.float64)
_GYRE_E3W_1D = np.asarray([
    9.950530581805197, 10.121161817424081, 10.43988191757876,
    10.914048646553056, 11.61852705940396, 12.66309229154649,
    14.207337494218677, 16.480319826389632, 19.804482396370304,
    24.620543509582376, 31.504122360713666, 41.15480650087369,
    54.32592726687034, 71.6584856658713, 93.40817408802695,
    119.13338905771104, 147.5206037556743, 176.54727935722144,
    204.00592227790025, 228.13252600238502, 247.99143632415962,
    263.47513523883026, 275.04366416278845, 283.4138796274283,
    289.33002250128766, 293.4427044280478, 296.26873304279707,
    298.19518745505684, 299.50127253556275, 300.38349000569724,
    300.97790853013066,
], dtype=np.float64)
_GYRE_GDEPT_1D = np.asarray([
    4.975265290902598, 15.09642710832668, 25.53630902590544,
    36.450357672458495, 48.068884731862454, 60.731977023408945,
    74.93931451762762, 91.41963434401725, 111.22411674038756,
    135.84466024996993, 167.3487826106836, 208.5035891115573,
    262.8295163784276, 334.4880020442989, 427.8961761323259,
    547.0295651900369, 694.5501689457112, 871.0974483029327,
    1075.103370580833, 1303.235896583218, 1551.2273329073776,
    1814.7024681462078, 2089.7461323089965, 2373.160011936425,
    2662.4900344377124, 2955.9327388657603, 3252.2014719085573,
    3550.396659363614, 3849.897931899177, 4150.281421904874,
    4451.259330435005,
], dtype=np.float64)
_GYRE_GDEPW_1D = np.asarray([
    0.0, 10.003514801805068, 20.268240849498397, 30.921769940293643,
    42.15337865240758, 54.24307690069986, 67.60340699749395,
    82.83848063339474, 100.82498062847901, 122.81826266525468,
    150.58089701837127, 186.51725071692533, 233.77214094225997,
    296.2153690306998, 378.20454174164377, 484.0399403305146,
    617.1487103445098, 779.2441431638937, 969.8437744146827,
    1186.4091853742266, 1425.0284323485214, 1681.289322759125,
    1950.996092054217, 2230.57687229554, 2517.21195505534,
    2808.788225197397, 3103.7777584240584, 3401.1024872672133,
    3700.0143263066284, 3999.9999999997844, 4300.710017215397,
], dtype=np.float64)


def gyre_vertical_ladder() -> dict[str, np.ndarray]:
    """Return copies of the source-produced 31-record GYRE MI96 ladder."""

    return {
        "e3t_1d": _GYRE_E3T_1D.copy(),
        "e3w_1d": _GYRE_E3W_1D.copy(),
        "gdept_1d": _GYRE_GDEPT_1D.copy(),
        "gdepw_1d": _GYRE_GDEPW_1D.copy(),
    }


def gyre_horizontal_coordinates() -> dict[str, np.ndarray]:
    """Transcribe the rotated GYRE T/U/V/F coordinates and beta-plane f."""

    nx, ny = 32, 22
    radius = float(NEMO_CONSTANTS_CONFIG.R_earth)
    omega = float(NEMO_CONSTANTS_CONFIG.Omega)
    rad = np.pi / 180.0
    spacing = 106000.0
    spacing_deg = spacing / (radius * rad)
    sin_alpha = -math.sqrt(2.0) * 0.5
    cos_alpha = math.sqrt(2.0) * 0.5
    lon0 = -85.0 + cos_alpha * spacing_deg * (ny - 2)
    lat0 = 29.0 + sin_alpha * spacing_deg * (ny - 2)
    i_m05 = np.arange(nx, dtype=np.float64) - 0.5
    i_m0 = np.arange(nx, dtype=np.float64)
    j_m05 = np.arange(ny, dtype=np.float64) - 0.5
    j_m0 = np.arange(ny, dtype=np.float64)

    def stagger(i, j):
        lon = lon0 + i[None, :] * spacing_deg * cos_alpha
        lon = lon + j[:, None] * spacing_deg * sin_alpha
        lat = lat0 - i[None, :] * spacing_deg * sin_alpha
        lat = lat + j[:, None] * spacing_deg * cos_alpha
        return lon, lat

    glamt, gphit = stagger(i_m05, j_m05)
    glamu, gphiu = stagger(i_m0, j_m05)
    glamv, gphiv = stagger(i_m05, j_m0)
    glamf, gphif = stagger(i_m0, j_m0)
    beta = 2.0 * omega * math.cos(rad * 29.0) / radius
    f0 = 2.0 * omega * math.sin(rad * 15.0)

    def coriolis(lat):
        return f0 + beta * np.abs(lat - 15.0) * rad * radius

    return {
        "glamt": glamt, "glamu": glamu, "glamv": glamv, "glamf": glamf,
        "gphit": gphit, "gphiu": gphiu, "gphiv": gphiv, "gphif": gphif,
        "ff_t": coriolis(gphit), "ff_f": coriolis(gphif),
    }


def _gyre_grid():
    """Build the source-rotated 106-km Cartesian C-grid."""

    source = gyre_horizontal_coordinates()
    grid = create_beta_plane_cgrid_geometry(
        22, 32, dx_m=106000.0, dy_m=106000.0, f0=0.0, beta=0.0,
        radius=float(NEMO_CONSTANTS_CONFIG.R_earth),
        cartesian_pseudo_lat=True, dtype=jnp.float64,
    )
    radians = np.pi / 180.0
    rotation = math.sqrt(2.0) * 0.5
    # The redundant west/south face layouts add one model face beyond the
    # native NEMO A2D arrays.  Their interior records are the exact source
    # staggerings; boundary extras are affine continuations and are walled.
    f_u_native = (
        2.0 * float(NEMO_CONSTANTS_CONFIG.Omega)
        * np.sin(15.0 * radians)
        + 2.0 * float(NEMO_CONSTANTS_CONFIG.Omega)
        * math.cos(29.0 * radians)
        / float(NEMO_CONSTANTS_CONFIG.R_earth)
        * np.abs(source["gphiu"] - 15.0) * radians
        * float(NEMO_CONSTANTS_CONFIG.R_earth)
    )
    f_u = np.concatenate([f_u_native, f_u_native[:, -1:] + (
        f_u_native[:, -1:] - f_u_native[:, -2:-1])], axis=1)
    f_v = np.concatenate([
        source["ff_f"][:1] - (source["ff_f"][1:2] - source["ff_f"][:1]),
        source["ff_f"],
    ], axis=0)
    return grid._replace(
        lat_T=jnp.asarray(source["gphit"] * radians, dtype=jnp.float64),
        lon_T=jnp.asarray(source["glamt"] * radians, dtype=jnp.float64),
        f_T=jnp.asarray(source["ff_t"], dtype=jnp.float64),
        f_u=jnp.asarray(f_u, dtype=jnp.float64),
        f_v=jnp.asarray(f_v, dtype=jnp.float64),
        cos_alpha_u=jnp.full((22, 33), rotation, dtype=jnp.float64),
        sin_alpha_u=jnp.full((22, 33), rotation, dtype=jnp.float64),
        cos_alpha_v=jnp.full((23, 32), rotation, dtype=jnp.float64),
        sin_alpha_v=jnp.full((23, 32), rotation, dtype=jnp.float64),
        native_lat_T_deg=jnp.asarray(source["gphit"], dtype=jnp.float64),
    )


def _gyre_initial_profiles(depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scalar-libm transcription of ``usrdef_istate.F90:62-76``."""

    def profile(d):
        w1 = (-math.tanh((500.0 - d) / 150.0) + 1.0) / 2.0
        w2 = (-math.tanh((d - 500.0) / 150.0) + 1.0) / 2.0
        t = ((16.0 - 12.0 * math.tanh((d - 400.0) / 700.0)) * w1
             + (15.0 * (1.0 - math.tanh((d - 50.0) / 1500.0))
                - 1.4 * math.tanh((d - 100.0) / 100.0)
                + 7.0 * (1500.0 - d) / 1500.0) * w2)
        s = ((36.25 - 1.13 * math.tanh((d - 305.0) / 460.0)) * w1
             + (35.55 + 1.25 * (5000.0 - d) / 5000.0
                - 1.62 * math.tanh((d - 60.0) / 650.0)
                + 0.2 * math.tanh((d - 35.0) / 100.0)
                + 0.2 * math.tanh((d - 1000.0) / 5000.0)) * w2)
        return t, s

    values = np.asarray([profile(float(d)) for d in depth_m], dtype=np.float64)
    return values[:, 0], values[:, 1]


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
        whole_step_identity="lane1_flux_up3",
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
        whole_step_identity="lane1_flux_up3",
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


def build_gyre_zco_card() -> NEMOTestcaseCard:
    """Certified GYRE: rotated beta-plane, MI96 zco, OMIP RK3 identity."""

    grid = _gyre_grid()
    wet = _closed_box_mask(22, 32)
    wet_np = np.asarray(wet)
    u_wet_native = wet_np * np.roll(wet_np, -1, axis=1)
    v_wet_native = wet_np * np.roll(wet_np, -1, axis=0)
    area_native = np.asarray(grid.dx_T) * np.asarray(grid.dy_T)
    reference_depth = float(_GYRE_GDEPW_1D[30])
    native_3d = (22, 32, 30)
    zco_thickness = np.broadcast_to(_GYRE_E3T_1D[:30], native_3d)
    umask_3d = np.broadcast_to(u_wet_native[..., None], native_3d)
    vmask_3d = np.broadcast_to(v_wet_native[..., None], native_3d)
    f_wet_native = u_wet_native * np.roll(u_wet_native, -1, axis=0)
    f_wet_native[-1, :] = 0.0
    fmask_3d = np.broadcast_to(f_wet_native[..., None], native_3d)
    literal_barotropic_operands = NemoEENBarotropicOperands(
        ff_f=np.asarray(grid.f_v)[1:],
        e3u_0=zco_thickness,
        e3v_0=zco_thickness,
        e3f_0=zco_thickness,
        umask=umask_3d,
        vmask=vmask_3d,
        fmask=fmask_3d,
        hu_0=reference_depth * u_wet_native,
        hv_0=reference_depth * v_wet_native,
        hf_0=reference_depth * f_wet_native,
        e1t=np.asarray(grid.dx_T),
        e2t=np.asarray(grid.dy_T),
        e1u=np.asarray(grid.dx_u)[:, 1:],
        e2u=np.asarray(grid.dy_u)[:, 1:],
        e1v=np.asarray(grid.dx_v)[1:],
        e2v=np.asarray(grid.dy_v)[1:],
        e1f=np.full((22, 32), 106000.0),
        e2f=np.full((22, 32), 106000.0),
    )
    z_ref = create_z_star_from_thicknesses(
        _GYRE_E3T_1D[:30],
        t_depth_ref_m=_GYRE_GDEPT_1D[:30],
        nemo_gdept_0_m=np.broadcast_to(_GYRE_GDEPT_1D[:30], native_3d),
        nemo_gdepw_0_m=np.broadcast_to(_GYRE_GDEPW_1D[:30], native_3d),
        nemo_e3t_0_m=np.broadcast_to(_GYRE_E3T_1D[:30], native_3d),
        nemo_e3w_0_m=np.broadcast_to(_GYRE_E3W_1D[:30], native_3d),
        nemo_hu_0_m=reference_depth * u_wet_native,
        nemo_hv_0_m=reference_depth * v_wet_native,
        nemo_e1e2t_m=area_native,
        nemo_e1e2u_m=area_native,
        nemo_e1e2v_m=area_native,
        nemo_e2u_m=np.full_like(area_native, 106000.0),
        nemo_e1v_m=np.full_like(area_native, 106000.0),
        nemo_een_barotropic_m=literal_barotropic_operands,
    )
    # The generic constructor deliberately recovers dz from a cumulative-sum
    # interface ladder.  NEMO stores both source-produced arrays and its
    # ``e3t_1d`` differs from that recovery by several deep-level ULPs.  Retain
    # both certified operands exactly, as NEMO does.
    z_ref = z_ref._replace(
        H_max=float(_GYRE_GDEPW_1D[30]),
        z_half_ref=jnp.asarray(-_GYRE_GDEPW_1D[:31], dtype=jnp.float64),
        dz_ref=jnp.asarray(_GYRE_E3T_1D[:30], dtype=jnp.float64),
    )
    bottom = jnp.where(wet > 0.0, 29, -1)
    z_coord = create_full_step_coordinate(z_ref, bottom)
    bathymetry = wet * _GYRE_GDEPW_1D[30]
    state = rest_state_latlon_cgrid_ocean(
        grid,
        z_coord,
        T_water_init_C=0.0,
        T_deep=0.0,
        S_uniform=0.0,
        H_max=float(_GYRE_GDEPW_1D[30]),
        land_mask_override=wet,
        H_bathy_override=bathymetry,
    )
    t_profile, s_profile = _gyre_initial_profiles(_GYRE_GDEPT_1D[:30])
    active = np.asarray(z_coord.is_active)
    temperature = np.where(active, t_profile[None, None, :], 0.0)
    salinity = np.where(active, s_profile[None, None, :], 0.0)
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(temperature, dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(salinity, dtype=jnp.float64)),
    )
    # GYRE resolves ln_bt_auto=T to nn_e=50 with nn_bt_flt=3 and
    # rn_bt_alpha=.07 (ocean.output:833-839).  Keep this executed value pinned;
    # auto-resolution is separately checked from the live card geometry.
    model_config = _model_config(
        barotropic_time_filter="nemo_ab3am4",
        n_barotropic_substeps=50,
        bbl_adv_option=0,
        bbl_gamma_s=0.0,
        whole_step_identity="gyre_vector_ene_c2",
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
        "GYRE-zco",
        recipe,
        14400.0,
        4320,
        1,
        0,
        0,
        0.0,
        0.0,
        surface_boundary_condition="gyre_usrdef_sbc",
    )
    validate_nemo_testcase_card(card)
    return card


def gyre_surface_boundary_condition(
    card: NEMOTestcaseCard, t_seconds: float
) -> GYRESurfaceBoundaryCondition:
    """Evaluate GYRE's source-defined seasonal SBC on its rotated T grid.

    ``t_seconds`` is NEMO's current ``kt * rn_Dt`` clock, matching
    ``usrdef_sbc.F90:84-107,109-145,161-184``.  Stress is returned in NEMO's
    grid-aligned ocean convention; no atmosphere/ocean sign conversion is
    applied in this transcription receipt.
    """

    if card.case != "GYRE-zco":
        raise ValueError("GYRE seasonal SBC requires the GYRE-zco card")
    lat = jnp.asarray(card.recipe.grid.native_lat_T_deg, dtype=jnp.float64)
    qsr = nemo_gyre_qsr(lat, t_seconds)
    t_star = nemo_gyre_t_star(lat, t_seconds)
    emp_raw = nemo_gyre_emp(lat, t_seconds)
    wet = jnp.asarray(card.recipe.land_mask) > 0.5
    # Runtime substep evidence resolves a subtle ownership point hidden by the
    # source string: glob_2Dsum receives unmasked ``emp``, but excludes GYRE's
    # 104 non-owned boundary-ring cells.  Those are the same cells represented
    # by ``wet=False`` in this cropped 22x32 card.  The oracle kt=1 eta update
    # pins the resulting 600-owned-cell numerator exactly.
    emp_mean = jnp.sum(emp_raw * wet) / jnp.sum(wet)
    emp = emp_raw - emp_mean * wet
    utau, vtau = nemo_gyre_wind(lat, t_seconds)
    taum = jnp.sqrt(utau * utau + vtau * vtau)
    wndm = jnp.sqrt(taum / (1.22 * 1.5e-3))
    return GYRESurfaceBoundaryCondition(
        qsr.astype(jnp.float64),
        t_star.astype(jnp.float64),
        emp.astype(jnp.float64),
        utau.astype(jnp.float64),
        vtau.astype(jnp.float64),
        taum.astype(jnp.float64),
        wndm.astype(jnp.float64),
    )


def validate_nemo_testcase_card(card: NEMOTestcaseCard) -> None:
    """Reject any card composition not exercised by its named oracle run."""
    expected = {
        "LOCK_EXCHANGE-zco": ("nemo_ab3am4", 1, 0, 0.0),
        "OVERFLOW-zps": ("nemo_boxcar1_ab3", 3, 2, 20.0),
        "GYRE-zco": ("nemo_ab3am4", 50, 0, 0.0),
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
    wet = np.asarray(card.recipe.land_mask) > 0.5
    if card.case == "GYRE-zco":
        if (cfg.barotropic_coriolis_split != "live"
                or cfg.barotropic.barotropic_coriolis != "ene_metric"
                or cfg.barotropic.barotropic_een_coefficient_evaluation
                != "nemo_literal"):
            raise ValueError(
                "GYRE-zco requires the live literal-ENE barotropic composition")
        require_rotation = (
            np.any(np.asarray(card.recipe.grid.f_T) != 0.0)
            and np.count_nonzero(np.any(wet, axis=1)) == 20
            and np.all(np.asarray(card.recipe.grid.sin_alpha_u) != 0.0)
        )
        if not require_rotation:
            raise ValueError("GYRE-zco requires a live rotated beta-plane")
        if card.surface_boundary_condition != "gyre_usrdef_sbc":
            raise ValueError("GYRE-zco requires the analytic usrdef_sbc card")
        return
    # The namelists select ENS, while these Cartesian cases have f=0 and only
    # one wet y row.  Prove the inherited rotation operator is structurally
    # eliminated; otherwise reject rather than silently run an AL81/Matsuno
    # third model.
    f_t = np.asarray(card.recipe.grid.f_T)
    if np.any(f_t != 0.0) or np.count_nonzero(np.any(wet, axis=1)) != 1:
        raise ValueError(
            f"{card.case} legacy rotation is permitted only when f=0 and "
            "the meridional operator is structurally absent")


def build_nemo_testcase_card(case: str) -> NEMOTestcaseCard:
    """Fail-closed dispatch for the certified phase-2 cards."""

    builders = {
        "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card,
        "OVERFLOW-zps": build_overflow_zps_card,
        "GYRE-zco": build_gyre_zco_card,
    }
    if case not in builders:
        raise ValueError(
            f"unknown NEMO testcase {case!r}; expected one of {sorted(builders)}"
        )
    return builders[case]()


__all__ = (
    "NEMOTestcaseCard",
    "GYRESurfaceBoundaryCondition",
    "build_gyre_zco_card",
    "build_lock_exchange_zco_card",
    "build_overflow_zps_card",
    "build_nemo_testcase_card",
    "gyre_horizontal_coordinates",
    "gyre_surface_boundary_condition",
    "gyre_vertical_ladder",
    "validate_nemo_testcase_card",
)
