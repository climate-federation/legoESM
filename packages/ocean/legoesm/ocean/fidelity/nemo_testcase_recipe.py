"""Pure-config legoESM cards for the certified NEMO 5.0.2 test cases.

The domain formulas are transcribed from NEMO's ``tests/LOCK_EXCHANGE`` and
``tests/OVERFLOW`` user definitions.  Dynamics are selected exclusively from
the shared lat-lon C-grid model; this module contains no solver or oracle I/O.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.barotropic_common import nemo_auto_substeps
from legoesm.ocean.eos import NemoSEOSConfig
from legoesm.ocean.fidelity.nemo_recipe import (
    NEMOModelRecipeConfig,
    NEMORecipe,
    nemo_gyre_emp,
    nemo_gyre_qsr,
    nemo_gyre_t_star,
    nemo_gyre_wind,
    nemo_gyre_zero_mean_emp,
    nemo_lat_lon_model_config,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig,
    OceanConvectionConfig,
)
from legoesm.ocean.physics.shortwave_penetration import ShortwavePenetrationConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    NemoEENBarotropicOperands,
    nemo_fe3mask_from_tmask,
    create_full_step_coordinate,
    create_partial_cell_coordinate,
    create_z_star_from_thicknesses,
)

from legoesm import constants


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
    surface_input_operator: str = "none"
    transcendentals: str = "libm"
    precision_policy: str = "fp64"
    # Selected oracle mechanisms that are not yet in the shared execution
    # program.  This is deliberately card data: callers and gates can refuse a
    # partial card rather than silently substituting another scheme.
    unmeasured_features: tuple[str, ...] = ()
    # Coupled-deck option, deliberately tri-state for legacy cards.  ORCA2
    # must spell this out so no caller can inherit an implicit iceberg choice.
    icebergs_enabled: bool | None = None
    iceberg_inputs: tuple[str, ...] | None = None


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
    bbl_adv_option: int, bbl_gamma_s: float,
    bbl_diffusive_option: int, bbl_aht_m2_s: float,
    whole_step_identity: str,
    tke_langmuir_evaluation: str | None,
) -> LatLonCGridOceanConfig:
    """The selectors shared by both certified ``key_qco + key_RK3`` runs."""

    if whole_step_identity not in {
        "lane1_flux_up3", "gyre_vector_ene_c2", "orca2_vector_een_c2",
        "vortex_flux_up3_een", "vortex_vector_een_c2",
    }:
        raise ValueError(
            "unknown whole_step_identity; expected 'lane1_flux_up3' or "
            "'gyre_vector_ene_c2' or 'orca2_vector_een_c2' or "
            "'vortex_flux_up3_een' or 'vortex_vector_een_c2'"
        )

    if whole_step_identity == "vortex_vector_een_c2":
        # DECISION 73 (operator note BJ).  The second VORTEX card is the first
        # one with ONE thing changed: namelist_cfg:182,185 select vector-
        # invariant momentum instead of the flux-form third-order upstream
        # scheme, and NEMO's own resolved run says what that means --
        # "vector form : keg + zad + vor is used" and "total vorticity =
        # Coriolis + relative vorticity" (round-3 ocean.output:723,739).
        #
        # So three things move together, and all three are NEMO's, read from
        # the compiled dispatch rather than chosen here:
        #   * dynadv.f90:135-138 replaces the flux-form momentum advection
        #     with the kinetic-energy gradient plus the vertical advection of
        #     momentum;
        #   * dynvor.f90:861-864 hands the SAME energy-and-enstrophy triad the
        #     relative vorticity as well as the planetary one, where the flux
        #     card's arm gets the planetary one plus a metric term;
        #   * namelist_cfg:183 nn_dynkeg = 0 selects the mean-of-squares
        #     kinetic-energy gradient, not the Hollingsworth correction.
        # Nothing else about the card moves: it is built from the flux card's
        # own configuration and only these fields are replaced, so "one
        # variable" is structural here and not a claim in prose.
        base = _model_config(
            barotropic_time_filter=barotropic_time_filter,
            n_barotropic_substeps=n_barotropic_substeps,
            bbl_adv_option=bbl_adv_option,
            bbl_gamma_s=bbl_gamma_s,
            bbl_diffusive_option=bbl_diffusive_option,
            bbl_aht_m2_s=bbl_aht_m2_s,
            whole_step_identity="vortex_flux_up3_een",
            tke_langmuir_evaluation=tke_langmuir_evaluation,
        )
        return base._replace(
            momentum_advection="vector_invariant",
            # dyn_adv no longer carries the momentum flux, so the flux-form
            # scheme selector must not be left behind pointing at the routine
            # the other card runs.  The field has no "unused" value, so this
            # card carries the SAME inert value GYRE's and ORCA2's
            # vector-invariant cards carry (the library default), which is the
            # existing convention for a selector NEMO never reaches on a
            # vector-form deck.
            momentum_flux_scheme="upwind",
            # dynadv.f90:138 CALL dyn_zad: the vertical advection of momentum
            # is the advective form, not the flux-form UP3 vertical flux the
            # other card runs.
            vertical_momentum_scheme="nemo_advective",
            # stp2d.f90:153 CALL wzv(..., np_velocity): the vertical velocity
            # dyn_zad consumes is NEMO's own wzv result -- the bottom-up
            # integral of the live-thickness horizontal divergence plus
            # NEMO's scale-factor term (sshwzv.f90:295-298) -- NOT legoESM's
            # generic z-star diagnosis, which redistributes the column's own
            # surface tendency through the column and forces the vertical
            # velocity to zero at the surface.  Measured on this card's own
            # kt=1 boundary (round-5 receipt): the generic form is the WHOLE
            # of the vector card's first-stage momentum error.  GYRE, DINO
            # and ORCA2 already select this arm; VORTEX inherited the
            # generic default from the flux card, which never calls dyn_zad.
            zad_qco_evaluation="nemo_literal",
            # stprk3.f90:225 -> stp2d.f90:149 -> stp2d.f90:153: NEMO's RK3
            # program leaves the PREVIOUS step's linear extrapolation in the
            # after slot, and that is what the first wzv call's scale-factor
            # term is built from.  STATED here, never inferred from the time
            # integrator or any other field (decision 75).
            nemo_first_wzv_after_ssh="rk3_extrapolated_carried",
            # nn_dynkeg = 0 (namelist_cfg:183); dynkeg.f90 takes its
            # mean-of-squares arm, not the Hollingsworth correction.
            ke_gradient_scheme="c2",
            # dynvor.f90:861-864, n_dynadv = np_VEC_c2: ntot = np_CRV, so the
            # triad is handed the relative vorticity PLUS the planetary one.
            # This is the arm the ORCA2 card already selects; nothing new is
            # written for it here.
            vorticity_scheme="een_total",
        )

    if whole_step_identity == "vortex_flux_up3_een":
        # VORTEX runs the SAME flux-form UP3 / FCT2 / hpg_sco / zdfcst RK3
        # program as the two tanks (namelist_cfg:143-152,182-185,198,252-260),
        # so build that identity and change only what its own namelist selects
        # differently.
        #
        # WHAT IS *NOT* CHANGED HERE, AND WHY THE CARD FAILS ITS EXECUTION
        # GATE.  namelist_cfg:193 selects ln_dynvor_een with a LIVE beta-plane
        # f, and dynvor.F90:874 routes that to np_EEN.  Under
        # ln_dynadv_vec=.false. (namelist_cfg:182) NEMO's dyn_vor takes its
        # FLUX-FORM arm, which calls vor_een on the PLANETARY vorticity alone
        # -- so on this case EEN *is* the Coriolis discretisation, a triad-
        # weighted f x u.  legoESM binds its own EEN arm to VECTOR-INVARIANT
        # momentum (ocean_model_latlon_cgrid.py:4392-4400 refuses the pair
        # outright, because its flux-form branch never receives f_vtx) and
        # gives the flux-form branch the 4-point C-grid average instead.
        # Selecting een_total here would not build; selecting nothing and
        # running the 4-point average would silently substitute a different
        # Coriolis operator for NEMO's.  The card therefore leaves the
        # inherited selectors alone and DECLARES the gap in
        # unmeasured_features, which makes
        # validate_nemo_testcase_card_for_execution refuse it.
        base = _model_config(
            barotropic_time_filter=barotropic_time_filter,
            n_barotropic_substeps=n_barotropic_substeps,
            bbl_adv_option=bbl_adv_option,
            bbl_gamma_s=bbl_gamma_s,
            bbl_diffusive_option=bbl_diffusive_option,
            bbl_aht_m2_s=bbl_aht_m2_s,
            whole_step_identity="lane1_flux_up3",
            tke_langmuir_evaluation=tke_langmuir_evaluation,
        )
        return base._replace(
            # namelist_cfg's namzdf block does NOT set ln_zad_Aimp, so
            # namelist_ref:1177 leaves it .false. and NEMO never compiles the
            # Courant-dependent implicit vertical advection.  The tanks share
            # this identity and their campaign decks resolve it .true., which
            # is why the inherited value is True; VORTEX must state its own.
            # This is executed physics (ocean_model_latlon_cgrid.py takes a
            # different arm), not metadata, so it is explicit here.
            adaptive_implicit_vertadv=False,
            # namelist_cfg:130 selects ln_seos, not the campaign's TEOS-10
            # (decision 69, operator note BG).  eosbn2.F90:300-302 is the
            # density statement; the coefficients are namelist_cfg:132-138 and
            # the unset references are eosbn2.F90:89-90.  Both are stated here
            # because a defaulted coefficient is a hidden choice: the shared
            # NemoSEOSConfig defaults are DINO's, not this case's.
            eos="nemo_seos",
            eos_nemo_seos=_VORTEX_SEOS,
            # This card's S-EOS is depth-blind (rn_mu1 = rn_mu2 = 0), so the
            # geometric-depth EOS argument the TEOS-10 cards need buys nothing
            # here and its ladder machinery would be dead weight.
            eos_depth="insitu",
            # namelist_cfg:193 ln_dynvor_een with namelist_cfg:182
            # ln_dynadv_vec=.false.: dynvor.F90:874 routes np_EEN and
            # dyn_vor_init (dynvor.F90:891-893) hands the flux-form arm
            # ntot = np_CME, whose metric term is bitwise zero on this
            # Cartesian mesh (usrdef_hgr.F90:160-163).  So the energy-and-
            # enstrophy triad IS this case's Coriolis operator, and the card
            # selects it rather than legoESM's 4-point C-grid average.
            vorticity_scheme="een_planetary",
            # The triad already carries f, so the Matsuno rotation must be off
            # or f would enter twice (dyn_vor is called once per stage).
            coriolis_scheme="explicit_ab2",
            # domzgr_substitute.h90:130 e3f_vor = e3f_0vor*(1+r3f), built by
            # dynvor.F90:918-950 and domqco.F90:233-246.
            een_e3f_scheme="nemo_avg4",
            # dynvor.F90:791-792,804-806 weight the transport by the neighbour
            # face width and normalise by the local one.
            een_metric_weighting="nemo",
            # vor_een never fills or masks zwz (ln_dynvor_msk is irrelevant on
            # the np_COR/np_CME arms), so the vertex field stays live at the
            # coast; a Neumann fill would be a different operator.
            een_q_boundary="nemo_live",
            # dynspg_ts.F90:1326-1345 (dyn_cor_2D_init, np_EEN): the barotropic
            # substeps run the SAME triad on ff_f/e3f_vor, accumulated over the
            # column with the live e3u/e3v(Kmm).  The split must be "live" so
            # the depth-mean of the baroclinic Coriolis above is subtracted
            # with the SAME stencil the substeps add back; a frozen or 4-point
            # subtraction would leave a residual rotation.
            barotropic=base.barotropic._replace(
                barotropic_coriolis="een_metric",
                barotropic_een_seed="nemo_kmm",
                barotropic_een_coefficient_evaluation="nemo_literal",
            ),
            barotropic_coriolis_split="live",
        )

    if whole_step_identity == "orca2_vector_een_c2":
        # ORCA2 runs the same coupled WS-RK3/QCO stage program as GYRE, but
        # dynvor.F90:1326-1332 selects EEN (not ENE), eosbn2.F90 selects EOS-80,
        # and traqsr.F90 selects RGB light.  Build this by specializing the
        # shared GYRE identity; no stage formula lives in this card.
        base = _model_config(
            barotropic_time_filter=barotropic_time_filter,
            n_barotropic_substeps=n_barotropic_substeps,
            bbl_adv_option=bbl_adv_option,
            bbl_gamma_s=bbl_gamma_s,
            bbl_diffusive_option=bbl_diffusive_option,
            bbl_aht_m2_s=bbl_aht_m2_s,
            whole_step_identity="gyre_vector_ene_c2",
            tke_langmuir_evaluation=tke_langmuir_evaluation,
        )
        # ORCA2 namelist_cfg:389 selects TKE, so zdfphy.F90:220-223 sets
        # l_zdfsh2=.TRUE. and :264-286 calls zdf_sh2 before zdf_tke.  Under
        # key_RK3, stprk3.F90:164-165 passes Kbb=Nbb and Kmm=Nbb: the executed
        # zdfsh2.F90:78-100 arm is face-native Nbb*Nbb (whole-step entry), with
        # avm summed on the faces and the same live-QCO face metric in both
        # divisor factors.  Never call Nbb "now" (time-level Rule 1d).
        tke = base.physics.vertical_mixing.tke._replace(
            n2_eos_form="eos80",
            tke_shear_production="nemo_face_native_nbb2",
            tke_shear_avm_weighting="nemo_face",
            tke_shear_metric_source="nemo_qco_live_face",
            # namelist_ref:1255-1259; zdftke.F90:255.  Mode 1 is the shared
            # source-literal tanh(10*fr_i) arm, not raw fr_i (mode 2).
            eice=1,
        )
        convection = base.physics.convection._replace(
            enhanced_diffusion=base.physics.convection.enhanced_diffusion._replace(
                n2_eos_form="eos80"
            )
        )
        physics = base.physics._replace(
            vertical_mixing=base.physics.vertical_mixing._replace(tke=tke),
            convection=convection,
            shortwave_penetration=ShortwavePenetrationConfig(
                scheme="nemo_qsr_rgb", nemo_time_step_s=10800.0
            ),
        )
        return base._replace(
            eos="nemo_eos80",
            physics=physics,
            nemo_first_wzv_after_ssh="rk3_extrapolated",
            # Round 163 (Decision 55, note AT): ORCA2-zps resolves the SAME
            # rk3_ws+vector_invariant+nemo_literal program GYRE-zco does (via
            # this shared base), but has never been measured under the
            # second continuity solve -- explicit False, not an inference
            # from `eos` (round-163 review BLOCKER, closed by making this an
            # explicit per-card config choice instead).
            nemo_stage_momentum_wzv_split=False,
            vorticity_scheme="een_total",
            # nn_e3f_typ=0 and ln_dynvor_msk=F, identical source meanings to
            # the already-shared literal EEN operands used by GYRE's ENE arm.
            een_e3f_scheme="nemo_avg4",
            een_metric_weighting="nemo",
            een_q_boundary="nemo_live",
            barotropic=base.barotropic._replace(
                barotropic_coriolis="een_metric",
            ),
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
            tke_dry_wmask=True,
            kappaM_max=float("inf"),
            tke_preclosure_coeff_source="carried_previous_step",
            tke_matrix_evaluation="nemo_literal",
            tke_solver_evaluation="nemo_literal",
            tke_etau_exponential_evaluation="jax_expression",
            tke_htau_evaluation="jax_expression",
            tke_mxl_raw_evaluation="factored",
            # Card-owned selector: GYRE chooses nemo_literal, ORCA2 retains
            # the vectorized parent arm, and non-TKE cards pass None.
            tke_langmuir_evaluation=(
                tke_langmuir_evaluation
                if tke_langmuir_evaluation is not None
                else config.physics.vertical_mixing.tke.tke_langmuir_evaluation
            ),
            tke_shear_evaluation_stage="step_entry",
            # Decision 36 (user, 2026-09-12): NEMO's shear production
            # statement, not legoESM's legacy centred one.  zdfsh2.f90:83-114
            # (R59TKE ppsrc) forms sh2 at the uw/vw faces from the product of
            # the Kmm and Kbb face velocity differences, the viscosity SUMMED
            # on the two faces, and the live stretched face metric
            # e3w_1d*(1+r3u); stprk3.f90:168 calls zdf_phy(kstp, Nbb, Nbb,
            # Nrhs), so both levels are the step-entry state.  Round 61
            # measured the legacy statement 4.1e-8 off NEMO in every wet
            # interface while every other closure operand agreed to 3e-17.
            # The DINO NEMO card selects nemo_face_native (its leap-frog
            # integrator carries the before-level state); under RK3 both
            # factors are the step-entry velocities, which is the now2 variant
            # (the model refuses nemo_face_native without a before state).
            tke_shear_production="nemo_face_native_now2",
            tke_shear_avm_weighting="nemo_face",
            tke_shear_metric_source="nemo_qco_live_face",
            # RK3 has no leapfrog eta-before carrier; the QCO live metric is
            # reconstructed at the current stage from the same raw ladder.
            tke_n2_evaluation_stage="step_entry",
            # stprk3.F90:154-181 evaluates the closure N2 at Nbb before any
            # stage.  On RK3 the canonical nemo_before operand is the same
            # whole-step entry tracer used by the EVD twin below.
            tke_n2_time_level="nemo_before",
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
                    # RK3 stprk3.F90:154-181 computes rn2b on Nbb, copies
                    # rn2=rn2b, and calls zdf_phy(Nbb,Nbb) before any stage.
                    # The canonical NEMO time-level option therefore samples
                    # the whole-step entry tracer for both EVD trigger arms.
                    evd_n2_time_level="nemo_now_before",
                ),
            ),
            shortwave_penetration=ShortwavePenetrationConfig(
                scheme="nemo_qsr_2bd", water_type="I", nemo_time_step_s=14400.0
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
            # stprk3.f90:225 -> stp2d.f90:149 -> stp2d.f90:153: NEMO's RK3
            # program leaves the PREVIOUS step's linear extrapolation in the
            # after slot, and that is what the first wzv call's scale-factor
            # term is built from.  STATED here, never inferred from the time
            # integrator or any other field (decision 75).
            nemo_first_wzv_after_ssh="rk3_extrapolated_carried",
            wzv_call2_evaluation="nemo_literal",
            # Round 163 (Decision 55, note AT): GYRE-zco's own measured,
            # landed choice -- explicit here, not inferred from EOS or any
            # other field.  ORCA2-zps specializes this SAME branch below and
            # overrides it back to False, because it resolves the identical
            # program but has never been measured under it.
            nemo_stage_momentum_wzv_split=True,
            vorticity_scheme="ene_total",
            # key_qco e3f_vor = e3f_0vor*(1+r3f); dom_qco_r3c_RK3 builds
            # r3f from the four-cell surface-weighted SSH average
            # (domqco.F90:233-246), not the generic vertex minimum.
            # Resolved nn_e3f_typ=0: e3f_0vor is the masked four-cell
            # reference sum divided by four, with live r3f applied only at
            # fully wet F points (dynvor.F90:918-950).
            een_e3f_scheme="nemo_avg4",
            # dynvor.F90:518-535 carries e2u/e1v into the transport operands
            # and divides the final U/V terms by e1u/e2v.
            een_metric_weighting="nemo",
            # Resolved ln_dynvor_msk=.false.: keep coastal relative
            # vorticity live instead of Neumann-filling the F-point q field.
            een_q_boundary="nemo_live",
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
            bbl_diffusive_option=bbl_diffusive_option,
            bbl_aht_m2_s=bbl_aht_m2_s,
            # NEMO's e3w(Kmm) implicit-solve divisor (trazdf.F90:219-221,
            # dynzdf.F90:200-203) comes WITH this identity -- it is not a
            # separate flag, because NEMO has no such switch.  Same removal as
            # the shared _model_config above; the GYRE card runs the same
            # routine, so it takes the same fix.
            zdf_implicit_solver_evaluation="nemo_literal",
            # GYRE resolves ln_non_lin=T + ln_drgimp=T together with
            # ln_dynspg_ts=T.  NEMO owns this as one composition: frozen Kmm
            # rCdU_bot in the explicit external-mode substeps, the baroclinic
            # residual correction, and the bottom-cell implicit dynzdf
            # diagonal (zdfdrg.F90:138-190; dynspg_ts.F90:699-705,1584-1643;
            # dynzdf.F90:148-160,293-305).  These are the already-canonical
            # DINO options, selected as a bundle rather than a GYRE-only
            # reimplementation.
            zdf_drag_in_matrix=True,
            zdf_baroclinic_only=True,
            barotropic_drag_substep=True,
            use_conservation_fixer=False,
            # GYRE HAS NO SALT FLUX.  usrdef_sbc.f90:160 is
            # `sfx (ji,jj) = 0.0_wp   ! no salt flux`, and the card compiles
            # key_qco (cpp_GYRE_OMIP_L2_P3_SM_YRPERT.fcm:1), so evaporation
            # minus precipitation is carried by the cell VOLUME and by nothing
            # else.  legoESM's library default is a VIRTUAL SALT FLUX -- a
            # salinity source the oracle does not have -- so the card selects
            # the volume channel HERE rather than leaving every harness to
            # remember a _replace.  Leaving it to the harnesses is what put two
            # programs on one card: the certified kt=1..10 gate applied the
            # pair, the from-rest year harness did not, and the two receipts
            # then disagreed about the same step.
            #
            # fix_eta_drift: OFF (decision 35, ASKED, user 2026-09-11: "no
            # hidden extras -- a fixer could hide model errors").  NEMO adds
            # emp LOCALLY: sshwzv.f90:137 `pssh(:,:,Kaa) = ( pssh(:,:,Kbb) -
            # rDt * ( r1_rho0 * emp(:,:) + zhdiv(:,:) ) ) * ssmask(:,:)` and,
            # inside the barotropic loop, dynspg_ts.f90:553; it has no
            # global projection anywhere in its free-surface path.  legoESM's
            # fix_eta_drift is a global uniform eta shift NEMO lacks (Rule 9).
            # On this card it corrected nothing -- NEMO de-means the card's
            # own emp (usrdef_sbc.f90:151-157; area mean 5.5e-22 kg/m2/s) --
            # and seeded a 4.3e-19 m / 1.4e-14 K difference at kt=2 (card
            # reconciliation receipt, section 7).  The model admits the
            # unprojected real_freshwater pair only under nemo_literal
            # barotropic continuity, where the source enters eta by NEMO's
            # own substep statement; this card selects that continuity below.
            freshwater_closure="real_freshwater",
            fix_eta_drift=False,
            barotropic=config.barotropic._replace(
                barotropic_diffusion_alpha=0.0,
                barotropic_face_depth="nemo_ssh_avg",
                barotropic_continuity_evaluation="nemo_literal",
                barotropic_transport_accumulation_evaluation="nemo_literal",
                barotropic_seed_face_depth="nemo_ssh_avg",
                barotropic_seed_evaluation="nemo_literal",
                # dynspg_ts.F90:484-500 seeds the window from the CARRIED
                # uu_b/vv_b (oce.F90:39,99), not from a 3-D reduction.
                nemo_prognostic_barotropic_state=True,
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
        # dynadv_up3.F90:166,169-170 -- the NEMO-referenced UP3 arm (the
        # T-point fluxes select the upwind curvature by the advected-
        # velocity pair).  See UP3_REFERENCE_SELECTOR.
        momentum_flux_scheme="nemo_up3",
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
        # dynspg_ts.F90:484-500 seeds the window from the CARRIED uu_b/vv_b
        # (oce.F90:39,99), not from a 3-D reduction.
        nemo_prognostic_barotropic_state=True,
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
        # DECISION 19, asked and answered by the user ("Do as NEMO does"):
        # both tanks resolve ln_drgimp = T (lock_kt1_10/ocean.output:560,
        # overflow_kt1_10/ocean.output:672) and ln_dynspg_ts = T (:752, :869),
        # so NEMO takes the branch that removes the barotropic velocity from
        # the implicit vertical solve on every level -- their own compiled arm
        # at tests/<CARD>_OMIP_L1_P3_R33ZDF/BLD/ppsrc/nemo/dynzdf.f90:186-191.
        # legoESM did not, and that was a transcription gap, not a choice.
        #
        # The companion statement at :192-200, which adds the barotropic
        # bottom stress back at the deepest wet level, is EXACTLY ZERO on
        # these two cards and is therefore not transcribed here: rCdU_bot in
        # each card's own round-33 record is 0.0 on every owned cell (0 of 390
        # on LOCK, 0 of 606 on OVERFLOW).  GYRE's is 5.0e-05 on 600 of 704, so
        # it is live there -- and there it is already carried, by
        # zdf_drag_in_matrix, which stays OFF here because nothing on these
        # cards needs it.
        zdf_baroclinic_only=True,
        bbl_adv_option=bbl_adv_option,
        bbl_gamma_s=bbl_gamma_s,
        adaptive_implicit_vertadv=True,
        implicit_vertical_mixing=True,
        # NEMO's e3w(Kmm) implicit-solve divisor (trazdf.F90:219-221,
        # dynzdf.F90:200-203) comes WITH this identity — it is not a
        # separate flag, because NEMO has no such switch.
        zdf_implicit_solver_evaluation="nemo_literal",
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
        bbl_diffusive_option=bbl_diffusive_option,
        bbl_aht_m2_s=bbl_aht_m2_s,
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
        ff_f=jnp.asarray(source["ff_f"], dtype=jnp.float64),
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
        nemo_prognostic_barotropic_velocity=True,
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
        # mesh_mask.nc:e3w_1d(1:20) is exactly 1 m on the shipped case.
        nemo_e3w_0_m=np.broadcast_to(
            np.ones(20, dtype=np.float64), (3, 130, 20)),
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
        bbl_diffusive_option=0,
        bbl_aht_m2_s=0.0,
        whole_step_identity="lane1_flux_up3",
        tke_langmuir_evaluation=None,
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
    _gdept_1d = 10.0 + 20.0 * np.arange(100, dtype=np.float64)
    z_ref = create_z_star_from_thicknesses(
        jnp.full((100,), 20.0),
        t_depth_ref_m=_gdept_1d,
        # tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:157-168 leaves gdept/e3w on the
        # uniform reference ladder under ld_zps, so mesh_mask.nc:e3w_0 is
        # exactly 20 m in every column INCLUDING the partial-bottom ones.
        # Carry the source operands on that 1-D ladder: both dynhpg (which
        # must not reconstruct W spacing from the clipped T-cell thickness)
        # and the raw-W ZDF/wAimp consumers read this one field.
        nemo_gdept_0_m=_gdept_1d,
        nemo_e3w_0_m=np.full((100,), 20.0, dtype=np.float64),
    )
    z_coord = create_partial_cell_coordinate(
        z_ref, bathymetry, bottom_index_rule="nemo_tpoint"
    )
    # Exact unmasked BBL face scale factors from this test case's source.
    # usrdef_zgr.F90:171-186 starts every e3t at 20 m, replaces the bottom
    # and first-below-bottom records by the partial thickness, then (the
    # bathymetry increases monotonically in i and is identical in j) assigns
    # e3u=e3v=e3t at the west/south source column.  trabbl.F90:529-531 later
    # gathers these arrays at BOTH adjacent bottom indices, including a level
    # below the shallower wet column; masked h_partial cannot reconstruct it.
    _h_partial = np.asarray(z_coord.h_partial, dtype=np.float64)
    _bottom = np.asarray(z_coord.bottom_level, dtype=np.int32)
    _raw_e3t = np.broadcast_to(
        np.asarray(z_coord.dz_ref, dtype=np.float64), _h_partial.shape
    ).copy()
    for _j, _i in np.argwhere(_bottom >= 0):
        _k = int(_bottom[_j, _i])
        _raw_e3t[_j, _i, _k] = _h_partial[_j, _i, _k]
        if _k + 1 < _raw_e3t.shape[-1]:
            _raw_e3t[_j, _i, _k + 1] = _h_partial[_j, _i, _k]
    z_coord = z_coord._replace(
        nemo_e3t_0=jnp.asarray(_raw_e3t),
        nemo_bbl_e3u_0=jnp.asarray(_raw_e3t[:, :-1, :]),
        nemo_bbl_e3v_0=jnp.asarray(_raw_e3t[:-1, :, :]),
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
        bbl_diffusive_option=0,
        bbl_aht_m2_s=1000.0,
        whole_step_identity="lane1_flux_up3",
        tke_langmuir_evaluation=None,
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
        ff_f=np.asarray(grid.ff_f),
        e3u_0=zco_thickness,
        e3v_0=zco_thickness,
        e3f_0=zco_thickness,
        umask=umask_3d,
        vmask=vmask_3d,
        fmask=fmask_3d,
        fe3mask=fmask_3d,
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
        nemo_prognostic_barotropic_velocity=True,
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
        bbl_diffusive_option=0,
        bbl_aht_m2_s=0.0,
        whole_step_identity="gyre_vector_ene_c2",
        tke_langmuir_evaluation="nemo_literal",
    )
    # stprk3 computes rn2b once from Nbb before zdf_phy, which consumes that
    # field in zdf_mxl, and then passes the same rn2b to ldf_slp.  Keep the
    # mixed-layer recurrence and the slope denominator on that one carried
    # step-entry field instead of evaluating eosbn2 a second time.
    model_config = model_config._replace(
        gm_redi=model_config.gm_redi._replace(
            slope_n2_evaluation="carried_step_entry"))
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
    emp = nemo_gyre_zero_mean_emp(emp_raw, wet)
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


def _orca2_depth_ladder(e3t_1d: np.ndarray, e3w_1d: np.ndarray):
    """NEMO ``e3_to_depth`` scalar recurrence for the active ORCA2 levels.

    ``depth_e3.F90:109-130`` evaluates the two assignments in this order.  A
    vector cumulative sum is not substituted because its reduction tree is a
    different floating-point program.
    """

    nlev = int(e3t_1d.size)
    gdepw = np.empty(nlev, dtype=np.float64)
    gdept = np.empty(nlev, dtype=np.float64)
    gdepw[0] = 0.0
    gdept[0] = 0.5 * float(e3w_1d[0])
    for k in range(1, nlev):
        gdepw[k] = gdepw[k - 1] + float(e3t_1d[k - 1])
        gdept[k] = gdept[k - 1] + float(e3w_1d[k])
    return gdept, gdepw


def _orca2_masks(bottom_level: np.ndarray, strait_shlat: np.ndarray):
    """Reproduce ``dommsk`` masks on the de-haloed ORCA2 T-fold domain.

    The interior products are ``dommsk.F90:146-172``.  The no-slip extension
    is :207-217 with resolved ``rn_shlat=2`` and the domain-file strait
    override is :226-238.  The two final-row assignments are the de-haloed
    jperio=4 T-fold images that NEMO's ``lbc_lnk`` writes.
    """

    bottom = np.asarray(bottom_level, dtype=np.int32)
    nlat, nlon = bottom.shape
    nlev = int(bottom.max(initial=0))
    tmask = np.arange(nlev)[None, None, :] < bottom[..., None]
    umask = tmask & np.roll(tmask, -1, axis=1)
    vmask = tmask & np.roll(tmask, -1, axis=0)
    fmask_free = (
        tmask
        & np.roll(tmask, -1, axis=1)
        & np.roll(tmask, -1, axis=0)
        & np.roll(np.roll(tmask, -1, axis=0), -1, axis=1)
    )

    # T-fold boundary images on a de-haloed ORCA grid (Iperio=1, NFold=1,
    # NFtype=T).  The V row maps the penultimate native row with the T
    # permutation; the F row has the one-point stagger offset.
    perm_t = (nlon - np.arange(nlon)) % nlon
    vmask[-1] = vmask[-2, perm_t]

    # Resolved rn_shlat=2: replace zero free-slip F masks adjacent to any wet
    # U/V face.  Keep this as floating-point because NEMO deliberately emits
    # values 0, 0.5, 1 and 2 after the strait-file override.
    u_north = np.roll(umask, -1, axis=0)
    v_east = np.roll(vmask, -1, axis=1)
    adjacent = np.maximum.reduce((umask, u_north, vmask, v_east))
    fmask = np.where(fmask_free, 1.0, 2.0 * adjacent.astype(np.float64))
    shlat = np.asarray(strait_shlat, dtype=np.float64)
    fmask = np.where(shlat[..., None] >= 0.0, shlat[..., None], fmask)
    perm_f = (nlon - np.arange(nlon) - 1) % nlon
    fmask[-1] = fmask[-2, perm_f]
    return tmask, umask, vmask, fmask


def build_orca2_zps_card(deck_root: str | Path) -> NEMOTestcaseCard:
    """Build the source-file-driven ORCA2+SI3 card through ocean kt=1 entry.

    This constructor intentionally requires the external deck path.  It reads
    NEMO's own domain and initial-condition files; it never synthesizes a
    nominal two-degree grid.  Selected mechanisms not yet supported by the
    shared WS-RK3 program remain named in ``unmeasured_features`` and make the
    Phase-2 gate stop before their first consumer.
    """

    try:
        import netCDF4  # noqa: N813
    except ImportError as exc:
        raise ImportError("netCDF4 is required to build the ORCA2 card") from exc

    root = Path(deck_root)
    domain_path = root / "ORCA_R2_zps_domcfg.nc"
    temperature_path = root / "data_1m_potential_temperature_nomask.nc"
    salinity_path = root / "data_1m_salinity_nomask.nc"
    required = (domain_path, temperature_path, salinity_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing ORCA2 deck files: {missing}")

    with netCDF4.Dataset(domain_path, "r") as ds:
        e3t_1d = np.asarray(ds.variables["e3t_1d"][:30], dtype=np.float64)
        e3w_1d = np.asarray(ds.variables["e3w_1d"][:30], dtype=np.float64)
        bottom = np.asarray(ds.variables["bottom_level"][:], dtype=np.int32)
        strait = np.asarray(ds.variables["strait_shlat"][:], dtype=np.float64)
        raw_e3 = {
            name: np.moveaxis(
                np.asarray(ds.variables[name][:30], dtype=np.float64), 0, -1
            )
            for name in ("e3t_0", "e3u_0", "e3v_0", "e3f_0")
        }
        metric = {
            name: np.asarray(ds.variables[name][:], dtype=np.float64)
            for name in (
                "e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
                "ff_f", "gphit",
            )
        }

    grid = create_tripole_grid(
        domain_path,
        radius=float(NEMO_CONSTANTS_CONFIG.R_earth),
        omega=float(NEMO_CONSTANTS_CONFIG.Omega),
        dtype=jnp.float64,
        min_dx_m=0.0,
        fold_convention="(n_lon-i)%n_lon",
        # ORCA2's domain file supplies ff_t and ff_f, so NEMO reads them
        # instead of recomputing 2*Omega*sin(lat) (domhgr.F90:222-227).  This
        # card is the NEMO-literal one, so it opts in; every other tripole
        # caller keeps the analytic default.
        use_mesh_coriolis=True,
    )
    tmask, umask, vmask, fmask = _orca2_masks(bottom, strait)
    # dommsk.F90:146-198 copies the four-T-cell free-slip mask into
    # fe3mask before rn_shlat and strait_shlat alter the vorticity fmask.
    fe3mask = np.asarray(
        nemo_fe3mask_from_tmask(tmask.astype(np.float64), grid=grid),
        dtype=np.float64,
    )
    gdept, gdepw = _orca2_depth_ladder(e3t_1d, e3w_1d)
    h_partial = raw_e3["e3t_0"] * tmask
    bathymetry = np.sum(h_partial, axis=-1)
    hu = np.sum(raw_e3["e3u_0"] * umask, axis=-1)
    hv = np.sum(raw_e3["e3v_0"] * vmask, axis=-1)
    # domain.F90:149-152 uses V masks on the F-column depth, not fmask.
    hf = np.sum(
        raw_e3["e3f_0"] * vmask * np.roll(vmask, -1, axis=1), axis=-1
    )
    hf[-1] = hf[-2, (grid.n_lon - np.arange(grid.n_lon) - 1) % grid.n_lon]
    operands = NemoEENBarotropicOperands(
        ff_f=metric["ff_f"],
        e3u_0=raw_e3["e3u_0"],
        e3v_0=raw_e3["e3v_0"],
        e3f_0=raw_e3["e3f_0"],
        umask=umask.astype(np.float64),
        vmask=vmask.astype(np.float64),
        fmask=fmask,
        fe3mask=fe3mask,
        hu_0=hu,
        hv_0=hv,
        hf_0=hf,
        e1t=metric["e1t"],
        e2t=metric["e2t"],
        e1u=metric["e1u"],
        e2u=metric["e2u"],
        e1v=metric["e1v"],
        e2v=metric["e2v"],
        e1f=metric["e1f"],
        e2f=metric["e2f"],
    )
    z_ref = create_z_star_from_thicknesses(
        e3t_1d,
        t_depth_ref_m=gdept,
        nemo_gdept_0_m=np.broadcast_to(gdept, h_partial.shape),
        nemo_gdepw_0_m=np.broadcast_to(gdepw, h_partial.shape),
        nemo_e3t_0_m=raw_e3["e3t_0"],
        # Under key_vco_1d3d, domzgr_substitute.h90:71-101 keeps e3w on its
        # 1-D reference ladder while T/U/V/F thicknesses are 3-D.
        nemo_e3w_0_m=np.broadcast_to(e3w_1d, h_partial.shape),
        nemo_hu_0_m=hu,
        nemo_hv_0_m=hv,
        nemo_e1e2t_m=metric["e1t"] * metric["e2t"],
        nemo_e1e2u_m=metric["e1u"] * metric["e2u"],
        nemo_e1e2v_m=metric["e1v"] * metric["e2v"],
        nemo_e2u_m=metric["e2u"],
        nemo_e1v_m=metric["e1v"],
        nemo_een_barotropic_m=operands,
    )
    z_coord = create_partial_cell_coordinate(
        z_ref, bathymetry, bottom_index_rule="nemo_tpoint"
    )._replace(
        h_partial=jnp.asarray(h_partial, dtype=jnp.float64),
        bottom_level=jnp.asarray(bottom - 1, dtype=jnp.int32),
        is_active=jnp.asarray(tmask),
        nemo_e3t_0=jnp.asarray(raw_e3["e3t_0"], dtype=jnp.float64),
        nemo_bbl_e3u_0=jnp.asarray(raw_e3["e3u_0"], dtype=jnp.float64),
        nemo_bbl_e3v_0=jnp.asarray(raw_e3["e3v_0"], dtype=jnp.float64),
    )

    # fld_read.F90:181-227: at kt=1 (0.0625 d), December and January are
    # centred at -15.5 and +15.5 d.  Preserve the source multiply-add order.
    after_weight = np.float64(249.0 / 496.0)
    before_weight = np.float64(1.0) - after_weight
    with netCDF4.Dataset(temperature_path, "r") as ds:
        t_dec = np.asarray(ds.variables["votemper"][11, :30], dtype=np.float64)
        t_jan = np.asarray(ds.variables["votemper"][0, :30], dtype=np.float64)
    with netCDF4.Dataset(salinity_path, "r") as ds:
        s_dec = np.asarray(ds.variables["vosaline"][11, :30], dtype=np.float64)
        s_jan = np.asarray(ds.variables["vosaline"][0, :30], dtype=np.float64)
    temperature = np.moveaxis(before_weight * t_dec + after_weight * t_jan, 0, -1)
    salinity = np.moveaxis(before_weight * s_dec + after_weight * s_jan, 0, -1)
    temperature = np.where(tmask, temperature, 0.0)
    salinity = np.where(tmask, salinity, 0.0)

    # iceistate.F90:262-291 creates one-category ice from the surface T/S and
    # hemisphere, :309-393 distributes it over jpl=5 while conserving volume,
    # and :398-427 removes the globally averaged snow+ice+pond mass from SSH.
    # The resolved pond arm is live (a_p=.2*a_i, h_p=.05 m), so it contributes
    # 9 kg m-2 wherever initial ice exists; omitting it misses the oracle SSH.
    surface_s = salinity[..., 0]
    surface_t = temperature[..., 0]
    freezing_t = (
        -0.0575
        + 1.710523e-3 * np.sqrt(surface_s)
        - 2.154996e-4 * surface_s
    ) * surface_s
    surface_wet = tmask[..., 0]
    initial_ice = ((surface_t - freezing_t) * surface_wet < 2.0) & surface_wet
    ice_thickness = np.where(np.asarray(grid.f_T) >= 0.0, 2.0, 1.0)
    snow_ice_pond_mass = np.where(
        initial_ice,
        0.9 * (constants.rho_snow * 0.2 + constants.rho_ice * ice_thickness)
        + constants.rho_water * 0.18 * 0.05,
        0.0,
    )
    # dom_uniq (domutl.F90:91-121) excludes the duplicated right half of the
    # T-fold row from glob_2Dsum.  On this de-haloed 180-column T-fold, the
    # retained half is i=0:89.  The exact reduction below reproduces the
    # source-produced -0.09343505872684969 m adjustment.
    unique_t = np.ones(surface_wet.shape, dtype=np.float64)
    unique_t[-1, grid.n_lon // 2 :] = 0.0
    area = metric["e1t"] * metric["e2t"]
    ssh_adjustment = np.sum(
        snow_ice_pond_mass / NEMO_CONSTANTS_CONFIG.rho_0 * area * unique_t
    ) / np.sum(area * surface_wet * unique_t)
    initial_ssh = np.where(surface_wet, -ssh_adjustment, 0.0)

    state = rest_state_latlon_cgrid_ocean(
        grid,
        z_coord,
        T_water_init_C=0.0,
        T_deep=0.0,
        S_uniform=0.0,
        H_max=float(np.max(bathymetry)),
        land_mask_override=tmask[..., 0],
        H_bathy_override=bathymetry,
        nemo_prognostic_barotropic_velocity=True,
    )
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(temperature, dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(salinity, dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(initial_ssh, dtype=jnp.float64)),
    )

    model_config = _model_config(
        barotropic_time_filter="nemo_ab3am4",
        n_barotropic_substeps=65,
        bbl_adv_option=0,
        bbl_gamma_s=0.0,
        bbl_diffusive_option=1,
        bbl_aht_m2_s=1000.0,
        whole_step_identity="orca2_vector_een_c2",
        tke_langmuir_evaluation="vectorized",
    )
    # zdfphy.F90:146-173 initializes the pre-closure coefficient memory
    # before zdf_tke_init.  This deck resolves nn_avb=0, nn_havtb=1, so avm_k
    # is constant rn_avm0=1.2e-4 while avt_k is rn_avt0=1.2e-5 times the
    # three literal latitude-band WHERE statements (:160-168).  Then
    # zdftke.F90:839-846 sees ln_zdfiwm=T and forces rn_emin=1e-10 before
    # tke_rst initializes en (:914-923).  Seed those carried card inputs here;
    # the shared step consumes the same state fields.
    lat_t = metric["gphit"]
    avtb_2d = np.ones(lat_t.shape, dtype=np.float64)
    select = (-15.0 <= lat_t) & (lat_t < -5.0)
    avtb_2d[select] = 1.0 - 0.09 * (lat_t[select] + 15.0)
    select = (-5.0 <= lat_t) & (lat_t < 5.0)
    avtb_2d[select] = 0.1
    select = (5.0 <= lat_t) & (lat_t < 15.0)
    avtb_2d[select] = 0.1 + 0.09 * (lat_t[select] - 5.0)
    wet_w = tmask[..., 1:30]
    avm_pre = np.float64(1.2e-4) * wet_w
    avt_pre = (avtb_2d[..., None] * np.float64(1.2e-5)) * wet_w
    en_pre = np.float64(1.0e-10) * wet_w
    dissl_pre = np.float64(1.0e-12) * wet_w
    state = state._replace(
        tke=Field(data=jnp.asarray(en_pre), name="tke",
                  dims=("lat", "lon", "level"), units="m^2/s^2"),
        tke_avm=Field(data=jnp.asarray(avm_pre), name="tke_avm",
                      dims=("lat", "lon", "level"), units="m^2/s"),
        tke_avt=Field(data=jnp.asarray(avt_pre), name="tke_avt",
                      dims=("lat", "lon", "level"), units="m^2/s"),
        tke_avm_surface=Field(
            data=jnp.asarray(np.float64(1.2e-4) * tmask[..., 0]),
            name="tke_avm_surface", dims=("lat", "lon"), units="m^2/s"),
        tke_dissl=Field(data=jnp.asarray(dissl_pre), name="tke_dissl",
                        dims=("lat", "lon", "level"), units="s^-1"),
    )
    tke_config = model_config.physics.vertical_mixing.tke._replace(
        tke_background=1.0e-10,
        # ORCA2 namelist_cfg:396 sets ln_zdfiwm=.TRUE., so zdf_tke_init takes
        # the FORCED arm: rn_emin=1e-10 AND rmxl_min=1e-3 (zdftke.F90:841-843).
        # The derived ln_zdfiwm=.FALSE. expression (:846) is never evaluated on
        # this card; with rn_emin=1e-10 it would return 1.0 m, a thousand times
        # NEMO's floor.
        nemo_derived_mxl_min=False,
        mxl_min=1.0e-3,
        # ORCA2 resolves ln_drg_OFF=.false.; tke_tke therefore applies the
        # bottom-friction Dirichlet value unconditionally at mbkt+1
        # (zdftke.F90:279-288).  nn_bc_bot is read for wave coupling but does
        # not guard this executed branch (declared/read only at :85/:762).
        bottom_tke_bc=True,
    )
    model_config = model_config._replace(
        physics=model_config.physics._replace(
            vertical_mixing=model_config.physics.vertical_mixing._replace(
                tke=tke_config)))
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=jnp.asarray(tmask[..., 0], dtype=jnp.float64),
        initial_state=state,
    )
    card = NEMOTestcaseCard(
        "ORCA2-zps",
        recipe,
        10800.0,
        2920,
        1,
        0,
        1,
        1000.0,
        0.0,
        surface_boundary_condition="ncar_core_sbcblk",
        surface_input_operator="nemo_fld_read",
        unmeasured_features=(
            "staged_gm_eiv",
            "linear_implicit_bottom_drag",
            "internal_wave_mixing",
            "spatial_lateral_viscosity",
            "freshwater_budget_carry",
            "si3_jpl5_layered_prather_state",
        ),
        icebergs_enabled=False,
        iceberg_inputs=(),
    )
    validate_nemo_testcase_card(card)
    return card


# ---------------------------------------------------------------------------
# VORTEX (tests/VORTEX) -- the PARENT grid only.
#
# cpp_VORTEX.fcm:1 compiles ``key_xios key_agrif key_qco key_vco_1d key_RK3``.
# The shipped case runs a 1:3 AGRIF zoom; legoESM has no nesting, so this card
# is the ROOT grid alone and the oracle build drops ``key_agrif``.  Every
# ``#if defined key_agrif`` block in the user routines is then absent and the
# compiled arm is the ``Agrif_Root()`` one, which is what is transcribed.
#
# namelist_cfg:19-24 (namusr_def): rn_dx = rn_dy = 30000 m, rn_dz = 500 m,
# rn_ppgphi0 = 38.5 deg, rn_ppumax = 1.0 m/s, nn_rot = 0.
_VORTEX_DX_M = 30000.0
_VORTEX_DY_M = 30000.0
_VORTEX_DZ_M = 500.0
_VORTEX_PPGPHI0_DEG = 38.5
_VORTEX_PPUMAX_M_S = 1.0
# usrdef_nam.F90:96-97,121: kpi = NINT(1800e3/rn_dx)+3, kpk = NINT(5000/rn_dz)+1.
_VORTEX_NI = 63
_VORTEX_NJ = 63
_VORTEX_NK = 11
_VORTEX_NLEV = _VORTEX_NK - 1        # jpkm1 wet levels; record 11 is the dummy
_VORTEX_H_M = 5000.0                 # usrdef_zgr.F90:126,187 flat bottom
# nameos rn_a0 (namelist_cfg:132).  eosbn2.F90:1890-1895 reads nameos
# unconditionally, so usrdef_istate.F90:19,88 consumes 0.28 whichever equation
# of state is selected.
_VORTEX_RN_A0 = 0.28
# namelist_cfg:130-140 (&nameos): ln_seos = .true., rn_a0 = 0.28 and EVERY
# other coefficient zero.  Decision 69 (operator note BG) makes VORTEX the one
# card that runs NEMO's simplified equation of state instead of the campaign's
# TEOS-10, because this eddy's temperature was DEFINED by inverting it
# (usrdef_istate.F90:83-88) and no other law reproduces its balance.
# eosbn2.F90:300-302 is the statement:
#     zn = - rn_a0*(1 + 0.5*rn_lambda1*zt + rn_mu1*zh)*zt
#          + rn_b0*(1 - 0.5*rn_lambda2*zs - rn_mu2*zh)*zs
#          - rn_nu*zt*zs
# with zt = T - rn_T0 and zs = S - rn_S0 (eosbn2.F90:295-296) and the
# unset rn_T0/rn_S0 taking eosbn2.F90:89-90's 10 degC / 35 PSU.  With
# rn_b0 = rn_lambda1 = rn_lambda2 = rn_mu1 = rn_mu2 = rn_nu = 0 this is
# rho = rho0 - 0.28*(T - 10): linear in temperature, salinity-blind, and
# depth-blind, so the depth argument's convention cannot matter here.
_VORTEX_SEOS = NemoSEOSConfig(
    rho0=float(NEMO_CONSTANTS_CONFIG.rho_0),   # phycst.F90 rho0, 1026 kg/m3
    a0=_VORTEX_RN_A0,                          # namelist_cfg:132 rn_a0
    b0=0.0,                                    # namelist_cfg:133 rn_b0
    lambda1=0.0,                               # namelist_cfg:134 rn_lambda1
    lambda2=0.0,                               # namelist_cfg:135 rn_lambda2
    mu1=0.0,                                   # namelist_cfg:136 rn_mu1
    mu2=0.0,                                   # namelist_cfg:137 rn_mu2
    nu=0.0,                                    # namelist_cfg:138 rn_nu
    T0=10.0,                                   # eosbn2.F90:89  rn_T0 (unset)
    S0=35.0,                                   # eosbn2.F90:90  rn_S0 (unset)
)


def vortex_horizontal_coordinates() -> dict[str, np.ndarray]:
    """Transcribe ``usrdef_hgr.F90`` for ``nn_rot = 0`` (root grid).

    Positions are the source's own KILOMETRE coordinates: VORTEX uses
    ``glam``/``gphi`` as a Cartesian offset from the domain centre, and
    ``usrdef_istate`` multiplies them back by 1.e3.  Operand order is kept so
    the initial state reproduces the Fortran bit for bit.
    """

    rad = math.pi / 180.0                            # phycst.F90:26
    omega = float(NEMO_CONSTANTS_CONFIG.Omega)       # phycst.F90:91
    radius = float(NEMO_CONSTANTS_CONFIG.R_earth)    # phycst.F90:37
    # usrdef_hgr.F90:83-84
    offset_x = (-float(_VORTEX_NI - 1) + 1.0) * 0.5 * 1.e-3 * _VORTEX_DX_M
    offset_y = (-float(_VORTEX_NJ - 1) + 1.0) * 0.5 * 1.e-3 * _VORTEX_DY_M
    # usrdef_hgr.F90:103-104 -- mig/mjg minus one, the 0-based global index
    zti = np.arange(_VORTEX_NI, dtype=np.float64)
    ztj = np.arange(_VORTEX_NJ, dtype=np.float64)
    # usrdef_hgr.F90:108-111 and :130-133 (the nn_rot==0 arm).  glamv = glamt,
    # glamf = glamu, gphiu = gphit, gphif = gphiv.
    lam_t = offset_x + _VORTEX_DX_M * 1.e-3 * (zti - 0.5)
    lam_u = lam_t + _VORTEX_DX_M * 1.e-3 * 0.5
    phi_t = offset_y + _VORTEX_DY_M * 1.e-3 * (ztj - 0.5)
    phi_v = phi_t + _VORTEX_DY_M * 1.e-3 * 0.5
    shape = (_VORTEX_NJ, _VORTEX_NI)
    glamt = np.broadcast_to(lam_t[None, :], shape).copy()
    glamu = np.broadcast_to(lam_u[None, :], shape).copy()
    gphit = np.broadcast_to(phi_t[:, None], shape).copy()
    gphiv = np.broadcast_to(phi_v[:, None], shape).copy()
    # usrdef_hgr.F90:174-177 -- the beta-plane Coriolis.  The km position is
    # scaled back to metres INSIDE the product, exactly as the source writes it.
    beta = 2.0 * omega * math.cos(rad * _VORTEX_PPGPHI0_DEG) / radius
    f0 = 2.0 * omega * math.sin(rad * _VORTEX_PPGPHI0_DEG)
    return {
        "glamt": glamt, "glamu": glamu, "glamv": glamt.copy(),
        "glamf": glamu.copy(),
        "gphit": gphit, "gphiu": gphit.copy(), "gphiv": gphiv,
        "gphif": gphiv.copy(),
        "ff_t": f0 + beta * gphit * 1.e+3,
        "ff_f": f0 + beta * gphiv * 1.e+3,
        "f0": f0, "beta": beta,
    }


def _vortex_grid(source: dict[str, np.ndarray]):
    """Uniform 30 km Cartesian beta-plane with NEMO's own Coriolis operands.

    ``create_beta_plane_cgrid_geometry`` already evaluates ``f0 + beta*y`` at
    each stagger from its own ``y``; the origins below align those ``y`` with
    ``gphit``/``gphiv``.  The f arrays are nevertheless replaced by the
    transcribed ones so the card carries the source's ``(beta*phi_km)*1e3``
    operand order rather than a re-derivation in metres.
    """

    grid = create_beta_plane_cgrid_geometry(
        _VORTEX_NJ, _VORTEX_NI,
        dx_m=_VORTEX_DX_M, dy_m=_VORTEX_DY_M,
        f0=source["f0"], beta=source["beta"],
        # y_c[j] = origin + (j+1/2)dy must equal gphit[j]*1e3.
        x_origin_m=float(source["glamt"][0, 0]) * 1.e3 - 0.5 * _VORTEX_DX_M,
        y_origin_m=float(source["gphit"][0, 0]) * 1.e3 - 0.5 * _VORTEX_DY_M,
        radius=float(NEMO_CONSTANTS_CONFIG.R_earth),
        cartesian_pseudo_lat=True,
        dtype=jnp.float64,
    )
    ff_t = source["ff_t"]
    ff_f = source["ff_f"]
    # The model's face arrays carry one redundant west/south record.  f here
    # depends on j alone, so the U array is a pure broadcast and the extra V
    # row is the affine continuation one cell south of gphiv[0].
    f_u = np.ascontiguousarray(
        np.broadcast_to(ff_t[:, :1], (_VORTEX_NJ, _VORTEX_NI + 1)))
    f_v = np.concatenate(
        [ff_f[:1] - (ff_f[1:2] - ff_f[:1]), ff_f], axis=0)
    return grid._replace(
        f_T=jnp.asarray(ff_t, dtype=jnp.float64),
        f_u=jnp.asarray(f_u, dtype=jnp.float64),
        f_v=jnp.asarray(f_v, dtype=jnp.float64),
        ff_f=jnp.asarray(ff_f, dtype=jnp.float64),
    )


def _vortex_analytic_scalars() -> dict[str, float]:
    """The scalars shared by ``usr_def_istate`` and ``usr_def_istate_ssh``."""

    omega = float(NEMO_CONSTANTS_CONFIG.Omega)
    rad = math.pi / 180.0
    # usrdef_istate.F90:69-75 (identical at :169-174 in the ssh routine)
    f0 = 2.0 * omega * math.sin(rad * _VORTEX_PPGPHI0_DEG)
    umax = _VORTEX_PPUMAX_M_S * math.copysign(1.0, f0)
    lam = math.sqrt(2.0) * 60.e3
    n2 = 3.e-3 ** 2
    height = 0.5 * 5000.0
    p0 = (float(NEMO_CONSTANTS_CONFIG.rho_0) * f0 * umax * lam
          * math.sqrt(math.exp(1.0) / 2.0))
    return {"f0": f0, "lam": lam, "n2": n2, "H": height, "P0": p0}


def vortex_initial_state_fields(source: dict[str, np.ndarray], tmask):
    """Transcribe ``usrdef_istate.F90`` in NEMO's own execution order.

    THE ORDER IS THE SOURCE'S: ``rst_read_ssh`` (restart.F90:461) calls
    ``usr_def_istate_ssh`` FIRST, ``dom_qco_zgr`` (domqco.F90:124) then builds
    ``r3t`` from that ssh, and only afterwards does ``istate.F90:127-130`` hand
    ``gdept(:,:,:,Kbb)`` -- the LIVE, ssh-stretched depth
    (domzgr_substitute.h90:139) -- to ``usr_def_istate``.  T, u and v are
    therefore functions of the stretched depth, not of ``gdept_1d``.

    Returns ``(ssh, T, S, u, v)`` on NEMO's own T/U/V index convention.
    """

    scalars = _vortex_analytic_scalars()
    grav = float(NEMO_CONSTANTS_CONFIG.g)
    rho0 = float(NEMO_CONSTANTS_CONFIG.rho_0)
    p0, lam, height = scalars["P0"], scalars["lam"], scalars["H"]
    tmask = np.asarray(tmask, dtype=np.float64)
    surface = tmask[:, :, 0]

    def gaussian(lam_km, phi_km):
        zx = np.asarray(lam_km) * 1.e3
        zy = np.asarray(phi_km) * 1.e3
        return np.exp(-(zx ** 2 + zy ** 2) / lam ** 2), zx, zy

    # --- usr_def_istate_ssh, usrdef_istate.F90:177-183 ---------------------
    bell_t, _, _ = gaussian(source["glamt"], source["gphit"])
    a_ssh = (-p0 * (1.0 - math.exp(-height))
             / (grav * (height - 1.0 + math.exp(-height))))
    rho_ssh = rho0 + a_ssh * bell_t
    ssh = p0 * bell_t / (rho_ssh * grav) * surface

    # --- domqco.F90:160 with domain.F90:158's reciprocal ------------------
    r1_ht_0 = surface / (_VORTEX_H_M + 1.0 - surface)
    r3t = ssh * r1_ht_0
    # domzgr_substitute.h90:139 -- gdept(Kbb) = gdept_0 * (1 + r3t(Kbb))
    gdept_1d = (np.arange(_VORTEX_NLEV, dtype=np.float64) + 0.5) * _VORTEX_DZ_M
    gdept = gdept_1d[None, None, :] * (1.0 + r3t[:, :, None])

    # --- temperature, usrdef_istate.F90:78-90 -----------------------------
    rho1 = rho0 * (1.0 + scalars["n2"] * gdept / grav)
    # EXP(zdt-zH) is evaluated only on the zdt < zH arm; clipping the argument
    # keeps the taken arm bit-identical and stops the deep levels overflowing.
    anomaly = (p0 * (1.0 - np.exp(np.minimum(gdept - height, 0.0)))
               * bell_t[:, :, None]
               / (grav * (height - 1.0 + math.exp(-height))))
    rho1 = np.where(gdept < height, rho1 - anomaly, rho1)
    temperature = (20.0 + (rho0 - rho1) / _VORTEX_RN_A0) * tmask
    salinity = 35.0 * tmask                       # usrdef_istate.F90:93

    # --- velocities, usrdef_istate.F90:96-139 -----------------------------
    a_vel = 2.0 * p0 / (scalars["f0"] * rho0 * lam ** 2)
    bell_u, _, zy_u = gaussian(source["glamu"], source["gphiu"])
    bell_v, zx_v, _ = gaussian(source["glamv"], source["gphiv"])

    def profile(depth):
        return ((height - 1.0 - depth
                 + np.exp(np.minimum(depth - height, 0.0)))
                / (height - 1.0 + math.exp(-height)))

    # ji+1 / jj+1 reach the closed land ring, whose tmask is zero; NEMO's own
    # DO_2D(0,0,0,0) plus the lbc_lnk at :141 leaves the same zeros there.
    east = np.concatenate([tmask[:, 1:], np.zeros_like(tmask[:, :1])], axis=1)
    north = np.concatenate([tmask[1:], np.zeros_like(tmask[:1])], axis=0)
    depth_u = 0.5 * (gdept + np.concatenate(
        [gdept[:, 1:], np.zeros_like(gdept[:, :1])], axis=1))
    depth_v = 0.5 * (gdept + np.concatenate(
        [gdept[1:], np.zeros_like(gdept[:1])], axis=0))
    u = np.where(
        depth_u < height,
        a_vel * profile(depth_u) * zy_u[:, :, None] * bell_u[:, :, None],
        0.0) * tmask * east
    v = np.where(
        depth_v < height,
        -(a_vel * profile(depth_v) * zx_v[:, :, None] * bell_v[:, :, None]),
        0.0) * tmask * north
    return ssh, temperature, salinity, u, v


def _vortex_barotropic_velocity(ssh, u, v, tmask):
    """Transcribe ``istate.F90:149-154`` (the RK3 ``Kbb`` arm).

    ``e3u(Kbb)`` and ``r1_hu(Kbb)`` are the key_qco macros
    (domzgr_substitute.h90:127,136), so the ``(1+r3u)`` factor is applied in
    the accumulation and removed in the divisor as two separate statements --
    it does not cancel in the bits.
    """

    tmask = np.asarray(tmask, dtype=np.float64)
    umask = tmask * np.concatenate(
        [tmask[:, 1:], np.zeros_like(tmask[:, :1])], axis=1)
    vmask = tmask * np.concatenate(
        [tmask[1:], np.zeros_like(tmask[:1])], axis=0)
    su = umask[:, :, 0]
    sv = vmask[:, :, 0]
    # domain.F90:159 -- r1_h*_0 = mask / (h*_0 + 1 - mask), h*_0 = 5000*mask.
    r1_hu_0 = su / (_VORTEX_H_M * su + 1.0 - su)
    r1_hv_0 = sv / (_VORTEX_H_M * sv + 1.0 - sv)
    # domqco.F90:166-169 with this mesh's uniform e1e2t (r1_e1e2u = 1/e1e2u).
    area = _VORTEX_DX_M * _VORTEX_DY_M
    ssh_east = np.concatenate([ssh[:, 1:], np.zeros_like(ssh[:, :1])], axis=1)
    ssh_north = np.concatenate([ssh[1:], np.zeros_like(ssh[:1])], axis=0)
    r3u = 0.5 * (area * ssh + area * ssh_east) * r1_hu_0 * (1.0 / area)
    r3v = 0.5 * (area * ssh + area * ssh_north) * r1_hv_0 * (1.0 / area)
    e3u = _VORTEX_DZ_M * (1.0 + r3u)
    e3v = _VORTEX_DZ_M * (1.0 + r3v)
    uu_b = np.zeros_like(su)
    vv_b = np.zeros_like(sv)
    for k in range(_VORTEX_NLEV):                  # DO_3D ... 1, jpkm1
        uu_b = uu_b + e3u * u[:, :, k] * umask[:, :, k]
        vv_b = vv_b + e3v * v[:, :, k] * vmask[:, :, k]
    return uu_b * (r1_hu_0 / (1.0 + r3u)), vv_b * (r1_hv_0 / (1.0 + r3v))


# Round 1 declared two gaps here, both of them the SAME missing operator seen
# from the two sides NEMO runs it on.  Round 2 transcribed it
# (vorticity_scheme="een_planetary" plus the matching barotropic arm), so the
# tuple is EMPTY and the card no longer fails its execution gate.  It is kept
# as a named symbol because the card validator checks the card against it: a
# future gap is declared by adding to this tuple, never by dropping the check.
VORTEX_UNMEASURED: tuple[str, ...] = ()


def build_vortex_zco_card(momentum: str = "flux") -> NEMOTestcaseCard:
    """VORTEX root grid: 63x63x10 beta-plane box, flat 5000 m zco bottom.

    ``momentum`` picks which of the two VORTEX decks this card is (decision
    73).  ``"flux"`` is the shipped deck -- flux-form third-order upstream
    momentum, where the energy-and-enstrophy triad degenerates to a Coriolis
    operator.  ``"vector"`` is the ORCA2/GYRE momentum scheme set, where the
    same triad runs on the live relative vorticity and the kinetic-energy
    gradient and vertical momentum advection join the step.  Everything else
    -- geometry, initial state, equation of state, tracer program, free
    surface, vertical physics -- is shared, which is what makes the pair a
    controlled comparison.
    """
    if momentum not in ("flux", "vector"):
        raise ValueError(
            f"unknown VORTEX momentum deck {momentum!r}; expected 'flux' "
            "(ln_dynadv_up3) or 'vector' (ln_dynadv_vec)")

    source = vortex_horizontal_coordinates()
    grid = _vortex_grid(source)
    wet = _closed_box_mask(_VORTEX_NJ, _VORTEX_NI)
    wet_np = np.asarray(wet)
    bathymetry = wet * _VORTEX_H_M
    native_3d = (_VORTEX_NJ, _VORTEX_NI, _VORTEX_NLEV)
    tmask = np.broadcast_to(wet_np[..., None], native_3d)
    thickness = np.full(native_3d, _VORTEX_DZ_M)
    gdept_1d = (np.arange(_VORTEX_NLEV, dtype=np.float64) + 0.5) * _VORTEX_DZ_M
    gdepw_1d = np.arange(_VORTEX_NLEV, dtype=np.float64) * _VORTEX_DZ_M
    u_wet = wet_np * np.roll(wet_np, -1, axis=1)
    u_wet[:, -1] = 0.0
    v_wet = wet_np * np.roll(wet_np, -1, axis=0)
    v_wet[-1, :] = 0.0
    fe3mask = np.asarray(nemo_fe3mask_from_tmask(jnp.asarray(tmask)))
    # namelist_cfg:99 rn_shlat = 0 (free slip), so dommsk.F90 leaves fmask
    # equal to the four-T-cell product it copied into fe3mask.
    fmask = fe3mask
    umask_3d = np.broadcast_to(u_wet[..., None], native_3d)
    vmask_3d = np.broadcast_to(v_wet[..., None], native_3d)
    area = np.full((_VORTEX_NJ, _VORTEX_NI), _VORTEX_DX_M * _VORTEX_DY_M)
    metric = np.full((_VORTEX_NJ, _VORTEX_NI), _VORTEX_DX_M)
    operands = NemoEENBarotropicOperands(
        ff_f=np.asarray(grid.ff_f),
        e3u_0=thickness, e3v_0=thickness, e3f_0=thickness,
        umask=umask_3d, vmask=vmask_3d,
        fmask=fmask, fe3mask=fe3mask,
        hu_0=_VORTEX_H_M * u_wet,
        hv_0=_VORTEX_H_M * v_wet,
        # domain.F90:149-152 builds the F-column depth from the V masks.
        hf_0=np.sum(
            thickness * vmask_3d * np.roll(vmask_3d, -1, axis=1), axis=-1),
        e1t=metric, e2t=metric, e1u=metric, e2u=metric,
        e1v=metric, e2v=metric, e1f=metric, e2f=metric,
    )
    z_ref = create_z_star_from_thicknesses(
        jnp.full((_VORTEX_NLEV,), _VORTEX_DZ_M),
        t_depth_ref_m=gdept_1d,
        nemo_gdept_0_m=np.broadcast_to(gdept_1d, native_3d),
        nemo_gdepw_0_m=np.broadcast_to(gdepw_1d, native_3d),
        nemo_e3t_0_m=thickness,
        # usrdef_zgr.F90:140-152: the depth<->e3 round trip returns a uniform
        # ladder, so e3w_1d is exactly rn_dz on every record including the
        # first (e3w_1d(1) = 2*(dept(1)-depw(1)) = rn_dz).
        nemo_e3w_0_m=thickness,
        nemo_hu_0_m=_VORTEX_H_M * u_wet,
        nemo_hv_0_m=_VORTEX_H_M * v_wet,
        nemo_e1e2t_m=area, nemo_e1e2u_m=area, nemo_e1e2v_m=area,
        nemo_e2u_m=metric, nemo_e1v_m=metric,
        nemo_een_barotropic_m=operands,
    )
    bottom = jnp.where(wet > 0.0, _VORTEX_NLEV - 1, -1)
    z_coord = create_full_step_coordinate(z_ref, bottom)
    ssh, temperature, salinity, u, v = vortex_initial_state_fields(
        source, tmask)
    uu_b, vv_b = _vortex_barotropic_velocity(ssh, u, v, tmask)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=0.0, T_deep=0.0, S_uniform=0.0,
        H_max=_VORTEX_H_M,
        land_mask_override=wet,
        H_bathy_override=bathymetry,
        nemo_prognostic_barotropic_velocity=True,
    )
    # The model's face arrays carry one redundant west/south record; the
    # phase-2 geometry mapping is model[:, 1:] == NEMO's own U column.
    zeros_u = np.zeros((_VORTEX_NJ, 1, _VORTEX_NLEV))
    zeros_v = np.zeros((1, _VORTEX_NI, _VORTEX_NLEV))
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(temperature, dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(salinity, dtype=jnp.float64)),
        u=state.u.replace(data=jnp.asarray(
            np.concatenate([zeros_u, u], axis=1), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(
            np.concatenate([zeros_v, v], axis=0), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(ssh, dtype=jnp.float64)),
        uu_b=state.uu_b.replace(data=jnp.asarray(
            np.concatenate([np.zeros((_VORTEX_NJ, 1)), uu_b], axis=1),
            dtype=jnp.float64)),
        vv_b=state.vv_b.replace(data=jnp.asarray(
            np.concatenate([np.zeros((1, _VORTEX_NI)), vv_b], axis=0),
            dtype=jnp.float64)),
    )
    # namelist_cfg:204-216: ln_bt_fw=T, nn_bt_flt=3, rn_bt_alpha=.07 and
    # ln_bt_auto=.false. with nn_e=48, so the substep count is PINNED by the
    # namelist -- it is not the resolved auto value the tanks use.
    model_config = _model_config(
        barotropic_time_filter="nemo_ab3am4",
        n_barotropic_substeps=48,
        bbl_adv_option=0, bbl_gamma_s=0.0,
        bbl_diffusive_option=0, bbl_aht_m2_s=0.0,
        whole_step_identity=("vortex_flux_up3_een" if momentum == "flux"
                             else "vortex_vector_een_c2"),
        tke_langmuir_evaluation=None,
    )._replace(
        wzv_call2_evaluation=("nemo_literal" if momentum == "vector"
                              else "generic"),
        nemo_stage_momentum_wzv_split=(True if momentum == "vector"
                                       else None))
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=wet,
        initial_state=state,
    )
    # namelist_cfg:32,41: nn_itend = 3000 at rn_Dt = 2880 s.
    card = NEMOTestcaseCard(
        "VORTEX-zco" if momentum == "flux" else "VORTEX_VEC-zco",
        recipe, 2880.0, 3000, 1, 0, 0, 0.0, 0.0,
        unmeasured_features=VORTEX_UNMEASURED,
    )
    validate_nemo_testcase_card(card)
    return card


def validate_nemo_testcase_card(card: NEMOTestcaseCard) -> None:
    """Reject any card composition not exercised by its named oracle run."""
    if card.transcendentals != "libm":
        raise ValueError(
            f"{card.case} requires scalar-libm certification transcendentals, "
            f"got {card.transcendentals!r}"
        )
    expected = {
        "LOCK_EXCHANGE-zco": ("nemo_ab3am4", 1, 0, 0.0, 0, 0.0),
        "OVERFLOW-zps": ("nemo_boxcar1_ab3", 3, 2, 20.0, 0, 1000.0),
        "GYRE-zco": ("nemo_ab3am4", 50, 0, 0.0, 0, 0.0),
        "ORCA2-zps": ("nemo_ab3am4", 65, 0, 0.0, 1, 1000.0),
        "VORTEX-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        # The vector-EEN deck differs from the flux deck only in the momentum
        # scheme set, so every row this table checks is the same row.
        "VORTEX_VEC-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
    }
    if card.case not in expected:
        raise ValueError(f"unknown NEMO testcase card {card.case!r}")
    cfg = card.recipe.model_config
    if (card.recipe.initial_state.uu_b is None
            or card.recipe.initial_state.vv_b is None):
        raise ValueError(
            f"{card.case} requires NEMO's prognostic uu_b/vv_b state pair")
    filt, count, bbl_option, gamma, diffusive, aht = expected[card.case]
    actual = (
        cfg.barotropic.barotropic_time_filter,
        cfg.barotropic.n_barotropic_substeps,
        cfg.bbl_adv_option,
        cfg.bbl_gamma_s,
        cfg.bbl_diffusive_option,
        cfg.bbl_aht_m2_s,
    )
    if actual != (filt, count, bbl_option, gamma, diffusive, aht):
        raise ValueError(
            f"{card.case} filter/substep/BBL composition {actual!r} does not "
            f"match the executed oracle "
            f"{(filt, count, bbl_option, gamma, diffusive, aht)!r}")
    if ((card.bbl_adv_option, card.bbl_gamma_s,
         card.bbl_diffusive_option, card.bbl_aht_m2_s)
            != (bbl_option, gamma, diffusive, aht)):
        raise ValueError(f"{card.case} card BBL metadata disagrees with config")
    required = {
        "barotropic_face_depth": "nemo_ssh_avg",
        "barotropic_continuity_evaluation": "nemo_literal",
        "barotropic_transport_accumulation_evaluation": "nemo_literal",
        "barotropic_seed_face_depth": "nemo_ssh_avg",
        "barotropic_seed_evaluation": "nemo_literal",
        "barotropic_pgf_evaluation": "nemo_literal",
        # dynspg_ts.F90:484-500 reads the CARRIED uu_b/vv_b at the window
        # seed; this card must select that identity by config, not inherit it
        # from whether the state happens to hold the pair.
        "nemo_prognostic_barotropic_state": True,
    }
    for field, value in required.items():
        got = getattr(cfg.barotropic, field)
        if got != value:
            raise ValueError(
                f"{card.case} requires {field}={value!r}, got {got!r}")
    if cfg.barotropic.barotropic_diffusion_alpha != 0.0:
        raise ValueError(
            f"{card.case} forbids unmatched live eta diffusion")
    # All three testcase cards run NEMO's hpg_sco (dynhpg.F90:305-393) with its
    # e3w(Kmm) trapezoid.  The MODEL-level guard can no longer carry this: the
    # trapezoid allow-list was widened to {"nemo_sco", "adcroft"} because
    # hpg_zco (dynhpg.F90:270-296) accumulates the same recurrence, so a card
    # silently swapped to "adcroft" now builds.  Pin it here, on the card,
    # which is what owns the certified identity.
    if cfg.pgf_scheme != "nemo_sco":
        raise ValueError(
            f"{card.case} requires NEMO's hpg_sco pressure gradient "
            f"(pgf_scheme='nemo_sco'), got {cfg.pgf_scheme!r}")
    if cfg.zdf_implicit_solver_evaluation != "nemo_literal":
        raise ValueError(
            f"{card.case} requires the NEMO literal implicit-ZDF program "
            "(which carries NEMO's e3w(Kmm) gradient divisor)")
    if not cfg.tracer_wall_neumann_fill:
        raise ValueError(
            f"{card.case} requires NEMO's closed-wall tracer halo fill")
    if cfg.barotropic.barotropic_reconcile_target != "velocity_avg":
        raise ValueError(
            f"{card.case} requires NEMO's prognostic uu_b(Kaa) velocity frame")
    wet = np.asarray(card.recipe.land_mask) > 0.5
    if card.case in ("GYRE-zco", "ORCA2-zps"):
        if (cfg.momentum_advection != "vector_invariant"
                or cfg.ke_gradient_scheme != "c2"):
            raise ValueError(
                f"{card.case} requires ln_dynadv_vec=.true. with nn_dynkeg=0 "
                "(momentum_advection='vector_invariant', "
                f"ke_gradient_scheme='c2'); got "
                f"{cfg.momentum_advection!r}, {cfg.ke_gradient_scheme!r}")
        tke = cfg.physics.vertical_mixing.tke
        expected_langmuir = (
            "nemo_literal" if card.case == "GYRE-zco" else "vectorized"
        )
        if tke.tke_langmuir_evaluation != expected_langmuir:
            raise ValueError(
                f"{card.case} requires tke_langmuir_evaluation="
                f"{expected_langmuir!r}, got "
                f"{tke.tke_langmuir_evaluation!r}")
        expected_baro_coriolis = (
            "ene_metric" if card.case == "GYRE-zco" else "een_metric"
        )
        if (cfg.barotropic_coriolis_split != "live"
                or cfg.barotropic.barotropic_coriolis != expected_baro_coriolis
                or cfg.barotropic.barotropic_een_coefficient_evaluation
                != "nemo_literal"):
            raise ValueError(
                f"{card.case} requires its live literal E[N]E barotropic "
                "composition")
        if card.case == "GYRE-zco":
            require_rotation = (
                np.any(np.asarray(card.recipe.grid.f_T) != 0.0)
                and np.count_nonzero(np.any(wet, axis=1)) == 20
                and np.all(np.asarray(card.recipe.grid.sin_alpha_u) != 0.0)
            )
            if not require_rotation:
                raise ValueError("GYRE-zco requires a live rotated beta-plane")
            if card.surface_boundary_condition != "gyre_usrdef_sbc":
                raise ValueError("GYRE-zco requires the analytic usrdef_sbc card")
        else:
            if cfg.eos != "nemo_eos80" or cfg.vorticity_scheme != "een_total":
                raise ValueError("ORCA2-zps requires EOS-80 and EEN vorticity")
            sh2_tuple = (
                tke.tke_shear_production,
                tke.tke_shear_avm_weighting,
                tke.tke_shear_evaluation_stage,
                tke.tke_shear_metric_source,
            )
            expected_sh2 = (
                "nemo_face_native_nbb2", "nemo_face", "step_entry",
                "nemo_qco_live_face",
            )
            if sh2_tuple != expected_sh2:
                raise ValueError(
                    "ORCA2-zps requires the RK3 zdf_sh2 Nbb*Nbb face-native "
                    f"selector tuple {expected_sh2!r}, got {sh2_tuple!r}")
            if not tke.bottom_tke_bc:
                raise ValueError(
                    "ORCA2-zps requires the executed zdftke bottom-friction "
                    "Dirichlet boundary (ln_drg_OFF=.false.)")
            if int(tke.eice) != 1:
                raise ValueError(
                    "ORCA2-zps requires NEMO nn_eice=1 ice attenuation "
                    "(zdftke.F90:255 tanh(10*fr_i))")
            if card.surface_boundary_condition != "ncar_core_sbcblk":
                raise ValueError("ORCA2-zps requires the NCAR/CORE sbcblk card")
            if card.surface_input_operator != "nemo_fld_read":
                raise ValueError("ORCA2-zps requires the shared NEMO fld_read map")
            if card.icebergs_enabled is not False or card.iceberg_inputs != ():
                raise ValueError(
                    "ORCA2-zps comparison card requires ln_icebergs=F and no "
                    "iceberg inputs"
                )
            if not card.unmeasured_features:
                raise ValueError(
                    "ORCA2-zps must fail-closed while selected mechanisms "
                    "remain unmeasured"
                )
        return
    if card.case in ("VORTEX-zco", "VORTEX_VEC-zco"):
        vector = card.case == "VORTEX_VEC-zco"
        # VORTEX is the first card on this identity with a LIVE rotation
        # operator, so the structural-elimination escape below must not be
        # reachable for it.  Round 1 declared the operator as a gap; round 2
        # transcribed it, so what is checked here is that the card selects
        # NEMO's energy-and-enstrophy triad rather than legoESM's 4-point
        # C-grid average, on both the baroclinic and the barotropic arm.
        if card.unmeasured_features != VORTEX_UNMEASURED:
            raise ValueError(
                "VORTEX-zco's declared gaps must match VORTEX_UNMEASURED "
                "exactly; a card that declares a different set has not been "
                "checked against what this validator proves")
        # The two decks differ in exactly what the vorticity routine is
        # handed, which is decided by the momentum advection form, not by the
        # vorticity namelist: both select ln_dynvor_een (namelist_cfg:193).
        # dynvor.f90:855-868 reads the resolved advection form and sets the
        # total vorticity to Coriolis plus the metric term under flux form and
        # to Coriolis plus the RELATIVE vorticity under vector form.
        if vector:
            if cfg.vorticity_scheme != "een_total":
                raise ValueError(
                    "VORTEX_VEC-zco requires vorticity_scheme='een_total': "
                    "namelist_cfg:182 selects vector form, so dynvor.f90:"
                    "861-864 sets ntot = np_CRV and the triad runs on the "
                    "relative vorticity as well as the planetary one")
            if cfg.momentum_advection != "vector_invariant":
                raise ValueError(
                    "VORTEX_VEC-zco requires vector-invariant momentum "
                    "(namelist_cfg:182 ln_dynadv_vec)")
            if cfg.ke_gradient_scheme != "c2":
                raise ValueError(
                    "VORTEX_VEC-zco requires the mean-of-squares kinetic "
                    "energy gradient (namelist_cfg:183 nn_dynkeg = 0), not "
                    "the Hollingsworth correction")
            if cfg.vertical_momentum_scheme != "nemo_advective":
                raise ValueError(
                    "VORTEX_VEC-zco runs dyn_zad for the vertical advection "
                    "of momentum (dynadv.f90:138); the flux-form vertical UP3 "
                    "flux is not called on this deck")
            if cfg.momentum_flux_scheme != "upwind":
                raise ValueError(
                    "VORTEX_VEC-zco must not leave the OTHER card's flux-form "
                    "momentum scheme selected; dyn_adv never calls one on a "
                    "vector-form deck, so this carries the same inert value "
                    "the GYRE and ORCA2 vector cards carry")
        elif cfg.vorticity_scheme != "een_planetary":
            raise ValueError(
                "VORTEX-zco requires vorticity_scheme='een_planetary': "
                "namelist_cfg:193 selects ln_dynvor_een and :182 selects "
                "flux form, so dynvor.F90:874 plus dyn_vor_init:891-893 run "
                "vor_een on np_CME, whose metric term vanishes on this "
                "Cartesian mesh. Any other scheme substitutes a different "
                "Coriolis operator for NEMO's")
        if not vector and cfg.momentum_advection != "flux_form":
            raise ValueError(
                "VORTEX-zco requires flux-form momentum "
                "(namelist_cfg:185 ln_dynadv_up3)")
        if cfg.coriolis_scheme != "explicit_ab2":
            raise ValueError(
                "VORTEX-zco carries f inside the triad, so the Matsuno "
                "rotation must be off (coriolis_scheme='explicit_ab2')")
        if (cfg.barotropic.barotropic_coriolis != "een_metric"
                or cfg.barotropic_coriolis_split != "live"):
            raise ValueError(
                "VORTEX-zco requires the matching barotropic EEN Coriolis "
                "(dynspg_ts.F90:1326-1345) and the live depth-mean split")
        # The np_CME metric term is dropped only because it is bitwise zero
        # here; prove that against the card's OWN mesh, not against the
        # namelist's promise (usrdef_hgr.F90:160-163).
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            assert_een_planetary_metric_term_vanishes,
        )
        if not vector:
            # The np_CME metric term is DROPPED on the flux card only because
            # this mesh makes it bitwise zero.  Under vector form NEMO never
            # forms it at all (dynvor.f90:861-864), so the proof belongs to
            # the flux card; running it here would assert something the
            # vector card does not rely on.
            assert_een_planetary_metric_term_vanishes(card.recipe.grid)
        if not np.any(np.asarray(card.recipe.grid.ff_f) != 0.0):
            raise ValueError(
                f"{card.case} requires a live beta-plane Coriolis at F "
                "points")
        if np.count_nonzero(np.any(wet, axis=1)) != 61:
            raise ValueError(
                "VORTEX-zco requires the 61 wet rows of its closed 63x63 box")
        if card.recipe.z_coord.nemo_een_barotropic is None:
            raise ValueError(
                "VORTEX-zco requires NEMO's frozen barotropic EEN operands")
        # namelist_cfg:77 ln_usr=.true. with usrdef_sbc.F90:60-68 writing
        # zeros; namelist_cfg:114 ln_drg_OFF; :158 ln_traldf_OFF;
        # :222 ln_dynldf_OFF.
        if card.surface_boundary_condition != "none":
            raise ValueError(
                "VORTEX-zco has no surface forcing (usrdef_sbc writes zeros)")
        if (cfg.lateral_viscosity.A_h, cfg.lateral_viscosity.B_h, cfg.K_h,
                cfg.bottom_drag.bottom_drag_r) != (0.0, 0.0, 0.0, 0.0):
            raise ValueError(
                "VORTEX-zco requires ln_traldf_OFF, ln_dynldf_OFF and "
                "ln_drg_OFF (no lateral diffusion, no bottom drag)")
        if (cfg.A_v, cfg.K_v) != (1.0e-4, 0.0):
            raise ValueError(
                "VORTEX-zco requires rn_avm0=1.0e-4 and rn_avt0=0.0")
        if cfg.adaptive_implicit_vertadv:
            raise ValueError(
                "VORTEX-zco leaves ln_zad_Aimp at its .false. default; the "
                "Courant-dependent implicit vertical advection must be OFF")
        # namelist_cfg:130-138 (&nameos), decision 69.  The eddy's temperature
        # is DEFINED by inverting this law (usrdef_istate.F90:83-88), so a card
        # on any other equation of state is not this experiment.  The
        # coefficients are checked field by field because the shared defaults
        # are DINO's and every one of them differs.
        if cfg.eos != "nemo_seos" or cfg.eos_nemo_seos != _VORTEX_SEOS:
            raise ValueError(
                "VORTEX-zco requires NEMO's simplified equation of state with "
                "its own &nameos coefficients (ln_seos, rn_a0=0.28 and every "
                "other coefficient zero); the shared NemoSEOSConfig defaults "
                "are DINO's and would silently run a different fluid")
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


def validate_nemo_testcase_card_for_execution(card: NEMOTestcaseCard) -> None:
    """Reject a structurally valid card until every selected arm is measured."""
    validate_nemo_testcase_card(card)
    if card.unmeasured_features:
        unresolved = ", ".join(card.unmeasured_features)
        raise ValueError(
            f"{card.case} is not execution-ready; unresolved selected arms: "
            f"{unresolved}"
        )


def build_nemo_testcase_card(
    case: str, *, deck_root: str | Path | None = None
) -> NEMOTestcaseCard:
    """Fail-closed dispatch for the certified phase-2 cards."""

    builders = {
        "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card,
        "OVERFLOW-zps": build_overflow_zps_card,
        "GYRE-zco": build_gyre_zco_card,
        "VORTEX-zco": build_vortex_zco_card,
        "VORTEX_VEC-zco": lambda: build_vortex_zco_card("vector"),
    }
    if case == "ORCA2-zps":
        if deck_root is None:
            raise ValueError("ORCA2-zps requires an explicit deck_root")
        return build_orca2_zps_card(deck_root)
    if case not in builders:
        raise ValueError(
            f"unknown NEMO testcase {case!r}; expected one of {sorted(builders)}"
        )
    return builders[case]()


def with_first_wzv_after_ssh(model_config, form):
    """MEASUREMENT ARM: run a card with a different after-SSH form stated.

    The cards state their own form and this never changes one of them; it
    exists so a gate can score the SAME card under both forms in one run and
    report a before/after pair, which is what an operator decision about
    switching a card needs.  ``None`` returns the configuration unchanged, so
    the default arm is byte-identical to no flag at all.

    Refuses a card that never reaches NEMO's own first ``wzv`` call: an arm
    that silently does nothing would report "no change" as if it were a
    measurement.
    """
    if form is None:
        return model_config
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        NEMO_FIRST_WZV_AFTER_SSH_FORMS,
    )
    if form not in NEMO_FIRST_WZV_AFTER_SSH_FORMS:
        raise ValueError(
            f"after-SSH arm {form!r} is not one of "
            f"{list(NEMO_FIRST_WZV_AFTER_SSH_FORMS)}")
    if getattr(model_config, "zad_qco_evaluation", "generic") != "nemo_literal":
        raise ValueError(
            "this card does not resolve NEMO's own first wzv call "
            "(zad_qco_evaluation is not 'nemo_literal'), so an after-SSH arm "
            "on it would measure nothing")
    return model_config._replace(nemo_first_wzv_after_ssh=form)


__all__ = (
    "NEMOTestcaseCard",
    "GYRESurfaceBoundaryCondition",
    "build_gyre_zco_card",
    "build_lock_exchange_zco_card",
    "build_overflow_zps_card",
    "build_orca2_zps_card",
    "build_vortex_zco_card",
    "VORTEX_UNMEASURED",
    "build_nemo_testcase_card",
    "gyre_horizontal_coordinates",
    "gyre_surface_boundary_condition",
    "gyre_vertical_ladder",
    "vortex_horizontal_coordinates",
    "vortex_initial_state_fields",
    "validate_nemo_testcase_card",
    "validate_nemo_testcase_card_for_execution",
    "with_first_wzv_after_ssh",
)
