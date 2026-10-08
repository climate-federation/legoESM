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
            # DECISION 85 (operator note BT), measured in rounds 194/198
            # and landed here.  In NEMO's vector-invariant RK3 stage the
            # continuity solve is called TWICE and the two calls are not the
            # same statement: stprk3_stg.f90:289-300 hands wzv the RAW stage
            # velocity for the momentum program, while the tracer program
            # re-solves it on the transports (traadv.f90:268); sshwzv.f90:
            # 271-299 is the solve itself.  Both fields are STATED here, per
            # decision 75 -- the split is NOT inferred from the time
            # integrator, the momentum form or wzv_call2_evaluation.
            wzv_call2_evaluation="nemo_literal",
            nemo_stage_momentum_wzv_split=True,
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
        # momentum (ocean_model_latlon_cgrid.py:4454-4462 refuses the pair
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
            # Round 77 (Decision 76): STATED on this card, never inherited
            # from the shared GYRE identity above and never inferred from the
            # time integrator.  ORCA2's OWN build runs the RK3 vector-invariant
            # program -- stp2d.f90:145-147 takes the "Vector Inv. Form"
            # Coriolis arm and :149 the "only KEG + ZAD in Vector Inv. Form"
            # advection -- and that program leaves the PREVIOUS step's linear
            # extrapolation ``ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)``
            # in the after slot (stprk3.f90:239-241).  stp2d.f90:152 turns
            # that slot into ``r3t(:,:,Kaa) = ssh(:,:,Kaa) * r1_ht_0``
            # immediately before ``CALL wzv( ..., np_velocity )`` at
            # stp2d.f90:156, so the first wzv call's scale-factor term is
            # built from the extrapolation, not from a continuity prediction.
            nemo_first_wzv_after_ssh="rk3_extrapolated",
            # Round 23 (Decision 58): ORCA2-zps resolves the SAME
            # rk3_ws+vector_invariant+nemo_literal program GYRE-zco does and
            # now takes NEMO's separately evaluated momentum continuity solve
            # under its own explicit card choice.  This is not inferred from
            # EOS or any other selector; the fail-closed field remains stated
            # on each card that resolves the program.
            nemo_stage_momentum_wzv_split=True,
            vorticity_scheme="een_total",
            # ORCA2 resolves nn_ahm_ijk_t = -30 (run ocean.output:1184), so the
            # lateral momentum viscosity coefficient is READ whole from
            # eddy_viscosity_3D.nc rather than built from the grid metrics
            # (ldfdyn.f90:348-353).  GYRE resolves the metric formula and keeps
            # the shared default.
            lateral_viscosity_coefficient_source="nemo_ahm_3d_file",
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
                    # zdfevd.f90:107-110 REPLACES the assembled avt by
                    # rn_evd where the trigger fires (zdfphy.f90:359 runs it
                    # AFTER the closure copy at :348-351), and :121/:133-135
                    # does the same to avm because this deck sets
                    # nn_evdm = 1 (gyre_omip_l2_namelist_cfg namzdf) --
                    # which is what nu_conv = K_conv = rn_evd = 100 states.
                    evd_composition="nemo_replace",
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
                # DECISION 90 (user): no default, every card states it, and
                # a NEMO card states NEMO's form (stp2d.f90:178-186).
                barotropic_slow_forcing_depth_evaluation="nemo_literal",
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
        # DECISION 90 (user, 2026-10-04): the field has no default and every
        # card states it; a NEMO card states NEMO's form.  stp2d.f90:178-179
        # (vector) / :183-184 (flux) weight the slow forcing with the
        # REFERENCE face thickness and divide by the stored reciprocal
        # r1_hu_0 -- no sea-surface stretching anywhere in it.  On a
        # full-step mesh this equals the per-level minimum of the two live
        # thicknesses (one per-face scalar cancels); over partial cells it
        # does not, which is what the seamount cards measured.
        barotropic_slow_forcing_depth_evaluation="nemo_literal",
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


# ORCA_R2 initial-condition hand alterations, transcribed from the compiled
# ``dtatsd.f90:218-253`` branch that this deck executes (``cn_cfg="ORCA"`` and
# ``nn_cfg=2`` come from the domain file's ``CfgName``/``CfgIndex``, and
# ``namelist_cfg`` sets ``ln_tsd_dmp = .true.``, so the branch is live).  NEMO
# applies them to the TIME-INTERPOLATED field before the land mask, and with
# ``ln_tint = .true.`` they are re-applied from the interpolated field at every
# call rather than accumulating.
#
# Index arithmetic, read off ``mppini.f90:1586-1594`` rather than inferred: the
# source writes global halo-frame indices ``ij0 = 101 + nn_hls`` and
# ``ii0 = 141 + nn_hls - 1``, and ``mi0``/``mj0`` map a global halo-frame index
# to ``index - nn_hls`` in the inner domain, so every ``nn_hls`` cancels and the
# boxes are the halo-independent inner one-based ranges below.  Level ranges are
# Fortran one-based inclusive.
_ORCA2_ALBORAN_BOX = (101, 109, 140, 154)      # (j0, j1, i0, i1), inner 1-based
_ORCA2_RED_SEA_BOX = (87, 96, 147, 159)
_ORCA2_ALBORAN_TEMPERATURE_OFFSETS_C = ((13, 13, -0.20), (14, 15, -0.35),
                                        (16, 25, -0.40))
_ORCA2_ALBORAN_SALINITY_OFFSETS_PSU = ((13, 13, -0.15), (14, 15, -0.25),
                                       (16, 17, -0.30), (18, 25, -0.35))
_ORCA2_RED_SEA_TEMPERATURES_C = ((4, 10, 7.0), (11, 13, 6.5), (14, 20, 6.0))


def build_orca2_ldf_dyn_coefficients(
    viscosity_path, tmask: np.ndarray, fmask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NEMO ``ldf_dyn_init`` with ``nn_ahm_ijk_t = -30``: the coefficient is READ.

    The resolved ORCA2 run prints ``nn_ahm_ijk_t = -30``, so the compiled
    routine computes no coefficient at all.  It opens ``eddy_viscosity_3D.nc``
    and reads the whole three-dimensional field at T points and at F points
    (``ldfdyn.f90:348-353``), each read carrying its own grid-point nature and
    north-fold sign (``'T'`` and ``'F'``, both ``+1``); the read path completes
    the field with the ordinary lateral boundary exchange for that nature
    (``iom.f90:958-975``).  Because the resolved operator is the laplacian
    (``ln_dynldf_lap = T``), levels one to ``jpkm1`` are then multiplied by
    ``tmask``/``fmask`` and the last level is left alone -- no square root,
    which is the bilaplacian arm (``ldfdyn.f90:388-393``).

    ``rn_Uv`` and ``rn_Lv`` are read and printed and this arm never consults
    them: ``zah0`` (``ldfdyn.f90:314``) is not referenced inside the
    ``CASE( -30 )`` block.

    The north-fold exchange is NOT applied here.  On the shipped input file it
    is the identity over the owned domain for both natures, and that is a
    MEASURED, gated statement rather than an assumption: the round-9 gate
    refuses unless the file's last owned T row is its own mirrored left half
    (``lbcnfd.f90:584-638``) and its last owned F row is the row below at the
    reversed longitude (``lbcnfd.f90:722-746``).

    Parameters
    ----------
    viscosity_path : path to ``eddy_viscosity_3D.nc``.
    tmask, fmask : the card's own ``(n_lat, n_lon, nlev)`` masks.

    Returns
    -------
    ahmt : (n_lat, n_lon, nlev)      T-point coefficient [m2/s].
    ahmf : (n_lat+1, n_lon+1, nlev)  F-point coefficient on legoESM's VERTEX
        layout, where ``vertex[j, i]`` is NEMO's F point ``(j-1, (i-1) mod
        n_lon)``; the south row has no NEMO source and is zero (a wall).
    """

    import netCDF4  # noqa: N813

    path = Path(viscosity_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing ORCA2 eddy viscosity file: {path}")
    nlev = tmask.shape[-1]
    with netCDF4.Dataset(path, "r") as ds:
        ds.set_auto_maskandscale(False)
        raw_t = np.asarray(ds.variables["ahmt_3d"][0], dtype=np.float64)
        raw_f = np.asarray(ds.variables["ahmf_3d"][0], dtype=np.float64)
    # File axes are (z, y, x); the card's are (y, x, z).
    ahmt = np.moveaxis(raw_t, 0, -1)[..., :nlev] * tmask
    ahmf_native = np.moveaxis(raw_f, 0, -1)[..., :nlev] * fmask
    n_lat, n_lon = ahmf_native.shape[0], ahmf_native.shape[1]
    ahmf = np.zeros((n_lat + 1, n_lon + 1, nlev), dtype=np.float64)
    columns = (np.arange(n_lon + 1) - 1) % n_lon
    ahmf[1:] = ahmf_native[:, columns]
    return ahmt, ahmf


def _orca2_box(field, box):
    """Inner one-based ``(j0, j1, i0, i1)`` box as a mutable view."""

    j0, j1, i0, i1 = box
    return field[j0 - 1:j1, i0 - 1:i1]


def apply_orca2_hand_alterations(temperature, salinity):
    """Apply the ORCA_R2 initial hand alterations IN PLACE, in source order.

    ``temperature`` and ``salinity`` are the time-interpolated, UNMASKED input
    fields shaped ``(nlat, nlon, nlev)``.  NEMO subtracts the Alboran Sea
    temperature then salinity increments and finally assigns the Red Sea deep
    temperatures, all before the land mask (``dtatsd.f90:218-253``, masked at
    ``:307-310``).
    """

    alboran_t = _orca2_box(temperature, _ORCA2_ALBORAN_BOX)
    for k0, k1, offset in _ORCA2_ALBORAN_TEMPERATURE_OFFSETS_C:
        alboran_t[..., k0 - 1:k1] += np.float64(offset)
    alboran_s = _orca2_box(salinity, _ORCA2_ALBORAN_BOX)
    for k0, k1, offset in _ORCA2_ALBORAN_SALINITY_OFFSETS_PSU:
        alboran_s[..., k0 - 1:k1] += np.float64(offset)
    red_sea_t = _orca2_box(temperature, _ORCA2_RED_SEA_BOX)
    for k0, k1, value in _ORCA2_RED_SEA_TEMPERATURES_C:
        red_sea_t[..., k0 - 1:k1] = np.float64(value)


def build_orca2_initial_ts(
    temperature_path,
    salinity_path,
    tmask,
    *,
    apply_hand_alterations: bool = True,
):
    """ORCA2's independent initial temperature and salinity, as NEMO builds it.

    The executed order is the compiled one: read the monthly input files, do
    ``fldread``'s two-record time interpolation, apply the ORCA_R2 hand
    alterations (``dtatsd.f90:218-253``), then mask.  The z/zps branch
    (``dtatsd.f90:307-310``) masks AFTER the copy, which is why the alterations
    are applied to the unmasked field.

    ``apply_hand_alterations=False`` is an ABLATION CONTROL for the fidelity
    gate -- it reproduces the pre-transcription state so the gate can show the
    alterations own the whole residual.  It is not a configuration knob and no
    card, recipe or driver exposes it; the card always takes the default, which
    is what NEMO executes.
    """

    import netCDF4  # noqa: N813

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

    if apply_hand_alterations:
        apply_orca2_hand_alterations(temperature, salinity)

    return (np.where(tmask, temperature, 0.0), np.where(tmask, salinity, 0.0))


# --- ORCA2 internal-wave mixing (NEMO zdfiwm; de Lavergne et al. 2020) ---
# zdf_iwm_init replaces the namelist backgrounds once the wave arm is on:
# the momentum background becomes the molecular viscosity and the tracer
# background a very small diffusive minimum, because the wave field is now
# what sets the interior background.
_ORCA2_IWM_AVMB = 1.4e-6   # NEMO rnu [m2/s]
_ORCA2_IWM_AVTB = 1.0e-10  # [m2/s]


def _orca2_iwm_forcing(path, surface_tmask: np.ndarray, lat_t, lon_t):
    """The six wave-power / decay-scale maps, as zdf_iwm_init reads them.

    Delegates to the shared loader, which masks the four power maps with the
    surface tracer mask (NEMO's ``smask0``), leaves the decay scales unmasked,
    guards a non-positive critical-slope scale before inverting it, and stores
    that inverse, which is what the scheme consumes.  The ORCA2 product is on
    this card's own grid, so the loader's coordinate check takes its
    pass-through branch and nothing is regridded; the assertion below is what
    makes that a checked fact rather than an assumption.
    """
    from legoesm.ocean.forcing.curvilinear_regrid import coords_match
    from legoesm.ocean.iwm_forcing import load_iwm_forcing, read_iwm_file

    raw = read_iwm_file(str(path))
    if not coords_match(raw["nav_lat"], raw["nav_lon"],
                        np.asarray(lat_t, dtype=np.float64),
                        np.asarray(lon_t, dtype=np.float64),
                        tol_deg=1.0e-3):
        raise ValueError(
            "the ORCA2 internal-wave product is not on this card's grid; "
            "this card reads it directly and never regrids it")
    return load_iwm_forcing(
        str(path), lat_t, lon_t,
        land_mask=np.asarray(surface_tmask, dtype=np.float64),
        coord_match_tol_deg=1.0e-3,
    )


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
    # ldfdyn.f90:348-353 opens this exact file name in the run directory; the
    # record's directory symlinks it to the deck, so the deck root is where it
    # is read from here.
    viscosity_path = root / "eddy_viscosity_3D.nc"
    # namelist_cfg's namzdf_iwm names this root for all six wave-power and
    # decay-scale fields, and the run log shows it opened six times.
    iwm_path = root / "zdfiwm_forcing_orca2.nc"
    required = (domain_path, temperature_path, salinity_path, viscosity_path,
                iwm_path)
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
                "ff_f", "gphit", "glamt",
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
    ahmt, ahmf = build_orca2_ldf_dyn_coefficients(
        viscosity_path, tmask.astype(np.float64), fmask)
    z_coord = create_partial_cell_coordinate(
        z_ref, bathymetry, bottom_index_rule="nemo_tpoint"
    )._replace(
        nemo_ldf_ahmt=jnp.asarray(ahmt, dtype=jnp.float64),
        nemo_ldf_ahmf=jnp.asarray(ahmf, dtype=jnp.float64),
        h_partial=jnp.asarray(h_partial, dtype=jnp.float64),
        bottom_level=jnp.asarray(bottom - 1, dtype=jnp.int32),
        is_active=jnp.asarray(tmask),
        nemo_e3t_0=jnp.asarray(raw_e3["e3t_0"], dtype=jnp.float64),
        nemo_bbl_e3u_0=jnp.asarray(raw_e3["e3u_0"], dtype=jnp.float64),
        nemo_bbl_e3v_0=jnp.asarray(raw_e3["e3v_0"], dtype=jnp.float64),
    )

    temperature, salinity = build_orca2_initial_ts(
        temperature_path, salinity_path, tmask)

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
        # zdfiwm.F90's initialisation runs AFTER zdf_tke_init and REPLACES the
        # namelist backgrounds, because the wave field now supplies the
        # interior background: avmb becomes the molecular viscosity, avtb a
        # very small diffusive minimum, and the equatorial 2-D shape becomes
        # uniform.  The carried avt_k/avm_k seeds above are set BEFORE that
        # reset (zdfphy.f90:227-228) and keep the namelist values; these two
        # floors are read AFTER it, every step, inside tke_avn's
        # MAX(zav, avmb) / MAX(zav, avtb_2d*avtb) (zdftke.f90:709-710).
        kappaM_min=_ORCA2_IWM_AVMB,
        kappaH_min=_ORCA2_IWM_AVTB,
    )
    # namelist_cfg's namzdf_iwm: ln_mevar=.false. (constant mixing efficiency)
    # and ln_tsdiff=.true. (salt and heat get different wave diffusivities).
    # The efficiency option is taken; the salt/heat differential is NOT, and it
    # is not silently dropped either: legoESM's implicit tracer solve carries
    # ONE diffusivity for both tracers, so a separate salt coefficient has
    # nowhere to go, and the gap is declared in the card's unmeasured features
    # below rather than hidden behind a False.  The heat and momentum halves of
    # the arm, which are what the retired mixing-length floor was standing in
    # for, are unaffected by that gap.
    iwm_config = model_config.physics.vertical_mixing.iwm._replace(
        enabled=True,
        mevar=False,
        tsdiff=False,
        require_forcing_maps=True,
    )
    model_config = model_config._replace(
        physics=model_config.physics._replace(
            vertical_mixing=model_config.physics.vertical_mixing._replace(
                tke=tke_config, iwm=iwm_config)))
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=jnp.asarray(tmask[..., 0], dtype=jnp.float64),
        initial_state=state,
        iwm_forcing=_orca2_iwm_forcing(
            iwm_path, tmask[..., 0], metric["gphit"], metric["glamt"]),
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
            # The deck's ln_tsdiff: NEMO gives salt a different wave-driven
            # diffusivity from heat, and the shared-K implicit tracer solve
            # cannot carry two.  The deck's second salt/heat differential,
            # ln_zdfddm, is unbuilt for the same reason.
            "internal_wave_salt_heat_differential",
            "double_diffusive_salt_heat_split",
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


class _VortexResolution(NamedTuple):
    """One rung of decision 74's VORTEX resolution ladder, fully resolved.

    No field here is derived at use time: the card states the grid it runs on
    and the receipt prints it.  ``ni``/``nj`` ARE what ``usr_def_nam`` computes
    from ``rn_dx`` (usrdef_nam.F90:138-139, kpi = NINT(1800e3/rn_dx)+3), but
    they are written out so a card can never silently disagree with the deck
    NEMO was run with.
    """

    suffix: str          # card-name segment; empty for the shipped 30 km deck
    dx_m: float          # namusr_def rn_dx
    dy_m: float          # namusr_def rn_dy
    ni: int              # usr_def_nam kpi
    nj: int              # usr_def_nam kpj
    dt_s: float          # namdom  rn_Dt
    n_steps: int         # namrun  nn_itend (run length; inert for a kt<=10 ladder)
    oracle_tag: str      # the acquisition variant that produced its record


# THE RULE IS NEMO'S OWN, NOT A CHOICE.  tests/VORTEX ships an AGRIF zoom whose
# space and time refinement ratios are pinned at 3 3 3
# (tests/VORTEX/EXPREF/AGRIF_FixedGrids.in:2, "22 41 22 41 3 3 3"), and the
# child namelist NEMO ships for that zoom (tests/VORTEX/EXPREF/1_namelist_cfg)
# differs from the parent in exactly three values: rn_dx and rn_dy
# 30000 -> 10000 (:21-22), rn_Dt 2880 -> 960 (:43) and nn_itend 3000 -> 6000
# (:34), plus a &namagrif sponge block (:103-108) that is meaningless without a
# nest.  rn_dz = 500 (:23), rn_ppgphi0, rn_ppumax, nn_rot, nn_e = 48 (:222) and
# every physics switch are UNCHANGED, and ln_dynldf_OFF / ln_traldf_OFF hold in
# parent and child alike, so there is no viscosity or diffusivity to rescale.
# So: dx -> dx/r, dt -> dt/r, everything else held.  The 10 km rung IS NEMO's
# own child deck; the 15 km rung is the same rule at r = 2.
#
# nn_itend has no ratio rule (6000 is not 3*3000), so the 15 km rung keeps the
# parent's 3000 and the discrepancy is registered as DECISION_NEEDED in round
# 208's receipt.  It is inert for every gate: the ladders run ten steps and the
# oracle decks pin nn_itend = 10 at every resolution.
_VORTEX_RESOLUTIONS: dict[str, _VortexResolution] = {
    "30km": _VortexResolution(
        "", _VORTEX_DX_M, _VORTEX_DY_M, _VORTEX_NI, _VORTEX_NJ,
        2880.0, 3000, "round2/round3"),
    "15km": _VortexResolution(
        "-15km", 15000.0, 15000.0, 123, 123, 1440.0, 3000,
        "round208_res15"),
    "10km": _VortexResolution(
        "-10km", 10000.0, 10000.0, 183, 183, 960.0, 6000,
        "round208_res10"),
}
def _vortex_nint(value: float) -> int:
    """Fortran ``NINT``: round half AWAY FROM ZERO, not Python's half-to-even.

    No rung here lands on a .5 case, but ``round()`` would silently disagree
    with ``usr_def_nam`` on one that did, and the statement being transcribed
    is Fortran's, so it is transcribed with Fortran's rounding rule.
    """
    if value < 0.0:
        return -_vortex_nint(-value)
    return int(math.floor(value + 0.5))


def validate_vortex_resolution(rung: "_VortexResolution") -> None:
    """Refuse a rung whose written-out cell count is not NEMO's own.

    ``usrdef_nam.F90:138-139`` computes ``kpi = NINT(1800e3/rn_dx)+3`` and
    ``kpj`` likewise; this runs that arithmetic as a CHECK on the value the
    rung writes out, never as its source.  A rung that disagreed with it would
    run a different box than the deck it cites.
    """
    want = (_vortex_nint(1800.e3 / rung.dx_m) + 3,
            _vortex_nint(1800.e3 / rung.dy_m) + 3)
    if (rung.ni, rung.nj) != want:
        raise ValueError(
            f"VORTEX rung {rung.suffix or '30km'} states {rung.ni}x"
            f"{rung.nj} cells, which is not usr_def_nam's "
            f"NINT(1800e3/rn_dx)+3 = {want[0]}x{want[1]}")


for _rung in _VORTEX_RESOLUTIONS.values():
    validate_vortex_resolution(_rung)
del _rung
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


def vortex_horizontal_coordinates(
    res: _VortexResolution | None = None,
) -> dict[str, np.ndarray]:
    """Transcribe ``usrdef_hgr.F90`` for ``nn_rot = 0`` (root grid).

    Positions are the source's own KILOMETRE coordinates: VORTEX uses
    ``glam``/``gphi`` as a Cartesian offset from the domain centre, and
    ``usrdef_istate`` multiplies them back by 1.e3.  Operand order is kept so
    the initial state reproduces the Fortran bit for bit.
    """

    res = res or _VORTEX_RESOLUTIONS["30km"]
    rad = math.pi / 180.0                            # phycst.F90:26
    omega = float(NEMO_CONSTANTS_CONFIG.Omega)       # phycst.F90:91
    radius = float(NEMO_CONSTANTS_CONFIG.R_earth)    # phycst.F90:37
    # usrdef_hgr.F90:83-84
    offset_x = (-float(res.ni - 1) + 1.0) * 0.5 * 1.e-3 * res.dx_m
    offset_y = (-float(res.nj - 1) + 1.0) * 0.5 * 1.e-3 * res.dy_m
    # usrdef_hgr.F90:103-104 -- mig/mjg minus one, the 0-based global index
    zti = np.arange(res.ni, dtype=np.float64)
    ztj = np.arange(res.nj, dtype=np.float64)
    # usrdef_hgr.F90:108-111 and :130-133 (the nn_rot==0 arm).  glamv = glamt,
    # glamf = glamu, gphiu = gphit, gphif = gphiv.
    lam_t = offset_x + res.dx_m * 1.e-3 * (zti - 0.5)
    lam_u = lam_t + res.dx_m * 1.e-3 * 0.5
    phi_t = offset_y + res.dy_m * 1.e-3 * (ztj - 0.5)
    phi_v = phi_t + res.dy_m * 1.e-3 * 0.5
    shape = (res.nj, res.ni)
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


def _vortex_grid(source: dict[str, np.ndarray], res: _VortexResolution | None = None):
    """Uniform 30 km Cartesian beta-plane with NEMO's own Coriolis operands.

    ``create_beta_plane_cgrid_geometry`` already evaluates ``f0 + beta*y`` at
    each stagger from its own ``y``; the origins below align those ``y`` with
    ``gphit``/``gphiv``.  The f arrays are nevertheless replaced by the
    transcribed ones so the card carries the source's ``(beta*phi_km)*1e3``
    operand order rather than a re-derivation in metres.
    """

    res = res or _VORTEX_RESOLUTIONS["30km"]
    grid = create_beta_plane_cgrid_geometry(
        res.nj, res.ni,
        dx_m=res.dx_m, dy_m=res.dy_m,
        f0=source["f0"], beta=source["beta"],
        # y_c[j] = origin + (j+1/2)dy must equal gphit[j]*1e3.
        x_origin_m=float(source["glamt"][0, 0]) * 1.e3 - 0.5 * res.dx_m,
        y_origin_m=float(source["gphit"][0, 0]) * 1.e3 - 0.5 * res.dy_m,
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
        np.broadcast_to(ff_t[:, :1], (res.nj, res.ni + 1)))
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


def vortex_initial_state_fields(
    source: dict[str, np.ndarray], tmask, *, ht_0=None,
):
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
    # ``ht_0`` is domain.F90:139-144's SUM_k e3t_0*tmask.  On the flat cards
    # every wet column is 5000 m, which is why the shipped decks could state
    # it as a scalar; a card with topography MUST pass its own, because the
    # depth handed to usr_def_istate is gdept_1d(k)*(1+ssh/ht_0) and nothing
    # else about the initial state sees the bottom.  Round 212 measured this:
    # it is the whole of the 2.787e-05 K difference between NEMO's seamount
    # and flat runs at kt=1, reproduced to the bit.
    if ht_0 is None:
        ht_0 = _VORTEX_H_M * surface
    ht_0 = np.asarray(ht_0, dtype=np.float64)
    r1_ht_0 = surface / (ht_0 + 1.0 - surface)
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


def _vortex_barotropic_velocity(
    ssh, u, v, tmask, res: _VortexResolution | None = None, *,
    e3u_0=None, e3v_0=None, hu_0=None, hv_0=None,
):
    """Transcribe ``istate.F90:149-154`` (the RK3 ``Kbb`` arm).

    ``e3u(Kbb)`` and ``r1_hu(Kbb)`` are the key_qco macros
    (domzgr_substitute.h90:127,136), so the ``(1+r3u)`` factor is applied in
    the accumulation and removed in the divisor as two separate statements --
    it does not cancel in the bits.
    """

    res = res or _VORTEX_RESOLUTIONS["30km"]
    tmask = np.asarray(tmask, dtype=np.float64)
    umask = tmask * np.concatenate(
        [tmask[:, 1:], np.zeros_like(tmask[:, :1])], axis=1)
    vmask = tmask * np.concatenate(
        [tmask[1:], np.zeros_like(tmask[:1])], axis=0)
    su = umask[:, :, 0]
    sv = vmask[:, :, 0]
    # domain.F90:143-144,:159 -- h*_0 = SUM_k e3*_0*mask and
    # r1_h*_0 = mask / (h*_0 + 1 - mask).  The flat cards' uniform 5000 m
    # column is the special case; a partial-cell card passes its own.
    if hu_0 is None:
        hu_0 = _VORTEX_H_M * su
    if hv_0 is None:
        hv_0 = _VORTEX_H_M * sv
    r1_hu_0 = su / (np.asarray(hu_0, dtype=np.float64) + 1.0 - su)
    r1_hv_0 = sv / (np.asarray(hv_0, dtype=np.float64) + 1.0 - sv)
    # domqco.F90:166-169 with this mesh's uniform e1e2t (r1_e1e2u = 1/e1e2u).
    area = res.dx_m * res.dy_m
    ssh_east = np.concatenate([ssh[:, 1:], np.zeros_like(ssh[:, :1])], axis=1)
    ssh_north = np.concatenate([ssh[1:], np.zeros_like(ssh[:1])], axis=0)
    r3u = 0.5 * (area * ssh + area * ssh_east) * r1_hu_0 * (1.0 / area)
    r3v = 0.5 * (area * ssh + area * ssh_north) * r1_hv_0 * (1.0 / area)
    # domzgr_substitute.h90:127 -- e3u(Kbb) = e3u_0(i,j,k)*(1+r3u*umask).
    # E3u_0 is e3t_1d(k) under key_vco_1d (a scalar here, the ladder being
    # uniform) and e3u_3d(i,j,k) under key_vco_1d3d.
    e3u_ref = (np.full((_VORTEX_NLEV,), _VORTEX_DZ_M) if e3u_0 is None
               else np.asarray(e3u_0, dtype=np.float64))
    e3v_ref = (np.full((_VORTEX_NLEV,), _VORTEX_DZ_M) if e3v_0 is None
               else np.asarray(e3v_0, dtype=np.float64))
    uu_b = np.zeros_like(su)
    vv_b = np.zeros_like(sv)
    for k in range(_VORTEX_NLEV):                  # DO_3D ... 1, jpkm1
        e3u = (e3u_ref[k] if e3u_ref.ndim == 1 else e3u_ref[:, :, k]) * (
            1.0 + r3u)
        e3v = (e3v_ref[k] if e3v_ref.ndim == 1 else e3v_ref[:, :, k]) * (
            1.0 + r3v)
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


def build_vortex_zco_card(
    momentum: str = "flux", resolution: str = "30km",
) -> NEMOTestcaseCard:
    """VORTEX root grid: a beta-plane box with a flat 5000 m zco bottom.

    ``resolution`` picks one rung of decision 74's ladder: ``"30km"`` is the
    shipped deck (63x63x10), ``"15km"`` and ``"10km"`` are NEMO's own
    refinement rule applied to it (123x123x10 and 183x183x10).  Every
    resolved value is stated in ``_VORTEX_RESOLUTIONS``; nothing about the
    grid is defaulted here.

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
    if resolution not in _VORTEX_RESOLUTIONS:
        raise ValueError(
            f"unknown VORTEX resolution {resolution!r}; expected one of "
            f"{sorted(_VORTEX_RESOLUTIONS)}")
    res = _VORTEX_RESOLUTIONS[resolution]

    source = vortex_horizontal_coordinates(res)
    grid = _vortex_grid(source, res)
    wet = _closed_box_mask(res.nj, res.ni)
    wet_np = np.asarray(wet)
    bathymetry = wet * _VORTEX_H_M
    native_3d = (res.nj, res.ni, _VORTEX_NLEV)
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
    area = np.full((res.nj, res.ni), res.dx_m * res.dy_m)
    metric = np.full((res.nj, res.ni), res.dx_m)
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
    uu_b, vv_b = _vortex_barotropic_velocity(ssh, u, v, tmask, res)
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
    zeros_u = np.zeros((res.nj, 1, _VORTEX_NLEV))
    zeros_v = np.zeros((1, res.ni, _VORTEX_NLEV))
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(temperature, dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(salinity, dtype=jnp.float64)),
        u=state.u.replace(data=jnp.asarray(
            np.concatenate([zeros_u, u], axis=1), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(
            np.concatenate([zeros_v, v], axis=0), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(ssh, dtype=jnp.float64)),
        uu_b=state.uu_b.replace(data=jnp.asarray(
            np.concatenate([np.zeros((res.nj, 1)), uu_b], axis=1),
            dtype=jnp.float64)),
        vv_b=state.vv_b.replace(data=jnp.asarray(
            np.concatenate([np.zeros((1, res.ni)), vv_b], axis=0),
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
    )
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=wet,
        initial_state=state,
    )
    # namelist_cfg:32,41 at 30 km (nn_itend = 3000 at rn_Dt = 2880 s); the
    # refined rungs carry their own rn_Dt and run length, both resolved in
    # _VORTEX_RESOLUTIONS from NEMO's shipped child deck.
    stem = "VORTEX-zco" if momentum == "flux" else "VORTEX_VEC-zco"
    card = NEMOTestcaseCard(
        stem.replace("-zco", f"{res.suffix}-zco"),
        recipe, res.dt_s, res.n_steps, 1, 0, 0, 0.0, 0.0,
        unmeasured_features=VORTEX_UNMEASURED,
    )
    validate_nemo_testcase_card(card)
    return card


# --- Decision 88 (user, 2026-10-03): VORTEX WITH TOPOGRAPHY ---------------
# The bathymetry is an EXPLICIT field recipe, stated here with every
# constant printed, exactly as the NEMO hook states it
# (scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/
#  vortex_smt_usrdef_zgr.F90):
#
#     h(x,y) = H0 - A * exp( -((x-x0)^2 + (y-y0)^2) / L^2 )
#
# H0, A, L and the 300 km westward offset are the user's (decision 88).
# glamt/gphit are in KILOMETRES in this configuration
# (tests/VORTEX/MY_SRC/usrdef_hgr.F90:78) and glamt increases with i at
# nn_rot = 0 (:108), so WEST is -glamt and x0 is negative.
_VORTEX_SMT_H0_M = 5000.0        # far-field depth              [m]
_VORTEX_SMT_A_M = 1000.0         # seamount height              [m]
_VORTEX_SMT_L_M = 150.0e3        # Gaussian radius              [m]
_VORTEX_SMT_X0_M = -300.0e3      # centre, i-direction          [m]
_VORTEX_SMT_Y0_M = 0.0           # centre, j-direction          [m]
# tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:204 -- ze3min = 0.1 * rn_dz.
_VORTEX_SMT_ZE3MIN_M = 0.1 * _VORTEX_DZ_M

# --- seamount mini-ladder rung SMT-1 (decision 93): ORCA2 rung 0's namzdf ---
# orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/
# namelist_cfg:417 / :418 / :411.
_SMT1_RN_AVM0 = 1.2e-4           # background vertical viscosity   [m2/s]
_SMT1_RN_AVT0 = 1.2e-5           # background vertical diffusivity [m2/s]
_SMT1_RN_EVD = 100.0             # enhanced vertical diffusion     [m2/s]

# --- rung SMT-2 (decision 93): ORCA2 rung 0's LINEAR bottom drag ---
# The rung-0 namelist writes ONE line, `&namdrg ln_lin = .true.`
# (namelist_cfg:270), and takes every companion from the reference namelist:
# &namdrg_bot rn_Cd0 = 1.e-3 (namelist_ref:834), rn_Uc0 = 0.4 (:835),
# ln_boost = .false. (:839), and &namdrg ln_drgimp = .true. (:817).
# zdfdrg.f90 np_lin then stores, once for the whole run,
#     rCd0_bot = rn_Cd0 * ssmask ;  rCdU_bot = - rCd0_bot * rn_Uc0
# i.e. a CONSTANT -4.0e-4 m/s on every wet column, never updated
# (l_zdfdrg = .FALSE.).
_SMT2_RN_CD0 = 1.0e-3            # linear drag coefficient         [-]
_SMT2_RN_UC0 = 0.4               # linear-law reference velocity   [m/s]

# --- rung SMT-3 (decision 93): ORCA2 rung 0's namtra_ldf ------------
_SMT3_RN_UD = 0.018              # lateral diffusive velocity       [m/s]
_SMT3_RN_LD = 200.0e3            # lateral diffusive length         [m]


def vortex_smt_bathymetry(glamt_km, gphit_km) -> np.ndarray:
    """The seamount, in the hook's own association (usrdef_zgr:167-170)."""
    zx = np.asarray(glamt_km, dtype=np.float64) * 1.0e3 - _VORTEX_SMT_X0_M
    zy = np.asarray(gphit_km, dtype=np.float64) * 1.0e3 - _VORTEX_SMT_Y0_M
    return _VORTEX_SMT_H0_M - _VORTEX_SMT_A_M * np.exp(
        -(zx * zx + zy * zy) / (_VORTEX_SMT_L_M * _VORTEX_SMT_L_M))


def vortex_smt_partial_cell_geometry(source: dict[str, np.ndarray]):
    """NEMO's resolved zps geometry for the seamount deck, statement by statement.

    Returns ``(zht, k_bot, e3t, e3u, e3v, e3f)`` with ``k_bot`` ONE-BASED and
    the scale factors carrying NEMO's ``jpk`` records (``_VORTEX_NLEV + 1``);
    record ``jpk`` is the below-bottom copy NEMO writes at ``ik+1``.

    THE LAND RING IS NOT MASKED HERE, and that is NEMO's own order, not an
    oversight: ``dom_zgr`` turns the first and last inner global row and
    column into land only AFTER ``usr_def_zgr`` returns
    (src/OCE/DOM/domzgr.F90:303-315), so those columns leave the hook with
    ``k_top = 1`` and a partial bottom cell.  ``mesh_mask.nc`` of the NEMO
    run shows exactly that (``mbathy = 10`` and ``e3t = 499.99997817 m`` on
    the ring), and a card that masked them would not reproduce it.
    """
    zht = vortex_smt_bathymetry(source["glamt"], source["gphit"])
    nlev = _VORTEX_NLEV
    gdepw_1d = np.arange(nlev + 1, dtype=np.float64) * _VORTEX_DZ_M
    ze3min = _VORTEX_SMT_ZE3MIN_M
    # usrdef_zgr:186-189 (OVERFLOW:209-212).  The downward loop's last write
    # wins, so k_bot is the number of W interfaces that clear the floor.
    k_bot = np.sum(
        gdepw_1d[None, None, :nlev] + ze3min <= zht[..., None], axis=-1)
    shape = zht.shape + (nlev + 1,)
    e3t = np.full(shape, _VORTEX_DZ_M, dtype=np.float64)   # :194-199
    jj, ii = np.nonzero(k_bot > 0)                         # :200-206, IF(ik>0)
    ik = k_bot[jj, ii] - 1
    e3t[jj, ii, ik] = np.minimum(zht[jj, ii], gdepw_1d[ik + 1]) - gdepw_1d[ik]
    e3t[jj, ii, ik + 1] = e3t[jj, ii, ik]

    def min_of_neighbours(a, axis):
        """usrdef_zgr:211-217 / :222-226 -- MIN with the next point.

        THE LAST COLUMN/ROW, and the reason, which differs between the
        faces (round 2's review corrected an earlier, wrong one):

        * ``e3f`` is formed from ``pe3v`` AFTER the hook's
          ``lbc_lnk(..., kfillmode = jpfillcopy)`` on ``pe3u``/``pe3v``
          (vortex_smt_usrdef_zgr.F90:218-219), so its outside neighbour IS a
          copy of the last point and the MIN there is that point.
        * ``e3u``/``e3v`` are formed from ``pe3t``, whose halo is NOT
          jpfillcopy-filled inside the hook (``dom_zgr`` fills it only after
          the hook returns, src/OCE/DOM/domzgr.F90:272-274).  NEMO really
          does MIN against the next T-cell there.  Copying agrees with that
          ONLY because this bathymetry deepens monotonically away from the
          seamount, so the cell just outside the east/north edge is never
          shallower.  It would NOT agree for a bathymetry that shoals at the
          edge, nor on a periodic domain, where NEMO wraps and this copies.
          All five fields are measured bit-for-bit against the run's own
          ``mesh_mask.nc`` by the round-2 geometry gate, which is what the
          claim rests on.
        """
        out = np.empty_like(a)
        if axis == 1:
            out[:, :-1] = np.minimum(a[:, :-1], a[:, 1:])
            out[:, -1] = a[:, -1]
        else:
            out[:-1] = np.minimum(a[:-1], a[1:])
            out[-1] = a[-1]
        return out

    e3u = min_of_neighbours(e3t, 1)
    e3v = min_of_neighbours(e3t, 0)
    e3f = min_of_neighbours(e3v, 1)       # from e3v, not e3u (zgr_zps:1194)
    return zht, k_bot, e3t, e3u, e3v, e3f


def build_vortex_smt_zps_card(
    momentum: str = "flux", rung: str = "smt0"
) -> NEMOTestcaseCard:
    """VORTEX with a Gaussian seamount and z partial bottom cells.

    Identical to :func:`build_vortex_zco_card` at 30 km in every namelist
    value; the ONE thing that differs is the bottom, and everything the
    bottom reaches: the column depth, the bottom level, the partial T-, U-,
    V- and F-cell thicknesses, and -- through ``ht_0`` -- the depth the
    initial state is evaluated on.

    ``rung`` selects the seamount mini-ladder step (decision 93, operator
    note CE).  ``"smt0"`` is the shipped VORTEX vertical-physics block;
    ``"smt1"`` moves ONE module -- namzdf -- to ORCA2 rung 0's values and
    nothing else; ``"smt2"`` is SMT-1 with ONE further module, namdrg, at
    rung 0's linear bottom drag; ``"smt3"`` adds ORCA2 rung 0's namtra_ldf
    block.  Only the vector deck is carried up the ladder, because ORCA2 is
    vector-invariant.
    """
    if momentum not in ("flux", "vector"):
        raise ValueError(
            f"unknown VORTEX_SMT momentum deck {momentum!r}; expected 'flux' "
            "(ln_dynadv_up3) or 'vector' (ln_dynadv_vec)")
    if rung not in ("smt0", "smt1", "smt2", "smt3"):
        raise ValueError(
            f"unknown VORTEX_SMT mini-ladder rung {rung!r}; expected 'smt0' "
            "(the shipped namzdf block), 'smt1' (ORCA2 rung 0's background "
            "mixing and enhanced vertical diffusion), 'smt2' (SMT-1 plus "
            "rung 0's linear bottom drag), or 'smt3' (SMT-2 plus rung 0's "
            "lateral tracer diffusion)")
    if rung in ("smt1", "smt2", "smt3") and momentum != "vector":
        raise ValueError(
            "the seamount mini-ladder (decision 93) is carried on the VECTOR "
            "deck only; there is no flux-form SMT-1 card")
    res = _VORTEX_RESOLUTIONS["30km"]
    source = vortex_horizontal_coordinates(res)
    grid = _vortex_grid(source, res)
    wet = _closed_box_mask(res.nj, res.ni)
    wet_np = np.asarray(wet)
    nlev = _VORTEX_NLEV
    native_3d = (res.nj, res.ni, nlev)

    zht, k_bot, e3t_jpk, e3u_jpk, e3v_jpk, e3f_jpk = (
        vortex_smt_partial_cell_geometry(source))
    e3t = e3t_jpk[:, :, :nlev]
    e3u_0 = e3u_jpk[:, :, :nlev]
    e3v_0 = e3v_jpk[:, :, :nlev]
    e3f_0 = e3f_jpk[:, :, :nlev]

    # dom_msk: a cell is wet when it is inside the closed box AND above the
    # bottom level the zps rule chose.
    k_idx = np.arange(nlev)[None, None, :]
    tmask = (wet_np[..., None] > 0.0) & (k_idx < k_bot[..., None])
    tmask = tmask.astype(np.float64)
    umask_3d = tmask * np.concatenate(
        [tmask[:, 1:], np.zeros_like(tmask[:, :1])], axis=1)
    vmask_3d = tmask * np.concatenate(
        [tmask[1:], np.zeros_like(tmask[:1])], axis=0)
    fe3mask = np.asarray(nemo_fe3mask_from_tmask(jnp.asarray(tmask)))
    fmask = fe3mask            # rn_shlat = 0, namelist_cfg:99

    # domain.F90:139-152.
    ht_0 = np.sum(e3t * tmask, axis=-1)
    hu_0 = np.sum(e3u_0 * umask_3d, axis=-1)
    hv_0 = np.sum(e3v_0 * vmask_3d, axis=-1)
    hf_0 = np.sum(
        e3f_0 * vmask_3d * np.roll(vmask_3d, -1, axis=1), axis=-1)

    gdept_1d = (np.arange(nlev, dtype=np.float64) + 0.5) * _VORTEX_DZ_M
    gdepw_1d = np.arange(nlev, dtype=np.float64) * _VORTEX_DZ_M
    area = np.full((res.nj, res.ni), res.dx_m * res.dy_m)
    metric = np.full((res.nj, res.ni), res.dx_m)
    operands = NemoEENBarotropicOperands(
        ff_f=np.asarray(grid.ff_f),
        e3u_0=e3u_0, e3v_0=e3v_0, e3f_0=e3f_0,
        umask=umask_3d, vmask=vmask_3d,
        fmask=fmask, fe3mask=fe3mask,
        hu_0=hu_0, hv_0=hv_0, hf_0=hf_0,
        e1t=metric, e2t=metric, e1u=metric, e2u=metric,
        e1v=metric, e2v=metric, e1f=metric, e2f=metric,
    )
    z_ref = create_z_star_from_thicknesses(
        jnp.full((nlev,), _VORTEX_DZ_M),
        t_depth_ref_m=gdept_1d,
        # domzgr_substitute.h90:71-78: under key_vco_1d3d gdept_0, gdepw_0 and
        # e3w_0 are STILL the 1-D ladder -- the key makes e3t/e3u/e3v/e3f 3-D
        # and nothing else.  This is NEMO's own zps shape, not a shortcut.
        nemo_gdept_0_m=np.broadcast_to(gdept_1d, native_3d),
        nemo_gdepw_0_m=np.broadcast_to(gdepw_1d, native_3d),
        nemo_e3t_0_m=e3t,
        nemo_e3w_0_m=np.broadcast_to(
            np.full((nlev,), _VORTEX_DZ_M), native_3d),
        nemo_hu_0_m=hu_0, nemo_hv_0_m=hv_0,
        nemo_e1e2t_m=area, nemo_e1e2u_m=area, nemo_e1e2v_m=area,
        nemo_e2u_m=metric, nemo_e1v_m=metric,
        nemo_een_barotropic_m=operands,
    )
    z_coord = create_partial_cell_coordinate(
        z_ref, jnp.asarray(np.where(wet_np > 0.0, zht, 0.0)),
        bottom_index_rule="nemo_zps_e3min",
        min_partial_thickness=_VORTEX_SMT_ZE3MIN_M,
    )
    # fix 5 of round 2's review: the card derives the geometry in numpy (the
    # statement-by-statement transcription above) and the shared factory
    # derives bottom_level / h_partial again in JAX from the same bathymetry.
    # Nothing used to pin the two together, so they are pinned here: outside
    # the closed box NEMO's k_bot survives unmasked (see the docstring) while
    # the model's column is dry, which is the only place they may differ.
    _bl = np.asarray(z_coord.bottom_level)
    _wetcol = wet_np > 0.0
    if not np.array_equal(_bl[_wetcol] + 1, k_bot[_wetcol]):
        raise ValueError(
            "VORTEX_SMT: the card's transcribed k_bot and the shared "
            "partial-cell factory's bottom_level disagree")
    if not np.array_equal(np.asarray(z_coord.h_partial)[
            np.asarray(z_coord.is_active)], e3t[np.asarray(
                z_coord.is_active)]):
        raise ValueError(
            "VORTEX_SMT: the card's transcribed e3t and the shared "
            "partial-cell factory's h_partial disagree on a wet cell")

    ssh, temperature, salinity, u, v = vortex_initial_state_fields(
        source, tmask, ht_0=ht_0)
    uu_b, vv_b = _vortex_barotropic_velocity(
        ssh, u, v, tmask, res, e3u_0=e3u_0, e3v_0=e3v_0,
        hu_0=hu_0, hv_0=hv_0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=0.0, T_deep=0.0, S_uniform=0.0,
        H_max=_VORTEX_H_M,
        land_mask_override=wet,
        H_bathy_override=jnp.asarray(ht_0),
        nemo_prognostic_barotropic_velocity=True,
    )
    zeros_u = np.zeros((res.nj, 1, nlev))
    zeros_v = np.zeros((1, res.ni, nlev))
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(temperature, dtype=jnp.float64)),
        S=state.S.replace(data=jnp.asarray(salinity, dtype=jnp.float64)),
        u=state.u.replace(data=jnp.asarray(
            np.concatenate([zeros_u, u], axis=1), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(
            np.concatenate([zeros_v, v], axis=0), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(ssh, dtype=jnp.float64)),
        uu_b=state.uu_b.replace(data=jnp.asarray(
            np.concatenate([np.zeros((res.nj, 1)), uu_b], axis=1),
            dtype=jnp.float64)),
        vv_b=state.vv_b.replace(data=jnp.asarray(
            np.concatenate([np.zeros((1, res.ni)), vv_b], axis=0),
            dtype=jnp.float64)),
    )
    model_config = _model_config(
        barotropic_time_filter="nemo_ab3am4",
        n_barotropic_substeps=48,
        bbl_adv_option=0, bbl_gamma_s=0.0,
        bbl_diffusive_option=0, bbl_aht_m2_s=0.0,
        whole_step_identity=("vortex_flux_up3_een" if momentum == "flux"
                             else "vortex_vector_een_c2"),
        tke_langmuir_evaluation=None,
    )
    # stp2d.F90:177-186 depth-averages the slow forcing with the REFERENCE
    # face thickness and the stored reciprocal, not with the live min-rule
    # thickness; the two differ only over partial cells (round 213).  Stated
    # on the card, explicitly, because it is a statement choice.
    model_config = model_config._replace(
        barotropic=model_config.barotropic._replace(
            # Stated on the seamount card too, where it was first measured,
            # rather than inherited silently from the shared block above.
            barotropic_slow_forcing_depth_evaluation="nemo_literal"))
    if rung in ("smt1", "smt2", "smt3"):
        # DECISION 93 (user), rung SMT-1.  ONE namelist module moves to ORCA2
        # rung 0's values; every line is cited to that deck
        # (phase3/orca2_rounds/round83/acquisition/
        #  orca2_rung0_restart_list_10step_a_np2/namelist_cfg):
        #
        #   :417  rn_avm0  = 1.2e-4   background vertical eddy VISCOSITY
        #   :418  rn_avt0  = 1.2e-5   background vertical eddy DIFFUSIVITY
        #   :409  ln_zdfevd = .true.  enhanced vertical diffusion ON
        #   :410  nn_evdm   = 0       evd on the TRACER only, never on avm
        #   :411  rn_evd    = 100.    the coefficient it writes [m2/s]
        #
        # Rung 0 keeps ln_zdfcst with the UNIFORM background (:419 nn_avb=0,
        # :420 nn_havtb=0, :421 ln_zdfcst=.true.), which is what the two
        # scalars below express -- there is no latitude shape and no profile.
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.convection.config import (
            EnhancedDiffusionConfig, OceanConvectionConfig,
        )
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig,
        )
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        model_config = model_config._replace(
            A_v=_SMT1_RN_AVM0, K_v=_SMT1_RN_AVT0,
            physics=OceanPhysicsConfig(
                # ln_zdfcst with a uniform background is NOT a closure: it is
                # the two scalars above, which the implicit solve already
                # carries as A_v / K_v.  Selecting a closure here would run a
                # second one.
                vertical_mixing=VerticalMixingConfig(scheme="none"),
                # Every other namzdf / namtra_ldf / namdyn_ldf / namdrg /
                # namsbc switch on this deck is OFF, and each is stated
                # rather than inherited (ln_traldf_OFF, ln_dynldf_OFF,
                # ln_drg_OFF, ln_usr writing zeros, ln_traqsr=.false.).
                lateral_mixing=LateralMixingConfig(scheme="none"),
                surface_forcing=SurfaceForcingConfig(scheme="none"),
                bottom_drag=BottomDragConfig(scheme="none"),
                shortwave_penetration=None,
                mle=None,
                constants=NEMO_CONSTANTS_CONFIG,
                convection=OceanConvectionConfig(
                    scheme="enhanced_diffusion",
                    enhanced_diffusion=EnhancedDiffusionConfig(
                        # zdfevd.F90:94  p_avt = rn_evd * wmask
                        K_conv=_SMT1_RN_EVD,
                        # zdfevd.F90:106  nn_evdm == 1 guards the MOMENTUM
                        # arm; rung 0 sets nn_evdm = 0, so the viscosity is
                        # never touched and the convective one is zero.
                        nu_conv=0.0,
                        # The stable branch contributes nothing of its own:
                        # the background is the namelist pair above.
                        K_bg=0.0, nu_bg=0.0,
                        smooth_transition=False,
                        # zdfevd.F90:93  MIN( rn2, rn2b ) <= -1.e-12
                        n2_threshold=-1.0e-12,
                        two_level_trigger=True,
                        # The S-EOS of this deck (decision 69), not the
                        # campaign TEOS-10: the trigger must see the same
                        # fluid the rest of the card runs.
                        n2_mode="nemo_bn2",
                        n2_eos_form="seos",
                        # zdfevd.f90:107-110 REPLACES the assembled avt by
                        # rn_evd where the trigger fires; zdfphy.f90:359
                        # runs it AFTER the background copy at :348-351, so
                        # the fired interfaces carry rn_evd ALONE and not
                        # rn_evd + rn_avt0.  nn_evdm = 0 (rung-0
                        # namelist_cfg:410) is the nu_conv = 0 above:
                        # zdfevd.f90:121 leaves avm untouched.
                        evd_composition="nemo_replace",
                    ),
                ),
            ),
        )
    if rung in ("smt2", "smt3"):
        # DECISION 93 (user), rung SMT-2: ORCA2 rung 0's LINEAR BOTTOM DRAG,
        # the second module of the mini-ladder.  The deck writes one line,
        # &namdrg ln_lin = .true. (rung-0 namelist_cfg:270), and takes every
        # companion from the reference namelist -- which is what rung 0 does
        # too, so the resolved set is rung 0's and nothing here is a choice:
        #
        #   ln_drg_OFF   = .false.   namelist_ref:812   (rung 0 leaves unset)
        #   ln_non_lin   = .false.   namelist_ref:814
        #   ln_loglayer  = .false.   namelist_ref:815
        #   ln_drgimp    = .true.    namelist_ref:817   -> implicit, in the
        #                                                 tridiagonal diagonal
        #   rn_Cd0       = 1.e-3     namelist_ref:834   (&namdrg_bot)
        #   rn_Uc0       = 0.4       namelist_ref:835
        #   ln_boost     = .false.   namelist_ref:839   -> zmsk_boost == ssmask
        #
        # zdfdrg.f90's np_lin branch stores the coefficient ONCE
        # (`l_zdfdrg = .FALSE.`): rCd0_bot = rn_Cd0 * ssmask and then
        # rCdU_bot = - rCd0_bot * rn_Uc0.  It never reads the velocity, which
        # is the whole difference from GYRE's np_non_lin.
        #
        # ln_drgimp = .true. with ln_dynspg_ts = .true. is ONE composition in
        # NEMO and is selected here as one, exactly as the GYRE card selects
        # it: the frozen rCdU_bot in the external-mode substeps
        # (dynspg_ts.f90:1245-1246, :1271-1272), the baroclinic residual
        # correction (dynzdf.f90:160-169), and the bottom-cell implicit
        # diagonal over PARTIAL cells (dynzdf.f90:306, :473 -- divisor
        # e3u_3d(iku)*(1+r3u(Kaa)*umask(iku)) at iku = mbku).
        model_config = model_config._replace(
            bottom_drag=model_config.bottom_drag._replace(
                bottom_drag_scheme="nemo_linear",
                bottom_drag_cd0=_SMT2_RN_CD0,
                bottom_drag_uc0=_SMT2_RN_UC0,
                # np_lin reads neither of these; they are stated so the card
                # records what the namelist resolved rather than leaving the
                # library default to speak for the run.
                bottom_drag_cdmax=0.1,
                bottom_drag_z0=3.0e-3,
                bottom_drag_ke0=2.5e-3,
                # ln_boost = .false.: no regional boost, and NEMO applies the
                # drag to the bottom cell alone, never over a K&E99 band.
                bottom_drag_bbl_thickness=0.0,
                bottom_drag_bg_velocity=0.0,
                bottom_drag_r=0.0,
            ),
            zdf_drag_in_matrix=True,
            zdf_baroclinic_only=True,
            barotropic_drag_substep=True,
        )
    if rung == "smt3":
        # DECISION 93 (user), rung SMT-3: ORCA2 rung 0's namtra_ldf block.
        # The deck selects laplacian standard isoneutral diffusion with MSC,
        # coefficient mode 20, rn_Ud=0.018 m/s and rn_Ld=200 km.  On this
        # uniform 30-km mesh mode 20 evaluates an equatorial coefficient of
        # 0.5*rn_Ud*MAX(e1u,e2u)=270 m2/s; the production operator evaluates
        # the face coefficient from the metric and rn_Ud rather than relying
        # on that derived scalar.
        from legoesm.ocean.physics.lateral_mixing.config import (
            GMRediConfig, VisbeckConfig,
        )
        model_config = model_config._replace(
            K_h=0.0,
            gm_redi=GMRediConfig(
                kappa_GM=0.0,
                kappa_Redi=0.5 * _SMT3_RN_UD * res.dx_m,
                S_max=0.01,
                visbeck=VisbeckConfig(enabled=False),
                slope_scheme="nemo_iso_lap",
                slope_density="neutral",
                slope_limit="nemo_cap",
                slope_positions="nemo_native",
                nemo_mld_slope_ramp=True,
                mld_criterion="n2_integral",
                slope_n2="nemo_bn2",
                slope_n2_evaluation="carried_step_entry",
                slope_prd_geometry_stage="current_step",
                slope_prd_evaluation="nemo_literal",
                slope_metric_evaluation="nemo_reciprocal",
                slope_face_thickness_evaluation="nemo_qco_live",
                redi_flux_face_thickness_evaluation="nemo_qco_live",
                slope_depth_evaluation="nemo_qco_live_literal",
                nemo_slope_shapiro=True,
                kappa_redi_horizontal_evaluation="nemo_metric_literal",
                kappa_redi_diffusive_velocity=_SMT3_RN_UD,
                redi_vertical_skew_evaluation="nemo_literal",
                redi_a33_evaluation="nemo_literal",
                redi_w_slope_stage_evaluation="nemo_post_slope_pair",
                msc_stabilize=True,
                implicit_K33=True,
            ),
        )
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid,
        z_coord=z_coord,
        land_mask=wet,
        initial_state=state,
    )
    if rung == "smt3":
        case_name = "VORTEX_SMT3_VEC-zps"
    elif rung == "smt2":
        case_name = "VORTEX_SMT2_VEC-zps"
    elif rung == "smt1":
        case_name = "VORTEX_SMT1_VEC-zps"
    elif momentum == "flux":
        case_name = "VORTEX_SMT-zps"
    else:
        case_name = "VORTEX_SMT_VEC-zps"
    card = NEMOTestcaseCard(
        case_name,
        recipe, res.dt_s, res.n_steps, 1, 0, 0, 0.0, 0.0,
        unmeasured_features=VORTEX_UNMEASURED,
    )
    validate_nemo_testcase_card(card)
    return card


# --- TSUNAMI (NEMO 5.0.2 tests/TSUNAMI), lane round 1: TRANSCRIPTION ONLY ---
# Every value below is the RESOLVED namelist (EXPREF/namelist_cfg over
# cfgs/SHARED/namelist_ref) or a usrdef formula, cited where it is read.
# The card is selected only by calling build_tsunami_zco_card; it is NOT in
# build_nemo_testcase_card's dispatch, so no existing gate or card can reach it.
#
# THE STEP PROGRAM IS NOT ANY CARD'S.  cpp_TSUNAMI.fcm compiles key_qco
# key_xios key_vco_1d and NOT key_RK3, so nemogcm.F90:186 calls stp_MLF, and
# tests/TSUNAMI/MY_SRC/stpmlf.F90 replaces NEMO's leapfrog step with
#   sbc -> dom_qco_r3c(ssh(Naa)) -> uu/vv(Nrhs)=0 -> dyn_spg -> dia_wri -> swap
# (stpmlf.F90:111-134): the split-explicit external mode ALONE.  No ssh_nxt,
# no 3-D momentum update (dyn_zdf/dyn_atf never run), no ssh_atf, no tracer,
# no zdf_phy.  Every whole_step_identity legoESM has is an RK3 program; the
# card therefore carries the closest existing barotropic arm and DECLARES the
# program gap in TSUNAMI_UNMEASURED, so validate_nemo_testcase_card_for_execution
# refuses it (the VORTEX round-1 pattern).


class TsunamiResolvedNamelist(NamedTuple):
    """The resolved TSUNAMI switch set, every one stated (none defaulted)."""

    # &namusr_def, namelist_cfg:19-29 (read at usrdef_nam.F90:74)
    rn_domszx_km: float = 2000.0
    rn_domszy_km: float = 2000.0
    rn_domszz_m: float = 100.0
    rn_dx_km: float = 10.0
    rn_dy_km: float = 10.0
    rn_0xratio: float = 0.2
    rn_0yratio: float = 0.4
    nn_fcase: int = 0
    rn_ppgphi0_deg: float = 38.5
    ln_Iperio: bool = True
    ln_Jperio: bool = True
    # &namrun namelist_cfg:40-43, &namdom :48 (rn_atfp from namelist_ref)
    nn_it000: int = 1
    nn_itend: int = 100
    rn_Dt_s: float = 1000.0
    rn_atfp: float = 0.1
    ln_rstart: bool = False
    # cpp_TSUNAMI.fcm: key_qco key_xios key_vco_1d, no key_RK3 -> stp_MLF
    key_RK3: bool = False
    key_qco: bool = True
    key_vco_1d: bool = True
    # &namdyn_spg namelist_cfg:163 + namelist_ref
    ln_dynspg_ts: bool = True
    ln_bt_fw: bool = True
    nn_bt_flt: int = 1
    rn_bt_alpha: float = 0.0
    ln_bt_auto: bool = True
    rn_bt_cmax: float = 0.8
    nn_e_resolved: int = 6        # dynspg_ts.F90:1240 CEILING(rn_Dt/rn_bt_cmax*zcmax)
    # &namdyn_vor namelist_cfg:153 + namelist_ref nn_e3f_typ
    ln_dynvor_een: bool = True
    nn_e3f_typ: int = 0
    # switches NEMO's TSUNAMI program never reaches (stpmlf.F90:111-134) but
    # which the deck selects; stated so nothing is inherited
    ln_dynadv_OFF: bool = True
    ln_traadv_OFF: bool = True
    ln_traldf_OFF: bool = True
    ln_dynldf_OFF: bool = True
    ln_hpg_sco: bool = True
    ln_zdfcst: bool = True
    rn_avm0_m2_s: float = 1.2e-4
    rn_avt0_m2_s: float = 1.2e-5
    ln_zad_Aimp: bool = False
    rn_shlat: float = 0.0
    ln_drg_OFF: bool = True
    ln_seos: bool = True
    ln_usr_sbc: bool = True
    nn_fsbc: int = 1
    # &nammpp nn_hls from namelist_ref (enters no formula: dom_hgr's lbc_lnk
    # wraps glamt/gphit periodically before usr_def_istate_ssh reads them)
    nn_hls: int = 2
    # selectors dyn_spg_ts reads on the executed path (dynspg_ts.F90:373-446,
    # 512-525, 601-692, 781-849); namelist_ref, not overridden by the deck
    ln_dynvor_msk: bool = False
    ln_apr_dyn: bool = False
    ln_rnf: bool = False
    ln_isf: bool = False
    ln_sdw: bool = False
    ln_bdy: bool = False
    ln_tide: bool = False
    ln_tide_pot: bool = False
    ln_wd_dl: bool = False
    ln_wd_dl_bc: bool = False
    ln_sshinc: bool = False
    ln_asmiau: bool = False


TSUNAMI_NAMELIST = TsunamiResolvedNamelist()
# the card carries none of these forcings/limiters/increments
_TSUNAMI_ABSENT_SPG_TERMS = (
    "ln_dynvor_msk", "ln_apr_dyn", "ln_rnf", "ln_isf", "ln_sdw", "ln_bdy",
    "ln_tide", "ln_tide_pot", "ln_wd_dl", "ln_wd_dl_bc", "ln_sshinc",
    "ln_asmiau",
)

_TSUNAMI_NLEV = 1                    # jpkm1 (usrdef_nam.F90:98 kpk = 2)
_TSUNAMI_H_M = TSUNAMI_NAMELIST.rn_domszz_m

# nameos: namelist_cfg:121 selects ln_seos and sets NO coefficient, so every
# one is namelist_ref's.  T and S are uniform (usrdef_istate.F90:65-66) and
# NEMO's TSUNAMI program never evaluates density; stated anyway.
_TSUNAMI_SEOS = NemoSEOSConfig(
    rho0=float(NEMO_CONSTANTS_CONFIG.rho_0),
    a0=1.6550e-1, b0=7.6554e-1, lambda1=5.9520e-2, lambda2=7.4914e-4,
    mu1=1.4970e-4, mu2=1.1090e-5, nu=2.4341e-3, T0=10.0, S0=35.0,
)

TSUNAMI_UNMEASURED: tuple[str, ...] = (
    # stpmlf.F90:111-134 under no key_RK3 (nemogcm.F90:186): the external mode
    # alone, MLF branch of dynspg_ts (dynspg_ts.F90:303-482, 914-999).
    "B1:stp_mlf_external_mode_only_program",
    # stpmlf.F90:118 builds r3(Naa) from the slot ssh dyn_spg later
    # overwrites, and finalize_lbc is never called, so the Kmm metric dyn_spg
    # reads lags the Kmm ssh (dynspg_ts.F90:484-491; dyn_cor_2D_init).
    "B2:mlf_lagged_qco_metric",
    # namelist_cfg:29 ln_Jperio: legoESM's y-wrap is a process-global halo
    # mode (halo_latlon.set_meridionally_periodic), not card data.
    "B3:j_periodic_card_topology",
    # namelist_cfg:28 ln_Iperio on a fully wet box: every certified NEMO card
    # has a closed ring, so the NEMO-literal barotropic arms have never run
    # across an open periodic seam.
    "B4:i_periodic_nemo_literal_barotropic",
    # usrdef_nam.F90:98 kpk = 2: ONE wet level.  The coordinate builds only
    # under allow_single_level=True; no model step has run on one level.
    "B5:single_wet_level_column",
)


def tsunami_horizontal_coordinates() -> dict[str, np.ndarray]:
    """Transcribe ``tests/TSUNAMI/MY_SRC/usrdef_hgr.F90:79-122``.

    Positions in KILOMETRES, as the source writes them.  ``mig(ji,0)`` is the
    1-based global interior index, so the 0-based ``i`` gives
    ``zti = (i + 1) - ii0``.
    """
    nl = TSUNAMI_NAMELIST
    ni = _vortex_nint(nl.rn_domszx_km / nl.rn_dx_km) + 1   # usrdef_nam.F90:92
    nj = _vortex_nint(nl.rn_domszy_km / nl.rn_dy_km) + 1   # usrdef_nam.F90:93
    ii0 = _vortex_nint(float(ni) * nl.rn_0xratio)          # usrdef_hgr.F90:79
    ij0 = _vortex_nint(float(nj) * nl.rn_0yratio)          # usrdef_hgr.F90:80
    zti = np.arange(1, ni + 1, dtype=np.float64) - ii0     # :90
    ztj = np.arange(1, nj + 1, dtype=np.float64) - ij0     # :91
    lam_t = nl.rn_dx_km * zti                              # :93
    lam_u = nl.rn_dx_km * (zti + 0.5)                      # :94
    phi_t = nl.rn_dy_km * ztj                              # :98
    phi_v = nl.rn_dy_km * (ztj + 0.5)                      # :99
    shape = (nj, ni)
    glamt = np.broadcast_to(lam_t[None, :], shape).copy()
    glamu = np.broadcast_to(lam_u[None, :], shape).copy()
    gphit = np.broadcast_to(phi_t[:, None], shape).copy()
    gphiv = np.broadcast_to(phi_v[:, None], shape).copy()
    # :120-123, nn_fcase = 0: an f-plane, f0 = 2*omega*SIN(rad*rn_ppgphi0).
    if nl.nn_fcase != 0:
        raise ValueError("TSUNAMI card transcribes nn_fcase = 0 only")
    rad = math.pi / 180.0                                  # phycst.F90 rad
    f0 = 2.0 * float(NEMO_CONSTANTS_CONFIG.Omega) * math.sin(
        rad * nl.rn_ppgphi0_deg)
    return {
        "ni": ni, "nj": nj, "ii0": ii0, "ij0": ij0,
        "glamt": glamt, "glamu": glamu, "glamv": glamt.copy(),
        "glamf": glamu.copy(),
        "gphit": gphit, "gphiu": gphit.copy(), "gphiv": gphiv,
        "gphif": gphiv.copy(),
        "ff_t": np.full(shape, f0), "ff_f": np.full(shape, f0), "f0": f0,
    }


def tsunami_initial_ssh(glamt_km: np.ndarray, gphit_km: np.ndarray) -> np.ndarray:
    """Transcribe ``usr_def_istate_ssh`` (usrdef_istate.F90:93-101).

    ``dom_hgr`` wraps ``glamt``/``gphit`` periodically before this routine
    runs (domhgr.F90:114), so ``MAXVAL`` over the halo-inclusive array is the
    interior maximum.  ``0.1`` is double under ``-fdefault-real-8``.
    """
    glamt_km = np.asarray(glamt_km, dtype=np.float64)
    gphit_km = np.asarray(gphit_km, dtype=np.float64)
    zdist = np.sqrt(glamt_km * glamt_km + gphit_km * gphit_km)   # :94
    zmax = np.max(zdist) / 20.0                                  # :96
    ssh = np.zeros_like(zdist)                                   # :99
    inside = zdist <= zmax                                       # :101
    ssh[inside] = 0.1 * np.cos(zdist[inside] / zmax * math.pi * 0.5)
    return ssh


def build_tsunami_zco_card() -> NEMOTestcaseCard:
    """TSUNAMI: doubly periodic 201x201 f-plane, one 100 m z level.

    Geometry from usrdef_nam/hgr/zgr, initial state from usrdef_istate, zero
    forcing (usrdef_sbc.F90:60-68).  NOT execution-ready: see
    ``TSUNAMI_UNMEASURED``.
    """
    nl = TSUNAMI_NAMELIST
    src = tsunami_horizontal_coordinates()
    ni, nj = src["ni"], src["nj"]
    dx_m, dy_m = nl.rn_dx_km * 1.e3, nl.rn_dy_km * 1.e3     # usrdef_hgr.F90:106-109
    grid = create_beta_plane_cgrid_geometry(
        nj, ni, dx_m=dx_m, dy_m=dy_m, f0=src["f0"], beta=0.0,
        x_origin_m=float(src["glamt"][0, 0]) * 1.e3 - 0.5 * dx_m,
        y_origin_m=float(src["gphit"][0, 0]) * 1.e3 - 0.5 * dy_m,
        radius=float(NEMO_CONSTANTS_CONFIG.R_earth),
        cartesian_pseudo_lat=True, dtype=jnp.float64,
    )
    grid = grid._replace(
        f_T=jnp.asarray(src["ff_t"], dtype=jnp.float64),
        f_u=jnp.full((nj, ni + 1), src["f0"], dtype=jnp.float64),
        f_v=jnp.full((nj + 1, ni), src["f0"], dtype=jnp.float64),
        ff_f=jnp.asarray(src["ff_f"], dtype=jnp.float64),
    )
    # usrdef_zgr.F90:187-189: k_top = 1, k_bot = jpkm1 EVERYWHERE -- no land
    # ring; periodicity (not walls) closes the box.
    wet = jnp.ones((nj, ni), dtype=jnp.float64)
    native_3d = (nj, ni, _TSUNAMI_NLEV)
    # usrdef_zgr.F90:128-154 with depth_e3.F90:68-73,125-130: zd = 100/1,
    # gdepw_1d = (0, 100), gdept_1d = (50, 150), e3t_1d = e3w_1d = (100, 100)
    # exactly, unchanged by the e3->depth round trip.  key_vco_1d makes every
    # e3*_0 the 1-D ladder (domzgr_substitute.h90:72-91); key_qco stretches it
    # by (1 + r3*) at run time (:126-140).  Record jpk is the dummy bottom.
    zd = nl.rn_domszz_m / float(_TSUNAMI_NLEV)
    thickness = np.full(native_3d, zd)
    gdept_1d = np.array([0.5 * zd])
    gdepw_1d = np.array([0.0])
    ones2 = np.ones((nj, ni))
    ones3 = np.ones(native_3d)
    area = np.full((nj, ni), dx_m * dy_m)
    metric_x = np.full((nj, ni), dx_m)
    metric_y = np.full((nj, ni), dy_m)
    operands = NemoEENBarotropicOperands(
        ff_f=np.asarray(grid.ff_f),
        e3u_0=thickness, e3v_0=thickness, e3f_0=thickness,
        # all-wet periodic box: every face and vertex mask is 1, and
        # rn_shlat = 0 (namelist_cfg:90) has no coast to act on
        umask=ones3, vmask=ones3, fmask=ones3, fe3mask=ones3,
        hu_0=_TSUNAMI_H_M * ones2, hv_0=_TSUNAMI_H_M * ones2,
        hf_0=_TSUNAMI_H_M * ones2,
        e1t=metric_x, e2t=metric_y, e1u=metric_x, e2u=metric_y,
        e1v=metric_x, e2v=metric_y, e1f=metric_x, e2f=metric_y,
    )
    z_ref = create_z_star_from_thicknesses(
        jnp.full((_TSUNAMI_NLEV,), zd),
        t_depth_ref_m=gdept_1d,
        nemo_gdept_0_m=np.broadcast_to(gdept_1d, native_3d),
        nemo_gdepw_0_m=np.broadcast_to(gdepw_1d, native_3d),
        nemo_e3t_0_m=thickness,
        nemo_e3w_0_m=thickness,
        nemo_hu_0_m=_TSUNAMI_H_M * ones2,
        nemo_hv_0_m=_TSUNAMI_H_M * ones2,
        nemo_e1e2t_m=area, nemo_e1e2u_m=area, nemo_e1e2v_m=area,
        nemo_e2u_m=metric_y, nemo_e1v_m=metric_x,
        nemo_een_barotropic_m=operands,
        allow_single_level=True,   # jpkm1 = 1 (usrdef_nam.F90:98)
    )
    z_coord = create_full_step_coordinate(
        z_ref, jnp.full((nj, ni), _TSUNAMI_NLEV - 1))
    ssh = tsunami_initial_ssh(src["glamt"], src["gphit"])
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=0.0, T_deep=0.0, S_uniform=0.0,
        H_max=_TSUNAMI_H_M,
        land_mask_override=wet,
        H_bathy_override=wet * _TSUNAMI_H_M,
        nemo_prognostic_barotropic_velocity=True,
    )
    # usrdef_istate.F90:65-68: T = 20, S = 30, u = v = 0; istate.F90 then
    # builds uu_b/vv_b from u = v = 0, i.e. exactly zero.
    state = state._replace(
        T=state.T.replace(data=jnp.full(native_3d, 20.0, dtype=jnp.float64)),
        S=state.S.replace(data=jnp.full(native_3d, 30.0, dtype=jnp.float64)),
        u=state.u.replace(data=jnp.zeros((nj, ni + 1, _TSUNAMI_NLEV), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.zeros((nj + 1, ni, _TSUNAMI_NLEV), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(ssh, dtype=jnp.float64)),
        uu_b=state.uu_b.replace(data=jnp.zeros((nj, ni + 1), dtype=jnp.float64)),
        vv_b=state.vv_b.replace(data=jnp.zeros((nj + 1, ni), dtype=jnp.float64)),
    )
    n_e = _resolved_auto_substeps(grid, np.asarray(wet) * _TSUNAMI_H_M, nl.rn_Dt_s)
    if n_e != nl.nn_e_resolved:
        raise ValueError(
            f"TSUNAMI ln_bt_auto resolves nn_e = {n_e}, card states "
            f"{nl.nn_e_resolved}")
    # Carrier: the VORTEX flux-form identity, whose barotropic arm is the
    # closest existing one (EEN dyn_cor_2D, transport-averaged flux-form
    # window).  Its 3-D program is NOT TSUNAMI's -- that is B1.
    base = _model_config(
        barotropic_time_filter="nemo_boxcar1_ab3",   # nn_bt_flt = 1, ln_bt_fw = T
        n_barotropic_substeps=nl.nn_e_resolved,
        bbl_adv_option=0, bbl_gamma_s=0.0,
        bbl_diffusive_option=0, bbl_aht_m2_s=0.0,
        whole_step_identity="vortex_flux_up3_een",
        tke_langmuir_evaluation=None,
    )
    model_config = base._replace(
        eos="nemo_seos", eos_nemo_seos=_TSUNAMI_SEOS, eos_depth="insitu",
        adaptive_implicit_vertadv=nl.ln_zad_Aimp,
        K_h=0.0,                                  # ln_traldf_OFF
        A_v=nl.rn_avm0_m2_s, K_v=nl.rn_avt0_m2_s,  # ln_zdfcst, namelist_ref
        lateral_viscosity=base.lateral_viscosity._replace(A_h=0.0),  # ln_dynldf_OFF
        bottom_drag=base.bottom_drag._replace(bottom_drag_r=0.0),    # ln_drg_OFF
    )
    recipe = NEMORecipe(
        model_config=model_config,
        physics_config=model_config.physics,
        grid=grid, z_coord=z_coord, land_mask=wet, initial_state=state,
    )
    card = NEMOTestcaseCard(
        "TSUNAMI-zco", recipe, nl.rn_Dt_s, nl.nn_itend, 1, 0, 0, 0.0, 0.0,
        unmeasured_features=TSUNAMI_UNMEASURED,
    )
    validate_nemo_testcase_card(card)
    return card


def _validate_tsunami_card(card: NEMOTestcaseCard) -> None:
    """TSUNAMI's own validator branch; refuses any drift from the deck."""
    nl = TSUNAMI_NAMELIST
    cfg = card.recipe.model_config
    if card.unmeasured_features != TSUNAMI_UNMEASURED:
        raise ValueError("TSUNAMI-zco must declare exactly TSUNAMI_UNMEASURED")
    on = [k for k in _TSUNAMI_ABSENT_SPG_TERMS if getattr(nl, k)]
    if on:
        raise ValueError(f"TSUNAMI-zco carries no term for {on}")
    if (card.dt_s, card.n_steps) != (nl.rn_Dt_s, nl.nn_itend):
        raise ValueError("TSUNAMI-zco rn_Dt/nn_itend disagree with the deck")
    if (card.recipe.initial_state.uu_b is None
            or card.recipe.initial_state.vv_b is None):
        raise ValueError("TSUNAMI-zco requires the prognostic uu_b/vv_b pair")
    got = (cfg.barotropic.barotropic_time_filter,
           cfg.barotropic.n_barotropic_substeps)
    if got != ("nemo_boxcar1_ab3", nl.nn_e_resolved):
        raise ValueError(f"TSUNAMI-zco barotropic filter/substeps {got!r}")
    if cfg.eos != "nemo_seos" or cfg.eos_nemo_seos != _TSUNAMI_SEOS:
        raise ValueError("TSUNAMI-zco requires namelist_ref's S-EOS set")
    if (cfg.lateral_viscosity.A_h, cfg.K_h,
            cfg.bottom_drag.bottom_drag_r) != (0.0, 0.0, 0.0):
        raise ValueError("TSUNAMI-zco has no lateral mixing and no drag")
    if (cfg.A_v, cfg.K_v) != (nl.rn_avm0_m2_s, nl.rn_avt0_m2_s):
        raise ValueError("TSUNAMI-zco zdfcst coefficients disagree")
    if cfg.adaptive_implicit_vertadv:
        raise ValueError("TSUNAMI-zco resolves ln_zad_Aimp = .false.")
    if cfg.barotropic.barotropic_coriolis != "een_metric":
        raise ValueError("TSUNAMI-zco requires the EEN barotropic Coriolis")
    if not np.all(np.asarray(card.recipe.land_mask) == 1.0):
        raise ValueError("TSUNAMI-zco is fully wet (usrdef_zgr.F90:187-189)")
    f0 = tsunami_horizontal_coordinates()["f0"]
    for name in ("f_T", "f_u", "f_v", "ff_f"):
        if not np.all(np.asarray(getattr(card.recipe.grid, name)) == f0):
            raise ValueError(f"TSUNAMI-zco {name} is not the f-plane f0")


def validate_nemo_testcase_card(card: NEMOTestcaseCard) -> None:
    """Reject any card composition not exercised by its named oracle run."""
    if card.transcendentals != "libm":
        raise ValueError(
            f"{card.case} requires scalar-libm certification transcendentals, "
            f"got {card.transcendentals!r}"
        )
    if card.case == "TSUNAMI-zco":
        _validate_tsunami_card(card)
        return
    expected = {
        "LOCK_EXCHANGE-zco": ("nemo_ab3am4", 1, 0, 0.0, 0, 0.0),
        "OVERFLOW-zps": ("nemo_boxcar1_ab3", 3, 2, 20.0, 0, 1000.0),
        "GYRE-zco": ("nemo_ab3am4", 50, 0, 0.0, 0, 0.0),
        "ORCA2-zps": ("nemo_ab3am4", 65, 0, 0.0, 1, 1000.0),
        "VORTEX-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        # The vector-EEN deck differs from the flux deck only in the momentum
        # scheme set, so every row this table checks is the same row.
        "VORTEX_VEC-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        # Decision 74's refined rungs.  nn_e = 48 is UNCHANGED by NEMO's own
        # child namelist (1_namelist_cfg:222), so the barotropic substep count
        # is the same row at every resolution -- it does not follow rn_Dt.
        "VORTEX-15km-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        "VORTEX_VEC-15km-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        "VORTEX-10km-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        # Decision 88's seamount pair: the SAME 30 km deck, so the SAME row.
        "VORTEX_SMT-zps": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        "VORTEX_SMT_VEC-zps": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        # Decision 93's mini-ladder rung 1: the SAME deck with namzdf moved
        # to ORCA2 rung 0, so every row this table checks is the same row.
        "VORTEX_SMT1_VEC-zps": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        # Rung 2 adds namdrg's linear bottom drag; the barotropic filter,
        # substep count and BBL block are untouched, so the same row again.
        "VORTEX_SMT2_VEC-zps": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        "VORTEX_SMT3_VEC-zps": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
        "VORTEX_VEC-10km-zco": ("nemo_ab3am4", 48, 0, 0.0, 0, 0.0),
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
    if card.case.startswith(
            ("VORTEX-", "VORTEX_VEC-", "VORTEX_SMT-", "VORTEX_SMT_VEC-",
             "VORTEX_SMT1_VEC-", "VORTEX_SMT2_VEC-", "VORTEX_SMT3_VEC-")):
        # Decision 88's seamount cards run the SAME two momentum decks; every
        # switch this branch checks is the same switch, so they are checked
        # by it rather than by a second copy of it.
        vector = card.case.startswith(
            ("VORTEX_VEC-", "VORTEX_SMT_VEC-", "VORTEX_SMT1_VEC-",
             "VORTEX_SMT2_VEC-", "VORTEX_SMT3_VEC-"))
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
        # usrdef_zgr.F90 closes the box, so exactly the two outermost rows
        # and columns are land at EVERY rung of the resolution ladder.  The
        # expected count is taken from the card's own mask shape rather than
        # hard-coded at 61, which would silently exempt the refined rungs.
        wet_rows = np.count_nonzero(np.any(wet, axis=1))
        if wet_rows != np.shape(wet)[0] - 2:
            raise ValueError(
                f"{card.case} requires the {np.shape(wet)[0] - 2} wet rows of "
                f"its closed {np.shape(wet)[0]}x{np.shape(wet)[1]} box, "
                f"got {wet_rows}")
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
        # EVERY VORTEX card but rung SMT-2 resolves ln_drg_OFF
        # (namelist_cfg:114), so none of them may carry a drag law -- checked
        # here, outside the rung block, because the older bottom_drag_r test
        # above passes any NEMO scheme whose legacy rate happens to be zero.
        if card.case not in ("VORTEX_SMT2_VEC-zps", "VORTEX_SMT3_VEC-zps") and (
                cfg.bottom_drag.bottom_drag_scheme != "legacy"
                or cfg.zdf_drag_in_matrix
                or cfg.barotropic_drag_substep):
            raise ValueError(
                f"{card.case} resolves ln_drg_OFF (namelist_cfg:114); "
                "only rung SMT-2 carries a drag law")
        # Decision 93 rung SMT-1 moves exactly this pair, and the enhanced
        # vertical diffusion that rung 0 runs beside it; every other card on
        # this identity keeps the shipped VORTEX namzdf block.
        if card.case in ("VORTEX_SMT1_VEC-zps", "VORTEX_SMT2_VEC-zps",
                          "VORTEX_SMT3_VEC-zps"):
            if (cfg.A_v, cfg.K_v) != (_SMT1_RN_AVM0, _SMT1_RN_AVT0):
                raise ValueError(
                    f"{card.case} requires ORCA2 rung 0's rn_avm0="
                    f"{_SMT1_RN_AVM0} and rn_avt0={_SMT1_RN_AVT0}")
            _ph = cfg.physics
            if _ph is None or _ph.convection.scheme != "enhanced_diffusion":
                raise ValueError(
                    f"{card.case} requires ln_zdfevd=.true. as an "
                    "explicit enhanced-vertical-diffusion selection")
            _ed = _ph.convection.enhanced_diffusion
            if (_ed.K_conv, _ed.nu_conv, _ed.K_bg, _ed.nu_bg) != (
                    _SMT1_RN_EVD, 0.0, 0.0, 0.0):
                raise ValueError(
                    f"{card.case} requires rn_evd=100 on the tracer "
                    "arm and nn_evdm=0 (no convective viscosity)")
            if (_ed.n2_threshold, _ed.two_level_trigger) != (-1.0e-12, True):
                raise ValueError(
                    f"{card.case} requires zdfevd's own trigger, "
                    "MIN(rn2, rn2b) <= -1.e-12 (zdfevd.F90:93)")
            if _ph.vertical_mixing.scheme != "none":
                raise ValueError(
                    f"{card.case} resolves ln_zdfcst with a uniform "
                    "background; a closure here would be a second one")
            # Rung SMT-2 and ONLY rung SMT-2 carries namdrg's linear drag.
            _bd = cfg.bottom_drag
            _want_lin = card.case in ("VORTEX_SMT2_VEC-zps",
                                      "VORTEX_SMT3_VEC-zps")
            if _want_lin:
                if _bd.bottom_drag_scheme != "nemo_linear":
                    raise ValueError(
                        f"{card.case} requires ORCA2 rung 0's linear "
                        "bottom drag (ln_lin, rung-0 namelist_cfg:270), got "
                        f"bottom_drag_scheme={_bd.bottom_drag_scheme!r}")
                if (_bd.bottom_drag_cd0, _bd.bottom_drag_uc0) != (
                        _SMT2_RN_CD0, _SMT2_RN_UC0):
                    raise ValueError(
                        f"{card.case} requires rn_Cd0="
                        f"{_SMT2_RN_CD0} (namelist_ref:834) and rn_Uc0="
                        f"{_SMT2_RN_UC0} (namelist_ref:835)")
                if (_bd.bottom_drag_bbl_thickness,
                        _bd.bottom_drag_bg_velocity,
                        _bd.bottom_drag_r) != (0.0, 0.0, 0.0):
                    raise ValueError(
                        f"{card.case} applies the drag to the bottom "
                        "cell alone, with no boost and no legacy rate "
                        "(ln_boost=.false., namelist_ref:839)")
                if not (cfg.zdf_drag_in_matrix and cfg.zdf_baroclinic_only
                        and cfg.barotropic_drag_substep):
                    raise ValueError(
                        f"{card.case} resolves ln_drgimp=.true. "
                        "(namelist_ref:817) with ln_dynspg_ts=.true., which "
                        "is one composition: the implicit bottom-cell "
                        "diagonal, the baroclinic-only solve and the frozen "
                        "rCdU_bot in the external-mode substeps")
            if card.case == "VORTEX_SMT3_VEC-zps":
                gm = cfg.gm_redi
                if gm is None:
                    raise ValueError(
                        "VORTEX_SMT3_VEC-zps requires ORCA2 rung 0's "
                        "laplacian isoneutral tracer diffusion")
                expected_gm = {
                    "kappa_GM": 0.0,
                    "kappa_Redi": 270.0,
                    "S_max": 0.01,
                    "slope_scheme": "nemo_iso_lap",
                    "slope_density": "neutral",
                    "slope_limit": "nemo_cap",
                    "slope_positions": "nemo_native",
                    "msc_stabilize": True,
                    "implicit_K33": True,
                    "kappa_redi_horizontal_evaluation":
                        "nemo_metric_literal",
                    "kappa_redi_diffusive_velocity": _SMT3_RN_UD,
                }
                for name, expected_value in expected_gm.items():
                    if getattr(gm, name) != expected_value:
                        raise ValueError(
                            "VORTEX_SMT3_VEC-zps requires the resolved "
                            f"namtra_ldf value {name}={expected_value!r}, got "
                            f"{getattr(gm, name)!r}")
                if cfg.K_h != 0.0:
                    raise ValueError(
                        "VORTEX_SMT3_VEC-zps runs isoneutral diffusion, not "
                        "a second geopotential K_h operator")
            elif cfg.gm_redi is not None:
                raise ValueError(
                    f"{card.case} resolves ln_traldf_OFF before SMT-3")
        elif (cfg.A_v, cfg.K_v) != (1.0e-4, 0.0):
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
        # Decision 74's resolution ladder (operator note BZ).
        "VORTEX-15km-zco": lambda: build_vortex_zco_card("flux", "15km"),
        "VORTEX_VEC-15km-zco": lambda: build_vortex_zco_card("vector", "15km"),
        "VORTEX-10km-zco": lambda: build_vortex_zco_card("flux", "10km"),
        "VORTEX_VEC-10km-zco": lambda: build_vortex_zco_card("vector", "10km"),
        # Decision 88: the same two momentum decks over a Gaussian seamount
        # with z partial bottom cells.
        "VORTEX_SMT-zps": build_vortex_smt_zps_card,
        "VORTEX_SMT_VEC-zps": lambda: build_vortex_smt_zps_card("vector"),
        # Decision 93's seamount mini-ladder, rung 1 (vector deck only).
        "VORTEX_SMT1_VEC-zps": lambda: build_vortex_smt_zps_card(
            "vector", "smt1"),
        # Rung 2: the same deck with namdrg's linear bottom drag.
        "VORTEX_SMT2_VEC-zps": lambda: build_vortex_smt_zps_card(
            "vector", "smt2"),
        "VORTEX_SMT3_VEC-zps": lambda: build_vortex_smt_zps_card(
            "vector", "smt3"),
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
    "apply_orca2_hand_alterations",
    "build_orca2_initial_ts",
    "build_orca2_zps_card",
    "build_vortex_zco_card",
    "build_vortex_smt_zps_card",
    "build_tsunami_zco_card",
    "tsunami_horizontal_coordinates",
    "tsunami_initial_ssh",
    "TSUNAMI_NAMELIST",
    "TSUNAMI_UNMEASURED",
    "vortex_smt_bathymetry",
    "vortex_smt_partial_cell_geometry",
    "validate_vortex_resolution",
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
