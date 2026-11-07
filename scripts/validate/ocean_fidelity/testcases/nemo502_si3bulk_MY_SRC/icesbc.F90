MODULE icesbc
   !!======================================================================
   !!                       ***  MODULE  icesbc  ***
   !! Sea-Ice :   air-ice sbc fields
   !!=====================================================================
   !! History :  4.0  !  2017-08  (C. Rousset)       Original code
   !!            4.0  !  2018     (many people)      SI3 [aka Sea Ice cube]
   !!----------------------------------------------------------------------
#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3' :                                     SI3 sea-ice model
   !!----------------------------------------------------------------------
   USE par_ice        ! SI3 parameters
   USE ice            ! sea-ice: variables
   USE sbc_oce        ! Surface boundary condition: ocean fields
   USE sbc_ice        ! Surface boundary condition: ice   fields
   USE usrdef_sbc     ! Surface boundary condition: user defined
   USE sbcblk         ! Surface boundary condition: bulk
   USE sbccpl         ! Surface boundary condition: coupled interface
   USE icealb         ! sea-ice: albedo
   !
   USE in_out_manager ! I/O manager
   USE iom            ! I/O manager library
   USE lib_mpp        ! MPP library
   USE lbclnk         ! lateral boundary conditions (or mpp links)
   USE timing         ! Timing
   USE fldread        !!GS: needed by agrif

   IMPLICIT NONE
   PRIVATE

   !                                     !!** ice-surface boundary conditions namelist (namsbc) **
   INTEGER ::   nn_flxdist       ! Redistribute heat flux over ice categories
   !                             !   =-1  Do nothing (needs N(cat) fluxes)
   !                             !   = 0  Average N(cat) fluxes then apply the average over the N(cat) ice
   !                             !   = 1  Average N(cat) fluxes then redistribute over the N(cat) ice using T-ice and albedo sensitivity
   !                             !   = 2  Redistribute a single flux over categories

   PUBLIC ice_sbc_tau   ! called by icestp.F90
   PUBLIC ice_sbc_flx   ! called by icestp.F90
   PUBLIC ice_sbc_init  ! called by icestp.F90

   INTEGER, SAVE :: num_l3bulk = -1
   LOGICAL, SAVE :: ll_l3bulk_opened = .FALSE.

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "read_nml_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/ICE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ice_sbc_tau( kt, ksbc, utau_ice, vtau_ice )
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE ice_sbc_tau  ***
      !!
      !! ** Purpose : provide surface boundary condition for sea ice (momentum)
      !!
      !! ** Action  : It provides the following fields:
      !!              utau_ice, vtau_ice : surface ice stress (U- & V-points) [N/m2]
      !!-------------------------------------------------------------------
      INTEGER                     , INTENT(in   ) ::   kt                   ! ocean time step
      INTEGER                     , INTENT(in   ) ::   ksbc                 ! type of sbc flux
      REAL(wp), DIMENSION(jpi,jpj), INTENT(  out) ::   utau_ice, vtau_ice   ! air-ice stress   [N/m2]
      !!
      INTEGER  ::   ji, jj                 ! dummy loop index
      REAL(wp), DIMENSION(A2D(nn_hls)) ::   zutau_ice, zvtau_ice   ! must have the same shape as utau_ice for sbc_cpl_ice_tau
      !!-------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('icesbc')
      !
      IF( kt == nit000 .AND. lwp ) THEN
         WRITE(numout,*)
         WRITE(numout,*)'ice_sbc_tau: Surface boundary condition for sea ice (momentum)'
         WRITE(numout,*)'~~~~~~~~~~~~~~~'
      ENDIF
      !
      SELECT CASE( ksbc )
         !
      CASE( jp_usr     )                 !--- User defined formulation
         !
         CALL usrdef_sbc_ice_tau( kt )
         !
      CASE( jp_blk     )                 !--- Forced formulation
         !
         CALL blk_ice_1( sf(jp_wndi)%fnow(:,:,1), sf(jp_wndj)%fnow(:,:,1), theta_air_zt(:,:), q_air_zt(:,:), & ! <<== in
            &            sf(jp_slp )%fnow(:,:,1), tm_su(:,:),                                                & ! <<== in
            &            putaui=utau_ice, pvtaui=vtau_ice )                                                    ! ==>> out
         CALL l3bulk_dump_tau( kt, utau_ice, vtau_ice )
         !
         !CASE( jp_abl    )              !--- ABL formulation (utau_ice & vtau_ice are computed in ablmod)
         !
      CASE( jp_purecpl )                 !--- Coupled formulation
         !
         CALL sbc_cpl_ice_tau( utau_ice, vtau_ice )
         !
      END SELECT
      !
      IF( ln_mixcpl) THEN                !--- Case of a mixed Bulk/Coupled formulation
         !
         CALL sbc_cpl_ice_tau( zutau_ice , zvtau_ice )
         !
         DO_2D( 0, 0, 0, 0 )
            utau_ice(ji,jj) = utau_ice(ji,jj) * xcplmask(ji,jj,0) + zutau_ice(ji,jj) * ( 1. - xcplmask(ji,jj,0) )
            vtau_ice(ji,jj) = vtau_ice(ji,jj) * xcplmask(ji,jj,0) + zvtau_ice(ji,jj) * ( 1. - xcplmask(ji,jj,0) )
         END_2D
         !
      ENDIF
      !
      CALL lbc_lnk( 'icesbc', utau_ice, 'T', -1.0_wp, vtau_ice, 'T', -1.0_wp )
      !
      IF( ln_timing )   CALL timing_stop('icesbc')
      !
   END SUBROUTINE ice_sbc_tau


   SUBROUTINE ice_sbc_flx( kt, ksbc )
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE ice_sbc_flx  ***
      !!
      !! ** Purpose : provide surface boundary condition for sea ice (flux)
      !!
      !! ** Action  : It provides the following fields used in sea ice model:
      !!                emp_oce , emp_ice                        = E-P over ocean and sea ice                    [Kg/m2/s]
      !!                sprecip                                  = solid precipitation                           [Kg/m2/s]
      !!                evap_ice                                 = sublimation                                   [Kg/m2/s]
      !!                qsr_tot , qns_tot                        = solar & non solar heat flux (total)           [W/m2]
      !!                qsr_ice , qns_ice                        = solar & non solar heat flux over ice          [W/m2]
      !!                dqns_ice                                 = non solar  heat sensistivity                  [W/m2]
      !!                qemp_oce, qemp_ice, qprec_ice, qevap_ice = sensible heat (associated with evap & precip) [W/m2]
      !!            + these fields
      !!                qsb_ice_bot                              = sensible heat at the ice bottom               [W/m2]
      !!                fhld, qlead                              = heat budget in the leads                      [W/m2]
      !!            + some fields that are not used outside this module:
      !!                qla_ice                                  = latent heat flux over ice                     [W/m2]
      !!                dqla_ice                                 = latent heat sensistivity                      [W/m2]
      !!                tprecip                                  = total  precipitation                          [Kg/m2/s]
      !!                alb_ice                                  = albedo above sea ice
      !!-------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt     ! ocean time step
      INTEGER, INTENT(in) ::   ksbc   ! flux formulation (user defined, bulk or Pure Coupled)
      !!--------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('icesbc')

      IF( kt == nit000 .AND. lwp ) THEN
         WRITE(numout,*)
         WRITE(numout,*)'ice_sbc_flx: Surface boundary condition for sea ice (flux)'
         WRITE(numout,*)'~~~~~~~~~~~~~~~'
      ENDIF
      !                     !== ice albedo ==!
      CALL ice_alb( ln_pnd_alb, t_su(:,:,:), h_i(:,:,:), h_s(:,:,:), a_ip_eff(:,:,:), &   ! <<== in
         &                                  h_ip(:,:,:), h_il(:,:,:), cloud_fra(:,:), &   ! <<== in
         &                                                             alb_ice(:,:,:) )   ! ==>> out
      !
      SELECT CASE( ksbc )   !== fluxes over sea ice ==!
      !
      CASE( jp_usr )              !--- user defined formulation
         !
                                  CALL usrdef_sbc_ice_flx( kt, h_s, h_i )
         !
      CASE( jp_blk, jp_abl )      !--- bulk formulation & ABL formulation
         !
                                  CALL blk_ice_2( t_su(:,:,:), h_s(:,:,:), h_i(:,:,:),                &   ! <<== in
                                     &            alb_ice(:,:,:), theta_air_zt(:,:), q_air_zt(:,:),   &   ! <<== in
                                     &            sf(jp_slp)%fnow(:,:,1), sf(jp_qlw)%fnow(:,:,1),     &   ! <<== in
                                     &            precip, sf(jp_snow)%fnow(:,:,1) )                       ! <<== in
                                  CALL l3bulk_dump_flux( kt, 1 )
                                  !
         IF( ln_mixcpl        )   CALL sbc_cpl_ice_flx( kt, picefr=at_i_b(:,:), palbi=alb_ice(:,:,:), &
                                     &                      psst=sst_m(:,:), pist=t_su(:,:,:),  &
                                     &                      phs=h_s(:,:,:), phi=h_i(:,:,:) )
         !
         IF( nn_flxdist /= -1 )   CALL ice_flx_dist( nn_flxdist, at_i(:,:), a_i(:,:,:), t_su(:,:,:), alb_ice(:,:,:), &   ! <<== in
            &                                                    qns_ice(:,:,:), qsr_ice(:,:,:), dqns_ice(:,:,:),    &   ! ==>> inout
            &                                                    evap_ice(:,:,:), devap_ice(:,:,:) )                     ! ==>> inout
         !
         !                        !    compute conduction flux and surface temperature (as in Jules surface module)
         IF( ln_cndflx .AND. .NOT.ln_cndemulate ) THEN
                                  CALL blk_ice_qcn( ln_virtual_itd, t_bo(:,:), h_s(:,:,:), h_i(:,:,:), &   ! <<== in
                                     &                              qcn_ice(:,:,:), qml_ice(:,:,:),          &   ! ==>> out
                                     &                              qns_ice(:,:,:), t_su(:,:,:) )             ! ==>> inout
         ENDIF
         !
      CASE ( jp_purecpl )         !--- coupled formulation
         !
                                  CALL sbc_cpl_ice_flx( kt, picefr=at_i_b(:,:), palbi=alb_ice(:,:,:), psst=sst_m(:,:), &
                                     &                      pist=t_su(:,:,:), phs=h_s(:,:,:), phi=h_i(:,:,:) )
                                  !
         IF( nn_flxdist /= -1 )   CALL ice_flx_dist( nn_flxdist, at_i(:,:), a_i(:,:,:), t_su(:,:,:), alb_ice(:,:,:), &   ! <<== in
            &                                                    qns_ice(:,:,:), qsr_ice(:,:,:), dqns_ice(:,:,:),    &   ! ==>> inout
            &                                                    evap_ice(:,:,:), devap_ice(:,:,:) )                     ! ==>> inout
         !
      END SELECT
!!$      CALL lbc_lnk( 'icesbc', t_su, 'T', 1.0_wp ) ! clem: t_su is needed for Met-Office only => necessary?
      !
      !                     !== some fluxes at the ice-ocean interface and in the leads
      CALL ice_flx_other
      CALL l3bulk_dump_flux( kt, 2 )
      !
      IF( ln_timing )   CALL timing_stop('icesbc')
      !
   END SUBROUTINE ice_sbc_flx


   SUBROUTINE l3bulk_header( kt, kstage, knval )
      INTEGER, INTENT(in) :: kt, kstage, knval
      CHARACTER(LEN=16), PARAMETER :: cmagic = 'NEMO_L3BULK_001 '
      IF( .NOT. ll_l3bulk_opened ) THEN
         OPEN( NEWUNIT=num_l3bulk, FILE='oracle_si3_bulk_operands.bin', STATUS='REPLACE', &
            &  ACCESS='STREAM', FORM='UNFORMATTED', ACTION='WRITE' )
         ll_l3bulk_opened = .TRUE.
      ENDIF
      WRITE(num_l3bulk) cmagic
      WRITE(num_l3bulk) 1, kt, kstage, knval, 64
   END SUBROUTINE l3bulk_header


   SUBROUTINE l3bulk_dump_tau( kt, putau, pvtau )
      INTEGER, INTENT(in) :: kt
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) :: putau, pvtau
      REAL(wp) :: z(17)
      INTEGER :: ji, jj
      ji = Nis0 ; jj = Njs0
      z = (/ sf(jp_wndi)%fnow(ji,jj,1), sf(jp_wndj)%fnow(ji,jj,1), &
         & theta_air_zt(ji,jj), q_air_zt(ji,jj), sf(jp_slp)%fnow(ji,jj,1), tm_su(ji,jj), &
         & rhoa(ji,jj), wndm_ice(ji,jj), Cd_ice(ji,jj), Ch_ice(ji,jj), Ce_ice(ji,jj), &
         & theta_zu_i(ji,jj), q_zu_i(ji,jj), fr_i(ji,jj), smask0(ji,jj), putau(ji,jj), pvtau(ji,jj) /)
      CALL l3bulk_header( kt, 0, SIZE(z) )
      WRITE(num_l3bulk) z
      FLUSH(num_l3bulk)
   END SUBROUTINE l3bulk_dump_tau


   SUBROUTINE l3bulk_dump_flux( kt, kstage )
      INTEGER, INTENT(in) :: kt, kstage
      REAL(wp) :: z1(50), z2(39)
      INTEGER :: ji, jj
      ji = Nis0 ; jj = Njs0
      IF( kstage == 1 ) THEN
         z1 = (/ t_su(ji,jj,1), h_s(ji,jj,1), h_i(ji,jj,1), a_ip_eff(ji,jj,1), h_ip(ji,jj,1), &
            & h_il(ji,jj,1), cloud_fra(ji,jj), theta_air_zt(ji,jj), q_air_zt(ji,jj), &
            & sf(jp_slp)%fnow(ji,jj,1), sf(jp_qlw)%fnow(ji,jj,1), precip(ji,jj), &
            & sf(jp_snow)%fnow(ji,jj,1), qsr(ji,jj), rhoa(ji,jj), wndm_ice(ji,jj), &
            & Ch_ice(ji,jj), Ce_ice(ji,jj), at_i_b(ji,jj), a_i_b(ji,jj,1), sst_m(ji,jj), &
            & qns_oce(ji,jj), qsr_oce(ji,jj), emp(ji,jj), fr_i(ji,jj), &
            & alb_ice(ji,jj,1), qsr_ice(ji,jj,1), qla_ice(ji,jj,1), dqla_ice(ji,jj,1), &
            & qns_ice(ji,jj,1), dqns_ice(ji,jj,1), evap_ice(ji,jj,1), devap_ice(ji,jj,1), &
            & tprecip(ji,jj), sprecip(ji,jj), emp_oce(ji,jj), emp_ice(ji,jj), emp_tot(ji,jj), &
            & qemp_oce(ji,jj), qemp_ice(ji,jj), qns_tot(ji,jj), qsr_tot(ji,jj), &
            & qprec_ice(ji,jj), qevap_ice(ji,jj,1), qtr_ice_top(ji,jj,1), &
            & l3_qsat_ice, l3_theta_ice, l3_qlw_ice, l3_qsb_ice, l3_dqlw_ice /)
         CALL l3bulk_header( kt, 1, SIZE(z1) )
         WRITE(num_l3bulk) z1
      ELSE
         z2 = (/ at_i(ji,jj), at_i_b(ji,jj), vt_i(ji,jj), &
            & u_ice(ji,jj), u_ice(ji-1,jj), v_ice(ji,jj), v_ice(ji,jj-1), &
            & ssu_m(ji,jj), ssu_m(ji-1,jj), ssv_m(ji,jj), ssv_m(ji,jj-1), drag_io(ji,jj), &
            & utau(ji,jj), vtau(ji,jj), frq_m(ji,jj), qsr_oce(ji,jj), qns_oce(ji,jj), &
            & qemp_oce(ji,jj), e3t_m(ji,jj), sst_m(ji,jj), t_bo(ji,jj), &
            & rDt_ice, r1_Dt_ice, rho0, rcp, rt0, epsi10, rn_amax_2d(ji,jj), smask0(ji,jj), &
            & MERGE(1._wp,0._wp,ln_icedyn), MERGE(1._wp,0._wp,ln_Cx_ice_frm), REAL(nn_frm,wp), &
            & MERGE(1._wp,0._wp,ln_leadhfx), MERGE(1._wp,0._wp,ln_icedO), &
            & MERGE(1._wp,0._wp,ln_icedH), rn_Cd_io, qsb_ice_bot(ji,jj), fhld(ji,jj), qlead(ji,jj) /)
         CALL l3bulk_header( kt, 2, SIZE(z2) )
         WRITE(num_l3bulk) z2
      ENDIF
      FLUSH(num_l3bulk)
   END SUBROUTINE l3bulk_dump_flux


   SUBROUTINE ice_flx_dist( k_flxdist, pat_i, pa_i, ptn_ice, palb_ice, &
      &                                pqns_ice, pqsr_ice, pdqn_ice, pevap_ice, pdevap_ice )
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE ice_flx_dist  ***
      !!
      !! ** Purpose :   update the ice surface boundary condition by averaging
      !!              and/or redistributing fluxes on ice categories
      !!
      !! ** Method  :   average then redistribute
      !!
      !! ** Action  :   depends on k_flxdist
      !!                = -1  Do nothing (needs N(cat) fluxes)
      !!                =  0  Average N(cat) fluxes then apply the average over the N(cat) ice
      !!                =  1  Average N(cat) fluxes then redistribute over the N(cat) ice
      !!                                                 using T-ice and albedo sensitivity
      !!                =  2  Redistribute a single flux over categories
      !!-------------------------------------------------------------------
      INTEGER                             , INTENT(in   ) ::   k_flxdist  ! redistributor
      REAL(wp), DIMENSION(A2D(nn_hls)    ), INTENT(in   ) ::   pat_i      ! ice concentration
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in   ) ::   pa_i       ! ice concentration
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in   ) ::   ptn_ice    ! ice surface temperature
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(in   ) ::   palb_ice   ! ice albedo
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(inout) ::   pqns_ice   ! non solar flux
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(inout) ::   pqsr_ice   ! net solar flux
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(inout) ::   pdqn_ice   ! non solar flux sensitivity
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(inout) ::   pevap_ice  ! sublimation
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(inout) ::   pdevap_ice ! sublimation sensitivity
      !
      INTEGER  ::   ji,jj,jl      ! dummy loop index
      !
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   z_qsr_m   ! Mean solar heat flux over all categories
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   z_qns_m   ! Mean non solar heat flux over all categories
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   z_evap_m  ! Mean sublimation over all categories
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   z_dqn_m   ! Mean d(qns)/dT over all categories
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   z_devap_m ! Mean d(evap)/dT over all categories
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zalb_m    ! Mean albedo over all categories
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   ztem_m    ! Mean temperature over all categories
      REAL(wp) :: ztmp
      !!----------------------------------------------------------------------

      SELECT CASE( k_flxdist )       !==  averaged on all ice categories  ==!
      !
      CASE( 0 , 1 )
         !
         ALLOCATE( z_qns_m(A2D(0)), z_qsr_m(A2D(0)), z_dqn_m(A2D(0)), z_evap_m(A2D(0)), z_devap_m(A2D(0)) )
         !
         DO_2D( 0, 0, 0, 0 )
            IF( pat_i(ji,jj) > 0._wp ) THEN
               ztmp =  1._wp / pat_i(ji,jj)
               z_qns_m  (ji,jj) = SUM( pa_i(ji,jj,:) * pqns_ice  (ji,jj,:) ) * ztmp
               z_qsr_m  (ji,jj) = SUM( pa_i(ji,jj,:) * pqsr_ice  (ji,jj,:) ) * ztmp
               z_dqn_m  (ji,jj) = SUM( pa_i(ji,jj,:) * pdqn_ice  (ji,jj,:) ) * ztmp
               z_evap_m (ji,jj) = SUM( pa_i(ji,jj,:) * pevap_ice (ji,jj,:) ) * ztmp
               z_devap_m(ji,jj) = SUM( pa_i(ji,jj,:) * pdevap_ice(ji,jj,:) ) * ztmp
            ELSE
               z_qns_m  (ji,jj) = 0._wp
               z_qsr_m  (ji,jj) = 0._wp
               z_dqn_m  (ji,jj) = 0._wp
               z_evap_m (ji,jj) = 0._wp
               z_devap_m(ji,jj) = 0._wp
            ENDIF
         END_2D
         !
         DO jl = 1, jpl
            pqns_ice  (A2D(0),jl) = z_qns_m  (A2D(0))
            pqsr_ice  (A2D(0),jl) = z_qsr_m  (A2D(0))
            pdqn_ice  (A2D(0),jl) = z_dqn_m  (A2D(0))
            pevap_ice (A2D(0),jl) = z_evap_m (A2D(0))
            pdevap_ice(A2D(0),jl) = z_devap_m(A2D(0))
         END DO
         !
         DEALLOCATE( z_qns_m, z_qsr_m, z_dqn_m, z_evap_m, z_devap_m )
         !
      END SELECT
      !
      SELECT CASE( k_flxdist )       !==  redistribution on all ice categories  ==!
      !
      CASE( 1 , 2 )
         !
         ALLOCATE( zalb_m(A2D(0)), ztem_m(A2D(0)) )
         !
         DO_2D( 0, 0, 0, 0 )
            IF( pat_i(ji,jj) > 0._wp ) THEN
               ztmp =  1._wp / pat_i(ji,jj)
               zalb_m(ji,jj) = SUM( pa_i(ji,jj,:) * palb_ice(ji,jj,:) ) * ztmp
               ztem_m(ji,jj) = SUM( pa_i(ji,jj,:) * ptn_ice (ji,jj,:) ) * ztmp
            ELSE
               zalb_m(ji,jj) = 0._wp
               ztem_m(ji,jj) = 0._wp
            ENDIF
         END_2D
         !
         DO jl = 1, jpl
            pqns_ice (A2D(0),jl) = pqns_ice (A2D(0),jl) + pdqn_ice  (A2D(0),jl) * ( ptn_ice(A2D(0),jl) - ztem_m(A2D(0)) )
            pevap_ice(A2D(0),jl) = pevap_ice(A2D(0),jl) + pdevap_ice(A2D(0),jl) * ( ptn_ice(A2D(0),jl) - ztem_m(A2D(0)) )
            pqsr_ice (A2D(0),jl) = pqsr_ice (A2D(0),jl) * ( 1._wp - palb_ice(A2D(0),jl) ) / ( 1._wp - zalb_m(A2D(0)) )
         END DO
         !
         DEALLOCATE( zalb_m, ztem_m )
         !
      END SELECT
      !
   END SUBROUTINE ice_flx_dist


   SUBROUTINE ice_flx_other
      !!-----------------------------------------------------------------------
      !!                   ***  ROUTINE ice_flx_other ***
      !!
      !! ** Purpose :   prepare necessary fields for thermo calculations
      !!
      !! ** Inputs  :   u_ice, v_ice, ssu_m, ssv_m, utau, vtau
      !!                frq_m, qsr_oce, qns_oce, qemp_oce, e3t_m, sst_m
      !! ** Outputs :   qsb_ice_bot, fhld, qlead
      !!-----------------------------------------------------------------------
      INTEGER  ::   ji, jj             ! dummy loop indices
      REAL(wp) ::   zdrag, zfric_u, zqld, zqfr, zqfr_neg, zqfr_pos, zu_io, zv_io, zu_iom1, zv_iom1
      REAL(wp) ::   zswitch
      REAL(wp), PARAMETER ::   zfric_umin = 0._wp       ! lower bound for the friction velocity (cice value=5.e-04)
      REAL(wp), PARAMETER ::   zch        = 0.0057_wp   ! heat transfer coefficient
      REAL(wp), DIMENSION(A2D(0)) ::  zfric, zvel       ! ice-ocean velocity (m/s) and frictional velocity (m2/s2)
      !!-----------------------------------------------------------------------
      !
      ! computation of friction velocity at T points
      IF( ln_icedyn ) THEN
         DO_2D( 0, 0, 0, 0 )
            zu_io   = u_ice(ji  ,jj  ) - ssu_m(ji  ,jj  )
            zu_iom1 = u_ice(ji-1,jj  ) - ssu_m(ji-1,jj  )
            zv_io   = v_ice(ji  ,jj  ) - ssv_m(ji  ,jj  )
            zv_iom1 = v_ice(ji  ,jj-1) - ssv_m(ji  ,jj-1)
            !
            ! FORM-DRAG: for nn_frm = 2 or 3: no impact on ocean-heat transfer coefficient!
            IF( ln_Cx_ice_frm .AND. nn_frm /= 1  ) THEN   ;   zdrag = drag_io(ji,jj)
            ELSE                                          ;   zdrag = rn_Cd_io      ; ENDIF
            !
            zfric(ji,jj) = zdrag * ( 0.5_wp * (  ( zu_io*zu_io + zu_iom1*zu_iom1 )   &   ! add () for NP repro
               &                               + ( zv_io*zv_io + zv_iom1*zv_iom1 ) ) ) * smask0(ji,jj)
            zvel (ji,jj) = 0.5_wp * SQRT( ( u_ice(ji-1,jj  ) + u_ice(ji,jj) ) * ( u_ice(ji-1,jj  ) + u_ice(ji,jj) ) + &
               &                          ( v_ice(ji  ,jj-1) + v_ice(ji,jj) ) * ( v_ice(ji  ,jj-1) + v_ice(ji,jj) ) )
         END_2D
      ELSE      !  if no ice dynamics => transfer directly the atmospheric stress to the ocean
         DO_2D( 0, 0, 0, 0 )
            zfric(ji,jj) = r1_rho0 * SQRT( utau(ji,jj)*utau(ji,jj) + vtau(ji,jj)*vtau(ji,jj) ) * smask0(ji,jj)
            zvel (ji,jj) = 0._wp
         END_2D
      ENDIF
      !
      !--------------------------------------------------------------------!
      ! Partial computation of forcing for the thermodynamic sea ice model
      !--------------------------------------------------------------------!
      DO_2D( 0, 0, 0, 0 )   ! needed for qlead
         IF( at_i(ji,jj) >= epsi10 ) THEN ; zswitch = smask0(ji,jj)
         ELSE                             ; zswitch = 0._wp
         ENDIF
         !
         ! --- Energy received in the lead from atm-oce exchanges, zqld is defined everywhere (J.m-2) --- !
         zqld =  smask0(ji,jj) * rDt_ice *  &
            &    ( ( 1._wp - at_i_b(ji,jj) ) * qsr_oce(ji,jj) * frq_m(ji,jj) +  &
            &      ( 1._wp - at_i_b(ji,jj) ) * qns_oce(ji,jj) + qemp_oce(ji,jj) )

         ! --- Energy needed to bring ocean surface layer until its freezing, zqfr is defined everywhere (J.m-2) --- !
         !     (mostly<0 but >0 if supercooling)
         zqfr     = rho0 * rcp * e3t_m(ji,jj) * ( t_bo(ji,jj) - ( sst_m(ji,jj) + rt0 ) ) * smask0(ji,jj)  ! both < 0 (t_bo < sst) and > 0 (t_bo > sst)
         zqfr_neg = MIN( zqfr , 0._wp )                                                                   ! only < 0
         zqfr_pos = MAX( zqfr , 0._wp )                                                                   ! only > 0

         ! --- Sensible ocean-to-ice heat flux (W/m2) --- !
         !     (mostly>0 but <0 if supercooling)
         zfric_u            = MAX( SQRT( zfric(ji,jj) ), zfric_umin )
         qsb_ice_bot(ji,jj) = zswitch * rho0 * rcp * zch * zfric_u * ( ( sst_m(ji,jj) + rt0 ) - t_bo(ji,jj) )

         ! upper bound for qsb_ice_bot: the heat retrieved from the ocean must be smaller than the heat necessary to reach
         !                              the freezing point, so that we do not have SST < T_freeze
         !                              This implies: qsb_ice_bot(ji,jj) * at_i(ji,jj) * rtdice <= - zqfr_neg
         !                              The following formulation is ok for both normal conditions and supercooling
         qsb_ice_bot(ji,jj) = zswitch * MIN( qsb_ice_bot(ji,jj), - zqfr_neg * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) )

         ! If conditions are always supercooled (such as at the mouth of ice-shelves), then ice grows continuously
         ! ==> stop ice formation by artificially setting up the turbulent fluxes to 0 when volume > 20m (arbitrary)
         IF( ( t_bo(ji,jj) - ( sst_m(ji,jj) + rt0 ) ) > 0._wp .AND. vt_i(ji,jj) >= 20._wp ) THEN
            zqfr               = 0._wp
            zqfr_pos           = 0._wp
            qsb_ice_bot(ji,jj) = 0._wp
         ENDIF
         !
         ! --- Energy Budget of the leads (qlead, J.m-2) --- !
         !     qlead is the energy received from the atm. in the leads.
         !     If warming (zqld >= 0), then the energy in the leads is used to melt ice (bottom melting) => fhld  (W/m2)
         !     If cooling (zqld <  0), then the energy in the leads is used to grow ice in open water    => qlead (J.m-2)
         IF( ( zqld - zqfr ) < 0._wp ) THEN
            fhld (ji,jj) = 0._wp
            ! upper bound for qlead: qlead should be equal to zqld
            !                        but before using this heat for ice formation, we suppose that the ocean cools down till the freezing point.
            !                        The energy for this cooling down is zqfr and freezing point is reached if zqfr = zqld
            !                        so the max heat that can be pulled out of the ocean is zqld - zqfr
            !                        The following formulation is ok for both normal conditions and supercooling
            qlead(ji,jj) = MIN( 0._wp , zqld - zqfr )
         ELSE
            ! upper bound for fhld: fhld should be equal to zqld
            !                        but we have to make sure that this heat will not make the sst drop below the freezing point
            !                        so the max heat that can be pulled out of the ocean is zqld - zqfr_pos
            !                        The following formulation is ok for both normal conditions and supercooling
            fhld (ji,jj) = zswitch * MAX( 0._wp, ( zqld - zqfr_pos ) * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) ) ! divided by at_i since this is (re)multiplied by a_i in icethd_dh.F90
            qlead(ji,jj) = 0._wp
         ENDIF
         !
         ! stop ice formation in open water if ice is very slow (i.e. landfast) and ice concentration reaches its max (minus a threshold at 0.001)
         !                                                                      and ice is thicker than 3 meters (arbitrary)
         !    Note: This threshold is necessary when reading a landfast mask otherwise ice grows up to its limit at 20m
         !          The limit for ice velocity should be 0.5 mm/s as for observations of landfast but it works better with larger values
         !          Hence, we use 5 mm/s
         IF( zvel(ji,jj) <= 5.e-03_wp .AND. at_i(ji,jj) >= (rn_amax_2d(ji,jj)-0.001_wp) .AND. vt_i(ji,jj) >= 3._wp )   qlead(ji,jj) = 0._wp
         !
         ! If the grid cell is almost fully covered by ice (no leads) and ice is thicker than 3 meters (arbitrary)
         ! => stop ice formation in open water
         IF( at_i(ji,jj) >= (1._wp - epsi10) .AND. vt_i(ji,jj) >= 3._wp )   qlead(ji,jj) = 0._wp
         !
         ! If ln_leadhfx is false
         ! => do not use energy of the leads to melt sea-ice
         IF( .NOT.ln_leadhfx )   fhld(ji,jj) = 0._wp
         !
      END_2D

      ! If ln_leadhfx is false
      ! => do not use transmitted solar flux to melt sea-ice (equivalent of setting frq_m=0)
      IF( .NOT.ln_leadhfx )   frq_m(:,:) = 0._wp

      ! In case we bypass open-water ice formation
      IF( .NOT. ln_icedO )  qlead(:,:) = 0._wp
      ! In case we bypass growing/melting from top and bottom
      IF( .NOT. ln_icedH ) THEN
         qsb_ice_bot(:,:) = 0._wp
         fhld       (:,:) = 0._wp
      ENDIF

   END SUBROUTINE ice_flx_other


   SUBROUTINE ice_sbc_init
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE ice_sbc_init  ***
      !!
      !! ** Purpose :   Physical constants and parameters linked to the ice dynamics
      !!
      !! ** Method  :   Read the namsbc namelist and check the ice-dynamic
      !!              parameter values called at the first timestep (nit000)
      !!
      !! ** input   :   Namelist namsbc
      !!-------------------------------------------------------------------
      INTEGER ::   ios, ioptio   ! Local integer
      !!
      NAMELIST/namsbc/ rn_Cd_io, nn_snwfra, rn_snwblow, nn_flxdist, ln_cndflx, ln_cndemulate, nn_qtrice
      !!-------------------------------------------------------------------
      !
      READ_NML_REF(numnam_ice,namsbc)
      READ_NML_CFG(numnam_ice,namsbc)
      IF(lwm) WRITE( numoni, namsbc )
      !
      IF(lwp) THEN                     ! control print
         WRITE(numout,*)
         WRITE(numout,*) 'ice_sbc_init: ice parameters for ice dynamics '
         WRITE(numout,*) '~~~~~~~~~~~~~~~~'
         WRITE(numout,*) '   Namelist namsbc:'
         WRITE(numout,*) '      drag coefficient for oceanic stress                       rn_Cd_io      = ', rn_Cd_io
         WRITE(numout,*) '      fraction of ice covered by snow (options 0,1,2)           nn_snwfra     = ', nn_snwfra
         WRITE(numout,*) '      coefficient for ice-lead partition of snowfall            rn_snwblow    = ', rn_snwblow
         WRITE(numout,*) '      Multicategory heat flux formulation                       nn_flxdist    = ', nn_flxdist
         WRITE(numout,*) '      Use conduction flux as surface condition                  ln_cndflx     = ', ln_cndflx
         WRITE(numout,*) '         emulate conduction flux                                ln_cndemulate = ', ln_cndemulate
         WRITE(numout,*) '      solar flux transmitted thru the surface scattering layer  nn_qtrice     = ', nn_qtrice
         WRITE(numout,*) '         = 0  Grenfell and Maykut 1977'
         WRITE(numout,*) '         = 1  Lebrun 2019'
      ENDIF
      !
      IF(lwp) WRITE(numout,*)
      SELECT CASE( nn_flxdist )         ! SI3 Multi-category heat flux formulation
      CASE( -1  )
         IF(lwp) WRITE(numout,*) '   SI3: use per-category fluxes (nn_flxdist = -1) '
      CASE(  0  )
         IF(lwp) WRITE(numout,*) '   SI3: use average per-category fluxes (nn_flxdist = 0) '
      CASE(  1  )
         IF(lwp) WRITE(numout,*) '   SI3: use average then redistribute per-category fluxes (nn_flxdist = 1) '
         IF( ln_cpl )         CALL ctl_stop( 'ice_thd_init: the chosen nn_flxdist for SI3 in coupled mode must be /=1' )
      CASE(  2  )
         IF(lwp) WRITE(numout,*) '   SI3: Redistribute a single flux over categories (nn_flxdist = 2) '
         IF( .NOT. ln_cpl )   CALL ctl_stop( 'ice_thd_init: the chosen nn_flxdist for SI3 in forced mode must be /=2' )
      CASE DEFAULT
         CALL ctl_stop( 'ice_thd_init: SI3 option, nn_flxdist, should be between -1 and 2' )
      END SELECT
      !
   END SUBROUTINE ice_sbc_init

#else
   !!----------------------------------------------------------------------
   !!   Default option :         Empty module         NO SI3 sea-ice model
   !!----------------------------------------------------------------------
#endif

   !!======================================================================
END MODULE icesbc
