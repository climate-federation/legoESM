MODULE icethd_zdf_BL99
   !!======================================================================
   !!                       ***  MODULE icethd_zdf_BL99 ***
   !!   sea-ice: vertical heat diffusion in sea ice (computation of temperatures)
   !!======================================================================
   !! History :       !  2003-02  (M. Vancoppenolle) original 1D code
   !!                 !  2005-06  (M. Vancoppenolle) 3d version
   !!            4.0  !  2018     (many people)      SI3 [aka Sea Ice cube]
   !!----------------------------------------------------------------------
#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3'                                       SI3 sea-ice model
   !!----------------------------------------------------------------------
   !!  ice_thd_zdf_BL99 : vertical diffusion computation
   !!----------------------------------------------------------------------
   USE par_ice        ! SI3 parameters
   USE par_kind, ONLY : wp
   USE par_oce , ONLY : jpij
   USE phycst  
   USE ice1D          ! sea-ice: thermodynamics variables
   USE icevar  , ONLY : ice_var_snwfra, ice_var_enthalpy

   IMPLICIT NONE
   PRIVATE

   PUBLIC   ice_thd_zdf_BL99   ! called by icethd_zdf

   INTEGER, SAVE ::   num_l3zdf = -1
   LOGICAL, SAVE ::   ll_l3zdf_opened = .FALSE.
   INTEGER, SAVE ::   kn_l3zdf_call = 0

   !!----------------------------------------------------------------------
   !! NEMO/ICE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ice_thd_zdf_BL99( k_cnd )
      !!-------------------------------------------------------------------
      !!                ***  ROUTINE ice_thd_zdf_BL99  ***
      !!
      !! ** Purpose : computes the time evolution of snow and sea-ice temperature
      !!              profiles, using the original Bitz and Lipscomb (1999) algorithm
      !!
      !! ** Method  : solves the heat equation diffusion with a Neumann boundary
      !!              condition at the surface and a Dirichlet one at the bottom.
      !!              Solar radiation is partially absorbed into the ice.
      !!              The specific heat and thermal conductivities depend on ice
      !!              salinity and temperature to take into account brine pocket
      !!              melting. The numerical scheme is an iterative Crank-Nicolson
      !!              on a non-uniform multilayer grid in the ice and snow system.
      !!
      !!           The successive steps of this routine are
      !!           1.  initialization of ice-snow layers thicknesses
      !!           2.  Internal absorbed and transmitted radiation
      !!           Then iterative procedure begins
      !!           3.  Thermal conductivity
      !!           4.  Kappa factors
      !!           5.  specific heat in the ice
      !!           6.  eta factors
      !!           7.  surface flux computation
      !!           8.  tridiagonal system terms
      !!           9.  solving the tridiagonal system with Gauss elimination
      !!           Iterative procedure ends according to a criterion on evolution
      !!           of temperature
      !!           10. Fluxes at the interfaces
      !!
      !! ** Inputs / Ouputs : (global commons)
      !!           surface temperature              : t_su_1d
      !!           ice/snow temperatures            : t_i_1d, t_s_1d
      !!           ice salinities                   : sz_i_1d
      !!           number of layers in the ice/snow : nlay_i, nlay_s
      !!           total ice/snow thickness         : h_i_1d, h_s_1d
      !!-------------------------------------------------------------------
      INTEGER, INTENT(in) ::   k_cnd     ! conduction flux (off, on, emulated)
      !
      INTEGER ::   ji, jk         ! spatial loop index
      INTEGER ::   jm             ! current reference number of equation
      INTEGER ::   iconv          ! number of iterations in iterative procedure
      INTEGER ::   iconv_max = 200 ! max number of iterations in iterative procedure
      !                            ! clem: this number is increased from 50 to 200 to (maybe) solve rare convergence pb while atmo coupling

      LOGICAL, DIMENSION(jpij) ::   l_T_converged   ! true when T converges (per grid point)
      !
      REAL(wp) ::   zg1s      =  2._wp        ! for the tridiagonal system
      REAL(wp) ::   zg1       =  2._wp        !
      REAL(wp) ::   zgamma    =  18009._wp    ! for specific heat
      REAL(wp) ::   zbeta     =  0.117_wp     ! for thermal conductivity (could be 0.13)
      REAL(wp) ::   zkimin    =  0.10_wp      ! minimum ice thermal conductivity
      REAL(wp) ::   ztsu_err  =  1.e-5_wp     ! range around which t_su is considered at 0C
      REAL(wp) ::   zdti_bnd  =  1.e-4_wp     ! maximal authorized error on temperature
      REAL(wp) ::   zhs_ssl   =  0.03_wp      ! surface scattering layer in the snow
      REAL(wp) ::   zhi_ssl   =  0.10_wp      ! surface scattering layer in the ice
      REAL(wp) ::   zh_min    =  1.e-3_wp     ! minimum ice/snow thickness for conduction
      REAL(wp) ::   ztmelts                   ! ice melting temperature
      REAL(wp) ::   zdti_max                  ! current maximal error on temperature
      REAL(wp) ::   zcpi                      ! Ice specific heat
      REAL(wp) ::   zhfx_err, zdq             ! diag errors on heat
      !
      REAL(wp), DIMENSION(jpij) ::   zraext_s     ! extinction coefficient of radiation in the snow
      REAL(wp), DIMENSION(jpij) ::   ztsub        ! surface temperature at previous iteration
      REAL(wp), DIMENSION(jpij) ::   zh_i, z1_h_i ! ice layer thickness
      REAL(wp), DIMENSION(jpij) ::   zh_s, z1_h_s ! snow layer thickness
      REAL(wp), DIMENSION(jpij) ::   zqns_ice_b   ! solar radiation absorbed at the surface
      REAL(wp), DIMENSION(jpij) ::   zdqns_ice_b  ! derivative of the surface flux function
      REAL(wp), DIMENSION(jpij,0:nlay_s) ::   zradtr_s    ! Radiation transmited through the snow
      REAL(wp), DIMENSION(jpij,0:nlay_s) ::   zradab_s    ! Radiation absorbed in the snow
      REAL(wp), DIMENSION(jpij,0:nlay_i) ::   zradtr_i    ! Radiation transmitted through the ice
      REAL(wp), DIMENSION(jpij,0:nlay_i) ::   zradab_i    ! Radiation absorbed in the ice
      REAL(wp), DIMENSION(jpij,0:nlay_i) ::   ztcond_i    ! Ice thermal conductivity
      REAL(wp), DIMENSION(jpij,0:nlay_i) ::   ztcond_i_cp ! copy
      REAL(wp), DIMENSION(jpij,nlay_i)   ::   ztiold      ! Old temperature in the ice
      REAL(wp), DIMENSION(jpij,nlay_s)   ::   ztsold      ! Old temperature in the snow
      REAL(wp), DIMENSION(jpij,0:nlay_i) ::   zkappa_i    ! Kappa factor in the ice
      REAL(wp), DIMENSION(jpij,0:nlay_i) ::   zeta_i      ! Eta factor in the ice
      REAL(wp), DIMENSION(jpij,0:nlay_s) ::   zkappa_s    ! Kappa factor in the snow
      REAL(wp), DIMENSION(jpij,0:nlay_s) ::   zeta_s      ! Eta factor in the snow
      REAL(wp), DIMENSION(jpij)          ::   zkappa_comb ! Combined snow and ice surface conductivity
      REAL(wp), DIMENSION(jpij)          ::   zq_ini      ! diag errors on heat
      REAL(wp), DIMENSION(jpij)          ::   zghe        ! G(he), th. conduct enhancement factor, mono-cat
      REAL(wp), DIMENSION(jpij)          ::   za_s_fra    ! ice fraction covered by snow
      REAL(wp), DIMENSION(jpij)          ::   isnow       ! snow presence (1) or not (0)
      REAL(wp), DIMENSION(jpij)          ::   isnow_comb  ! snow presence for met-office
      !
      INTEGER  ::   jm_min    ! reference number of top equation
      INTEGER  ::   jm_max    ! reference number of bottom equation
      REAL(wp) ::   zfnet     ! surface flux function
      REAL(wp), DIMENSION(nlay_i)   ::   ztib        ! Temporary temperature in the ice to check the convergence
      REAL(wp), DIMENSION(nlay_s)   ::   ztsb        ! Temporary temperature in the snow to check the convergence
      REAL(wp), DIMENSION(nlay_i+nlay_s+1)   ::   zindterm    ! 'Ind'ependent term
      REAL(wp), DIMENSION(nlay_i+nlay_s+1)   ::   zindtbis    ! Temporary 'ind'ependent term
      REAL(wp), DIMENSION(nlay_i+nlay_s+1)   ::   zdiagbis    ! Temporary 'dia'gonal term
      REAL(wp), DIMENSION(nlay_i+nlay_s+1,3) ::   ztrid       ! Tridiagonal system terms
      !
      ! Mono-category
      REAL(wp) ::   zepsilon   ! determines thres. above which computation of G(h) is done
      REAL(wp) ::   zhe        ! dummy factor
      REAL(wp) ::   zcnd_i     ! mean sea ice thermal conductivity
      LOGICAL  ::   ll_l3zdf_capture
      !!------------------------------------------------------------------

      kn_l3zdf_call = kn_l3zdf_call + 1
      ll_l3zdf_capture = kn_l3zdf_call == 1 .OR. kn_l3zdf_call == 3

      ! --- diag error on heat diffusion - PART 1 --- !
      DO ji = 1, npti
         zq_ini(ji) = ( SUM( e_i_1d(ji,1:nlay_i) ) * h_i_1d(ji) * r1_nlay_i +  &
            &           SUM( e_s_1d(ji,1:nlay_s) ) * h_s_1d(ji) * r1_nlay_s )
      END DO

      ! calculate ice fraction covered by snow for radiation
      CALL ice_var_snwfra( h_s_1d(1:npti), za_s_fra(1:npti) )

      !------------------
      ! 1) Initialization
      !------------------
      !
      ! extinction radiation in the snow
      IF    ( nn_qtrice == 0 ) THEN   ! constant
         zraext_s(1:npti) = rn_kappa_s
      ELSEIF( nn_qtrice == 1 ) THEN   ! depends on melting/freezing conditions
         WHERE( t_su_1d(1:npti) < rt0 )   ;   zraext_s(1:npti) = rn_kappa_sdry   ! no surface melting
         ELSEWHERE                        ;   zraext_s(1:npti) = rn_kappa_smlt   !    surface melting
         END WHERE
      ENDIF
      !
      ! thicknesses
      DO ji = 1, npti
         ! ice thickness
         IF( h_i_1d(ji) > 0._wp ) THEN
            zh_i  (ji) = MAX( zh_min , h_i_1d(ji) ) * r1_nlay_i ! set a minimum thickness for conduction
            z1_h_i(ji) = 1._wp / zh_i(ji)                       !       it must be very small
         ELSE
            zh_i  (ji) = 0._wp
            z1_h_i(ji) = 0._wp
         ENDIF
         ! snow thickness
         IF( h_s_1d(ji) > 0._wp ) THEN
            zh_s  (ji) = MAX( zh_min , h_s_1d(ji) ) * r1_nlay_s ! set a minimum thickness for conduction
            z1_h_s(ji) = 1._wp / zh_s(ji)                       !       it must be very small
            isnow (ji) = 1._wp
         ELSE
            zh_s  (ji) = 0._wp
            z1_h_s(ji) = 0._wp
            isnow (ji) = 0._wp
         ENDIF
         ! for Met-Office
         IF( h_s_1d(ji) < zh_min ) THEN
            isnow_comb(ji) = h_s_1d(ji) / zh_min
         ELSE
            isnow_comb(ji) = 1._wp
         ENDIF
      END DO
      ! clem: we should apply correction on snow thickness to take into account snow fraction
      !       it must be a distribution, so it is a bit complicated
      !
      ! Store initial temperatures and non solar heat fluxes
      IF( k_cnd == np_cnd_OFF .OR. k_cnd == np_cnd_EMU ) THEN
         ztsub      (1:npti) = t_su_1d(1:npti)                          ! surface temperature at iteration n-1
         t_su_1d    (1:npti) = MIN( t_su_1d(1:npti), rt0 - ztsu_err )   ! required to leave the choice between melting or not
         zdqns_ice_b(1:npti) = dqns_ice_1d(1:npti)                      ! derivative of incoming nonsolar flux
         zqns_ice_b (1:npti) = qns_ice_1d(1:npti)                       ! store previous qns_ice_1d value
         !
      ENDIF
      !
      ztsold (1:npti,:) = t_s_1d(1:npti,:)   ! Old snow temperature
      ztiold (1:npti,:) = t_i_1d(1:npti,:)   ! Old ice temperature

      !-------------
      ! 2) Radiation
      !-------------
      ! --- Transmission/absorption of solar radiation in the ice --- !
      zradtr_s(1:npti,0) = qtr_ice_top_1d(1:npti)
      DO jk = 1, nlay_s
         DO ji = 1, npti
            !                             ! radiation transmitted below the layer-th snow layer
            zradtr_s(ji,jk) = zradtr_s(ji,0) * EXP( - zraext_s(ji) * MAX( 0._wp, zh_s(ji) * REAL(jk) - zhs_ssl ) )
            !                             ! radiation absorbed by the layer-th snow layer
            zradab_s(ji,jk) = zradtr_s(ji,jk-1) - zradtr_s(ji,jk)
         END DO
      END DO
      !
      zradtr_i(1:npti,0) = zradtr_s(1:npti,nlay_s) * za_s_fra(1:npti) + qtr_ice_top_1d(1:npti) * ( 1._wp - za_s_fra(1:npti) )
      DO jk = 1, nlay_i
         DO ji = 1, npti
            !                             ! radiation transmitted below the layer-th ice layer
            zradtr_i(ji,jk) =           za_s_fra(ji)   * zradtr_s(ji,nlay_s)                       &   ! part covered by snow
               &                                       * EXP( - rn_kappa_i * MAX( 0._wp, zh_i(ji) * REAL(jk) - zh_min  ) ) &
               &            + ( 1._wp - za_s_fra(ji) ) * qtr_ice_top_1d(ji)                        &   ! part snow free
               &                                       * EXP( - rn_kappa_i * MAX( 0._wp, zh_i(ji) * REAL(jk) - zhi_ssl ) )
            !                             ! radiation absorbed by the layer-th ice layer
            zradab_i(ji,jk) = zradtr_i(ji,jk-1) - zradtr_i(ji,jk)
         END DO
      END DO
         !
      DO ji = 1, npti
         qtr_ice_bot_1d(ji) = zradtr_i(ji,nlay_i)   ! record radiation transmitted below the ice
      END DO
      IF( ll_l3zdf_capture ) THEN
         CALL l3zdf_header( 0, 0, 25 )
         WRITE(num_l3zdf) t_su_1d(1), t_i_1d(1,1:nlay_i), t_s_1d(1,1:nlay_s), &
            & sz_i_1d(1,1:nlay_i), h_i_1d(1), h_s_1d(1), ztsub(1), &
            & qns_ice_1d(1), zdqns_ice_b(1), qsr_ice_1d(1), qtr_ice_top_1d(1), &
            & t_bo_1d(1), zradab_s(1,1:nlay_s), zradab_i(1,1:nlay_i), qtr_ice_bot_1d(1)
      ENDIF
      !
      !
      iconv = 0          ! number of iterations
      !
      l_T_converged(:) = .FALSE.
      ! Convergence calculated until all sub-domain grid points have converged
      ! Calculations keep going for all grid points until sub-domain convergence (vectorisation optimisation)
      ! but values are not taken into account (results independant of MPI partitioning)
      !
      !                                                                            !============================!
      DO WHILE ( ( .NOT. ALL (l_T_converged(1:npti)) ) .AND. iconv < iconv_max )   ! Iterative procedure begins !
         !                                                                         !============================!
         iconv = iconv + 1
         !
         !--------------------------------
         ! 3) Sea ice thermal conductivity
         !--------------------------------
         IF( ln_cndi_U64 ) THEN         !-- Untersteiner (1964) formula: k = k0 + beta.S/T
            !
            DO ji = 1, npti
               ztcond_i_cp(ji,0)      = rcnd_i + zbeta * sz_i_1d(ji,1)      / MIN( -epsi10, t_i_1d(ji,1) - rt0 )
               ztcond_i_cp(ji,nlay_i) = rcnd_i + zbeta * sz_i_1d(ji,nlay_i) / MIN( -epsi10, t_bo_1d(ji)  - rt0 )
            END DO
            DO jk = 1, nlay_i-1
               DO ji = 1, npti
                  ztcond_i_cp(ji,jk) = rcnd_i + zbeta * 0.5_wp * ( sz_i_1d(ji,jk) + sz_i_1d(ji,jk+1) ) /  &
                     &                    MIN( -epsi10, 0.5_wp * (  t_i_1d(ji,jk) +  t_i_1d(ji,jk+1) ) - rt0 )
               END DO
            END DO
            !
         ELSEIF( ln_cndi_P07 ) THEN     !-- Pringle et al formula: k = k0 + beta1.S/T - beta2.T
            !
            DO ji = 1, npti
               ztcond_i_cp(ji,0)      = rcnd_i + 0.09_wp  *  sz_i_1d(ji,1)      / MIN( -epsi10, t_i_1d(ji,1) - rt0 )  &
                  &                            - 0.011_wp * ( t_i_1d(ji,1) - rt0 )
               ztcond_i_cp(ji,nlay_i) = rcnd_i + 0.09_wp  *  sz_i_1d(ji,nlay_i) / MIN( -epsi10, t_bo_1d(ji)  - rt0 )  &
                  &                            - 0.011_wp * ( t_bo_1d(ji) - rt0 )
            END DO
            DO jk = 1, nlay_i-1
               DO ji = 1, npti
                  ztcond_i_cp(ji,jk) = rcnd_i + 0.09_wp  *   0.5_wp * ( sz_i_1d(ji,jk) + sz_i_1d(ji,jk+1) ) /       &
                     &                         MIN( -epsi10, 0.5_wp * (  t_i_1d(ji,jk) +  t_i_1d(ji,jk+1) ) - rt0 ) &
                     &                        - 0.011_wp * ( 0.5_wp * (  t_i_1d(ji,jk) +  t_i_1d(ji,jk+1) ) - rt0 )
               END DO
            END DO
            !
         ENDIF

         ! Variable used after iterations
         ! Value must be frozen after convergence for MPP independance reason
         DO ji = 1, npti
            IF ( .NOT. l_T_converged(ji) ) &
               ztcond_i(ji,:) = MAX( zkimin, ztcond_i_cp(ji,:) )
         END DO
         !
         !--- G(he) : enhancement of thermal conductivity in mono-category case
         ! Computation of effective thermal conductivity G(h)
         ! Used in mono-category case only to simulate an ITD implicitly
         ! Fichefet and Morales Maqueda, JGR 1997
         zghe(1:npti) = 1._wp
         !
         IF( ln_virtual_itd ) THEN
            !
            zepsilon = 0.1_wp
            DO ji = 1, npti
               IF( .NOT. l_T_converged(ji) ) THEN
                  zcnd_i = SUM( ztcond_i(ji,:) ) / REAL( nlay_i+1, wp )                            ! Mean sea ice thermal conductivity
                  zhe = ( rcnd_s * h_i_1d(ji) + zcnd_i * h_s_1d(ji) ) / ( rcnd_s + zcnd_i )        ! Effective thickness he (zhe)
                  IF( zhe >=  zepsilon * 0.5_wp * EXP(1._wp) )  &
                     &   zghe(ji) = MIN( 2._wp, 0.5_wp * ( 1._wp + LOG( 2._wp * zhe / zepsilon ) ) )   ! G(he)
               ENDIF
            END DO
            !
         ENDIF
         !
         !-----------------
         ! 4) kappa factors
         !-----------------
         DO ji = 1, npti
            !
            IF ( .NOT. l_T_converged(ji) ) THEN
               !
               !--- Snow
               ! Variable used after iterations
               ! Value must be frozen after convergence for MPP independance reason
               DO jk = 0, nlay_s-1
                  zkappa_s(ji,jk) = zghe(ji) * rcnd_s * z1_h_s(ji)
               END DO
               zkappa_s(ji,nlay_s) = isnow(ji) * zghe(ji) * rcnd_s * ztcond_i(ji,0) &   ! Snow-ice interface
                  &                            / ( 0.5_wp * ( ztcond_i(ji,0) * zh_s(ji) + rcnd_s * zh_i(ji) ) )
               !
               !--- Ice
               ! Variable used after iterations
               ! Value must be frozen after convergence for MPP independance reason
               DO jk = 0, nlay_i
                  zkappa_i(ji,jk) = zghe(ji) * ztcond_i(ji,jk) * z1_h_i(ji)
               END DO
               ! Calculate combined surface snow and ice conductivity to pass through the coupler (met-office)
               zkappa_comb(ji) = isnow_comb(ji) * zkappa_s(ji,0) + ( 1._wp - isnow_comb(ji) ) * zkappa_i(ji,0)
               ! If there is snow then use the same snow-ice interface conductivity for the top layer of ice
               IF( h_s_1d(ji) > 0._wp )   zkappa_i(ji,0) = zkappa_s(ji,nlay_s)   ! Snow-ice interface
               !
            ENDIF
            !
         END DO

         IF( ll_l3zdf_capture ) THEN
            CALL l3zdf_header( 1, iconv, 20 )
            WRITE(num_l3zdf) t_su_1d(1), t_i_1d(1,1:nlay_i), t_s_1d(1,1:nlay_s), &
               & ztcond_i(1,0:nlay_i), zkappa_i(1,0:nlay_i), &
               & zkappa_s(1,0:nlay_s), isnow(1)
         ENDIF
         
         !--------------------------------------
         ! 5) Sea ice specific heat, eta factors
         !--------------------------------------
         DO jk = 1, nlay_i
            DO ji = 1, npti
               IF ( .NOT. l_T_converged(ji) ) THEN
                  zcpi = rcpi + zgamma * sz_i_1d(ji,jk) / MAX( ( t_i_1d(ji,jk) - rt0 ) * ( ztiold(ji,jk) - rt0 ), epsi10 )
                  zeta_i(ji,jk) = rDt_ice * r1_rhoi * z1_h_i(ji) / zcpi
               ENDIF
            END DO
         END DO
         !
         DO jk = 1, nlay_s
            DO ji = 1, npti
               IF ( .NOT. l_T_converged(ji) ) &
                  & zeta_s(ji,jk) = rDt_ice * r1_rhos * r1_rcpi * z1_h_s(ji)
            END DO
         END DO
         !
         !
         IF( k_cnd == np_cnd_OFF .OR. k_cnd == np_cnd_EMU ) THEN
            !----------------------------------------!
            !                                        !
            !   Conduction flux is off or emulated   !
            !                                        !
            !----------------------------------------!
            !
            ! ==> The original BL99 temperature computation is used
            !       (with qsr_ice, qns_ice and dqns_ice as inputs)
            !
            DO ji = 1, npti
               !
               !----------------------------
               ! 6) surface flux computation
               !----------------------------
               ! update of the non solar flux according to the update in T_su
               ! Variable used after iterations
               ! Value must be frozen after convergence for MPP independance reason
               IF ( .NOT. l_T_converged(ji) ) THEN
                  !
                  qns_ice_1d(ji) = qns_ice_1d(ji) + dqns_ice_1d(ji) * ( t_su_1d(ji) - ztsub(ji) )
                  !
                  zfnet = qsr_ice_1d(ji) - qtr_ice_top_1d(ji) + qns_ice_1d(ji) ! net heat flux = net - transmitted solar + non solar

                  IF( ll_l3zdf_capture .AND. ji == 1 ) THEN
                     CALL l3zdf_header( 2, iconv, 8 )
                     WRITE(num_l3zdf) zeta_i(1,1:nlay_i), zeta_s(1,1:nlay_s), qns_ice_1d(1), zfnet
                  ENDIF
                  !
                  ! before temperatures
                  ztib(:) = t_i_1d(ji,:)
                  ztsb(:) = t_s_1d(ji,:)
                  !
                  !----------------------------
                  ! 7) tridiagonal system terms
                  !----------------------------
                  ! layer denotes the number of the layer in the snow or in the ice
                  ! jm denotes the reference number of the equation in the tridiagonal
                  ! system, terms of tridiagonal system are indexed as following :
                  ! 1 is subdiagonal term, 2 is diagonal and 3 is superdiagonal one
                  
                  ! ice interior terms (top equation has the same form as the others)
                  ztrid   (:,:) = 0._wp
                  zindterm(:)   = 0._wp
                  zindtbis(:)   = 0._wp
                  zdiagbis(:)   = 0._wp

                  DO jm = nlay_s + 2, nlay_s + nlay_i
                     jk = jm - nlay_s - 1
                     ztrid   (jm,1) =       - zeta_i(ji,jk) *   zkappa_i(ji,jk-1)
                     ztrid   (jm,2) = 1._wp + zeta_i(ji,jk) * ( zkappa_i(ji,jk-1) + zkappa_i(ji,jk) )
                     ztrid   (jm,3) =       - zeta_i(ji,jk) *                       zkappa_i(ji,jk)
                     zindterm(jm)   = ztiold(ji,jk) + zeta_i(ji,jk) * zradab_i(ji,jk)
                  END DO

                  jm =  nlay_s + nlay_i + 1
                  ! ice bottom term
                  ztrid   (jm,1) =       - zeta_i(ji,nlay_i) *   zkappa_i(ji,nlay_i-1)
                  ztrid   (jm,2) = 1._wp + zeta_i(ji,nlay_i) * ( zkappa_i(ji,nlay_i-1) + zkappa_i(ji,nlay_i) * zg1 )
                  ztrid   (jm,3) = 0._wp
                  zindterm(jm)   = ztiold(ji,nlay_i) + zeta_i(ji,nlay_i) *  &
                     &         ( zradab_i(ji,nlay_i) + zkappa_i(ji,nlay_i) * zg1 * t_bo_1d(ji) )

                  !                               !---------------------!
                  IF( h_s_1d(ji) > 0._wp ) THEN   !  snow-covered cells !
                     !                            !---------------------!
                     ! snow interior terms (bottom equation has the same form as the others)
                     DO jm = 3, nlay_s + 1
                        jk = jm - 1
                        ztrid   (jm,1) =       - zeta_s(ji,jk) *   zkappa_s(ji,jk-1)
                        ztrid   (jm,2) = 1._wp + zeta_s(ji,jk) * ( zkappa_s(ji,jk-1) + zkappa_s(ji,jk) )
                        ztrid   (jm,3) =       - zeta_s(ji,jk) *                       zkappa_s(ji,jk)
                        zindterm(jm)   = ztsold(ji,jk) + zeta_s(ji,jk) * zradab_s(ji,jk)
                     END DO

                     ! case of only one layer in the ice (ice equation is altered)
                     IF( nlay_i == 1 ) THEN
                        ztrid   (nlay_s+2,3) = 0._wp
                        zindterm(nlay_s+2)   = zindterm(nlay_s+2) + zeta_i(ji,1) * zkappa_i(ji,1) * t_bo_1d(ji)
                     ENDIF

                     IF( t_su_1d(ji) < rt0 ) THEN   !--  case 1 : no surface melting

                        jm_min = 1
                        jm_max = nlay_i + nlay_s + 1

                        ! surface equation
                        ztrid   (1,1) = 0._wp
                        ztrid   (1,2) = zdqns_ice_b(ji) - zg1s * zkappa_s(ji,0)
                        ztrid   (1,3) =                   zg1s * zkappa_s(ji,0)
                        zindterm(1)   = zdqns_ice_b(ji) * t_su_1d(ji) - zfnet

                        ! first layer of snow equation
                        ztrid   (2,1) =       - zeta_s(ji,1) *                    zkappa_s(ji,0) * zg1s
                        ztrid   (2,2) = 1._wp + zeta_s(ji,1) * ( zkappa_s(ji,1) + zkappa_s(ji,0) * zg1s )
                        ztrid   (2,3) =       - zeta_s(ji,1) *   zkappa_s(ji,1)
                        zindterm(2)   = ztsold(ji,1) + zeta_s(ji,1) * zradab_s(ji,1)

                     ELSE                            !--  case 2 : surface is melting
                        !
                        jm_min = 2
                        jm_max = nlay_i + nlay_s + 1

                        ! first layer of snow equation
                        ztrid   (2,1) = 0._wp
                        ztrid   (2,2) = 1._wp + zeta_s(ji,1) * ( zkappa_s(ji,1) + zkappa_s(ji,0) * zg1s )
                        ztrid   (2,3) =       - zeta_s(ji,1) *   zkappa_s(ji,1)
                        zindterm(2)   = ztsold(ji,1) + zeta_s(ji,1) * ( zradab_s(ji,1) + zkappa_s(ji,0) * zg1s * t_su_1d(ji) )
                     ENDIF
                     !                            !---------------------!
                  ELSE                            ! cells without snow  !
                     !                            !---------------------!
                     !
                     IF( t_su_1d(ji) < rt0 ) THEN   !--  case 1 : no surface melting
                        !
                        jm_min = nlay_s + 1
                        jm_max = nlay_i + nlay_s + 1

                        ! surface equation
                        ztrid   (jm_min,1) = 0._wp
                        ztrid   (jm_min,2) = zdqns_ice_b(ji) - zkappa_i(ji,0) * zg1
                        ztrid   (jm_min,3) =                   zkappa_i(ji,0) * zg1
                        zindterm(jm_min)   = zdqns_ice_b(ji) * t_su_1d(ji) - zfnet

                        ! first layer of ice equation
                        ztrid   (jm_min+1,1) =       - zeta_i(ji,1) *                    zkappa_i(ji,0) * zg1
                        ztrid   (jm_min+1,2) = 1._wp + zeta_i(ji,1) * ( zkappa_i(ji,1) + zkappa_i(ji,0) * zg1 )
                        ztrid   (jm_min+1,3) =       - zeta_i(ji,1) *   zkappa_i(ji,1)
                        zindterm(jm_min+1)   = ztiold(ji,1) + zeta_i(ji,1) * zradab_i(ji,1)

                        ! case of only one layer in the ice (surface & ice equations are altered)
                        IF( nlay_i == 1 ) THEN
                           ztrid   (jm_min,1)   = 0._wp
                           ztrid   (jm_min,2)   = zdqns_ice_b(ji)      -   zkappa_i(ji,0) * 2._wp
                           ztrid   (jm_min,3)   =                          zkappa_i(ji,0) * 2._wp
                           ztrid   (jm_min+1,1) =       - zeta_i(ji,1) *   zkappa_i(ji,0) * 2._wp
                           ztrid   (jm_min+1,2) = 1._wp + zeta_i(ji,1) * ( zkappa_i(ji,0) * 2._wp + zkappa_i(ji,1) )
                           ztrid   (jm_min+1,3) = 0._wp
                           zindterm(jm_min+1)   = ztiold(ji,1) + zeta_i(ji,1) * (zradab_i(ji,1) + zkappa_i(ji,1) * t_bo_1d(ji))
                        ENDIF

                     ELSE                            !--  case 2 : surface is melting

                        jm_min = nlay_s + 2
                        jm_max = nlay_i + nlay_s + 1

                        ! first layer of ice equation
                        ztrid   (jm_min,1) = 0._wp
                        ztrid   (jm_min,2) = 1._wp + zeta_i(ji,1) * ( zkappa_i(ji,1) + zkappa_i(ji,0) * zg1 )
                        ztrid   (jm_min,3) =       - zeta_i(ji,1) *   zkappa_i(ji,1)
                        zindterm(jm_min)   = ztiold(ji,1) + zeta_i(ji,1) * (zradab_i(ji,1) + zkappa_i(ji,0) * zg1 * t_su_1d(ji))

                        ! case of only one layer in the ice (surface & ice equations are altered)
                        IF( nlay_i == 1 ) THEN
                           ztrid   (jm_min,1) = 0._wp
                           ztrid   (jm_min,2) = 1._wp + zeta_i(ji,1) * ( zkappa_i(ji,0) * 2._wp + zkappa_i(ji,1) )
                           ztrid   (jm_min,3) = 0._wp
                           zindterm(jm_min)   = ztiold(ji,1) + zeta_i(ji,1) * ( zradab_i(ji,1) + zkappa_i(ji,1) * t_bo_1d(ji) ) &
                              &                    + t_su_1d(ji) * zeta_i(ji,1) * zkappa_i(ji,0) * 2._wp
                        ENDIF

                     ENDIF
                  ENDIF

                  IF( ll_l3zdf_capture .AND. ji == 1 ) THEN
                     CALL l3zdf_header( 3, iconv, 28 )
                     WRITE(num_l3zdf) ztrid(:,1), ztrid(:,2), ztrid(:,3), zindterm(:)
                  ENDIF
                  !
                  zindtbis(jm_min) = zindterm(jm_min)
                  zdiagbis(jm_min) = ztrid   (jm_min,2)
                  !
                  !
                  !------------------------------
                  ! 8) tridiagonal system solving
                  !------------------------------
                  ! Solve the tridiagonal system with Gauss elimination method.
                  ! Thomas algorithm, from Computational fluid Dynamics, J.D. ANDERSON, McGraw-Hill 1984

                  DO jm = jm_min+1, jm_max
                     zdiagbis(jm) = ztrid   (jm,2) - ztrid(jm,1) * ztrid   (jm-1,3) / zdiagbis(jm-1)
                     zindtbis(jm) = zindterm(jm  ) - ztrid(jm,1) * zindtbis(jm-1  ) / zdiagbis(jm-1)
                  END DO

                  IF( ll_l3zdf_capture .AND. ji == 1 ) THEN
                     CALL l3zdf_header( 4, iconv, 14 )
                     WRITE(num_l3zdf) zdiagbis(:), zindtbis(:)
                  ENDIF

                  ! ice temperatures
                  ! Variable used after iterations
                  ! Value must be frozen after convergence for MPP independance reason
                  t_i_1d(ji,nlay_i) = zindtbis(jm_max) / zdiagbis(jm_max)
                  
                  DO jm = nlay_i + nlay_s, nlay_s + 2, -1
                     jk = jm - nlay_s - 1
                     t_i_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_i_1d(ji,jk+1) ) / zdiagbis(jm)
                  END DO

                  ! snow temperatures
                  ! Variables used after iterations
                  ! Value must be frozen after convergence for MPP independance reason
                  IF ( h_s_1d(ji) > 0._wp ) &
                     &   t_s_1d(ji,nlay_s) = ( zindtbis(nlay_s+1) - ztrid(nlay_s+1,3) * t_i_1d(ji,1) ) / zdiagbis(nlay_s+1)

                  DO jm = nlay_s, 2, -1
                     jk = jm - 1
                     IF ( h_s_1d(ji) > 0._wp ) &
                        &   t_s_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_s_1d(ji,jk+1) ) / zdiagbis(jm)
                  END DO
                  
                  ! surface temperature
                  ztsub(ji) = t_su_1d(ji)
                  IF( t_su_1d(ji) < rt0 ) THEN
                     t_su_1d(ji) = ( zindtbis(jm_min) - ztrid(jm_min,3) *  &
                        &          ( isnow(ji) * t_s_1d(ji,1) + ( 1._wp - isnow(ji) ) * t_i_1d(ji,1) ) ) / zdiagbis(jm_min)
                  ENDIF
                  IF( ll_l3zdf_capture .AND. ji == 1 ) THEN
                     CALL l3zdf_header( 5, iconv, 9 )
                     WRITE(num_l3zdf) t_su_1d(1), t_s_1d(1,1:nlay_s), &
                        & t_i_1d(1,1:nlay_i), ztsub(1), qns_ice_1d(1)
                  ENDIF
                  !
                  !--------------------------------------------------------------
                  ! 9) Has the scheme converged?, end of the iterative procedure
                  !--------------------------------------------------------------
                  ! check that nowhere it has started to melt
                  ! zdti_max is a measure of error, it has to be under zdti_bnd                  
                  zdti_max = 0._wp

                  t_su_1d(ji) = MAX( MIN( t_su_1d(ji) , rt0 ) , rt0 - 100._wp )
                  zdti_max    = MAX( zdti_max, ABS( t_su_1d(ji) - ztsub(ji) ) )

                  IF( h_s_1d(ji) > 0._wp ) THEN
                     DO jk = 1, nlay_s
                        t_s_1d(ji,jk) = MAX( MIN( t_s_1d(ji,jk), rt0 ), rt0 - 100._wp )
                        zdti_max      = MAX ( zdti_max , ABS( t_s_1d(ji,jk) - ztsb(jk) ) )
                     END DO
                  ENDIF

                  DO jk = 1, nlay_i
                     ztmelts       = -rTmlt * sz_i_1d(ji,jk) + rt0
                     t_i_1d(ji,jk) =  MAX( MIN( t_i_1d(ji,jk), ztmelts ), rt0 - 100._wp )
                     zdti_max      =  MAX( zdti_max, ABS( t_i_1d(ji,jk) - ztib(jk) ) )
                  END DO

                  ! convergence test
                  IF( ln_zdf_chkcvg ) THEN
                     tice_cvgerr_1d(ji) = zdti_max
                     tice_cvgstp_1d(ji) = REAL(iconv)
                  ENDIF

                  IF( zdti_max < zdti_bnd )   l_T_converged(ji) = .TRUE.

                  IF( ll_l3zdf_capture .AND. ji == 1 ) THEN
                     CALL l3zdf_header( 6, iconv, 10 )
                     WRITE(num_l3zdf) t_su_1d(1), t_s_1d(1,1:nlay_s), &
                        & t_i_1d(1,1:nlay_i), zdti_max, &
                        & MERGE( 1._wp, 0._wp, l_T_converged(1) ), REAL( iconv, wp )
                  ENDIF

               ENDIF
               
            END DO     
            !
         ELSEIF( k_cnd == np_cnd_ON ) THEN
            !----------------------------------------!
            !                                        !
            !      Conduction flux is on             !
            !                                        !
            !----------------------------------------!
            !
            ! ==> we use a modified BL99 solver with conduction flux (qcn_ice) as forcing term
            !
            DO ji = 1, npti
               !
               IF ( .NOT. l_T_converged(ji) ) THEN
                  ! before temperatures
                  ztib(:) = t_i_1d(ji,:)
                  ztsb(:) = t_s_1d(ji,:)
                  !
                  !----------------------------
                  ! 7) tridiagonal system terms
                  !----------------------------
                  ! layer denotes the number of the layer in the snow or in the ice
                  ! jm denotes the reference number of the equation in the tridiagonal
                  ! system, terms of tridiagonal system are indexed as following :
                  ! 1 is subdiagonal term, 2 is diagonal and 3 is superdiagonal one

                  ! ice interior terms (top equation has the same form as the others)
                  ztrid   (:,:) = 0._wp
                  zindterm(:)   = 0._wp
                  zindtbis(:)   = 0._wp
                  zdiagbis(:)   = 0._wp

                  DO jm = nlay_s + 2, nlay_s + nlay_i
                     jk = jm - nlay_s - 1
                     ztrid   (jm,1) =       - zeta_i(ji,jk) *   zkappa_i(ji,jk-1)
                     ztrid   (jm,2) = 1._wp + zeta_i(ji,jk) * ( zkappa_i(ji,jk-1) + zkappa_i(ji,jk) )
                     ztrid   (jm,3) =       - zeta_i(ji,jk) *                       zkappa_i(ji,jk)
                     zindterm(jm)   = ztiold(ji,jk) + zeta_i(ji,jk) * zradab_i(ji,jk)
                  END DO

                  ! ice bottom term
                  jm =  nlay_s + nlay_i + 1
                  ztrid   (jm,1) =       - zeta_i(ji,nlay_i) *   zkappa_i(ji,nlay_i-1)
                  ztrid   (jm,2) = 1._wp + zeta_i(ji,nlay_i) * ( zkappa_i(ji,nlay_i-1) + zkappa_i(ji,nlay_i) * zg1 )
                  ztrid   (jm,3) = 0._wp
                  zindterm(jm)   = ztiold(ji,nlay_i) + zeta_i(ji,nlay_i) *  &
                     &              ( zradab_i(ji,nlay_i) + zkappa_i(ji,nlay_i) * zg1 * t_bo_1d(ji) )

                  !                               !---------------------!
                  IF( h_s_1d(ji) > 0._wp ) THEN   !  snow-covered cells !
                     !                            !---------------------!
                     ! snow interior terms (bottom equation has the same form as the others)
                     DO jm = 3, nlay_s + 1
                        jk = jm - 1
                        ztrid   (jm,1) =       - zeta_s(ji,jk) *   zkappa_s(ji,jk-1)
                        ztrid   (jm,2) = 1._wp + zeta_s(ji,jk) * ( zkappa_s(ji,jk-1) + zkappa_s(ji,jk) )
                        ztrid   (jm,3) =       - zeta_s(ji,jk) *                       zkappa_s(ji,jk)
                        zindterm(jm)   = ztsold(ji,jk) + zeta_s(ji,jk) * zradab_s(ji,jk)
                     END DO

                     ! case of only one layer in the ice (ice equation is altered)
                     IF ( nlay_i == 1 ) THEN
                        ztrid   (nlay_s+2,3) = 0._wp
                        zindterm(nlay_s+2)   = zindterm(nlay_s+2) + zeta_i(ji,1) * zkappa_i(ji,1) * t_bo_1d(ji)
                     ENDIF

                     jm_min = 2
                     jm_max = nlay_i + nlay_s + 1

                     ! first layer of snow equation
                     ztrid   (2,1) = 0._wp
                     ztrid   (2,2) = 1._wp + zeta_s(ji,1) * zkappa_s(ji,1)
                     ztrid   (2,3) =       - zeta_s(ji,1) * zkappa_s(ji,1)
                     zindterm(2)   = ztsold(ji,1) + zeta_s(ji,1) * ( zradab_s(ji,1) + qcn_ice_1d(ji) )

                     !                            !---------------------!
                  ELSE                            ! cells without snow  !
                     !                            !---------------------!
                     jm_min = nlay_s + 2
                     jm_max = nlay_i + nlay_s + 1

                     ! first layer of ice equation
                     ztrid   (jm_min,1) = 0._wp
                     ztrid   (jm_min,2) = 1._wp + zeta_i(ji,1) * zkappa_i(ji,1)
                     ztrid   (jm_min,3) =       - zeta_i(ji,1) * zkappa_i(ji,1)
                     zindterm(jm_min)   = ztiold(ji,1) + zeta_i(ji,1) * ( zradab_i(ji,1) + qcn_ice_1d(ji) )

                     ! case of only one layer in the ice (surface & ice equations are altered)
                     IF( nlay_i == 1 ) THEN
                        ztrid   (jm_min,1) = 0._wp
                        ztrid   (jm_min,2) = 1._wp + zeta_i(ji,1) * zkappa_i(ji,1)
                        ztrid   (jm_min,3) = 0._wp
                        zindterm(jm_min)   = ztiold(ji,1) + zeta_i(ji,1) *  &
                           &                                     ( zradab_i(ji,1) + zkappa_i(ji,1) * t_bo_1d(ji) + qcn_ice_1d(ji) )
                     ENDIF

                  ENDIF
                  !
                  zindtbis(jm_min) = zindterm(jm_min)
                  zdiagbis(jm_min) = ztrid   (jm_min,2)
                  !
                  !
                  !------------------------------
                  ! 8) tridiagonal system solving
                  !------------------------------
                  ! Solve the tridiagonal system with Gauss elimination method.
                  ! Thomas algorithm, from Computational fluid Dynamics, J.D. ANDERSON, McGraw-Hill 1984

                  DO jm = jm_min+1, jm_max
                     zdiagbis(jm) = ztrid   (jm,2) - ztrid(jm,1) * ztrid   (jm-1,3) / zdiagbis(jm-1)
                     zindtbis(jm) = zindterm(jm  ) - ztrid(jm,1) * zindtbis(jm-1  ) / zdiagbis(jm-1)
                  END DO

                  ! ice temperatures
                  ! Variable used after iterations
                  ! Value must be frozen after convergence for MPP independance reason
                  t_i_1d(ji,nlay_i) = zindtbis(jm_max) / zdiagbis(jm_max)

                  DO jm = nlay_i + nlay_s, nlay_s + 2, -1
                     jk = jm - nlay_s - 1
                     t_i_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_i_1d(ji,jk+1) ) / zdiagbis(jm)
                  END DO

                  ! snow temperatures
                  ! Variables used after iterations
                  ! Value must be frozen after convergence for MPP independance reason
                  IF ( h_s_1d(ji) > 0._wp ) &
                     &   t_s_1d(ji,nlay_s) = ( zindtbis(nlay_s+1) - ztrid(nlay_s+1,3) * t_i_1d(ji,1) ) / zdiagbis(nlay_s+1)

                  DO jm = nlay_s, 2, -1
                     jk = jm - 1
                     IF ( h_s_1d(ji) > 0._wp ) &
                        &   t_s_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_s_1d(ji,jk+1) ) / zdiagbis(jm)
                  END DO
                  !
                  !--------------------------------------------------------------
                  ! 9) Has the scheme converged?, end of the iterative procedure
                  !--------------------------------------------------------------
                  ! check that nowhere it has started to melt
                  ! zdti_max is a measure of error, it has to be under zdti_bnd
                  zdti_max = 0._wp

                  IF( h_s_1d(ji) > 0._wp ) THEN
                     DO jk = 1, nlay_s
                        t_s_1d(ji,jk) = MAX( MIN( t_s_1d(ji,jk), rt0 ), rt0 - 100._wp )
                        zdti_max      = MAX ( zdti_max , ABS( t_s_1d(ji,jk) - ztsb(jk) ) )
                     END DO
                  ENDIF

                  DO jk = 1, nlay_i
                     ztmelts       = -rTmlt * sz_i_1d(ji,jk) + rt0
                     t_i_1d(ji,jk) =  MAX( MIN( t_i_1d(ji,jk), ztmelts ), rt0 - 100._wp )
                     zdti_max      =  MAX ( zdti_max, ABS( t_i_1d(ji,jk) - ztib(jk) ) )
                  END DO

                  ! convergence test
                  IF( ln_zdf_chkcvg ) THEN
                     tice_cvgerr_1d(ji) = zdti_max
                     tice_cvgstp_1d(ji) = REAL(iconv)
                  ENDIF

                  IF( zdti_max < zdti_bnd )   l_T_converged(ji) = .TRUE.

               ENDIF

            END DO

         ENDIF ! k_cnd

      END DO  ! End of the do while iterative procedure
      !
      !-----------------------------
      ! 10) Fluxes at the interfaces
      !-----------------------------
      !
      IF( k_cnd == np_cnd_OFF .OR. k_cnd == np_cnd_EMU ) THEN
         !
         DO ji = 1, npti
            ! --- ice conduction fluxes (positive downward)
            qcn_ice_bot_1d(ji) = - zkappa_i(ji,nlay_i) * zg1 * ( t_bo_1d(ji ) - t_i_1d (ji,nlay_i) )                ! bottom
            qcn_ice_top_1d(ji) = -           isnow(ji)   * zkappa_s(ji,0) * zg1s * ( t_s_1d(ji,1) - t_su_1d(ji) ) & ! surface
               &                 - ( 1._wp - isnow(ji) ) * zkappa_i(ji,0) * zg1  * ( t_i_1d(ji,1) - t_su_1d(ji) )
            ! --- Diagnose the heat loss due to changing non-solar / conduction flux
            hfx_err_dif_1d(ji) = hfx_err_dif_1d(ji) - ( qns_ice_1d(ji) - zqns_ice_b(ji) ) * a_i_1d(ji)           
         END DO
         !
      ELSEIF( k_cnd == np_cnd_ON ) THEN
         !
         DO ji = 1, npti
            ! --- ice conduction fluxes (positive downward)
            qcn_ice_bot_1d(ji) = - zkappa_i(ji,nlay_i) * zg1 * ( t_bo_1d(ji ) - t_i_1d (ji,nlay_i) ) ! bottom
            qcn_ice_top_1d(ji) = qcn_ice_1d(ji)                                                      ! surface
         END DO
         !
         ! --- surface ice temperature
         IF( ln_cndemulate ) THEN
            DO ji = 1, npti
               t_su_1d(ji) = ( qcn_ice_top_1d(ji) +          isnow(ji)   * zkappa_s(ji,0) * zg1s * t_s_1d(ji,1) + &
                  &                                ( 1._wp - isnow(ji) ) * zkappa_i(ji,0) * zg1  * t_i_1d(ji,1) ) &
                  &          / MAX( epsi10, isnow(ji) * zkappa_s(ji,0) * zg1s + ( 1._wp - isnow(ji) ) * zkappa_i(ji,0) * zg1 )
               t_su_1d(ji) = MAX( MIN( t_su_1d(ji), rt0 ), rt0 - 100._wp )  ! cap t_su
            END DO
         ENDIF
         !
      ENDIF
      !
      ! --- Diagnose the heat loss due to non-fully converged temperature solution (should not be larger than 10-4 W-m2)
      !
      IF( k_cnd == np_cnd_OFF .OR. k_cnd == np_cnd_ON ) THEN

         CALL ice_var_enthalpy

         ! zhfx_err = correction on the diagnosed heat flux due to non-convergence of the algorithm used to solve heat equation
         DO ji = 1, npti
            zdq = - zq_ini(ji) + ( SUM( e_i_1d(ji,1:nlay_i) ) * h_i_1d(ji) * r1_nlay_i +  &
               &                   SUM( e_s_1d(ji,1:nlay_s) ) * h_s_1d(ji) * r1_nlay_s )

            IF( k_cnd == np_cnd_OFF ) THEN

               IF( t_su_1d(ji) < rt0 ) THEN  ! case T_su < 0degC
                  zhfx_err = ( qns_ice_1d(ji)     + qsr_ice_1d(ji)     - zradtr_i(ji,nlay_i) - qcn_ice_bot_1d(ji)  &
                     &       + zdq * r1_Dt_ice ) * a_i_1d(ji)
               ELSE                          ! case T_su = 0degC
                  zhfx_err = ( qcn_ice_top_1d(ji) + qtr_ice_top_1d(ji) - zradtr_i(ji,nlay_i) - qcn_ice_bot_1d(ji)  &
                     &       + zdq * r1_Dt_ice ) * a_i_1d(ji)
               ENDIF

            ELSEIF( k_cnd == np_cnd_ON ) THEN

               zhfx_err    = ( qcn_ice_top_1d(ji) + qtr_ice_top_1d(ji) - zradtr_i(ji,nlay_i) - qcn_ice_bot_1d(ji)  &
                  &          + zdq * r1_Dt_ice ) * a_i_1d(ji)

            ENDIF
            !
            ! total heat sink to be sent to the ocean
            hfx_err_dif_1d(ji) = hfx_err_dif_1d(ji) + zhfx_err
            !
            ! hfx_dif = Heat flux diagnostic of sensible heat used to warm/cool ice in W.m-2
            hfx_dif_1d(ji) = hfx_dif_1d(ji) - zdq * r1_Dt_ice * a_i_1d(ji)
            !
         END DO
         !
      ENDIF
      !
      !--------------------------------------------------------------------
      ! 11) reset inner snow and ice temperatures, update conduction fluxes
      !--------------------------------------------------------------------
      ! effective conductivity and 1st layer temperature (needed by Met Office)
      ! this is a conductivity at mid-layer, hence the factor 2
      DO ji = 1, npti
         IF( h_i_1d(ji) >= zhi_ssl ) THEN
            cnd_ice_1d(ji) = 2._wp * zkappa_comb(ji)
            !!cnd_ice_1d(ji) = 2._wp * zkappa_i(ji,0)
         ELSE
            cnd_ice_1d(ji) = 2._wp * ztcond_i(ji,0) / zhi_ssl ! cnd_ice is capped by: cond_i/zhi_ssl
         ENDIF
         t1_ice_1d(ji) = isnow(ji) * t_s_1d(ji,1) + ( 1._wp - isnow(ji) ) * t_i_1d(ji,1)
      END DO
      !
      IF( k_cnd == np_cnd_EMU ) THEN
         ! Restore temperatures to their initial values
         t_s_1d    (1:npti,:) = ztsold        (1:npti,:)
         t_i_1d    (1:npti,:) = ztiold        (1:npti,:)
         qcn_ice_1d(1:npti)   = qcn_ice_top_1d(1:npti)
      ENDIF
      !
      ! --- SIMIP diagnostics (Snow-ice interfacial temperature)
      DO ji = 1, npti
         IF( h_s_1d(ji) >= zhs_ssl ) THEN
            t_si_1d(ji) = (   rcnd_s         * h_i_1d(ji) * r1_nlay_i * t_s_1d(ji,nlay_s)   &
               &            + ztcond_i(ji,1) * h_s_1d(ji) * r1_nlay_s * t_i_1d(ji,1)      ) &
               &          / ( rcnd_s         * h_i_1d(ji) * r1_nlay_i &
               &            + ztcond_i(ji,1) * h_s_1d(ji) * r1_nlay_s )
         ELSE
            t_si_1d(ji) = t_su_1d(ji)
         ENDIF
      END DO
      IF( ll_l3zdf_capture ) THEN
         FLUSH(num_l3zdf)
      ENDIF
      !
   END SUBROUTINE ice_thd_zdf_BL99


   SUBROUTINE l3zdf_header( kframe, kiter, knval )
      INTEGER, INTENT(in) :: kframe, kiter, knval
      INTEGER :: ios
      CHARACTER(LEN=16), PARAMETER :: cmagic = 'NEMO_L3ZDF_001'
      IF( .NOT. ll_l3zdf_opened ) THEN
         OPEN( NEWUNIT=num_l3zdf, FILE='oracle_si3_zdf_operands.bin', STATUS='REPLACE', &
            & ACCESS='STREAM', FORM='UNFORMATTED', ACTION='WRITE', IOSTAT=ios )
         IF( ios /= 0 ) STOP 'l3zdf_header: cannot open operand stream'
         ll_l3zdf_opened = .TRUE.
      ENDIF
      WRITE(num_l3zdf) cmagic
      WRITE(num_l3zdf) 1, kn_l3zdf_call, kframe, kiter, npti, nlay_i, nlay_s, STORAGE_SIZE(1._wp), knval
   END SUBROUTINE l3zdf_header

#else
   !!----------------------------------------------------------------------
   !!   Default option       Dummy Module             No SI3 sea-ice model
   !!----------------------------------------------------------------------
#endif

   !!======================================================================
END MODULE icethd_zdf_BL99
