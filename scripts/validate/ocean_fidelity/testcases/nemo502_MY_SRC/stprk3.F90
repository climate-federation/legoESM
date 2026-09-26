MODULE stprk3
   !!======================================================================
   !!                       ***  MODULE stpRK3  ***
   !! Time-stepping   : manager of the ocean, tracer and ice time stepping
   !!                   using a 3rd order Rung Kuta  with fixed or quasi-eulerian coordinate
   !!======================================================================
   !! History :  4.5  !  2021-01  (S. Techene, G. Madec, N. Ducousso, F. Lemarie)  Original code
   !!   NEMO     
   !!----------------------------------------------------------------------
#if defined key_RK3
#if defined key_qco   ||   defined key_linssh
   !!----------------------------------------------------------------------
   !!   'key_qco'                        Quasi-Eulerian vertical coordinate
   !!                          OR
   !!   'key_linssh                       Fixed in time vertical coordinate
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   stp_RK3       : NEMO 3rd order Runge-Kutta time-stepping 
   !!----------------------------------------------------------------------
   USE step_oce       ! time stepping used modules
   USE trd_oce        ! trends: ocean variables
   USE domqco         ! quasi-eulerian coordinate      (dom_qco_r3c routine)
   USE stprk3_stg     ! RK3 stages
   USE stp2d          ! external mode solver
   USE lib_mpp, ONLY : l_perpetual_ts

   IMPLICIT NONE
   PRIVATE

   PUBLIC   stp_RK3   ! called by nemogcm.F90

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

#if defined key_agrif
   RECURSIVE SUBROUTINE stp_RK3( )
      INTEGER             ::   kstp   ! ocean time-step index
#else
   SUBROUTINE stp_RK3( kstp )
      INTEGER, INTENT(in) ::   kstp   ! ocean time-step index
#endif
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE stp_RK3  ***
      !!
      !! ** Purpose : - Time stepping of OCE  (momentum and active tracer Eqs.) (RK3)
      !!              - Time stepping of SI3 (dynamic and thermodynamic Eqs.)   (FBS)
      !!              - Time stepping of TRC  (passive tracer Eqs.)
      !!
      !! ** Method  : -1- Update forcings and data
      !!              -2- Update ocean physics
      !!              -3- Compute the after (Naa) ssh and velocity 
      !!              -4- diagnostics and output at Now (Nnn)
      !!              -4- Compute the after (Naa) T-S
      !!              -5- Update now 
      !!              -6- Update the horizontal velocity
      !!              -7- Compute the diagnostics variables (rd,N2, hdiv,w)
      !!              -8- Outputs and diagnostics
      !!----------------------------------------------------------------------
      INTEGER ::   ji, jj, jk, jtile   ! dummy loop indice
      INTEGER ::   itraj, ios_traj
      CHARACTER(LEN=96) ::   cl_traj
      CHARACTER(LEN=16) ::   cl_magic
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) ::   zgdept
      !! ---------------------------------------------------------------------
#if defined key_agrif
      IF( nstop > 0 )   RETURN   ! avoid to go further if an error was detected during previous time step (child grid)
      kstp = nit000 + Agrif_Nb_Step()
      Kbb_a = Nbb   ; Kmm_a = Nbb   ;   Krhs_a = Nrhs   ! agrif_oce module copies of time level indices
      IF( lk_agrif_debug ) THEN
         IF( Agrif_Root() .AND. lwp)   WRITE(*,*) '---'
         IF(lwp)   WRITE(*,*) 'Grid Number', Agrif_Fixed(),' time step ', kstp, 'int tstep', Agrif_NbStepint()
      ENDIF
      IF( kstp == nit000 + 1 )   lk_agrif_fstep = .FALSE.
# if defined key_xios
      IF( Agrif_Nbstepint() == 0 )   CALL iom_swap( cxios_context )
# endif
#endif
      !
      IF( ln_timing )   CALL timing_start('stp_RK3')
      !
      ! Lane-1 certified oracle: exact step-entry Nbb state.  This is a
      ! write-only instrument; no prognostic array is assigned here.
      IF( lwp .AND. ( kstp == nit000 .OR. kstp == ( nit000 + nitend ) / 2 .OR. kstp == nitend ) ) THEN
         IF( STORAGE_SIZE( 1._wp ) /= 64 )   CALL ctl_stop( 'lane1 trajectory requires 64-bit wp' )
         WRITE(cl_traj,'("oracle_step_entry_kt",I8.8,".bin")') kstp
         OPEN( NEWUNIT=itraj, FILE=TRIM(cl_traj), ACCESS='STREAM', FORM='UNFORMATTED', &
            &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios_traj )
         IF( ios_traj /= 0 )   CALL ctl_stop( 'cannot open lane1 trajectory file' )
         cl_magic = 'NEMO_L1_ENTRY_1'
         WRITE(itraj) cl_magic, 1, kstp, Nbb, jpi, jpj, jpk, jpts, STORAGE_SIZE(1._wp)
         WRITE(itraj) ts(:,:,:,:,Nbb), uu(:,:,:,Nbb), vv(:,:,:,Nbb), ssh(:,:,Nbb)
         CLOSE(itraj)
         WRITE(numout,*) 'LANE1_STEP_ENTRY_DUMP ', kstp, Nbb, STORAGE_SIZE(1._wp), TRIM(cl_traj)
      ENDIF
      !
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! update I/O and calendar
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      !
      IF( kstp == nit000 ) THEN                       ! initialize IOM context (must be done after nemo_init for AGRIF+XIOS+OASIS)
                             CALL iom_init( cxios_context, ld_closedef=.FALSE. )   ! for model grid (including possible AGRIF zoom)
         IF( l_diamlr   )    CALL dia_mlr_iom_init    ! with additional setup for multiple-linear-regression analysis
                             CALL iom_init_closedef
                             CALL dia_hth_init        ! called here since it uses iom_use
                             CALL dia_ptr_init        ! called here since it uses iom_use
                             CALL dia_ar5_init        ! called here since it uses iom_use
                             CALL dia_hsb_init( Nbb ) ! heat content, salt content and volume budgets
                             CALL dia_25h_init( Nbb ) ! 25h mean  outputs
                             CALL rk3_dia( -1 )       ! Store diagnostic logicals
      ENDIF
      IF( kstp == nitrst .AND. lwxios ) THEN
                             CALL iom_swap(                     cw_ocerst_cxt )
                             CALL iom_init_closedef(            cw_ocerst_cxt )
                             CALL iom_setkt( kstp - nit000 + 1, cw_ocerst_cxt )
#if defined key_top
         IF( ln_top      ) THEN
                             CALL iom_swap(                     cw_toprst_cxt )
                             CALL iom_init_closedef(            cw_toprst_cxt )
                             CALL iom_setkt( kstp - nit000 + 1, cw_toprst_cxt )
         ENDIF
#endif
      ENDIF

      IF( kstp + nn_fsbc - 1 == nitrst .AND. lwxios ) THEN
#if defined key_si3
         IF( nn_ice == 2 ) THEN
                             CALL iom_swap(                     cw_icerst_cxt )
                             CALL iom_init_closedef(            cw_icerst_cxt )
                             CALL iom_setkt( kstp - nit000 + 1, cw_icerst_cxt )
         ENDIF
#endif
         IF( ln_abl      ) THEN
                             CALL iom_swap(                     cw_ablrst_cxt )
                             CALL iom_init_closedef(            cw_ablrst_cxt )
                             CALL iom_setkt( kstp - nit000 + 1, cw_ablrst_cxt )
         ENDIF
      ENDIF
      IF( kstp /= nit000 )   CALL day( kstp )         ! Calendar (day was already called at nit000 in day_init)
                             CALL iom_setkt( kstp - nit000 + 1,      cxios_context          )   ! tell IOM we are at time step kstp


      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Update external forcing (tides, open boundaries, ice shelf interaction and surface boundary condition (including sea-ice)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      IF( ln_tide    )   CALL tide_update( kstp )                          ! update tide potential
      IF( ln_apr_dyn )   CALL sbc_apr    ( kstp )                          ! atmospheric pressure (NB: call before bdy_dta which needs ssh_ib)
      IF( ln_bdy     )   CALL bdy_dta    ( kstp, Nbb )                     ! update dynamic & tracer data at open boundaries
      IF( ln_isf     )   CALL isf_stp    ( kstp, Nbb )                     ! update iceshelf geometry
                         CALL sbc        ( kstp, Nbb, Nbb )                ! Sea Boundary Condition (including sea-ice)
!!$      IF( ln_isf     )   CALL isf_stp    ( kstp, Nbb )                     ! update iceshelf geometry
                         !clem: problem with isf and cpl: sbcfwb needs isf but isf needs fwf from sbccpl

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Update stochastic parameters and random T/S fluctuations
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF( ln_sto_eos )   CALL sto_par( kstp )                              ! Stochastic parameters
      IF( ln_sto_eos )   CALL sto_pts( ts(:,:,:,:,Nnn)  )                  ! Random T/S fluctuations

!!gm  ocean physic computed at stage 3 ?

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Ocean physics update
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      !  THERMODYNAMICS
!!gm only Before is needed for bn2 and rab  (except at stage 3 for n2)
!!gm issue with   Nnn  used in rad(Nbb) 
                         CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )       ! before local thermal/haline expension ratio at T-points
                         CALL bn2    ( ts(:,:,:,:,Nbb), rab_b, rn2b, Nbb ) ! before Brunt-Vaisala frequency
!!gm
                         rab_n = rab_b
                         rn2   = rn2b
!!gm sh2 computed at the end of the time-step
!!gm       or  call zdf_phy at the end !
      !  VERTICAL PHYSICS
!!st                         CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )   ! vertical physics update (top/bot drag, avt, avs, avm + MLD)
                         CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )   ! vertical physics update (top/bot drag, avt, avs, avm + MLD)
!!gm gdep
      !  LATERAL  PHYSICS
      !
      IF( l_ldfslp ) THEN                             ! slope of lateral mixing
         IF( ln_traldf_triad ) THEN
                         CALL ldf_slp_triad( kstp, Nbb, Nbb )        ! before slope for triad operator
         ELSE
                         CALL eos ( ts, Nbb, rhd )                   ! before in situ density
                         CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )   ! before slope for standard operator
         ENDIF
      ENDIF
      !                                                                        ! eddy diffusivity coeff.
      IF( l_ldftra_time .OR. l_ldfeiv_time )   CALL ldf_tra( kstp, Nbb, Nbb )  !       and/or eiv coeff.
      IF( l_ldfeke                         )   CALL ldf_eke_eiv( kstp, Nbb )   ! GEOMETRIC param. (update of eiv coefficient)
      IF( l_ldfdyn_time                    )   CALL ldf_dyn( kstp, Nbb )       ! eddy viscosity coeff.


      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      !  RK3 : single first external mode computation
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )         ! out: ssh, (uu_b,vv_b) at Naa and (un_adv,vn_adv) between Nbb and Naa

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      !  RK3 time integration
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

      CALL rk3_dia( 0 )                                ! Diagnostics switched off for stage 1 & 2
      !
      ! Stage 1 :
      CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )
      !
      Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs    ! Swap: Nbb unchanged, Nnn <==> Naa
      !
      ! Stage 2 :
      CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )
      !
      Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs    ! Swap: Nbb unchanged, Nnn <==> Naa
      !
      ! Stage 3 :
      CALL rk3_dia( 1 )                                ! Diagnostics switched on for stage 3
      !
      CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )
      !
      IF ( l_ldfeke   )  CALL ldf_eke( kstp, Nbb )                ! GEOMETRIC param. (time evolution of eiv coefficient)
      !
      IF ( .NOT. l_perpetual_ts ) THEN
         !
         Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs    ! Swap: Nnn unchanged, Nbb <==> Naa

         ! linear extrapolation of ssh to compute ww at the beginning of the next time-step
         ! ssh(n+1) = 2*ssh(n) - ssh(n-1)    
         ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)
         !!st: ssh recomputed at the begining of stp2d
         !
      ENDIF

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! diagnostics and outputs
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
!==>>>  at Nbb  no more Nnn 
     
      IF( ln_diacfl  )   CALL dia_cfl   ( kstp,      Nbb )      ! Courant number diagnostics
      IF( ln_diadct  )   CALL dia_dct   ( kstp,      Nbb )      ! Transports

      IF( ln_tile ) CALL dom_tile_start         ! [tiling] DIA tiling loop
      DO jtile = 1, nijtile
         IF( ln_tile ) CALL dom_tile( ntsi, ntsj, ntei, ntej, ktile = jtile )
         IF( l_hth )     CALL dia_hth   ( kstp,      Nbb )      ! Thermocline depth (20 degres isotherm depth)
                         CALL dia_ar5   ( kstp,      Nbb )      ! ar5 diag
         IF( l_diaptr )  CALL dia_ptr   ( kstp,      Nbb )      ! Poleward adv/ldf TRansports diagnostics
#if defined key_xios
                         CALL dia_wri   ( kstp,      Nbb )      ! ocean model: outputs
#endif
      END DO
      IF( ln_tile ) CALL dom_tile_stop

#if ! defined key_xios
                         CALL dia_wri   ( kstp,      Nbb )      ! Ocean model outputs (default, tiling-unaware variant of 'dia_wri')
#endif
      IF( l_diadetide )  CALL dia_detide( kstp )                ! Weights computation for daily detiding of model diagnostics
      IF( l_diamlr  )    CALL dia_mlr                           ! Update time used in multiple-linear-regression analysis
      IF( l_diahsb  )    CALL dia_hsb   ( kstp, Nbb, Nbb )      ! - ML - global conservation diagnostics
      
!!gm : This does not only concern the dynamics ==>>> add a new title
!!gm2: why ouput restart before AGRIF update?
!!
!!jc: That would be better, but see comment above
!!
!!====>>>> to be modified for RK3
      IF( lrst_oce   )   CALL rst_write    ( kstp, Nbb, Nnn, Naa )   ! write output ocean restart file
      IF( ln_sto_eos )   CALL sto_rst_write( kstp )   ! write restart file for stochastic parameters

#if defined key_agrif
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! AGRIF recursive integration
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                         Kbb_a = Nbb; Kmm_a = Nbb; Krhs_a = Nrhs      ! agrif_oce module copies of time level indices
                         CALL Agrif_Integrate_ChildGrids( stp_RK3 )       ! allows to finish all the Child Grids before updating

#endif
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Control
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                         CALL stp_ctl      ( kstp, Nbb )
#if defined key_agrif
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! AGRIF update
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      IF( Agrif_NbStepint() == 0 .AND. nstop == 0 )   &
         &               CALL Agrif_update_all( )                  ! Update all components

#endif
      IF( ln_diaobs .AND. nstop == 0 )   &
         &               CALL dia_obs( kstp, Nbb )  ! obs-minus-model (assimilation) diags (after dynamics update)

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! File manipulation at the end of the first time step
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      IF( kstp == nit000 ) THEN                          ! 1st time step only
                                        CALL iom_close( numror )   ! close input  ocean restart file
         IF( lrxios )                   CALL iom_context_finalize( cr_ocerst_cxt )
         IF(lwm)                        CALL FLUSH    ( numond )   ! flush output namelist oce
         IF(lwm .AND. numoni /= -1 )    CALL FLUSH    ( numoni )   ! flush output namelist ice (if exist)
      ENDIF

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Coupled mode
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      IF( lk_oasis .AND. nstop == 0 )   CALL sbc_cpl_snd( kstp, Nbb, Nnn )     ! coupled mode : field exchanges
      !
#if defined key_xios
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Finalize contextes if end of simulation or error detected
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      IF( kstp == nitend .OR. nstop > 0 ) THEN
                      CALL iom_context_finalize(      cxios_context          ) ! needed for XIOS+AGRIF
      ENDIF
#endif
      !
      IF( ln_timing )   CALL timing_stop('stp_RK3')
      !
   END SUBROUTINE stp_RK3


   SUBROUTINE rk3_dia( kswitch )
      !!----------------------------------------------------------------------
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kswitch   ! on/off/init = 1/0/-1
      !!
      LOGICAL, SAVE ::   ll_trddyn, ll_trdtrc, ll_trdtra  ! call trd at stage 3 only
      LOGICAL, SAVE ::   ll_diaptr, ll_diaar5, ll_ldfeiv_dia
      !!----------------------------------------------------------------------
      !
      SELECT CASE( kswitch ) 
      CASE ( 1 )                ! diagnostic activated (on)
         l_trdtra = ll_trdtra
         l_trdtrc = ll_trdtrc
         l_trddyn = ll_trddyn
         l_diaptr = ll_diaptr
         l_diaar5 = ll_diaar5
         l_ldfeiv_dia = ll_ldfeiv_dia
      CASE ( 0 )                ! diagnostic desactivated (off)
         l_trdtra  = .FALSE.
         l_trdtrc  = .FALSE.
         l_trddyn  = .FALSE.
         l_diaptr  = .FALSE.
         l_diaar5  = .FALSE.
         l_ldfeiv_dia  = .FALSE.
      CASE ( -1 )
         ll_trdtra = l_trdtra
         ll_trdtrc = l_trdtrc
         ll_trddyn = l_trddyn
         ll_diaptr = l_diaptr
         ll_diaar5 = l_diaar5
         ! special diags for eiv
         ll_ldfeiv_dia = iom_use('uoce_eiv')    .OR. iom_use('ueiv_masstr')   .OR. &
            &            iom_use('ueiv_heattr') .OR. iom_use('ueiv_heattr3d') .OR. &
            &            iom_use('ueiv_salttr') .OR. iom_use('ueiv_salttr3d') .OR. &
            &            iom_use('voce_eiv')    .OR. iom_use('veiv_masstr')   .OR. &
            &            iom_use('veiv_heattr') .OR. iom_use('veiv_heattr3d') .OR. &
            &            iom_use('veiv_salttr') .OR. iom_use('veiv_salttr3d') .OR. &
            &            iom_use('woce_eiv')    .OR. iom_use('weiv_masstr')   .OR. &
            &            iom_use('sophteiv')    .OR. iom_use('sopsteiv')
         !
      END SELECT
      !
   END SUBROUTINE rk3_dia

#else
   !!----------------------------------------------------------------------
   !!   default option             EMPTY MODULE           qco not activated
   !!----------------------------------------------------------------------
#endif
#else
   !!----------------------------------------------------------------------
   !!   default option             EMPTY MODULE           RK3 not activated
   !!----------------------------------------------------------------------
#endif  
   !!======================================================================
END MODULE stprk3
