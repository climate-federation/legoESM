MODULE icethd
   !!======================================================================
   !!                  ***  MODULE icethd   ***
   !!   sea-ice : master routine for thermodynamics
   !!======================================================================
   !! History :  1.0  !  2000-01  (M.A. Morales Maqueda, H. Goosse, T. Fichefet) original code 1D
   !!            4.0  !  2018     (many people)       SI3 [aka Sea Ice cube]
   !!----------------------------------------------------------------------
#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3'                                       SI3 sea-ice model
   !!----------------------------------------------------------------------
   !!   ice_thd       : thermodynamics of sea ice
   !!   ice_thd_init  : initialisation of sea-ice thermodynamics
   !!----------------------------------------------------------------------
   USE par_ice        ! SI3 parameters
   USE phycst         ! physical constants
   USE par_oce
   USE ice            ! sea-ice: variables
   USE sbc_oce , ONLY : sss_m, sst_m, frq_m, sprecip
   USE sbc_ice , ONLY : qsr_ice, qns_ice, dqns_ice, evap_ice, qprec_ice, qml_ice, qcn_ice, qtr_ice_top
   USE ice1D          ! sea-ice: thermodynamics variables
   USE icethd_zdf     ! sea-ice: vertical heat diffusion
   USE icethd_dh      ! sea-ice: ice-snow growth and melt
   USE icethd_da      ! sea-ice: lateral melting
   USE icethd_sal     ! sea-ice: salinity
   USE icethd_do      ! sea-ice: growth in open water
   USE icethd_pnd     ! sea-ice: melt ponds
   USE iceitd  , ONLY : ice_itd_rem
   USE icecor         ! sea-ice: corrections
   USE icetab         ! sea-ice: 1D <==> 2D transformation
   USE icectl         ! sea-ice: control print
   !
   USE in_out_manager ! I/O manager
   USE iom            , ONLY : iom_miss_val, iom_put       ! I/O manager library
   USE lib_mpp        ! MPP library
   USE lbclnk         ! lateral boundary conditions (or mpp links)
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC   ice_thd         ! called by limstp module
   PUBLIC   ice_thd_init    ! called by ice_init

   !! for convergence tests
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) ::   ztice_cvgerr, ztice_cvgstp
   !! for sanity checks in drainage and flushing
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) ::   zcfl_flush, zcfl_drain, zsneg_flush, zsneg_drain
   LOGICAL , ALLOCATABLE, DIMENSION(:,:,:) ::   llmsk
   CHARACTER(LEN=50)      ::   clname="cfl_icesalt.ascii"    ! ascii filename
   INTEGER , DIMENSION(3) ::   iloc
   REAL(wp)               ::   zcfl_drain_max, zcfl_flush_max
   INTEGER                ::   numcfl                        ! outfile unit
   INTEGER, SAVE          ::   num_l3thd = -1               ! write-only oracle stream
   LOGICAL, SAVE          ::   ll_l3thd_opened = .FALSE.    ! NEWUNIT values may be negative
   INTEGER, SAVE          ::   num_l3zin = -1               ! exact ZDF-entry forcing stream
   LOGICAL, SAVE          ::   ll_l3zin_opened = .FALSE.
   INTEGER, SAVE          ::   num_l3rea = -1               ! kt3 arithmetic-replay operands
   LOGICAL, SAVE          ::   ll_l3rea_opened = .FALSE.

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "read_nml_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/ICE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ice_thd( kt )
      !!-------------------------------------------------------------------
      !!                ***  ROUTINE ice_thd  ***
      !!
      !! ** Purpose : This routine manages ice thermodynamics
      !!
      !! ** Action : - computation of oceanic sensible heat flux at the ice base
      !!                              energy budget in the leads
      !!                              net fluxes on top of ice and of ocean
      !!             - selection of grid cells with ice
      !!                - call ice_thd_zdf  for vertical heat diffusion
      !!                - call ice_thd_dh   for vertical ice growth and melt
      !!                - call ice_thd_pnd  for melt ponds
      !!                - call ice_thd_temp to  retrieve temperature from ice enthalpy
      !!                - call ice_thd_sal  for ice desalination
      !!                - call ice_thd_temp to  retrieve temperature from ice enthalpy
      !!                - call ice_thd_mono for extra lateral ice melt if active virtual thickness distribution
      !!                - call ice_thd_da   for lateral ice melt
      !!             - back to the geographic grid
      !!                - call ice_thd_rem  for remapping thickness distribution
      !!                - call ice_thd_do   for ice growth in leads
      !!-------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt    ! number of iteration
      !
      INTEGER  ::   ji, jj, jk, jl   ! dummy loop indices
      !!-------------------------------------------------------------------

      ! controls
      IF( ln_timing    )   CALL timing_start('icethd')                                                             ! timing
      IF( ln_icediachk )   CALL ice_cons_hsm(0, 'icethd', rdiag_v, rdiag_s, rdiag_t, rdiag_fv, rdiag_fs, rdiag_ft) ! conservation
      IF( ln_icediachk )   CALL ice_cons2D  (0, 'icethd',  diag_v,  diag_s,  diag_t,  diag_fv,  diag_fs,  diag_ft) ! conservation

      IF( kt == nit000 .AND. lwp ) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'ice_thd: sea-ice thermodynamics'
         WRITE(numout,*) '~~~~~~~'
      ENDIF

      ! convergence tests
      IF( ln_zdf_chkcvg ) THEN
         ALLOCATE( ztice_cvgerr(A2D(0),jpl) , ztice_cvgstp(A2D(0),jpl) )
         ztice_cvgerr = 0._wp ; ztice_cvgstp = 0._wp
      ENDIF
      !
      IF( ln_sal_chk )   CALL ice_thd_salchk( kt, 1 )
      CALL l3thd_dump_global( kt, 0 )                        ! registered ENTRY frame
      !
      !-------------------------------------------------------------------------------------------!
      ! Thermodynamic computation (only on grid points covered by ice) => loop over ice categories
      !-------------------------------------------------------------------------------------------!
      !
      CALL ice_thd_frazil             !--- frazil ice: collection thickness (ht_i_new) & fraction of frazil (fraz_frac)
      !
      DO jl = 1, jpl

#if defined key_si3_1D      
         DO_2D( 0, 0, 0, 0 )
            ! select ice covered grid points
            npti = 0 ; nptidx(:) = 0
            IF ( a_i(ji,jj,jl) > epsi10 ) THEN
               npti         = 1
               nptidx(npti) = (jj - 1) * jpi + ji
            ENDIF
#else
         ! select ice covered grid points
         npti = 0 ; nptidx(:) = 0
         DO_2D( 0, 0, 0, 0 )
            IF ( a_i(ji,jj,jl) > epsi10 ) THEN
               npti         = npti  + 1
               nptidx(npti) = (jj - 1) * jpi + ji
            ENDIF
         END_2D
#endif
         
         IF( npti > 0 ) THEN  ! If there is no ice, do nothing.
            !
                              CALL ice_thd_1d2d( jl, 1 )            ! --- Move to 1D arrays --- !
            !                                                       ! --- & Change units of e_i, e_s from J/m2 to J/m3 --- !
                              CALL l3zin_dump( kt, jl )
            !
            dh_s_tot  (1:npti) = 0._wp                              ! --- some init --- !  (important to have them here)
            dh_i_sum  (1:npti) = 0._wp ; dh_i_bom(1:npti) = 0._wp ; dh_i_itm(1:npti) = 0._wp
            dh_i_sub  (1:npti) = 0._wp ; dh_i_bog(1:npti) = 0._wp
            dh_snowice(1:npti) = 0._wp ; dh_s_sum(1:npti) = 0._wp ; dh_s_itm(1:npti) = 0._wp
            !
                              CALL ice_thd_zdf                      ! --- Ice-Snow temperature --- !
                              IF( kt == 3 ) CALL l3rea_dump( kt, jl )
                              CALL l3thd_dump_1d( kt, 1 )
            !
            IF( ln_icedH )    CALL ice_thd_dh                       ! --- Growing/Melting --- !
                              CALL l3thd_dump_1d( kt, 2 )
            !
                              CALL ice_thd_temp                     ! --- Temperature update --- !
                              CALL l3thd_dump_1d( kt, 3 )
            !
                              CALL ice_thd_sal                      ! --- Ice salinity --- !
                              CALL l3thd_dump_1d( kt, 4 )
            !
                              CALL ice_thd_temp                     ! --- Temperature update --- !
                              CALL l3thd_dump_1d( kt, 5 )
            !
            IF( ln_icedH .AND. ln_virtual_itd ) &
               &              CALL ice_thd_mono                     ! --- Extra lateral melting if virtual_itd --- !
            !
            IF( ln_icedA )    CALL ice_thd_da                       ! --- Lateral melting --- !
            !
                              CALL ice_thd_1d2d( jl, 2 )            ! --- Change units of e_i, e_s from J/m3 to J/m2 --- !
            !                                                       ! --- & Move to 2D arrays --- !
         ENDIF
         !
#if defined key_si3_1D
         END_2D
#endif
         !
      END DO
      !
      IF( ln_icediachk )   CALL ice_cons_hsm(1, 'icethd', rdiag_v, rdiag_s, rdiag_t, rdiag_fv, rdiag_fs, rdiag_ft)
      IF( ln_icediachk )   CALL ice_cons2D  (1, 'icethd',  diag_v,  diag_s,  diag_t,  diag_fv,  diag_fs,  diag_ft)
      !
      IF ( ln_pnd .AND. ln_icedH ) &
         &                    CALL ice_thd_pnd( kt )                ! --- Melt ponds --- !
      !
      IF( jpl > 1  )          CALL ice_itd_rem( kt )                ! --- Transport ice between thickness categories --- !
      !
      IF( ln_icedO )          CALL ice_thd_do                       ! --- Frazil ice growth in leads --- !
                              CALL l3thd_dump_global( kt, 6 )
      !
                              CALL ice_cor( kt , 2 )                ! --- Corrections --- !
      !
      oa_i(A2D(0),:) = oa_i(A2D(0),:) + a_i(A2D(0),:) * rDt_ice     ! --- Ice natural aging incrementation
      !
      !                                                             ! --- LBC for the halos --- !
      CALL lbc_lnk( 'icethd', a_i , 'T', 1._wp, v_i , 'T', 1._wp, v_s , 'T', 1._wp, sv_i, 'T', 1._wp, oa_i, 'T', 1._wp, &
         &                    t_su, 'T', 1._wp, a_ip, 'T', 1._wp, v_ip, 'T', 1._wp, v_il, 'T', 1._wp )
      CALL lbc_lnk( 'icethd', e_i , 'T', 1._wp, e_s , 'T', 1._wp, szv_i , 'T', 1._wp )
      !
      at_i(:,:) = SUM( a_i, dim=3 )
      DO_2D( 0, 0, 0, 0 )                                           ! --- Ice velocity corrections
         IF( at_i(ji,jj) == 0._wp ) THEN   ! if ice has melted
            IF( at_i(ji+1,jj) == 0._wp )   u_ice(ji  ,jj) = 0._wp   ! right side
            IF( at_i(ji-1,jj) == 0._wp )   u_ice(ji-1,jj) = 0._wp   ! left side
            IF( at_i(ji,jj+1) == 0._wp )   v_ice(ji,jj  ) = 0._wp   ! upper side
            IF( at_i(ji,jj-1) == 0._wp )   v_ice(ji,jj-1) = 0._wp   ! bottom side
         ENDIF
      END_2D
      CALL lbc_lnk( 'icethd', u_ice, 'U', -1.0_wp, v_ice, 'V', -1.0_wp )
      !
      ! convergence tests
      IF( ln_zdf_chkcvg ) THEN
         CALL iom_put( 'tice_cvgerr', ztice_cvgerr ) ; DEALLOCATE( ztice_cvgerr )
         CALL iom_put( 'tice_cvgstp', ztice_cvgstp ) ; DEALLOCATE( ztice_cvgstp )
      ENDIF
      !
      ! sanity checks for salt drainage and flushing
      IF( ln_sal_chk )   CALL ice_thd_salchk( kt, 2 )
      
      ! controls
      IF( ln_icectl )   CALL ice_prt    (kt, iiceprt, jiceprt, 1, ' - ice thermodyn. - ') ! prints
      IF( sn_cfctl%l_prtctl )   &
        &               CALL ice_prt3D  ('icethd')                                        ! prints
      CALL l3thd_dump_global( kt, 7 )                                                      ! registered EXIT frame
      IF( ln_timing )   CALL timing_stop('icethd')                                        ! timing
      !
   END SUBROUTINE ice_thd


   SUBROUTINE l3thd_open
      INTEGER :: ios
      IF( .NOT. ll_l3thd_opened ) THEN
         OPEN( NEWUNIT=num_l3thd, FILE='oracle_si3_thd_frames.bin', STATUS='REPLACE', &
            &  ACCESS='STREAM', FORM='UNFORMATTED', ACTION='WRITE', IOSTAT=ios )
         IF( ios /= 0 ) CALL ctl_stop( 'l3thd_open: cannot open oracle_si3_thd_frames.bin' )
         ll_l3thd_opened = .TRUE.
      ENDIF
   END SUBROUTINE l3thd_open


   SUBROUTINE l3thd_header( kt, kstage, kpayload )
      INTEGER, INTENT(in) :: kt, kstage, kpayload
      CHARACTER(LEN=16), PARAMETER :: cmagic = 'NEMO_L3THD_001'
      CALL l3thd_open
      WRITE(num_l3thd) cmagic
      WRITE(num_l3thd) 1, kt, kstage, kpayload, jpi, jpj, jpl, nlay_i, nlay_s, npti, STORAGE_SIZE(1._wp)
   END SUBROUTINE l3thd_header


   SUBROUTINE l3thd_dump_global( kt, kstage )
      INTEGER, INTENT(in) :: kt, kstage
      CALL l3thd_header( kt, kstage, 0 )
      WRITE(num_l3thd) a_i, v_i, v_s, sv_i, oa_i, t_su, a_ip, v_ip, v_il, e_i, e_s, szv_i
      IF( kstage == 7 ) THEN
         ! NEMO can revisit nitend during terminal handling.  Keep the stream
         ! open until process teardown so STATUS='REPLACE' cannot truncate it.
         FLUSH(num_l3thd)
      ENDIF
   END SUBROUTINE l3thd_dump_global


   SUBROUTINE l3thd_dump_1d( kt, kstage )
      INTEGER, INTENT(in) :: kt, kstage
      CALL l3thd_header( kt, kstage, 1 )
      WRITE(num_l3thd) a_i_1d(1:npti), h_i_1d(1:npti), h_s_1d(1:npti), t_su_1d(1:npti), &
         & e_i_1d(1:npti,1:nlay_i), e_s_1d(1:npti,1:nlay_s), sz_i_1d(1:npti,1:nlay_i)
   END SUBROUTINE l3thd_dump_1d


   SUBROUTINE l3zin_dump( kt, kl )
      INTEGER, INTENT(in) :: kt, kl
      INTEGER :: ios
      CHARACTER(LEN=16), PARAMETER :: cmagic = 'NEMO_L3ZIN_002'
      IF( .NOT. ll_l3zin_opened ) THEN
         OPEN( NEWUNIT=num_l3zin, FILE='oracle_si3_zdf_inputs.bin', STATUS='REPLACE', &
            & ACCESS='STREAM', FORM='UNFORMATTED', ACTION='WRITE', IOSTAT=ios )
         IF( ios /= 0 ) CALL ctl_stop( 'l3zin_dump: cannot open ZDF-input stream' )
         ll_l3zin_opened = .TRUE.
      ENDIF
      WRITE(num_l3zin) cmagic
      WRITE(num_l3zin) 2, kt, kl, npti, STORAGE_SIZE(1._wp), ( 13 + nlay_s ) * npti
      WRITE(num_l3zin) qns_ice_1d(1:npti), qsr_ice_1d(1:npti), dqns_ice_1d(1:npti), &
         & qtr_ice_top_1d(1:npti), t_bo_1d(1:npti), sss_1d(1:npti), &
         & evap_ice_1d(1:npti), sprecip_1d(1:npti), qprec_ice_1d(1:npti), &
         & qcn_ice_1d(1:npti), qsb_ice_bot_1d(1:npti), fhld_1d(1:npti), qml_ice_1d(1:npti), &
         & t_s_1d(1:npti,1:nlay_s)
      FLUSH(num_l3zin)
   END SUBROUTINE l3zin_dump


   SUBROUTINE l3rea_dump( kt, kl )
      INTEGER, INTENT(in) :: kt, kl
      INTEGER :: ios
      CHARACTER(LEN=16), PARAMETER :: cmagic = 'NEMO_L3REA_001'
      IF( .NOT. ll_l3rea_opened ) THEN
         OPEN( NEWUNIT=num_l3rea, FILE='oracle_si3_reassoc_operands.bin', STATUS='REPLACE', &
            & ACCESS='STREAM', FORM='UNFORMATTED', ACTION='WRITE', IOSTAT=ios )
         IF( ios /= 0 ) CALL ctl_stop( 'l3rea_dump: cannot open arithmetic-replay stream' )
         ll_l3rea_opened = .TRUE.
      ENDIF
      WRITE(num_l3rea) cmagic
      WRITE(num_l3rea) 1, kt, kl, npti, nlay_i, nlay_s, STORAGE_SIZE(1._wp), 18 * npti
      WRITE(num_l3rea) t_su_1d(1:npti), t_i_1d(1:npti,1:nlay_i), &
         & t_s_1d(1:npti,1:nlay_s), sz_i_1d(1:npti,1:nlay_i), &
         & e_i_1d(1:npti,1:nlay_i), e_s_1d(1:npti,1:nlay_s), &
         & qns_ice_1d(1:npti), dqns_ice_1d(1:npti)
      FLUSH(num_l3rea)
   END SUBROUTINE l3rea_dump


   SUBROUTINE ice_thd_temp
      !!-----------------------------------------------------------------------
      !!                   ***  ROUTINE ice_thd_temp ***
      !!
      !! ** Purpose :   Computes sea ice temperature (Kelvin) from enthalpy
      !!
      !! ** Method  :   Formula (Bitz and Lipscomb, 1999)
      !!-------------------------------------------------------------------
      INTEGER  ::   ii, jk   ! dummy loop indices
      REAL(wp) ::   ztmelts, zbbb, zccc  ! local scalar
      !!-------------------------------------------------------------------
      ! Recover ice temperature
      DO jk = 1, nlay_i
         DO ii = 1, npti
            IF( h_i_1d(ii) > 0._wp ) THEN
               ztmelts       = -rTmlt * sz_i_1d(ii,jk)
               ! Conversion q(S,T) -> T (second order equation)
               zbbb          = ( rcp - rcpi ) * ztmelts + e_i_1d(ii,jk) * r1_rhoi - rLfus
               zccc          = SQRT( MAX( zbbb * zbbb - 4._wp * rcpi * rLfus * ztmelts, 0._wp ) )
               t_i_1d(ii,jk) = rt0 - ( zbbb + zccc ) * 0.5_wp * r1_rcpi
            ELSE
               t_i_1d(ii,jk) = rt0
            ENDIF
         END DO
      END DO
      !
   END SUBROUTINE ice_thd_temp


   SUBROUTINE ice_thd_mono
      !!-----------------------------------------------------------------------
      !!                   ***  ROUTINE ice_thd_mono ***
      !!
      !! ** Purpose :   Lateral melting in case virtual_itd
      !!                          ( dA = A/2h dh )
      !!-----------------------------------------------------------------------
      INTEGER  ::   ii                 ! dummy loop indices
      REAL(wp) ::   zhi_bef            ! ice thickness before thermo
      REAL(wp) ::   zdh_mel, zda_mel   ! net melting
      REAL(wp) ::   zvi, zvs           ! ice/snow volumes
      !!-----------------------------------------------------------------------
      !
      DO ii = 1, npti
         zdh_mel = MIN( 0._wp, dh_i_itm(ii) + dh_i_sum(ii) + dh_i_bom(ii) + dh_snowice(ii) + dh_i_sub(ii) )
         IF( zdh_mel < 0._wp .AND. a_i_1d(ii) > 0._wp )  THEN
            zvi          = a_i_1d(ii) * h_i_1d(ii)
            zvs          = a_i_1d(ii) * h_s_1d(ii)
            ! lateral melting = concentration change
            zhi_bef     = h_i_1d(ii) - zdh_mel
            zda_mel     = MAX( -a_i_1d(ii) , a_i_1d(ii) * zdh_mel / ( 2._wp * MAX( zhi_bef, epsi20 ) ) )
            a_i_1d(ii)  = MAX( epsi20, a_i_1d(ii) + zda_mel )
            ! adjust thickness
            h_i_1d(ii) = zvi / a_i_1d(ii)
            h_s_1d(ii) = zvs / a_i_1d(ii)
            ! retrieve total concentration
            at_i_1d(ii) = a_i_1d(ii)
         END IF
      END DO
      !
   END SUBROUTINE ice_thd_mono

   SUBROUTINE ice_thd_salchk( kt, kn )
      !!-----------------------------------------------------------------------
      !!                   ***  ROUTINE ice_thd_salchk ***
      !!
      !! ** Purpose :   checking salt drainage and flushing
      !!-----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt, kn 
      !
      !INTEGER ::   jk   ! dummy loop indices
      !!-----------------------------------------------------------------------
     
      ! sanity checks for salt flushing and drainage
      IF( kn == 1 ) THEN
         !
         ALLOCATE( llmsk(A2D(0),jpl) )
         ALLOCATE( zcfl_flush(A2D(0),jpl) , zcfl_drain(A2D(0),jpl), zsneg_flush(A2D(0),jpl) , zsneg_drain(A2D(0),jpl) )
         !
         zcfl_flush = 0._wp ; zcfl_drain = 0._wp
         zsneg_flush = 0._wp ; zsneg_drain = 0._wp
         !
      ELSEIF( kn == 2 ) THEN
         !
         CALL iom_put( 'sice_flush_neg', zsneg_flush ) ; DEALLOCATE( zsneg_flush )
         CALL iom_put( 'sice_drain_neg', zsneg_drain ) ; DEALLOCATE( zsneg_drain )
         !
         CALL iom_put( 'cfl_flush', zcfl_flush )
         CALL iom_put( 'cfl_drain', zcfl_drain )

         !                    ! calculate maximum values and locations
         llmsk(Nis0:Nie0,Njs0:Nje0,:) = h_i(Nis0:Nie0,Njs0:Nje0,:) > rn_himin        ! define only where h > 0.10m
         CALL mpp_maxloc( 'icethd', zcfl_drain, llmsk, zcfl_drain_max, iloc )
         CALL mpp_maxloc( 'icethd', zcfl_flush, llmsk, zcfl_flush_max, iloc )

         IF( lwp ) THEN       ! write out to file
            WRITE(numcfl,FMT='(2x,i6,3x,a10,4x,f8.4,1x,i4,1x,i4,1x,i4)') kt, 'Max Cdrain', zcfl_drain_max, iloc(1), iloc(2), iloc(3)
            WRITE(numcfl,FMT='(11x,     a10,4x,f8.4,1x,i4,1x,i4,1x,i4)')     'Max Cflush', zcfl_flush_max, iloc(1), iloc(2), iloc(3)
         ENDIF
         DEALLOCATE( zcfl_flush, zcfl_drain )
         DEALLOCATE( llmsk )

         IF( kt == nitend .AND. lwp )   CLOSE( numcfl )
      ENDIF
      
   END SUBROUTINE ice_thd_salchk

   SUBROUTINE ice_thd_1d2d( kl, kn )
      !!-----------------------------------------------------------------------
      !!                   ***  ROUTINE ice_thd_1d2d ***
      !!
      !! ** Purpose :   move arrays from 1d to 2d and the reverse
      !!-----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kl   ! index of the ice category
      INTEGER, INTENT(in) ::   kn   ! 1= from 2D to 1D   ;   2= from 1D to 2D
      !
      INTEGER ::   jk   ! dummy loop indices
      !!-----------------------------------------------------------------------
      !
      SELECT CASE( kn )
      !                    !---------------------!
      CASE( 1 )            !==  from 2D to 1D  ==!
         !                 !---------------------!
         CALL tab_2d_1d( npti, nptidx(1:npti), at_i_1d(1:npti), at_i             )
         CALL tab_2d_1d( npti, nptidx(1:npti), a_i_1d (1:npti), a_i (:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), h_i_1d (1:npti), h_i (:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), h_s_1d (1:npti), h_s (:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), t_su_1d(1:npti), t_su(:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), s_i_1d (1:npti), s_i (:,:,kl)     )
         DO jk = 1, nlay_s
            CALL tab_2d_1d( npti, nptidx(1:npti), t_s_1d(1:npti,jk), t_s(:,:,jk,kl)    )
            CALL tab_2d_1d( npti, nptidx(1:npti), e_s_1d(1:npti,jk), e_s(:,:,jk,kl)    )
         END DO
         DO jk = 1, nlay_i
            CALL tab_2d_1d( npti, nptidx(1:npti), t_i_1d  (1:npti,jk), t_i  (:,:,jk,kl)  )
            CALL tab_2d_1d( npti, nptidx(1:npti), e_i_1d  (1:npti,jk), e_i  (:,:,jk,kl)  )
            CALL tab_2d_1d( npti, nptidx(1:npti), sz_i_1d (1:npti,jk), sz_i (:,:,jk,kl)  )
         END DO
         !
         CALL tab_2d_1d( npti, nptidx(1:npti), qprec_ice_1d  (1:npti), qprec_ice            )
         CALL tab_2d_1d( npti, nptidx(1:npti), qsr_ice_1d    (1:npti), qsr_ice (:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), qns_ice_1d    (1:npti), qns_ice (:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), evap_ice_1d   (1:npti), evap_ice(:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), dqns_ice_1d   (1:npti), dqns_ice(:,:,kl)     )
         CALL tab_2d_1d( npti, nptidx(1:npti), t_bo_1d       (1:npti), t_bo                 )
         CALL tab_2d_1d( npti, nptidx(1:npti), sprecip_1d    (1:npti), sprecip              )
         CALL tab_2d_1d( npti, nptidx(1:npti), qsb_ice_bot_1d(1:npti), qsb_ice_bot          )
         CALL tab_2d_1d( npti, nptidx(1:npti), fhld_1d       (1:npti), fhld                 )

         CALL tab_2d_1d( npti, nptidx(1:npti), qml_ice_1d    (1:npti), qml_ice    (:,:,kl) )
         CALL tab_2d_1d( npti, nptidx(1:npti), qcn_ice_1d    (1:npti), qcn_ice    (:,:,kl) )
         CALL tab_2d_1d( npti, nptidx(1:npti), qtr_ice_top_1d(1:npti), qtr_ice_top(:,:,kl) )
         !
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_snw_sni_1d(1:npti), wfx_snw_sni   )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_snw_sum_1d(1:npti), wfx_snw_sum   )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_sub_1d    (1:npti), wfx_sub       )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_snw_sub_1d(1:npti), wfx_snw_sub   )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_ice_sub_1d(1:npti), wfx_ice_sub   )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_err_sub_1d(1:npti), wfx_err_sub   )
         !
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_bog_1d (1:npti), wfx_bog          )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_bom_1d (1:npti), wfx_bom          )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_sum_1d (1:npti), wfx_sum          )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_sni_1d (1:npti), wfx_sni          )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_res_1d (1:npti), wfx_res          )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_spr_1d (1:npti), wfx_spr          )
         CALL tab_2d_1d( npti, nptidx(1:npti), wfx_lam_1d (1:npti), wfx_lam          )
         !
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_bog_1d (1:npti), sfx_bog          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_bom_1d (1:npti), sfx_bom          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_sum_1d (1:npti), sfx_sum          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_sni_1d (1:npti), sfx_sni          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_bri_1d (1:npti), sfx_bri          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_res_1d (1:npti), sfx_res          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_sub_1d (1:npti), sfx_sub          )
         CALL tab_2d_1d( npti, nptidx(1:npti), sfx_lam_1d (1:npti), sfx_lam          )
         !
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_thd_1d    (1:npti), hfx_thd       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_spr_1d    (1:npti), hfx_spr       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_sum_1d    (1:npti), hfx_sum       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_bom_1d    (1:npti), hfx_bom       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_bog_1d    (1:npti), hfx_bog       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_dif_1d    (1:npti), hfx_dif       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_opw_1d    (1:npti), hfx_opw       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_snw_1d    (1:npti), hfx_snw       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_sub_1d    (1:npti), hfx_sub       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_res_1d    (1:npti), hfx_res       )
         CALL tab_2d_1d( npti, nptidx(1:npti), hfx_err_dif_1d(1:npti), hfx_err_dif   )
         !
         ! ocean surface fields
         CALL tab_2d_1d( npti, nptidx(1:npti), sst_1d(1:npti), sst_m )
         CALL tab_2d_1d( npti, nptidx(1:npti), sss_1d(1:npti), sss_m )
         CALL tab_2d_1d( npti, nptidx(1:npti), frq_m_1d(1:npti), frq_m )
         !
         ! to update ice age
         CALL tab_2d_1d( npti, nptidx(1:npti), o_i_1d (1:npti), o_i (:,:,kl) )
         CALL tab_2d_1d( npti, nptidx(1:npti), oa_i_1d(1:npti), oa_i(:,:,kl) )
         !
         ! --- Change units of e_i, e_s from J/m2 to J/m3 --- !
         ! Here we make sure that we don't divide by very small, but physically
         ! meaningless, products of sea ice thicknesses/snow depths and sea ice
         ! concentration
         DO jk = 1, nlay_i
            WHERE( (h_i_1d(1:npti) * a_i_1d(1:npti)) > epsi20 )
               e_i_1d(1:npti,jk) = e_i_1d(1:npti,jk) / (h_i_1d(1:npti) * a_i_1d(1:npti)) * nlay_i
            ELSEWHERE
               e_i_1d(1:npti,jk) = 0._wp
            ENDWHERE
         END DO
         DO jk = 1, nlay_s
            WHERE( (h_s_1d(1:npti) * a_i_1d(1:npti)) > epsi20 )
               e_s_1d(1:npti,jk) = e_s_1d(1:npti,jk) / (h_s_1d(1:npti) * a_i_1d(1:npti)) * nlay_s
            ELSEWHERE
               e_s_1d(1:npti,jk) = 0._wp
            ENDWHERE
         END DO
         !
         !                 !---------------------!
      CASE( 2 )            !==  from 1D to 2D  ==!
         !                 !---------------------!
         ! --- Change units of e_i, e_s from J/m3 to J/m2 --- !
         DO jk = 1, nlay_i
            e_i_1d(1:npti,jk) = e_i_1d(1:npti,jk) * h_i_1d(1:npti) * a_i_1d(1:npti) * r1_nlay_i
         END DO
         DO jk = 1, nlay_s
            e_s_1d(1:npti,jk) = e_s_1d(1:npti,jk) * h_s_1d(1:npti) * a_i_1d(1:npti) * r1_nlay_s
         END DO
         !
         ! Change thickness to volume (replaces routine ice_var_eqv2glo)
         v_i_1d  (1:npti)   = h_i_1d (1:npti)   * a_i_1d (1:npti)
         v_s_1d  (1:npti)   = h_s_1d (1:npti)   * a_i_1d (1:npti)
         sv_i_1d (1:npti)   = s_i_1d (1:npti)   * v_i_1d (1:npti)
         oa_i_1d (1:npti)   = o_i_1d (1:npti)   * a_i_1d (1:npti)
         DO jk = 1, nlay_i
            szv_i_1d(1:npti,jk) = sz_i_1d(1:npti,jk) * v_i_1d (1:npti) * r1_nlay_i
         ENDDO
         CALL tab_1d_2d( npti, nptidx(1:npti), at_i_1d(1:npti), at_i             )
         CALL tab_1d_2d( npti, nptidx(1:npti), a_i_1d (1:npti), a_i (:,:,kl)     )
         CALL tab_1d_2d( npti, nptidx(1:npti), h_i_1d (1:npti), h_i (:,:,kl)     )
         CALL tab_1d_2d( npti, nptidx(1:npti), h_s_1d (1:npti), h_s (:,:,kl)     )
         CALL tab_1d_2d( npti, nptidx(1:npti), t_su_1d(1:npti), t_su(:,:,kl)     )
         CALL tab_1d_2d( npti, nptidx(1:npti), s_i_1d (1:npti), s_i (:,:,kl)     )
         DO jk = 1, nlay_s
            CALL tab_1d_2d( npti, nptidx(1:npti), t_s_1d(1:npti,jk), t_s(:,:,jk,kl)    )
            CALL tab_1d_2d( npti, nptidx(1:npti), e_s_1d(1:npti,jk), e_s(:,:,jk,kl)    )
         END DO
         DO jk = 1, nlay_i
            CALL tab_1d_2d( npti, nptidx(1:npti), t_i_1d  (1:npti,jk), t_i  (:,:,jk,kl)  )
            CALL tab_1d_2d( npti, nptidx(1:npti), e_i_1d  (1:npti,jk), e_i  (:,:,jk,kl)  )
            CALL tab_1d_2d( npti, nptidx(1:npti), sz_i_1d (1:npti,jk), sz_i (:,:,jk,kl)  )
            CALL tab_1d_2d( npti, nptidx(1:npti), szv_i_1d(1:npti,jk), szv_i(:,:,jk,kl)  )
         END DO
         !
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_snw_sni_1d(1:npti), wfx_snw_sni )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_snw_sum_1d(1:npti), wfx_snw_sum )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_sub_1d    (1:npti), wfx_sub     )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_snw_sub_1d(1:npti), wfx_snw_sub )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_ice_sub_1d(1:npti), wfx_ice_sub )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_err_sub_1d(1:npti), wfx_err_sub )
         !
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_bog_1d (1:npti), wfx_bog        )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_bom_1d (1:npti), wfx_bom        )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_sum_1d (1:npti), wfx_sum        )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_sni_1d (1:npti), wfx_sni        )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_res_1d (1:npti), wfx_res        )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_spr_1d (1:npti), wfx_spr        )
         CALL tab_1d_2d( npti, nptidx(1:npti), wfx_lam_1d (1:npti), wfx_lam        )
         !
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_bog_1d (1:npti), sfx_bog        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_bom_1d (1:npti), sfx_bom        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_sum_1d (1:npti), sfx_sum        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_sni_1d (1:npti), sfx_sni        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_bri_1d (1:npti), sfx_bri        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_res_1d (1:npti), sfx_res        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_sub_1d (1:npti), sfx_sub        )
         CALL tab_1d_2d( npti, nptidx(1:npti), sfx_lam_1d (1:npti), sfx_lam        )
         !
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_thd_1d    (1:npti), hfx_thd     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_spr_1d    (1:npti), hfx_spr     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_sum_1d    (1:npti), hfx_sum     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_bom_1d    (1:npti), hfx_bom     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_bog_1d    (1:npti), hfx_bog     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_dif_1d    (1:npti), hfx_dif     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_opw_1d    (1:npti), hfx_opw     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_snw_1d    (1:npti), hfx_snw     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_sub_1d    (1:npti), hfx_sub     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_res_1d    (1:npti), hfx_res     )
         CALL tab_1d_2d( npti, nptidx(1:npti), hfx_err_dif_1d(1:npti), hfx_err_dif )
         !
         CALL tab_1d_2d( npti, nptidx(1:npti), qns_ice_1d    (1:npti), qns_ice    (:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), qtr_ice_bot_1d(1:npti), qtr_ice_bot(:,:,kl) )
         ! effective conductivity and 1st layer temperature (ln_cndflx=T)
         CALL tab_1d_2d( npti, nptidx(1:npti), cnd_ice_1d(1:npti), cnd_ice(:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), t1_ice_1d (1:npti), t1_ice (:,:,kl) )
         ! Melt ponds
         CALL tab_1d_2d( npti, nptidx(1:npti), dh_i_sum  (1:npti) , dh_i_sum_2d(:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), dh_s_sum  (1:npti) , dh_s_sum_2d(:,:,kl) )
         ! SIMIP diagnostics
         CALL tab_1d_2d( npti, nptidx(1:npti), t_si_1d       (1:npti), t_si       (:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), qcn_ice_bot_1d(1:npti), qcn_ice_bot(:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), qcn_ice_top_1d(1:npti), qcn_ice_top(:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), qml_ice_1d    (1:npti), qml_ice    (:,:,kl) )
         ! extensive variables
         CALL tab_1d_2d( npti, nptidx(1:npti), v_i_1d (1:npti), v_i (:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), v_s_1d (1:npti), v_s (:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), sv_i_1d(1:npti), sv_i(:,:,kl) )
         CALL tab_1d_2d( npti, nptidx(1:npti), oa_i_1d(1:npti), oa_i(:,:,kl) )
         ! check convergence of heat diffusion scheme
         IF( ln_zdf_chkcvg ) THEN
            CALL tab_1d_2d( npti, nptidx(1:npti), tice_cvgerr_1d(1:npti), ztice_cvgerr(:,:,kl) )
            CALL tab_1d_2d( npti, nptidx(1:npti), tice_cvgstp_1d(1:npti), ztice_cvgstp(:,:,kl) )
         ENDIF
         ! sanity check for salt scheme
         IF( ln_sal_chk ) THEN
            CALL tab_1d_2d( npti, nptidx(1:npti), sneg_flush_1d (1:npti), zsneg_flush (:,:,kl) )
            CALL tab_1d_2d( npti, nptidx(1:npti), sneg_drain_1d (1:npti), zsneg_drain (:,:,kl) )
            CALL tab_1d_2d( npti, nptidx(1:npti), cfl_flush_1d(1:npti), zcfl_flush(:,:,kl) )
            CALL tab_1d_2d( npti, nptidx(1:npti), cfl_drain_1d(1:npti), zcfl_drain(:,:,kl) )
         ENDIF
         !
      END SELECT
      !
   END SUBROUTINE ice_thd_1d2d


   SUBROUTINE ice_thd_init
      !!-------------------------------------------------------------------
      !!                   ***  ROUTINE ice_thd_init ***
      !!
      !! ** Purpose :   Physical constants and parameters associated with
      !!                ice thermodynamics
      !!
      !! ** Method  :   Read the namthd namelist and check the parameters
      !!                called at the first timestep (nit000)
      !!
      !! ** input   :   Namelist namthd
      !!-------------------------------------------------------------------
      INTEGER  ::   ios   ! Local integer output status for namelist read
      !!
      NAMELIST/namthd/ ln_icedH, ln_icedA, ln_icedO, ln_leadhfx
      !!-------------------------------------------------------------------
      !
      READ_NML_REF(numnam_ice,namthd)
      READ_NML_CFG(numnam_ice,namthd)
      IF(lwm) WRITE( numoni, namthd )
      !
      IF(lwp) THEN                          ! control print
         WRITE(numout,*)
         WRITE(numout,*) 'ice_thd_init: Ice Thermodynamics'
         WRITE(numout,*) '~~~~~~~~~~~~'
         WRITE(numout,*) '   Namelist namthd:'
         WRITE(numout,*) '      activate ice thick change from top/bot (T) or not (F)                ln_icedH   = ', ln_icedH
         WRITE(numout,*) '      activate lateral melting (T) or not (F)                              ln_icedA   = ', ln_icedA
         WRITE(numout,*) '      activate ice growth in open-water (T) or not (F)                     ln_icedO   = ', ln_icedO
         WRITE(numout,*) '      heat in the leads is used to melt sea-ice before warming the ocean   ln_leadhfx = ', ln_leadhfx
     ENDIF
      !
                       CALL ice_thd_zdf_init   ! set ice heat diffusion parameters
      IF( ln_icedA )   CALL ice_thd_da_init    ! set ice lateral melting parameters
      IF( ln_icedO )   CALL ice_thd_do_init    ! set ice growth in open water parameters
                       CALL ice_thd_sal_init   ! set ice salinity parameters
                       CALL ice_thd_pnd_init   ! set melt ponds parameters
      !
      IF( ln_sal_chk ) THEN
         ! create output ascii file
         CALL ctl_opn( numcfl, clname, 'UNKNOWN', 'FORMATTED', 'SEQUENTIAL', 1, numout, lwp, 1 )
         WRITE(numcfl,*) 'Timestep  Direction   Max C     i    j    k'
         WRITE(numcfl,*) '*******************************************'
      ENDIF
   END SUBROUTINE ice_thd_init

#else
   !!----------------------------------------------------------------------
   !!   Default option         Dummy module          NO  SI3 sea-ice model
   !!----------------------------------------------------------------------
#endif

   !!======================================================================
END MODULE icethd
