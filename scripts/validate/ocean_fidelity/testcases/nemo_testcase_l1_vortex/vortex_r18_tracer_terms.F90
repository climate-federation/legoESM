MODULE vortex_r18_tracer_terms
   !!======================================================================
   !! Read-only stage-1/2/3 TRACER-term writer for VORTEX_SMT round 7
   !! (lane round 218).
   !!
   !! Rounds 192 and 200 instrumented the MOMENTUM side of stp_RK3_stg
   !! (vortex_r8_stage_terms, vortex_r16_stage_terms).  Neither carries a
   !! single tracer array, so no admitted seamount record says what NEMO's
   !! FCT tracer advection was handed or what it produced.  This writer adds
   !! exactly that, at every stage:
   !!
   !!   * the three advective transports zFu, zFv, zFw AS tra_adv receives
   !!     them -- i.e. after tra_adv_trp has updated (flux form) or computed
   !!     (vector-invariant form) them, which is the only point at which
   !!     zFw exists on the vector card (stprk3_stg.F90:287 comment);
   !!   * the before/now tracer fields and the three r3t time levels, which
   !!     are every non-geometry operand of the FCT statements;
   !!   * the tracer right-hand side after tra_adv + tra_sbc_RK3;
   !!   * the after-tracer at the end of the stage (which is the stage 1/2
   !!     thickness-weighted time step, or stage 3's tra_zdf).
   !!
   !! Every write happens AFTER a compiled statement has completed, on a
   !! copy of what NEMO already holds; this module performs no arithmetic on
   !! any field NEMO consumes, and it writes only at kstp == nit000.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE par_oce        , ONLY : jpi, jpj, jpk
   USE oce            , ONLY : ww
   USE dom_oce        , ONLY : r3t
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: vortex_r18_tracer_begin
   PUBLIC :: vortex_r18_tracer_rhs
   PUBLIC :: vortex_r18_tracer_state
   PUBLIC :: vortex_r18_tracer_finish

   INTEGER, SAVE :: nunit = -1
   INTEGER, SAVE :: nstage = 0
   LOGICAL, SAVE :: lopen = .FALSE.

CONTAINS

   SUBROUTINE vortex_r18_tracer_begin( kstp, kstg, Kbb, Kmm, Kaa, Krhs, pts, &
      &                                pFu, pFv, pFw )
      INTEGER, INTENT(in) :: kstp, kstg, Kbb, Kmm, Kaa, Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pts
      REAL(wp), DIMENSION(:,:,:)    , INTENT(in) :: pFu, pFv, pFw
      INTEGER :: ios
      CHARACTER(LEN=16) :: clmagic
      CHARACTER(LEN=128) :: clfile

      IF( lopen ) CALL ctl_stop( 'VORTEX R18 tracer writer opened twice' )
      IF( kstg < 1 .OR. kstg > 3 ) &
         CALL ctl_stop( 'VORTEX R18 tracer writer accepts stages 1, 2 and 3 only' )
      IF( STORAGE_SIZE(1._wp) /= 64 ) &
         CALL ctl_stop( 'VORTEX R18 tracer writer requires 64-bit wp' )
      IF( SIZE(pts,4) < 2 ) &
         CALL ctl_stop( 'VORTEX R18 tracer writer needs both active tracers' )

      WRITE(clfile,'("oracle_tracer_terms_kt",I8.8,"_s",I1,".bin")') kstp, kstg
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM', &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open VORTEX R18 tracer-term file' )

      nstage = kstg
      lopen = .TRUE.
      clmagic = 'NEMO_L1_TRATRM1'
      WRITE(nunit) clmagic, 1, kstp, kstg, Kbb, Kmm, Kaa, Krhs, &
         &         jpi, jpj, jpk, 15, 0, 0, 0, STORAGE_SIZE(1._wp)
      CALL write3( 'zfu', pFu )
      CALL write3( 'zfv', pFv )
      CALL write3( 'zfw', pFw )
      CALL write3( 'ww' , ww  )
      CALL write3( 'tsb_t', pts(:,:,:,1,Kbb) )
      CALL write3( 'tsb_s', pts(:,:,:,2,Kbb) )
      CALL write3( 'tsm_t', pts(:,:,:,1,Kmm) )
      CALL write3( 'tsm_s', pts(:,:,:,2,Kmm) )
      CALL write2( 'r3t_kbb', r3t(:,:,Kbb) )
      CALL write2( 'r3t_kmm', r3t(:,:,Kmm) )
      CALL write2( 'r3t_kaa', r3t(:,:,Kaa) )
   END SUBROUTINE vortex_r18_tracer_begin


   SUBROUTINE vortex_r18_tracer_rhs( cdname, Krhs, pts )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      INTEGER, INTENT(in) :: Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pts
      IF( .NOT. lopen ) RETURN
      CALL write3( TRIM(cdname)//'_t', pts(:,:,:,1,Krhs) )
      CALL write3( TRIM(cdname)//'_s', pts(:,:,:,2,Krhs) )
   END SUBROUTINE vortex_r18_tracer_rhs


   SUBROUTINE vortex_r18_tracer_state( cdname, Kaa, pts )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      INTEGER, INTENT(in) :: Kaa
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pts
      IF( .NOT. lopen ) RETURN
      CALL write3( TRIM(cdname)//'_t', pts(:,:,:,1,Kaa) )
      CALL write3( TRIM(cdname)//'_s', pts(:,:,:,2,Kaa) )
   END SUBROUTINE vortex_r18_tracer_state


   SUBROUTINE vortex_r18_tracer_finish( kstg )
      INTEGER, INTENT(in) :: kstg
      IF( .NOT. lopen ) RETURN
      IF( kstg /= nstage ) CALL ctl_stop( 'VORTEX R18 tracer writer stage drift' )
      CLOSE(nunit)
      WRITE(numout,*) 'VORTEX_R18_TRACER_TERM_DUMP stage ', kstg
      nunit = -1
      nstage = 0
      lopen = .FALSE.
   END SUBROUTINE vortex_r18_tracer_finish


   SUBROUTINE write3( cdname, pfield )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      clname = cdname
      WRITE(nunit) clname, 3, SIZE(pfield,1), SIZE(pfield,2), SIZE(pfield,3)
      WRITE(nunit) pfield
   END SUBROUTINE write3


   SUBROUTINE write2( cdname, pfield )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      clname = cdname
      WRITE(nunit) clname, 2, SIZE(pfield,1), SIZE(pfield,2), 1
      WRITE(nunit) pfield
   END SUBROUTINE write2

END MODULE vortex_r18_tracer_terms
