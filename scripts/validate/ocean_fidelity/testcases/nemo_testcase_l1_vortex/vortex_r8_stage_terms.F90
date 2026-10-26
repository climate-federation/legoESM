MODULE vortex_r8_stage_terms
   !!======================================================================
   !! Read-only stage-2/3 momentum-term writer for VORTEX round 8.
   !!
   !! One self-describing stream is opened after each stage's WZV call and
   !! closed after the common barotropic correction.  The caller writes the
   !! accumulator only AFTER a compiled statement has completed; this module
   !! performs no arithmetic on a field NEMO consumes.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE par_oce        , ONLY : jpi, jpj, jpk
   USE oce            , ONLY : ww, ssh
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: vortex_r8_stage_begin
   PUBLIC :: vortex_r8_stage_rhs
   PUBLIC :: vortex_r8_stage_state
   PUBLIC :: vortex_r8_stage_finish

   INTEGER, SAVE :: nunit = -1
   INTEGER, SAVE :: nstage = 0
   LOGICAL, SAVE :: lopen = .FALSE.

CONTAINS

   SUBROUTINE vortex_r8_stage_begin( kstp, kstg, Kbb, Kmm, Kaa, Krhs, puu, pvv )
      INTEGER, INTENT(in) :: kstp, kstg, Kbb, Kmm, Kaa, Krhs
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      INTEGER :: ios, ngrps
      CHARACTER(LEN=16) :: clmagic
      CHARACTER(LEN=128) :: clfile

      IF( lopen ) CALL ctl_stop( 'VORTEX R8 stage writer opened twice' )
      IF( kstg /= 2 .AND. kstg /= 3 ) &
         CALL ctl_stop( 'VORTEX R8 stage writer accepts only stages 2 and 3' )
      IF( STORAGE_SIZE(1._wp) /= 64 ) &
         CALL ctl_stop( 'VORTEX R8 stage writer requires 64-bit wp' )

      WRITE(clfile,'("oracle_stage_terms_kt",I8.8,"_s",I1,".bin")') kstp, kstg
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM', &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open VORTEX R8 stage-term file' )

      nstage = kstg
      lopen = .TRUE.
      ngrps = MERGE( 18, 20, kstg == 2 )
      clmagic = 'NEMO_L1_STGTRM1'
      WRITE(nunit) clmagic, 1, kstp, kstg, Kbb, Kmm, Kaa, Krhs, &
         &         jpi, jpj, jpk, ngrps, 0, 0, 0, STORAGE_SIZE(1._wp)
      CALL write3( 'kmm_u', puu(:,:,:,Kmm) )
      CALL write3( 'kmm_v', pvv(:,:,:,Kmm) )
      CALL write2( 'ssh_kmm', ssh(:,:,Kmm) )
      CALL write3( 'ww', ww )
      CALL vortex_r8_stage_rhs( 'base', Krhs, puu, pvv )
   END SUBROUTINE vortex_r8_stage_begin


   SUBROUTINE vortex_r8_stage_rhs( cdname, Krhs, puu, pvv )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      INTEGER, INTENT(in) :: Krhs
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      IF( .NOT. lopen ) RETURN
      CALL write3( TRIM(cdname)//'_u', puu(:,:,:,Krhs) )
      CALL write3( TRIM(cdname)//'_v', pvv(:,:,:,Krhs) )
   END SUBROUTINE vortex_r8_stage_rhs


   SUBROUTINE vortex_r8_stage_state( cdname, Kaa, puu, pvv )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      INTEGER, INTENT(in) :: Kaa
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      IF( .NOT. lopen ) RETURN
      CALL write3( TRIM(cdname)//'_u', puu(:,:,:,Kaa) )
      CALL write3( TRIM(cdname)//'_v', pvv(:,:,:,Kaa) )
   END SUBROUTINE vortex_r8_stage_state


   SUBROUTINE vortex_r8_stage_finish( kstg, Kaa, puu, pvv )
      INTEGER, INTENT(in) :: kstg, Kaa
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      IF( .NOT. lopen ) RETURN
      IF( kstg /= nstage ) CALL ctl_stop( 'VORTEX R8 stage writer stage drift' )
      CALL vortex_r8_stage_state( 'out', Kaa, puu, pvv )
      CLOSE(nunit)
      WRITE(numout,*) 'VORTEX_R8_STAGE_TERM_DUMP stage ', kstg
      nunit = -1
      nstage = 0
      lopen = .FALSE.
   END SUBROUTINE vortex_r8_stage_finish


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

END MODULE vortex_r8_stage_terms
