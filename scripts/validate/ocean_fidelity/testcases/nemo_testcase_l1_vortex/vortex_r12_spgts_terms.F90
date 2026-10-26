MODULE vortex_r12_spgts_terms
   !!======================================================================
   !! Read-only per-substep writer for NEMO's split-explicit barotropic
   !! solve (dyn_spg_ts), VORTEX round 12 / campaign round 196.
   !!
   !! One self-describing stream per baroclinic step.  The caller opens it
   !! before the sub-time-step loop, names the substep it is inside, and
   !! writes an array only AFTER a compiled statement has completed.  This
   !! module performs no arithmetic on any field NEMO consumes and holds no
   !! reference to one: every array arrives as an INTENT(in) argument.
   !!
   !! RECORD FORMAT (operator note BD: self-describing, never size-predicted)
   !!   CHARACTER(16) magic = 'NEMO_L1_SPGTS1'
   !!   sixteen default INTEGERs, the 64-bit word size LAST
   !!   then (CHARACTER(16) name, INTEGER rank, n1, n2, n3) followed by the
   !!   payload record, repeated to end of file.
   !! Group names carry their frame: 'i000_<name>' is the loop-entry frame,
   !! 'jNNN_<name>' is substep NNN, 'o000_<name>' is the loop-exit frame.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: spgts_r12_open
   PUBLIC :: spgts_r12_substep
   PUBLIC :: spgts_r12_w2
   PUBLIC :: spgts_r12_w1
   PUBLIC :: spgts_r12_close

   INTEGER, SAVE :: nunit = -1
   INTEGER, SAVE :: nsub  = 0
   LOGICAL, SAVE :: lopen = .FALSE.

CONTAINS

   SUBROUTINE spgts_r12_open( kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk,   &
      &                       kcycle, kis0, kjs0, kie0, kje0 )
      INTEGER, INTENT(in) :: kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk
      INTEGER, INTENT(in) :: kcycle, kis0, kjs0, kie0, kje0
      INTEGER            :: ios
      CHARACTER(LEN=16)  :: clmagic
      CHARACTER(LEN=128) :: clfile
      !
      IF( lopen )   CALL ctl_stop( 'VORTEX R12 spgts writer opened twice' )
      IF( STORAGE_SIZE(1._wp) /= 64 )                                      &
         &   CALL ctl_stop( 'VORTEX R12 spgts writer requires 64-bit wp' )
      WRITE(clfile,'("oracle_spgts_kt",I8.8,".bin")') kstp
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM',             &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop( 'cannot open VORTEX R12 spgts file' )
      clmagic = 'NEMO_L1_SPGTS1'
      WRITE(nunit) clmagic, 1, kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk,   &
         &         kcycle, kis0, kjs0, kie0, kje0, STORAGE_SIZE(1._wp)
      nsub  = 0
      lopen = .TRUE.
   END SUBROUTINE spgts_r12_open


   SUBROUTINE spgts_r12_substep( kn )
      !! Name the frame the following writes belong to.  kn = 0 is the
      !! loop-entry frame, kn = -1 the loop-exit frame.
      INTEGER, INTENT(in) :: kn
      IF( .NOT. lopen )   RETURN
      nsub = kn
   END SUBROUTINE spgts_r12_substep


   SUBROUTINE spgts_r12_w2( cdname, pfield )
      CHARACTER(LEN=*)        , INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      IF( .NOT. lopen )   RETURN
      clname = frame_name( cdname )
      WRITE(nunit) clname, 2, SIZE(pfield,1), SIZE(pfield,2), 1
      WRITE(nunit) pfield
   END SUBROUTINE spgts_r12_w2


   SUBROUTINE spgts_r12_w1( cdname, pfield )
      CHARACTER(LEN=*)      , INTENT(in) :: cdname
      REAL(wp), DIMENSION(:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      IF( .NOT. lopen )   RETURN
      clname = frame_name( cdname )
      WRITE(nunit) clname, 1, SIZE(pfield,1), 1, 1
      WRITE(nunit) pfield
   END SUBROUTINE spgts_r12_w1


   SUBROUTINE spgts_r12_close( kstp, kcycle )
      INTEGER, INTENT(in) :: kstp, kcycle
      IF( .NOT. lopen )   RETURN
      CLOSE(nunit)
      WRITE(numout,*) 'VORTEX_R12_SPGTS_DUMP ', kstp, kcycle
      nunit = -1
      nsub  = 0
      lopen = .FALSE.
   END SUBROUTINE spgts_r12_close


   CHARACTER(LEN=16) FUNCTION frame_name( cdname )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      IF( nsub == 0 ) THEN
         WRITE(frame_name,'("i000_",A)') TRIM(cdname)
      ELSE IF( nsub < 0 ) THEN
         WRITE(frame_name,'("o000_",A)') TRIM(cdname)
      ELSE
         WRITE(frame_name,'("j",I3.3,"_",A)') nsub, TRIM(cdname)
      ENDIF
   END FUNCTION frame_name

END MODULE vortex_r12_spgts_terms
