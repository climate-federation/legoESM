MODULE g1_spgts_record
   !!======================================================================
   !! Read-only per-substep writer for NEMO's split-explicit barotropic
   !! solve (dyn_spg_ts), grid ladder G1 round 4 (the VORTEX round-196 writer).
   !!
   !! One self-describing stream per baroclinic step.  The caller opens it
   !! before the sub-time-step loop, names the substep it is inside, and
   !! writes an array only AFTER a compiled statement has completed.  This
   !! module performs no arithmetic on any field NEMO consumes and holds no
   !! reference to one: every array arrives as an INTENT(in) argument.
   !!
   !! RECORD FORMAT (operator note BD: self-describing, never size-predicted)
   !!   CHARACTER(16) magic = 'NEMO_G1_SPGTS1'
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
      IF( lopen )   CALL ctl_stop( 'G1 spgts writer opened twice' )
      IF( STORAGE_SIZE(1._wp) /= 64 )                                      &
         &   CALL ctl_stop( 'G1 spgts writer requires 64-bit wp' )
      !                          ! G1: only the steps the round-4 walk needs
      IF( kstp /= 1 .AND. kstp /= 2 .AND. kstp /= 17 .AND.                  &
         &   kstp /= 31 .AND. kstp /= 41 )   RETURN
      WRITE(clfile,'("oracle_g1spgts_kt",I8.8,".bin")') kstp
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM',             &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop( 'cannot open G1 spgts file' )
      clmagic = 'NEMO_G1_SPGTS1'
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
      WRITE(numout,*) 'G1_SPGTS_DUMP ', kstp, kcycle
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

END MODULE g1_spgts_record
