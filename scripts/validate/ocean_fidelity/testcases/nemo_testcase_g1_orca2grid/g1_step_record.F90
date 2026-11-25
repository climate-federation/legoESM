MODULE g1_step_record
   !!======================================================================
   !! Read-only per-step writer for the ORCA2G1 RK3 step (grid ladder G1).
   !! The TSUNAMI lane's writer (tsunami_r1_step_record.F90) with new names.
   !!
   !! One self-describing stream per step, oracle_g1step_kt########.bin.
   !! Every array arrives INTENT(in); the module does no arithmetic on any
   !! field NEMO consumes.  Single-rank runs only (the arrays are the local
   !! domain INCLUDING halos, so the fold and seam halo rows are recorded).
   !!
   !! RECORD FORMAT (self-describing, never size-predicted)
   !!   CHARACTER(16) magic = 'NEMO_G1_STEP_01'
   !!   fourteen default INTEGERs: version, step, Nbb, Nnn, Naa, Nrhs,
   !!   jpi, jpj, jpk, Nis0, Njs0, Nie0, Nje0, then the word size LAST
   !!   then (CHARACTER(16) name, INTEGER rank, n1, n2, n3) + payload,
   !!   repeated to end of file.  Names carry their frame:
   !!   'e_' step entry, 'b_' after stp_2D, '1_'..'3_' after each RK3
   !!   stage, 'f_' after the final swap and ssh extrapolation.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: g1r_open, g1r_frame, g1r_w2, g1r_close

   INTEGER         , SAVE :: nunit = -1
   LOGICAL         , SAVE :: lopen = .FALSE.
   CHARACTER(LEN=1), SAVE :: cframe = 'e'

CONTAINS

   SUBROUTINE g1r_open( kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk,   &
      &                 kis0, kjs0, kie0, kje0 )
      INTEGER, INTENT(in) :: kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk
      INTEGER, INTENT(in) :: kis0, kjs0, kie0, kje0
      INTEGER            :: ios
      CHARACTER(LEN=16)  :: clmagic
      CHARACTER(LEN=128) :: clfile
      !
      IF( lopen )   CALL ctl_stop( 'G1 step writer opened twice' )
      IF( STORAGE_SIZE(1._wp) /= 64 )                                   &
         &   CALL ctl_stop( 'G1 step writer requires 64-bit wp' )
      WRITE(clfile,'("oracle_g1step_kt",I8.8,".bin")') kstp
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM',          &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop( 'cannot open G1 step file' )
      clmagic = 'NEMO_G1_STEP_01'
      WRITE(nunit) clmagic, 1, kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk, &
         &         kis0, kjs0, kie0, kje0, STORAGE_SIZE(1._wp)
      cframe = 'e'
      lopen  = .TRUE.
   END SUBROUTINE g1r_open


   SUBROUTINE g1r_frame( cdframe )
      CHARACTER(LEN=1), INTENT(in) :: cdframe
      cframe = cdframe
   END SUBROUTINE g1r_frame


   SUBROUTINE g1r_w2( cdname, pfield )
      CHARACTER(LEN=*)        , INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      IF( .NOT. lopen )   RETURN
      WRITE(clname,'(A,"_",A)') cframe, TRIM(cdname)
      WRITE(nunit) clname, 2, SIZE(pfield,1), SIZE(pfield,2), 1
      WRITE(nunit) pfield
   END SUBROUTINE g1r_w2


   SUBROUTINE g1r_close( kstp )
      INTEGER, INTENT(in) :: kstp
      IF( .NOT. lopen )   RETURN
      CLOSE(nunit)
      WRITE(numout,*) 'G1_STEP_DUMP ', kstp
      nunit = -1
      lopen = .FALSE.
   END SUBROUTINE g1r_close

END MODULE g1_step_record
