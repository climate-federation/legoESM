MODULE tsunami_r1_step_record
   !!======================================================================
   !! Read-only per-step writer for TSUNAMI's stp_MLF (lane round 1).
   !!
   !! One self-describing stream per step, oracle_tsustep_kt########.bin.
   !! Every array arrives INTENT(in); the module does no arithmetic on any
   !! field NEMO consumes.
   !!
   !! RECORD FORMAT (self-describing, never size-predicted)
   !!   CHARACTER(16) magic = 'NEMO_L1_TSUSTP1'
   !!   fourteen default INTEGERs: version, step, Nbb, Nnn, Naa, Nrhs,
   !!   jpi, jpj, jpk, Nis0, Njs0, Nie0, Nje0, then the word size LAST
   !!   then (CHARACTER(16) name, INTEGER rank, n1, n2, n3) + payload,
   !!   repeated to end of file.  Names carry their frame:
   !!   'e_' step entry, 'r_' after dom_qco_r3c, 'a_' after dyn_spg.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: tsu_open, tsu_frame, tsu_w2, tsu_close

   INTEGER         , SAVE :: nunit = -1
   LOGICAL         , SAVE :: lopen = .FALSE.
   CHARACTER(LEN=1), SAVE :: cframe = 'e'

CONTAINS

   SUBROUTINE tsu_open( kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk,   &
      &                 kis0, kjs0, kie0, kje0 )
      INTEGER, INTENT(in) :: kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk
      INTEGER, INTENT(in) :: kis0, kjs0, kie0, kje0
      INTEGER            :: ios
      CHARACTER(LEN=16)  :: clmagic
      CHARACTER(LEN=128) :: clfile
      !
      IF( lopen )   CALL ctl_stop( 'TSUNAMI R1 step writer opened twice' )
      IF( STORAGE_SIZE(1._wp) /= 64 )                                   &
         &   CALL ctl_stop( 'TSUNAMI R1 step writer requires 64-bit wp' )
      WRITE(clfile,'("oracle_tsustep_kt",I8.8,".bin")') kstp
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM',          &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop( 'cannot open TSUNAMI R1 step file' )
      clmagic = 'NEMO_L1_TSUSTP1'
      WRITE(nunit) clmagic, 1, kstp, kbb, kmm, kaa, krhs, kpi, kpj, kpk, &
         &         kis0, kjs0, kie0, kje0, STORAGE_SIZE(1._wp)
      cframe = 'e'
      lopen  = .TRUE.
   END SUBROUTINE tsu_open


   SUBROUTINE tsu_frame( cdframe )
      CHARACTER(LEN=1), INTENT(in) :: cdframe
      cframe = cdframe
   END SUBROUTINE tsu_frame


   SUBROUTINE tsu_w2( cdname, pfield )
      CHARACTER(LEN=*)        , INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      IF( .NOT. lopen )   RETURN
      WRITE(clname,'(A,"_",A)') cframe, TRIM(cdname)
      WRITE(nunit) clname, 2, SIZE(pfield,1), SIZE(pfield,2), 1
      WRITE(nunit) pfield
   END SUBROUTINE tsu_w2


   SUBROUTINE tsu_close( kstp )
      INTEGER, INTENT(in) :: kstp
      IF( .NOT. lopen )   RETURN
      CLOSE(nunit)
      WRITE(numout,*) 'TSUNAMI_R1_STEP_DUMP ', kstp
      nunit = -1
      lopen = .FALSE.
   END SUBROUTINE tsu_close

END MODULE tsunami_r1_step_record
