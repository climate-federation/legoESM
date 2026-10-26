MODULE swg_r1_stage_entry_record
   !! Read-only SWG stage-entry writer.  One self-describing stream file per
   !! (step, stage); every field is INTENT(in) and no NEMO-consumed value is
   !! assigned.  Format: CHARACTER(16) magic, fifteen INTEGER header values,
   !! then repeated CHARACTER(16), rank/n1/n2/n3 INTEGERs, float64 payload.
   USE par_kind       , ONLY : wp
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: swg_open, swg_w2, swg_close
   INTEGER, SAVE :: nunit = -1
   LOGICAL, SAVE :: lopen = .FALSE.
CONTAINS
   SUBROUTINE swg_open( kstp, kstage, kbb, kmm, kaa, krhs, kpi, kpj, kpk, &
      &                 kis0, kjs0, kie0, kje0 )
      INTEGER, INTENT(in) :: kstp, kstage, kbb, kmm, kaa, krhs, kpi, kpj, kpk
      INTEGER, INTENT(in) :: kis0, kjs0, kie0, kje0
      INTEGER :: ios
      CHARACTER(LEN=16) :: clmagic
      CHARACTER(LEN=128) :: clfile
      IF( lopen ) CALL ctl_stop( 'SWG stage-entry writer opened twice' )
      IF( STORAGE_SIZE(1._wp) /= 64 ) CALL ctl_stop( 'SWG writer requires 64-bit wp' )
      WRITE(clfile,'("oracle_swgstage_kt",I8.8,"_s",I1,".bin")') kstp, kstage
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM', FORM='UNFORMATTED', &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open SWG stage-entry file' )
      clmagic = 'NEMO_L1_SWGSTG1'
      WRITE(nunit) clmagic, 1, kstp, kstage, kbb, kmm, kaa, krhs, kpi, kpj, &
         &         kpk, kis0, kjs0, kie0, kje0, STORAGE_SIZE(1._wp)
      lopen = .TRUE.
   END SUBROUTINE swg_open

   SUBROUTINE swg_w2( cdname, pfield )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      IF( .NOT. lopen ) RETURN
      WRITE(clname,'(A)') TRIM(cdname)
      WRITE(nunit) clname, 2, SIZE(pfield,1), SIZE(pfield,2), 1
      WRITE(nunit) pfield
   END SUBROUTINE swg_w2

   SUBROUTINE swg_close( kstp, kstage )
      INTEGER, INTENT(in) :: kstp, kstage
      IF( .NOT. lopen ) RETURN
      CLOSE(nunit)
      WRITE(numout,*) 'SWG_R1_STAGE_ENTRY_DUMP ', kstp, kstage
      nunit = -1
      lopen = .FALSE.
   END SUBROUTINE swg_close
END MODULE swg_r1_stage_entry_record
