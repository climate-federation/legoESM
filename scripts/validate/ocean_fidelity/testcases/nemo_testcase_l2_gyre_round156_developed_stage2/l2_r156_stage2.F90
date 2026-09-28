MODULE l2_r156_stage2
   !! Round-156 WRITE-only developed-state RK3 STAGE-2 momentum record.
   !!
   !! Every stage-2 momentum writer in this MY_SRC is gated on kstp == nit000,
   !! so NEMO's stage-2 RHS families, its N+1/3 velocity and its N+1/3 free-
   !! surface ratios do not exist at the developed day-180 step.  Round 155
   !! measured that the stage-2 velocity uu(Kmm) owns 95.2% by rms of the
   !! developed stage-3 transport difference, so those operands are exactly
   !! what the next walk needs.
   !!
   !! Self-describing: a 16-character magic, sixteen header integers, then one
   !! (name, rank, n1, n2, n3, payload) group per pair until EOF.  The dims
   !! are the ARGUMENT's own extents, not jpi/jpj/jpk: the stage barotropic
   !! correction allocates its zub/zvb on the interior tile bounds, so a
   !! group that claimed jpi,jpj would mislabel it.  Nothing here is read
   !! back by NEMO; every argument is INTENT(in).
   USE par_kind, ONLY : wp
   USE dom_oce,  ONLY : jpi, jpj, jpk, ntsi, ntei, ntsj, ntej
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r156_stage2_open, r156_stage2_pair3, r156_stage2_pair2
   PUBLIC :: r156_stage2_scal, r156_stage2_close
   !  NEWUNIT hands back a NEGATIVE unit number, so "is the file open" is a
   !  LOGICAL here and never a sign test on the unit -- a sign test silently
   !  skipped every group and left an 80-byte header behind.
   INTEGER, SAVE :: nunit = 0
   LOGICAL, SAVE :: lopen = .FALSE.
   LOGICAL, SAVE :: ldone = .FALSE.
CONTAINS

   SUBROUTINE r156_stage2_open( kt, kstg, Kbb, Kmm, Krhs, Kaa )
      INTEGER, INTENT(in) :: kt, kstg, Kbb, Kmm, Krhs, Kaa
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      IF( ldone ) RETURN
      IF( STORAGE_SIZE(1._wp) /= 64 ) ERROR STOP 'R156_STAGE2_REQUIRES_FP64'
      magic = 'NEMO_L2_R156ST2'
      OPEN( NEWUNIT=nunit, FILE='oracle_developed_stage2_kt00001081.bin', &
         & ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE', &
         & ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) ERROR STOP 'R156_STAGE2_OPEN_FAILED'
      WRITE(nunit) magic
      WRITE(nunit) 1, kt, kstg, Kbb, Kmm, Krhs, Kaa, jpi, jpj, jpk, &
         & STORAGE_SIZE(1._wp), 19, ntsi, ntei, ntsj, ntej
      lopen = .TRUE.
   END SUBROUTINE r156_stage2_open

   SUBROUTINE r156_stage2_pair3( cdname, pu, pv )
      CHARACTER(LEN=16), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pu, pv
      IF( .NOT.lopen ) RETURN
      IF( ANY( SHAPE(pu) /= SHAPE(pv) ) ) ERROR STOP 'R156_STAGE2_PAIR3_SHAPE'
      WRITE(nunit) cdname
      WRITE(nunit) 3, SIZE(pu,1), SIZE(pu,2), SIZE(pu,3)
      WRITE(nunit) pu
      WRITE(nunit) pv
   END SUBROUTINE r156_stage2_pair3

   SUBROUTINE r156_stage2_pair2( cdname, pu, pv )
      CHARACTER(LEN=16), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pu, pv
      IF( .NOT.lopen ) RETURN
      IF( ANY( SHAPE(pu) /= SHAPE(pv) ) ) ERROR STOP 'R156_STAGE2_PAIR2_SHAPE'
      WRITE(nunit) cdname
      WRITE(nunit) 2, SIZE(pu,1), SIZE(pu,2), 1
      WRITE(nunit) pu
      WRITE(nunit) pv
   END SUBROUTINE r156_stage2_pair2

   SUBROUTINE r156_stage2_scal( cdname, pa, pb )
      CHARACTER(LEN=16), INTENT(in) :: cdname
      REAL(wp), INTENT(in) :: pa, pb
      IF( .NOT.lopen ) RETURN
      WRITE(nunit) cdname
      WRITE(nunit) 0, 1, 1, 1
      WRITE(nunit) pa
      WRITE(nunit) pb
   END SUBROUTINE r156_stage2_scal

   SUBROUTINE r156_stage2_close()
      IF( .NOT.lopen ) RETURN
      CLOSE(nunit)
      nunit = 0
      lopen = .FALSE.
      ldone = .TRUE.
   END SUBROUTINE r156_stage2_close

END MODULE l2_r156_stage2
