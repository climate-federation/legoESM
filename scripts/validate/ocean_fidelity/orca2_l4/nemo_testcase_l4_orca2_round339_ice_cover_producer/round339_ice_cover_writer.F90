MODULE round339_ice_cover_writer
   !! Passive full-array fr_i record immediately after the full-SI3 ice step.
   USE par_kind, ONLY : wp
   USE dom_oce, ONLY : nimpp, njmpp, narea, ntsi, ntei, ntsj, ntej
   USE in_out_manager, ONLY : numout
   USE lib_mpp, ONLY : ctl_stop
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r339_write

CONTAINS

   FUNCTION checksum2(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER(int64) :: checksum, word
      INTEGER :: ji, jj
      checksum = 0_int64
      DO jj = 1, SIZE(value,2) ; DO ji = 1, SIZE(value,1)
         word = TRANSFER(value(ji,jj),word)
         checksum = IEOR(checksum,word)
      END DO ; END DO
   END FUNCTION checksum2

   SUBROUTINE r339_write(kt, pfr_i, kn_fsbc)
      INTEGER, INTENT(in) :: kt, kn_fsbc
      REAL(wp), INTENT(in) :: pfr_i(:,:)
      INTEGER :: iunit, ios
      CHARACTER(LEN=96) :: clfile
      CHARACTER(LEN=16) :: clmagic
      IF( MOD(kt-1,kn_fsbc) /= 0 )   RETURN
      IF( kt < 1 .OR. kt > 2920 )   RETURN
      IF( kn_fsbc /= 2 )   CALL ctl_stop('round339 ice-cover writer requires nn_fsbc=2')
      IF( STORAGE_SIZE(1._wp) /= 64 )   CALL ctl_stop('round339 ice-cover writer requires fp64 wp')
      WRITE(clfile,'("oracle_r339_fr_i_kt",I8.8,"_r",I4.4,".bin")') kt, narea-1
      OPEN( NEWUNIT=iunit, FILE=TRIM(clfile), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop('round339 ice-cover writer cannot open output')
      clmagic = 'NEMO_R339_ICE1'
      WRITE(iunit) clmagic
      WRITE(iunit) 1, kt, narea-1, nimpp, njmpp, SIZE(pfr_i,1), SIZE(pfr_i,2), &
         & ntsi, ntei, ntsj, ntej, STORAGE_SIZE(1._wp), kn_fsbc
      WRITE(iunit) checksum2(pfr_i)
      WRITE(iunit) pfr_i
      CLOSE(iunit)
      WRITE(numout,*) 'ORCA2_R339_FR_I ', kt, narea-1, TRIM(clfile)
   END SUBROUTINE r339_write

END MODULE round339_ice_cover_writer

