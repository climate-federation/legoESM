MODULE l4_r92_rhs_frames
   !! Rank-complete, write-only rung-0 stage-1 momentum accumulator recorder.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r92_rhs_start, r92_rhs_put_pair, r92_rhs_finish

   INTEGER, SAVE :: r92_unit = -1
   INTEGER, SAVE :: r92_fields = 0

CONTAINS

   SUBROUTINE r92_rhs_start(kt, Kbb, Krhs)
      INTEGER, INTENT(in) :: kt, Kbb, Krhs
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(kt /= nit000) RETURN
      IF(r92_unit /= -1) CALL ctl_stop('round92: RHS record already open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round92: RHS record requires fp64')
      WRITE(filename,'("oracle_r92_rhs_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=r92_unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round92: cannot open RHS record')
      magic = 'NEMO_L4_R92RHS1'
      WRITE(r92_unit) magic
      WRITE(r92_unit) 1, kt, Kbb, Krhs, mpprank, jpi, jpj, jpk, &
         & nimpp, njmpp, ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 10
      r92_fields = 0
   END SUBROUTINE r92_rhs_start

   SUBROUTINE r92_put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r92_unit) field
      WRITE(r92_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(r92_unit) value
      r92_fields = r92_fields + 1
   END SUBROUTINE r92_put3

   SUBROUTINE r92_rhs_put_pair(kt, boundary, u_rhs, v_rhs)
      INTEGER, INTENT(in) :: kt
      CHARACTER(LEN=*), INTENT(in) :: boundary
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: u_rhs, v_rhs
      IF(kt /= nit000) RETURN
      IF(r92_unit == -1) CALL ctl_stop('round92: RHS record is not open')
      CALL r92_put3('after_' // TRIM(boundary) // '_u', u_rhs)
      CALL r92_put3('after_' // TRIM(boundary) // '_v', v_rhs)
   END SUBROUTINE r92_rhs_put_pair

   SUBROUTINE r92_rhs_finish(kt)
      INTEGER, INTENT(in) :: kt
      IF(kt /= nit000) RETURN
      IF(r92_unit == -1) CALL ctl_stop('round92: RHS record is not open')
      IF(r92_fields /= 10) CALL ctl_stop('round92: RHS field count is not ten')
      CLOSE(r92_unit)
      r92_unit = -1
   END SUBROUTINE r92_rhs_finish

END MODULE l4_r92_rhs_frames
