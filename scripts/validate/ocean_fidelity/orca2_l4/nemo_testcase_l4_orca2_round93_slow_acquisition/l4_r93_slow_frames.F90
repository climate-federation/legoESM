MODULE l4_r93_slow_frames
   !! Rank-complete, write-only rung-0 slow-forcing boundary recorder.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r93_slow_start, r93_slow_put2, r93_slow_put_pair, r93_slow_finish

   INTEGER, SAVE :: r93_unit = -1
   INTEGER, SAVE :: r93_fields = 0

CONTAINS
   SUBROUTINE r93_slow_start(kt, Kbb, Kaa, Krhs)
      INTEGER, INTENT(in) :: kt, Kbb, Kaa, Krhs
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(kt /= nit000) RETURN
      IF(r93_unit /= -1) CALL ctl_stop('round93: slow record already open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round93: slow record requires fp64')
      WRITE(filename,'("oracle_r93_slow_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=r93_unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round93: cannot open slow record')
      magic = 'NEMO_L4_R93SLOW1'
      WRITE(r93_unit) magic
      WRITE(r93_unit) 1, kt, Kbb, Kaa, Krhs, mpprank, jpi, jpj, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 14
      r93_fields = 0
   END SUBROUTINE r93_slow_start

   SUBROUTINE r93_slow_put2(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      IF(r93_unit == -1) CALL ctl_stop('round93: slow record is not open')
      field = name
      WRITE(r93_unit) field
      WRITE(r93_unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(r93_unit) value
      r93_fields = r93_fields + 1
   END SUBROUTINE r93_slow_put2

   SUBROUTINE r93_slow_put_pair(kt, boundary, u_value, v_value)
      INTEGER, INTENT(in) :: kt
      CHARACTER(LEN=*), INTENT(in) :: boundary
      REAL(wp), DIMENSION(:,:), INTENT(in) :: u_value, v_value
      IF(kt /= nit000) RETURN
      CALL r93_slow_put2(TRIM(boundary) // '_u', u_value)
      CALL r93_slow_put2(TRIM(boundary) // '_v', v_value)
   END SUBROUTINE r93_slow_put_pair

   SUBROUTINE r93_slow_finish(kt)
      INTEGER, INTENT(in) :: kt
      IF(kt /= nit000) RETURN
      IF(r93_fields /= 14) CALL ctl_stop('round93: slow field count is not fourteen')
      CLOSE(r93_unit)
      r93_unit = -1
   END SUBROUTINE r93_slow_finish
END MODULE l4_r93_slow_frames
