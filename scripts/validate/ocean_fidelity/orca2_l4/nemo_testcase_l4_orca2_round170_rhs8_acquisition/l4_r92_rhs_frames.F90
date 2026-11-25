MODULE l4_r92_rhs_frames
   !! Rank-complete, write-only rung-0 kt=8 momentum accumulator recorder.
   !! The module name and public calls deliberately match the admitted
   !! round-92 instrumentation already present in stp2d.F90; only the selected
   !! step, magic and output name change.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r92_rhs_start, r92_rhs_put_pair, r92_rhs_finish

   INTEGER, PARAMETER :: target_kt = 8
   INTEGER, SAVE :: rhs_unit = -1
   INTEGER, SAVE :: rhs_fields = 0

CONTAINS

   SUBROUTINE r92_rhs_start(kt, Kbb, Krhs)
      INTEGER, INTENT(in) :: kt, Kbb, Krhs
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(kt /= target_kt) RETURN
      IF(rhs_unit /= -1) CALL ctl_stop('round170: RHS8 record already open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round170: RHS8 record requires fp64')
      WRITE(filename,'("oracle_r170_rhs8_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=rhs_unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round170: cannot open RHS8 record')
      magic = 'NEMO_L4_R170RH8'
      WRITE(rhs_unit) magic
      WRITE(rhs_unit) 1, kt, Kbb, Krhs, mpprank, jpi, jpj, jpk, &
         & nimpp, njmpp, ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 10
      rhs_fields = 0
   END SUBROUTINE r92_rhs_start

   SUBROUTINE rhs_put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(rhs_unit) field
      WRITE(rhs_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(rhs_unit) value
      rhs_fields = rhs_fields + 1
   END SUBROUTINE rhs_put3

   SUBROUTINE r92_rhs_put_pair(kt, boundary, u_rhs, v_rhs)
      INTEGER, INTENT(in) :: kt
      CHARACTER(LEN=*), INTENT(in) :: boundary
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: u_rhs, v_rhs
      IF(kt /= target_kt) RETURN
      IF(rhs_unit == -1) CALL ctl_stop('round170: RHS8 record is not open')
      CALL rhs_put3('after_' // TRIM(boundary) // '_u', u_rhs)
      CALL rhs_put3('after_' // TRIM(boundary) // '_v', v_rhs)
   END SUBROUTINE r92_rhs_put_pair

   SUBROUTINE r92_rhs_finish(kt)
      INTEGER, INTENT(in) :: kt
      IF(kt /= target_kt) RETURN
      IF(rhs_unit == -1) CALL ctl_stop('round170: RHS8 record is not open')
      IF(rhs_fields /= 10) CALL ctl_stop('round170: RHS8 field count is not ten')
      CLOSE(rhs_unit)
      rhs_unit = -1
   END SUBROUTINE r92_rhs_finish

END MODULE l4_r92_rhs_frames
