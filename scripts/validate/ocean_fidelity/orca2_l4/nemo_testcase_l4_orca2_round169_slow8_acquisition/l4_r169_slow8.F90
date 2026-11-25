MODULE l4_r169_slow8
   !! Rank-complete, self-describing WRITE-only kt=8 slow-forcing recorder.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r169_start, r169_put1, r169_put2, r169_put3, r169_finish

   INTEGER, PARAMETER :: r169_kt = 8, r169_expected = 23
   INTEGER, SAVE :: r169_unit = -1, r169_fields = 0

CONTAINS
   SUBROUTINE r169_start(kt, Kbb, Kaa, Krhs)
      INTEGER, INTENT(in) :: kt, Kbb, Kaa, Krhs
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=256) :: filename
      IF(kt /= r169_kt) RETURN
      IF(r169_unit /= -1) CALL ctl_stop('round169: slow8 record already open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round169: slow8 requires fp64')
      WRITE(filename,'("oracle_r169_slow8_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=r169_unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round169: cannot open slow8 record')
      magic = 'NEMO_L4_R169SLW8'
      WRITE(r169_unit) magic
      WRITE(r169_unit) 1, kt, Kbb, Kaa, Krhs, mpprank, jpi, jpj, jpk, &
         & nimpp, njmpp, ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), &
         & r169_expected
      r169_fields = 0
   END SUBROUTINE r169_start

   SUBROUTINE r169_header(name, ndim, n1, n2, n3)
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER, INTENT(in) :: ndim, n1, n2, n3
      CHARACTER(LEN=16) :: field
      IF(r169_unit == -1) CALL ctl_stop('round169: slow8 record is not open')
      field = name
      WRITE(r169_unit) field
      WRITE(r169_unit) ndim, n1, n2, n3
      r169_fields = r169_fields + 1
   END SUBROUTINE r169_header

   SUBROUTINE r169_put1(kt, name, value)
      INTEGER, INTENT(in) :: kt
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value
      IF(kt /= r169_kt) RETURN
      CALL r169_header(name, 1, 1, 1, 1)
      WRITE(r169_unit) value
   END SUBROUTINE r169_put1

   SUBROUTINE r169_put2(kt, name, value)
      INTEGER, INTENT(in) :: kt
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      IF(kt /= r169_kt) RETURN
      CALL r169_header(name, 2, SIZE(value,1), SIZE(value,2), 1)
      WRITE(r169_unit) value
   END SUBROUTINE r169_put2

   SUBROUTINE r169_put3(kt, name, value)
      INTEGER, INTENT(in) :: kt
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      IF(kt /= r169_kt) RETURN
      CALL r169_header(name, 3, SIZE(value,1), SIZE(value,2), SIZE(value,3))
      WRITE(r169_unit) value
   END SUBROUTINE r169_put3

   SUBROUTINE r169_finish(kt)
      INTEGER, INTENT(in) :: kt
      IF(kt /= r169_kt) RETURN
      IF(r169_fields /= r169_expected) &
         & CALL ctl_stop('round169: slow8 field count is not twenty-three')
      CLOSE(r169_unit)
      r169_unit = -1
   END SUBROUTINE r169_finish
END MODULE l4_r169_slow8
