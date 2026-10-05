MODULE l4_r84_frames
   !! Round-84 write-only ORCA2 hierarchy rung-0 entry/stage recorder.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk, jpts, jp_tem, jp_sal
   USE oce,            ONLY : ts, uu, vv, ssh
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r84_dump_frame

CONTAINS

   SUBROUTINE put2(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE put3(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE r84_dump_frame(kt, stage, level)
      INTEGER, INTENT(in) :: kt, stage, level
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(kt < nit000 .OR. kt > nit000 + 9) RETURN
      IF(stage < 0 .OR. stage > 3) CALL ctl_stop('round84: bad frame stage')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round84: frame requires fp64')
      IF(jpts /= 2) CALL ctl_stop('round84: frame requires T/S')
      WRITE(filename,'("oracle_r84_frame_rank",I4.4,"_kt",I8.8,"_s",I1,".bin")') &
         & mpprank, kt, stage
      OPEN(NEWUNIT=unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round84: cannot open frame')
      magic = 'NEMO_L4_R84FRM1'
      WRITE(unit) magic
      WRITE(unit) 1, kt, stage, level, mpprank, jpi, jpj, jpk, jpts, &
         & nimpp, njmpp, ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 5
      CALL put3(unit, 'T', ts(:,:,:,jp_tem,level))
      CALL put3(unit, 'S', ts(:,:,:,jp_sal,level))
      CALL put3(unit, 'u', uu(:,:,:,level))
      CALL put3(unit, 'v', vv(:,:,:,level))
      CALL put2(unit, 'ssh', ssh(:,:,level))
      CLOSE(unit)
   END SUBROUTINE r84_dump_frame

END MODULE l4_r84_frames
