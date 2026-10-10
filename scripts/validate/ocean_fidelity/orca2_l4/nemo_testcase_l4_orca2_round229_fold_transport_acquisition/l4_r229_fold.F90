MODULE l4_r229_fold
   !! Round-229 write-only, rank-complete tracer-consumer operand record.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r229_dump_fold_operands

CONTAINS

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

   SUBROUTINE r229_dump_fold_operands(kt, stage, zfv, temp, salt, e3t_live, tmask_live)
      INTEGER, INTENT(in) :: kt, stage
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: zfv, temp, salt, e3t_live, tmask_live
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(stage /= 1) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round229: record requires fp64')
      WRITE(filename,'("oracle_r229_fold_rank",I4.4,"_kt",I8.8,"_s1.bin")') mpprank, kt
      OPEN(NEWUNIT=unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round229: cannot open fold operand record')
      magic = 'NEMO_L4_R229FLD1'
      WRITE(unit) magic
      WRITE(unit) 1, kt, stage, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 5
      CALL put3(unit, 'zFv_after_trp', zfv)
      CALL put3(unit, 'T_Kmm', temp)
      CALL put3(unit, 'S_Kmm', salt)
      CALL put3(unit, 'e3t_Kmm', e3t_live)
      CALL put3(unit, 'tmask', tmask_live)
      CLOSE(unit)
   END SUBROUTINE r229_dump_fold_operands

END MODULE l4_r229_fold
