MODULE l4_r104_een_accum
   !! Rank-complete, self-describing, write-only EEN pre-scale operands.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r104_een_accum_dump

CONTAINS
   SUBROUTINE r104_een_accum_dump(acc_u_nw, acc_u_ne, acc_u_sw, acc_u_se, &
      & acc_v_sw, acc_v_se, acc_v_nw, acc_v_ne, scl_u_nw, scl_u_ne, &
      & scl_u_sw, scl_u_se, scl_v_sw, scl_v_se, scl_v_nw, scl_v_ne)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: acc_u_nw, acc_u_ne, acc_u_sw, acc_u_se
      REAL(wp), DIMENSION(:,:), INTENT(in) :: acc_v_sw, acc_v_se, acc_v_nw, acc_v_ne
      REAL(wp), DIMENSION(:,:), INTENT(in) :: scl_u_nw, scl_u_ne, scl_u_sw, scl_u_se
      REAL(wp), DIMENSION(:,:), INTENT(in) :: scl_v_sw, scl_v_se, scl_v_nw, scl_v_ne
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round104: EEN operand record requires fp64')
      WRITE(filename,'("oracle_r104_een_accum_rank",I4.4,"_kt",I8.8,".bin")') mpprank, nit000
      OPEN(NEWUNIT=unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round104: cannot open EEN operand record')
      magic = 'NEMO_L4_R104EA1'
      WRITE(unit) magic
      WRITE(unit) 1, nit000, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 16
      CALL write_2d(unit, 'acc_u_nw', acc_u_nw)
      CALL write_2d(unit, 'acc_u_ne', acc_u_ne)
      CALL write_2d(unit, 'acc_u_sw', acc_u_sw)
      CALL write_2d(unit, 'acc_u_se', acc_u_se)
      CALL write_2d(unit, 'acc_v_sw', acc_v_sw)
      CALL write_2d(unit, 'acc_v_se', acc_v_se)
      CALL write_2d(unit, 'acc_v_nw', acc_v_nw)
      CALL write_2d(unit, 'acc_v_ne', acc_v_ne)
      CALL write_2d(unit, 'scl_u_nw', scl_u_nw)
      CALL write_2d(unit, 'scl_u_ne', scl_u_ne)
      CALL write_2d(unit, 'scl_u_sw', scl_u_sw)
      CALL write_2d(unit, 'scl_u_se', scl_u_se)
      CALL write_2d(unit, 'scl_v_sw', scl_v_sw)
      CALL write_2d(unit, 'scl_v_se', scl_v_se)
      CALL write_2d(unit, 'scl_v_nw', scl_v_nw)
      CALL write_2d(unit, 'scl_v_ne', scl_v_ne)
      CLOSE(unit)
      WRITE(numout,*) 'ORCA2_R104_EEN_ACCUM_DUMP ', nit000, mpprank
   END SUBROUTINE r104_een_accum_dump

   SUBROUTINE write_2d(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, ntei-ntsi+1, ntej-ntsj+1, 1
      WRITE(unit) value(ntsi:ntei,ntsj:ntej)
   END SUBROUTINE write_2d
END MODULE l4_r104_een_accum
