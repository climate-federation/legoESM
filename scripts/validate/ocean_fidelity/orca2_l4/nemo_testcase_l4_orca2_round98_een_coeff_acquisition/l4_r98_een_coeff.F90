MODULE l4_r98_een_coeff
   !! Rank-complete, self-describing, write-only frozen EEN coefficients.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r98_een_coeff_dump

CONTAINS
   SUBROUTINE r98_een_coeff_dump(kt, Kmm, Kscheme, ffu_nw, ffu_ne, ffu_sw, ffu_se, &
      & ffv_sw, ffv_se, ffv_nw, ffv_ne)
      INTEGER, INTENT(in) :: kt, Kmm, Kscheme
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ffu_nw, ffu_ne, ffu_sw, ffu_se
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ffv_sw, ffv_se, ffv_nw, ffv_ne
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(kt /= nit000) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round98: EEN coefficient record requires fp64')
      WRITE(filename,'("oracle_r98_een_coeff_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round98: cannot open EEN coefficient record')
      magic = 'NEMO_L4_R98EEN1'
      WRITE(unit) magic
      WRITE(unit) 1, kt, Kmm, Kscheme, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 8
      CALL write_2d(unit, 'ffu_nw', ffu_nw)
      CALL write_2d(unit, 'ffu_ne', ffu_ne)
      CALL write_2d(unit, 'ffu_sw', ffu_sw)
      CALL write_2d(unit, 'ffu_se', ffu_se)
      CALL write_2d(unit, 'ffv_sw', ffv_sw)
      CALL write_2d(unit, 'ffv_se', ffv_se)
      CALL write_2d(unit, 'ffv_nw', ffv_nw)
      CALL write_2d(unit, 'ffv_ne', ffv_ne)
      CLOSE(unit)
      WRITE(numout,*) 'ORCA2_R98_EEN_COEFF_DUMP ', kt, Kmm, Kscheme, mpprank
   END SUBROUTINE r98_een_coeff_dump

   SUBROUTINE write_2d(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(unit) value
   END SUBROUTINE write_2d
END MODULE l4_r98_een_coeff
