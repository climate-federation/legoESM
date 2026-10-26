MODULE l4_r69_surface
   !! Round-69 WRITE-only ORCA2 ocean surface-operand recorder.
   !! No value is ever read back into NEMO state.
   USE dom_oce, ONLY : wp, jpi, jpj, jpts, nn_hls, ssmask
   USE sbc_oce, ONLY : utau, vtau, taum, qsr, qns, emp, sfx, rnf, fr_i
   USE sbcrnf, ONLY : rnf_tsc
   USE lib_mpp, ONLY : ctl_stop, mpprank
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: l4_r69_dump

CONTAINS

   SUBROUTINE put2(unit, name, value, mask)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      REAL(wp), DIMENSION(:,:), INTENT(in) :: mask
      REAL(wp), DIMENSION(SIZE(value,1),SIZE(value,2)) :: canonical
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, SIZE(value, 1), SIZE(value, 2), 1
      canonical(:,:) = 0._wp
      WHERE(mask /= 0._wp) canonical = value
      WRITE(unit) canonical
   END SUBROUTINE put2

   SUBROUTINE put3(unit, name, value, mask)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      REAL(wp), DIMENSION(:,:), INTENT(in) :: mask
      REAL(wp), DIMENSION(SIZE(value,1),SIZE(value,2),SIZE(value,3)) :: canonical
      INTEGER :: jk
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 3, SIZE(value, 1), SIZE(value, 2), SIZE(value, 3)
      canonical(:,:,:) = 0._wp
      DO jk = 1, SIZE(value, 3)
         WHERE(mask /= 0._wp) canonical(:,:,jk) = value(:,:,jk)
      ENDDO
      WRITE(unit) canonical
   END SUBROUTINE put3

   SUBROUTINE l4_r69_dump(kt, klevel)
      INTEGER, INTENT(in) :: kt, klevel
      INTEGER :: unit, ios, ib, ie, jb, je
      INTEGER, PARAMETER :: nfields = 10
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename

      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round69: surface record requires fp64')
      IF(jpts /= 2) CALL ctl_stop('round69: surface record requires T/S')
      ib = 1 + nn_hls
      ie = jpi - nn_hls
      jb = 1 + nn_hls
      je = jpj - nn_hls
      WRITE(filename,'("oracle_r69_surface_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round69: cannot open surface record')
      magic = 'NEMO_L4_R69SFC1'
      WRITE(unit) magic
      WRITE(unit) 1, kt, klevel, mpprank, nfields, ie-ib+1, je-jb+1, STORAGE_SIZE(1._wp)
      CALL put2(unit, 'utau', utau(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'vtau', vtau(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'taum', taum(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'qsr', qsr(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'qns', qns(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'emp', emp(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'sfx', sfx(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'rnf', rnf(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put2(unit, 'fr_i', fr_i(ib:ie,jb:je), ssmask(ib:ie,jb:je))
      CALL put3(unit, 'rnf_tsc', rnf_tsc(ib:ie,jb:je,1:jpts), ssmask(ib:ie,jb:je))
      CLOSE(unit)
   END SUBROUTINE l4_r69_dump

END MODULE l4_r69_surface
