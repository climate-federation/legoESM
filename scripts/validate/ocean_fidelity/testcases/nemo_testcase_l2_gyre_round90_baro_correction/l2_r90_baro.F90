MODULE l2_r90_baro
   !! Round-90 WRITE-only correction-site recorder.  Captured values are never
   !! read back by NEMO; the only active call is GYRE kt=2, RK3 stage 1.
   USE dom_oce
   USE oce,            ONLY : uu_b, vv_b
   USE in_out_manager, ONLY : lwp, numout, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r90_baro_begin, r90_baro_finish

   INTEGER, SAVE :: r90_unit = -1
   LOGICAL, SAVE :: r90_active = .FALSE.
   CHARACTER(LEN=16), PARAMETER :: r90_magic = 'NEMO_L2_R90BARO1'

#  include "domzgr_substitute.h90"

CONTAINS

   SUBROUTINE put2(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r90_unit) field ; WRITE(r90_unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(r90_unit) value
   END SUBROUTINE put2

   SUBROUTINE put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r90_unit) field ; WRITE(r90_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(r90_unit) value
   END SUBROUTINE put3

   SUBROUTINE put_ref(name, family)
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER, INTENT(in) :: family
      INTEGER :: ji, jj, jk
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: z
      ALLOCATE(z(jpi,jpj,jpk)); z(:,:,:) = 0._wp
      DO jk=1,jpk ; DO jj=1,jpj ; DO ji=1,jpi
         SELECT CASE(family)
         CASE(1) ; z(ji,jj,jk)=e3u_0(ji,jj,jk)
         CASE(2) ; z(ji,jj,jk)=e3v_0(ji,jj,jk)
         CASE DEFAULT ; CALL ctl_stop('round90: invalid reference e3 family')
         END SELECT
      END DO ; END DO ; END DO
      CALL put3(name,z); DEALLOCATE(z)
   END SUBROUTINE put_ref

   SUBROUTINE r90_baro_begin(kt,kstg,Kaa,puu,pvv,pzub,pzvb)
      INTEGER, INTENT(in) :: kt,kstg,Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pzub,pzvb
      INTEGER :: ios
      CHARACTER(LEN=64) :: filename
      r90_active = lwp .AND. kt == nit000 + 1 .AND. kstg == 1
      IF(.NOT.r90_active) RETURN
      IF(r90_unit /= -1) CALL ctl_stop('round90: nested correction record')
      IF(l_istiled) CALL ctl_stop('round90: whole-array writer refuses tiling')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round90: writer requires 64-bit wp')
      WRITE(filename,'("oracle_baro_correction_kt",I8.8,"_s",I1,".bin")') kt,kstg
      OPEN(NEWUNIT=r90_unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round90: cannot open correction record')
      WRITE(r90_unit) r90_magic
      WRITE(r90_unit) 1,kt,kstg,Kaa,jpi,jpj,jpk,jpkm1,ntsi,ntei,ntsj,ntej,STORAGE_SIZE(1._wp)
      CALL put3('baro_raw_u      ',puu(:,:,:,Kaa)); CALL put3('baro_raw_v      ',pvv(:,:,:,Kaa))
      CALL put2('baro_target_u   ',uu_b(:,:,Kaa)); CALL put2('baro_target_v   ',vv_b(:,:,Kaa))
      CALL put2('baro_zub        ',pzub); CALL put2('baro_zvb        ',pzvb)
      CALL put_ref('baro_e3u_0     ',1); CALL put_ref('baro_e3v_0     ',2)
      CALL put2('baro_r1_hu_0   ',r1_hu_0); CALL put2('baro_r1_hv_0   ',r1_hv_0)
      CALL put3('baro_umask     ',umask); CALL put3('baro_vmask     ',vmask)
      WRITE(numout,*) 'ROUND90_BARO_BEGIN ',kt,kstg,Kaa,TRIM(filename)
   END SUBROUTINE r90_baro_begin

   SUBROUTINE r90_baro_finish(kt,kstg,Kaa,puu,pvv)
      INTEGER, INTENT(in) :: kt,kstg,Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      IF(.NOT.r90_active) RETURN
      IF(r90_unit == -1) CALL ctl_stop('round90: correction record was not open')
      CALL put3('baro_final_u    ',puu(:,:,:,Kaa)); CALL put3('baro_final_v    ',pvv(:,:,:,Kaa))
      CLOSE(r90_unit); r90_unit=-1; r90_active=.FALSE.
      WRITE(numout,*) 'ROUND90_BARO_END ',kt,kstg
   END SUBROUTINE r90_baro_finish
END MODULE l2_r90_baro
