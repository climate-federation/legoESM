MODULE l1_r50_pair
   !! Round-50 WRITE-only kt=3 source-order recorder for OVERFLOW.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r50_mom_begin, r50_mom_rhd, r50_mom_uv, r50_mom_finish, &
      & r50_tra_begin, r50_tra_ts, r50_tra_finish

   INTEGER, SAVE :: mom_unit = -1, tra_unit = -1
   INTEGER, SAVE :: mom_expected = 0, mom_written = 0
   INTEGER, SAVE :: tra_expected = 0, tra_written = 0
   LOGICAL, SAVE :: mom_active = .FALSE., tra_active = .FALSE.

#  include "do_loop_substitute.h90"

CONTAINS

   SUBROUTINE put2(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(T2D(0)), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, ntei-ntsi+1, ntej-ntsj+1, 1
      WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE put3(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(T2D(0),jpkm1), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 3, ntei-ntsi+1, ntej-ntsj+1, jpkm1
      WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE mom_put_uv(label,puu,pvv,slot)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: slot
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      IF(.NOT.mom_active) RETURN
      CALL put3(mom_unit,TRIM(label)//'_u',puu(ntsi:ntei,ntsj:ntej,1:jpkm1,slot))
      CALL put3(mom_unit,TRIM(label)//'_v',pvv(ntsi:ntei,ntsj:ntej,1:jpkm1,slot))
      mom_written = mom_written + 2
   END SUBROUTINE mom_put_uv

   SUBROUTINE r50_mom_begin(kt,kstg,Kbb,Kmm,Krhs,Kaa,pts,pssh,puu,pvv)
      INTEGER, INTENT(in) :: kt,kstg,Kbb,Kmm,Krhs,Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      REAL(wp), DIMENSION(jpi,jpj,jpt), INTENT(in) :: pssh
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      mom_active = lwp .AND. kt == nit000+2
      IF(.NOT.mom_active) RETURN
      IF(mom_unit /= -1) CALL ctl_stop('round50: nested momentum record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round50: momentum requires fp64')
      IF(jpts /= 2) CALL ctl_stop('round50: momentum requires T/S')
      SELECT CASE(kstg)
      CASE(1)
         mom_expected = 16
      CASE(2)
         mom_expected = 21
      CASE(3)
         mom_expected = 23
      CASE DEFAULT
         CALL ctl_stop('round50: bad momentum stage')
      END SELECT
      WRITE(filename,'("oracle_r50_momentum_kt",I8.8,"_s",I1,".bin")') kt,kstg
      OPEN(NEWUNIT=mom_unit,FILE=TRIM(filename),ACCESS='STREAM', &
         & FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round50: cannot open momentum record')
      magic = 'NEMO_L1_R50MOM1'
      WRITE(mom_unit) magic
      WRITE(mom_unit) 1,kt,kstg,Kbb,Kmm,Krhs,Kaa,ntei-ntsi+1, &
         & ntej-ntsj+1,jpkm1,ntsi,ntsj,STORAGE_SIZE(1._wp),mom_expected
      mom_written = 0
      CALL put3(mom_unit,'Kbb_T',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Kbb))
      CALL put3(mom_unit,'Kbb_S',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Kbb))
      CALL put2(mom_unit,'Kbb_ssh',pssh(ntsi:ntei,ntsj:ntej,Kbb))
      CALL put3(mom_unit,'Kmm_T',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Kmm))
      CALL put3(mom_unit,'Kmm_S',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Kmm))
      CALL put2(mom_unit,'Kmm_ssh',pssh(ntsi:ntei,ntsj:ntej,Kmm))
      mom_written = 6
      CALL mom_put_uv('rhs_entry',puu,pvv,Krhs)
   END SUBROUTINE r50_mom_begin

   SUBROUTINE r50_mom_rhd(prhd)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: prhd
      IF(.NOT.mom_active) RETURN
      CALL put3(mom_unit,'rhd',prhd(ntsi:ntei,ntsj:ntej,1:jpkm1))
      mom_written = mom_written + 1
   END SUBROUTINE r50_mom_rhd

   SUBROUTINE r50_mom_uv(label,puu,pvv,slot)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: slot
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      CALL mom_put_uv(label,puu,pvv,slot)
   END SUBROUTINE r50_mom_uv

   SUBROUTINE r50_mom_finish(puu,pvv,Kaa)
      INTEGER, INTENT(in) :: Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      IF(.NOT.mom_active) RETURN
      CALL mom_put_uv('postbar_kaa',puu,pvv,Kaa)
      IF(mom_written /= mom_expected) CALL ctl_stop('round50: momentum field count')
      CLOSE(mom_unit)
      mom_unit = -1
      mom_active = .FALSE.
   END SUBROUTINE r50_mom_finish

   SUBROUTINE tra_put_ts(label,pts,slot)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: slot
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      IF(.NOT.tra_active) RETURN
      CALL put3(tra_unit,TRIM(label)//'_T', &
         & pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,slot))
      CALL put3(tra_unit,TRIM(label)//'_S', &
         & pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,slot))
      tra_written = tra_written + 2
   END SUBROUTINE tra_put_ts

   SUBROUTINE r50_tra_begin(kt,kstg,Kbb,Kmm,Krhs,Kaa,pts,pssh)
      INTEGER, INTENT(in) :: kt,kstg,Kbb,Kmm,Krhs,Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      REAL(wp), DIMENSION(jpi,jpj,jpt), INTENT(in) :: pssh
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      tra_active = lwp .AND. kt == nit000+2
      IF(.NOT.tra_active) RETURN
      IF(tra_unit /= -1) CALL ctl_stop('round50: nested tracer record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round50: tracer requires fp64')
      IF(jpts /= 2) CALL ctl_stop('round50: tracer requires T/S')
      SELECT CASE(kstg)
      CASE(1,2)
         tra_expected = 15
      CASE(3)
         tra_expected = 21
      CASE DEFAULT
         CALL ctl_stop('round50: bad tracer stage')
      END SELECT
      WRITE(filename,'("oracle_r50_tracer_kt",I8.8,"_s",I1,".bin")') kt,kstg
      OPEN(NEWUNIT=tra_unit,FILE=TRIM(filename),ACCESS='STREAM', &
         & FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round50: cannot open tracer record')
      magic = 'NEMO_L1_R50TRA1'
      WRITE(tra_unit) magic
      WRITE(tra_unit) 1,kt,kstg,Kbb,Kmm,Krhs,Kaa,ntei-ntsi+1, &
         & ntej-ntsj+1,jpkm1,ntsi,ntsj,STORAGE_SIZE(1._wp),tra_expected
      tra_written = 0
      CALL put3(tra_unit,'Kbb_T',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Kbb))
      CALL put3(tra_unit,'Kbb_S',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Kbb))
      CALL put2(tra_unit,'Kbb_ssh',pssh(ntsi:ntei,ntsj:ntej,Kbb))
      CALL put3(tra_unit,'Kmm_T',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Kmm))
      CALL put3(tra_unit,'Kmm_S',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Kmm))
      CALL put2(tra_unit,'Kmm_ssh',pssh(ntsi:ntei,ntsj:ntej,Kmm))
      tra_written = 6
      CALL tra_put_ts('rhs_entry',pts,Krhs)
   END SUBROUTINE r50_tra_begin

   SUBROUTINE r50_tra_ts(label,pts,slot)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: slot
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      CALL tra_put_ts(label,pts,slot)
   END SUBROUTINE r50_tra_ts

   SUBROUTINE r50_tra_finish(pts,pssh,Kaa)
      INTEGER, INTENT(in) :: Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      REAL(wp), DIMENSION(jpi,jpj,jpt), INTENT(in) :: pssh
      IF(.NOT.tra_active) RETURN
      CALL tra_put_ts('Kaa',pts,Kaa)
      CALL put2(tra_unit,'Kaa_ssh',pssh(ntsi:ntei,ntsj:ntej,Kaa))
      tra_written = tra_written + 1
      IF(tra_written /= tra_expected) CALL ctl_stop('round50: tracer field count')
      CLOSE(tra_unit)
      tra_unit = -1
      tra_active = .FALSE.
   END SUBROUTINE r50_tra_finish

END MODULE l1_r50_pair
