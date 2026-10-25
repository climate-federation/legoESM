MODULE round333_traldf_writer
   !! Passive full-array recorder for kt=2 stage-3 traldf_iso_lap.
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE par_kind       , ONLY : wp
   USE par_oce        , ONLY : jpi, jpj, jpk, jpts
   USE dom_oce        , ONLY : narea
   USE in_out_manager , ONLY : nit000
   USE lib_mpp        , ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: r333_ldf_active, r333_ldf_arm, r333_ldf_begin, r333_ldf_finish
   PUBLIC :: r333_ldf_disarm
   PUBLIC :: r333_dit, r333_djt, r333_dkt
   PUBLIC :: r333_A11, r333_A22, r333_A13, r333_A23
   PUBLIC :: r333_hmsku, r333_hmskv, r333_fu, r333_fv
   PUBLIC :: r333_vmsku, r333_vmskv, r333_ahu_w, r333_ahv_w
   PUBLIC :: r333_A31, r333_A32, r333_fw_lower, r333_fw_upper
   PUBLIC :: r333_rhs_before, r333_horizontal_seen, r333_vertical_seen
   PUBLIC :: r333_rhs_seen

   LOGICAL, SAVE :: r333_ldf_active=.FALSE., armed=.FALSE., written=.FALSE.
   INTEGER, SAVE :: arm_kt=-1, arm_stage=-1, arm_Kbb=-1, arm_Kmm=-1, arm_Krhs=-1
   REAL(wp), ALLOCATABLE, SAVE :: r333_pt_kbb(:,:,:,:), r333_rhs_before(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_dit(:,:,:,:), r333_djt(:,:,:,:), r333_dkt(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_A11(:,:,:,:), r333_A22(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_A13(:,:,:,:), r333_A23(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_hmsku(:,:,:,:), r333_hmskv(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_fu(:,:,:,:), r333_fv(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_vmsku(:,:,:,:), r333_vmskv(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_ahu_w(:,:,:,:), r333_ahv_w(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_A31(:,:,:,:), r333_A32(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_fw_lower(:,:,:,:), r333_fw_upper(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_horizontal_seen(:,:,:,:), r333_vertical_seen(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_rhs_seen(:,:,:,:)

CONTAINS

   SUBROUTINE r333_ldf_arm(kt,kstage,Kbb,Kmm,Krhs)
      INTEGER, INTENT(in) :: kt,kstage,Kbb,Kmm,Krhs
      IF(kt /= nit000+1 .OR. kstage /= 3) RETURN
      IF(armed .OR. r333_ldf_active .OR. written) CALL ctl_stop('round333: duplicate traldf arm')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round333: traldf record requires fp64')
      arm_kt=kt; arm_stage=kstage; arm_Kbb=Kbb; arm_Kmm=Kmm; arm_Krhs=Krhs
      armed=.TRUE.
   END SUBROUTINE r333_ldf_arm

   SUBROUTINE r333_ldf_begin(kt,Kbb,Kmm,Krhs,pt)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      IF(.NOT.armed) RETURN
      IF(ANY((/kt,Kbb,Kmm,Krhs/) /= (/arm_kt,arm_Kbb,arm_Kmm,arm_Krhs/))) &
         & CALL ctl_stop('round333: traldf time levels moved')
      IF(SIZE(pt,1)/=jpi .OR. SIZE(pt,2)/=jpj .OR. SIZE(pt,3)/=jpk .OR. &
         & SIZE(pt,4)/=jpts) CALL ctl_stop('round333: traldf input extent moved')
      ALLOCATE(r333_pt_kbb(jpi,jpj,jpk,jpts),r333_rhs_before(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_dit(jpi,jpj,jpk,jpts),r333_djt(jpi,jpj,jpk,jpts),r333_dkt(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_A11(jpi,jpj,jpk,jpts),r333_A22(jpi,jpj,jpk,jpts), &
         & r333_A13(jpi,jpj,jpk,jpts),r333_A23(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_hmsku(jpi,jpj,jpk,jpts),r333_hmskv(jpi,jpj,jpk,jpts), &
         & r333_fu(jpi,jpj,jpk,jpts),r333_fv(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_vmsku(jpi,jpj,jpk,jpts),r333_vmskv(jpi,jpj,jpk,jpts), &
         & r333_ahu_w(jpi,jpj,jpk,jpts),r333_ahv_w(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_A31(jpi,jpj,jpk,jpts),r333_A32(jpi,jpj,jpk,jpts), &
         & r333_fw_lower(jpi,jpj,jpk,jpts),r333_fw_upper(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_horizontal_seen(jpi,jpj,jpk,jpts), &
         & r333_vertical_seen(jpi,jpj,jpk,jpts),r333_rhs_seen(jpi,jpj,jpk,jpts))
      r333_pt_kbb=pt(:,:,:,:,Kbb); r333_rhs_before=pt(:,:,:,:,Krhs)
      r333_dit=0._wp; r333_djt=0._wp; r333_dkt=0._wp
      r333_A11=0._wp; r333_A22=0._wp; r333_A13=0._wp; r333_A23=0._wp
      r333_hmsku=0._wp; r333_hmskv=0._wp; r333_fu=0._wp; r333_fv=0._wp
      r333_vmsku=0._wp; r333_vmskv=0._wp; r333_ahu_w=0._wp; r333_ahv_w=0._wp
      r333_A31=0._wp; r333_A32=0._wp; r333_fw_lower=0._wp; r333_fw_upper=0._wp
      r333_horizontal_seen=0._wp; r333_vertical_seen=0._wp; r333_rhs_seen=0._wp
      r333_ldf_active=.TRUE.
   END SUBROUTINE r333_ldf_begin

   SUBROUTINE r333_ldf_finish(kt,Kbb,Kmm,Krhs,pt,e3t,e3u,e3v,tmask,umask,vmask,wmask, &
      & ahtu,ahtv,uslp,vslp,wslpi,wslpj,ah_wslp2,akz,r3t,r3u,r3v,e2_e1u,e1_e2v, &
      & e2u,e1v,e1t,e2t,e1e2t,r1_e1e2t,e3w_1d)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: e3t,e3u,e3v,tmask,umask,vmask,wmask
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: ahtu,ahtv,uslp,vslp,wslpi,wslpj,ah_wslp2,akz
      REAL(wp), DIMENSION(:,:), INTENT(in) :: r3t,r3u,r3v,e2_e1u,e1_e2v,e2u,e1v,e1t,e2t,e1e2t,r1_e1e2t
      REAL(wp), DIMENSION(:), INTENT(in) :: e3w_1d
      REAL(wp), ALLOCATABLE :: rhs_after(:,:,:,:),rhs_increment(:,:,:,:)
      INTEGER :: unit,ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(.NOT.r333_ldf_active) RETURN
      IF(ANY((/kt,Kbb,Kmm,Krhs/) /= (/arm_kt,arm_Kbb,arm_Kmm,arm_Krhs/))) &
         & CALL ctl_stop('round333: traldf finish levels moved')
      ALLOCATE(rhs_after(jpi,jpj,jpk,jpts),rhs_increment(jpi,jpj,jpk,jpts))
      rhs_after=pt(:,:,:,:,Krhs)
      rhs_increment=rhs_after-r333_rhs_before
      WRITE(filename,'("oracle_r333_traldf_kt",I8.8,"_r",I4.4,".bin")') kt,narea-1
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round333: cannot open traldf record')
      magic='NEMO_L4_R333LI'
      WRITE(unit) magic,1,kt,arm_stage,Kbb,Kmm,Krhs,narea-1,jpi,jpj,jpk,jpts, &
         & STORAGE_SIZE(1._wp),53
      CALL put4(unit,'pt_kbb',r333_pt_kbb); CALL put4(unit,'rhs_before',r333_rhs_before)
      CALL put4(unit,'rhs_after',rhs_after); CALL put4(unit,'rhs_increment',rhs_increment)
      CALL put4(unit,'dit',r333_dit); CALL put4(unit,'djt',r333_djt); CALL put4(unit,'dkt',r333_dkt)
      CALL put4(unit,'A11',r333_A11); CALL put4(unit,'A22',r333_A22)
      CALL put4(unit,'A13',r333_A13); CALL put4(unit,'A23',r333_A23)
      CALL put4(unit,'hmsku',r333_hmsku); CALL put4(unit,'hmskv',r333_hmskv)
      CALL put4(unit,'zfu',r333_fu); CALL put4(unit,'zfv',r333_fv)
      CALL put4(unit,'vmsku',r333_vmsku); CALL put4(unit,'vmskv',r333_vmskv)
      CALL put4(unit,'zahu_w',r333_ahu_w); CALL put4(unit,'zahv_w',r333_ahv_w)
      CALL put4(unit,'A31',r333_A31); CALL put4(unit,'A32',r333_A32)
      CALL put4(unit,'zfw_lower',r333_fw_lower); CALL put4(unit,'zfw_upper',r333_fw_upper)
      CALL put4(unit,'horizontal_seen',r333_horizontal_seen)
      CALL put4(unit,'vertical_seen',r333_vertical_seen); CALL put4(unit,'rhs_seen',r333_rhs_seen)
      CALL put3(unit,'e3t_3d',e3t); CALL put3(unit,'e3u_3d',e3u); CALL put3(unit,'e3v_3d',e3v)
      CALL put3(unit,'tmask',tmask); CALL put3(unit,'umask',umask); CALL put3(unit,'vmask',vmask)
      CALL put3(unit,'wmask',wmask); CALL put3(unit,'ahtu',ahtu); CALL put3(unit,'ahtv',ahtv)
      CALL put3(unit,'uslp',uslp); CALL put3(unit,'vslp',vslp)
      CALL put3(unit,'wslpi',wslpi); CALL put3(unit,'wslpj',wslpj)
      CALL put3(unit,'ah_wslp2',ah_wslp2); CALL put3(unit,'akz',akz)
      CALL put2(unit,'r3t_kmm',r3t); CALL put2(unit,'r3u_kmm',r3u); CALL put2(unit,'r3v_kmm',r3v)
      CALL put2(unit,'e2_e1u',e2_e1u); CALL put2(unit,'e1_e2v',e1_e2v)
      CALL put2(unit,'e2u',e2u); CALL put2(unit,'e1v',e1v); CALL put2(unit,'e1t',e1t)
      CALL put2(unit,'e2t',e2t); CALL put2(unit,'e1e2t',e1e2t); CALL put2(unit,'r1_e1e2t',r1_e1e2t)
      CALL put1(unit,'e3w_1d',e3w_1d)
      CLOSE(unit,IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round333: cannot close traldf record')
      DEALLOCATE(rhs_after,rhs_increment)
      r333_ldf_active=.FALSE.; written=.TRUE.
   END SUBROUTINE r333_ldf_finish

   SUBROUTINE r333_ldf_disarm(kt,kstage)
      INTEGER, INTENT(in) :: kt,kstage
      IF(kt/=nit000+1 .OR. kstage/=3) RETURN
      IF(.NOT.armed .OR. r333_ldf_active .OR. .NOT.written) &
         & CALL ctl_stop('round333: incomplete traldf record')
      armed=.FALSE.
   END SUBROUTINE r333_ldf_disarm

   SUBROUTINE put4(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 4,SIZE(value,1),SIZE(value,2),SIZE(value,3),SIZE(value,4),1,1,1,1
      WRITE(unit) checksum4(value); WRITE(unit) value
   END SUBROUTINE put4

   SUBROUTINE put3(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),1,1,1,1,1
      WRITE(unit) checksum3(value); WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE put2(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 2,SIZE(value,1),SIZE(value,2),1,1,1,1,1,1
      WRITE(unit) checksum2(value); WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE put1(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 1,SIZE(value,1),1,1,1,1,1,1,1
      WRITE(unit) checksum1(value); WRITE(unit) value
   END SUBROUTINE put1

   FUNCTION checksum4(value) RESULT(checksum)
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i,j,k,n
      checksum=0_int64
      DO n=1,SIZE(value,4); DO k=1,SIZE(value,3); DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j,k,n),word); checksum=IEOR(checksum,word)
      END DO; END DO; END DO; END DO
   END FUNCTION checksum4

   FUNCTION checksum3(value) RESULT(checksum)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i,j,k
      checksum=0_int64
      DO k=1,SIZE(value,3); DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j,k),word); checksum=IEOR(checksum,word)
      END DO; END DO; END DO
   END FUNCTION checksum3

   FUNCTION checksum2(value) RESULT(checksum)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i,j
      checksum=0_int64
      DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j),word); checksum=IEOR(checksum,word)
      END DO; END DO
   END FUNCTION checksum2

   FUNCTION checksum1(value) RESULT(checksum)
      REAL(wp), DIMENSION(:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i
      checksum=0_int64
      DO i=1,SIZE(value,1)
         word=TRANSFER(value(i),word); checksum=IEOR(checksum,word)
      END DO
   END FUNCTION checksum1
END MODULE round333_traldf_writer
