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
   PUBLIC :: r333_ldf_disarm, r333_ldf_grad, r333_ldf_horizontal
   PUBLIC :: r333_ldf_vertical_before, r333_ldf_vertical_after
   PUBLIC :: r333_ldf_bottom_after, r333_ldf_a33_first
   PUBLIC :: r333_ldf_a33_raw, r333_ldf_a33_lap
   PUBLIC :: r333_dit, r333_djt, r333_dkt
   PUBLIC :: r333_A11, r333_A22, r333_A13, r333_A23
   PUBLIC :: r333_hmsku, r333_hmskv, r333_fu, r333_fv
   PUBLIC :: r333_vmsku, r333_vmskv, r333_ahu_w, r333_ahv_w
   PUBLIC :: r333_A31, r333_A32, r333_fw_lower, r333_fw_upper
   PUBLIC :: r333_rhs_before, r333_horizontal_seen, r333_vertical_seen
   PUBLIC :: r333_rhs_seen, r333_divergence, r333_divisor
   PUBLIC :: r333_rhs_preadd, r333_rhs_postadd

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
   REAL(wp), ALLOCATABLE, SAVE :: r333_divergence(:,:,:,:), r333_divisor(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_rhs_preadd(:,:,:,:), r333_rhs_postadd(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_a33_msku(:,:,:), r333_a33_mskv(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_a33_ahu_w(:,:,:), r333_a33_ahv_w(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_a33_pah(:,:,:), r333_a33_pakz_raw(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_a33_ze3w2(:,:,:), r333_a33_zcoef0(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_a33_pakz_final(:,:,:), r333_a33_seen(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r333_rDt(:,:,:), r333_r1_Dt(:,:,:)

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
      ALLOCATE(r333_divergence(jpi,jpj,jpk,jpts),r333_divisor(jpi,jpj,jpk,jpts), &
         & r333_rhs_preadd(jpi,jpj,jpk,jpts),r333_rhs_postadd(jpi,jpj,jpk,jpts))
      ALLOCATE(r333_a33_msku(jpi,jpj,jpk),r333_a33_mskv(jpi,jpj,jpk), &
         & r333_a33_ahu_w(jpi,jpj,jpk),r333_a33_ahv_w(jpi,jpj,jpk), &
         & r333_a33_pah(jpi,jpj,jpk),r333_a33_pakz_raw(jpi,jpj,jpk), &
         & r333_a33_ze3w2(jpi,jpj,jpk),r333_a33_zcoef0(jpi,jpj,jpk), &
         & r333_a33_pakz_final(jpi,jpj,jpk),r333_a33_seen(jpi,jpj,jpk), &
         & r333_rDt(jpi,jpj,jpk),r333_r1_Dt(jpi,jpj,jpk))
      r333_pt_kbb=pt(:,:,:,:,Kbb); r333_rhs_before=pt(:,:,:,:,Krhs)
      r333_dit=0._wp; r333_djt=0._wp; r333_dkt=0._wp
      r333_A11=0._wp; r333_A22=0._wp; r333_A13=0._wp; r333_A23=0._wp
      r333_hmsku=0._wp; r333_hmskv=0._wp; r333_fu=0._wp; r333_fv=0._wp
      r333_vmsku=0._wp; r333_vmskv=0._wp; r333_ahu_w=0._wp; r333_ahv_w=0._wp
      r333_A31=0._wp; r333_A32=0._wp; r333_fw_lower=0._wp; r333_fw_upper=0._wp
      r333_horizontal_seen=0._wp; r333_vertical_seen=0._wp; r333_rhs_seen=0._wp
      r333_divergence=0._wp; r333_divisor=0._wp
      r333_rhs_preadd=0._wp; r333_rhs_postadd=0._wp
      r333_a33_msku=0._wp; r333_a33_mskv=0._wp
      r333_a33_ahu_w=0._wp; r333_a33_ahv_w=0._wp
      r333_a33_pah=0._wp; r333_a33_pakz_raw=0._wp
      r333_a33_ze3w2=0._wp; r333_a33_zcoef0=0._wp
      r333_a33_pakz_final=0._wp; r333_a33_seen=0._wp
      r333_rDt=0._wp; r333_r1_Dt=0._wp
      r333_ldf_active=.TRUE.
   END SUBROUTINE r333_ldf_begin

   SUBROUTINE r333_ldf_grad(jn,jk,ilo,ihi,jlo,jhi,dit,djt, &
      & dilo,dihi,djlo,djhi,dkt,dkt_next)
      INTEGER, INTENT(in) :: jn,jk,ilo,ihi,jlo,jhi,dilo,dihi,djlo,djhi
      REAL(wp), DIMENSION(:,:), INTENT(in) :: dit,djt,dkt,dkt_next
      IF(.NOT.r333_ldf_active) RETURN
      r333_dit(ilo:ihi,jlo:jhi,jk,jn)=dit
      r333_djt(ilo:ihi,jlo:jhi,jk,jn)=djt
      r333_dkt(dilo:dihi,djlo:djhi,jk,jn)=dkt
      r333_dkt(dilo:dihi,djlo:djhi,jk+1,jn)=dkt_next
   END SUBROUTINE r333_ldf_grad

   SUBROUTINE r333_ldf_horizontal(jn,jk,ilo,ihi,jlo,jhi,zfu,zfv)
      INTEGER, INTENT(in) :: jn,jk,ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zfu,zfv
      IF(.NOT.r333_ldf_active) RETURN
      r333_fu(ilo:ihi,jlo:jhi,jk,jn)=zfu
      r333_fv(ilo:ihi,jlo:jhi,jk,jn)=zfv
      r333_horizontal_seen(ilo:ihi,jlo:jhi,jk,jn)=1._wp
   END SUBROUTINE r333_ldf_horizontal

   SUBROUTINE r333_ldf_vertical_before(jn,jk,ilo,ihi,jlo,jhi,zfw,rhs)
      INTEGER, INTENT(in) :: jn,jk,ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zfw,rhs
      IF(.NOT.r333_ldf_active) RETURN
      r333_fw_lower(ilo:ihi,jlo:jhi,jk,jn)=zfw
      r333_rhs_preadd(ilo:ihi,jlo:jhi,jk,jn)=rhs
   END SUBROUTINE r333_ldf_vertical_before

   SUBROUTINE r333_ldf_vertical_after(jn,jk,ilo,ihi,jlo,jhi,zfw,rhs)
      INTEGER, INTENT(in) :: jn,jk,ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zfw,rhs
      IF(.NOT.r333_ldf_active) RETURN
      r333_fw_upper(ilo:ihi,jlo:jhi,jk,jn)=zfw
      r333_rhs_postadd(ilo:ihi,jlo:jhi,jk,jn)=rhs
      r333_vertical_seen(ilo:ihi,jlo:jhi,jk,jn)=1._wp
      r333_rhs_seen(ilo:ihi,jlo:jhi,jk,jn)=1._wp
   END SUBROUTINE r333_ldf_vertical_after

   SUBROUTINE r333_ldf_bottom_after(jn,jk,ilo,ihi,jlo,jhi,rhs)
      INTEGER, INTENT(in) :: jn,jk,ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:), INTENT(in) :: rhs
      IF(.NOT.r333_ldf_active) RETURN
      r333_fw_upper(ilo:ihi,jlo:jhi,jk,jn)=0._wp
      r333_rhs_postadd(ilo:ihi,jlo:jhi,jk,jn)=rhs
      r333_vertical_seen(ilo:ihi,jlo:jhi,jk,jn)=1._wp
      r333_rhs_seen(ilo:ihi,jlo:jhi,jk,jn)=1._wp
   END SUBROUTINE r333_ldf_bottom_after

   SUBROUTINE r333_ldf_a33_first(Kmm,ilo,ihi,jlo,jhi,pah,wmask,umask,vmask, &
      & ahtu,ahtv,wslpi,wslpj)
      INTEGER, INTENT(in) :: Kmm,ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pah,wmask,umask,vmask,ahtu,ahtv,wslpi,wslpj
      INTEGER :: i,j,k
      REAL(wp) :: zu,zv
      IF(.NOT.r333_ldf_active) RETURN
      DO k=2,jpk-1; DO j=jlo,jhi; DO i=ilo,ihi
         zu=wmask(i,j,k)/MAX((umask(i,j,k-1)+umask(i-1,j,k)) + &
            & (umask(i-1,j,k-1)+umask(i,j,k)),1._wp)
         zv=wmask(i,j,k)/MAX((vmask(i,j,k-1)+vmask(i,j-1,k)) + &
            & (vmask(i,j-1,k-1)+vmask(i,j,k)),1._wp)
         r333_a33_msku(i,j,k)=zu; r333_a33_mskv(i,j,k)=zv
         r333_a33_ahu_w(i,j,k)=((ahtu(i,j,k-1)+ahtu(i-1,j,k)) + &
            & (ahtu(i-1,j,k-1)+ahtu(i,j,k)))*zu
         r333_a33_ahv_w(i,j,k)=((ahtv(i,j,k-1)+ahtv(i,j-1,k)) + &
            & (ahtv(i,j-1,k-1)+ahtv(i,j,k)))*zv
         r333_a33_pah(i,j,k)=pah(i,j,k)
      END DO; END DO; END DO
   END SUBROUTINE r333_ldf_a33_first

   SUBROUTINE r333_ldf_a33_raw(ilo,ihi,jlo,jhi,pakz)
      INTEGER, INTENT(in) :: ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pakz
      IF(.NOT.r333_ldf_active) RETURN
      r333_a33_pakz_raw(ilo:ihi,jlo:jhi,2:jpk-1)=pakz(ilo:ihi,jlo:jhi,2:jpk-1)
   END SUBROUTINE r333_ldf_a33_raw

   SUBROUTINE r333_ldf_a33_lap(Kmm,ilo,ihi,jlo,jhi,pakz,e3w_1d,r3t,rDt,r1_Dt)
      INTEGER, INTENT(in) :: Kmm,ilo,ihi,jlo,jhi
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pakz
      REAL(wp), DIMENSION(:), INTENT(in) :: e3w_1d
      REAL(wp), DIMENSION(:,:), INTENT(in) :: r3t
      REAL(wp), INTENT(in) :: rDt,r1_Dt
      INTEGER :: i,j,k
      IF(.NOT.r333_ldf_active) RETURN
      DO k=2,jpk-1; DO j=jlo,jhi; DO i=ilo,ihi
         r333_a33_ze3w2(i,j,k)=(e3w_1d(k)*(1._wp+r3t(i,j))) * &
            & (e3w_1d(k)*(1._wp+r3t(i,j)))
         r333_a33_zcoef0(i,j,k)=rDt*(r333_a33_pakz_raw(i,j,k) + &
            & r333_a33_pah(i,j,k)/r333_a33_ze3w2(i,j,k))
         r333_a33_pakz_final(i,j,k)=pakz(i,j,k)
         r333_a33_seen(i,j,k)=1._wp
         r333_rDt(i,j,k)=rDt; r333_r1_Dt(i,j,k)=r1_Dt
      END DO; END DO; END DO
   END SUBROUTINE r333_ldf_a33_lap

   SUBROUTINE r333_ldf_finish(kt,Kbb,Kmm,Krhs,pt,e3t,e3u,e3v,tmask,umask,vmask,wmask, &
      & ahtu,ahtv,uslp,vslp,wslpi,wslpj,ah_wslp2,akz,r3t,r3u,r3v,e2_e1u,e1_e2v, &
      & e2u,e1v,e1u,e2v,e1t,e2t,e1e2t,r1_e1e2t,e3w_1d)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: e3t,e3u,e3v,tmask,umask,vmask,wmask
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: ahtu,ahtv,uslp,vslp,wslpi,wslpj,ah_wslp2,akz
      REAL(wp), DIMENSION(:,:), INTENT(in) :: r3t,r3u,r3v,e2_e1u,e1_e2v,e2u,e1v,e1u,e2v,e1t,e2t,e1e2t,r1_e1e2t
      REAL(wp), DIMENSION(:), INTENT(in) :: e3w_1d
      REAL(wp), ALLOCATABLE :: rhs_after(:,:,:,:),rhs_increment(:,:,:,:)
      INTEGER :: unit,ios,i,j,k,n
      REAL(wp) :: zu,zv
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(.NOT.r333_ldf_active) RETURN
      IF(ANY((/kt,Kbb,Kmm,Krhs/) /= (/arm_kt,arm_Kbb,arm_Kmm,arm_Krhs/))) &
         & CALL ctl_stop('round333: traldf finish levels moved')
      ALLOCATE(rhs_after(jpi,jpj,jpk,jpts),rhs_increment(jpi,jpj,jpk,jpts))
      rhs_after=pt(:,:,:,:,Krhs)
      rhs_increment=rhs_after-r333_rhs_before
      DO n=1,jpts; DO k=1,jpk; DO j=1,jpj; DO i=1,jpi
         IF(r333_horizontal_seen(i,j,k,n)==1._wp) THEN
            r333_A11(i,j,k,n)=e2_e1u(i,j)*e3u(i,j,k)
            r333_A22(i,j,k,n)=e1_e2v(i,j)*e3v(i,j,k)
            zu=1._wp/MAX((wmask(i+1,j,k)+wmask(i,j,k+1)) + &
               & (wmask(i+1,j,k+1)+wmask(i,j,k)),1._wp)
            zv=1._wp/MAX((wmask(i,j+1,k)+wmask(i,j,k+1)) + &
               & (wmask(i,j+1,k+1)+wmask(i,j,k)),1._wp)
            r333_hmsku(i,j,k,n)=zu; r333_hmskv(i,j,k,n)=zv
            r333_A13(i,j,k,n)=-e2u(i,j)*uslp(i,j,k)*zu
            r333_A23(i,j,k,n)=-e1v(i,j)*vslp(i,j,k)*zv
         ENDIF
         IF(r333_vertical_seen(i,j,k,n)==1._wp) THEN
            r333_divergence(i,j,k,n)=((r333_fu(i,j,k,n)-r333_fu(i-1,j,k,n)) + &
               & (r333_fv(i,j,k,n)-r333_fv(i,j-1,k,n))) + &
               & (r333_fw_lower(i,j,k,n)-r333_fw_upper(i,j,k,n))
            r333_divisor(i,j,k,n)=r1_e1e2t(i,j)/e3t(i,j,k)
            IF(k<jpk-1) THEN
               zu=wmask(i,j,k)/MAX((umask(i,j,k)+umask(i-1,j,k+1)) + &
                  & (umask(i-1,j,k)+umask(i,j,k+1)),1._wp)
               zv=wmask(i,j,k)/MAX((vmask(i,j,k)+vmask(i,j-1,k+1)) + &
                  & (vmask(i,j-1,k)+vmask(i,j,k+1)),1._wp)
               r333_vmsku(i,j,k,n)=zu; r333_vmskv(i,j,k,n)=zv
               r333_ahu_w(i,j,k,n)=((ahtu(i,j,k)+ahtu(i-1,j,k+1)) + &
                  & (ahtu(i-1,j,k)+ahtu(i,j,k+1)))*zu
               r333_ahv_w(i,j,k,n)=((ahtv(i,j,k)+ahtv(i,j-1,k+1)) + &
                  & (ahtv(i,j-1,k)+ahtv(i,j,k+1)))*zv
               r333_A31(i,j,k,n)=-r333_ahu_w(i,j,k,n)*e2t(i,j)*zu*wslpi(i,j,k+1)
               r333_A32(i,j,k,n)=-r333_ahv_w(i,j,k,n)*e1t(i,j)*zv*wslpj(i,j,k+1)
            ENDIF
         ENDIF
      END DO; END DO; END DO; END DO
      WRITE(filename,'("oracle_r333_traldf_kt",I8.8,"_r",I4.4,".bin")') kt,narea-1
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round333: cannot open traldf record')
      magic='NEMO_L4_R333LI'
      WRITE(unit) magic,1,kt,arm_stage,Kbb,Kmm,Krhs,narea-1,jpi,jpj,jpk,jpts, &
         & STORAGE_SIZE(1._wp),71
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
      CALL put4(unit,'divergence',r333_divergence); CALL put4(unit,'divisor',r333_divisor)
      CALL put4(unit,'rhs_preadd',r333_rhs_preadd); CALL put4(unit,'rhs_postadd',r333_rhs_postadd)
      CALL put3(unit,'e3t_3d',e3t); CALL put3(unit,'e3u_3d',e3u); CALL put3(unit,'e3v_3d',e3v)
      CALL put3(unit,'tmask',tmask); CALL put3(unit,'umask',umask); CALL put3(unit,'vmask',vmask)
      CALL put3(unit,'wmask',wmask); CALL put3(unit,'ahtu',ahtu); CALL put3(unit,'ahtv',ahtv)
      CALL put3(unit,'uslp',uslp); CALL put3(unit,'vslp',vslp)
      CALL put3(unit,'wslpi',wslpi); CALL put3(unit,'wslpj',wslpj)
      CALL put3(unit,'ah_wslp2',ah_wslp2); CALL put3(unit,'akz',akz)
      CALL put3(unit,'a33_msku',r333_a33_msku); CALL put3(unit,'a33_mskv',r333_a33_mskv)
      CALL put3(unit,'a33_ahu_w',r333_a33_ahu_w); CALL put3(unit,'a33_ahv_w',r333_a33_ahv_w)
      CALL put3(unit,'a33_pah',r333_a33_pah); CALL put3(unit,'a33_pakz_raw',r333_a33_pakz_raw)
      CALL put3(unit,'a33_ze3w2',r333_a33_ze3w2); CALL put3(unit,'a33_zcoef0',r333_a33_zcoef0)
      CALL put3(unit,'a33_pakz_final',r333_a33_pakz_final); CALL put3(unit,'a33_seen',r333_a33_seen)
      CALL put3(unit,'rDt',r333_rDt); CALL put3(unit,'r1_Dt',r333_r1_Dt)
      CALL put2(unit,'r3t_kmm',r3t); CALL put2(unit,'r3u_kmm',r3u); CALL put2(unit,'r3v_kmm',r3v)
      CALL put2(unit,'e2_e1u',e2_e1u); CALL put2(unit,'e1_e2v',e1_e2v)
      CALL put2(unit,'e2u',e2u); CALL put2(unit,'e1v',e1v)
      CALL put2(unit,'e1u',e1u); CALL put2(unit,'e2v',e2v); CALL put2(unit,'e1t',e1t)
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
