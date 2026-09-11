MODULE l2_r46_stage
   !! Round-46 WRITE-only momentum-stage recorder.  No captured value is read
   !! back by NEMO.  The executing call sites are patched in stp2d,
   !! stprk3_stg and dynadv; inactive calls are no-ops.
   USE dom_oce
   USE par_oce,        ONLY : jp_tem, jp_sal
   USE oce,            ONLY : uu_b, vv_b, rhd, ts, ssh
   USE zdf_oce,        ONLY : en, avm_k, avt_k
   USE zdftke,         ONLY : dissl
   USE in_out_manager, ONLY : lwp, numout, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r46_begin, r46_rhs, r46_ww, r46_state, r46_finish

   INTEGER, SAVE :: r46_unit = -1
   LOGICAL, SAVE :: r46_active = .FALSE.
   LOGICAL, SAVE :: saw_hpg = .FALSE., saw_vor = .FALSE., saw_keg = .FALSE.
   LOGICAL, SAVE :: saw_zad = .FALSE., saw_ldf = .FALSE., saw_zdf = .FALSE.
   CHARACTER(LEN=16), PARAMETER :: r46_magic = 'NEMO_L2_R46STG1'

#  include "domzgr_substitute.h90"

CONTAINS

   SUBROUTINE put0(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r46_unit) field ; WRITE(r46_unit) 0, 1, 1, 1
      WRITE(r46_unit) value
   END SUBROUTINE put0

   SUBROUTINE put2(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r46_unit) field ; WRITE(r46_unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(r46_unit) value
   END SUBROUTINE put2

   SUBROUTINE put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r46_unit) field ; WRITE(r46_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(r46_unit) value
   END SUBROUTINE put3

   SUBROUTINE put_live(name, family, level)
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER, INTENT(in) :: family, level
      INTEGER :: ji, jj, jk
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: z
      ALLOCATE(z(jpi,jpj,jpk))
      z(:,:,:) = 0._wp
      DO jk=1,jpk ; DO jj=1,jpj ; DO ji=1,jpi
         SELECT CASE(family)
         CASE(1) ; z(ji,jj,jk)=e3t(ji,jj,jk,level)
         CASE(2) ; z(ji,jj,jk)=e3u(ji,jj,jk,level)
         CASE(3) ; z(ji,jj,jk)=e3v(ji,jj,jk,level)
         CASE(4) ; z(ji,jj,jk)=e3w(ji,jj,jk,level)
         CASE DEFAULT ; CALL ctl_stop('round46: invalid e3 family')
         END SELECT
      END DO ; END DO ; END DO
      CALL put3(name,z)
      DEALLOCATE(z)
   END SUBROUTINE put_live

   SUBROUTINE put_ref(name, family)
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER, INTENT(in) :: family
      INTEGER :: ji, jj, jk
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: z
      ALLOCATE(z(jpi,jpj,jpk)); z(:,:,:) = 0._wp
      DO jk=1,jpk ; DO jj=1,jpj ; DO ji=1,jpi
         SELECT CASE(family)
         CASE(1) ; z(ji,jj,jk)=E3t_0(ji,jj,jk)
         CASE(2) ; z(ji,jj,jk)=E3u_0(ji,jj,jk)
         CASE(3) ; z(ji,jj,jk)=E3v_0(ji,jj,jk)
         CASE(4) ; z(ji,jj,jk)=E3w_0(ji,jj,jk)
         CASE DEFAULT ; CALL ctl_stop('round46: invalid reference e3 family')
         END SELECT
      END DO ; END DO ; END DO
      CALL put3(name,z); DEALLOCATE(z)
   END SUBROUTINE put_ref

   SUBROUTINE r46_begin(kt,kstg,Kbb,Kmm,Krhs,Kaa,puu,pvv)
      INTEGER, INTENT(in) :: kt,kstg,Kbb,Kmm,Krhs,Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      INTEGER :: ios
      CHARACTER(LEN=64) :: filename
      r46_active = lwp .AND. kt >= nit000 .AND. kt <= nit000+1
      IF(.NOT.r46_active) RETURN
      IF(r46_unit /= -1) CALL ctl_stop('round46: nested stage record')
      IF(l_istiled) CALL ctl_stop('round46: whole-array writer refuses tiling')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round46: writer requires 64-bit wp')
      IF(kstg < 1 .OR. kstg > 3) CALL ctl_stop('round46: invalid stage')
      WRITE(filename,'("oracle_momstage_kt",I8.8,"_s",I1,".bin")') kt,kstg
      OPEN(NEWUNIT=r46_unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round46: cannot open stage record')
      WRITE(r46_unit) r46_magic
      WRITE(r46_unit) 1,kt,kstg,Kbb,Kmm,Krhs,Kaa,jpi,jpj,jpk,jpkm1,ntsi,ntei,ntsj,ntej,STORAGE_SIZE(1._wp)
      saw_hpg=.FALSE.; saw_vor=.FALSE.; saw_keg=.FALSE.; saw_zad=.FALSE.; saw_ldf=.FALSE.; saw_zdf=.FALSE.
      CALL put3('u_Kbb           ',puu(:,:,:,Kbb)); CALL put3('v_Kbb           ',pvv(:,:,:,Kbb))
      CALL put3('u_Kmm           ',puu(:,:,:,Kmm)); CALL put3('v_Kmm           ',pvv(:,:,:,Kmm))
      CALL put3('u_Kaa_in        ',puu(:,:,:,Kaa)); CALL put3('v_Kaa_in        ',pvv(:,:,:,Kaa))
      CALL put3('rhd_in          ',rhd)
      CALL put3('T_Kbb           ',ts(:,:,:,jp_tem,Kbb))
      CALL put3('S_Kbb           ',ts(:,:,:,jp_sal,Kbb))
      CALL put2('ssh_Kbb         ',ssh(:,:,Kbb))
      CALL put3('T_Kmm           ',ts(:,:,:,jp_tem,Kmm))
      CALL put3('S_Kmm           ',ts(:,:,:,jp_sal,Kmm))
      CALL put2('ssh_Kmm         ',ssh(:,:,Kmm))
      CALL put3('T_Kaa_in        ',ts(:,:,:,jp_tem,Kaa))
      CALL put3('S_Kaa_in        ',ts(:,:,:,jp_sal,Kaa))
      CALL put2('ssh_Kaa         ',ssh(:,:,Kaa))
      CALL put0('r1_Dt           ',r1_Dt)
      CALL put3('tke_en          ',en)
      CALL put3('tke_avm_k       ',avm_k)
      CALL put3('tke_avt_k       ',avt_k)
      CALL put3('tke_dissl       ',dissl)
      CALL put2('r3t_Kbb         ',r3t(:,:,Kbb)); CALL put2('r3u_Kbb         ',r3u(:,:,Kbb))
      CALL put2('r3v_Kbb         ',r3v(:,:,Kbb))
      CALL put2('r3t_Kmm         ',r3t(:,:,Kmm)); CALL put2('r3u_Kmm         ',r3u(:,:,Kmm))
      CALL put2('r3v_Kmm         ',r3v(:,:,Kmm))
      CALL put2('r3t_Kaa         ',r3t(:,:,Kaa)); CALL put2('r3u_Kaa         ',r3u(:,:,Kaa))
      CALL put2('r3v_Kaa         ',r3v(:,:,Kaa))
      CALL put_live('e3t_Kbb        ',1,Kbb); CALL put_live('e3u_Kbb        ',2,Kbb)
      CALL put_live('e3v_Kbb        ',3,Kbb); CALL put_live('e3w_Kbb        ',4,Kbb)
      CALL put_live('e3t_Kmm        ',1,Kmm); CALL put_live('e3u_Kmm        ',2,Kmm)
      CALL put_live('e3v_Kmm        ',3,Kmm); CALL put_live('e3w_Kmm        ',4,Kmm)
      CALL put_live('e3t_Kaa        ',1,Kaa); CALL put_live('e3u_Kaa        ',2,Kaa)
      CALL put_live('e3v_Kaa        ',3,Kaa); CALL put_live('e3w_Kaa        ',4,Kaa)
      CALL put_ref('e3t_0           ',1); CALL put_ref('e3u_0           ',2)
      CALL put_ref('e3v_0           ',3); CALL put_ref('e3w_0           ',4)
      CALL put3('umask           ',umask); CALL put3('vmask           ',vmask)
      CALL put3('tmask           ',tmask); CALL put3('wmask           ',wmask)
      CALL put2('e1e2t           ',e1e2t); CALL put2('e1e2u           ',e1e2u); CALL put2('e1e2v           ',e1e2v)
      CALL put2('r1_e1e2t        ',r1_e1e2t); CALL put2('e2u             ',e2u); CALL put2('e1v             ',e1v)
      CALL put2('r1_e1e2u        ',r1_e1e2u); CALL put2('r1_e1e2v        ',r1_e1e2v)
      CALL put2('r1_e1u          ',r1_e1u); CALL put2('r1_e2v          ',r1_e2v)
      CALL put2('uu_b_Kbb        ',uu_b(:,:,Kbb)); CALL put2('vv_b_Kbb        ',vv_b(:,:,Kbb))
      CALL put2('uu_b_Kmm        ',uu_b(:,:,Kmm)); CALL put2('vv_b_Kmm        ',vv_b(:,:,Kmm))
      CALL put2('uu_b_Kaa        ',uu_b(:,:,Kaa)); CALL put2('vv_b_Kaa        ',vv_b(:,:,Kaa))
      CALL r46_rhs('rhs_entry       ',puu,pvv,Krhs)
      WRITE(numout,*) 'ROUND46_STAGE_BEGIN ',kt,kstg,Kbb,Kmm,Krhs,Kaa,TRIM(filename)
   END SUBROUTINE r46_begin

   SUBROUTINE r46_rhs(label,puu,pvv,Krhs)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: Krhs
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      IF(.NOT.r46_active) RETURN
      CALL put3(TRIM(label)//'_u',puu(:,:,:,Krhs)); CALL put3(TRIM(label)//'_v',pvv(:,:,:,Krhs))
      IF(INDEX(label,'hpg')>0) saw_hpg=.TRUE.; IF(INDEX(label,'vor')>0) saw_vor=.TRUE.
      IF(INDEX(label,'keg')>0) saw_keg=.TRUE.; IF(INDEX(label,'zad')>0) saw_zad=.TRUE.
      IF(INDEX(label,'ldf')>0) saw_ldf=.TRUE.; IF(INDEX(label,'zdf')>0) saw_zdf=.TRUE.
   END SUBROUTINE r46_rhs

   SUBROUTINE r46_ww(pww)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pww
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: z
      IF(.NOT.r46_active) RETURN
      CALL put3('ww              ',pww)
      ALLOCATE(z(jpi,jpj,jpk)); z(:,:,:)=0._wp
      CALL put3('wsd_effective   ',z); DEALLOCATE(z)
   END SUBROUTINE r46_ww

   SUBROUTINE r46_state(label,puu,pvv,Kaa)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu,pvv
      IF(.NOT.r46_active) RETURN
      CALL put3(TRIM(label)//'_u',puu(:,:,:,Kaa)); CALL put3(TRIM(label)//'_v',pvv(:,:,:,Kaa))
   END SUBROUTINE r46_state

   SUBROUTINE r46_finish(kt,kstg)
      INTEGER, INTENT(in) :: kt,kstg
      IF(.NOT.r46_active) RETURN
      CALL put0('has_hpg         ',MERGE(1._wp,0._wp,saw_hpg)); CALL put0('has_vor         ',MERGE(1._wp,0._wp,saw_vor))
      CALL put0('has_keg         ',MERGE(1._wp,0._wp,saw_keg)); CALL put0('has_zad         ',MERGE(1._wp,0._wp,saw_zad))
      CALL put0('has_ldf         ',MERGE(1._wp,0._wp,saw_ldf)); CALL put0('has_zdf         ',MERGE(1._wp,0._wp,saw_zdf))
      CLOSE(r46_unit); r46_unit=-1; r46_active=.FALSE.
      WRITE(numout,*) 'ROUND46_STAGE_END ',kt,kstg
   END SUBROUTINE r46_finish
END MODULE l2_r46_stage
