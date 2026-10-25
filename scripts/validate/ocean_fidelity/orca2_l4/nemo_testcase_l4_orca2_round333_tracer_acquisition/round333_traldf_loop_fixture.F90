PROGRAM round333_traldf_loop_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE, INTRINSIC :: ieee_arithmetic, ONLY : ieee_is_negative
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi,jpj,jpk,jpts
   USE dom_oce, ONLY : narea
   USE round333_traldf_writer
   IMPLICIT NONE
   INTEGER, PARAMETER :: ntsi=2, ntei=jpi-1, ntsj=2, ntej=jpj-1, jpkm1=jpk-1
   INTEGER, PARAMETER :: Kbb=3, Kmm=2, Krhs=1
   REAL(wp), PARAMETER :: rDt=0.25_wp, r1_Dt=4._wp
   REAL(wp) :: ref_pt(jpi,jpj,jpk,jpts,3),obs_pt(jpi,jpj,jpk,jpts,3)
   REAL(wp) :: e3t_3d(jpi,jpj,jpk),e3u_3d(jpi,jpj,jpk),e3v_3d(jpi,jpj,jpk)
   REAL(wp) :: tmask(jpi,jpj,jpk),umask(jpi,jpj,jpk),vmask(jpi,jpj,jpk),wmask(jpi,jpj,jpk)
   REAL(wp) :: ahtu(jpi,jpj,jpk),ahtv(jpi,jpj,jpk),uslp(jpi,jpj,jpk),vslp(jpi,jpj,jpk)
   REAL(wp) :: wslpi(jpi,jpj,jpk),wslpj(jpi,jpj,jpk),ah_wslp2(jpi,jpj,jpk),akz(jpi,jpj,jpk)
   REAL(wp) :: pakz_raw(jpi,jpj,jpk),e3w_1d(jpk)
   REAL(wp) :: r3t(jpi,jpj,3),r3u(jpi,jpj,3),r3v(jpi,jpj,3)
   REAL(wp) :: e2_e1u(jpi,jpj),e1_e2v(jpi,jpj),e2u(jpi,jpj),e1v(jpi,jpj)
   REAL(wp) :: e1u(jpi,jpj),e2v(jpi,jpj),e1t(jpi,jpj),e2t(jpi,jpj)
   REAL(wp) :: e1e2t(jpi,jpj),r1_e1e2t(jpi,jpj)
   INTEGER(int64) :: ref_bits(SIZE(ref_pt)),obs_bits(SIZE(obs_pt))
   CHARACTER(LEN=8) :: argument
   INTEGER :: rank,ji,jj,jk,jn

   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   narea=rank+1
   DO jk=1,jpk
      e3w_1d(jk)=1.25_wp+0.2_wp*jk+0.01_wp*rank
      DO jj=1,jpj; DO ji=1,jpi
         e3t_3d(ji,jj,jk)=1.1_wp+0.01_wp*(ji+2*jj+3*jk+rank)
         e3u_3d(ji,jj,jk)=1.2_wp+0.01_wp*(2*ji+jj+2*jk+rank)
         e3v_3d(ji,jj,jk)=1.3_wp+0.01_wp*(ji+3*jj+jk+rank)
         tmask(ji,jj,jk)=1._wp; umask(ji,jj,jk)=1._wp
         vmask(ji,jj,jk)=1._wp; wmask(ji,jj,jk)=1._wp
         ahtu(ji,jj,jk)=0.7_wp+0.01_wp*(ji+jj+jk+rank)
         ahtv(ji,jj,jk)=0.8_wp+0.01_wp*(ji+2*jj+jk+rank)
         uslp(ji,jj,jk)=0.001_wp*REAL(ji+jj+jk+rank,wp)
         vslp(ji,jj,jk)=0.0015_wp*REAL(ji+2*jj+jk+rank,wp)
         wslpi(ji,jj,jk)=0.0007_wp*REAL(ji+jj+jk+rank,wp)
         wslpj(ji,jj,jk)=0.0009_wp*REAL(ji+2*jj+jk+rank,wp)
         DO jn=1,jpts
            ref_pt(ji,jj,jk,jn,Krhs)=REAL(1000*jn+100*jk+10*jj+ji+rank,wp)/113._wp
            ref_pt(ji,jj,jk,jn,Kmm)=REAL(2000*jn+100*jk+10*jj+ji+rank,wp)/109._wp
            ref_pt(ji,jj,jk,jn,Kbb)=REAL(3000*jn+100*jk+10*jj+ji+rank,wp)/107._wp
         END DO
      END DO; END DO
   END DO
   wmask(:,:,jpk)=0._wp
   umask(1,1,1)=-0._wp
   DO jj=1,jpj; DO ji=1,jpi
      r3t(ji,jj,:)=0._wp; r3u(ji,jj,:)=0._wp; r3v(ji,jj,:)=0._wp
      r3t(ji,jj,Kmm)=0.002_wp*REAL(ji+jj+rank,wp)
      r3u(ji,jj,Kmm)=0.003_wp*REAL(ji+2*jj+rank,wp)
      r3v(ji,jj,Kmm)=0.004_wp*REAL(2*ji+jj+rank,wp)
      e2_e1u(ji,jj)=1.05_wp+0.001_wp*(ji+jj+rank)
      e1_e2v(ji,jj)=0.95_wp+0.001_wp*(2*ji+jj+rank)
      e2u(ji,jj)=1.2_wp+0.01_wp*(ji+rank); e1v(ji,jj)=1.3_wp+0.01_wp*(jj+rank)
      e1u(ji,jj)=1.4_wp+0.01_wp*(ji+jj+rank); e2v(ji,jj)=1.5_wp+0.01_wp*(ji+jj+rank)
      e1t(ji,jj)=1.6_wp+0.01_wp*(ji+rank); e2t(ji,jj)=1.7_wp+0.01_wp*(jj+rank)
      e1e2t(ji,jj)=e1t(ji,jj)*e2t(ji,jj); r1_e1e2t(ji,jj)=1._wp/e1e2t(ji,jj)
   END DO; END DO
   CALL compute_a33
   obs_pt=ref_pt
   CALL reference_kernel(ref_pt)
   CALL r333_ldf_arm(2,3,Kbb,Kmm,Krhs)
   CALL r333_ldf_begin(2,Kbb,Kmm,Krhs,obs_pt)
   CALL r333_ldf_a33_first(Kmm,ntsi,ntei,ntsj,ntej,ah_wslp2,wmask,umask,vmask, &
      & ahtu,ahtv,wslpi,wslpj)
   CALL r333_ldf_a33_raw(ntsi,ntei,ntsj,ntej,pakz_raw)
   CALL r333_ldf_a33_lap(Kmm,ntsi,ntei,ntsj,ntej,akz,e3w_1d,r3t(:,:,Kmm),rDt,r1_Dt)
   CALL observed_kernel(obs_pt)
   CALL r333_ldf_finish(2,Kbb,Kmm,Krhs,obs_pt,e3t_3d,e3u_3d,e3v_3d,tmask,umask,vmask,wmask, &
      & ahtu,ahtv,uslp,vslp,wslpi,wslpj,ah_wslp2,akz,r3t(:,:,Kmm),r3u(:,:,Kmm), &
      & r3v(:,:,Kmm),e2_e1u,e1_e2v,e2u,e1v,e1u,e2v,e1t,e2t,e1e2t,r1_e1e2t,e3w_1d)
   CALL r333_ldf_disarm(2,3)
   ref_bits=TRANSFER(ref_pt,ref_bits); obs_bits=TRANSFER(obs_pt,obs_bits)
   IF(ANY(ref_bits/=obs_bits)) ERROR STOP 'actual patched traldf scheme changed arithmetic'
   IF(.NOT.ANY(r333_horizontal_seen==1._wp) .OR. .NOT.ANY(r333_rhs_seen==1._wp)) &
      & ERROR STOP 'actual patched traldf scheme did not execute observers'
   IF(.NOT.ANY((r333_dit==0._wp) .AND. ieee_is_negative(r333_dit))) &
      & ERROR STOP 'actual patched traldf scheme lost negative zero'
   WRITE(*,'(A,1X,I0)') 'R333_TRALDF_PATCHED_LOOP_FIXTURE_PASS',rank

CONTAINS

   SUBROUTINE compute_a33
      REAL(wp) :: zmsku,zmskv,zahu,zahv,ze3w2,zcoef0
      ah_wslp2=0._wp; pakz_raw=0._wp; akz=0._wp
      DO jk=2,jpk-1; DO jj=ntsj,ntej; DO ji=ntsi,ntei
         zmsku=wmask(ji,jj,jk)/MAX((umask(ji,jj,jk-1)+umask(ji-1,jj,jk)) + &
            & (umask(ji-1,jj,jk-1)+umask(ji,jj,jk)),1._wp)
         zmskv=wmask(ji,jj,jk)/MAX((vmask(ji,jj,jk-1)+vmask(ji,jj-1,jk)) + &
            & (vmask(ji,jj-1,jk-1)+vmask(ji,jj,jk)),1._wp)
         zahu=((ahtu(ji,jj,jk-1)+ahtu(ji-1,jj,jk)) + &
            & (ahtu(ji-1,jj,jk-1)+ahtu(ji,jj,jk)))*zmsku
         zahv=((ahtv(ji,jj,jk-1)+ahtv(ji,jj-1,jk)) + &
            & (ahtv(ji,jj-1,jk-1)+ahtv(ji,jj,jk)))*zmskv
         ah_wslp2(ji,jj,jk)=zahu*wslpi(ji,jj,jk)*wslpi(ji,jj,jk) + &
            & zahv*wslpj(ji,jj,jk)*wslpj(ji,jj,jk)
         pakz_raw(ji,jj,jk)=(((ahtu(ji,jj,jk)+ahtu(ji,jj,jk-1))/(e1u(ji,jj)*e1u(ji,jj)) + &
            & (ahtu(ji-1,jj,jk)+ahtu(ji-1,jj,jk-1))/(e1u(ji-1,jj)*e1u(ji-1,jj))) + &
            & ((ahtv(ji,jj,jk)+ahtv(ji,jj,jk-1))/(e2v(ji,jj)*e2v(ji,jj)) + &
            & (ahtv(ji,jj-1,jk)+ahtv(ji,jj-1,jk-1))/(e2v(ji,jj-1)*e2v(ji,jj-1))))*0.25_wp
         ze3w2=(e3w_1d(jk)*(1._wp+r3t(ji,jj,Kmm)))*(e3w_1d(jk)*(1._wp+r3t(ji,jj,Kmm)))
         zcoef0=rDt*(pakz_raw(ji,jj,jk)+ah_wslp2(ji,jj,jk)/ze3w2)
         akz(ji,jj,jk)=MAX(zcoef0-0.5_wp,0._wp)*ze3w2*r1_Dt
      END DO; END DO; END DO
   END SUBROUTINE compute_a33

   SUBROUTINE reference_kernel(pt)
      REAL(wp), INTENT(inout) :: pt(jpi,jpj,jpk,jpts,3)
      REAL(wp) :: zdit(1:jpi,1:jpj,0:1),zdjt(1:jpi,1:jpj,0:1),zdkt(1:jpi,1:jpj,0:1)
      REAL(wp) :: zfu(1:jpi,1:jpj),zfv(1:jpi,1:jpj),zfw(ntsi:ntei,ntsj:ntej)
      REAL(wp) :: zmsku,zmskv,zahu_w,zahv_w,zfw_kp1,zA11,zA22,zA13,zA23,zA31,zA32
      INTEGER :: ik,ikp1,iis
#define iso_lap
#define INN 0
#define pt_in(i,j,k,n,t) pt(i,j,k,n,t)
#define T2D(H) ntsi-(H):ntei+(H),ntsj-(H):ntej+(H)
#define DO_2D(L,R,B,T) DO jj=ntsj-(B),ntej+(T); DO ji=ntsi-(L),ntei+(R)
#define END_2D END DO; END DO
#define e3u(i,j,k,t) (e3u_3d(i,j,k)*(1._wp+r3u(i,j,t)*umask(i,j,k)))
#define e3v(i,j,k,t) (e3v_3d(i,j,k)*(1._wp+r3v(i,j,t)*vmask(i,j,k)))
#define e3t(i,j,k,t) (e3t_3d(i,j,k)*(1._wp+r3t(i,j,t)*tmask(i,j,k)))
#define e3w(i,j,k,t) (e3w_1d(k)*(1._wp+r3t(i,j,t)))
      DO jn=1,jpts; DO jk=1,jpk-1
#include <traldf_iso_scheme_unpatched.h90>
      END DO; END DO
#undef e3w
#undef e3t
#undef e3v
#undef e3u
#undef END_2D
#undef DO_2D
#undef T2D
#undef pt_in
#undef INN
#undef iso_lap
   END SUBROUTINE reference_kernel

   SUBROUTINE observed_kernel(pt)
      REAL(wp), INTENT(inout) :: pt(jpi,jpj,jpk,jpts,3)
      REAL(wp) :: zdit(1:jpi,1:jpj,0:1),zdjt(1:jpi,1:jpj,0:1),zdkt(1:jpi,1:jpj,0:1)
      REAL(wp) :: zfu(1:jpi,1:jpj),zfv(1:jpi,1:jpj),zfw(ntsi:ntei,ntsj:ntej)
      REAL(wp) :: zmsku,zmskv,zahu_w,zahv_w,zfw_kp1,zA11,zA22,zA13,zA23,zA31,zA32
      INTEGER :: ik,ikp1,iis
#define iso_lap
#define INN 0
#define pt_in(i,j,k,n,t) pt(i,j,k,n,t)
#define T2D(H) ntsi-(H):ntei+(H),ntsj-(H):ntej+(H)
#define DO_2D(L,R,B,T) DO jj=ntsj-(B),ntej+(T); DO ji=ntsi-(L),ntei+(R)
#define END_2D END DO; END DO
#define e3u(i,j,k,t) (e3u_3d(i,j,k)*(1._wp+r3u(i,j,t)*umask(i,j,k)))
#define e3v(i,j,k,t) (e3v_3d(i,j,k)*(1._wp+r3v(i,j,t)*vmask(i,j,k)))
#define e3t(i,j,k,t) (e3t_3d(i,j,k)*(1._wp+r3t(i,j,t)*tmask(i,j,k)))
#define e3w(i,j,k,t) (e3w_1d(k)*(1._wp+r3t(i,j,t)))
      DO jn=1,jpts; DO jk=1,jpk-1
#include <traldf_iso_scheme_patched.h90>
      END DO; END DO
#undef e3w
#undef e3t
#undef e3v
#undef e3u
#undef END_2D
#undef DO_2D
#undef T2D
#undef pt_in
#undef INN
#undef iso_lap
   END SUBROUTINE observed_kernel
END PROGRAM round333_traldf_loop_fixture
