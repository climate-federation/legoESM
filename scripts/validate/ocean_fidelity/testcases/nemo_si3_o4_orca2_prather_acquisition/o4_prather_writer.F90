MODULE o4_prather_writer
   !! Passive write-only boundary observer for SI3 rung O4.
   USE, INTRINSIC :: iso_fortran_env, ONLY : int32, real64
   USE par_kind, ONLY : wp
   USE par_ice, ONLY : jpl,nlay_i,nlay_s,rDt_ice
   USE dom_oce, ONLY : jpi,jpj,nn_hls,Nis0,Nie0,Njs0,Nje0,narea, &
      e1u,e2v,r1_e1u,r1_e2v,e2u,e1v,e1e2t,r1_e1e2t,tmask,umask,vmask
   USE ice
   USE sbc_oce, ONLY : sss_m,sst_m
   USE sbc_ice, ONLY : snwice_mass
   USE icedyn_adv_pra
   USE o4_prather_tap, ONLY : o4_zcfl_delayed, o4_icycle
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o4_capture_adv, o4_capture_stage

   CHARACTER(len=8), PARAMETER :: magic='O4PRA001'
   INTEGER(int32), PARAMETER :: version=1_int32
   INTEGER(int32), SAVE :: c_kt=-1, c_rank=-1, c_jpi=-1, c_jpj=-1
   INTEGER(int32), SAVE :: c_jpl=-1, c_nlay_i=-1, c_nlay_s=-1, c_nn_hls=-1
   INTEGER(int32), SAVE :: c_nis0=-1, c_nie0=-1, c_njs0=-1, c_nje0=-1
   REAL(real64), SAVE :: c_rdt=-1.0_real64, c_hi_max=-1.0_real64
   REAL(wp), SAVE :: c_zcfl_current=0.0_wp

CONTAINS

   SUBROUTINE o4_capture_adv(kind,kt)
      INTEGER,INTENT(in)::kind,kt
      IF(kind/=2 .AND. kind/=5) ERROR STOP 'O4 advection capture kind mismatch'
      IF(kind==2) THEN
         CALL o4_set_context(kt,narea-1,jpi,jpj,jpl,nlay_i,nlay_s,nn_hls, &
            Nis0,Nie0,Njs0,Nje0,rDt_ice,hi_max(jpl))
         c_zcfl_current=MAXVAL(ABS(u_ice(:,:))*rDt_ice*r1_e1u(:,:))
         c_zcfl_current=MAX(c_zcfl_current,MAXVAL(ABS(v_ice(:,:))*rDt_ice*r1_e2v(:,:)))
         CALL o4_grid
         CALL o4_state(2)
         CALL o4_moments(3)
      ELSE
         IF(c_kt/=kt .OR. o4_icycle<1) ERROR STOP 'O4 CFL capture missing before exit'
         CALL o4_cfl(c_zcfl_current,o4_zcfl_delayed,o4_icycle)
         CALL o4_state(5)
         CALL o4_moments(6)
      ENDIF
   END SUBROUTINE o4_capture_adv

   SUBROUTINE o4_capture_stage(kind,kt)
      INTEGER,INTENT(in)::kind,kt
      IF(kind<7 .OR. kind>9) ERROR STOP 'O4 stage capture kind mismatch'
      IF(c_kt/=kt) ERROR STOP 'O4 stage capture step mismatch'
      CALL o4_state(kind)
   END SUBROUTINE o4_capture_stage

   SUBROUTINE o4_set_context(kt,rank,jpi,jpj,jpl,nlay_i,nlay_s,nn_hls, &
                             nis0,nie0,njs0,nje0,rdt,hi_max_top)
      INTEGER, INTENT(in) :: kt,rank,jpi,jpj,jpl,nlay_i,nlay_s,nn_hls
      INTEGER, INTENT(in) :: nis0,nie0,njs0,nje0
      REAL(wp), INTENT(in) :: rdt,hi_max_top
      c_kt=INT(kt,int32); c_rank=INT(rank,int32)
      c_jpi=INT(jpi,int32); c_jpj=INT(jpj,int32); c_jpl=INT(jpl,int32)
      c_nlay_i=INT(nlay_i,int32); c_nlay_s=INT(nlay_s,int32)
      c_nn_hls=INT(nn_hls,int32); c_nis0=INT(nis0,int32); c_nie0=INT(nie0,int32)
      c_njs0=INT(njs0,int32); c_nje0=INT(nje0,int32)
      c_rdt=REAL(rdt,real64); c_hi_max=REAL(hi_max_top,real64)
      IF(c_nie0-c_nis0+1/=c_jpi-2*c_nn_hls .OR. &
         c_nje0-c_njs0+1/=c_jpj-2*c_nn_hls) ERROR STOP 'O4 context extent mismatch'
   END SUBROUTINE o4_set_context

   SUBROUTINE open_record(u,kind,nfields,icycle,zcfl_current,zcfl_delayed)
      INTEGER, INTENT(out) :: u
      INTEGER, INTENT(in) :: kind,nfields,icycle
      REAL(wp), INTENT(in) :: zcfl_current,zcfl_delayed
      INTEGER(int32), DIMENSION(16) :: meta
      REAL(real64), DIMENSION(4) :: scalars
      CHARACTER(len=256) :: name
      INTEGER :: ios
      IF(c_kt<1 .OR. c_jpi<1) ERROR STOP 'O4 context not initialized'
      WRITE(name,'("o4_prather_records/o4_kind",I2.2,"_step",I4.4,"_rank",I4.4,".bin")') &
         kind,c_kt,c_rank
      OPEN(NEWUNIT=u,FILE=TRIM(name),FORM='UNFORMATTED',ACCESS='STREAM', &
           STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) ERROR STOP 'O4 record open failed'
      meta=[INT(kind,int32),c_kt,c_rank,c_jpi,c_jpj,c_jpl,c_nlay_i,c_nlay_s, &
            c_nn_hls,c_nis0,c_nie0,c_njs0,c_nje0,INT(nfields,int32), &
            INT(icycle,int32),INT(MOD(c_kt-1,2),int32)]
      scalars=[c_rdt,c_hi_max,REAL(zcfl_current,real64),REAL(zcfl_delayed,real64)]
      WRITE(u) magic; WRITE(u) version; WRITE(u) meta; WRITE(u) scalars
   END SUBROUTINE open_record

   SUBROUTINE put_f2(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)/=c_jpi .OR. SIZE(a,2)/=c_jpj) ERROR STOP 'O4 F2 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_f2

   SUBROUTINE put_i2(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)/=c_nie0-c_nis0+1 .OR. SIZE(a,2)/=c_nje0-c_njs0+1) &
         ERROR STOP 'O4 I2 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_i2

   SUBROUTINE put_i3(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=c_jpi .OR. SIZE(a,2)/=c_jpj .OR. SIZE(a,3)/=c_jpl) &
         ERROR STOP 'O4 I3 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_i3

   SUBROUTINE put_ii4(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=c_jpi .OR. SIZE(a,2)/=c_jpj .OR. &
         SIZE(a,3)/=c_nlay_i .OR. SIZE(a,4)/=c_jpl) ERROR STOP 'O4 II4 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_ii4

   SUBROUTINE put_is4(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=c_jpi .OR. SIZE(a,2)/=c_jpj .OR. &
         SIZE(a,3)/=c_nlay_s .OR. SIZE(a,4)/=c_jpl) ERROR STOP 'O4 IS4 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_is4

   SUBROUTINE o4_grid
      INTEGER::u
      CALL open_record(u,1,15,0,0._wp,0._wp)
      CALL put_f2(u,e1u); CALL put_f2(u,e2v); CALL put_f2(u,r1_e1u); CALL put_f2(u,r1_e2v)
      CALL put_f2(u,e2u); CALL put_f2(u,e1v); CALL put_f2(u,e1e2t); CALL put_f2(u,r1_e1e2t)
      CALL put_f2(u,tmask(:,:,1)); CALL put_f2(u,umask(:,:,1)); CALL put_f2(u,vmask(:,:,1))
      CALL put_f2(u,u_ice); CALL put_f2(u,v_ice); CALL put_f2(u,sss_m); CALL put_f2(u,sst_m); CLOSE(u)
   END SUBROUTINE o4_grid

   SUBROUTINE o4_state(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u
      IF(kind/=2 .AND. kind/=5 .AND. kind/=7 .AND. kind/=8 .AND. kind/=9) ERROR STOP 'O4 state kind mismatch'
      CALL open_record(u,kind,34,0,0._wp,0._wp)
      CALL put_f2(u,ato_i)
      CALL put_i3(u,h_i); CALL put_i3(u,h_s); CALL put_i3(u,h_ip); CALL put_i3(u,h_il)
      CALL put_i3(u,v_i); CALL put_i3(u,v_s); CALL put_i3(u,sv_i); CALL put_i3(u,oa_i)
      CALL put_i3(u,a_i); CALL put_i3(u,a_ip); CALL put_i3(u,v_ip); CALL put_i3(u,v_il)
      CALL put_i3(u,t_su)
      CALL put_is4(u,e_s); CALL put_ii4(u,e_i); CALL put_ii4(u,szv_i)
      CALL put_ii4(u,sz_i); CALL put_ii4(u,t_i); CALL put_is4(u,t_s)
      CALL put_f2(u,at_i); CALL put_f2(u,vt_i); CALL put_f2(u,snwice_mass)
      CALL put_i2(u,wfx_res); CALL put_i2(u,hfx_res); CALL put_i2(u,sfx_res)
      CALL put_i2(u,diag_adv_mass); CALL put_i2(u,diag_adv_salt); CALL put_i2(u,diag_adv_heat)
      CALL put_i2(u,diag_trp_vi); CALL put_i2(u,diag_trp_vs); CALL put_i2(u,diag_trp_ei)
      CALL put_i2(u,diag_trp_es); CALL put_i2(u,diag_trp_sv)
      CLOSE(u)
   END SUBROUTINE o4_state

   SUBROUTINE o4_moments(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u
      IF(kind/=3 .AND. kind/=6) ERROR STOP 'O4 moment kind mismatch'
      CALL open_record(u,kind,55,0,0._wp,0._wp)
      CALL put_i3(u,sxice); CALL put_i3(u,syice); CALL put_i3(u,sxxice); CALL put_i3(u,syyice); CALL put_i3(u,sxyice)
      CALL put_i3(u,sxsn); CALL put_i3(u,sysn); CALL put_i3(u,sxxsn); CALL put_i3(u,syysn); CALL put_i3(u,sxysn)
      CALL put_i3(u,sxa); CALL put_i3(u,sya); CALL put_i3(u,sxxa); CALL put_i3(u,syya); CALL put_i3(u,sxya)
      CALL put_i3(u,sxage); CALL put_i3(u,syage); CALL put_i3(u,sxxage); CALL put_i3(u,syyage); CALL put_i3(u,sxyage)
      CALL put_i3(u,sxsal); CALL put_i3(u,sysal); CALL put_i3(u,sxxsal); CALL put_i3(u,syysal); CALL put_i3(u,sxysal)
      CALL put_i3(u,sxap); CALL put_i3(u,syap); CALL put_i3(u,sxxap); CALL put_i3(u,syyap); CALL put_i3(u,sxyap)
      CALL put_i3(u,sxvp); CALL put_i3(u,syvp); CALL put_i3(u,sxxvp); CALL put_i3(u,syyvp); CALL put_i3(u,sxyvp)
      CALL put_i3(u,sxvl); CALL put_i3(u,syvl); CALL put_i3(u,sxxvl); CALL put_i3(u,syyvl); CALL put_i3(u,sxyvl)
      CALL put_is4(u,sxc0); CALL put_is4(u,syc0); CALL put_is4(u,sxxc0); CALL put_is4(u,syyc0); CALL put_is4(u,sxyc0)
      CALL put_ii4(u,sxe); CALL put_ii4(u,sye); CALL put_ii4(u,sxxe); CALL put_ii4(u,syye); CALL put_ii4(u,sxye)
      CALL put_ii4(u,sxsi); CALL put_ii4(u,sysi); CALL put_ii4(u,sxxsi); CALL put_ii4(u,syysi); CALL put_ii4(u,sxysi)
      CLOSE(u)
   END SUBROUTINE o4_moments

   SUBROUTINE o4_cfl(zcfl_current,zcfl_delayed,icycle)
      REAL(wp),INTENT(in)::zcfl_current,zcfl_delayed
      INTEGER,INTENT(in)::icycle
      INTEGER::u
      CALL open_record(u,4,0,icycle,zcfl_current,zcfl_delayed); CLOSE(u)
   END SUBROUTINE o4_cfl
END MODULE o4_prather_writer
