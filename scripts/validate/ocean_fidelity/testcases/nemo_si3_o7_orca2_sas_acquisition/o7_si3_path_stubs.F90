! Minimal runtime used to execute the patched sbcblk/icealb/icesbc/icestp path.
MODULE par_oce
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=real64
   INTEGER :: jpi=12, jpj=10, jpk=2, nn_hls=2
   INTEGER :: ntsi=3, ntei=10, ntsj=3, ntej=8
   INTEGER :: Nis0=3, Nie0=10, Njs0=3, Nje0=8
END MODULE par_oce

MODULE empty_o7_modules
END MODULE empty_o7_modules

MODULE cyclone; END MODULE cyclone
MODULE fldread
   USE par_oce
   IMPLICIT NONE
   TYPE FLD_N
      CHARACTER(LEN=64) :: clname='NOT USED'
      REAL(wp) :: freqh=0._wp
      LOGICAL :: ln_tint=.FALSE.
   END TYPE
   TYPE FLD
      REAL(wp), ALLOCATABLE :: fnow(:,:,:), fdta(:,:,:,:)
      CHARACTER(LEN=64) :: clrootname='NOT USED'
      REAL(wp) :: freqh=0._wp, zsgn=1._wp
      LOGICAL :: ln_tint=.FALSE.
   END TYPE
CONTAINS
   SUBROUTINE fld_fill(sf,slf,dir,routine,description,namelist_name)
      TYPE(FLD),INTENT(out)::sf(:); TYPE(FLD_N),INTENT(in)::slf(:)
      CHARACTER(LEN=*),INTENT(in)::dir,routine,description,namelist_name
      sf%clrootname='O7_FIXTURE'; sf%freqh=0._wp; sf%ln_tint=.FALSE.
   END SUBROUTINE
   SUBROUTINE fld_read(kt,nn,sf); INTEGER,INTENT(in)::kt,nn; TYPE(FLD),INTENT(inout)::sf(:); END SUBROUTINE
END MODULE fldread
MODULE sbcdcy
CONTAINS
   FUNCTION sbc_dcy(a) RESULT(r)
      REAL(8),INTENT(in)::a(:,:); REAL(8)::r(SIZE(a,1),SIZE(a,2)); r=a
   END FUNCTION
END MODULE sbcdcy
MODULE sbcwave; REAL(8) :: cdn_wave=0.; END MODULE sbcwave
MODULE trc_oce; END MODULE trc_oce
MODULE sbcblk_algo_ncar
CONTAINS
   SUBROUTINE turb_ncar(zqt,zu,sst,tair,ssq,qair,wnd,cd,ch,ce,theta,qzu,uzu,nb_iter)
      REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:); INTEGER,INTENT(in)::nb_iter
      REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:),uzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair; uzu=wnd
   END SUBROUTINE
END MODULE sbcblk_algo_ncar
MODULE sbcblk_algo_coare3p0
CONTAINS
   SUBROUTINE turb_coare3p0(kt,zqt,zu,sst,tair,ssq,qair,wnd,cs,wl,cd,ch,ce,theta,qzu,uzu,nb_iter,Qsw,rad_lw,slp)
      INTEGER,INTENT(in)::kt,nb_iter; REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:),Qsw(:,:),rad_lw(:,:),slp(:,:)
      LOGICAL,INTENT(in)::cs,wl; REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:),uzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair; uzu=wnd
   END SUBROUTINE
END MODULE sbcblk_algo_coare3p0
MODULE sbcblk_algo_coare3p6
CONTAINS
   SUBROUTINE turb_coare3p6(kt,zqt,zu,sst,tair,ssq,qair,wnd,cs,wl,cd,ch,ce,theta,qzu,uzu,nb_iter,Qsw,rad_lw,slp)
      INTEGER,INTENT(in)::kt,nb_iter; REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:),Qsw(:,:),rad_lw(:,:),slp(:,:)
      LOGICAL,INTENT(in)::cs,wl; REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:),uzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair; uzu=wnd
   END SUBROUTINE
END MODULE sbcblk_algo_coare3p6
MODULE sbcblk_algo_ecmwf
CONTAINS
   SUBROUTINE turb_ecmwf(kt,zqt,zu,sst,tair,ssq,qair,wnd,cs,wl,cd,ch,ce,theta,qzu,uzu,nb_iter,Qsw,rad_lw,slp)
      INTEGER,INTENT(in)::kt,nb_iter; REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:),Qsw(:,:),rad_lw(:,:),slp(:,:)
      LOGICAL,INTENT(in)::cs,wl; REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:),uzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair; uzu=wnd
   END SUBROUTINE
END MODULE sbcblk_algo_ecmwf
MODULE sbcblk_algo_andreas
CONTAINS
   SUBROUTINE turb_andreas(zqt,zu,sst,tair,ssq,qair,wnd,cd,ch,ce,theta,qzu,uzu,nb_iter)
      REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:); INTEGER,INTENT(in)::nb_iter
      REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:),uzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair; uzu=wnd
   END SUBROUTINE
END MODULE sbcblk_algo_andreas
MODULE sbcblk_algo_mfs
CONTAINS
   SUBROUTINE turb_mfs(zqt,zu,tsk,tair,humi,ssq,wnd,wi,wj,u,v,cloud,qsr,sst,cd,ch,ce,theta,qzu,uzu,rspeed,rhoa,qlwn,nb_iter,slp)
      REAL(8),INTENT(in)::zqt,zu,tsk(:,:),tair(:,:),humi(:,:),ssq(:,:),wnd(:,:),wi(:,:),wj(:,:),u(:,:),v(:,:),cloud(:,:),qsr(:,:),sst(:,:),slp(:,:)
      INTEGER,INTENT(in)::nb_iter; REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:),uzu(:,:),rspeed(:,:),rhoa(:,:),qlwn(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=humi; uzu=wnd; rspeed=wnd; rhoa=1.2; qlwn=0.
   END SUBROUTINE
END MODULE sbcblk_algo_mfs
MODULE sbcblk_algo_ice_an05
CONTAINS
   SUBROUTINE turb_ice_an05(zqt,zu,sst,tair,ssq,qair,wnd,cd,ch,ce,theta,qzu)
      REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:)
      REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair
   END SUBROUTINE
END MODULE sbcblk_algo_ice_an05
MODULE sbcblk_algo_ice_lu12
CONTAINS
   SUBROUTINE turb_ice_lu12(zqt,zu,sst,tair,ssq,qair,wnd,fr,cd,ch,ce,theta,qzu)
      REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:),fr(:,:)
      REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair
   END SUBROUTINE
END MODULE sbcblk_algo_ice_lu12
MODULE sbcblk_algo_ice_lg15
CONTAINS
   SUBROUTINE turb_ice_lg15(zqt,zu,sst,tair,ssq,qair,wnd,fr,cd,ch,ce,theta,qzu)
      REAL(8),INTENT(in)::zqt,zu,sst(:,:),tair(:,:),ssq(:,:),qair(:,:),wnd(:,:),fr(:,:)
      REAL(8),INTENT(out)::cd(:,:),ch(:,:),ce(:,:),theta(:,:),qzu(:,:)
      cd=1.4e-3; ch=1.2e-3; ce=1.2e-3; theta=tair; qzu=qair
   END SUBROUTINE
END MODULE sbcblk_algo_ice_lg15
MODULE sbccpl
CONTAINS
   SUBROUTINE sbc_cpl_ice_tau(ut,vt)
      REAL(8),INTENT(out)::ut(:,:),vt(:,:); ut=0.; vt=0.
   END SUBROUTINE
   SUBROUTINE sbc_cpl_ice_flx(kt,picefr,palbi,psst,pist,phs,phi)
      INTEGER,INTENT(in)::kt
      REAL(8),INTENT(in),OPTIONAL::picefr(:,:),palbi(:,:,:),psst(:,:),pist(:,:,:),phs(:,:,:),phi(:,:,:)
   END SUBROUTINE
END MODULE sbccpl
MODULE bdyice
CONTAINS
   SUBROUTINE bdy_ice(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
END MODULE bdyice
MODULE eosbn2
CONTAINS
   SUBROUTINE eos_fzp(s,t,kbnd,pdep)
      REAL(8),INTENT(in)::s(:,:); REAL(8),INTENT(out)::t(:,:); INTEGER,INTENT(in)::kbnd
      REAL(8),INTENT(in),OPTIONAL::pdep(:,:); t=-1.8
   END SUBROUTINE
END MODULE eosbn2
MODULE icedyn
CONTAINS
   SUBROUTINE ice_dyn(kt,kmm); INTEGER,INTENT(in)::kt,kmm; END SUBROUTINE
   SUBROUTINE ice_dyn_init; END SUBROUTINE
END MODULE icedyn
MODULE icethd
CONTAINS
   SUBROUTINE ice_thd(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
   SUBROUTINE ice_thd_init; END SUBROUTINE
END MODULE icethd
MODULE icedia
   LOGICAL :: ln_icediachk=.FALSE.
CONTAINS
   SUBROUTINE diag_set0; END SUBROUTINE
   SUBROUTINE diag_trends(k); INTEGER,INTENT(in)::k; END SUBROUTINE
   SUBROUTINE ice_dia(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
   SUBROUTINE ice_dia_init; END SUBROUTINE
   SUBROUTINE ice_drift_wri(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
   SUBROUTINE ice_drift_init; END SUBROUTINE
END MODULE icedia
MODULE icewri
CONTAINS
   SUBROUTINE ice_wri(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
END MODULE icewri
MODULE icerst
   LOGICAL :: lrst_ice=.FALSE.
CONTAINS
   SUBROUTINE ice_rst_opn(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
   SUBROUTINE ice_rst_write(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
   SUBROUTINE ice_rst_read(kbb,kmm,kaa); INTEGER,INTENT(in)::kbb,kmm,kaa; END SUBROUTINE
END MODULE icerst
MODULE icectl
   LOGICAL :: ln_icectl=.FALSE.
CONTAINS
   SUBROUTINE ice_ctl(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
END MODULE icectl
MODULE iceistate
CONTAINS
   SUBROUTINE ice_istate_init; END SUBROUTINE
   SUBROUTINE ice_istate(kt,kbb,kmm,kaa); INTEGER,INTENT(in)::kt,kbb,kmm,kaa; END SUBROUTINE
END MODULE iceistate
MODULE iceitd
CONTAINS
   SUBROUTINE ice_itd_init; END SUBROUTINE
END MODULE iceitd
MODULE icefrm
CONTAINS
   SUBROUTINE ice_frm(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
END MODULE icefrm
MODULE prtctl
   TYPE control_type
      LOGICAL :: l_prtctl=.FALSE.
   END TYPE
   TYPE(control_type) :: sn_cfctl
CONTAINS
   SUBROUTINE prt_ctl(tab2d_1,tab3d_1,clinfo1,mask1,tab2d_2,tab3d_2,clinfo2,mask2,kdim)
      REAL(8),INTENT(in),OPTIONAL::tab2d_1(:,:),tab3d_1(:,:,:),tab2d_2(:,:),tab3d_2(:,:,:)
      REAL(8),INTENT(in),OPTIONAL::mask1(:,:,:),mask2(:,:,:)
      CHARACTER(LEN=*),INTENT(in),OPTIONAL::clinfo1,clinfo2
      INTEGER,INTENT(in),OPTIONAL::kdim
   END SUBROUTINE
END MODULE prtctl

MODULE timing
   LOGICAL :: ln_timing=.FALSE.
CONTAINS
   SUBROUTINE timing_start(name); CHARACTER(LEN=*),INTENT(in)::name; END SUBROUTINE
   SUBROUTINE timing_stop(name); CHARACTER(LEN=*),INTENT(in)::name; END SUBROUTINE
END MODULE timing

MODULE bdy_oce
   LOGICAL :: ln_bdy=.FALSE.
END MODULE bdy_oce

MODULE lib_mpp
   INTEGER :: narea=1, jpni=1, jpnj=1
CONTAINS
   SUBROUTINE ctl_stop(a,b,c,d,e,f)
      CHARACTER(LEN=*),INTENT(in)::a
      CHARACTER(LEN=*),INTENT(in),OPTIONAL::b,c,d,e,f
      ERROR STOP a
   END SUBROUTINE
   SUBROUTINE ctl_warn(a,b); CHARACTER(LEN=*),INTENT(in)::a; CHARACTER(LEN=*),INTENT(in),OPTIONAL::b; END SUBROUTINE
   SUBROUTINE ctl_nam(ios,name,required); INTEGER,INTENT(in)::ios; CHARACTER(LEN=*),INTENT(in)::name; LOGICAL,INTENT(in)::required; END SUBROUTINE
   SUBROUTINE mpp_sum(name,value); CHARACTER(LEN=*),INTENT(in)::name; INTEGER,INTENT(inout)::value; END SUBROUTINE
END MODULE lib_mpp

MODULE in_out_manager
   IMPLICIT NONE
   LOGICAL :: lwp=.FALSE., lwm=.FALSE.
   INTEGER :: nit000=1, numout=6, numoni=6, numond=6
   CHARACTER(LEN=4096) :: numnam_ref='', numnam_cfg='', numnam_ice='', numnam_ice_ref='', numnam_ice_cfg=''
   CHARACTER(LEN=256) :: ctmp1=''
END MODULE in_out_manager

MODULE iom
   IMPLICIT NONE
   INTERFACE iom_put
      MODULE PROCEDURE iom_put_2d, iom_put_3d
   END INTERFACE
CONTAINS
   LOGICAL FUNCTION iom_use(name); CHARACTER(LEN=*),INTENT(in)::name; iom_use=.FALSE.; END FUNCTION
   SUBROUTINE iom_put_2d(name,a); CHARACTER(LEN=*),INTENT(in)::name; REAL(8),INTENT(in)::a(:,:); END SUBROUTINE
   SUBROUTINE iom_put_3d(name,a); CHARACTER(LEN=*),INTENT(in)::name; REAL(8),INTENT(in)::a(:,:,:); END SUBROUTINE
END MODULE iom

MODULE lbclnk
   IMPLICIT NONE
   INTERFACE lbc_lnk
      MODULE PROCEDURE lbc_lnk_2d
   END INTERFACE
CONTAINS
   SUBROUTINE lbc_lnk_2d(name,a,nata,sgna,b,natb,sgnb)
      CHARACTER(LEN=*),INTENT(in)::name,nata,natb
      REAL(8),INTENT(in)::sgna,sgnb
      REAL(8),INTENT(inout)::a(:,:),b(:,:)
      a(:,1:2)=0.; b(:,1:2)=0.; a(:,9:10)=sgna*a(:,7:8); b(:,9:10)=sgnb*b(:,7:8)
   END SUBROUTINE
END MODULE lbclnk

MODULE lib_fortran
   INTERFACE glob_2dsum
      MODULE PROCEDURE glob_2dsum_named
   END INTERFACE
CONTAINS
   REAL(8) FUNCTION glob_2dsum_named(name,a)
      CHARACTER(LEN=*),INTENT(in)::name; REAL(8),INTENT(in)::a(:,:); glob_2dsum_named=SUM(a)
   END FUNCTION
END MODULE lib_fortran

MODULE phycst
   USE par_oce
   IMPLICIT NONE
   REAL(wp), PARAMETER :: rt0=273.15_wp, ralb_oce=0.06_wp, rho0=1026._wp
   REAL(wp), PARAMETER :: rcp=3991.86795711963_wp, r1_rho0=1._wp/rho0
   REAL(wp), PARAMETER :: rLfus=3.34e5_wp, rLsub=2.834e6_wp, r1_rLsub=1._wp/rLsub
   REAL(wp), PARAMETER :: stefan=5.670374419e-8_wp, emiss_i=0.97_wp, rcp_air=1004.64_wp
   REAL(wp), PARAMETER :: epsi10=1.e-10_wp, epsi10_unused=epsi10
   REAL(wp), PARAMETER :: albo=0.06_wp, rcpi=2100._wp, rhos=330._wp
   REAL(wp), PARAMETER :: rdct_qsat_salt=0.98_wp
END MODULE phycst

MODULE par_ice
   USE par_oce
   IMPLICIT NONE
   INTEGER, PARAMETER :: jpl=5
   LOGICAL :: ln_icedyn=.TRUE., ln_icethd=.FALSE., ln_virtual_itd=.FALSE.
   LOGICAL :: ln_cndflx=.FALSE., ln_cndemulate=.FALSE., ln_icedH=.FALSE., ln_icedO=.FALSE., ln_leadhfx=.FALSE.
   LOGICAL :: ln_pnd_alb=.TRUE., ln_c1d=.FALSE.
   INTEGER :: nn_qtrice=0, nn_ice=2, nn_snwfra=0
   REAL(wp) :: rDt_ice=5400._wp, r1_Dt_ice=1._wp/5400._wp
   REAL(wp) :: rcnd_s=0.31_wp, rcnd_i=2.0_wp, rn_Cd_io=5.e-3_wp, rn_snwblow=1._wp
END MODULE par_ice

MODULE dom_oce
   USE par_oce
   IMPLICIT NONE
   REAL(wp), ALLOCATABLE :: smask0(:,:), tmask(:,:,:), rhoa(:,:)
CONTAINS
   SUBROUTINE o7_init_dom
      IF(ALLOCATED(smask0)) DEALLOCATE(smask0,tmask,rhoa)
      ALLOCATE(smask0(Nis0:Nie0,Njs0:Nje0),tmask(jpi,jpj,jpk),rhoa(Nis0:Nie0,Njs0:Nje0))
      smask0=1._wp; tmask=1._wp; rhoa=1.2_wp
   END SUBROUTINE
END MODULE dom_oce

MODULE oce
   USE par_oce
   IMPLICIT NONE
   REAL(wp), ALLOCATABLE :: uu(:,:,:,:), vv(:,:,:,:)
CONTAINS
   SUBROUTINE o7_init_oce
      IF(ALLOCATED(uu)) DEALLOCATE(uu,vv)
      ALLOCATE(uu(jpi,jpj,jpk,2),vv(jpi,jpj,jpk,2)); uu=0._wp; vv=0._wp
   END SUBROUTINE
END MODULE oce

MODULE ice1D
END MODULE ice1D

MODULE ice
   USE par_oce
   USE par_ice
   USE dom_oce
   USE phycst
   IMPLICIT NONE
   REAL(wp), ALLOCATABLE :: u_ice(:,:), v_ice(:,:), a_i(:,:,:), a_i_b(:,:,:), at_i_b(:,:)
   REAL(wp), ALLOCATABLE :: t_su(:,:,:), h_i(:,:,:), h_s(:,:,:), a_ip_eff(:,:,:)
   REAL(wp), ALLOCATABLE :: h_ip(:,:,:), h_il(:,:,:), at_i(:,:), vt_i(:,:), drag_ia(:,:)
   REAL(wp), ALLOCATABLE :: u_oce(:,:), v_oce(:,:), hfx_err_dif(:,:), drag_io(:,:)
   REAL(wp), ALLOCATABLE :: qsb_ice_bot(:,:), fhld(:,:), qlead(:,:)
   INTEGER :: kt_ice=0
CONTAINS
   SUBROUTINE o7_init_ice(rank)
      INTEGER,INTENT(in)::rank; INTEGER::i,j,l
      IF(ALLOCATED(u_ice)) DEALLOCATE(u_ice,v_ice,a_i,a_i_b,at_i_b,t_su,h_i,h_s,a_ip_eff, &
         h_ip,h_il,at_i,vt_i,drag_ia,u_oce,v_oce,hfx_err_dif,drag_io,qsb_ice_bot,fhld,qlead)
      ALLOCATE(u_ice(jpi,jpj),v_ice(jpi,jpj),a_i(Nis0:Nie0,Njs0:Nje0,jpl),a_i_b(Nis0:Nie0,Njs0:Nje0,jpl),at_i_b(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(t_su(jpi,jpj,jpl),h_i(jpi,jpj,jpl),h_s(jpi,jpj,jpl),a_ip_eff(Nis0:Nie0,Njs0:Nje0,jpl))
      ALLOCATE(h_ip(jpi,jpj,jpl),h_il(jpi,jpj,jpl),at_i(Nis0:Nie0,Njs0:Nje0),vt_i(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(drag_ia(Nis0:Nie0,Njs0:Nje0),drag_io(Nis0:Nie0,Njs0:Nje0),u_oce(jpi,jpj),v_oce(jpi,jpj),hfx_err_dif(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(qsb_ice_bot(Nis0:Nie0,Njs0:Nje0),fhld(Nis0:Nie0,Njs0:Nje0),qlead(Nis0:Nie0,Njs0:Nje0))
      DO j=1,jpj; DO i=1,jpi
         u_ice(i,j)=rank+0.01_wp*i; v_ice(i,j)=-rank-0.01_wp*j
         u_oce(i,j)=0._wp; v_oce(i,j)=0._wp
         DO l=1,jpl
            t_su(i,j,l)=270._wp+0.01_wp*(i+j+l+100*rank)
            h_i(i,j,l)=0.5_wp; h_s(i,j,l)=0.05_wp; h_ip(i,j,l)=0.02_wp; h_il(i,j,l)=0.005_wp
         END DO
      END DO; END DO
      a_i_b=-0._wp; a_i=0.12_wp; at_i_b=0.6_wp; a_ip_eff=0.1_wp; at_i=0.6_wp; vt_i=1._wp
      drag_ia=1.4e-3_wp; drag_io=5.e-3_wp; hfx_err_dif=0._wp; qsb_ice_bot=0._wp; fhld=0._wp; qlead=0._wp
      h_s(Nis0,Njs0,1)=-0._wp
   END SUBROUTINE
   SUBROUTINE store_fields
   END SUBROUTINE
END MODULE ice

MODULE sbc_oce
   USE par_oce
   IMPLICIT NONE
   INTEGER, PARAMETER :: jp_usr=1, jp_blk=2, jp_abl=3, jp_purecpl=4
   INTEGER :: nn_fsbc=1, jpka=1
   LOGICAL :: ln_blk=.TRUE., ln_abl=.FALSE., ln_mixcpl=.FALSE., ln_dm2dc=.FALSE., ln_cpl=.FALSE.
   REAL(wp) :: rn_Dt=5400._wp, pp_cldf=0.5_wp
   REAL(wp), ALLOCATABLE :: e1e2t(:,:), ssu_m(:,:), ssv_m(:,:), e3t_m(:,:), sss_m(:,:), sst_m(:,:)
   REAL(wp), ALLOCATABLE :: emp(:,:), qns_oce(:,:), qsr_oce(:,:), qemp_oce(:,:), frq_m(:,:), t_bo(:,:)
   REAL(wp), ALLOCATABLE :: utau(:,:), vtau(:,:), cloud_fra(:,:), qsr(:,:), qns(:,:), taum(:,:), tsk_m(:,:)
   REAL(wp), ALLOCATABLE :: wndm(:,:), xcplmask(:,:,:)
   REAL(wp) :: ght_abl(2)=[0._wp,10._wp]
CONTAINS
   SUBROUTINE o7_init_sbc_oce(rank)
      INTEGER,INTENT(in)::rank
      IF(ALLOCATED(e1e2t)) DEALLOCATE(e1e2t,ssu_m,ssv_m,e3t_m,sss_m,sst_m,emp,qns_oce,qsr_oce, &
         qemp_oce,frq_m,t_bo,utau,vtau,cloud_fra,qsr,qns,taum,tsk_m,wndm,xcplmask)
      ALLOCATE(e1e2t(jpi,jpj),ssu_m(jpi,jpj),ssv_m(jpi,jpj),e3t_m(jpi,jpj),sss_m(jpi,jpj),sst_m(jpi,jpj))
      ALLOCATE(emp(Nis0:Nie0,Njs0:Nje0),qns_oce(Nis0:Nie0,Njs0:Nje0),qsr_oce(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(qemp_oce(Nis0:Nie0,Njs0:Nje0),frq_m(Nis0:Nie0,Njs0:Nje0),t_bo(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(utau(jpi,jpj),vtau(jpi,jpj),cloud_fra(Nis0:Nie0,Njs0:Nje0),qsr(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(qns(Nis0:Nie0,Njs0:Nje0),taum(Nis0:Nie0,Njs0:Nje0),tsk_m(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(wndm(jpi,jpj))
      ALLOCATE(xcplmask(jpi,jpj,0:0))
      e1e2t=1._wp; ssu_m=0._wp; ssv_m=0._wp; e3t_m=10._wp; sss_m=34._wp; sst_m=-1.5_wp
      emp=0._wp; qns_oce=-30._wp; qsr_oce=100._wp; qemp_oce=0._wp; frq_m=0.2_wp; t_bo=271.6_wp
      utau=-0._wp; vtau=-0._wp; cloud_fra=0.5_wp; qsr=100._wp; qns=-30._wp; taum=0._wp; tsk_m=271._wp; wndm=0._wp; xcplmask=1._wp
   END SUBROUTINE
END MODULE sbc_oce

MODULE sbc_ice
   USE par_oce
   USE par_ice
   IMPLICIT NONE
   REAL(wp), ALLOCATABLE :: fr_i(:,:),tm_su(:,:),theta_air_zt(:,:),q_air_zt(:,:),utau_ice(:,:),vtau_ice(:,:)
   REAL(wp), ALLOCATABLE :: wndm_ice(:,:),alb_ice(:,:,:),rn_amax_2d(:,:),qsr_ice(:,:,:),qns_ice(:,:,:)
   REAL(wp), ALLOCATABLE :: dqns_ice(:,:,:),qla_ice(:,:,:),dqla_ice(:,:,:),evap_ice(:,:,:),devap_ice(:,:,:)
   REAL(wp), ALLOCATABLE :: qevap_ice(:,:,:),qtr_ice_top(:,:,:),qcn_ice(:,:,:),qml_ice(:,:,:),tprecip(:,:),sprecip(:,:),emp_oce(:,:)
   REAL(wp), ALLOCATABLE :: emp_ice(:,:),emp_tot(:,:),qemp_ice(:,:),qns_tot(:,:),qsr_tot(:,:),qprec_ice(:,:)
CONTAINS
   SUBROUTINE o7_init_sbc_ice(rank)
      INTEGER,INTENT(in)::rank
      IF(ALLOCATED(fr_i)) DEALLOCATE(fr_i,tm_su,theta_air_zt,q_air_zt,utau_ice,vtau_ice,wndm_ice, &
         alb_ice,rn_amax_2d,qsr_ice,qns_ice,dqns_ice,qla_ice,dqla_ice,evap_ice,devap_ice,qevap_ice, &
         qtr_ice_top,qcn_ice,qml_ice,tprecip,sprecip,emp_oce,emp_ice,emp_tot,qemp_ice,qns_tot,qsr_tot,qprec_ice)
      ALLOCATE(fr_i(Nis0:Nie0,Njs0:Nje0),tm_su(Nis0:Nie0,Njs0:Nje0),theta_air_zt(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(q_air_zt(Nis0:Nie0,Njs0:Nje0),utau_ice(jpi,jpj),vtau_ice(jpi,jpj),wndm_ice(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(alb_ice(Nis0:Nie0,Njs0:Nje0,jpl),rn_amax_2d(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(qsr_ice(Nis0:Nie0,Njs0:Nje0,jpl),qns_ice(Nis0:Nie0,Njs0:Nje0,jpl),dqns_ice(Nis0:Nie0,Njs0:Nje0,jpl))
      ALLOCATE(qla_ice(Nis0:Nie0,Njs0:Nje0,jpl),dqla_ice(Nis0:Nie0,Njs0:Nje0,jpl),evap_ice(Nis0:Nie0,Njs0:Nje0,jpl))
      ALLOCATE(devap_ice(Nis0:Nie0,Njs0:Nje0,jpl),qevap_ice(Nis0:Nie0,Njs0:Nje0,jpl),qtr_ice_top(Nis0:Nie0,Njs0:Nje0,jpl))
      ALLOCATE(qcn_ice(Nis0:Nie0,Njs0:Nje0,jpl),qml_ice(Nis0:Nie0,Njs0:Nje0,jpl))
      ALLOCATE(tprecip(Nis0:Nie0,Njs0:Nje0),sprecip(Nis0:Nie0,Njs0:Nje0),emp_oce(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(emp_ice(Nis0:Nie0,Njs0:Nje0),emp_tot(Nis0:Nie0,Njs0:Nje0),qemp_ice(Nis0:Nie0,Njs0:Nje0))
      ALLOCATE(qns_tot(Nis0:Nie0,Njs0:Nje0),qsr_tot(Nis0:Nie0,Njs0:Nje0),qprec_ice(Nis0:Nie0,Njs0:Nje0))
      fr_i=0.6_wp; tm_su=268._wp+rank; theta_air_zt=270._wp+rank; q_air_zt=0.002_wp
      utau_ice=-0._wp; vtau_ice=-0._wp; wndm_ice=0._wp; alb_ice=0._wp; rn_amax_2d=0.95_wp
      qsr_ice=0._wp; qns_ice=0._wp; dqns_ice=0._wp; qla_ice=0._wp; dqla_ice=0._wp
      evap_ice=0._wp; devap_ice=0._wp; qevap_ice=0._wp; qtr_ice_top=0._wp; qcn_ice=0._wp; qml_ice=0._wp
      tprecip=0._wp; sprecip=0._wp; emp_oce=0._wp; emp_ice=0._wp; emp_tot=0._wp
      qemp_ice=0._wp; qns_tot=0._wp; qsr_tot=0._wp; qprec_ice=0._wp
   END SUBROUTINE
END MODULE sbc_ice

MODULE sbc_phy
   USE par_oce
   INTERFACE theta_exner
      MODULE PROCEDURE theta_exner_2d, theta_exner_scalar
   END INTERFACE
   INTERFACE q_sat
      MODULE PROCEDURE q_sat_2d, q_sat_scalar
   END INTERFACE
   INTERFACE pres_temp
      MODULE PROCEDURE pres_temp_2d, pres_temp_scalar
   END INTERFACE
CONTAINS
   FUNCTION theta_exner_2d(t,p) RESULT(r)
      REAL(wp),INTENT(in)::t(:,:),p(:,:); REAL(wp)::r(SIZE(t,1),SIZE(t,2)); r=t
   END FUNCTION
   REAL(wp) FUNCTION theta_exner_scalar(t,p); REAL(wp),INTENT(in)::t,p; theta_exner_scalar=t; END FUNCTION
   FUNCTION q_sat_2d(t,p,l_ice) RESULT(r)
      REAL(wp),INTENT(in)::t(:,:),p(:,:); LOGICAL,INTENT(in),OPTIONAL::l_ice
      REAL(wp)::r(SIZE(t,1),SIZE(t,2)); r=0.003_wp
   END FUNCTION
   REAL(wp) FUNCTION q_sat_scalar(t,p,l_ice)
      REAL(wp),INTENT(in)::t,p; LOGICAL,INTENT(in),OPTIONAL::l_ice; q_sat_scalar=0.003_wp
   END FUNCTION
   REAL(wp) FUNCTION dq_sat_dt_ice(t,p); REAL(wp),INTENT(in)::t,p; dq_sat_dt_ice=1.e-4_wp; END FUNCTION
   FUNCTION q_air_rh(rh,t,p) RESULT(r)
      REAL(wp),INTENT(in)::rh(:,:),t(:,:),p(:,:); REAL(wp)::r(SIZE(rh,1),SIZE(rh,2)); r=rh*0.003_wp
   END FUNCTION
   FUNCTION pres_temp_2d(q,p,z,pta,ptpot) RESULT(r)
      REAL(wp),INTENT(in)::q(:,:),p(:,:),z
      REAL(wp),INTENT(in),OPTIONAL::pta(:,:),ptpot(:,:)
      REAL(wp)::r(SIZE(q,1),SIZE(q,2)); r=p
   END FUNCTION
   REAL(wp) FUNCTION pres_temp_scalar(q,p,z,pta,ptpot)
      REAL(wp),INTENT(in)::q,p,z
      REAL(wp),INTENT(in),OPTIONAL::pta,ptpot
      pres_temp_scalar=p
   END FUNCTION
   REAL(wp) FUNCTION rho_air(t,q,p); REAL(wp),INTENT(in)::t,q,p; rho_air=1.2_wp; END FUNCTION
   FUNCTION qlw_net(q,t) RESULT(r)
      REAL(wp),INTENT(in)::q(:,:),t(:,:); REAL(wp)::r(SIZE(q,1),SIZE(q,2)); r=q
   END FUNCTION
   SUBROUTINE bulk_formula(zu,sst,ssq,theta,qzu,cd,ch,ce,wnd,uzu,slp,rhoa,taum,sen,lat,pEvap,pfact_evap)
      REAL(wp),INTENT(in)::zu,sst(:,:),ssq(:,:),theta(:,:),qzu(:,:),cd(:,:),ch(:,:),ce(:,:),wnd(:,:),uzu(:,:),slp(:,:),rhoa(:,:)
      REAL(wp),INTENT(out)::taum(:,:),sen(:,:),lat(:,:),pEvap(:,:); REAL(wp),INTENT(in)::pfact_evap
      taum=0._wp; sen=0._wp; lat=0._wp; pEvap=0._wp
   END SUBROUTINE
END MODULE sbc_phy

MODULE icevar
   USE par_oce
   USE par_ice
CONTAINS
   SUBROUTINE ice_var_snwfra(ph_snw,fraction)
      REAL(wp),INTENT(in)::ph_snw(jpi,jpj,jpl); REAL(wp),INTENT(out)::fraction(jpi,jpj,jpl)
      fraction=MIN(0.5_wp,ph_snw)
   END SUBROUTINE
   SUBROUTINE ice_var_snwblow(at,zsnw); REAL(wp),INTENT(in)::at(:,:); REAL(wp),INTENT(out)::zsnw(:,:); zsnw=0._wp; END SUBROUTINE
   SUBROUTINE ice_var_glo2eqv(k); INTEGER,INTENT(in)::k; END SUBROUTINE
   SUBROUTINE ice_var_agg(k); INTEGER,INTENT(in)::k; END SUBROUTINE
END MODULE icevar

MODULE usrdef_sbc
   USE par_oce
CONTAINS
   SUBROUTINE usrdef_sbc_ice_tau(kt); INTEGER,INTENT(in)::kt; END SUBROUTINE
   SUBROUTINE usrdef_sbc_ice_flx(kt,hs,hi); INTEGER,INTENT(in)::kt; REAL(wp),INTENT(in)::hs(:,:,:),hi(:,:,:); END SUBROUTINE
END MODULE usrdef_sbc
