MODULE par_kind
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   INTEGER, PARAMETER :: wp=real64
END MODULE par_kind

MODULE par_ice
   USE par_kind, ONLY: wp
   INTEGER,PARAMETER :: jpi=12,jpj=10,jpl=2,nlay_i=2,nlay_s=1,nn_hls=2
   INTEGER,PARAMETER :: Nis0=3,Nie0=10,Njs0=3,Nje0=8,ntsi=3,ntei=10,ntsj=3,ntej=8
   INTEGER :: nn_icesal=4
   REAL(wp) :: rDt_ice=10800._wp,r1_Dt_ice=1._wp/10800._wp
   LOGICAL :: ln_pnd_LEV=.TRUE.,ln_pnd_TOPO=.FALSE.
END MODULE par_ice

MODULE dom_oce
   USE par_kind, ONLY: wp
   USE par_ice, ONLY: jpi,jpj,nn_hls,Nis0,Nie0,Njs0,Nje0
   INTEGER :: narea=1
   REAL(wp) :: e1u(jpi,jpj),e2v(jpi,jpj),r1_e1u(jpi,jpj),r1_e2v(jpi,jpj)
   REAL(wp) :: e2u(jpi,jpj),e1v(jpi,jpj),e1e2t(jpi,jpj),r1_e1e2t(jpi,jpj)
   REAL(wp) :: tmask(jpi,jpj,1),umask(jpi,jpj,1),vmask(jpi,jpj,1)
END MODULE dom_oce

MODULE phycst
   USE par_kind, ONLY: wp
   REAL(wp), PARAMETER :: rhoi=917._wp,rhos=330._wp,rhow=1026._wp,rho0=1026._wp
   REAL(wp), PARAMETER :: r1_rhos=1._wp/rhos,epsi10=1.e-10_wp,epsi20=1.e-20_wp
END MODULE phycst

MODULE ice
   USE par_kind, ONLY: wp
   USE par_ice, ONLY: jpi,jpj,jpl,nlay_i,nlay_s,Nis0,Nie0,Njs0,Nje0
   USE dom_oce
   REAL(wp) :: wfx_res(Nis0:Nie0,Njs0:Nje0),hfx_res(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: sfx_res(Nis0:Nie0,Njs0:Nje0),diag_adv_mass(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: diag_adv_salt(Nis0:Nie0,Njs0:Nje0),diag_adv_heat(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: diag_trp_vi(Nis0:Nie0,Njs0:Nje0),diag_trp_vs(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: diag_trp_ei(Nis0:Nie0,Njs0:Nje0),diag_trp_es(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: diag_trp_sv(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: u_ice(jpi,jpj),v_ice(jpi,jpj),ato_i(jpi,jpj),at_i(jpi,jpj),vt_i(jpi,jpj)
   REAL(wp) :: hi_max(jpl)=[99._wp,99._wp]
   REAL(wp) :: h_i(jpi,jpj,jpl),h_s(jpi,jpj,jpl),h_ip(jpi,jpj,jpl),h_il(jpi,jpj,jpl)
   REAL(wp) :: v_i(jpi,jpj,jpl),v_s(jpi,jpj,jpl),sv_i(jpi,jpj,jpl),oa_i(jpi,jpj,jpl)
   REAL(wp) :: a_i(jpi,jpj,jpl),a_ip(jpi,jpj,jpl),v_ip(jpi,jpj,jpl),v_il(jpi,jpj,jpl)
   REAL(wp) :: t_su(jpi,jpj,jpl),e_s(jpi,jpj,nlay_s,jpl),e_i(jpi,jpj,nlay_i,jpl)
   REAL(wp) :: szv_i(jpi,jpj,nlay_i,jpl),sz_i(jpi,jpj,nlay_i,jpl),t_i(jpi,jpj,nlay_i,jpl)
   REAL(wp) :: t_s(jpi,jpj,nlay_s,jpl)
END MODULE ice

MODULE sbc_oce
   USE par_kind, ONLY: wp
   USE par_ice, ONLY: jpi,jpj
   INTEGER :: nn_fsbc=1
   REAL(wp) :: sss_m(jpi,jpj),sst_m(jpi,jpj)
END MODULE sbc_oce

MODULE sbc_ice
   USE par_kind, ONLY: wp
   USE par_ice, ONLY: jpi,jpj
   REAL(wp) :: snwice_mass(jpi,jpj)
END MODULE sbc_ice

MODULE icevar
   USE par_kind, ONLY: wp
CONTAINS
   SUBROUTINE ice_var_zapneg(ihls,dt,vi,vs,svi,oai,ai,aip,vip,vil,es,ei,szvi)
      INTEGER,INTENT(in)::ihls
      REAL(wp),INTENT(in)::dt
      REAL(wp),DIMENSION(:,:,:),INTENT(inout)::vi,vs,svi,oai,ai,aip,vip,vil
      REAL(wp),DIMENSION(:,:,:,:),INTENT(inout)::es,ei,szvi
   END SUBROUTINE ice_var_zapneg
END MODULE icevar

MODULE in_out_manager
   LOGICAL :: lwp=.FALSE.,ln_rstart=.FALSE.,lrst_ice=.FALSE.
   INTEGER :: numout=6,nit000=1,numrir=10,numriw=11,nitrst=2
CONTAINS
   SUBROUTINE ctl_stop(a,b)
      CHARACTER(len=*),INTENT(in)::a
      CHARACTER(len=*),INTENT(in),OPTIONAL::b
      ERROR STOP 'unexpected ctl_stop in O4 bounds fixture'
   END SUBROUTINE ctl_stop
END MODULE in_out_manager

MODULE iom
   USE par_kind, ONLY: wp
   INTEGER,PARAMETER :: jpdom_auto=1
   INTERFACE iom_get
      MODULE PROCEDURE iom_get_any
   END INTERFACE
   INTERFACE iom_rstput
      MODULE PROCEDURE iom_rstput_any
   END INTERFACE
CONTAINS
   INTEGER FUNCTION iom_varid(unit,name,ldstop)
      INTEGER,INTENT(in)::unit
      CHARACTER(len=*),INTENT(in)::name
      LOGICAL,INTENT(in),OPTIONAL::ldstop
      iom_varid=0
   END FUNCTION iom_varid
   SUBROUTINE iom_get_any(unit,domain,name,a,psgn)
      INTEGER,INTENT(in)::unit,domain
      CHARACTER(len=*),INTENT(in)::name
      REAL(wp),DIMENSION(..),INTENT(out)::a
      REAL(wp),INTENT(in),OPTIONAL::psgn
      SELECT RANK(a)
      RANK(3); a=0._wp
      RANK DEFAULT; ERROR STOP 'unexpected iom_get rank'
      END SELECT
   END SUBROUTINE iom_get_any
   SUBROUTINE iom_rstput_any(iter,nitrst,unit,name,a)
      INTEGER,INTENT(in)::iter,nitrst,unit
      CHARACTER(len=*),INTENT(in)::name
      REAL(wp),DIMENSION(..),INTENT(in)::a
   END SUBROUTINE iom_rstput_any
END MODULE iom

MODULE lib_mpp
   USE par_kind, ONLY: wp
   USE dom_oce, ONLY: narea
   INTERFACE mpp_sum
      MODULE PROCEDURE mpp_sum_i
   END INTERFACE
CONTAINS
   SUBROUTINE mpp_max(name,value,cdelay)
      CHARACTER(len=*),INTENT(in)::name
      REAL(wp),INTENT(inout)::value
      CHARACTER(len=*),INTENT(in),OPTIONAL::cdelay
   END SUBROUTINE mpp_max
   SUBROUTINE mpp_sum_i(name,value)
      CHARACTER(len=*),INTENT(in)::name
      INTEGER,INTENT(inout)::value
   END SUBROUTINE mpp_sum_i
END MODULE lib_mpp

MODULE lbclnk
   USE par_kind, ONLY: wp
CONTAINS
   SUBROUTINE lbc_lnk(name, &
      a1,c1,s1,a2,c2,s2,a3,c3,s3,a4,c4,s4,a5,c5,s5,a6,c6,s6,a7,c7,s7,a8,c8,s8, &
      a9,c9,s9,a10,c10,s10,a11,c11,s11,a12,c12,s12,a13,c13,s13,a14,c14,s14,a15,c15,s15,a16,c16,s16, &
      a17,c17,s17,a18,c18,s18,a19,c19,s19,a20,c20,s20,a21,c21,s21,a22,c22,s22,a23,c23,s23,a24,c24,s24, &
      a25,c25,s25,a26,c26,s26,a27,c27,s27,a28,c28,s28,a29,c29,s29,a30,c30,s30,a31,c31,s31,a32,c32,s32, &
      a33,c33,s33,a34,c34,s34,a35,c35,s35,a36,c36,s36,a37,c37,s37,a38,c38,s38,a39,c39,s39,a40,c40,s40, &
      a41,c41,s41,a42,c42,s42,a43,c43,s43,a44,c44,s44,a45,c45,s45,a46,c46,s46,a47,c47,s47,a48,c48,s48,ldfull)
      CHARACTER(len=*),INTENT(in)::name,c1
      REAL(wp),DIMENSION(..),INTENT(inout)::a1
      REAL(wp),INTENT(in)::s1
      CHARACTER(len=*),INTENT(in),OPTIONAL::c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
      CHARACTER(len=*),INTENT(in),OPTIONAL::c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40
      CHARACTER(len=*),INTENT(in),OPTIONAL::c41,c42,c43,c44,c45,c46,c47,c48
      REAL(wp),DIMENSION(..),INTENT(inout),OPTIONAL::a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20
      REAL(wp),DIMENSION(..),INTENT(inout),OPTIONAL::a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40
      REAL(wp),DIMENSION(..),INTENT(inout),OPTIONAL::a41,a42,a43,a44,a45,a46,a47,a48
      REAL(wp),INTENT(in),OPTIONAL::s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20
      REAL(wp),INTENT(in),OPTIONAL::s21,s22,s23,s24,s25,s26,s27,s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40
      REAL(wp),INTENT(in),OPTIONAL::s41,s42,s43,s44,s45,s46,s47,s48
      LOGICAL,INTENT(in),OPTIONAL::ldfull
   END SUBROUTINE lbc_lnk
END MODULE lbclnk
