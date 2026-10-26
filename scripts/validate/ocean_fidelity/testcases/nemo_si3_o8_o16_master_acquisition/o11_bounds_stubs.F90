MODULE par_kind
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   INTEGER, PARAMETER :: wp=real64
END MODULE par_kind
MODULE par_ice
   INTEGER, PARAMETER :: nlay_i=10, nlay_s=5
END MODULE par_ice
MODULE phycst
   USE par_kind, ONLY: wp
   REAL(wp), PARAMETER :: rpi=3.1415926535897932384626433832795_wp, rt0=273.15_wp
   REAL(wp), PARAMETER :: rhoi=917._wp, rhos=330._wp
END MODULE phycst
MODULE ice1D
   USE par_kind, ONLY: wp
   USE par_ice, ONLY: nlay_i
   INTEGER :: npti=3, nn_icesal=4
   REAL(wp) :: rDt_ice=10800._wp, r1_Dt_ice=1._wp/10800._wp
   REAL(wp), PARAMETER :: r1_nlay_i=0.1_wp, r1_nlay_s=0.2_wp
   REAL(wp) :: at_i_1d(3), a_i_1d(3), h_i_1d(3), h_s_1d(3), s_i_1d(3)
   REAL(wp) :: sst_1d(3), t_bo_1d(3), sfx_lam_1d(3), hfx_thd_1d(3), wfx_lam_1d(3)
   REAL(wp) :: sz_i_1d(3,nlay_i), e_i_1d(3,nlay_i), e_s_1d(3,5)
END MODULE ice1D
MODULE in_out_manager
   IMPLICIT NONE
   CHARACTER(LEN=256) :: numnam_ice_ref='&namthd_da rn_beta=1.0, rn_dmin=8.0 /'
   CHARACTER(LEN=256) :: numnam_ice_cfg='&namthd_da /'
   INTEGER :: numout=6, numoni=6
   LOGICAL :: lwp=.FALSE., lwm=.FALSE.
END MODULE in_out_manager
MODULE lib_mpp
   IMPLICIT NONE
CONTAINS
   SUBROUTINE ctl_stop(message); CHARACTER(LEN=*),INTENT(in)::message; ERROR STOP message; END SUBROUTINE
   SUBROUTINE ctl_warn(message); CHARACTER(LEN=*),INTENT(in)::message; END SUBROUTINE
   SUBROUTINE ctl_nam(ios,label,last); INTEGER,INTENT(in)::ios; CHARACTER(LEN=*),INTENT(in)::label; LOGICAL,INTENT(in)::last; IF(ios/=0) ERROR STOP label; END SUBROUTINE
END MODULE lib_mpp
MODULE o8_o16_master_writer
   USE, INTRINSIC :: iso_fortran_env, ONLY: int64
   USE par_kind, ONLY: wp
   USE ice1D
   IMPLICIT NONE
CONTAINS
   SUBROUTINE o8o16_category(kind,jl,beta,dmin)
      INTEGER,INTENT(in)::kind,jl
      REAL(wp),INTENT(in),OPTIONAL::beta,dmin
      IF(kind/=9 .OR. jl/=0) ERROR STOP 'bounds fixture observer call mismatch'
      IF(.NOT.PRESENT(beta) .OR. .NOT.PRESENT(dmin)) ERROR STOP 'bounds fixture missing DA scalars'
      IF(SIZE(e_i_1d,2)/=10 .OR. SIZE(e_s_1d,2)/=5) ERROR STOP 'bounds fixture layer extent'
      IF(TRANSFER(sfx_lam_1d(1),0_int64)/=TRANSFER(-0._wp,0_int64)) ERROR STOP 'bounds fixture lost -0'
      IF(beta/=1._wp .OR. dmin/=8._wp) ERROR STOP 'bounds fixture namelist scalar mismatch'
   END SUBROUTINE o8o16_category
END MODULE o8_o16_master_writer
