MODULE par_oce
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=real64, jpi=12, jpj=10, nn_hls=2
   INTEGER, PARAMETER :: ntsi=3, ntei=10, ntsj=3, ntej=8
   INTEGER, PARAMETER :: Nis0=ntsi, Nie0=ntei, Njs0=ntsj, Nje0=ntej
END MODULE par_oce

MODULE phycst
   USE par_oce
   IMPLICIT NONE
   REAL(wp), PARAMETER :: rt0=273.15_wp, ralb_oce=0.06_wp
END MODULE phycst

MODULE par_ice
   USE par_oce
   IMPLICIT NONE
   INTEGER, PARAMETER :: jpl=5
END MODULE par_ice

MODULE dom_oce
   USE par_oce
   IMPLICIT NONE
   ! NEMO 5.0.2 dom_oce.F90 declares smask0 on A2D(0), without halos.
   REAL(wp), DIMENSION(Nis0:Nie0,Njs0:Nje0) :: smask0=1._wp
END MODULE dom_oce

MODULE icevar
   USE par_oce
   USE par_ice
   IMPLICIT NONE
CONTAINS
   SUBROUTINE ice_var_snwfra(ph_snw, fraction)
      REAL(wp), INTENT(in), DIMENSION(Nis0-nn_hls:Nie0+nn_hls,Njs0-nn_hls:Nje0+nn_hls,jpl) :: ph_snw
      REAL(wp), INTENT(out), DIMENSION(Nis0-nn_hls:Nie0+nn_hls,Njs0-nn_hls:Nje0+nn_hls,jpl) :: fraction
      fraction=MIN(0.5_wp,ph_snw)
   END SUBROUTINE ice_var_snwfra
END MODULE icevar

MODULE in_out_manager
   IMPLICIT NONE
   CHARACTER(LEN=512) :: numnam_ice_ref = '&namalb rn_alb_sdry=.85, rn_alb_smlt=.75, rn_alb_idry=.60, rn_alb_imlt=.50, rn_alb_dpnd=.36, rn_alb_lpnd=.21, rn_alb_hpiv=1.0 /'
   CHARACTER(LEN=32) :: numnam_ice_cfg = '&namalb /'
   INTEGER :: numoni=99, numout=6
   LOGICAL :: lwm=.FALSE., lwp=.FALSE.
END MODULE in_out_manager

MODULE lib_mpp
   IMPLICIT NONE
CONTAINS
   SUBROUTINE ctl_nam(ios,name,required)
      INTEGER, INTENT(in) :: ios
      CHARACTER(LEN=*), INTENT(in) :: name
      LOGICAL, INTENT(in) :: required
      IF( ios /= 0 .AND. required ) ERROR STOP name
   END SUBROUTINE ctl_nam
END MODULE lib_mpp

MODULE timing
   IMPLICIT NONE
   LOGICAL :: ln_timing=.FALSE.
CONTAINS
   SUBROUTINE timing_start(name)
      CHARACTER(LEN=*), INTENT(in) :: name
   END SUBROUTINE timing_start
   SUBROUTINE timing_stop(name)
      CHARACTER(LEN=*), INTENT(in) :: name
   END SUBROUTINE timing_stop
END MODULE timing
