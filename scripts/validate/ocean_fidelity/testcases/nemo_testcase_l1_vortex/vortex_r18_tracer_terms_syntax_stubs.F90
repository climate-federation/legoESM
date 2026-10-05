MODULE par_kind
   INTEGER, PARAMETER :: wp = KIND(1.0D0)
END MODULE par_kind
MODULE par_oce
   INTEGER, PARAMETER :: jpi = 67, jpj = 67, jpk = 11
END MODULE par_oce
MODULE oce
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi, jpj, jpk
   REAL(wp), DIMENSION(jpi,jpj,jpk) :: ww
END MODULE oce
MODULE dom_oce
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi, jpj
   REAL(wp), DIMENSION(jpi,jpj,3) :: r3t
END MODULE dom_oce
MODULE in_out_manager
   INTEGER :: numout = 6
END MODULE in_out_manager
MODULE lib_mpp
CONTAINS
   SUBROUTINE ctl_stop( cdmessage )
      CHARACTER(LEN=*), INTENT(in) :: cdmessage
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
