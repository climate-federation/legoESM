MODULE dom_oce
   USE, INTRINSIC :: iso_fortran_env, ONLY : real64
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=real64, jpk=4, jpkm1=3
   INTEGER :: ntsi=1, ntei=3, ntsj=1, ntej=2, narea=1
END MODULE dom_oce

MODULE in_out_manager
   IMPLICIT NONE
   INTEGER :: nit000=1
END MODULE in_out_manager

MODULE lib_mpp
   IMPLICIT NONE
CONTAINS
   SUBROUTINE ctl_stop(message)
      CHARACTER(LEN=*), INTENT(in) :: message
      WRITE(*,*) TRIM(message)
      ERROR STOP 1
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
