MODULE par_kind
   INTEGER, PARAMETER :: wp=KIND(1.0D0)
END MODULE par_kind
MODULE par_oce
   INTEGER, PARAMETER :: jpi=5,jpj=4,jpk=3,jpts=2
END MODULE par_oce
MODULE dom_oce
   INTEGER :: narea=1
END MODULE dom_oce
MODULE in_out_manager
   INTEGER, PARAMETER :: nit000=1
END MODULE in_out_manager
MODULE lib_mpp
CONTAINS
   SUBROUTINE ctl_stop(message)
      CHARACTER(LEN=*), INTENT(in) :: message
      WRITE(*,*) 'CTL_STOP ',TRIM(message)
      ERROR STOP 71
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
