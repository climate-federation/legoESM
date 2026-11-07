MODULE par_kind
   INTEGER, PARAMETER :: wp = KIND(1.0D0)
END MODULE par_kind

MODULE par_oce
   INTEGER, PARAMETER :: jpi=8, jpj=7, jpk=6, jp_tem=1
END MODULE par_oce

MODULE in_out_manager
   INTEGER :: numout=6
END MODULE in_out_manager

MODULE lib_mpp
CONTAINS
   SUBROUTINE ctl_stop(message)
      CHARACTER(LEN=*), INTENT(in) :: message
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
