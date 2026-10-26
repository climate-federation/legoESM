MODULE par_kind
   IMPLICIT NONE
   PUBLIC
   INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(12,307)
END MODULE par_kind

MODULE in_out_manager
   IMPLICIT NONE
   PUBLIC
   INTEGER :: numout = 6
END MODULE in_out_manager

MODULE lib_mpp
   IMPLICIT NONE
   PUBLIC
CONTAINS
   SUBROUTINE ctl_stop( cd1 )
      CHARACTER(LEN=*), INTENT(in) :: cd1
      WRITE(*,*) TRIM(cd1)
      STOP 1
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
