! Minimal stand-in for WRF frame/module_wrf_error.F (only the two routines
! module_cu_kfeta.F calls).  wrf_error_fatal does NOT stop: it records the
! message so the harness can flag the column (the oracle would abort the run).
MODULE module_wrf_error
  IMPLICIT NONE
  INTEGER :: n_fatal = 0
CONTAINS
  SUBROUTINE wrf_message(str)
    CHARACTER(LEN=*), INTENT(IN) :: str
    WRITE(0,'(A)') 'WRF_MESSAGE: '//TRIM(str)
  END SUBROUTINE wrf_message
  SUBROUTINE wrf_error_fatal(str)
    CHARACTER(LEN=*), INTENT(IN) :: str
    n_fatal = n_fatal + 1
    WRITE(0,'(A)') 'WRF_FATAL(continued by harness): '//TRIM(str)
  END SUBROUTINE wrf_error_fatal
END MODULE module_wrf_error
