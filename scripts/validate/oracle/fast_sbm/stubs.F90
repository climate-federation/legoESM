subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  print *, 'WRF_ERROR_FATAL: ', trim(msg)
  stop 99
end subroutine wrf_error_fatal
