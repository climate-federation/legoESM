! Minimal stand-ins for the CAM modules that mapz_module.F90 uses:
! the real kind, the log unit, an abort routine and the timers.
module shr_kind_mod
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
  integer, parameter :: shr_kind_i4 = selected_int_kind(6)
end module shr_kind_mod
module cam_logfile
  integer :: iulog = 6
end module cam_logfile
module cam_abortutils
contains
  subroutine endrun(msg)
    character(len=*), intent(in) :: msg
    write(*,*) 'ENDRUN: ', msg
    stop 1
  end subroutine endrun
end module cam_abortutils
module FVperf_module
contains
  subroutine FVstartclock(a, b)
    integer, optional :: a, b
  end subroutine FVstartclock
  subroutine FVstopclock(a, b)
    integer, optional :: a, b
  end subroutine FVstopclock
end module FVperf_module
