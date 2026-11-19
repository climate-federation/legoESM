! Minimal stubs for the CESM infrastructure shr_orb_mod.F90 uses (abort, log
! unit/level); everything numerical is compiled unmodified from cime5.6.47.
module shr_sys_mod
  implicit none
contains
  subroutine shr_sys_abort(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) print *, 'SHR_SYS_ABORT: ', trim(msg)
    stop 1
  end subroutine shr_sys_abort
end module shr_sys_mod

module shr_log_mod
  implicit none
  integer :: shr_log_Level = 0
  integer :: shr_log_Unit  = 6
end module shr_log_mod
