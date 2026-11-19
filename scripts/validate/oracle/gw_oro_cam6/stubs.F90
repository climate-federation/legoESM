! Minimal CESM/CAM support stubs for the standalone gw_oro oracle.  None of
! them touch the gravity-wave numerics:
!   shr_kind_mod  - shr_kind_r8 = selected_real_kind(12), exactly as cime.
!   shr_log_mod / shr_sys_mod / cam_abortutils / cam_logfile - error paths.
!   physconst     - rair/rga are referenced only by interpolate_data's
!                   vertinterp-type routines, which this driver never calls.
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
  integer, parameter :: shr_kind_cl = 256
end module shr_kind_mod
module shr_log_mod
  implicit none
contains
  function shr_log_errMsg(file, line) result(msg)
    character(len=*), intent(in) :: file
    integer, intent(in) :: line
    character(len=512) :: msg
    write(msg,'(a,":",i0)') trim(file), line
  end function shr_log_errMsg
end module shr_log_mod
module shr_sys_mod
  implicit none
contains
  subroutine shr_sys_abort(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'ABORT: ', trim(msg)
    stop 1
  end subroutine shr_sys_abort
end module shr_sys_mod
module cam_abortutils
  implicit none
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'ENDRUN: ', trim(msg)
    stop 1
  end subroutine endrun
end module cam_abortutils
module cam_logfile
  implicit none
  integer :: iulog = 6
end module cam_logfile
module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  real(r8), parameter :: rair = huge(1._r8)   ! unused by lininterp
  real(r8), parameter :: rga = huge(1._r8)    ! unused by lininterp
end module physconst
