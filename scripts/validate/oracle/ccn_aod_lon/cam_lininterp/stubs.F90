! Minimal stand-ins for the CAM modules that CAM's src/utils/interpolate_data.F90
! (cam_cesm2_1_rel_60, a03b84b7) USEs, so that file compiles unmodified outside
! CAM.  Nothing here enters lininterp_init / lininterp2d1d: rair/rga are read
! only by interpolate_data's vertical routine, which this harness never calls.
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod

module cam_logfile
  implicit none
  integer :: iulog = 6
end module cam_logfile

module cam_abortutils
  implicit none
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'endrun: ', trim(msg)
    error stop 1
  end subroutine endrun
end module cam_abortutils

module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  real(r8), parameter :: rair = 0._r8, rga = 0._r8   ! unused by lininterp
end module physconst
