! Minimal CAM infrastructure stubs for the standalone E3SM gw oracle.
! shr_kind_r8 = selected_real_kind(12) as in CIME shr_kind_mod.
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
  integer, parameter :: shr_kind_cl = 256
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
    if (present(msg)) write(*,*) 'ENDRUN: ', trim(msg)
    stop 1
  end subroutine endrun
end module cam_abortutils
