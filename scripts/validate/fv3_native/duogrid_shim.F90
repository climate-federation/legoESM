! Minimal shims so the VERBATIM duogrid global_grid.F90 (mirror of the
! Mouallem duo-grid variant, luanfs/FV3_container @ 7d06431e) compiles
! standalone for the phase-3 halo oracle.  Only build-plumbing is stubbed;
! no numerics live here.
module platform_mod
  implicit none
  integer, parameter :: r8_kind = selected_real_kind(15)
end module platform_mod

module constants_mod
  use platform_mod, only: r8_kind
  implicit none
  real(kind=r8_kind), parameter :: pi_8 = 4.0_r8_kind * atan(1.0_r8_kind)
  real(kind=r8_kind), parameter :: GRAV = 9.80_r8_kind      ! unused by grid gen
  real(kind=r8_kind), parameter :: OMEGA = 7.292e-5_r8_kind ! unused by grid gen
end module constants_mod

module mpp_mod
  implicit none
  integer, parameter :: FATAL = 1
contains
  integer function mpp_pe()
    mpp_pe = 0
  end function mpp_pe
  integer function mpp_root_pe()
    mpp_root_pe = 0
  end function mpp_root_pe
  subroutine mpp_error(level, msg)
    integer, intent(in) :: level
    character(len=*), intent(in) :: msg
    write(*,*) 'mpp_error: ', msg
    stop 1
  end subroutine mpp_error
end module mpp_mod
