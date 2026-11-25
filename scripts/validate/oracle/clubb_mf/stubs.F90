! Minimal CAM/CCPP infrastructure for the clubb_mf oracle driver.  Physical
! constants are SET AT RUNTIME by the driver from legoESM's constants, so both
! sides of the oracle comparison use identical values.  No physics here.
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
  integer, parameter :: shr_kind_i8 = selected_int_kind(13)
  integer, parameter :: shr_kind_in = kind(1)
  integer, parameter :: shr_kind_cl = 256
end module shr_kind_mod

module ccpp_kinds
  use shr_kind_mod, only: shr_kind_r8
  implicit none
  integer, parameter :: kind_phys = shr_kind_r8
end module ccpp_kinds

module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  real(r8) :: epsilo, latvap, latice, rh2o, cpair, tmelt, h2otrip
end module physconst

module spmd_utils
  implicit none
  logical :: masterproc = .false.
end module spmd_utils

module cam_logfile
  implicit none
  integer :: iulog = 6
end module cam_logfile

module cam_abortutils
  implicit none
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(0,*) 'ENDRUN: ', trim(msg)
    error stop 1
  end subroutine endrun
end module cam_abortutils

module shr_assert_mod
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
contains
  subroutine shr_assert_in_domain(var, varname, msg, ge, le)
    real(r8), intent(in) :: var
    character(len=*), intent(in), optional :: varname, msg
    real(r8), intent(in), optional :: ge, le
    if (present(ge)) then
      if (var < ge) error stop 'shr_assert_in_domain'
    end if
    if (present(le)) then
      if (var > le) error stop 'shr_assert_in_domain'
    end if
  end subroutine shr_assert_in_domain
end module shr_assert_mod

module error_messages
  implicit none
contains
  subroutine handle_errmsg(errmsg, subname)
    character(len=*), intent(in) :: errmsg
    character(len=*), intent(in), optional :: subname
    if (len_trim(errmsg) > 0) then
      write(0,*) 'ERRMSG: ', trim(errmsg)
      error stop 1
    end if
  end subroutine handle_errmsg
end module error_messages

module namelist_utils
  implicit none
contains
  subroutine find_group_name(unit, group, status)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: group
    integer, intent(out) :: status
    status = 1
  end subroutine find_group_name
end module namelist_utils
