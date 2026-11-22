! Minimal stand-ins for the CAM modules conv_water.F90 (cam_cesm2_1_rel_60)
! uses, so the routine compiles unmodified outside CAM.  Only what
! conv_water_readnl, conv_water_register, conv_water_init and conv_water_4rad
! touch is provided.  PVER is set at build time (-DPVER_VAL=..).
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod

module spmd_utils
  implicit none
  logical, parameter :: masterproc = .true.
end module spmd_utils

module ppgrid
  implicit none
  integer, parameter :: pcols = 64
  integer, parameter :: pver = PVER_VAL
  integer, parameter :: pverp = pver + 1
end module ppgrid

module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  real(r8), parameter :: gravit = 9.80616_r8
  real(r8), parameter :: latvap = 2.501e6_r8
  real(r8), parameter :: latice = 3.337e5_r8
  real(r8), parameter :: mwdry = 28.966_r8
  real(r8), parameter :: cpair = 1004.64_r8
end module physconst

module cam_abortutils
  implicit none
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'ENDRUN: ', trim(msg)
    stop 1
  end subroutine endrun
end module cam_abortutils

module perf_mod
  implicit none
end module perf_mod

module cam_logfile
  implicit none
  integer :: iulog = 6
end module cam_logfile

module mpishorthand
  implicit none
end module mpishorthand

module namelist_utils
  implicit none
contains
  subroutine find_group_name(unit, group, status)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: group
    integer, intent(out) :: status
    rewind(unit)
    status = 0
  end subroutine find_group_name
end module namelist_utils

module units
  implicit none
contains
  integer function getunit()
    getunit = 41
  end function getunit
  subroutine freeunit(u)
    integer, intent(in) :: u
  end subroutine freeunit
end module units

module constituents
  implicit none
  integer, parameter :: pcnst = 3
contains
  subroutine cnst_add(name)
    character(len=*), intent(in) :: name
  end subroutine cnst_add
  subroutine cnst_get_ind(name, ind)
    character(len=*), intent(in) :: name
    integer, intent(out) :: ind
    select case (trim(name))
    case ('CLDLIQ')
      ind = 2
    case ('CLDICE')
      ind = 3
    case default
      ind = 1
    end select
  end subroutine cnst_get_ind
end module constituents

module physics_buffer
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver
  implicit none
  integer, parameter :: dtype_r8 = 1
  integer, parameter :: nfld = 16
  type physics_buffer_desc
    integer :: dummy = 0
  end type physics_buffer_desc
  real(r8), target :: store(pcols, pver, nfld)
  character(len=16) :: names(nfld) = ''
  integer :: nnames = 0
  interface pbuf_get_field
    module procedure get_2d
  end interface
contains
  integer function pbuf_get_index(name, errcode)
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: errcode
    integer :: i
    if (present(errcode)) errcode = 0
    do i = 1, nnames
      if (names(i) == trim(name)) then
        pbuf_get_index = i
        return
      end if
    end do
    nnames = nnames + 1
    names(nnames) = trim(name)
    store(:, :, nnames) = 0._r8
    pbuf_get_index = nnames
  end function pbuf_get_index
  subroutine pbuf_add_field(name, scope, dtype, dims, idx)
    character(len=*), intent(in) :: name, scope
    integer, intent(in) :: dtype, dims(:)
    integer, intent(out) :: idx
    idx = pbuf_get_index(name)
  end subroutine pbuf_add_field
  integer function pbuf_old_tim_idx()
    pbuf_old_tim_idx = 1
  end function pbuf_old_tim_idx
  subroutine get_2d(pbuf, idx, field, start, kount)
    type(physics_buffer_desc), pointer :: pbuf(:)
    integer, intent(in) :: idx
    real(r8), pointer :: field(:,:)
    integer, intent(in), optional :: start(:), kount(:)
    field => store(:, :, idx)
  end subroutine get_2d
end module physics_buffer

module physics_types
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver
  implicit none
  type physics_state
    integer :: ncol = 0
    integer :: lchnk = 0
    real(r8) :: pdel(pcols, pver)
    real(r8) :: q(pcols, pver, 3)
  end type physics_state
end module physics_types

module cam_history
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
contains
  subroutine outfld(name, field, idim, lchnk)
    character(len=*), intent(in) :: name
    integer, intent(in) :: idim, lchnk
    real(r8), intent(in) :: field(idim, *)
  end subroutine outfld
  subroutine addfld(name, dims, avgflag, units, long_name)
    character(len=*), intent(in) :: name, avgflag, units, long_name
    character(len=*), intent(in) :: dims(:)
  end subroutine addfld
end module cam_history

module phys_control
  implicit none
contains
  subroutine phys_getopts(microp_scheme_out)
    character(len=16), intent(out), optional :: microp_scheme_out
    if (present(microp_scheme_out)) microp_scheme_out = 'MG'
  end subroutine phys_getopts
end module phys_control
