! Minimal shim for the corner-Lagrange oracle (fill_corner_region_2d +
! compute_lagrange_coeff + lagrange_poly_interp_2d).
module cornerlag_shim_mod
  implicit none
  public
  integer, parameter :: R_GRID = selected_real_kind(15)
  integer, parameter :: f_p = selected_real_kind(20)
  integer :: interporder = 3          ! fv_duogrid.F90:80

  type fv_grid_bounds_type
    integer :: is, ie, js, je
    integer :: isd, ied, jsd, jed
    integer :: ng
  end type fv_grid_bounds_type

  type duogrid_type
    type(fv_grid_bounds_type) :: bd
    logical :: rmp_w = .true., rmp_e = .true.
    logical :: rmp_s = .true., rmp_n = .true.
    logical :: rmp_sw = .true., rmp_se = .true.
    logical :: rmp_nw = .true., rmp_ne = .true.
    real(kind=R_GRID), allocatable :: a_pt(:, :, :)   ! (2, isd:ied, jsd:jed)
    real(kind=R_GRID), allocatable :: xp(:, :, :, :)
    real(kind=R_GRID), allocatable :: xm(:, :, :, :)
    real(kind=R_GRID), allocatable :: yp(:, :, :, :)
    real(kind=R_GRID), allocatable :: ym(:, :, :, :)
  end type duogrid_type

contains
  subroutine timing_on(name)
    character(len=*), intent(in) :: name
    if (len(name) < 0) continue
  end subroutine timing_on
  subroutine timing_off(name)
    character(len=*), intent(in) :: name
    if (len(name) < 0) continue
  end subroutine timing_off
  integer function mpp_pe()
    mpp_pe = 0
  end function mpp_pe
  integer function mpp_root_pe()
    mpp_root_pe = 0
  end function mpp_root_pe
end module cornerlag_shim_mod
