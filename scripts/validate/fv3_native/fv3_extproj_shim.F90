! Minimal type/constant shim for the ext-projection oracle
! (a2stag_metrics + cubed_a2d/a2c_halo).  Mirrors only the fields the
! extracted routines consume.
module extproj_shim_mod
  implicit none
  public
  integer, parameter :: R_GRID = selected_real_kind(15)   ! r8
  integer, parameter :: f_p = selected_real_kind(20)      ! long double

  type fv_grid_bounds_type
    integer :: is, ie, js, je
    integer :: isd, ied, jsd, jed
    integer :: ng
  end type fv_grid_bounds_type

  type duogrid_type
    type(fv_grid_bounds_type) :: bd
    real(kind=R_GRID), allocatable :: a_pt(:, :, :)     ! (2, isd:ied, jsd:jed)
    real(kind=R_GRID), allocatable :: b_pt(:, :, :)     ! (2, isd:ied+1, jsd:jed+1)
    real(kind=R_GRID), allocatable :: vlon_ext(:, :, :) ! (isd:ied, jsd:jed, 3)
    real(kind=R_GRID), allocatable :: vlat_ext(:, :, :)
    real(kind=R_GRID), allocatable :: ew_ext(:, :, :, :) ! (3, isd:ied+1, jsd:jed, 2)
    real(kind=R_GRID), allocatable :: es_ext(:, :, :, :) ! (3, isd:ied, jsd:jed+1, 2)
  end type duogrid_type

  ! a2stag_metrics takes a gridstruct it never reads on this path
  type fv_grid_type
    integer :: unused = 0
  end type fv_grid_type

contains

  ! fill_corner_region is called by cubed_a2d/a2c_halo on the
  ! geographic inputs; the ORACLE feeds fully-defined analytic fields
  ! over the whole ng=4 data domain, so the call must be a no-op here.
  ! (The python side fills corners upstream of projection; this oracle
  ! certifies the PROJECTION + BASES only.)
  subroutine fill_corner_region(vel, bd, dg, istag, jstag)
    type(fv_grid_bounds_type), intent(IN) :: bd
    type(duogrid_type), intent(in) :: dg
    integer, intent(IN) :: istag, jstag
    real, dimension(bd%isd:, bd%jsd:) :: vel
    if (istag == -999999) vel(bd%isd, bd%jsd) = 0.  ! silence unused warnings
  end subroutine fill_corner_region

end module extproj_shim_mod
