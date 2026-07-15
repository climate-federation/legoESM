! Minimal type shims so the VERBATIM sw_core / tp_core / a2b_edge
! extractions compile standalone for the phase-4 one-step oracles.  Only
! the derived types' CONSUMED fields are declared (inventory: grep
! gridstruct%/flagstruct% over the extracted ranges); the only numerics
! here are verbatim copies (great_circle_dist, f_p quad emulation dropped
! to double exactly like the -DNO_QUAD_PRECISION production builds).
module mpp_mod
  implicit none
contains
  integer function mpp_pe()
    mpp_pe = 0
  end function mpp_pe
end module mpp_mod

module swcore_shim_mod
  implicit none

  integer, parameter :: R_GRID = kind(1.d0)
  real, parameter :: big_number = 1.E8     ! fv_grid_utils big_number
  real, parameter :: tiny_number = 1.d-8   ! fv_grid_utils tiny_number
  integer, parameter :: XDir = 1           ! fv_mp_mod fill directions
  integer, parameter :: YDir = 2

  ! fill_corners module-scope bounds (fv_mp_mod globals); the driver must
  ! call set_fill_corner_bounds before any fill_corners use.
  integer :: is = -huge(1), ie = -huge(1), js = -huge(1), je = -huge(1)
  integer :: isd = -huge(1), ied = -huge(1)
  integer :: jsd = -huge(1), jed = -huge(1)
  integer :: ng = -huge(1)

  type fv_grid_bounds_type
    integer :: is, ie, js, je
    integer :: isd, ied, jsd, jed
    integer :: ng
  end type fv_grid_bounds_type

  type fv_grid_type
    logical :: bounded_domain = .false.
    logical :: sw_corner = .true., se_corner = .true.
    logical :: ne_corner = .true., nw_corner = .true.
    logical :: stretched_grid = .false.
    integer :: grid_type = 0
    real :: da_min = -9.e9, da_max = -9.e9
    real :: da_min_c = -9.e9, da_max_c = -9.e9
    ! corner-node positions (a2b_ord4)
    real(kind=R_GRID), allocatable, dimension(:, :, :) :: grid, agrid
    ! A-grid cell metrics
    real, allocatable, dimension(:, :) :: rarea, dxa, dya, cosa_s, rsin2
    real, allocatable, dimension(:, :) :: rdxa, rdya, f0
    ! D-edge lengths
    real, allocatable, dimension(:, :) :: dx, dy, rdx, rdy
    ! C-face metrics
    real, allocatable, dimension(:, :) :: dxc, dyc, rdxc, rdyc
    real, allocatable, dimension(:, :) :: cosa_u, cosa_v, sina_u, sina_v
    real, allocatable, dimension(:, :) :: rsin_u, rsin_v
    real, allocatable, dimension(:, :) :: divg_u, divg_v, del6_u, del6_v
    ! B-node metrics
    real, allocatable, dimension(:, :) :: rarea_c, fC, cosa, sina, rsina
    real, allocatable, dimension(:, :) :: area, area_c
    ! cubed->latlon (do_f3d only; allocated but unused in the oracle)
    real, allocatable, dimension(:, :) :: a11, a12, a21, a22, w00
    ! A->B edge interpolation factors (a2b_ord4)
    real, allocatable, dimension(:) :: edge_s, edge_n, edge_w, edge_e
    ! 9-position sub-grid tangents
    real, allocatable, dimension(:, :, :) :: sin_sg, cos_sg
  end type fv_grid_type

  type fv_flags_type
    integer :: grid_type = 0
    integer :: npx, npy
    logical :: do_diss_est = .false.
    logical :: do_f3d = .false.
    logical :: prevent_diss_cooling = .false.
    real :: lim_fac = 1.0
  end type fv_flags_type

contains

  subroutine set_fill_corner_bounds(bd)
    type(fv_grid_bounds_type), intent(in) :: bd
    is = bd%is; ie = bd%ie; js = bd%js; je = bd%je
    isd = bd%isd; ied = bd%ied; jsd = bd%jsd; jed = bd%jed
    ng = bd%ng
  end subroutine set_fill_corner_bounds

  ! VERBATIM fv_grid_utils.F90:1974-1996 (f_p == double here, matching
  ! -DNO_QUAD_PRECISION production builds)
  real function great_circle_dist( q1, q2, radius )
      real(kind=R_GRID), intent(IN)           :: q1(2), q2(2)
      real(kind=R_GRID), intent(IN), optional :: radius

      real (kind=R_GRID):: p1(2), p2(2)
      real (kind=R_GRID):: beta
      integer n

      do n=1,2
         p1(n) = q1(n)
         p2(n) = q2(n)
      enddo

      beta = asin( sqrt( sin((p1(2)-p2(2))/2.)**2 + cos(p1(2))*cos(p2(2))*   &
                         sin((p1(1)-p2(1))/2.)**2 ) ) * 2.

      if ( present(radius) ) then
           great_circle_dist = radius * beta
      else
           great_circle_dist = beta   ! Returns the angle
      endif

  end function great_circle_dist

end module swcore_shim_mod
