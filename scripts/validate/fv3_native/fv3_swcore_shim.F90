! Minimal type shims so the VERBATIM sw_core extraction compiles standalone
! for the phase-4 c_sw one-step oracle.  Only the derived types' CONSUMED
! fields are declared (inventory: grep gridstruct%/flagstruct% over the
! extracted ranges); no numerics live here.
module mpp_mod
  implicit none
contains
  integer function mpp_pe()
    mpp_pe = 0
  end function mpp_pe
end module mpp_mod

module swcore_shim_mod
  implicit none

  type fv_grid_bounds_type
    integer :: is, ie, js, je
    integer :: isd, ied, jsd, jed
    integer :: ng
  end type fv_grid_bounds_type

  type fv_grid_type
    logical :: bounded_domain = .false.
    logical :: sw_corner = .true., se_corner = .true.
    logical :: ne_corner = .true., nw_corner = .true.
    ! A-grid cell metrics
    real, allocatable, dimension(:, :) :: rarea, dxa, dya, cosa_s, rsin2
    ! D-edge lengths
    real, allocatable, dimension(:, :) :: dx, dy
    ! C-face metrics
    real, allocatable, dimension(:, :) :: dxc, dyc, rdxc, rdyc
    real, allocatable, dimension(:, :) :: cosa_u, cosa_v, sina_u, sina_v
    real, allocatable, dimension(:, :) :: rsin_u, rsin_v
    ! B-node metrics
    real, allocatable, dimension(:, :) :: rarea_c, fC
    ! 9-position sub-grid tangents
    real, allocatable, dimension(:, :, :) :: sin_sg, cos_sg
  end type fv_grid_type

  type fv_flags_type
    integer :: grid_type = 0
    integer :: npx, npy
  end type fv_flags_type

end module swcore_shim_mod
