! Shim for the BOUNDED-conventions gridstruct oracle (grid_utils_init
! verbatim, bounded_domain=.true.).  Only build plumbing; no numerics.
! Every routine that upstream gates on .not.bounded_domain is a
! LOUD-STOP stub — if one fires, the bounded flag did not reach the
! extract and the oracle run is invalid.
module bounded_gs_shim_mod
  implicit none
  public
  integer, parameter :: R_GRID = selected_real_kind(15)
  integer, parameter :: f_p = selected_real_kind(20)
  real(kind=R_GRID), parameter :: big_number = 1.d8
  real(kind=R_GRID), parameter :: tiny_number = 1.d-8
  ! Zenodo duo run log: "Radius is 6371200.0" (FMS constants_mod)
  real(kind=R_GRID), parameter :: radius = 6371.2d3
  real(kind=R_GRID), parameter :: pi = 4.0d0 * atan(1.0d0)
  real(kind=R_GRID), parameter :: todeg = 180.0d0 / pi
  real(kind=R_GRID), parameter :: torad = pi / 180.0d0
  integer, parameter :: XDir = 1, YDir = 2
  integer, parameter :: CORNER = 1, CENTER = 2, SCALAR_PAIR = 3
  integer, parameter :: AGRID_PARAM = 4, DGRID_NE = 5
  integer, parameter :: CGRID_NE_PARAM = 6
  integer, parameter :: FATAL = 1
  logical :: symm_grid = .true.

  type shim_bd_type
    integer :: is, ie, js, je
    integer :: isd, ied, jsd, jed
    integer :: ng
  end type shim_bd_type

  type shim_gridstruct_type
    logical :: bounded_domain = .true.
    logical :: stretched_grid = .false.
    logical :: sw_corner = .true., se_corner = .true.
    logical :: nw_corner = .true., ne_corner = .true.
    real(kind=R_GRID) :: da_min, da_max, da_min_c, da_max_c
    real(kind=R_GRID), allocatable :: grid(:, :, :), agrid(:, :, :)
    real(kind=R_GRID), allocatable :: grid_64(:, :, :), agrid_64(:, :, :)
    real(kind=R_GRID), allocatable :: area(:, :), area_c(:, :)
    real(kind=R_GRID), allocatable :: area_64(:, :), area_c_64(:, :)
    real(kind=R_GRID), allocatable :: dx(:, :), dy(:, :)
    real(kind=R_GRID), allocatable :: dx_64(:, :), dy_64(:, :)
    real(kind=R_GRID), allocatable :: dxa(:, :), dya(:, :)
    real(kind=R_GRID), allocatable :: dxa_64(:, :), dya_64(:, :)
    real(kind=R_GRID), allocatable :: dxc(:, :), dyc(:, :)
    real(kind=R_GRID), allocatable :: dxc_64(:, :), dyc_64(:, :)
    real(kind=R_GRID), allocatable :: sina(:, :), cosa(:, :)
    real(kind=R_GRID), allocatable :: sina_64(:, :), cosa_64(:, :)
    real, allocatable :: rsina(:, :), rsin2(:, :)
    real, allocatable :: rsin_u(:, :), rsin_v(:, :)
    real, allocatable :: cosa_u(:, :), cosa_v(:, :), cosa_s(:, :)
    real, allocatable :: sina_u(:, :), sina_v(:, :)
    real, allocatable :: divg_u(:, :), divg_v(:, :)
    real, allocatable :: del6_u(:, :), del6_v(:, :)
    real, allocatable :: sin_sg(:, :, :), cos_sg(:, :, :)
    real(kind=R_GRID), allocatable :: ec1(:, :, :), ec2(:, :, :)
    real(kind=R_GRID), allocatable :: ew(:, :, :, :), es(:, :, :, :)
    real(kind=R_GRID), allocatable :: ee1(:, :, :), ee2(:, :, :)
    real(kind=R_GRID), allocatable :: en1(:, :, :), en2(:, :, :)
    real(kind=R_GRID), allocatable :: eww(:, :), ess(:, :)
    real, allocatable :: l2c_u(:, :), l2c_v(:, :)
    real, allocatable :: edge_s(:), edge_n(:), edge_w(:), edge_e(:)
    real, allocatable :: edge_vect_s(:), edge_vect_n(:)
    real, allocatable :: edge_vect_w(:), edge_vect_e(:)
  end type shim_gridstruct_type

  type shim_neststruct_type
    logical :: nested = .false.
  end type shim_neststruct_type

  type shim_flagstruct_type
    logical :: do_schmidt = .false., do_cube_transform = .false.
    logical :: hybrid_z = .false., hydrostatic = .true.
    logical :: external_eta = .false.
    real(kind=R_GRID) :: stretch_fac = 1.0d0
    character(len=16) :: npz_type = ""
  end type shim_flagstruct_type

  type shim_atm_type
    type(shim_bd_type) :: bd
    type(shim_gridstruct_type) :: gridstruct
    type(shim_flagstruct_type) :: flagstruct
    type(shim_neststruct_type) :: neststruct
    real, allocatable :: ak(:), bk(:)
    real :: ptop = 1.0
    integer :: ks = 0
    integer :: domain = 0
    integer :: npz = 1
    integer :: ng = 3
  end type shim_atm_type


  interface fill_corners
    module procedure fill_corners_2d
    module procedure fill_corners_pair
  end interface fill_corners

  interface mpp_update_domains
    module procedure mpp_update_domains_2d
    module procedure mpp_update_domains_3d
    module procedure mpp_update_domains_pair
  end interface mpp_update_domains

contains

  subroutine mpp_update_domains_2d(f, domain, position, complete, flags, &
                                   gridtype)
    real(kind=R_GRID), dimension(:, :), intent(inout) :: f
    integer, intent(in), optional :: domain, position, flags, gridtype
    logical, intent(in), optional :: complete
    ! no-op: the driver pre-fills mpp-state halos
    if (size(f) < 0) f(1, 1) = 0.0d0
  end subroutine mpp_update_domains_2d

  subroutine mpp_update_domains_3d(f, domain, position, complete, flags, &
                                   gridtype)
    real(kind=R_GRID), dimension(:, :, :), intent(inout) :: f
    integer, intent(in), optional :: domain, position, flags, gridtype
    logical, intent(in), optional :: complete
    if (size(f) < 0) f(1, 1, 1) = 0.0d0
  end subroutine mpp_update_domains_3d

  subroutine mpp_update_domains_pair(a, b, domain, position, complete, &
                                     flags, gridtype)
    real(kind=R_GRID), dimension(:, :), intent(inout) :: a, b
    integer, intent(in), optional :: domain, position, flags, gridtype
    logical, intent(in), optional :: complete
    if (size(a) < 0) a(1, 1) = b(1, 1)
  end subroutine mpp_update_domains_pair

  subroutine fill_corners_2d(q, npx, npy, FILL, AGRID, BGRID)
    real(kind=R_GRID), dimension(:, :), intent(inout) :: q
    integer, intent(in) :: npx, npy
    integer, intent(in), optional :: FILL
    logical, intent(in), optional :: AGRID, BGRID
    write (*, *) "fill_corners fired under bounded conventions", npx, npy
    stop 3
    if (size(q) < 0) q(1, 1) = 0.0d0
  end subroutine fill_corners_2d

  subroutine fill_corners_pair(x, y, npx, npy, DGRID, AGRID, VECTOR)
    real(kind=R_GRID), dimension(:, :), intent(inout) :: x, y
    integer, intent(in) :: npx, npy
    logical, intent(in), optional :: DGRID, AGRID, VECTOR
    write (*, *) "fill_corners(pair) fired under bounded", npx, npy
    stop 3
    if (size(x) < 0) x(1, 1) = y(1, 1)
  end subroutine fill_corners_pair

  subroutine fill_ghost(q, npx, npy, value, bd)
    type(shim_bd_type), intent(in) :: bd
    real(kind=R_GRID), dimension(bd%isd:, bd%jsd:), intent(inout) :: q
    integer, intent(in) :: npx, npy
    real(kind=R_GRID), intent(in) :: value
    write (*, *) "fill_ghost fired under bounded conventions", npx, value
    stop 3
    if (size(q) < 0) q(bd%isd, bd%jsd) = 0.0d0
  end subroutine fill_ghost

  subroutine global_mx(q, n_g, qmin, qmax, bd)
    type(shim_bd_type), intent(in) :: bd
    integer, intent(in) :: n_g
    real(kind=R_GRID), intent(in) :: q(bd%is - n_g:, bd%js - n_g:)
    real(kind=R_GRID), intent(out) :: qmin, qmax
    integer :: ni, nj
    ni = bd%ie - bd%is + 1
    nj = bd%je - bd%js + 1
    qmin = minval(q(bd%is:bd%ie, bd%js:bd%je))
    qmax = maxval(q(bd%is:bd%ie, bd%js:bd%je))
    if (ni < 0) write (*, *) nj
  end subroutine global_mx

  subroutine global_mx_c(q, i1, i2, j1, j2, qmin, qmax)
    integer, intent(in) :: i1, i2, j1, j2
    real(kind=R_GRID), intent(in) :: q(i1:, j1:)
    real(kind=R_GRID), intent(out) :: qmin, qmax
    qmin = minval(q(i1:i2, j1:j2))
    qmax = maxval(q(i1:i2, j1:j2))
  end subroutine global_mx_c

  ! tail-section machinery the metric oracle does not exercise
  subroutine set_eta(npz, ks, ptop, ak, bk, npz_type)
    integer, intent(in) :: npz
    integer, intent(out) :: ks
    real, intent(out) :: ptop
    real, intent(inout) :: ak(:), bk(:)
    character(len=*), intent(in) :: npz_type
    ks = 0; ptop = 1.0; ak = 0.0; bk = 0.0
    if (npz < 0) write (*, *) npz_type
  end subroutine set_eta

  subroutine van2d_init()
  end subroutine van2d_init

  subroutine init_cubed_to_latlon(gridstruct, hydrostatic, agrid, &
                                  grid_type, ord, bd)
    type(shim_gridstruct_type), intent(in) :: gridstruct
    logical, intent(in) :: hydrostatic
    real(kind=R_GRID), intent(in) :: agrid(:, :, :)
    integer, intent(in) :: grid_type, ord
    type(shim_bd_type), intent(in) :: bd
    if (grid_type < -99) write (*, *) hydrostatic, ord, bd%is, &
      gridstruct%bounded_domain, agrid(1, 1, 1)
  end subroutine init_cubed_to_latlon

  subroutine edge_factors(edge_s, edge_n, edge_w, edge_e, non_ortho, &
                          grid, agrid, npx, npy, bd)
    type(shim_bd_type), intent(in) :: bd
    real, intent(inout) :: edge_s(:), edge_n(:), edge_w(:), edge_e(:)
    logical, intent(in) :: non_ortho
    real(kind=R_GRID), intent(in) :: grid(:, :, :), agrid(:, :, :)
    integer, intent(in) :: npx, npy
    edge_s = 0.5; edge_n = 0.5; edge_w = 0.5; edge_e = 0.5
    if (npx < 0) write (*, *) non_ortho, npy, bd%is, grid(1, 1, 1), &
      agrid(1, 1, 1)
  end subroutine edge_factors

  subroutine extend_cube_s(grid, agrid, bd, npx, npy, bounded_domain, &
                           stretched)
    type(shim_bd_type), intent(in) :: bd
    real(kind=R_GRID), intent(inout) :: grid(:, :, :), agrid(:, :, :)
    integer, intent(in) :: npx, npy
    logical, intent(in) :: bounded_domain, stretched
    if (npx < 0) write (*, *) npy, bd%is, bounded_domain, stretched, &
      grid(1, 1, 1), agrid(1, 1, 1)
  end subroutine extend_cube_s

  logical function is_master()
    is_master = .true.
  end function is_master

  subroutine mpp_error(level, msg)
    integer, intent(in) :: level
    character(len=*), intent(in) :: msg
    write (*, *) "mpp_error: ", msg
    stop 1
    if (level < 0) continue
  end subroutine mpp_error

  integer function mpp_pe()
    mpp_pe = 0
  end function mpp_pe

  integer function mpp_root_pe()
    mpp_root_pe = 0
  end function mpp_root_pe

end module bounded_gs_shim_mod
