! Minimal FMS stubs so Isca v1.0 hs_forcing.F90 compiles standalone, unmodified.
! No physics here.  constants_mod KAPPA / CP_AIR / GRAV are SET BY THE DRIVER from its
! input file (legoESM values written by make_fixture.py); the sentinels make an unset
! value fail loudly.  Every procedure the default HS path does not reach stops.
module constants_mod
  implicit none
  real, public :: KAPPA = -huge(1.0), CP_AIR = -huge(1.0), GRAV = -huge(1.0)
  real, public, parameter :: PI = 3.14159265358979323846
  real, public :: SECONDS_PER_DAY = 86400.0
  real, public :: orbital_period = 365.0*86400.0
  real, public :: stefan = -huge(1.0), solar_const = -huge(1.0)   ! top_down option only (not run)
end module constants_mod

module time_manager_mod
  implicit none
  type time_type
    integer :: s = 0, d = 0
  end type
contains
  subroutine get_time(t, seconds, days)
    type(time_type), intent(in) :: t
    integer, intent(out) :: seconds
    integer, intent(out), optional :: days
    seconds = t%s
    if (present(days)) days = t%d
  end subroutine
end module time_manager_mod

module transforms_mod
  implicit none
  type domain2d
    integer :: dummy = 0
  end type
  type(domain2d), save :: grid_domain
  integer, save :: g_is = 1, g_ie = 1, g_js = 1, g_je = 1
contains
  subroutine get_grid_domain(is, ie, js, je)
    integer, intent(out) :: is, ie, js, je
    is = g_is; ie = g_ie; js = g_js; je = g_je
  end subroutine
end module transforms_mod

module fms_mod
  use transforms_mod, only: domain2d
  implicit none
  integer, parameter :: FATAL=2, WARNING=1, NOTE=0
contains
  integer function open_namelist_file()
    integer :: u
    open(newunit=u, file='input.nml', status='old', action='read')
    open_namelist_file = u
  end function
  logical function file_exist(f)
    character(len=*), intent(in) :: f
    inquire(file=f, exist=file_exist)
  end function
  subroutine close_file(u)
    integer, intent(in) :: u
    close(u)
  end subroutine
  integer function mpp_pe()
    mpp_pe = 0
  end function
  integer function mpp_root_pe()
    mpp_root_pe = 0
  end function
  integer function stdlog()
    logical, save :: opened = .false.
    integer, save :: u = -1
    if (.not. opened) then
      open(newunit=u, file='logfile.out', status='replace', action='write')
      opened = .true.
    end if
    stdlog = u
  end function
  subroutine write_version_number(v, t)
    character(len=*), intent(in) :: v, t
  end subroutine
  subroutine set_domain(d)
    type(domain2d), intent(in) :: d
  end subroutine
  subroutine error_mesg(a, b, lvl)
    character(len=*), intent(in) :: a, b
    integer, intent(in) :: lvl
    write(0,*) 'ISCA_MSG ', trim(a), ': ', trim(b)
    if (lvl == FATAL) stop 'ISCA FATAL'
  end subroutine
  integer function check_nml_error(io, name)
    integer, intent(in) :: io
    character(len=*), intent(in) :: name
    check_nml_error = io
    if (io > 0) then
      write(0,*) 'NML ERROR ', name, io
      stop 'NML'
    end if
  end function
  function uppercase(s) result(u)
    character(len=*), intent(in) :: s
    character(len=len(s)) :: u
    integer :: i, c
    u = s
    do i = 1, len(s)
      c = ichar(s(i:i))
      if (c >= 97 .and. c <= 122) u(i:i) = char(c-32)
    end do
  end function
  subroutine read_data(f, n, a, d)
    character(len=*), intent(in) :: f, n
    real, intent(inout) :: a(:,:)
    type(domain2d), intent(in) :: d
    stop 'stub read_data (top_down option not run)'
  end subroutine
  subroutine write_data(f, n, a, d)
    character(len=*), intent(in) :: f, n
    real, intent(in) :: a(:,:)
    type(domain2d), intent(in) :: d
    stop 'stub write_data (top_down option not run)'
  end subroutine
end module fms_mod

module diag_manager_mod
  use time_manager_mod, only: time_type
  implicit none
  interface send_data
    module procedure send2, send3
  end interface
contains
  integer function register_diag_field(m, n, axes, t, l, u, missing_value, range)
    character(len=*), intent(in) :: m, n, l, u
    integer, intent(in) :: axes(:)
    type(time_type), intent(in) :: t
    real, intent(in), optional :: missing_value, range(2)
    register_diag_field = 0
  end function
  logical function send2(id, a, t)
    integer, intent(in) :: id
    real, intent(in) :: a(:,:)
    type(time_type), intent(in) :: t
    send2 = .true.
  end function
  logical function send3(id, a, t)
    integer, intent(in) :: id
    real, intent(in) :: a(:,:,:)
    type(time_type), intent(in) :: t
    send3 = .true.
  end function
end module diag_manager_mod

module field_manager_mod
  implicit none
  integer, parameter :: MODEL_ATMOS = 1
contains
  integer function parse(text, label, value)
    character(len=*), intent(in) :: text, label
    real, intent(out) :: value
    stop 'stub parse (no tracers)'
  end function
end module field_manager_mod

module tracer_manager_mod
  implicit none
contains
  subroutine get_number_tracers(model, num_tracers)
    integer, intent(in) :: model
    integer, intent(out), optional :: num_tracers
    if (present(num_tracers)) num_tracers = 0
  end subroutine
  logical function query_method(m, model, n, name, control)
    character(len=*), intent(in) :: m
    integer, intent(in) :: model, n
    character(len=*), intent(out) :: name, control
    stop 'stub query_method (no tracers)'
  end function
end module tracer_manager_mod

module interpolator_mod
  implicit none
  integer, parameter :: CONSTANT = 1, INTERP_WEIGHTED_P = 10
  type interpolate_type
    integer :: dummy = 0
  end type
contains
  subroutine interpolator_init(c, f, lonb, latb, data_out_of_bounds)
    type(interpolate_type), intent(inout) :: c
    character(len=*), intent(in) :: f
    real, intent(in) :: lonb(:,:), latb(:,:)
    integer, intent(in) :: data_out_of_bounds(:)
    stop 'stub interpolator_init (file options not run)'
  end subroutine
  subroutine interpolator(c, p, x, name)
    type(interpolate_type), intent(inout) :: c
    real, intent(in) :: p(:,:,:)
    real, intent(out) :: x(:,:,:)
    character(len=*), intent(in) :: name
    stop 'stub interpolator'
  end subroutine
  subroutine interpolator_end(c)
    type(interpolate_type), intent(inout) :: c
  end subroutine
end module interpolator_mod

module astronomy_mod
  use time_manager_mod, only: time_type
  implicit none
  real, public :: obliq = 0.0, ecc = 0.0   ! read only by the top_down option (not run)
contains
  subroutine astronomy_init()
  end subroutine
  subroutine diurnal_exoplanet(lat, lon, Time, coszen, fracsun, rrsun)
    real, intent(in) :: lat(:,:), lon(:,:)
    type(time_type), intent(in) :: Time
    real, intent(out) :: coszen(:,:), fracsun(:,:), rrsun
    stop 'stub diurnal_exoplanet (exoplanet options not run)'
  end subroutine
end module astronomy_mod
