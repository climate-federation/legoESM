! Minimal FMS stubs so Isca v1.0 two_stream_gray_rad.F90 compiles standalone.
! No physics here; constants_mod values are set at run time by drv_gray.F90.
module constants_mod
  ! stefan / grav / cp_air / pstd_mks are SET BY THE DRIVER from its input file
  ! (the generator writes legoESM's constants there); the sentinels make an unset
  ! value fail loudly (negative radiative constants -> FPE / sign-wrong fluxes).
  implicit none
  real, public :: stefan = -huge(1.0)
  real, public :: grav   = -huge(1.0)
  real, public :: cp_air = -huge(1.0)
  real, public :: pstd_mks = -huge(1.0)
  real, public :: pstd_mks_earth = 101325.0 ! Isca value; read only by GEEN/BYRNE (not run)
  real, public :: seconds_per_sol = 86400.0
  real, public :: orbital_period = 365.0*86400.0
end module constants_mod

module fms_mod
  implicit none
  integer, parameter :: FATAL=2, WARNING=1, NOTE=0
contains
  integer function open_file(fname, action)
    character(len=*), intent(in) :: fname
    character(len=*), intent(in), optional :: action
    integer :: u
    if (present(action)) then
      if (action == 'read') then
        open(newunit=u, file=fname, status='old', action='read')
      else
        open(newunit=u, file=fname, position='append', action='write')
      end if
    else
      open(newunit=u, file=fname)
    end if
    open_file = u
  end function
  subroutine close_file(u)
    integer, intent(in) :: u
    close(u)
  end subroutine
  integer function mpp_pe()
    mpp_pe = 0
  end function
  subroutine error_mesg(a, b, lvl)
    character(len=*), intent(in) :: a, b
    integer, intent(in) :: lvl
    write(0,*) 'ISCA_MSG ', trim(a), ': ', trim(b)
    if (lvl == FATAL) stop 'ISCA FATAL'
  end subroutine
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
  integer function check_nml_error(io, name)
    integer, intent(in) :: io
    character(len=*), intent(in) :: name
    check_nml_error = io
    if (io > 0) then
      write(0,*) 'NML ERROR ', name, io
      stop 'NML'
    end if
  end function
end module fms_mod

module time_manager_mod
  implicit none
  type time_type
    integer :: s = 0, d = 0
  end type
  interface operator(+)
    module procedure tadd
  end interface
  interface operator(-)
    module procedure tsub
  end interface
  interface operator(/=)
    module procedure tne
  end interface
contains
  type(time_type) function tadd(a, b)
    type(time_type), intent(in) :: a, b
    tadd%s = a%s + b%s; tadd%d = a%d + b%d
  end function
  type(time_type) function tsub(a, b)
    type(time_type), intent(in) :: a, b
    tsub%s = a%s - b%s; tsub%d = a%d - b%d
  end function
  logical function tne(a, b)
    type(time_type), intent(in) :: a, b
    tne = (a%s /= b%s) .or. (a%d /= b%d)
  end function
  type(time_type) function length_of_year()
    length_of_year%d = 365
  end function
  real function length_of_day()
    length_of_day = 86400.0
  end function
  subroutine get_time(t, seconds, days)
    type(time_type), intent(in) :: t
    integer, intent(out) :: seconds
    integer, intent(out), optional :: days
    seconds = t%s
    if (present(days)) then
      days = t%d
    else
      seconds = t%s + 86400*t%d
    end if
  end subroutine
end module time_manager_mod

module astronomy_mod
  implicit none
contains
  subroutine astronomy_init()
  end subroutine
  subroutine diurnal_solar(lat, lon, gmt, time_since_ae, cosz, fracsun, rrsun, dt)
    real, intent(in) :: lat(:,:), lon(:,:), gmt, time_since_ae
    real, intent(out) :: cosz(:,:), fracsun(:,:), rrsun
    real, intent(in), optional :: dt
    stop 'stub diurnal_solar must not be reached (do_seasonal=.false.)'
  end subroutine
end module astronomy_mod

module interpolator_mod
  implicit none
  integer, parameter :: ZERO = 0
  type interpolate_type
    integer :: dummy = 0
  end type
contains
  subroutine interpolator_init(c, f, lonb, latb, data_out_of_bounds)
    type(interpolate_type), intent(inout) :: c
    character(len=*), intent(in) :: f
    real, intent(in) :: lonb(:,:), latb(:,:)
    integer, intent(in) :: data_out_of_bounds(:)
    stop 'stub interpolator_init (do_read_co2=.false.)'
  end subroutine
  subroutine interpolator(c, t, p, x, name)
    use time_manager_mod, only: time_type
    type(interpolate_type), intent(inout) :: c
    type(time_type), intent(in) :: t
    real, intent(in) :: p(:,:,:)
    real, intent(out) :: x(:,:,:)
    character(len=*), intent(in) :: name
    stop 'stub interpolator'
  end subroutine
  subroutine interpolator_end(c)
    type(interpolate_type), intent(inout) :: c
  end subroutine
end module interpolator_mod

module diag_manager_mod
  use time_manager_mod, only: time_type
  implicit none
  interface register_diag_field
    module procedure reg_arr, reg_scal
  end interface
  interface send_data
    module procedure send0, send2, send3
  end interface
contains
  integer function reg_arr(m, n, axes, t, l, u, missing_value)
    character(len=*), intent(in) :: m, n, l, u
    integer, intent(in) :: axes(:)
    type(time_type), intent(in) :: t
    real, intent(in), optional :: missing_value
    reg_arr = 0
  end function
  integer function reg_scal(m, n, t, l, u, missing_value)
    character(len=*), intent(in) :: m, n, l, u
    type(time_type), intent(in) :: t
    real, intent(in), optional :: missing_value
    reg_scal = 0
  end function
  logical function send0(id, a, t)
    integer, intent(in) :: id
    real, intent(in) :: a
    type(time_type), intent(in) :: t
    send0 = .true.
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
