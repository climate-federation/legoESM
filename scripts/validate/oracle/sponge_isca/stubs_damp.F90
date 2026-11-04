! Minimal FMS stubs so Isca v1.0 damping_driver.f90 compiles standalone, unmodified.
! No physics here.  constants_mod CP_AIR / GRAV are SET BY THE DRIVER from its input
! file (legoESM values written by make_fixture.py); the sentinels make an unset value
! fail loudly.  Every procedure the Rayleigh-only path does not reach stops.
module constants_mod
  implicit none
  real, public :: CP_AIR = -huge(1.0), GRAV = -huge(1.0)
  real, public, parameter :: PI = 3.14159265358979323846
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
    stop 'stub get_time (const_drag option not run)'
  end subroutine
  type(time_type) function length_of_year()
    stop 'stub length_of_year (const_drag option not run)'
  end function
end module time_manager_mod

module fms_mod
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
end module fms_mod

module diag_manager_mod
  use time_manager_mod, only: time_type
  implicit none
  interface send_data
    module procedure send2, send3
  end interface
contains
  integer function register_diag_field(m, n, axes, t, l, u, missing_value)
    character(len=*), intent(in) :: m, n, l, u
    integer, intent(in) :: axes(:)
    type(time_type), intent(in) :: t
    real, intent(in), optional :: missing_value
    register_diag_field = 0
  end function
  integer function register_static_field(m, n, axes, l, u)
    character(len=*), intent(in) :: m, n, l, u
    integer, intent(in) :: axes(:)
    stop 'stub register_static_field (mg_drag option not run)'
  end function
  logical function send2(id, a, t)
    integer, intent(in) :: id
    real, intent(in) :: a(:,:)
    type(time_type), intent(in) :: t
    stop 'stub send_data (all diag ids are 0)'
  end function
  logical function send3(id, a, t, rmask)
    integer, intent(in) :: id
    real, intent(in) :: a(:,:,:)
    type(time_type), intent(in) :: t
    real, intent(in), optional :: rmask(:,:,:)
    stop 'stub send_data (all diag ids are 0)'
  end function
end module diag_manager_mod

module mg_drag_mod
  implicit none
contains
  subroutine mg_drag(is, js, delt, u, v, t, pf, ph, zf, zh, dtx, dty, dtt, tbx, tby, tsf, kbot)
    integer, intent(in) :: is, js
    real, intent(in) :: delt
    real, intent(in), dimension(:,:,:) :: u, v, t, pf, ph, zf, zh
    real, intent(out), dimension(:,:,:) :: dtx, dty, dtt, tsf
    real, intent(out), dimension(:,:) :: tbx, tby
    integer, intent(in), optional, dimension(:,:) :: kbot
    stop 'stub mg_drag'
  end subroutine
  subroutine mg_drag_init(lonb, latb, hprime)
    real, intent(in) :: lonb(:), latb(:)
    real, intent(out) :: hprime(:,:)
    stop 'stub mg_drag_init'
  end subroutine
  subroutine mg_drag_end()
    stop 'stub mg_drag_end'
  end subroutine
end module mg_drag_mod

module cg_drag_mod
  use time_manager_mod, only: time_type
  implicit none
contains
  subroutine cg_drag_init(lonb, latb, pref, Time, axes)
    real, intent(in) :: lonb(:), latb(:), pref(:)
    type(time_type), intent(in) :: Time
    integer, intent(in) :: axes(4)
    stop 'stub cg_drag_init'
  end subroutine
  subroutine cg_drag_calc(is, js, lat, pf, zf, t, u, v, Time, delt, gx, gy)
    integer, intent(in) :: is, js
    real, intent(in) :: lat(:,:)
    real, intent(in), dimension(:,:,:) :: pf, zf, t, u, v
    type(time_type), intent(in) :: Time
    real, intent(in) :: delt
    real, intent(out), dimension(:,:,:) :: gx, gy
    stop 'stub cg_drag_calc'
  end subroutine
  subroutine cg_drag_end()
    stop 'stub cg_drag_end'
  end subroutine
end module cg_drag_mod

module topo_drag_mod
  implicit none
contains
  subroutine topo_drag(is, js, u, v, t, pf, ph, zf, zh, zpbl, tx, ty, dtx, dty, ts)
    integer, intent(in) :: is, js
    real, intent(in), dimension(:,:,:) :: u, v, t, pf, ph, zf, zh
    real, intent(in) :: zpbl(:,:)
    real, intent(out), dimension(:,:) :: tx, ty
    real, intent(out), dimension(:,:,:) :: dtx, dty, ts
    stop 'stub topo_drag'
  end subroutine
  subroutine topo_drag_init(lonb, latb, ierr)
    real, intent(in) :: lonb(:), latb(:)
    integer, intent(out) :: ierr
    stop 'stub topo_drag_init'
  end subroutine
  subroutine topo_drag_end()
    stop 'stub topo_drag_end'
  end subroutine
end module topo_drag_mod
