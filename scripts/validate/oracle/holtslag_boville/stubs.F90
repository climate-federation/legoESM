! Minimal CAM/E3SM infrastructure stubs for the Holtslag-Boville oracle harness.
! Only what hb_diff.F90, pbl_utils.F90, coords_1d.F90, linear_1d_operators.F90,
! vdiff_lu_solver.F90 and diffusion_solver.F90 import.  No physics lives here;
! physical constants are passed through the oracle's own init routines.
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
  integer, parameter :: shr_kind_r4 = selected_real_kind(6)
  integer, parameter :: shr_kind_in = kind(1)
  integer, parameter :: shr_kind_cl = 256
  integer, parameter :: shr_kind_cs = 80
end module shr_kind_mod

module shr_log_mod
  implicit none
contains
  function shr_log_errMsg(file, line) result(msg)
    character(len=*), intent(in) :: file
    integer, intent(in) :: line
    character(len=512) :: msg
    write(msg, '(a,":",i0)') trim(file), line
  end function shr_log_errMsg
end module shr_log_mod

module shr_sys_mod
  implicit none
contains
  subroutine shr_sys_abort(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(0,*) 'ABORT: ', trim(msg)
    stop 99
  end subroutine shr_sys_abort
end module shr_sys_mod

module spmd_utils
  implicit none
  logical :: masterproc = .false.
end module spmd_utils

module cam_logfile
  implicit none
  integer :: iulog = 6
end module cam_logfile

module ppgrid
  implicit none
  ! L32 (CAM6 grid of the input columns); pcols >= number of columns
  integer, parameter :: pver = 32
  integer, parameter :: pverp = pver + 1
  integer, parameter :: pcols = 32
  integer, parameter :: begchunk = 1, endchunk = 1
end module ppgrid

module phys_debug
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols
  implicit none
contains
  subroutine phys_debug_hbdiff1(lchnk, pblh, zl, zh)
    integer,  intent(in) :: lchnk
    real(r8), intent(in) :: pblh(pcols), zl(pcols), zh(pcols)
  end subroutine phys_debug_hbdiff1
end module phys_debug

module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, begchunk, endchunk
  implicit none
  ! only rairv(:,1,lchnk) is read by compute_vdiff (top-interface dpidz_sq);
  ! the driver fills it with the run's rair.
  real(r8) :: rairv(pcols, pver, begchunk:endchunk)
end module physconst

module phys_control
  implicit none
contains
  subroutine phys_getopts(use_spcam_out)
    logical, intent(out), optional :: use_spcam_out
    if (present(use_spcam_out)) use_spcam_out = .false.
  end subroutine phys_getopts
end module phys_control

module beljaars_drag_cam
  implicit none
  ! CAM4/HB configuration: no Beljaars SGO drag -> the (tmpi1) KE-dissipation branch
  logical :: do_beljaars = .false.
end module beljaars_drag_cam
