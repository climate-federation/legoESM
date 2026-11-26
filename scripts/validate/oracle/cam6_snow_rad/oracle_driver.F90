! CAM6 snow-for-radiation oracle driver (tests/atmosphere/oracle_fixtures/cam6_snow_rad.npz).
!
! The executable statements are VERBATIM line ranges of CAM cam_cesm2_1_rel_60,
! pulled in by #include (extracted by build.sh):
!   ci_concld.inc   = clubb_intr.F90 L2512   concld = min(cloud_frac-alst+deepcu, 0.8)
!   mgcam_snow.inc  = micro_mg_cam.F90 L2309-2321  cldfsnow and icswp
!   mgcam_des.inc   = micro_mg_cam.F90 L2576-2583  snow diameter (avg_diameter)
!   mgcam_desum.inc = micro_mg_cam.F90 L2639       des in micron
! micro_mg_utils.F90 (avg_diameter, rhosn, rhows, mincld) is compiled unmodified.
! cloud_frac = alst at L2512 (CLUBB's fraction before the ast assembly), so
! concld = min(deepcu, 0.8).  gravit is legoESM's g, so the comparison tests
! the logic, not the constant.
!
! usage: oracle_driver <in.bin> <out.bin>
!   in.bin : int32 n, then float64 cld deepcu qliq qs ns rho pdel (n each)
!   out.bin: float64 cldfsnow icswp des (n each)
program snow_oracle
  use micro_mg_utils, only: r8, mincld, rhosn, rhows, avg_diameter
  implicit none
  integer, parameter :: ixcldliq = 1
  real(r8), parameter :: gravit = GRAVIT_VAL
  integer :: n, i, k, u, ngrdcol, top_lev
  character(len=512) :: fin, fout
  real(r8), allocatable :: cld(:,:), concld(:,:), qsout(:,:), cldfsnow(:,:), icswp(:,:)
  real(r8), allocatable :: cloud_frac(:,:), alst(:,:), deepcu(:,:)
  real(r8), allocatable :: qs_grid(:,:), ns_grid(:,:), rho_grid(:,:), dsout2_grid(:,:), des_grid(:,:)
  real(r8), allocatable :: a(:,:)
  type :: state_t
     real(r8), allocatable :: q(:,:,:)
     real(r8), allocatable :: pdel(:,:)
  end type state_t
  type(state_t) :: state_loc

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) n
  allocate(a(n, 7))
  read(u) a
  close(u)
  ! one grid column per point, one level (ngrdcol = n, pver = top_lev = 1)
  ngrdcol = n
  top_lev = 1
  allocate(cld(n,1), concld(n,1), qsout(n,1), cldfsnow(n,1), icswp(n,1))
  allocate(cloud_frac(n,1), alst(n,1), deepcu(n,1))
  allocate(qs_grid(n,1), ns_grid(n,1), rho_grid(n,1), dsout2_grid(n,1), des_grid(n,1))
  allocate(state_loc%q(n,1,1), state_loc%pdel(n,1))
  cld(:,1) = a(:,1)
  deepcu(:,1) = a(:,2)
  state_loc%q(:,1,ixcldliq) = a(:,3)
  qsout(:,1) = a(:,4)
  qs_grid(:,1) = a(:,4)
  ns_grid(:,1) = a(:,5)
  rho_grid(:,1) = a(:,6)
  state_loc%pdel(:,1) = a(:,7)
  alst = 0.3_r8
  cloud_frac = alst
  k = 1
  do i = 1, n
#include "ci_concld.inc"
  end do
  do i = 1, n
#include "mgcam_snow.inc"
  end do
  dsout2_grid = 0._r8
  des_grid = 0._r8
#include "mgcam_des.inc"
  do i = 1, n
#include "mgcam_desum.inc"
  end do
  open(newunit=u, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(u) cldfsnow(:,1), icswp(:,1), des_grid(:,1)
  close(u)
end program snow_oracle
