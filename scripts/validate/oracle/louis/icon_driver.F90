! Driver: executes the VERBATIM LTG82 unstable block of ICON
! src/atm_phy_schemes/mo_turb_vdiff_diag.f90 lines 518-530 (extracted by sed into
! icon_frag.inc at build time; not hand-copied).  Constants cb, cc and cons5 = 3*cb*cc are
! taken from mo_turb_vdiff_params.f90:35-44 (cb = cc = 5).  km, kh enter as the neutral
! K = l^2*S fed from legoESM, so the outputs are K_m, K_h after the Louis-79 factor.
program icon_driver
  implicit none
  integer, parameter :: wp = selected_real_kind(13,300)
  integer :: n, i, jl, jk
  real(wp) :: cb, cc, cons5, zonethird, zri, lmix, zalh2, zucf, kneu
  real(wp) :: pghf(1,2), km(1,1), kh(1,1)
  character(len=512) :: fin, fout
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old'); open(20, file=trim(fout), status='replace')
  read(10,*) n, cb, cc
  cons5 = 3._wp*cb*cc; zonethird = 1._wp/3._wp
  jl = 1; jk = 1
  do i = 1, n
    read(10,*) zri, lmix, pghf(1,1), pghf(1,2), kneu
    km(1,1) = kneu; kh(1,1) = kneu
#include "icon_frag.inc"
    write(20,'(2es25.17e3)') km(1,1), kh(1,1)
  end do
  print *, 'ICON_DRIVER_DONE', n
end program icon_driver
