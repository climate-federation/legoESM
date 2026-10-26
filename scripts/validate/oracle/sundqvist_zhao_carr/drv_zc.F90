! Driver: ccpp-physics v7.0.1 zhaocarr_precpd_run (unmodified).
! usage: drv_zc <infile> <outfile>
! infile line 1: ncol km dt grav hvap hfus ttp cp eps epsm1 prautco psautco evpco wminco_l wminco_i
!   then ncol*km lines (column-major over columns, levels BOTTOM->TOP): del prsl q cwm t u00k
! outfile: per column one line "rn sr", then km lines: q cwm t rainp (bottom->top)
program drv_zc
  use machine, only: kind_phys
  use zhaocarr_precpd, only: zhaocarr_precpd_run
  implicit none
  character(len=512) :: fin, fout
  integer :: ncol, km, i, k, errflg
  real(kind_phys) :: dt, grav, hvap, hfus, ttp, cp, eps, epsm1
  real(kind_phys) :: prautco(2), psautco(2), evpco, wminco(2)
  real(kind_phys), allocatable :: del(:,:), prsl(:,:), q(:,:), cwm(:,:), t(:,:), u00k(:,:), rainp(:,:)
  real(kind_phys), allocatable :: rn(:), sr(:), wk1(:)
  character(len=256) :: errmsg
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=fin, status='old')
  read(10,*) ncol, km, dt, grav, hvap, hfus, ttp, cp, eps, epsm1, prautco(1), psautco(1), evpco, wminco(1), wminco(2)
  prautco(2) = prautco(1); psautco(2) = psautco(1)
  allocate(del(ncol,km), prsl(ncol,km), q(ncol,km), cwm(ncol,km), t(ncol,km), u00k(ncol,km), rainp(ncol,km))
  allocate(rn(ncol), sr(ncol), wk1(ncol))
  do i = 1, ncol
    do k = 1, km
      read(10,*) del(i,k), prsl(i,k), q(i,k), cwm(i,k), t(i,k), u00k(i,k)
    end do
  end do
  close(10)
  wk1 = 1.0_kind_phys
  call zhaocarr_precpd_run(ncol, km, dt, del, prsl, q, cwm, t, rn, grav, hvap, hfus, ttp, cp, eps, epsm1, &
       sr, rainp, u00k, psautco, prautco, evpco, wminco, wk1, .false., 1, errmsg, errflg)
  if (errflg /= 0) then
    write(0,*) 'ZC ERR ', trim(errmsg); stop 1
  end if
  open(11, file=fout, status='replace')
  do i = 1, ncol
    write(11,'(2ES25.17)') rn(i), sr(i)
    do k = 1, km
      write(11,'(4ES25.17)') q(i,k), cwm(i,k), t(i,k), rainp(i,k)
    end do
  end do
  close(11)
end program
