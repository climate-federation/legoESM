! Driver: ccpp-physics v7.0.1 zhaocarr_gscond_run (unmodified), two-time-level (dtf = dt).
! usage: drv_gscond <infile> <outfile>
! infile line 1: ncol km dt hvap grav hfus ttp rd cp eps epsm1 rv
!   then per column one line "ps psp", then km lines (levels BOTTOM->TOP): prsl q cwm t tp qp u
! outfile: per column km lines: t q cwm tp qp (bottom->top)
program drv_gscond
  use machine, only: kind_phys
  use zhaocarr_gscond, only: zhaocarr_gscond_run
  implicit none
  character(len=512) :: fin, fout
  integer :: ncol, km, i, k, errflg
  real(kind_phys) :: dt, hvap, grav, hfus, ttp, rd, cp, eps, epsm1, rv, psat
  real(kind_phys), allocatable :: prsl(:,:), q(:,:), clw1(:,:), clw2(:,:), cwm(:,:), t(:,:)
  real(kind_phys), allocatable :: tp(:,:), qp(:,:), tp1(:,:), qp1(:,:), u(:,:)
  real(kind_phys), allocatable :: ps(:), psp(:), psp1(:)
  character(len=256) :: errmsg
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=fin, status='old')
  read(10,*) ncol, km, dt, hvap, grav, hfus, ttp, rd, cp, eps, epsm1, rv
  psat = 0.0_kind_phys   ! unused by zhaocarr_gscond_run
  allocate(prsl(ncol,km), q(ncol,km), clw1(ncol,km), clw2(ncol,km), cwm(ncol,km), t(ncol,km))
  allocate(tp(ncol,km), qp(ncol,km), tp1(ncol,km), qp1(ncol,km), u(ncol,km))
  allocate(ps(ncol), psp(ncol), psp1(ncol))
  do i = 1, ncol
    read(10,*) ps(i), psp(i)
    do k = 1, km
      read(10,*) prsl(i,k), q(i,k), clw1(i,k), t(i,k), tp(i,k), qp(i,k), u(i,k)
    end do
  end do
  close(10)
  clw2 = 0.0_kind_phys
  tp1 = tp; qp1 = qp; psp1 = psp
  call zhaocarr_gscond_run(ncol, km, dt, dt, prsl, ps, q, clw1, clw2, cwm, t, tp, qp, psp, &
       psat, hvap, grav, hfus, ttp, rd, cp, eps, epsm1, rv, tp1, qp1, psp1, u, .false., 1, &
       errmsg, errflg)
  if (errflg /= 0) then
    write(0,*) 'GSCOND ERR ', trim(errmsg); stop 1
  end if
  open(11, file=fout, status='replace')
  do i = 1, ncol
    do k = 1, km
      write(11,'(5ES26.17E3)') t(i,k), q(i,k), cwm(i,k), tp(i,k), qp(i,k)
    end do
  end do
  close(11)
end program
