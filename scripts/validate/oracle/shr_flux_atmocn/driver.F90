! Standalone driver for cime5.6.47 shr_flux_atmOcn.
! usage: driver <consts:lego|shr> <max_iter> <tol> <coa:0|1> <infile> <outfile>
! infile: first line N; then N lines: zbot ubot vbot thbot qbot rbot tbot ts
program drv
  use shr_kind_mod, only: R8 => SHR_KIND_R8, IN => SHR_KIND_IN
  use shr_flux_mod, only: shr_flux_atmOcn, shr_flux_adjust_constants
  implicit none
  character(len=256) :: a_c, a_it, a_tol, a_coa, fin, fout
  integer :: n, i, maxit, icoa
  real(R8) :: tol
  real(R8), allocatable :: zbot(:),ubot(:),vbot(:),thbot(:),qbot(:),rbot(:),tbot(:),ts(:)
  real(R8), allocatable :: zero(:), one(:), us(:), vs(:), prec(:)
  real(R8), allocatable :: sen(:),lat(:),lwup(:),evap(:),e16(:),ehdo(:),e18(:),taux(:),tauy(:)
  real(R8), allocatable :: tref(:),qref(:),duu10n(:),ustar(:),re(:),ssq(:)
  integer(IN), allocatable :: mask(:)
  ! legoESM constants (packages/core/legoesm/constants.py; make_fixture.py asserts they agree)
  real(R8), parameter :: L_RD=287.05_R8, L_RV=461.51_R8, L_CPD=1004.64_R8, L_CPV=1846.0_R8
  real(R8), parameter :: L_G=9.80616_R8, L_KARMAN=0.4_R8, L_LV=2.501e6_R8
  call get_command_argument(1,a_c); call get_command_argument(2,a_it)
  call get_command_argument(3,a_tol); call get_command_argument(4,a_coa)
  call get_command_argument(5,fin); call get_command_argument(6,fout)
  read(a_it,*) maxit; read(a_tol,*) tol; read(a_coa,*) icoa
  if (trim(a_c) == 'lego') then
     call shr_flux_adjust_constants(zvir=L_RV/L_RD-1.0_R8, cpair=L_CPD, cpvir=L_CPV/L_CPD-1.0_R8, &
          karman=L_KARMAN, gravit=L_G, latvap=L_LV, flux_convergence_tolerance=tol, &
          flux_convergence_max_iteration=maxit, coldair_outbreak_mod=(icoa==1))
  else if (trim(a_c) == 'shr') then
     call shr_flux_adjust_constants(flux_convergence_tolerance=tol, &
          flux_convergence_max_iteration=maxit, coldair_outbreak_mod=(icoa==1))
  else
     stop 'bad consts'
  end if
  open(10,file=trim(fin),status='old'); read(10,*) n
  allocate(zbot(n),ubot(n),vbot(n),thbot(n),qbot(n),rbot(n),tbot(n),ts(n),zero(n),one(n),us(n),vs(n),prec(n))
  allocate(sen(n),lat(n),lwup(n),evap(n),e16(n),ehdo(n),e18(n),taux(n),tauy(n),tref(n),qref(n),duu10n(n),ustar(n),re(n),ssq(n),mask(n))
  do i=1,n
     read(10,*) zbot(i),ubot(i),vbot(i),thbot(i),qbot(i),rbot(i),tbot(i),ts(i)
  end do
  close(10)
  zero=0.0_R8; one=1.0_R8; us=0.0_R8; vs=0.0_R8; prec=0.0_R8; mask=1
  ! seq_flux_atmocn_minwind = 0.5 (CESM2 default), gust_fac = 0 (CESM2.1 default)
  call shr_flux_atmOcn(n, zbot, ubot, vbot, thbot, prec, 0.0_R8, &
       qbot, zero, zero, zero, rbot, tbot, us, vs, ts, mask, 0.5_R8, &
       sen, lat, lwup, one, one, one, evap, e16, ehdo, e18, &
       taux, tauy, tref, qref, duu10n, ustar, re, ssq)
  open(11,file=trim(fout),status='replace')
  write(11,'(a)') '# sen lat lwup evap taux tauy tref qref duu10n ustar re ssq'
  do i=1,n
     write(11,'(12es26.17e3)') sen(i),lat(i),lwup(i),evap(i),taux(i),tauy(i),tref(i),qref(i),duu10n(i),ustar(i),re(i),ssq(i)
  end do
  close(11)
  write(*,*) 'ORACLE_OK n=',n,' consts=',trim(a_c),' maxit=',maxit,' tol=',tol,' coa=',icoa
end program drv
