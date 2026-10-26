! Standalone driver for gSAM 1.8.8 SRC/rad_simple.f90 (unmodified).
! Input simple_in.txt:  ncol ; per column: nzm dz ; adz(1:nzm) ; zi(1:nzm+1) ;
!   rho(1:nzm) ; rhow(1:nzm+1) ; qv(1:nzm) ; qcl(1:nzm) ; qci(1:nzm)   (bottom-up)
! Output simple_out.txt per column: radlwdn(1:nzm) = net upward LW flux at
!   faces 1..nzm (rad_simple's local flux(k)), qrad(1:nzm) = FTHRL [K/s].
program drv_simple
  use grid
  use vars
  use rad, only: qrad
  implicit none
  integer :: ncol, ic
  character(len=*), parameter :: F = '(*(es25.17e3,1x))'
  open(10, file='simple_in.txt', status='old', action='read')
  open(20, file='simple_out.txt', status='replace', action='write')
  read(10,*) ncol
  do ic = 1, ncol
    read(10,*) nzm, dz
    nz = nzm + 1
    if (allocated(adz)) deallocate(adz, zi, rho, rhow, radlwdn, radqrlw, qcl, qci, qv, t, qrad)
    allocate(adz(nzm), zi(nz), rho(nzm), rhow(nz), radlwdn(nz), radqrlw(nz))
    allocate(qcl(1,1,nzm), qci(1,1,nzm), qv(1,1,nzm), t(1,1,nzm), qrad(1,1,nzm))
    read(10,*) adz
    read(10,*) zi
    read(10,*) rho
    read(10,*) rhow
    read(10,*) qv(1,1,:)
    read(10,*) qcl(1,1,:)
    read(10,*) qci(1,1,:)
    t = 0.0
    radlwdn = 0.0; radqrlw = 0.0
    call rad_simple()
    write(20,'(a,i4)') 'COL', ic
    write(20,F) radlwdn(1:nzm)
    write(20,F) qrad(1,1,:)
  end do
  close(20)
  write(*,*) 'SIMPLE_DRV_OK', ncol
end program drv_simple
