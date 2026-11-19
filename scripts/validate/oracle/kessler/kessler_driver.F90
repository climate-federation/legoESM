! Runs the UNMODIFIED ESCOMP atmos_phys0_31_000 kessler_run on columns read
! from a text file (levels top-down, k=1 = model top; lyr_surf=nz, lyr_toa=1).
! usage: kessler_driver <infile> <outfile>
! input : ncol nz dt cpair rair lv pref_Pa rhoqr
!         then per column nz lines: rho z pk theta qv qc qr
! output: per column: precl [m_water/s], then nz lines: theta qv qc qr relhum
program kessler_driver
  use ccpp_kinds, only: kind_phys
  use kessler, only: kessler_init, kessler_run
  implicit none
  character(len=512) :: fin, fout, errmsg
  character(len=64) :: sname
  integer :: ncol, nz, col, k, errflg
  real(kind_phys) :: dt, cpair, rair, lv, pref, rhoqr
  real(kind_phys), allocatable :: rho(:,:), z(:,:), pk(:,:), th(:,:), qv(:,:), &
       qc(:,:), qr(:,:), cp2(:,:), r2(:,:), precl(:), relhum(:,:)
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(10, file=fin, status='old'); read(10,*) ncol, nz, dt, cpair, rair, lv, pref, rhoqr
  allocate(rho(ncol,nz), z(ncol,nz), pk(ncol,nz), th(ncol,nz), qv(ncol,nz), &
       qc(ncol,nz), qr(ncol,nz), cp2(ncol,nz), r2(ncol,nz), precl(ncol), relhum(ncol,nz))
  do col = 1, ncol
    do k = 1, nz
      read(10,*) rho(col,k), z(col,k), pk(col,k), th(col,k), qv(col,k), qc(col,k), qr(col,k)
    end do
  end do
  close(10)
  cp2 = cpair; r2 = rair
  call kessler_init(lv, pref, rhoqr, errmsg, errflg)
  call kessler_run(ncol, nz, dt, nz, 1, cp2, r2, rho, z, pk, th, qv, qc, qr, &
       precl, relhum, sname, errmsg, errflg)
  if (errflg /= 0) then; print *, 'ERR ', trim(errmsg); stop 2; end if
  open(11, file=fout, status='replace')
  do col = 1, ncol
    write(11,'(ES26.17E3)') precl(col)
    do k = 1, nz
      write(11,'(5ES26.17E3)') th(col,k), qv(col,k), qc(col,k), qr(col,k), relhum(col,k)
    end do
  end do
  close(11)
  print *, 'KESSLER_DRIVER_OK', ncol, nz
end program kessler_driver
