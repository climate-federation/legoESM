! Standalone driver: Isca v1.0 qe_moist_convection (SIMPLE_BETTS_MILLER) on columns.
! Input file: line 1 = ncol nlev sat_mode dt ; per column 4 records T(nlev) q(nlev)
! p_full(nlev) p_half(nlev+1)   (index 1 = model top, nlev = lowest level; q specific).
! input.nml in CWD sets qe_moist_convection_nml (tau_bm, rhbm).
program drv_sbm
  use qe_moist_convection_mod
  use sat_vapor_pres_mod, only: sat_mode
  implicit none
  integer :: ncol, nlev, i, u, v, kLZB
  real :: dt, cape1, cin1
  character(len=512) :: fin, fout
  real, allocatable :: T(:,:,:), q(:,:,:), pf(:,:,:), ph(:,:,:)
  real, allocatable :: rain(:,:), snow(:,:), klzbs(:,:), cape(:,:), cin(:,:), itq(:,:), itt(:,:)
  real, allocatable :: ddT(:,:,:), ddq(:,:,:), qref(:,:,:), Tref(:,:,:), Tp(:), rp(:)
  integer, allocatable :: flag(:,:)
  logical, allocatable :: coldT(:,:)
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), status='old')
  read(u,*) ncol, nlev, sat_mode, dt
  allocate(T(ncol,1,nlev), q(ncol,1,nlev), pf(ncol,1,nlev), ph(ncol,1,nlev+1))
  do i = 1, ncol
    read(u,*) T(i,1,:); read(u,*) q(i,1,:); read(u,*) pf(i,1,:); read(u,*) ph(i,1,:)
  end do
  close(u)
  allocate(rain(ncol,1), snow(ncol,1), klzbs(ncol,1), cape(ncol,1), cin(ncol,1), itq(ncol,1), itt(ncol,1))
  allocate(ddT(ncol,1,nlev), ddq(ncol,1,nlev), qref(ncol,1,nlev), Tref(ncol,1,nlev), flag(ncol,1), coldT(ncol,1))
  allocate(Tp(nlev), rp(nlev))
  coldT = .false.
  call qe_moist_convection_init()
  call qe_moist_convection(dt, T, q, pf, ph, coldT, rain, snow, ddT, ddq, qref, flag, &
       klzbs, cape, cin, itq, itt, Tref)
  open(newunit=v, file=trim(fout), status='replace')
  write(v,'(a,2i6)') 'NCOL_NLEV', ncol, nlev
  do i = 1, ncol
    call qe_diag_parcel(pf(i,1,:), ph(i,1,:), T(i,1,:), q(i,1,:)/(1.0 - q(i,1,:)), kLZB, Tp, rp, cape1, cin1)
    write(v,'(a,i4,i4,i6,i6)') 'COL', i, flag(i,1), nint(klzbs(i,1)), kLZB
    write(v,'(*(es25.16e3,1x))') rain(i,1)/dt, cape(i,1), cin(i,1), cape1, cin1
    write(v,"(*(es25.16e3,1x))") ddT(i,1,:)/dt
    write(v,"(*(es25.16e3,1x))") ddq(i,1,:)/dt
    write(v,'(*(es25.16e3,1x))') Tref(i,1,:)
    write(v,'(*(es25.16e3,1x))') qref(i,1,:)
    write(v,'(*(es25.16e3,1x))') Tp
    write(v,'(*(es25.16e3,1x))') rp
  end do
  close(v)
  write(*,*) 'DRV_SBM_OK', ncol
end program drv_sbm
