! Standalone driver: GFDL atmos_phys 2021.02 moist_conv (Manabe moist convective
! adjustment) on columns.  Same input format as drv_sbm.  coldT=.false. (HL=HLv),
! no cloud tracers (beta detrainment path off), namelist defaults (HC=1,
! TOLmin=.02, TOLmax=.10, ITSMOD=30, do_simple=.false.).
program drv_mca
  use moist_conv_mod
  use sat_vapor_pres_mod, only: sat_mode
  use time_manager_mod, only: time_type
  use fms_mod, only: n_notes, n_warnings
  implicit none
  integer :: ncol, nlev, i, u, v, axes(4)
  real :: dt
  character(len=512) :: fin, fout
  type(time_type) :: Time
  real, allocatable :: T(:,:,:), q(:,:,:), pf(:,:,:), ph(:,:,:), T0(:,:,:), q0(:,:,:)
  real, allocatable :: Tdel(:,:,:), Qdel(:,:,:), rain(:,:), snow(:,:), tr(:,:,:,:), qtr(:,:,:,:)
  logical, allocatable :: coldT(:,:), conv(:,:,:)
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), status='old')
  read(u,*) ncol, nlev, sat_mode, dt
  allocate(T(ncol,1,nlev), q(ncol,1,nlev), pf(ncol,1,nlev), ph(ncol,1,nlev+1))
  do i = 1, ncol
    read(u,*) T(i,1,:); read(u,*) q(i,1,:); read(u,*) pf(i,1,:); read(u,*) ph(i,1,:)
  end do
  close(u)
  T0 = T; q0 = q
  allocate(Tdel(ncol,1,nlev), Qdel(ncol,1,nlev), rain(ncol,1), snow(ncol,1))
  allocate(tr(ncol,1,nlev,1), qtr(ncol,1,nlev,1), coldT(ncol,1), conv(ncol,1,nlev))
  tr = 0.; coldT = .false.; axes = 1
  call moist_conv_init(axes, Time, (/ .false. /))
  call moist_conv(T, q, pf, ph, coldT, Tdel, Qdel, rain, snow, 1.0/dt, Time, 1, 1, tr, qtr, Conv=conv)
  open(newunit=v, file=trim(fout), status='replace')
  write(v,'(a,2i6,2i8)') 'NCOL_NLEV', ncol, nlev, n_notes, n_warnings
  do i = 1, ncol
    write(v,'(a,i4,i6)') 'COL', i, count(conv(i,1,:))
    write(v,'(*(es25.16e3,1x))') rain(i,1), snow(i,1)
    write(v,'(*(es25.16e3,1x))') Tdel(i,1,:)
    write(v,'(*(es25.16e3,1x))') Qdel(i,1,:)
    write(v,'(*(i2,1x))') merge(1, 0, conv(i,1,:))
  end do
  close(v)
  write(*,*) 'DRV_MCA_OK', ncol, ' notes(ALTOL doubled)=', n_notes, ' warnings=', n_warnings
end program drv_mca
