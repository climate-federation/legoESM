! Standalone driver for Isca v1.0 hs_forcing.F90 (compiled UNMODIFIED with stubs_hs.F90).
! Calls the PUBLIC hs_forcing_init and hs_forcing as Isca's atmosphere does, with all
! columns along the i dimension (j = 1), no tracers, and reports udt / vdt (= the
! Rayleigh-damping tendencies: hs_forcing adds them to the zero-initialised udt / vdt)
! and tdt (Newtonian damping + dissipative heating; recorded, not compared).
! Every float hs_forcing reads is in argv(1); the namelist values are written by this
! driver to ./input.nml (es25.17e3 = exact double) before hs_forcing_init reads it.
!
! usage: drv_hs <in.txt> <out.txt>
! Input (list-directed):
!   ncol nlev
!   sigma_b kf dt               (namelist sigma_b, kf [1/s, > 0 branch]; dt [s])
!   kappa cp_air grav           (constants_mod)
!   per column: lat lon
!               p_half(1:nlev+1) (TOA first, Pa)
!               p_full(1:nlev), u(1:nlev), v(1:nlev), t(1:nlev)
program drv_hs
  use hs_forcing_mod, only: hs_forcing_init, hs_forcing
  use constants_mod, only: KAPPA, CP_AIR, GRAV
  use time_manager_mod, only: time_type
  use transforms_mod, only: g_ie
  implicit none
  integer :: ncol, nlev, i, u, axes(4)
  real :: sigma_b, kf, dt
  real, allocatable :: lat(:,:), lon(:,:), ph(:,:,:), pf(:,:,:), uu(:,:,:), vv(:,:,:), tt(:,:,:)
  real, allocatable :: udt(:,:,:), vdt(:,:,:), tdt(:,:,:), zf(:,:,:), r(:,:,:,:), rdt(:,:,:,:)
  real, allocatable :: lonb(:,:), latb(:,:)
  type(time_type) :: tm
  character(len=512) :: fin, fout
  character(len=*), parameter :: F = '(*(es25.17e3,1x))', E = 'es25.17e3'

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old', action='read')
  read(10,*) ncol, nlev
  read(10,*) sigma_b, kf, dt
  read(10,*) KAPPA, CP_AIR, GRAV
  allocate(lat(ncol,1), lon(ncol,1), ph(ncol,1,nlev+1), pf(ncol,1,nlev), uu(ncol,1,nlev), &
           vv(ncol,1,nlev), tt(ncol,1,nlev), udt(ncol,1,nlev), vdt(ncol,1,nlev), tdt(ncol,1,nlev), &
           zf(ncol,1,nlev), r(ncol,1,nlev,0), rdt(ncol,1,nlev,0), lonb(ncol+1,2), latb(ncol+1,2))
  do i = 1, ncol
    read(10,*) lat(i,1), lon(i,1)
    read(10,*) ph(i,1,:)
    read(10,*) pf(i,1,:)
    read(10,*) uu(i,1,:)
    read(10,*) vv(i,1,:)
    read(10,*) tt(i,1,:)
  end do
  close(10)
  open(newunit=u, file='input.nml', status='replace', action='write')
  write(u,'(a)') "&hs_forcing_nml"
  write(u,'(a,'//E//',a)') " sigma_b=", sigma_b, ","
  write(u,'(a,'//E//')') " kf=", kf
  write(u,'(a)') "/"
  close(u)
  g_ie = ncol
  axes = (/1,2,3,4/); lonb = 0.; latb = 0.; zf = 0.
  call hs_forcing_init(axes, tm, lonb, latb, lat)
  udt = 0.; vdt = 0.; tdt = 0.
  call hs_forcing(1, ncol, 1, 1, dt, tm, lon, lat, ph, pf, uu, vv, tt, r, uu, vv, tt, r, &
                  udt, vdt, tdt, rdt, zf)
  open(20, file=trim(fout), status='replace', action='write')
  do i = 1, ncol
    write(20,'(a,i4)') 'COL', i
    write(20,F) udt(i,1,:)
    write(20,F) vdt(i,1,:)
    write(20,F) tdt(i,1,:)
  end do
  close(20)
  write(*,*) 'HS_DRV_OK', ncol
end program drv_hs
