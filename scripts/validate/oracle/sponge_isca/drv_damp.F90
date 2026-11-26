! Standalone driver for Isca v1.0 damping_driver.f90 (compiled UNMODIFIED with
! stubs_damp.F90).  Calls the PUBLIC damping_driver_init / damping_driver with only
! do_rayleigh and do_conserve_energy on, ONE column per init (Isca's pref is one
! profile for the whole domain; ours is per column, so each column gets its own
! init), and reports udt / vdt / tdt (the Rayleigh sponge tendencies:
! damping_driver adds them to the zero-initialised udt / vdt / tdt).
! Every float the path reads is in argv(1); namelist values are written by this
! driver to ./input.nml (es25.17e3 = exact double) before each init reads it.
!
! usage: drv_damp <in.txt> <out.txt>
! Input (list-directed):
!   ncol nlev
!   trayfric sponge_pbottom dt   (trayfric [s], > 0 branch: rfactr = 1/trayfric)
!   cp_air grav                  (constants_mod)
!   per column: pref(1:nlev+1)   (full-level reference pressure, then PSTD; Pa)
!               p_half(1:nlev+1) (TOA first, Pa)
!               p_full(1:nlev), u(1:nlev), v(1:nlev), t(1:nlev)
program drv_damp
  use damping_driver_mod, only: damping_driver_init, damping_driver
  use constants_mod, only: CP_AIR, GRAV
  use time_manager_mod, only: time_type
  implicit none
  integer :: ncol, nlev, i, u, axes(4)
  real :: trayfric, pbot, dt
  real, allocatable :: pref(:), ph(:,:,:), pf(:,:,:), uu(:,:,:), vv(:,:,:), tt(:,:,:), q(:,:,:)
  real, allocatable :: udt(:,:,:), vdt(:,:,:), tdt(:,:,:), qdt(:,:,:), zf(:,:,:), zh(:,:,:)
  real, allocatable :: r(:,:,:,:), rdt(:,:,:,:), lat(:,:), zpbl(:,:), sgs(:,:)
  real :: lonb(2), latb(2)
  type(time_type) :: tm
  character(len=512) :: fin, fout
  character(len=*), parameter :: F = '(*(es25.17e3,1x))', E = 'es25.17e3'

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old', action='read')
  read(10,*) ncol, nlev
  read(10,*) trayfric, pbot, dt
  read(10,*) CP_AIR, GRAV
  allocate(pref(nlev+1), ph(1,1,nlev+1), pf(1,1,nlev), uu(1,1,nlev), vv(1,1,nlev), &
           tt(1,1,nlev), q(1,1,nlev), udt(1,1,nlev), vdt(1,1,nlev), tdt(1,1,nlev), &
           qdt(1,1,nlev), zf(1,1,nlev), zh(1,1,nlev+1), r(1,1,nlev,0), rdt(1,1,nlev,0), &
           lat(1,1), zpbl(1,1), sgs(1,1))
  open(newunit=u, file='input.nml', status='replace', action='write')
  write(u,'(a)') "&damping_driver_nml"
  write(u,'(a,'//E//',a)') " trayfric=", trayfric, ","
  write(u,'(a,'//E//',a)') " sponge_pbottom=", pbot, ","
  write(u,'(a)') " do_rayleigh=.true., do_conserve_energy=.true."
  write(u,'(a)') "/"
  close(u)
  axes = (/1,2,3,4/); lonb = 0.; latb = 0.; zf = 0.; zh = 0.; q = 0.; lat = 0.; zpbl = 0.
  open(20, file=trim(fout), status='replace', action='write')
  do i = 1, ncol
    read(10,*) pref
    read(10,*) ph(1,1,:)
    read(10,*) pf(1,1,:)
    read(10,*) uu(1,1,:)
    read(10,*) vv(1,1,:)
    read(10,*) tt(1,1,:)
    call damping_driver_init(lonb, latb, pref, axes, tm, sgs)
    udt = 0.; vdt = 0.; tdt = 0.; qdt = 0.
    call damping_driver(1, 1, lat, tm, dt, pf, ph, zf, zh, uu, vv, tt, q, r, &
                        udt, vdt, tdt, qdt, rdt, zpbl)
    write(20,'(a,i4)') 'COL', i
    write(20,F) udt(1,1,:)
    write(20,F) vdt(1,1,:)
    write(20,F) tdt(1,1,:)
  end do
  close(10)
  close(20)
  write(*,*) 'DAMP_DRV_OK', ncol
end program drv_damp
