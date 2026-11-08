! Standalone driver for Isca v1.0 two_stream_gray_rad (B_FRIERSON).
! Calls the oracle's PUBLIC entry points two_stream_gray_rad_init/_down/_up
! exactly as Isca's idealised_moist_phys does (down -> up), and reads back the
! module flux arrays (made public by a visibility-only edit of a build copy; see
! build.sh).  Every float the oracle uses is read from argv(1); the namelist
! values are written by this driver to ./input.nml (es25.17e3 = exact double)
! before two_stream_gray_rad_init reads it.
!
! usage: drv_gray <in.txt> <out.txt>
! Input (list-directed):
!   ncol nlev
!   solar_constant del_sol del_sw sw_diff ir_tau_eq ir_tau_pole linear_tau
!     wv_exponent solar_exponent atm_abs                     (namelist)
!   stefan grav cp_air                                        (constants_mod)
!   per column: lat albedo t_surf pstd_mks
!               p_half(1:nlev+1)   (TOA first, Pa)
!               t(1:nlev)          (K)
! q is passed as 0 (B_FRIERSON does not read it).
program drv_gray
  use two_stream_gray_rad_mod
  use constants_mod, only: pstd_mks, stefan, grav, cp_air
  use time_manager_mod, only: time_type
  implicit none
  integer :: ncol, nlev, ic, axes(4), u
  real :: nml(10), lat_in, alb_in, ts_in, pstd_in
  real, allocatable :: ph(:,:,:), t(:,:,:), q(:,:,:), tdt(:,:,:)
  real :: lat(1,1), lon(1,1), alb(1,1), ts(1,1), nsd(1,1), sld(1,1)
  real :: lonb(2,2), latb(2,2)
  type(time_type) :: tm
  character(len=512) :: fin, fout
  character(len=*), parameter :: F = '(*(es25.17e3,1x))', E = 'es25.17e3'

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old', action='read')
  read(10,*) ncol, nlev
  read(10,*) nml
  read(10,*) stefan, grav, cp_air
  open(newunit=u, file='input.nml', status='replace', action='write')
  write(u,'(a)') "&two_stream_gray_rad_nml"
  write(u,'(a)') " rad_scheme='frierson', do_seasonal=.false.,"
  write(u,'(a,'//E//',a)') " solar_constant=", nml(1), ","
  write(u,'(a,'//E//',a)') " del_sol=", nml(2), ","
  write(u,'(a,'//E//',a)') " del_sw=", nml(3), ","
  write(u,'(a,'//E//',a)') " sw_diff=", nml(4), ","
  write(u,'(a,'//E//',a)') " ir_tau_eq=", nml(5), ","
  write(u,'(a,'//E//',a)') " ir_tau_pole=", nml(6), ","
  write(u,'(a,'//E//',a)') " linear_tau=", nml(7), ","
  write(u,'(a,'//E//',a)') " wv_exponent=", nml(8), ","
  write(u,'(a,'//E//',a)') " solar_exponent=", nml(9), ","
  write(u,'(a,'//E//')') " atm_abs=", nml(10)
  write(u,'(a)') "/"
  close(u)
  allocate(ph(1,1,nlev+1), t(1,1,nlev), q(1,1,nlev), tdt(1,1,nlev))
  q = 0.0
  axes = (/1,2,3,4/); lonb = 0.; latb = 0.; lon = 0.
  call two_stream_gray_rad_init(1, 1, 1, 1, nlev, axes, tm, lonb, latb, 1800.0)
  open(20, file=trim(fout), status='replace', action='write')
  do ic = 1, ncol
    read(10,*) lat_in, alb_in, ts_in, pstd_in
    read(10,*) ph(1,1,:)
    read(10,*) t(1,1,:)
    lat = lat_in; alb = alb_in; ts = ts_in
    pstd_mks = pstd_in
    call two_stream_gray_rad_down(1, 1, tm, lat, lon, ph, t, nsd, sld, alb, q)
    tdt = 0.0
    call two_stream_gray_rad_up(1, 1, tm, lat, ph, ts, t, tdt, alb)
    write(20,'(a,i4)') 'COL', ic
    write(20,F) insolation(1,1), nsd(1,1), sld(1,1), olr(1,1), net_lw_surf(1,1)
    write(20,F) lw_up(1,1,:)
    write(20,F) lw_down(1,1,:)
    write(20,F) sw_up(1,1,:)
    write(20,F) sw_down(1,1,:)
    write(20,F) tdt_rad(1,1,:)
    write(20,F) tdt_solar(1,1,:)
  end do
  close(20)
  close(10)
  write(*,*) 'GRAY_DRV_OK', ncol
end program drv_gray
