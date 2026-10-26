! Standalone driver for the CAM6 (cam_cesm2_1_rel_60) OROGRAPHIC gravity-wave
! drag oracle.  It reproduces gw_drag.F90's gw_tend orographic branch on
! columns read from a text file:
!   gw_prof (gw_drag.F90:1369) -> landfrac scaling of effgw/sgh
!   (gw_drag.F90:1757-1764, gw_lndscl_sgh = .true.) -> gw_oro_src ->
!   gw_drag_prof(lapply_effgw_in = gw_apply_tndmax = .true.) with gw
!   vertical diffusion of dse and the constituents ON (gw_drag.F90:1781-1787).
! gw_oro_south_fac = 1 (namelist default) is a no-op and is not applied.
! The Newtonian-cooling profile alpha is gw_init's table (gw_drag.F90:420-519,
! copied verbatim below) interpolated onto pref_edge with CAM's REAL
! interpolate_data lininterp (extrap_method_bndry), exactly as gw_init does.
! kvtt = 0 (no molecular diffusion, gw_drag.F90:1394-1395).  Constants
! (gravit, rair, cpair) are READ from the input so the oracle runs with the
! caller's constants.  All gw_* / coords_1d / linear_1d_operators /
! vdiff_lu_solver / interpolate_data sources are compiled UNMODIFIED.
program gw_oro_driver
  use gw_utils, only: r8
  use coords_1d, only: Coords1D
  use gw_common, only: GWBand, gw_common_init, gw_prof, gw_drag_prof
  use gw_oro, only: gw_oro_src
  use interpolate_data, only: lininterp
  implicit none
  integer :: ncol, pver, ncnst, i, k, m, iu, ou
  real(r8) :: gravit, rair, cpair, dt, effgw_oro, fcrit2, prndl, wavelength
  real(r8), allocatable :: pint(:,:), t(:,:), u(:,:), v(:,:), zm(:,:)
  real(r8), allocatable :: sgh(:), lfrac(:), pref_edge(:), alpha(:)
  real(r8), allocatable :: rhoi(:,:), nm(:,:), ni(:,:), piln(:,:), dse(:,:)
  real(r8), allocatable :: q(:,:,:), kvtt(:,:), effgw(:), sgh_scaled(:)
  real(r8), allocatable :: tau(:,:,:), ubm(:,:), ubi(:,:), xv(:), yv(:), c(:,:)
  real(r8), allocatable :: utgw(:,:), vtgw(:,:), ttgw(:,:), qtgw(:,:,:)
  real(r8), allocatable :: egwdffi(:,:), gwut(:,:,:), dttdf(:,:), dttke(:,:)
  integer, allocatable :: src_level(:), tend_level(:)
  real(r8), pointer :: vramp(:) => null()
  type(Coords1D) :: p
  type(GWBand) :: band_oro
  character(len=512) :: errstring, fin, fout
  ! gw_init (gw_drag.F90:416-460): Newtonian cooling coefficients [1/day]
  ! and the pressure levels [hPa] they were computed on.
  integer, parameter :: nalph = 71
  real(r8) :: alpha0(nalph) = [ &
       0.1_r8,         0.1_r8,         0.1_r8,         0.1_r8,         &
       0.1_r8,         0.1_r8,         0.1_r8,         0.1_r8,         &
       0.1_r8,         0.1_r8,         0.10133333_r8,  0.104_r8,       &
       0.108_r8,       0.112_r8,       0.116_r8,       0.12066667_r8,  &
       0.126_r8,       0.132_r8,       0.138_r8,       0.144_r8,       &
       0.15133333_r8,  0.16_r8,        0.17_r8,        0.18_r8,        &
       0.19_r8,        0.19933333_r8,  0.208_r8,       0.216_r8,       &
       0.224_r8,       0.232_r8,       0.23466667_r8,  0.232_r8,       &
       0.224_r8,       0.216_r8,       0.208_r8,       0.20133333_r8,  &
       0.196_r8,       0.192_r8,       0.188_r8,       0.184_r8,       &
       0.18266667_r8,  0.184_r8,       0.188_r8,       0.192_r8,       &
       0.196_r8,       0.19333333_r8,  0.184_r8,       0.168_r8,       &
       0.152_r8,       0.136_r8,       0.12133333_r8,  0.108_r8,       &
       0.096_r8,       0.084_r8,       0.072_r8,       0.061_r8,       &
       0.051_r8,       0.042_r8,       0.033_r8,       0.024_r8,       &
       0.017666667_r8, 0.014_r8,       0.013_r8,       0.012_r8,       &
       0.011_r8,       0.010333333_r8, 0.01_r8,        0.01_r8,        &
       0.01_r8,        0.01_r8,        0.01_r8                         &
       ]
  real(r8) :: palph(nalph) = [ &
       2.06115E-06_r8, 2.74280E-06_r8, 3.64988E-06_r8, 4.85694E-06_r8, &
       6.46319E-06_r8, 8.60065E-06_r8, 1.14450E-05_r8, 1.52300E-05_r8, &
       2.02667E-05_r8, 2.69692E-05_r8, 3.58882E-05_r8, 4.77568E-05_r8, &
       6.35507E-05_r8, 8.45676E-05_r8, 0.000112535_r8, 0.000149752_r8, &
       0.000199277_r8, 0.000265180_r8, 0.000352878_r8, 0.000469579_r8, &
       0.000624875_r8, 0.000831529_r8, 0.00110653_r8,  0.00147247_r8,  &
       0.00195943_r8,  0.00260744_r8,  0.00346975_r8,  0.00461724_r8,  &
       0.00614421_r8,  0.00817618_r8,  0.0108801_r8,   0.0144783_r8,   &
       0.0192665_r8,   0.0256382_r8,   0.0341170_r8,   0.0453999_r8,   &
       0.0604142_r8,   0.0803939_r8,   0.106981_r8,    0.142361_r8,    &
       0.189442_r8,    0.252093_r8,    0.335463_r8,    0.446404_r8,    &
       0.594036_r8,    0.790490_r8,    1.05192_r8,     1.39980_r8,     &
       1.86273_r8,     2.47875_r8,     3.29851_r8,     4.38936_r8,     &
       5.84098_r8,     7.77266_r8,     10.3432_r8,     13.7638_r8,     &
       18.3156_r8,     24.3728_r8,     32.4332_r8,     43.1593_r8,     &
       57.4326_r8,     76.4263_r8,     101.701_r8,     135.335_r8,     &
       180.092_r8,     239.651_r8,     318.907_r8,     424.373_r8,     &
       564.718_r8,     751.477_r8,     1000._r8                        &
       ]

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old', action='read')
  read(iu,*) ncol, pver, ncnst
  read(iu,*) gravit, rair, cpair
  read(iu,*) dt, effgw_oro, fcrit2, prndl, wavelength
  allocate(pref_edge(pver+1), alpha(pver+1))
  read(iu,*) pref_edge
  allocate(pint(ncol,pver+1), t(ncol,pver), u(ncol,pver), v(ncol,pver), &
       zm(ncol,pver), q(ncol,pver,ncnst), sgh(ncol), lfrac(ncol))
  do i = 1, ncol
     read(iu,*) sgh(i), lfrac(i)
     read(iu,*) pint(i,:)
     read(iu,*) t(i,:)
     read(iu,*) u(i,:)
     read(iu,*) v(i,:)
     read(iu,*) zm(i,:)
     do m = 1, ncnst
        read(iu,*) q(i,:,m)
     end do
  end do
  close(iu)

  ! gw_init (gw_drag.F90:509-519) verbatim.
  do k = 1, nalph
     alpha0(k) = alpha0(k) / 86400._r8
     alpha0(k) = max(alpha0(k), 1.e-6_r8)
     palph(k) = palph(k)*1.e2_r8
  end do
  call lininterp (alpha0  ,palph, nalph , alpha  , pref_edge , pver+1)

  ! gw_init: ktop = 1 (no gw_top_taper / tau_0_ubc), qbo scaling 1.
  call gw_common_init(pver, .false., 1, gravit, rair, alpha, prndl, 1._r8, errstring)
  if (trim(errstring) /= "") stop 'gw_common_init'
  ! gw_init: band_oro = GWBand(0, gw_dc, fcrit2, wavelength_mid); dc unused for ngwv=0.
  band_oro = GWBand(0, 2.5_r8, fcrit2, wavelength)
  p = Coords1D(pint)
  allocate(rhoi(ncol,pver+1), nm(ncol,pver), ni(ncol,pver+1), piln(ncol,pver+1), &
       dse(ncol,pver), kvtt(ncol,pver+1), effgw(ncol), sgh_scaled(ncol))
  allocate(tau(ncol,0:0,pver+1), ubm(ncol,pver), ubi(ncol,pver+1), xv(ncol), yv(ncol), &
       c(ncol,0:0), utgw(ncol,pver), vtgw(ncol,pver), ttgw(ncol,pver), qtgw(ncol,pver,ncnst), &
       egwdffi(ncol,pver+1), gwut(ncol,pver,0:0), dttdf(ncol,pver), dttke(ncol,pver), &
       src_level(ncol), tend_level(ncol))
  piln = log(pint)
  kvtt = 0._r8
  ! state%s = cpair*T + gravit*zm + phis; phis is a per-column constant and
  ! the conservative diffusion annihilates it, so it is omitted here.
  dse = cpair*t + gravit*zm
  call gw_prof(ncol, p, cpair, t, rhoi, nm, ni)
  where (lfrac >= epsilon(1._r8))
     effgw = effgw_oro*lfrac
     sgh_scaled = sgh/sqrt(lfrac)
  elsewhere
     effgw = 0._r8
     sgh_scaled = 0._r8
  end where
  call gw_oro_src(ncol, band_oro, p, u, v, t, sgh_scaled, zm, nm, &
       src_level, tend_level, tau, ubm, ubi, xv, yv, c)
  call gw_drag_prof(ncol, band_oro, p, src_level, tend_level, dt, &
       t, vramp, piln, rhoi, nm, ni, ubm, ubi, xv, yv, &
       effgw, c, kvtt, q, dse, tau, utgw, vtgw, &
       ttgw, qtgw, egwdffi, gwut, dttdf, dttke, &
       lapply_effgw_in=.true.)

  open(newunit=ou, file=trim(fout), status='replace', action='write')
  write(ou,'(a,3i6)') '# ncol pver ncnst', ncol, pver, ncnst
  write(ou,'(a,100es25.16e3)') 'alpha', alpha
  do i = 1, ncol
     write(ou,'(a,i4,i6,4es25.16e3)') 'col', i, src_level(i), xv(i), yv(i), effgw(i), sgh_scaled(i)
     write(ou,'(a,100es25.16e3)') 'tau', tau(i,0,:)
     write(ou,'(a,100es25.16e3)') 'utgw', utgw(i,:)
     write(ou,'(a,100es25.16e3)') 'vtgw', vtgw(i,:)
     write(ou,'(a,100es25.16e3)') 'ttgw', ttgw(i,:)/cpair
     write(ou,'(a,100es25.16e3)') 'dttke', dttke(i,:)/cpair
     write(ou,'(a,100es25.16e3)') 'dttdf', dttdf(i,:)/cpair
     do m = 1, ncnst
        write(ou,'(a,i0,100es25.16e3)') 'qtgw', m, qtgw(i,:,m)
     end do
  end do
  close(ou)
  write(*,*) 'ORACLE_OK', ncol, pver, ncnst
end program gw_oro_driver
