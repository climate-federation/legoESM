! Per-process oracle driver for the tier-1 MG2 ice follow-ups (oracle rows
! 26, 30, 31): the cloud-ice slope and its bounds, the Ferrier ice->snow
! autoconversion and snow riming of droplets.  Calls the UNMODIFIED
! micro_mg_utils.F90 routines (CAM cam_cesm2_1_rel_60) the way micro_mg_tend
! calls them; the dv/mu/rhof/asn and in-precip snow lines are copied verbatim
! from micro_mg2_0.F90 (:933-944, :1358-1375, :1403-1413), as in
! ../morrison_mg2_ice_snow/mg2_snow_procs.F90.
!
! Point input (first line n, then one line per point):
!   T p rho qiic niic[#/kg] qcic ncic[#/kg] qsic nsic[#/kg]   (in-cloud)
! Output (one line per point, 9 columns):
!   niic lami n0i prci nprci pgam lamc psacws npsacws
! usage: mg2_ice_rest_procs points.txt out.txt dcs
program mg2_ice_rest_procs
  use micro_mg_utils, only: r8, as, bs, mg_liq_props, mg_snow_props, mg_ice_props, &
       size_dist_param_liq, size_dist_param_basic, accrete_cloud_water_snow, &
       ice_autoconversion
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: n, i
  character(len=512) :: fin, fout, adcs
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8) :: dcs, rhosu
  real(r8), allocatable, dimension(:) :: t, p, rho, qiic, niic, qcic, ncic, qsic, nsic, &
       dv, mu, rhof, asn, lami, n0i, prci, nprci, pgam, lamc, lams, n0s, uns, psacws, npsacws
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  call get_command_argument(3, adcs); read(adcs, *) dcs
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=dcs, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  open(10, file=trim(fin), status='old'); read(10, *) n
  allocate(t(n), p(n), rho(n), qiic(n), niic(n), qcic(n), ncic(n), qsic(n), nsic(n), &
       dv(n), mu(n), rhof(n), asn(n), lami(n), n0i(n), prci(n), nprci(n), pgam(n), &
       lamc(n), lams(n), n0s(n), uns(n), psacws(n), npsacws(n))
  do i = 1, n
     read(10, *) t(i), p(i), rho(i), qiic(i), niic(i), qcic(i), ncic(i), qsic(i), nsic(i)
  end do
  close(10)
  ! verbatim micro_mg2_0.F90:933-944
  dv = 8.794E-5_r8 * t**1.81_r8 / p
  mu = 1.496E-6_r8 * t**1.5_r8 / (t + 120._r8)
  rhosu = 85000._r8/(rair * tmelt)
  rhof = (rhosu/rho)**0.54_r8
  asn = as*rhof
  ! ice PSD + autoconversion, :1331-1348
  call size_dist_param_basic(mg_ice_props, qiic, niic, lami, n, n0=n0i)
  call ice_autoconversion(t, qiic, lami, n0i, dcs, prci, nprci, n)
  ! liquid PSD (adjusts ncic in place), :1302; snow PSD :1400-1401
  call size_dist_param_liq(mg_liq_props, qcic, ncic, rho, pgam, lamc, n)
  call size_dist_param_basic(mg_snow_props, qsic, nsic, lams, n, n0=n0s)
  ! verbatim :1403-1413 (number-weighted snow fall speed)
  where (lams > 0._r8)
     uns = min(asn*gamma(1._r8+bs)/lams**bs,1.2_r8*rhof)
  elsewhere
     uns = 0._r8
  end where
  call accrete_cloud_water_snow(t, rho, asn, uns, mu, qcic, ncic, qsic, pgam, lamc, &
       lams, n0s, psacws, npsacws, n)
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(9(1x,es25.16e3))') niic(i), lami(i), n0i(i), prci(i), nprci(i), &
          pgam(i), lamc(i), psacws(i), npsacws(i)
  end do
  close(11)
  print *, 'MG2_ICE_REST_PROCS_OK n=', n, ' dcs=', dcs
end program mg2_ice_rest_procs
