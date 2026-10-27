! Point oracle for CAM6 MG2 snow riming of cloud water + Hallett-Mossop
! splinters.  Calls the UNMODIFIED micro_mg_utils.F90 routines (CAM
! cam_cesm2_1_rel_60) in micro_mg_tend's order: size_dist_param_liq (:1296),
! the in-precip snow lines (:1359-1375), size_dist_param_basic(snow) and
! uns (:1400-1412), accrete_cloud_water_snow (:1485) and
! secondary_ice_production (:1490).  rhof/asn/mu verbatim from :888-899.
!
! Point input (one line per point, after a first line holding n):
!   qcic ncic[#/kg] rho qs[grid] ns[#/kg grid] precip_frac T
! Output (one line per point):
!   psacws_raw npsacws psacws_left msacwi nsacwi lams n0s uns nsic_adj ncic_adj
! usage: mg2_rime points.txt out.txt
program mg2_rime_driver
  use micro_mg_utils, only: r8, qsmall, as, bs, mg_liq_props, mg_snow_props, &
       size_dist_param_liq, size_dist_param_basic, accrete_cloud_water_snow, &
       secondary_ice_production
  use micro_mg2_0, only: micro_mg_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: n, i
  character(len=512) :: fin, fout
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8), allocatable, dimension(:) :: qcic, ncic, rho, qs, ns, precip_frac, t, &
       pgam, lamc, qsic, nsic, lams, n0s, uns, rhof, asn, mu, psacws, npsacws, &
       psraw, msacwi, nsacwi
  real(r8) :: rhosu
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=500.e-6_r8, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  open(10, file=trim(fin), status='old'); read(10, *) n
  allocate(qcic(n), ncic(n), rho(n), qs(n), ns(n), precip_frac(n), t(n), pgam(n), &
       lamc(n), qsic(n), nsic(n), lams(n), n0s(n), uns(n), rhof(n), asn(n), mu(n), &
       psacws(n), npsacws(n), psraw(n), msacwi(n), nsacwi(n))
  do i = 1, n
     read(10, *) qcic(i), ncic(i), rho(i), qs(i), ns(i), precip_frac(i), t(i)
  end do
  close(10)
  call size_dist_param_liq(mg_liq_props, qcic, ncic, rho, pgam, lamc, n)
  ! verbatim micro_mg2_0.F90:888-899
  mu = 1.496E-6_r8 * t**1.5_r8 / (t + 120._r8)
  rhosu = 85000._r8/(rair * tmelt)
  rhof = (rhosu/rho)**0.54_r8
  asn = as*rhof
  ! verbatim micro_mg2_0.F90:1359-1375
  qsic(:) = qs(:)/precip_frac(:)
  nsic(:) = ns(:)/precip_frac(:)
  qsic(:)=min(qsic(:),0.01_r8)
  where (qsic(:) < qsmall)
     qsic(:)=0._r8
     nsic(:)=0._r8
  end where
  nsic(:)=max(nsic(:),0._r8)
  ! verbatim micro_mg2_0.F90:1400-1412
  call size_dist_param_basic(mg_snow_props, qsic, nsic, lams, n, n0=n0s)
  where (lams > 0._r8)
     uns = min(asn*gamma(1._r8+bs)/lams**bs,1.2_r8*rhof)
  elsewhere
     uns = 0._r8
  end where
  call accrete_cloud_water_snow(t, rho, asn, uns, mu, qcic, ncic, qsic, pgam, lamc, &
       lams, n0s, psacws, npsacws, n)
  psraw = psacws
  call secondary_ice_production(t, psacws, msacwi, nsacwi, n)
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(10(1x,es25.16e3))') psraw(i), npsacws(i), psacws(i), msacwi(i), &
          nsacwi(i), lams(i), n0s(i), uns(i), nsic(i), ncic(i)
  end do
  close(11)
  print *, 'MG2_RIME_OK n=', n
end program mg2_rime_driver
