! Per-process oracle driver for the CAM6 MG2 two-moment SNOW kernels, Bigg rain
! freezing and the ice->snow autoconversion.  Calls the UNMODIFIED
! micro_mg_utils.F90 routines (CAM cam_cesm2_1_rel_60) the way micro_mg_tend
! calls them; the in-precip assignments and dv/mu/sc/rhof/arn/asn lines are
! copied verbatim from micro_mg2_0.F90 (:933-944, :1312-1322, :1358-1375,
! :1403-1413).  Extends the tier-1 audit driver
! (oracle_audit/morrison_ice_phase/fortran/mg2/mg2_ice_procs.F90).
!
! Point input (first line n, then one line per point):
!   T p rho qfrac qcic ncic[#/kg] qs ns[#/kg] (grid) lcldm precip_frac qiic
!   qr nr[#/kg] (grid) qi ni[#/kg] (in-cloud ice, icldm = 1)
! with q = qfrac*qvl(T,p).
! Output (one line per point, 28 columns):
!   qsic nsic lams n0s ums uns nsagg pgam lamc ncic_adj psacws npsacws bergs
!   prds qvl qvi qric nric lamr mnuccr nnuccr lami n0i niic prci nprci pre am
! usage: mg2_snow_procs points.txt out.txt dcs
program mg2_snow_procs
  use micro_mg_utils, only: r8, pi, ar, as, bs, qsmall, mg_liq_props, mg_rain_props, &
       mg_snow_props, mg_ice_props, size_dist_param_liq, size_dist_param_basic, &
       snow_self_aggregation, accrete_cloud_water_snow, bergeron_process_snow, &
       evaporate_sublimate_precip, heterogeneous_rain_freezing, ice_autoconversion, rhosn
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: qsat_water => wv_sat_qsat_water, qsat_ice => wv_sat_qsat_ice, &
       wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: n, i
  character(len=512) :: fin, fout, adcs
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8) :: dcs, rhosu, es
  real(r8), allocatable, dimension(:) :: t, p, rho, qfrac, qcic, ncic, qs, ns, lcldm, pf, &
       qiic, qr, nr, qi, ni, q, qvl, qvi, dv, mu, sc, rhof, arn, asn, qsic, nsic, lams, n0s, &
       ums, uns, nsagg, pgam, lamc, psacws, npsacws, bergs, prds, qric, nric, lamr, n0r, &
       mnuccr, nnuccr, lami, n0i, niic, prci, nprci, pre, am
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  call get_command_argument(3, adcs); read(adcs, *) dcs
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=dcs, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  open(10, file=trim(fin), status='old'); read(10, *) n
  allocate(t(n), p(n), rho(n), qfrac(n), qcic(n), ncic(n), qs(n), ns(n), lcldm(n), pf(n), &
       qiic(n), qr(n), nr(n), qi(n), ni(n), q(n), qvl(n), qvi(n), dv(n), mu(n), sc(n), &
       rhof(n), arn(n), asn(n), qsic(n), nsic(n), lams(n), n0s(n), ums(n), uns(n), nsagg(n), &
       pgam(n), lamc(n), psacws(n), npsacws(n), bergs(n), prds(n), qric(n), nric(n), &
       lamr(n), n0r(n), mnuccr(n), nnuccr(n), lami(n), n0i(n), niic(n), prci(n), nprci(n), &
       pre(n), am(n))
  do i = 1, n
     read(10, *) t(i), p(i), rho(i), qfrac(i), qcic(i), ncic(i), qs(i), ns(i), lcldm(i), &
          pf(i), qiic(i), qr(i), nr(i), qi(i), ni(i)
  end do
  close(10)
  do i = 1, n
     call qsat_water(t(i), p(i), es, qvl(i))
     call qsat_ice(t(i), p(i), es, qvi(i))
  end do
  q = qfrac*qvl
  ! verbatim micro_mg2_0.F90:933-944
  dv = 8.794E-5_r8 * t**1.81_r8 / p
  mu = 1.496E-6_r8 * t**1.5_r8 / (t + 120._r8)
  sc = mu/(rho*dv)
  rhosu = 85000._r8/(rair * tmelt)
  rhof = (rhosu/rho)**0.54_r8
  arn = ar*rhof
  asn = as*rhof
  ! liquid PSD (adjusts ncic in place), :1302
  call size_dist_param_liq(mg_liq_props, qcic, ncic, rho, pgam, lamc, n)
  ! verbatim micro_mg2_0.F90:1312-1322 (rain) and :1358-1375 (snow)
  qric(:) = qr(:)/pf(:)
  nric(:) = nr(:)/pf(:)
  qric(:)=min(qric(:),0.01_r8)
  where (qric(:).lt.qsmall)
     qric(:)=0._r8
     nric(:)=0._r8
  end where
  nric(:)=max(nric(:),0._r8)
  qsic(:) = qs(:)/pf(:)
  nsic(:) = ns(:)/pf(:)
  qsic(:)=min(qsic(:),0.01_r8)
  where (qsic(:) < qsmall)
     qsic(:)=0._r8
     nsic(:)=0._r8
  end where
  nsic(:)=max(nsic(:),0._r8)
  call size_dist_param_basic(mg_rain_props, qric, nric, lamr, n, n0=n0r)
  call size_dist_param_basic(mg_snow_props, qsic, nsic, lams, n, n0=n0s)
  ! verbatim :1403-1413 (gamma_bs_plus4/1 evaluated as in micro_mg_init)
  where (lams > 0._r8)
     ums = min(asn*gamma(4._r8+bs)/(6._r8*lams**bs),1.2_r8*rhof)
     uns = min(asn*gamma(1._r8+bs)/lams**bs,1.2_r8*rhof)
  elsewhere
     ums = 0._r8
     uns = 0._r8
  end where
  ! ice PSD + autoconversion (icldm = 1: in-cloud = grid), :1331-1348
  niic = ni
  call size_dist_param_basic(mg_ice_props, qi, niic, lami, n, n0=n0i)
  call ice_autoconversion(t, qi, lami, n0i, dcs, prci, nprci, n)
  ! snow processes in micro_mg_tend order, :1482-1531
  call snow_self_aggregation(t, rho, asn, rhosn, qsic, nsic, nsagg, n)
  call accrete_cloud_water_snow(t, rho, asn, uns, mu, qcic, ncic, qsic, pgam, lamc, &
       lams, n0s, psacws, npsacws, n)
  call heterogeneous_rain_freezing(t, qric, nric, lamr, mnuccr, nnuccr, n)
  call evaporate_sublimate_precip(t, rho, dv, mu, sc, q, qvl, qvi, lcldm, pf, arn, asn, &
       qcic, qiic, qric, qsic, lamr, n0r, lams, n0s, pre, prds, am, n)
  call bergeron_process_snow(t, rho, dv, mu, sc, qvl, qvi, asn, qcic, qsic, lams, n0s, &
       bergs, n)
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(28(1x,es25.16e3))') qsic(i), nsic(i), lams(i), n0s(i), ums(i), uns(i), &
          nsagg(i), pgam(i), lamc(i), ncic(i), psacws(i), npsacws(i), bergs(i), prds(i), &
          qvl(i), qvi(i), qric(i), nric(i), lamr(i), mnuccr(i), nnuccr(i), lami(i), n0i(i), &
          niic(i), prci(i), nprci(i), pre(i), am(i)
  end do
  close(11)
  print *, 'MG2_SNOW_PROCS_OK n=', n, ' dcs=', dcs
end program mg2_snow_procs
