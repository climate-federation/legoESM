! Point oracle for CAM6 MG2 cloud-ice deposition / sublimation / Bergeron,
! snow Bergeron, snow sublimation and ice->snow autoconversion.  Calls the
! UNMODIFIED micro_mg_utils.F90 routines (CAM cam_cesm2_1_rel_60); the
! in-cloud / in-precip assignments and qvl/qvi, dv/mu/sc/rhof lines are
! verbatim from micro_mg2_0.F90 (:905-928 with qsfm = 1, :933-944,
! :1236-1250, :1359-1375).
!
! Point input (one line per point, after a first line holding n):
!   T q qi[grid] ni[#/kg grid] qcic qs[grid] ns[#/kg grid] precip_frac icldm lcldm p
! Output (one line per point):
!   vap_dep ice_sublim berg bergs prds prci nprci qvl qvi lami n0i lams n0s
! usage: mg2_ice points.txt out.txt
program mg2_ice_driver
  use micro_mg_utils, only: r8, qsmall, as, mg_ice_props, mg_snow_props, &
       size_dist_param_basic, ice_autoconversion, ice_deposition_sublimation, &
       bergeron_process_snow, evaporate_sublimate_precip
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: qsat_water => wv_sat_qsat_water, qsat_ice => wv_sat_qsat_ice, &
       wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  real(r8), parameter :: dcs = 500.e-6_r8
  integer :: n, i
  character(len=512) :: fin, fout
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8), allocatable, dimension(:) :: t, q, qi, ni, qcic, qs, ns, precip_frac, icldm, &
       lcldm, p, rho, qiic, niic, qsic, nsic, qvl, qvi, es, dv, mu, sc, rhof, asn, arn, &
       lami, n0i, lams, n0s, prci, nprci, berg, vap_dep, ice_sublim, bergs, zero, pre, &
       prds, am, lamr, n0r
  real(r8) :: rhosu
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=dcs, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  open(10, file=trim(fin), status='old'); read(10, *) n
  allocate(t(n), q(n), qi(n), ni(n), qcic(n), qs(n), ns(n), precip_frac(n), icldm(n), &
       lcldm(n), p(n), rho(n), qiic(n), niic(n), qsic(n), nsic(n), qvl(n), qvi(n), es(n), &
       dv(n), mu(n), sc(n), rhof(n), asn(n), arn(n), lami(n), n0i(n), lams(n), n0s(n), &
       prci(n), nprci(n), berg(n), vap_dep(n), ice_sublim(n), bergs(n), zero(n), pre(n), &
       prds(n), am(n), lamr(n), n0r(n))
  do i = 1, n
     read(10, *) t(i), q(i), qi(i), ni(i), qcic(i), qs(i), ns(i), precip_frac(i), &
          icldm(i), lcldm(i), p(i)
  end do
  close(10)
  zero = 0._r8; lamr = 0._r8; n0r = 0._r8
  rho = p/(rair*t)
  ! verbatim micro_mg2_0.F90:905-928, qsfm = 1
  do i = 1, n
     call qsat_water(t(i), p(i), es(i), qvl(i))
     if (t(i) >= tmelt) then
        qvi(i) = qvl(i)
     else
        call qsat_ice(t(i), p(i), es(i), qvi(i))
     end if
  end do
  ! verbatim micro_mg2_0.F90:933-944 (dv, mu, sc) and :893-899 (rhof, asn)
  dv = 8.794E-5_r8 * t**1.81_r8 / p
  mu = 1.496E-6_r8 * t**1.5_r8 / (t + 120._r8)
  sc = mu/(rho*dv)
  rhosu = 85000._r8/(rair * tmelt)
  rhof = (rhosu/rho)**0.54_r8
  asn = as*rhof
  arn = 0._r8
  ! verbatim micro_mg2_0.F90:1238-1250 (nicons off)
  do i = 1, n
     if (qi(i) >= qsmall) then
        qiic(i) = min(qi(i)/icldm(i), 5.e-3_r8)
        niic(i) = max(ni(i)/icldm(i), 0._r8)
     else
        qiic(i) = 0._r8
        niic(i) = 0._r8
     end if
  end do
  ! verbatim micro_mg2_0.F90:1359-1375
  qsic(:) = qs(:)/precip_frac(:)
  nsic(:) = ns(:)/precip_frac(:)
  qsic(:) = min(qsic(:), 0.01_r8)
  where (qsic(:) < qsmall)
     qsic(:) = 0._r8
     nsic(:) = 0._r8
  end where
  nsic(:) = max(nsic(:), 0._r8)
  call size_dist_param_basic(mg_ice_props, qiic, niic, lami, n, n0=n0i)
  call ice_autoconversion(t, qiic, lami, n0i, dcs, prci, nprci, n)
  call size_dist_param_basic(mg_snow_props, qsic, nsic, lams, n, n0=n0s)
  call evaporate_sublimate_precip(t, rho, dv, mu, sc, q, qvl, qvi, lcldm, precip_frac, &
       arn, asn, qcic, qiic, zero, qsic, lamr, n0r, lams, n0s, pre, prds, am, n)
  call bergeron_process_snow(t, rho, dv, mu, sc, qvl, qvi, asn, qcic, qsic, lams, n0s, &
       bergs, n)
  call ice_deposition_sublimation(t, q, qi, ni, icldm, rho, dv, qvl, qvi, berg, vap_dep, &
       ice_sublim, n)
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(13(1x,es25.16e3))') vap_dep(i), ice_sublim(i), berg(i), bergs(i), prds(i), &
          prci(i), nprci(i), qvl(i), qvi(i), lami(i), n0i(i), lams(i), n0s(i)
  end do
  close(11)
  print *, 'MG2_ICE_OK n=', n
end program mg2_ice_driver
