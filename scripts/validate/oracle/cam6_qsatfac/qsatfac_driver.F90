! Point oracle for CAM6 do_subgrid_growth (cam_cesm2_1_rel_60): the sub-grid
! saturation factor qsatfac of cldfrc2m.F90 aist_vector (iceopt 5) and its use
! as qsfm in MG2.  The qsatfac lines are verbatim from cldfrc2m.F90:1010-1024
! (qs, esl, esi; CAM's wv_saturation qsat_water/svp_water/svp_ice are the
! default-scheme wv_sat_methods routines, Goff-Gratch) and :1086-1117 (rhi,
! rhdif, aist, qsatfac; the per-point scalars qs, esl, esi are named qsx,
! eslx, esix here, the arithmetic is unchanged).  The MG2 lines are verbatim from micro_mg2_0.F90:
! 905-928 (qvl/qvi with the qsfm scaling of :922-927), :933 (relhum) and the
! ice nucleation humidity test of :1118, then the UNMODIFIED micro_mg_utils
! routines that read qvl/qvi, as in morrison_mg2_warm_rain/mg2_ice_driver.F90.
!
! Point input (one line per point, after a first line holding n):
!   T q qi[grid] ni[#/kg grid] qcic qs[grid] ns[#/kg grid] precip_frac icldm lcldm p
!   rhmini rhmaxi
! Output (one line per point):
!   qsatfac aist_raw qvl qvi relhum_rhi vap_dep ice_sublim berg bergs prds
! usage: qsatfac_driver points.txt out.txt
program qsatfac_driver
  use micro_mg_utils, only: r8, qsmall, as, mg_ice_props, mg_snow_props, &
       size_dist_param_basic, ice_deposition_sublimation, &
       bergeron_process_snow, evaporate_sublimate_precip
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: qsat_water => wv_sat_qsat_water, qsat_ice => wv_sat_qsat_ice, &
       svp_water => wv_sat_svp_water, svp_ice => wv_sat_svp_ice, wv_sat_methods_init
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
       lcldm, p, rho, qiic, niic, qsic, nsic, qvl, qvi, esl, esi, dv, mu, sc, rhof, asn, arn, &
       lami, n0i, lams, n0s, berg, vap_dep, ice_sublim, bergs, zero, pre, &
       prds, am, lamr, n0r, rhmini_in, rhmaxi_in, qsatfac, aist_raw, qsat_in, esat_in, &
       qsfm, relhum, gate
  real(r8) :: rhosu, qv, qsx, eslx, esix, rhi, rhdif, aist, rhmini, rhmaxi
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
       lcldm(n), p(n), rho(n), qiic(n), niic(n), qsic(n), nsic(n), qvl(n), qvi(n), esl(n), &
       esi(n), dv(n), mu(n), sc(n), rhof(n), asn(n), arn(n), lami(n), n0i(n), lams(n), &
       n0s(n), berg(n), vap_dep(n), ice_sublim(n), bergs(n), zero(n), pre(n), &
       prds(n), am(n), lamr(n), n0r(n), rhmini_in(n), rhmaxi_in(n), qsatfac(n), &
       aist_raw(n), qsat_in(n), esat_in(n), qsfm(n), relhum(n), gate(n))
  do i = 1, n
     read(10, *) t(i), q(i), qi(i), ni(i), qcic(i), qs(i), ns(i), precip_frac(i), &
          icldm(i), lcldm(i), p(i), rhmini_in(i), rhmaxi_in(i)
  end do
  close(10)
  zero = 0._r8; lamr = 0._r8; n0r = 0._r8
  rho = p/(rair*t)
  ! verbatim cldfrc2m.F90:1010-1024 (aist_vector; qv = state q, qi = cloud ice)
  call qsat_water(t(1:n), p(1:n), esat_in(1:n), qsat_in(1:n))
  do i = 1, n
     qv = q(i)
     qsx = qsat_in(i)
     eslx = svp_water(t(i))
     esix = svp_ice(t(i))
     rhmini = rhmini_in(i)
     rhmaxi = rhmaxi_in(i)
     ! verbatim cldfrc2m.F90:1086-1117 (iceopt 5)
     rhi = (qv+qi(i))/qsx * (eslx/esix)
     if (rhmaxi .eq. rhmini) then
        if (rhi .gt. rhmini) then
           rhdif = 1._r8
        else
           rhdif = 0._r8
        end if
     else
        rhdif = (rhi-rhmini) / (rhmaxi - rhmini)
     end if
     aist = min(1.0_r8, max(rhdif,0._r8)**2)
     aist_raw(i) = aist
     qsatfac(i) = max(min(qv / qsx, 1._r8), (1._r8 - aist) * rhmini + aist * rhmaxi)
  end do
  ! verbatim micro_mg2_0.F90:880 (qsfm = qsatfac) and :905-928
  qsfm = qsatfac
  do i = 1, n
     call qsat_water(t(i), p(i), esl(i), qvl(i))
     if (t(i) >= tmelt) then
        esi(i)=esl(i)
        qvi(i)=qvl(i)
     else
        call qsat_ice(t(i), p(i), esi(i), qvi(i))
        qvi(i) = qsfm(i) * qvi(i)
        esi(i) = qsfm(i) * esi(i)
        qvl(i) = qsfm(i) * qvl(i)
        esl(i) = qsfm(i) * esl(i)
     end if
  end do
  ! verbatim micro_mg2_0.F90:933 and the humidity factor of :1118
  relhum = q / max(qvl, qsmall)
  gate = relhum*esl/esi
  ! verbatim micro_mg2_0.F90:886-888 (dv, mu, sc) and :893-899 (rhof, asn)
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
  call size_dist_param_basic(mg_snow_props, qsic, nsic, lams, n, n0=n0s)
  call evaporate_sublimate_precip(t, rho, dv, mu, sc, q, qvl, qvi, lcldm, precip_frac, &
       arn, asn, qcic, qiic, zero, qsic, lamr, n0r, lams, n0s, pre, prds, am, n)
  call bergeron_process_snow(t, rho, dv, mu, sc, qvl, qvi, asn, qcic, qsic, lams, n0s, &
       bergs, n)
  call ice_deposition_sublimation(t, q, qi, ni, icldm, rho, dv, qvl, qvi, berg, vap_dep, &
       ice_sublim, n)
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(10(1x,es25.16e3))') qsatfac(i), aist_raw(i), qvl(i), qvi(i), gate(i), &
          vap_dep(i), ice_sublim(i), berg(i), bergs(i), prds(i)
  end do
  close(11)
  print *, 'QSATFAC_OK n=', n
end program qsatfac_driver
