! Per-process oracle driver for the CAM6 MG2 warm-rain kernels.  Calls the
! UNMODIFIED micro_mg_utils.F90 routines (CAM cam_cesm2_1_rel_60) in the order
! micro_mg_tend calls them (micro_mg2_0.F90:1296-1335, :1367-1368,
! :1505-1527).  The in-precip assignment and the dv/mu/sc/arn lines are
! copied verbatim from micro_mg2_0.F90 (:1312-1328 and :933-944).
!
! Point input (one line per point, after a first line holding n):
!   qcic ncic[#/kg] rho relvar qr[kg/kg grid] nr[#/kg grid] T p qfrac lcldm
!   precip_frac qiic          with q = qfrac*qvl(T,p)
! Output (one line per point):
!   prc nprc nprc1 ncic_adj pra npra nragg lamr n0r nric qric pre qvl
! usage: mg2_procs_driver points.txt out.txt
program mg2_procs_driver
  use micro_mg_utils, only: r8, pi, ar, qsmall, mg_liq_props, mg_rain_props, &
       size_dist_param_liq, size_dist_param_basic, kk2000_liq_autoconversion, &
       accrete_cloud_water_rain, self_collection_rain, evaporate_sublimate_precip
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: qsat_water => wv_sat_qsat_water, wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: n, i
  character(len=512) :: fin, fout
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8), allocatable, dimension(:) :: qcic, ncic, rho, relvar, qr, nr, t, p, qfrac, &
       lcldm, precip_frac, qiic, qric, nric, pgam, lamc, prc, nprc, nprc1, pra, npra, &
       accre, nragg, lamr, n0r, dv, mu, sc, arn, asn, q, qvl, qvi, es, zero, pre, prds, &
       am, lams, n0s
  real(r8) :: rhosu
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=500.e-6_r8, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  open(10, file=trim(fin), status='old'); read(10, *) n
  allocate(qcic(n), ncic(n), rho(n), relvar(n), qr(n), nr(n), t(n), p(n), qfrac(n), &
       lcldm(n), precip_frac(n), qiic(n), qric(n), nric(n), pgam(n), lamc(n), prc(n), &
       nprc(n), nprc1(n), pra(n), npra(n), accre(n), nragg(n), lamr(n), n0r(n), dv(n), &
       mu(n), sc(n), arn(n), asn(n), q(n), qvl(n), qvi(n), es(n), zero(n), pre(n), &
       prds(n), am(n), lams(n), n0s(n))
  do i = 1, n
     read(10, *) qcic(i), ncic(i), rho(i), relvar(i), qr(i), nr(i), t(i), p(i), &
          qfrac(i), lcldm(i), precip_frac(i), qiic(i)
  end do
  close(10)
  accre = 1._r8; zero = 0._r8
  ! micro_mg_tend order: size_dist_param_liq (adjusts ncic) -> kk2000
  call size_dist_param_liq(mg_liq_props, qcic, ncic, rho, pgam, lamc, n)
  call kk2000_liq_autoconversion(.false., qcic, ncic, rho, relvar, prc, nprc, nprc1, n)
  ! verbatim micro_mg2_0.F90:1312-1328
  qric(:) = qr(:)/precip_frac(:)
  nric(:) = nr(:)/precip_frac(:)
  qric(:)=min(qric(:),0.01_r8)
  where (qric(:).lt.qsmall)
     qric(:)=0._r8
     nric(:)=0._r8
  end where
  nric(:)=max(nric(:),0._r8)
  call size_dist_param_basic(mg_rain_props, qric, nric, lamr, n, n0=n0r)
  call accrete_cloud_water_rain(.false., qric, qcic, ncic, relvar, accre, pra, npra, n)
  call self_collection_rain(rho, qric, nric, nragg, n)
  ! verbatim micro_mg2_0.F90:933-944
  dv = 8.794E-5_r8 * t**1.81_r8 / p
  mu = 1.496E-6_r8 * t**1.5_r8 / (t + 120._r8)
  sc = mu/(rho*dv)
  rhosu = 85000._r8/(rair * tmelt)
  arn = ar*(rhosu/rho)**0.54_r8
  asn = 0._r8
  do i = 1, n
     call qsat_water(t(i), p(i), es(i), qvl(i))
  end do
  qvi = qvl
  q = qfrac*qvl
  lams = 0._r8; n0s = 0._r8
  call evaporate_sublimate_precip(t, rho, dv, mu, sc, q, qvl, qvi, lcldm, precip_frac, &
       arn, asn, qcic, qiic, qric, zero, lamr, n0r, lams, n0s, pre, prds, am, n)
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(13(1x,es25.16e3))') prc(i), nprc(i), nprc1(i), ncic(i), pra(i), &
          npra(i), nragg(i), lamr(i), n0r(i), nric(i), qric(i), pre(i), qvl(i)
  end do
  close(11)
  print *, 'MG2_PROCS_OK n=', n
end program mg2_procs_driver
