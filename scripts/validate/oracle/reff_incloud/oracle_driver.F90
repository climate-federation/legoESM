! CAM6 MG2 effective-radius oracle driver for
! tests/atmosphere/oracle_fixtures/reff_incloud.npz.
!
! The executable MG2 statements are VERBATIM line ranges of CAM cam_cesm2_1_rel_60
! src/physics/cam/micro_mg2_0.F90, pulled in by #include (extracted by build.sh):
!   mg2_cld.inc  = L877-879   lcldm/icldm = max(cloud fraction, mincld)
!   mg2_dum.inc  = L2739-2768 in-cloud dumc/dumnc/dumi/dumni, nccons, 5 g/kg cap
!   mg2_ice.inc  = L2772-2808 effi from size_dist_param_basic (mg_ice_props)
!   mg2_liq.inc  = L2811-2865 effc from size_dist_param_liq (mg_liq_props)
! and of src/physics/cam/micro_mg_cam.F90, the grid REL that CAM6 radiation reads
! (pbuf REL; rel_grid => rel without subcolumns, L2406):
!   mgcam_icw.inc = L2296      icwmrst = min(q/max(mincld,liqcldf), 0.005)
!   mgcam_rel.inc = L2534-2554 ncic = nc/max(mincld,liqcldf), size_dist_param_liq,
!                              REL where icwmrst >= qsmall, else 10
! (the glue assignments L2445/2447/2455/2512 are restated below).
! micro_mg_utils.F90 (mincld, qsmall, size-distribution routines) is compiled
! unmodified.  The WRF v4.6.1 M2005 PSD that legoESM implements (module_mp_morr_two_moment.F
! L3918-3949 with DUM = legoESM moist density, L4024-4029) is then evaluated on MG2's
! in-cloud (dumc, ncnst): wrf_liq.inc / wrf_effc.inc.
! One point per call (mgncol = nlev = 1) so ncnst (an MG2 scalar) can vary per point.
!
! usage: oracle_driver <in.bin> <out.bin>
!   in.bin : n, dcs, rhow, then qc qi ni ncnst cldf rho (n each; ni per kg grid mean,
!            ncnst per m3 in-cloud, cldf = CAM6 ast, rho = legoESM moist density)
!   out.bin: lcldm dumc effc_mg2 effi_mg2 ni_adj effc_wrf rel_cam  (n each)
program oracle_driver
  use micro_mg_utils, only: r8, pi, rhoi, rhows, mincld, qsmall, micro_mg_utils_init, &
       mg_liq_props, mg_ice_props, size_dist_param_liq, size_dist_param_basic
  implicit none
  integer, parameter :: mgncol = 1, nlev = 1
  integer :: n, j, i, k, u
  real(r8) :: dcs, rhow_lego
  character(len=512) :: fin, fout
  character(128) :: errstring
  real(r8), allocatable :: a_qc(:), a_qi(:), a_ni(:), a_nc(:), a_cf(:), a_rho(:)
  real(r8), allocatable :: o_lcldm(:), o_dumc(:), o_effc(:), o_effi(:), o_niadj(:), o_wrf(:), o_rel(:)
  ! micro_mg_cam locals (shapes (pcols,pver) = (1,1); top_lev = ngrdcol = 1)
  type :: state_t
     real(r8) :: q(1,1,2)
  end type state_t
  type(state_t) :: state_loc
  integer, parameter :: ixcldliq = 1, ixnumliq = 2
  integer :: ngrdcol, top_lev
  real(r8), dimension(1,1) :: icwmrst, icwmrst_grid, liqcldf_grid, nc_grid, rho_grid, &
       ncic_grid, mu_grid, lambdac_grid, rel_grid
  ! MG2 locals (micro_mg2_0.F90 declarations, shapes (mgncol,nlev))
  real(r8), dimension(mgncol,nlev) :: cldn, liqcldf, icecldf, cldm, lcldm, icldm, rho, &
       qc, qctend, qi, qitend, nc, nctend, ni, nitend, qr, qrtend, nr, nrtend, qs, qstend, &
       ns, nstend, precip_frac, dumc, dumi, dumnc, dumni, dumr, dumnr, dums, dumns, &
       lami, dum_2D, effi, sadice, deffi, re_ice, pgam, lamc, effc, lamcrad, pgamrad, effc_fn
  real(r8) :: ncnst, ninst, deltat, dumni0, dum
  logical :: nccons, nicons, do_cldice
  ! WRF M2005 locals
  real(r8) :: QC3D(1), NC3D(1), PGAM3(1), LAMC3(1), EFFC3(1), RHOM(1)
  real(r8) :: CONS26, LAMMIN, LAMMAX, DUM3, QSMALLW

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) n, dcs, rhow_lego
  allocate(a_qc(n), a_qi(n), a_ni(n), a_nc(n), a_cf(n), a_rho(n))
  allocate(o_lcldm(n), o_dumc(n), o_effc(n), o_effi(n), o_niadj(n), o_wrf(n), o_rel(n))
  ngrdcol = 1; top_lev = 1
  read(u) a_qc, a_qi, a_ni, a_nc, a_cf, a_rho
  close(u)
  ! Only dcs reaches effc/effi (ice lambda bounds); the thermodynamic constants
  ! set rain/snow properties that are not used here.
  call micro_mg_utils_init(r8, 461.5_r8, 1004.6_r8, 273.15_r8, 2.501e6_r8, 3.337e5_r8, dcs, errstring)
  if (len_trim(errstring) > 0) stop 'micro_mg_utils_init failed'
  nccons = .true.; nicons = .false.; do_cldice = .true.; ninst = 0._r8; deltat = 1._r8
  QSMALLW = 1.E-14_r8; CONS26 = pi/6._r8*rhow_lego
  do j = 1, n
     cldn = a_cf(j); liqcldf = a_cf(j); icecldf = a_cf(j)   ! CAM6: alst_mic, aist_mic => ast
     qc = a_qc(j); qi = a_qi(j); ni = a_ni(j); ncnst = a_nc(j); rho = a_rho(j)
     nc = ncnst/rho*max(a_cf(j), mincld)
     qctend = 0._r8; qitend = 0._r8; nctend = 0._r8; nitend = 0._r8
     qr = 0._r8; qrtend = 0._r8; nr = 0._r8; nrtend = 0._r8; qs = 0._r8; qstend = 0._r8
     ns = 0._r8; nstend = 0._r8; precip_frac = 1._r8; re_ice = 0._r8
#include "mg2_cld.inc"
#include "mg2_dum.inc"
     o_lcldm(j) = lcldm(1,1); o_dumc(j) = dumc(1,1)
#include "mg2_ice.inc"
     o_niadj(j) = merge(1._r8, 0._r8, nitend(1,1) /= 0._r8)
#include "mg2_liq.inc"
     o_effc(j) = effc(1,1); o_effi(j) = effi(1,1)
     ! micro_mg_cam grid REL from the same grid-mean state (qc, nc = ncnst/rho*lcldm)
     state_loc%q(1,1,ixcldliq) = a_qc(j); state_loc%q(1,1,ixnumliq) = nc(1,1)
     liqcldf = a_cf(j)
     i = 1; k = 1
#include "mgcam_icw.inc"
     icwmrst_grid = icwmrst; liqcldf_grid = liqcldf      ! L2445, L2447
     nc_grid = state_loc%q(:,:,ixnumliq); rho_grid = rho  ! L2455, L2512
#include "mgcam_rel.inc"
     o_rel(j) = rel_grid(1,1)
     ! WRF M2005 PSD on MG2's in-cloud water and the in-cloud number per kg of air
     QC3D(1) = o_dumc(j); NC3D(1) = ncnst/a_rho(j); RHOM(1) = a_rho(j)
     PGAM3 = 0._r8; LAMC3 = 0._r8
     K = 1
#include "wrf_liq.inc"
#include "wrf_effc.inc"
     o_wrf(j) = EFFC3(1)
  end do
  open(newunit=u, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(u) o_lcldm, o_dumc, o_effc, o_effi, o_niadj, o_wrf, o_rel
  close(u)
  print *, 'ORACLE_DONE', n
end program oracle_driver
