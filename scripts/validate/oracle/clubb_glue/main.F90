! Standalone oracle driver: CAM cam_cesm2_1_rel_60 clubb_intr.F90 host glue
! (host->CLUBB mapping, post-CLUBB clubbtop / water return / energy fixer /
! tendencies, alst/deepcu/aist/ast/cloud_frac assembly), tropopause.F90
! (twmo + level search + polar rule; analytic fallback stands in for the
! climatology exactly as legoESM documents), cldfrc2m.F90 aist_vector (iceopt=5).
! Every physics line comes from an include file holding VERBATIM oracle lines
! (gen.sh).  The CLUBB core is replaced by prescribed increments read from file
! (identical to the ones handed to legoESM's fake core).
! argv(1) = trop_cloud_top_press [Pa]; output to stdout.
program oracle_glue
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, pverp
  use physconst
  use wv_sat_methods, only: wv_sat_methods_init
  use cldfrc2m_x
  use tropo_x
  implicit none
  integer, parameter :: ixq = 1, ixcldliq = 2, ixcldice = 3, ixnumice = 4
  real(r8), parameter :: p0_clubb = 1.0e5_r8
  logical, parameter :: apply_to_heat = .false., clubb_do_energyfix = .true.
  logical, parameter :: single_column = .false.
  type(physics_state) :: state1
  type cam_in_t
     real(r8) :: shf(pcols), cflx(pcols,1), wsx(pcols), wsy(pcols), landfrac(pcols), snowhland(pcols)
  end type cam_in_t
  type(cam_in_t) :: cam_in
  type ptend_t
     real(r8) :: u(pcols,pver), v(pcols,pver), s(pcols,pver), q(pcols,pver,4)
  end type ptend_t
  type(ptend_t) :: ptend_loc
  integer :: ncol, nlev_in, i, k, top_lev, nlev, clubbtop, ios
  integer :: troplev(pcols)
  real(r8) :: hdtime, trop_cloud_top_press, dlats(pcols)
  real(r8) :: pref_mid(pver)
  character(len=64) :: arg
  ! host arrays (CAM layout)
  real(r8) :: exner_clubb(pcols,pver), thv(pcols,pver)
  real(r8) :: rtm(pcols,pverp), rvm(pcols,pverp), um(pcols,pverp), vm(pcols,pverp), thlm(pcols,pverp)
  real(r8) :: rcm(pcols,pverp), rtp2(pcols,pverp), cloud_frac(pcols,pverp)
  real(r8) :: upwp(pcols,pverp), vpwp(pcols,pverp), wprcp(pcols,pverp), rcm_in_layer(pcols,pverp)
  real(r8) :: cmfmc(pcols,pverp), cmfmc_sh(pcols,pverp), dp_icwmr(pcols,pver)
  real(r8) :: alst(pcols,pver), qlst(pcols,pver), deepcu(pcols,pver), shalcu(pcols,pver), concld(pcols,pver)
  real(r8) :: aist(pcols,pver), qsatfac(pcols,pver), ast(pcols,pver), qist(pcols,pver)
  real(r8) :: rhmini(pcols), rhmaxi(pcols), frac_limit, ic_limit
  real(r8) :: radf_clubb(pcols,pver), qrl(pcols,pver), grid_dx(pcols), grid_dy(pcols)
  real(r8) :: se_a(pcols), se_b(pcols), ke_a(pcols), ke_b(pcols), wv_a(pcols), wv_b(pcols)
  real(r8) :: wl_a(pcols), wl_b(pcols), te_a(pcols), te_b(pcols)
  real(r8) :: clubb_s(pver), se_upper_a, se_upper_b, tw_upper_a, tw_upper_b
  real(r8) :: se_upper_diss, tw_upper_diss, se_dis, rtm_test
  logical :: apply_to_surface
  ! CLUBB-grid column arrays
  real(r8), dimension(pverp+1) :: zi_g, zt_g, p_in_Pa, exner, rho_ds_zt, invrs_rho_ds_zt, rho_in, &
       thv_ds_zt, rfrzm, radf, qrl_clubb, rho_zt, wm_zt, rho_ds_zm, rho_zm, invrs_rho_ds_zm, &
       thv_ds_zm, wm_zm
  real(r8) :: dz_g(pver), sfc_elevation, host_dx, host_dy
  real(r8) :: wpthlp_sfc, wprtp_sfc, upwp_sfc, vpwp_sfc
  ! inputs
  real(r8) :: lat(pcols), ps(pcols), shf(pcols), E(pcols), taux(pcols), tauy(pcols)
  real(r8), dimension(pcols,pver) :: pf, dp, T, qv, qc, qi, u, v, zf, icw, drt, dthl, du, dv, rcm_o, cf_o
  real(r8), dimension(pcols,pverp) :: ph, zh, mf, rtp2_o
  ! saved host->CLUBB inputs for comparison (CLUBB order, 1 = ghost)
  real(r8), dimension(pcols,pverp+1) :: o_thlm, o_rtm, o_thv, o_rho, o_exner, o_p
  real(r8) :: o_sfc(pcols,5)
  integer :: o_clubbtop(pcols)
  character(len=*), parameter :: fmt = '(*(es27.17e3))'

  call get_command_argument(1, arg); read(arg,*) trop_cloud_top_press
  hdtime = 600._r8
  frac_limit = 0.01_r8   ! clubb_intr.F90:1356
  ic_limit   = 1.e-12_r8 ! clubb_intr.F90:1357
  call tropo_init()
  call wv_sat_methods_init(r8, tmelt, h2otrip, tboil, ttrice, epsilo, arg)

  call get_command_argument(2, arg); open(10, file=trim(arg), status='old')
  read(10,*) ncol, nlev_in
  if (ncol /= pcols .or. nlev_in /= pver) stop 'dims'
  read(10,*) pref_mid
  do i = 1, ncol
     read(10,*) lat(i), ps(i), shf(i), E(i), taux(i), tauy(i)
     read(10,*) pf(i,:); read(10,*) dp(i,:); read(10,*) T(i,:); read(10,*) qv(i,:)
     read(10,*) qc(i,:); read(10,*) qi(i,:); read(10,*) u(i,:); read(10,*) v(i,:)
     read(10,*) zf(i,:); read(10,*) icw(i,:); read(10,*) drt(i,:); read(10,*) dthl(i,:)
     read(10,*) du(i,:); read(10,*) dv(i,:); read(10,*) rcm_o(i,:); read(10,*) cf_o(i,:)
     read(10,*) ph(i,:); read(10,*) zh(i,:); read(10,*) mf(i,:); read(10,*) rtp2_o(i,:)
  end do
  close(10)

  ! ---- state1 ----
  state1%ncol = ncol
  state1%lat = lat
  state1%t = T; state1%pmid = pf; state1%pdel = dp; state1%pint = ph
  state1%zm = zf; state1%zi = zh; state1%u = u; state1%v = v; state1%omega = 0._r8
  state1%phis = 0._r8
  state1%q = 0._r8
  state1%q(:,:,ixq) = qv; state1%q(:,:,ixcldliq) = qc; state1%q(:,:,ixcldice) = qi
  state1%s = cpair*T + gravit*zf            ! dry static energy, phis = 0
  cam_in%shf = shf; cam_in%cflx(:,1) = E; cam_in%wsx = taux; cam_in%wsy = tauy
  cam_in%landfrac = 0._r8; cam_in%snowhland = 0._r8
  radf_clubb = 0._r8; qrl = 0._r8; grid_dx = 1.e5_r8; grid_dy = 1.e5_r8
  cmfmc = mf; cmfmc_sh = 0._r8; dp_icwmr = icw

  ! ---- top_lev: ref_pres.F90 press_lim_idx(trop_cloud_top_press, top=.true.) ----
  top_lev = pver+1
  do k = 1, pver
     if (pref_mid(k) > trop_cloud_top_press) then
        top_lev = k
        exit
     end if
  end do
  nlev = pver + 1 - top_lev

  ! ---- troplev: tropopause_findChemTrop (twmo, polar rule, backup) ----
  troplev = NOTFOUND
  call tropopause_twmo(state1, troplev)
  dlats(:ncol) = state1%lat(:ncol) * 180._r8/pi
  call polar(state1, troplev)
  ! backup: CAM uses TROP_ALG_CLIMATE (monthly file); legoESM documents the
  ! analytic surface as its stand-in, so the oracle uses it too (flagged per column).
  write(*,'(a,*(i4))') 'TWMO_POLAR_TROPLEV', troplev(:ncol)
  call tropopause_analytic(state1, troplev)
  write(*,'(a,*(i4))') 'TROPLEV', troplev(:ncol)
  write(*,'(a,i4)') 'TOP_LEV', top_lev

  ! ---- host -> CLUBB state (clubb_intr.F90:1534-1550, 1601-1606) ----
  include 'inc_exner.inc'
  do k = 1, pver
    do i = 1, ncol
      include 'inc_init.inc'
    end do
  end do
  include 'inc_thv.inc'

  do i = 1, ncol
     zt_g = 0._r8; zi_g = 0._r8; rho_ds_zt = 1._r8; invrs_rho_ds_zt = 1._r8; rho_in = 1._r8
     thv_ds_zt = 0._r8; wm_zt = 0._r8; rho_ds_zm = 1._r8; thv_ds_zm = 0._r8; rho_zt = 1._r8
     include 'inc_grid.inc'
     include 'inc_wm.inc'
     call zt2zm_cols()
     include 'inc_sfc.inc'
     o_thlm(i,:) = 0._r8; o_rtm(i,:) = 0._r8
     do k = 1, nlev
        o_thlm(i,k+1) = thlm(i,pver-k+1); o_rtm(i,k+1) = rtm(i,pver-k+1)
     end do
     o_thv(i,:) = thv_ds_zt; o_rho(i,:) = rho_ds_zt; o_exner(i,:) = exner; o_p(i,:) = p_in_Pa
     o_sfc(i,:) = (/ wpthlp_sfc, wprtp_sfc, upwp_sfc, vpwp_sfc, rho_ds_zm(1) /)

     ! ---- "CLUBB core": prescribed increments on the levels CLUBB runs (k >= top_lev) ----
     do k = top_lev, pver
        um(i,k)   = um(i,k)   + du(i,k)
        vm(i,k)   = vm(i,k)   + dv(i,k)
        thlm(i,k) = thlm(i,k) + dthl(i,k)
        rtm(i,k)  = rtm(i,k)  + drt(i,k)
        rcm(i,k)  = rcm_o(i,k)
        cloud_frac(i,k) = min(cf_o(i,k), 1._r8)    ! clubb_intr.F90:1993
     end do
     rtp2(i,:) = rtp2_o(i,:)
     include 'inc_toplev_reset.inc'
     ! ---- post-CLUBB: clubbtop, upper-level water/energy, fixer, tendencies ----
     include 'inc_post.inc'
     enddo        ! closes the "do k=1,pver" tendency loop opened in inc_post.inc
     o_clubbtop(i) = clubbtop
  end do

  ! ---- cloud fraction assembly (clubb_intr.F90:2478-2588), on the INPUT state ----
  include 'inc_alst.inc'
  include 'inc_deepcu.inc'
  include 'inc_aist.inc'
  include 'inc_ast.inc'

  ! ---- output ----
  do i = 1, ncol
     write(*,'(a,i3,i4)') 'CLUBBTOP', i, o_clubbtop(i)
     write(*,'(a,i3)') 'COL', i
     write(*,fmt) o_sfc(i,:)
     write(*,fmt) o_thlm(i,2:pver+1); write(*,fmt) o_rtm(i,2:pver+1)
     write(*,fmt) o_thv(i,2:pver+1); write(*,fmt) o_rho(i,2:pver+1)
     write(*,fmt) o_exner(i,2:pver+1); write(*,fmt) o_p(i,2:pver+1)
     write(*,fmt) ptend_loc%u(i,:); write(*,fmt) ptend_loc%v(i,:)
     write(*,fmt) ptend_loc%s(i,:)/cpair
     write(*,fmt) ptend_loc%q(i,:,ixq); write(*,fmt) ptend_loc%q(i,:,ixcldliq)
     write(*,fmt) alst(i,:); write(*,fmt) deepcu(i,:); write(*,fmt) aist(i,:)
     write(*,fmt) cloud_frac(i,1:pver)
  end do

contains

  subroutine polar(pstate, tropLev)
    type(physics_state), intent(in) :: pstate
    integer, intent(inout) :: tropLev(pcols)
    include 'inc_polar.inc'
  end subroutine polar

  subroutine zt2zm_cols()
    ! STUB for CLUBB zt2zm_api (grid_class linear_interpolated_azm).  Only
    ! rho_ds_zm(1) is consumed (surface flux scaling), and there the ghost
    ! value rho_ds_zt(1) = rho_ds_zt(2) makes the result rho_ds_zt(2) for any weight.
    integer :: kk, nz
    real(r8) :: w
    nz = nlev + 1
    do kk = 1, nz-1
       w = (zi_g(kk) - zt_g(kk)) / (zt_g(kk+1) - zt_g(kk))
       rho_ds_zm(kk) = w*(rho_ds_zt(kk+1) - rho_ds_zt(kk)) + rho_ds_zt(kk)
       thv_ds_zm(kk) = w*(thv_ds_zt(kk+1) - thv_ds_zt(kk)) + thv_ds_zt(kk)
    end do
    rho_ds_zm(nz) = rho_ds_zt(nz); thv_ds_zm(nz) = thv_ds_zt(nz)
    rho_zm = rho_ds_zm; invrs_rho_ds_zm = 0._r8; invrs_rho_ds_zm(1:nz) = 1._r8/rho_ds_zm(1:nz); wm_zm = 0._r8
  end subroutine zt2zm_cols

  function zt2zm_api(a) result(b)
    real(r8), intent(in) :: a(:)
    real(r8) :: b(size(a))
    integer :: kk, nz
    real(r8) :: w
    nz = nlev + 1
    b = a
    do kk = 1, nz-1
       w = (zi_g(kk) - zt_g(kk)) / (zt_g(kk+1) - zt_g(kk))
       b(kk) = w*(a(kk+1) - a(kk)) + a(kk)
    end do
  end function zt2zm_api
end program oracle_glue
