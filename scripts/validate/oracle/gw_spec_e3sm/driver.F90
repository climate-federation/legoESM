! Standalone driver for the E3SM v3.0.1 gw_tend SPECTRAL (+ optional orographic)
! branches, built against UNMODIFIED E3SM v3.0.1 gw_common / gw_front /
! gw_convect / gw_oro / gw_diffusion / vdiff_lu_solver / gw_utils and
! components/eam/src/control/interpolate_data.F90 (lininterp).
!
! Mirrors components/eam/src/physics/cam/gw_drag.F90 (tag v3.0.1, d4ca7d06):
!   gw_init  :287-288 (do_spectral_waves / orographic_only), :291-344 (cref,
!             alpha profile via lininterp on pref_edge / alpha=1e-6),
!             :346 kbotbg, :417-420 kfront, :453-456 k700
!   gw_tend  :722-724 gw_prof; :749-801 Beres; :803-866 frontal (C&M);
!             :871-938 orographic (closure block INCLUDED VERBATIM from :901-938)
! Infrastructure replaced: pbuf (frontgf / ttend_dp read from file), outfld
! (text dump), physics_state (plain arrays), cpairv(:,:,lchnk) = cpair,
! dycore_is('UNSTRUCTURED') -> input flag do_taper_front.
program gw_spec_driver
  use gw_utils, only: r8
  use gw_common, only: gw_common_init, gw_prof, gw_drag_prof, &
       momentum_energy_conservation, orographic_only
  use gw_front, only: gw_front_init, gw_cm_src
  use gw_convect, only: gw_convect_init, gw_beres_src
  use gw_oro, only: gw_oro_init, gw_oro_src
  use interpolate_data, only: lininterp
  implicit none
  integer :: ncol, pver, pgwv, i, k, l, iu, ou
  integer :: iconv, ifront, ioro, ifix, alpha_mode, itaper, kmode
  logical :: use_gw_convect, use_gw_front, use_gw_oro, use_gw_energy_fix
  logical :: do_spectral_waves, do_latitude_taper
  integer :: kbotbg, kfront, k700, maxh, maxuh, kbotbg_e, kfront_e, k700_e
  integer, parameter :: ktop = 0, lchnk = 1
  real(r8) :: gravit, rair, cpair, dt, dc, kwv, fcrit2
  real(r8) :: effgw_beres, effgw_cm, effgw_oro, taubgnd, frontgfc, hcf, hdsf, alpha_uniform
  integer, parameter :: nalph=66
  include 'alpha_table.inc'
  real(r8), allocatable :: pref_edge(:), alpha(:), cref(:), mfcc(:,:,:)
  real(r8), allocatable :: pint(:,:), pmid(:,:), t(:,:), u(:,:), v(:,:), zm(:,:), dse(:,:)
  real(r8), allocatable :: frontgf(:,:), ttend_dp(:,:), q(:,:,:), lat(:), sgh(:)
  real(r8), allocatable :: piln(:,:), dpm(:,:), rdpm(:,:), rhoi(:,:), ti(:,:), nm(:,:), ni(:,:)
  real(r8), allocatable :: tau(:,:,:), ubm(:,:), ubi(:,:), xv(:), yv(:), c(:,:)
  real(r8), allocatable :: utgw(:,:), vtgw(:,:), ttgw(:,:), qtgw(:,:,:), taucd(:,:,:)
  real(r8), allocatable :: egwdffi(:,:), gwut(:,:,:), dttdf(:,:), dttke(:,:), kvtt(:,:)
  real(r8), allocatable :: hdepth(:), maxq0(:), dE(:), cpairv(:,:,:)
  integer, allocatable :: src_level(:), tend_level(:)
  type ptend_t
     real(r8), allocatable :: u(:,:), v(:,:), s(:,:)
  end type ptend_t
  type cam_in_t
     real(r8), allocatable :: landfrac(:)
  end type cam_in_t
  type(ptend_t) :: ptend
  type(cam_in_t) :: cam_in
  character(len=512) :: errstring, fin, fout

  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old', action='read')
  read(iu,*) ncol, pver, pgwv
  read(iu,*) gravit, rair, cpair, dt
  read(iu,*) dc, kwv, fcrit2
  read(iu,*) iconv, ifront, ioro, ifix, itaper
  read(iu,*) effgw_beres, effgw_cm, effgw_oro
  read(iu,*) taubgnd, frontgfc, hcf, hdsf
  read(iu,*) alpha_mode, alpha_uniform
  read(iu,*) kmode, kbotbg, kfront, k700
  allocate(pref_edge(pver+1), alpha(0:pver), cref(-pgwv:pgwv))
  read(iu,*) pref_edge
  use_gw_convect = iconv == 1; use_gw_front = ifront == 1
  use_gw_oro = ioro == 1; use_gw_energy_fix = ifix == 1
  if (use_gw_convect) then
     read(iu,*) maxh, maxuh
     allocate(mfcc(maxh,-maxuh:maxuh,-pgwv:pgwv))
     read(iu,*) mfcc
  end if
  allocate(pint(ncol,pver+1), pmid(ncol,pver), t(ncol,pver), u(ncol,pver), v(ncol,pver), &
       zm(ncol,pver), dse(ncol,pver), frontgf(ncol,pver), ttend_dp(ncol,pver), &
       q(ncol,pver,1), lat(ncol), sgh(ncol))
  allocate(cam_in%landfrac(ncol))
  do i = 1, ncol
     read(iu,*) lat(i), sgh(i), cam_in%landfrac(i)
     read(iu,*) pint(i,:)
     read(iu,*) pmid(i,:)
     read(iu,*) t(i,:)
     read(iu,*) u(i,:)
     read(iu,*) v(i,:)
     read(iu,*) zm(i,:)
     read(iu,*) dse(i,:)
     read(iu,*) frontgf(i,:)
     read(iu,*) ttend_dp(i,:)
     read(iu,*) q(i,:,1)
  end do
  close(iu)

  ! ---- gw_init (gw_drag.F90:287-344) ----
  do_spectral_waves = (pgwv > 0 .and. (use_gw_front .or. use_gw_convect))
  orographic_only = (use_gw_oro .and. .not. do_spectral_waves)
  cref = (/ (dc * l, l = -pgwv, pgwv) /)
  if (.not. orographic_only) then
     do k=1,nalph
        alpha0(k) = alpha0(k) / 86400._r8
        alpha0(k) = max(alpha0(k), 1.e-6_r8)
        palph(k) = palph(k)*1.e2_r8
     end do
     call lininterp (alpha0  ,palph, nalph , alpha  , pref_edge , pver+1)
  else
     alpha = 1.e-6_r8
  end if
  if (alpha_mode == 0) alpha = alpha_uniform
  kbotbg_e = maxloc(pref_edge, 1, (pref_edge < 50000._r8)) - 1
  kfront_e = -1; k700_e = -1
  do k = 0, pver
     if (pref_edge(k+1) < 60000._r8) kfront_e = k+1
  end do
  do k = 0, pver
     if (pref_edge(k+1) < 70000._r8) k700_e = k+1
  end do
  if (kmode == 0) then
     kbotbg = kbotbg_e; kfront = kfront_e; k700 = k700_e
  end if
  call gw_common_init(pver, pgwv, dc, cref, .false., .false., 0, ktop, kbotbg, &
       fcrit2, kwv, gravit, rair, alpha, errstring)
  if (trim(errstring) /= "") stop 'gw_common_init'
  if (use_gw_front) then
     call gw_front_init(taubgnd, frontgfc, kfront, errstring)
     if (trim(errstring) /= "") stop 'gw_front_init'
  end if
  if (use_gw_convect) then
     call gw_convect_init(k700, mfcc, errstring)
     if (trim(errstring) /= "") stop 'gw_convect_init'
  end if
  if (use_gw_oro) then
     call gw_oro_init(errstring)
     if (trim(errstring) /= "") stop 'gw_oro_init'
  end if

  allocate(piln(ncol,pver+1), dpm(ncol,pver), rdpm(ncol,pver), rhoi(ncol,0:pver), &
       ti(ncol,0:pver), nm(ncol,pver), ni(ncol,0:pver), tau(ncol,-pgwv:pgwv,0:pver), &
       ubm(ncol,pver), ubi(ncol,0:pver), xv(ncol), yv(ncol), c(ncol,-pgwv:pgwv), &
       utgw(ncol,pver), vtgw(ncol,pver), ttgw(ncol,pver), qtgw(ncol,pver,1), &
       taucd(ncol,0:pver,4), egwdffi(ncol,0:pver), gwut(ncol,pver,-pgwv:pgwv), &
       dttdf(ncol,pver), dttke(ncol,pver), kvtt(ncol,0:pver), hdepth(ncol), maxq0(ncol), &
       dE(ncol), cpairv(ncol,pver,1), src_level(ncol), tend_level(ncol))
  allocate(ptend%u(ncol,pver), ptend%v(ncol,pver), ptend%s(ncol,pver))
  ! state1%lnpint / pdel / rpdel equivalents
  piln = log(pint)
  dpm = pint(:,2:pver+1) - pint(:,1:pver)
  rdpm = 1._r8/dpm
  kvtt = 0._r8
  cpairv = cpair
  ptend%u = 0._r8; ptend%v = 0._r8; ptend%s = 0._r8
  ubm = 0._r8; ubi = 0._r8

  open(newunit=ou, file=trim(fout), status='replace', action='write')
  write(ou,'(a,4i6)') 'hdr', ncol, pver, pgwv, merge(1,0,orographic_only)
  write(ou,'(a,6i6)') 'kidx', kbotbg, kfront, k700, kbotbg_e, kfront_e, k700_e
  write(ou,'(a,1000es25.16e3)') 'alpha', alpha

  call gw_prof(ncol, cpair, t, pmid, pint, rhoi, ti, nm, ni)
  call dump2('rhoi', rhoi); call dump2('ni', ni); call dump2('nm', nm)

  if (do_spectral_waves) then
     if (use_gw_convect) then
        call gw_beres_src(ncol, pgwv, lat, u, v, ttend_dp, &
             zm, src_level, tend_level, tau, ubm, ubi, xv, yv, c, &
             hdepth, maxq0, hcf, hdsf)
        do_latitude_taper = .false.
        call dumpsrc('beres')
        call gw_drag_prof(ncol, pgwv, src_level, tend_level, do_latitude_taper, dt, &
             lat, t,    ti, pmid, pint, dpm,   rdpm, &
             piln, rhoi,       nm,   ni, ubm,  ubi,  xv,    yv,   &
             effgw_beres, c,   kvtt, q,  dse,  tau,  utgw,  vtgw, &
             ttgw, qtgw,  taucd,     egwdffi,  gwut, dttdf, dttke)
        do k = 1, pver
           ptend%u(:ncol,k) = utgw(:,k)
           ptend%v(:ncol,k) = vtgw(:,k)
           ptend%s(:ncol,k) = ttgw(:,k)
        end do
        call dumpprof('beres_pre')
        call momentum_energy_conservation(ncol, tend_level, dt, taucd, &
             pint, dpm, u, v, ptend%u, ptend%v, ptend%s, utgw, vtgw, ttgw)
        call dumpprof('beres')
        write(ou,'(a,1000es25.16e3)') 'beres_hdepth', hdepth
        write(ou,'(a,1000es25.16e3)') 'beres_maxq0', maxq0
     end if
     if (use_gw_front) then
        call gw_cm_src(ncol, pgwv, kbotbg, u, v, frontgf, &
             src_level, tend_level, tau, ubm, ubi, xv, yv, c)
        do_latitude_taper = (itaper == 1)
        call dumpsrc('front')
        call gw_drag_prof(ncol, pgwv, src_level, tend_level, do_latitude_taper, dt, &
             lat, t,    ti, pmid, pint, dpm,   rdpm, &
             piln, rhoi,       nm,   ni, ubm,  ubi,  xv,    yv,   &
             effgw_cm,    c,   kvtt, q,  dse,  tau,  utgw,  vtgw, &
             ttgw, qtgw,  taucd,     egwdffi,  gwut, dttdf, dttke)
        do k = 1, pver
           ptend%u(:ncol,k) = ptend%u(:ncol,k) + utgw(:,k)
           ptend%v(:ncol,k) = ptend%v(:ncol,k) + vtgw(:,k)
           ptend%s(:ncol,k) = ptend%s(:ncol,k) + ttgw(:,k)
        end do
        call dumpprof('front_pre')
        call momentum_energy_conservation(ncol, tend_level, dt, taucd, &
             pint, dpm, u, v, ptend%u, ptend%v, ptend%s, utgw, vtgw, ttgw)
        call dumpprof('front')
     end if
  end if

  if (use_gw_oro) then
     call gw_oro_src(ncol, &
          u, v, t, sgh(:ncol), pmid, pint, dpm, zm, nm, &
          src_level, tend_level, tau, ubm, ubi, xv, yv, c)
     do_latitude_taper = .false.
     call gw_drag_prof(ncol, 0, src_level, tend_level, do_latitude_taper, dt, &
          lat, t,    ti, pmid, pint, dpm,   rdpm, &
          piln, rhoi,       nm,   ni, ubm,  ubi,  xv,    yv,   &
          effgw_oro,   c,   kvtt, q,  dse,  tau,  utgw,  vtgw, &
          ttgw, qtgw,  taucd,     egwdffi,  gwut(:,:,0:0), dttdf, dttke)
     include 'oro_closure.inc'
     do i = 1, ncol
        write(ou,'(a,i4,2i6)') 'oro_lev', i, src_level(i), tend_level(i)
     end do
     call dump2('oro_utgw', utgw); call dump2('oro_vtgw', vtgw); call dump2('oro_ttgw', ttgw)
  end if

  call dump2('ptend_u', ptend%u); call dump2('ptend_v', ptend%v); call dump2('ptend_s', ptend%s)
  close(ou)
  write(*,*) 'E3SM_SPEC_ORACLE_OK', ncol, pver, pgwv, orographic_only, kbotbg, kfront, k700

contains

  subroutine dump2(name, a)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: a(:,:)
    integer :: ii
    do ii = 1, size(a,1)
       write(ou,'(a,1x,i4,1x,1000es25.16e3)') name, ii, a(ii,:)
    end do
  end subroutine dump2

  subroutine dumpsrc(pf)
    character(len=*), intent(in) :: pf
    integer :: ii, ll
    do ii = 1, ncol
       write(ou,'(a,1x,i4,2i6,2es25.16e3)') pf//'_lev', ii, src_level(ii), tend_level(ii), xv(ii), yv(ii)
       write(ou,'(a,1x,i4,1x,1000es25.16e3)') pf//'_c', ii, c(ii,:)
       write(ou,'(a,1x,i4,1x,1000es25.16e3)') pf//'_ubi_src', ii, ubi(ii,0:src_level(ii))
       do ll = -pgwv, pgwv
          write(ou,'(a,1x,i4,i5,1x,1000es25.16e3)') pf//'_tau0', ii, ll, tau(ii,ll,:)
       end do
    end do
  end subroutine dumpsrc

  subroutine dumpprof(pf)
    character(len=*), intent(in) :: pf
    integer :: ii, ll
    call dump2(pf//'_utgw', utgw); call dump2(pf//'_vtgw', vtgw); call dump2(pf//'_ttgw', ttgw)
    call dump2(pf//'_dttdf', dttdf); call dump2(pf//'_dttke', dttke)
    call dump2(pf//'_egwdffi', egwdffi)
    call dump2(pf//'_taucd_ew', taucd(:,:,1) + taucd(:,:,2))
    call dump2(pf//'_taucd_ns', taucd(:,:,3) + taucd(:,:,4))
    do ii = 1, ncol
       do ll = -pgwv, pgwv
          write(ou,'(a,1x,i4,i5,1x,1000es25.16e3)') pf//'_tau', ii, ll, tau(ii,ll,:)
       end do
    end do
  end subroutine dumpprof

end program gw_spec_driver
