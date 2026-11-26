! Oracle driver: runs CAM's UNMODIFIED clubb_mf integrate_mf (atmospheric_physics
! atmos_phys0_34_002, cec8c93ed2) twice per column (ztopm1 = 500 m, ddcp = cbm1 = 0;
! then with step 1's memory), and replays CAM's own poisson
! draws (same lambda, same u seed) so the JAX reference can take them as input.
! build.sh makes the module's `poisson` PUBLIC; no other source change.
program clubb_mf_driver
  use ccpp_kinds, only: kp => kind_phys
  use physconst, only: epsilo, latvap, latice, rh2o, cpair, tmelt, h2otrip
  use wv_saturation, only: wv_sat_init
  use clubb_mf, only: clubb_mf_init, integrate_mf, poisson
  implicit none
  integer :: ncol, nzm, nzt, nup, ic, step, k, i, iu, ou
  integer :: Lopt, up_ndt, cp_ndt, kseed, fl(17), errflg
  real(kp) :: c(11), rp(15), dtime, gravit, zvir, pi
  character(len=256) :: errmsg, fin, fout
  real(kp), allocatable :: zt(:), zm(:), dzt(:), rho_zt(:), p_zt(:), iex_zt(:), u(:), v(:), thl(:), qt(:), &
       thv(:), th(:), qv(:), qc(:)
  real(kp), allocatable :: rho_zm(:), p_zm(:), iex_zm(:), w(:), thl_zm(:), qt_zm(:), thv_zm(:), th_zm(:), &
       qv_zm(:), qc_zm(:), tke(:), wpthlp(:), wpthvp(:), wpqtp(:)
  real(kp) :: s(7), cbm1, kctop
  real(kp), allocatable :: ztopm1(:), ddcp(:), mcape(:), ztop(:), dL0(:), lam(:,:)
  integer, allocatable :: enti(:,:)
  real(kp), allocatable, dimension(:,:) :: upa, dna, upw, dnw, upmf, upqt, dnqt, upthl, dnthl, upthv, dnthv, &
       upth, dnth, upqc, dnqc, upbuoy, upent, updet
  real(kp), allocatable, dimension(:) :: dry_a, moist_a, dry_w, moist_w, dry_qt, moist_qt, dry_thl, moist_thl, &
       dry_u, moist_u, dry_v, moist_v, moist_qc, ae, ac, aup, adn, aw, awup, awdn, aww, awwup, awwdn, &
       awthlup, awqtup, awuup, awvup, awthldn, awqtdn, awudn, awvdn, awthl, awqt, awu, awv, &
       thlflxup, qtflxup, uflxup, vflxup, thlflxdn, qtflxdn, uflxdn, vflxdn, thlflx, qtflx, uflx, vflx, thvflx, &
       sqtup, sthlup, sqtdn, sthldn, sqt, sthl, sac, sev, precc, mfup, entup, detup, mfdn, entdn, detdn
  real(kp) :: u_seed(4)

  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old'); open(newunit=ou, file=trim(fout), status='replace')
  read(iu,*) ncol, nzm, nup
  nzt = nzm - 1
  read(iu,*) c      ! dtime cpair gravit latvap latice zvir h2otrip pi epsilo rh2o tmelt
  dtime = c(1); cpair = c(2); gravit = c(3); latvap = c(4); latice = c(5); zvir = c(6); h2otrip = c(7)
  pi = c(8); epsilo = c(9); rh2o = c(10); tmelt = c(11)
  read(iu,*) Lopt, up_ndt, cp_ndt, kseed
  read(iu,*) rp     ! a0 b0 L0 ent0 alphturb max_L0 fdd ddalph ddbeta pwfac ddexp pwmin pwmax cldfrac_fac (15th unused)
  read(iu,*) fl     ! do_clubb_mf diag precip rad coldpool coldpool_init coldpool_perplume lscale_perplume ustar
                    ! mixd rhtke cmt addtke aloft aspd pblcull invswitch   (0/1)
  call wv_sat_init()
  call clubb_mf_init(Lopt, rp(1), rp(2), rp(3), rp(4), rp(5), nup, rp(6), rp(7), rp(8), rp(9), rp(10), rp(11), &
       up_ndt, cp_ndt, kseed, rp(12), rp(13), rp(14), &
       fl(1)==1, fl(2)==1, fl(3)==1, fl(4)==1, fl(5)==1, fl(6)==1, fl(7)==1, fl(8)==1, fl(9)==1, fl(10)==1, &
       fl(11)==1, fl(12)==1, fl(13)==1, fl(14)==1, fl(15)==1, fl(16)==1, fl(17)==1, errmsg, errflg)
  if (errflg /= 0) then
    write(0,*) trim(errmsg); error stop 2
  end if

  allocate(zt(nzt), dzt(nzt), rho_zt(nzt), p_zt(nzt), iex_zt(nzt), u(nzt), v(nzt), thl(nzt), qt(nzt), thv(nzt), &
       th(nzt), qv(nzt), qc(nzt))
  allocate(zm(nzm), rho_zm(nzm), p_zm(nzm), iex_zm(nzm), w(nzm), thl_zm(nzm), qt_zm(nzm), thv_zm(nzm), th_zm(nzm), &
       qv_zm(nzm), qc_zm(nzm), tke(nzm), wpthlp(nzm), wpthvp(nzm), wpqtp(nzm))
  allocate(ztopm1(nup), ddcp(nup), mcape(nup), ztop(nup), dL0(nup), lam(nzt,nup), enti(nzt,nup))
  allocate(upa(nzm,nup), dna(nzm,nup), upw(nzm,nup), dnw(nzm,nup), upmf(nzm,nup), upqt(nzm,nup), dnqt(nzm,nup), &
       upthl(nzm,nup), dnthl(nzm,nup), upthv(nzm,nup), dnthv(nzm,nup), upth(nzm,nup), dnth(nzm,nup), &
       upqc(nzm,nup), dnqc(nzm,nup), upbuoy(nzm,nup), upent(nzm,nup), updet(nzm,nup))
  allocate(dry_a(nzm), moist_a(nzm), dry_w(nzm), moist_w(nzm), dry_qt(nzm), moist_qt(nzm), dry_thl(nzm), &
       moist_thl(nzm), dry_u(nzm), moist_u(nzm), dry_v(nzm), moist_v(nzm), moist_qc(nzm), ae(nzm), ac(nzm), &
       aup(nzm), adn(nzm), aw(nzm), awup(nzm), awdn(nzm), aww(nzm), awwup(nzm), awwdn(nzm), awthlup(nzm), &
       awqtup(nzm), awuup(nzm), awvup(nzm), awthldn(nzm), awqtdn(nzm), awudn(nzm), awvdn(nzm), awthl(nzm), &
       awqt(nzm), awu(nzm), awv(nzm), thlflxup(nzm), qtflxup(nzm), uflxup(nzm), vflxup(nzm), thlflxdn(nzm), &
       qtflxdn(nzm), uflxdn(nzm), vflxdn(nzm), thlflx(nzm), qtflx(nzm), uflx(nzm), vflx(nzm), thvflx(nzm), &
       sqtup(nzt), sthlup(nzt), sqtdn(nzt), sthldn(nzt), sqt(nzt), sthl(nzt), sac(nzt), sev(nzt), precc(nzm), &
       mfup(nzm), entup(nzm), detup(nzm), mfdn(nzm), entdn(nzm), detdn(nzm))

  write(ou,*) ncol, nzm, nup
  do ic = 1, ncol
    read(iu,*) zt; read(iu,*) dzt; read(iu,*) rho_zt; read(iu,*) p_zt; read(iu,*) iex_zt
    read(iu,*) u; read(iu,*) v; read(iu,*) thl; read(iu,*) qt; read(iu,*) thv; read(iu,*) th; read(iu,*) qv
    read(iu,*) qc
    read(iu,*) zm; read(iu,*) rho_zm; read(iu,*) p_zm; read(iu,*) iex_zm; read(iu,*) w; read(iu,*) thl_zm
    read(iu,*) qt_zm; read(iu,*) thv_zm; read(iu,*) th_zm; read(iu,*) qv_zm; read(iu,*) qc_zm; read(iu,*) tke
    read(iu,*) wpthlp; read(iu,*) wpthvp; read(iu,*) wpqtp
    read(iu,*) s      ! ustar ths wthl_sfc wqt_sfc pblh tpert lts
    ! Call 1 starts from a NONZERO plume-top memory (vote V26, not CAM's 0 cold start) so a
    ! culled plume's kept memory is a nonzero value; call 2 starts from call 1's output.
    ztopm1 = 500._kp; ddcp = 0._kp; cbm1 = 0._kp
    do step = 1, 2
      call wo1(ztopm1); call wo1(ddcp); call wo0(cbm1)          ! memory going IN
      call integrate_mf(nzm, nzt, dtime, cpair, gravit, latvap, latice, zvir, h2otrip, pi, &
           rho_zm, zm, p_zm, iex_zm, rho_zt, dzt, zt, p_zt, iex_zt, u, v, thl, qt, thv, w, th, qv, qc, &
           thl_zm, qt_zm, thv_zm, th_zm, qv_zm, qc_zm, s(1), s(2), s(3), s(4), s(5), tke, s(6), s(7), &
           wpthlp, wpthvp, wpqtp, ztopm1, ddcp, cbm1, mcape, upa, dna, upw, dnw, upmf, upqt, dnqt, &
           upthl, dnthl, upthv, dnthv, upth, dnth, upqc, dnqc, upbuoy, upent, updet, &
           dry_a, moist_a, dry_w, moist_w, dry_qt, moist_qt, dry_thl, moist_thl, dry_u, moist_u, dry_v, moist_v, &
           moist_qc, ae, ac, aup, adn, aw, awup, awdn, aww, awwup, awwdn, awthlup, awqtup, awuup, awvup, &
           awthldn, awqtdn, awudn, awvdn, awthl, awqt, awu, awv, thlflxup, qtflxup, uflxup, vflxup, &
           thlflxdn, qtflxdn, uflxdn, vflxdn, thlflx, qtflx, uflx, vflx, thvflx, sqtup, sthlup, sqtdn, sthldn, &
           sqt, sthl, sac, sev, precc, ztop, dL0, mfup, entup, detup, mfdn, entdn, detdn, kctop)
      ! Replay CAM's draws: integrate_mf's entf (dzt/dynamic_L0, 0 at the top level) and u seed.
      enti = 0
      if (any(dL0 > 0._kp)) then
        lam = 0._kp
        do i = 1, nup
          do k = 1, nzt - 1
            lam(k,i) = dzt(k) / dL0(i)
          end do
        end do
        u_seed(1) = u(1); u_seed(2) = u(1 + kseed); u_seed(3) = u(2 + kseed); u_seed(4) = u(3 + kseed)
        call poisson(nzt, nup, 1, nzt, 1, lam, enti, u_seed, pi)
      end if
      call wo2(upa); call wo2(upw); call wo2(upthl); call wo2(upqt); call wo2(upthv); call wo2(upqc)
      call wo2(upth); call wo2(upent); call wo2(updet)
      call wo1(thlflx); call wo1(qtflx); call wo1(uflx); call wo1(vflx); call wo1(thvflx)
      call wo1(dL0); call wo1(ztopm1); call wo1(ddcp); call wo0(cbm1); call wo1(sqt); call wo1(sthl)
      write(ou,'(*(i0,1x))') enti
    end do
  end do
  write(ou,*) 'DRIVER_OK'
contains
  subroutine wo0(x)
    real(kp), intent(in) :: x
    write(ou,'(es26.17e3)') x
  end subroutine wo0
  subroutine wo1(x)
    real(kp), intent(in) :: x(:)
    write(ou,'(*(es26.17e3,1x))') x
  end subroutine wo1
  subroutine wo2(x)
    real(kp), intent(in) :: x(:,:)
    write(ou,'(*(es26.17e3,1x))') x
  end subroutine wo2
end program clubb_mf_driver
