! Standalone driver reproducing CAM6 zm_conv_tend (zm_conv_intr.F90:366-918)
! with the pbuf/outfld/ptend plumbing replaced by local arrays:
!   zm_convr -> physics_update(T,q) -> zm_conv_evap -> physics_update ->
!   momtran(u,v) -> convtran(CLDLIQ, CLDICE, NUMLIQ, NUMICE, RAINQM, SNOWQM,
!   NUMRAI): the MG2 is_convtran1 constituents legoESM carries (micro_mg_cam.F90:416-435)
! Input/outputs are plain text (list-directed); see make_fixture.py.
program zm_driver
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, pverp
  use physconst, only: set_constants, cpair, gravit, tmelt, epsilo, h2otrip
  use wv_saturation, only: wv_sat_init_stub
  use zm_microphysics, only: zm_aero_t, zm_conv_t
  use zm_conv, only: zm_convi, zm_convr, zm_conv_evap, momtran, convtran, &
                     zm_ient_tol, zm_ient_eps, dbg_mb, dbg_tl, dbg_tp, dbg_lcl, dbg_lel, &
                     dbg_mx, dbg_jd, dbg_j0, dbg_jlcl, dbg_lengath, dbg_nobracket
  implicit none

  integer :: ncol, nlev, const_mode, sat_mode, num_cin, i, k, ii, limcnv, lengath, iu
  real(r8) :: ztodt, ient_tol, ient_eps, c0_lnd, c0_ocn, ke, ke_lnd, momcu, momcd
  character(len=512) :: fin, fout
  real(r8) :: pref_edge(pverp)
  real(r8), dimension(pcols) :: landfrac, pblh, tpert, phis
  real(r8), dimension(pcols,pver) :: t, q, pmid, pdel, zm, u, v, cld, qc, qi
  real(r8), dimension(pcols,pver) :: nc, ni, qr, qsn, nr
  real(r8), dimension(pcols,pverp) :: pint, zi
  ! zm_convr outputs
  real(r8), dimension(pcols) :: prec, jctop, jcbot, cape, dsubcld, rliq, rice
  real(r8), dimension(pcols,pver) :: qtnd, heat, dlf, cme, zdu, rprd, ql, dif, dnlf, dnif
  real(r8), dimension(pcols,pver) :: mu, md, du, eu, ed, dp
  real(r8), dimension(pcols,pverp) :: mcon, pflx
  integer, dimension(pcols) :: jt, maxg, ideep
  real(r8), pointer :: org(:,:), orgt(:,:), org2d(:,:)
  type(zm_conv_t) :: conv
  type(zm_aero_t) :: aero
  ! evap
  real(r8), dimension(pcols,pver) :: t1, q1, tend_s, tend_q, snwprd, snwevmlt, ntprprd, ntsnprd
  real(r8), dimension(pcols,pverp) :: flxprec, flxsnow
  real(r8), dimension(pcols) :: snow, prec_convr
  ! momtran
  real(r8) :: winds(pcols,pver,2), wind_tends(pcols,pver,2), pguall(pcols,pver,2), pgdall(pcols,pver,2)
  real(r8) :: icwu(pcols,pver,2), icwd(pcols,pver,2), seten(pcols,pver)
  logical :: l_windt(2)
  ! convtran on [Q, CLDLIQ, CLDICE, NUMLIQ, NUMICE, RAINQM, SNOWQM, NUMRAI];
  ! Q itself is not transported (lq(1)=.false.)
  integer, parameter :: ntr = 8
  real(r8) :: qall(pcols,pver,ntr), dqall(pcols,pver,ntr), fracis(pcols,pver,ntr), fake_dpdry(pcols,pver)
  logical :: lq(ntr)
  ! scattered gathered fields
  real(r8), dimension(pcols,pver) :: mu_s, md_s, du_s, eu_s, ed_s, dp_s
  real(r8), dimension(pcols) :: dsub_s
  integer, dimension(pcols) :: jt_s, mx_s

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old', action='read')
  read(iu,*) ncol, nlev
  if (nlev /= pver .or. ncol > pcols) then
     write(*,*) 'dimension mismatch', ncol, nlev, pcols, pver; stop 3
  end if
  read(iu,*) ztodt
  read(iu,*) const_mode, sat_mode, ient_tol, ient_eps
  read(iu,*) c0_lnd, c0_ocn, ke, ke_lnd, momcu, momcd, num_cin
  read(iu,*) pref_edge
  t = 0; q = 0; pmid = 0; pdel = 0; zm = 0; u = 0; v = 0; cld = 0; qc = 0; qi = 0
  nc = 0; ni = 0; qr = 0; qsn = 0; nr = 0
  pint = 0; zi = 0; landfrac = 0; pblh = 0; tpert = 0; phis = 0
  do i = 1, ncol
     read(iu,*) landfrac(i), pblh(i), tpert(i)
     read(iu,*) t(i,:)
     read(iu,*) q(i,:)
     read(iu,*) pmid(i,:)
     read(iu,*) pint(i,:)
     read(iu,*) pdel(i,:)
     read(iu,*) zm(i,:)
     read(iu,*) zi(i,:)
     read(iu,*) u(i,:)
     read(iu,*) v(i,:)
     read(iu,*) cld(i,:)
     read(iu,*) qc(i,:)
     read(iu,*) qi(i,:)
     read(iu,*) nc(i,:)
     read(iu,*) ni(i,:)
     read(iu,*) qr(i,:)
     read(iu,*) qsn(i,:)
     read(iu,*) nr(i,:)
  end do
  close(iu)
  ! pad unused columns with copies of column 1 (never read: loops run to ncol)
  do i = ncol+1, pcols
     t(i,:) = t(1,:); q(i,:) = q(1,:); pmid(i,:) = pmid(1,:); pint(i,:) = pint(1,:)
     pdel(i,:) = pdel(1,:); zm(i,:) = zm(1,:); zi(i,:) = zi(1,:)
  end do

  call set_constants(const_mode)
  call wv_sat_init_stub(sat_mode, tmelt, h2otrip, epsilo)
  zm_ient_tol = ient_tol
  zm_ient_eps = ient_eps

  ! zm_conv_init (zm_conv_intr.F90:334-345)
  limcnv = 0
  if (pref_edge(1) >= 4.e3_r8) then
     limcnv = 1
  else
     do k = 1, pver
        if (pref_edge(k) < 4.e3_r8 .and. pref_edge(k+1) >= 4.e3_r8) then
           limcnv = k
           exit
        end if
     end do
     if (limcnv == 0) limcnv = pverp
  end if
  call zm_convi(limcnv, c0_lnd, c0_ocn, ke, ke_lnd, momcu, momcd, num_cin, .false., &
                .false., no_deep_pbl_in=.false.)

  nullify(org, orgt, org2d)
  call zm_convr(1, ncol, &
                t, q, prec, jctop, jcbot, &
                pblh, zm, phis, zi, qtnd, &
                heat, pmid, pint, pdel, &
                0.5_r8*ztodt, mcon, cme, cape, &
                tpert, dlf, pflx, zdu, rprd, &
                mu, md, du, eu, ed, &
                dp, dsubcld, jt, maxg, ideep, &
                ql, rliq, landfrac, &
                org, orgt, org2d, &
                dif, dnlf, dnif, conv, &
                aero, rice)
  lengath = count(ideep > 0)
  prec_convr = prec

  ! physics_update(state1, ptend_loc, ztodt): T += s*dt/cpair, Q += q*dt then qneg3 (qmin 1e-12)
  t1 = t + heat*ztodt/cpair
  q1 = max(q + qtnd*ztodt, 1.e-12_r8)

  tend_s = 0; tend_q = 0
  call zm_conv_evap(ncol, 1, t1, pmid, pdel, q1, landfrac, &
                    tend_s, snwprd, snwevmlt, tend_q, &
                    rprd, cld, ztodt, &
                    prec, snow, ntprprd, ntsnprd, flxprec, flxsnow, conv%sprd)

  winds(:,:,1) = u
  winds(:,:,2) = v
  l_windt = .true.
  call momtran(1, ncol, l_windt, winds, 2, mu, md, du, eu, ed, dp, dsubcld, &
               jt, maxg, ideep, 1, lengath, 0, wind_tends, pguall, pgdall, icwu, icwd, ztodt, seten)

  qall(:,:,1) = max(q + (qtnd + tend_q)*ztodt, 1.e-12_r8)
  qall(:,:,2) = qc
  qall(:,:,3) = qi
  qall(:,:,4) = nc
  qall(:,:,5) = ni
  qall(:,:,6) = qr
  qall(:,:,7) = qsn
  qall(:,:,8) = nr
  lq = .true.
  lq(1) = .false.
  fracis = 1._r8
  fake_dpdry = 0._r8
  dqall = 0._r8
  call convtran(1, lq, qall, ntr, mu, md, du, eu, ed, dp, dsubcld, &
                jt, maxg, ideep, 1, lengath, 0, fracis, dqall, fake_dpdry, ztodt)

  ! scatter the gathered cloud-model arrays
  mu_s = 0; md_s = 0; du_s = 0; eu_s = 0; ed_s = 0; dp_s = 0; dsub_s = 0; jt_s = 0; mx_s = 0
  do ii = 1, lengath
     i = ideep(ii)
     mu_s(i,:) = mu(ii,:); md_s(i,:) = md(ii,:); du_s(i,:) = du(ii,:)
     eu_s(i,:) = eu(ii,:); ed_s(i,:) = ed(ii,:); dp_s(i,:) = dp(ii,:)
     dsub_s(i) = dsubcld(ii); jt_s(i) = jt(ii); mx_s(i) = maxg(ii)
  end do

  open(newunit=iu, file=trim(fout), status='replace', action='write')
  write(iu,'(A,1X,I6,1X,I6,1X,I6)') 'DIMS', ncol, pver, lengath
  write(iu,'(A,1X,I6)') 'LIMCNV', limcnv
  call w1('cape', cape); call w1('prec_convr', prec_convr); call w1('prec', prec)
  call w1('snow', snow); call w1('rliq', rliq); call w1('jctop', jctop); call w1('jcbot', jcbot)
  call w1('dsubcld', dsub_s); call w1('mb', dbg_mb); call w1('tl', dbg_tl)
  call wi('jt', jt_s); call wi('mx', mx_s); call wi('lcl', dbg_lcl); call wi('lel', dbg_lel)
  call wi('mx_buoy', dbg_mx); call wi('jd', dbg_jd); call wi('j0', dbg_j0); call wi('jlcl', dbg_jlcl)
  call wi('ideep', ideep)
  do k = 1, 4
     write(fin,'(A,I1)') 'nobracket_call', k
     call wi(trim(fin), dbg_nobracket(:,k))
  end do
  call w2('qtnd', qtnd); call w2('heat', heat); call w2('dlf', dlf); call w2('cme', cme)
  call w2('zdu', zdu); call w2('rprd', rprd); call w2('ql', ql); call w2('tp', dbg_tp)
  call w2('mu', mu_s); call w2('md', md_s); call w2('du', du_s); call w2('eu', eu_s)
  call w2('ed', ed_s); call w2('dp', dp_s)
  call w2p('mcon', mcon); call w2p('pflx', pflx)
  call w2('evap_tend_s', tend_s); call w2('evap_tend_q', tend_q); call w2('snwprd', snwprd)
  call w2('snwevmlt', snwevmlt); call w2('ntprprd', ntprprd); call w2('ntsnprd', ntsnprd)
  call w2p('flxprec', flxprec); call w2p('flxsnow', flxsnow)
  call w2('dudt', wind_tends(:,:,1)); call w2('dvdt', wind_tends(:,:,2)); call w2('seten', seten)
  call w2('pgu_u', pguall(:,:,1)); call w2('pgd_u', pgdall(:,:,1))
  call w2('icwu_u', icwu(:,:,1)); call w2('icwd_u', icwd(:,:,1))
  call w2('dqc_convtran', dqall(:,:,2)); call w2('dqi_convtran', dqall(:,:,3))
  call w2('dnc_convtran', dqall(:,:,4)); call w2('dni_convtran', dqall(:,:,5))
  call w2('dqr_convtran', dqall(:,:,6)); call w2('dqs_convtran', dqall(:,:,7))
  call w2('dnr_convtran', dqall(:,:,8))
  close(iu)
  write(*,*) 'ORACLE_DONE lengath=', lengath, ' limcnv=', limcnv

contains
  subroutine w1(name, a)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: a(:)
    write(iu,'(A,1X,A,1X,I6)') 'F1', name, ncol
    write(iu,'(ES25.17E3)') a(1:ncol)
  end subroutine w1
  subroutine wi(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    write(iu,'(A,1X,A,1X,I6)') 'I1', name, ncol
    write(iu,'(I8)') a(1:ncol)
  end subroutine wi
  subroutine w2(name, a)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: a(:,:)
    integer :: j
    write(iu,'(A,1X,A,1X,I6,1X,I6)') 'F2', name, ncol, size(a,2)
    do j = 1, ncol
       write(iu,'(ES25.17E3)') a(j,:)
    end do
  end subroutine w2
  subroutine w2p(name, a)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: a(:,:)
    call w2(name, a)
  end subroutine w2p
end program zm_driver
