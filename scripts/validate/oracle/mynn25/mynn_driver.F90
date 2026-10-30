! Oracle driver for WRF v4.8.0 MYNN-EDMF (phys/MYNN-EDMF/module_bl_mynnedmf.F90,
! submodule 8f8e8b59021249949c04d1686edc5a02ca325627), level-2.5 closure with the
! WRF namelist defaults for every other option (Registry.EM_COMMON):
!   bl_mynn_tkeadvect=.false., tke_budget=0, bl_mynn_cloudpdf=2,
!   bl_mynn_mixlength=1, bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0,
!   bl_mynn_edmf_dd=0, bl_mynn_mixscalars=0, bl_mynn_mixaerosols=1,
!   bl_mynn_mixnumcon=0, bl_mynn_output=0, bl_mynn_cloudmix=1, bl_mynn_mixqt=0,
!   bl_mynn_ess=1, spp_pbl=0, no chemistry; FLAG_QC=FLAG_QI=.true., all other
!   species flags .false. (FLAG_OZONE is a .false. parameter in the WRF driver).
! The column core ``mynnedmf`` is called directly with specific contents (the
! WRF driver's mixing-ratio <-> specific conversion is host glue, not called).
!
! Input (stdin, list-directed): ncol nlev nstep, then per column
!   dt dx xland ps ts ust hfx qfx wspd
!   then bottom-up level arrays: dz u v w th p exner rho sqv sqc sqi
! Each column is integrated nstep steps: step 1 is a cold start (initflag=1),
! later steps initflag=0 with the turbulence state carried and u, v, th, sqv,
! sqc, sqi advanced by the scheme's own tendencies (tk = th*exner).
! Output (stdout): for every column and every step, the inputs the step saw
! (state and carried turbulence state) and every output, as "name n v1 .. vn".
program mynn_driver
  use ccpp_kind_types, only: kind_phys
  use module_bl_mynnedmf, only: mynnedmf
#ifdef MYNN_DUMP
  use mynn_dump_mod, only: mynn_dump_on
#endif
  implicit none
  integer :: ncol, nlev, nstep, ic, is, kts, kte
  real(kind_phys) :: dt, dx, xland, ps, ts, ust, hfx, qfx, wspd
  real(kind_phys) :: qsfc, ch, znt, uoce, voce
  real(kind_phys), allocatable, dimension(:) :: dz, u, v, w, th, p, ex, rho, tk, &
       sqv, sqc, sqi, sqs, qnc, qni, qnwfa, qnifa, qnbca, ozone, rthraten, &
       qke, qke_adv, el, sh, sm, kh, km, tsq, qsq, cov, &
       du, dv, dth, dqv, dqc, dqi, dqnc, dqni, dqs, dqnwfa, dqnifa, dqnbca, dozone, &
       dqke, qwt, qshear, qbuoy, qdiss, qc_bl, qi_bl, cldfra_bl, &
       edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, &
       sub_thl, sub_sqv, det_thl, det_sqv, pattern_spp
  real(kind_phys), allocatable :: chem(:,:), settle(:,:), scalars(:,:)
  real(kind_phys) :: vdep(1), frp, emis_ant_no
  real(kind_phys) :: pblh, excess_h, excess_q, maxmf, maxwidth, ztop_plume, &
       maxmf_dd, maxwidth_dd, maxtkeprod, cldtop_cooling, ent_eff
  integer :: kpbl, initflag, errflg
  character(len=256) :: errmsg

  read(*,*) ncol, nlev, nstep
  kts = 1
  kte = nlev
  allocate(dz(nlev), u(nlev), v(nlev), w(nlev), th(nlev), p(nlev), ex(nlev), &
       rho(nlev), tk(nlev), sqv(nlev), sqc(nlev), sqi(nlev), sqs(nlev), qnc(nlev), &
       qni(nlev), qnwfa(nlev), qnifa(nlev), qnbca(nlev), ozone(nlev), rthraten(nlev), &
       qke(nlev), qke_adv(nlev), el(nlev), sh(nlev), sm(nlev), kh(nlev), km(nlev), &
       tsq(nlev), qsq(nlev), cov(nlev), du(nlev), dv(nlev), dth(nlev), dqv(nlev), &
       dqc(nlev), dqi(nlev), dqnc(nlev), dqni(nlev), dqs(nlev), dqnwfa(nlev), &
       dqnifa(nlev), dqnbca(nlev), dozone(nlev), dqke(nlev), qwt(nlev), qshear(nlev), &
       qbuoy(nlev), qdiss(nlev), qc_bl(nlev), qi_bl(nlev), cldfra_bl(nlev), &
       edmf_a(nlev), edmf_w(nlev), edmf_qt(nlev), edmf_thl(nlev), edmf_ent(nlev), &
       edmf_qc(nlev), sub_thl(nlev), sub_sqv(nlev), det_thl(nlev), det_sqv(nlev), &
       pattern_spp(nlev))
  allocate(chem(nlev,1), settle(nlev,1), scalars(nlev,1))

  write(*,'(a,3i6)') 'dims ', ncol, nlev, nstep
  do ic = 1, ncol
     read(*,*) dt, dx, xland, ps, ts, ust, hfx, qfx, wspd
     read(*,*) dz
     read(*,*) u
     read(*,*) v
     read(*,*) w
     read(*,*) th
     read(*,*) p
     read(*,*) ex
     read(*,*) rho
     read(*,*) sqv
     read(*,*) sqc
     read(*,*) sqi
     ! species the scheme is not given (flags .false.) and unused surface inputs
     sqs = 0.0_kind_phys; qnc = 0.0_kind_phys; qni = 0.0_kind_phys
     qnwfa = 0.0_kind_phys; qnifa = 0.0_kind_phys; qnbca = 0.0_kind_phys
     ozone = 0.0_kind_phys; rthraten = 0.0_kind_phys; pattern_spp = 0.0_kind_phys
     chem = 0.0_kind_phys; settle = 0.0_kind_phys; scalars = 0.0_kind_phys
     vdep = 0.0_kind_phys; frp = 0.0_kind_phys; emis_ant_no = 0.0_kind_phys
     qsfc = 0.0_kind_phys; ch = 0.0_kind_phys; znt = 0.0_kind_phys
     uoce = 0.0_kind_phys; voce = 0.0_kind_phys
     ! carried state before the cold start (overwritten by the initflag=1 branch)
     qke = 0.0_kind_phys; qke_adv = 0.0_kind_phys; el = 0.0_kind_phys
     sh = 0.0_kind_phys; sm = 0.0_kind_phys; tsq = 0.0_kind_phys
     qsq = 0.0_kind_phys; cov = 0.0_kind_phys; qc_bl = 0.0_kind_phys
     qi_bl = 0.0_kind_phys; cldfra_bl = 0.0_kind_phys
     pblh = 0.0_kind_phys; kpbl = 0
     do is = 1, nstep
        initflag = 0
        if (is == 1) initflag = 1
        tk = th*ex
        write(*,'(a,2i6)') 'step ', ic, is
        call put0('in_dt', dt); call put0('in_dx', dx); call put0('in_xland', xland)
        call put0('in_ps', ps); call put0('in_ts', ts); call put0('in_ust', ust)
        call put0('in_hfx', hfx); call put0('in_qfx', qfx); call put0('in_wspd', wspd)
        call put0('in_initflag', real(initflag, kind_phys))
        call put1('in_dz', dz); call put1('in_u', u); call put1('in_v', v)
        call put1('in_w', w); call put1('in_th', th); call put1('in_tk', tk)
        call put1('in_p', p); call put1('in_exner', ex); call put1('in_rho', rho)
        call put1('in_sqv', sqv); call put1('in_sqc', sqc); call put1('in_sqi', sqi)
        call put1('in_qke', qke); call put1('in_el', el); call put1('in_sh', sh)
        call put1('in_sm', sm); call put1('in_tsq', tsq); call put1('in_qsq', qsq)
        call put1('in_cov', cov); call put1('in_qc_bl', qc_bl)
        call put1('in_qi_bl', qi_bl); call put1('in_cldfra_bl', cldfra_bl)
        call put0('in_pblh', pblh); call put0('in_kpbl', real(kpbl, kind_phys))
#ifdef MYNN_DUMP
        mynn_dump_on = .true.
#endif
        call mynnedmf(i=1, j=1, initflag=initflag, restart=.false., cycling=.false., &
             delt=dt, dz1=dz, dx=dx, u1=u, v1=v, w1=w, th1=th, sqv1=sqv, sqc1=sqc, &
             sqi1=sqi, sqs1=sqs, qnc1=qnc, qni1=qni, qnwfa1=qnwfa, qnifa1=qnifa, &
             qnbca1=qnbca, ozone1=ozone, pres1=p, ex1=ex, rho1=rho, tk1=tk, &
             xland=xland, ts=ts, qsfc=qsfc, ps=ps, ust=ust, ch=ch, hfx=hfx, qfx=qfx, &
             znt=znt, wspd=wspd, uoce=uoce, voce=voce, qke1=qke, qke_adv1=qke_adv, &
             el1=el, sh1=sh, sm1=sm, kh1=kh, km1=km, nchem=1, ndvel=1, chem1=chem, &
             vdep=vdep, frp=frp, emis_ant_no=emis_ant_no, mix_chem=.false., &
             enh_mix=.false., settle1=settle, scalars=scalars, nscalars=1, &
             tsq1=tsq, qsq1=qsq, cov1=cov, du1=du, dv1=dv, dth1=dth, dqv1=dqv, &
             dqc1=dqc, dqi1=dqi, dqnc1=dqnc, dqni1=dqni, dqs1=dqs, dqnwfa1=dqnwfa, &
             dqnifa1=dqnifa, dqnbca1=dqnbca, dozone1=dozone, rthraten1=rthraten, &
             pblh=pblh, kpbl=kpbl, maxwidth_dd=maxwidth_dd, maxwidth=maxwidth, &
             maxmf=maxmf, ztop_plume=ztop_plume, excess_h=excess_h, excess_q=excess_q, &
             maxmf_dd=maxmf_dd, maxtkeprod=maxtkeprod, cldtop_cooling=cldtop_cooling, &
             ent_eff=ent_eff, dqke1=dqke, qwt1=qwt, qshear1=qshear, qbuoy1=qbuoy, &
             qdiss1=qdiss, qc_bl1=qc_bl, qi_bl1=qi_bl, cldfra_bl1=cldfra_bl, &
             bl_mynn_tkeadvect=.false., tke_budget=0, bl_mynn_cloudpdf=2, &
             bl_mynn_mixlength=1, bl_mynn_closure=2.5_kind_phys, bl_mynn_ess=1, &
             bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0, &
             bl_mynn_mixscalars=0, bl_mynn_mixaerosols=1, bl_mynn_mixnumcon=0, &
             bl_mynn_output=0, bl_mynn_cloudmix=1, bl_mynn_mixqt=0, bl_mynn_edmf_dd=0, &
             edmf_a1=edmf_a, edmf_w1=edmf_w, edmf_qt1=edmf_qt, edmf_thl1=edmf_thl, &
             edmf_ent1=edmf_ent, edmf_qc1=edmf_qc, sub_thl1=sub_thl, sub_sqv1=sub_sqv, &
             det_thl1=det_thl, det_sqv1=det_sqv, spp_pbl=0, pattern_spp_pbl1=pattern_spp, &
             FLAG_QC=.true., FLAG_QI=.true., FLAG_QNC=.false., FLAG_QNI=.false., &
             FLAG_QS=.false., FLAG_QNWFA=.false., FLAG_QNIFA=.false., FLAG_QNBCA=.false., &
             FLAG_OZONE=.false., KTS=kts, KTE=kte, errmsg=errmsg, errflg=errflg)
#ifdef MYNN_DUMP
        mynn_dump_on = .false.
#endif
        if (errflg /= 0) then
           write(0,*) 'mynnedmf error: ', trim(errmsg)
           stop 1
        endif
        call put1('out_du', du); call put1('out_dv', dv); call put1('out_dth', dth)
        call put1('out_dqv', dqv); call put1('out_dqc', dqc); call put1('out_dqi', dqi)
        call put1('out_qke', qke); call put1('out_el', el); call put1('out_sh', sh)
        call put1('out_sm', sm); call put1('out_kh', kh); call put1('out_km', km)
        call put1('out_tsq', tsq); call put1('out_qsq', qsq); call put1('out_cov', cov)
        call put1('out_qc_bl', qc_bl); call put1('out_qi_bl', qi_bl)
        call put1('out_cldfra_bl', cldfra_bl)
        call put1('out_edmf_a', edmf_a); call put1('out_edmf_w', edmf_w)
        call put1('out_edmf_qt', edmf_qt); call put1('out_edmf_thl', edmf_thl)
        call put1('out_edmf_ent', edmf_ent); call put1('out_edmf_qc', edmf_qc)
        call put0('out_pblh', pblh); call put0('out_kpbl', real(kpbl, kind_phys))
        call put0('out_maxmf', maxmf); call put0('out_maxwidth', maxwidth)
        call put0('out_ztop_plume', ztop_plume); call put0('out_excess_h', excess_h)
        call put0('out_excess_q', excess_q)
        ! advance the resolved state with the scheme's own tendencies
        u = u + du*dt
        v = v + dv*dt
        th = th + dth*dt
        sqv = sqv + dqv*dt
        sqc = sqc + dqc*dt
        sqi = sqi + dqi*dt
     enddo
  enddo
  write(*,'(a)') 'END_OK'
contains
  subroutine put0(name, x)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: x
    write(*,'(a,1x,i4,1x,es26.17e3)') name, 1, x
  end subroutine put0
  subroutine put1(name, x)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: x(:)
    write(*,'(a,1x,i4,*(1x,es26.17e3))') name, size(x), x
  end subroutine put1
end program mynn_driver
