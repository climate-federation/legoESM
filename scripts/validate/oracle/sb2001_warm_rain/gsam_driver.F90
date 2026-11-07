! Standalone driver for the gSAM 1.8.8 M2005 oracle (legoESM-constant scratch
! copy).  Reproduces SRC/MICRO_M2005/microphysics.f90 micro_proc for ONE column:
!   (optional) satadj_liquid(T_l, q_t) -> M2005MICRO_GRAUPEL(...) with the
!   arguments in micro_proc's order; level 1 = SURFACE (SAM convention).
! usage:
!   gsam_driver satadj <stock|lego> <thresh_mult> <maxiter> in.txt out.txt
!       in: n / T qv qc p   (q_t = qv+qc, T_l = T - fac_cond*qc, as micro_proc)
!       out: T_new qv_new qc_new
!   gsam_driver m2005 <irain> <fixpgam> <do_satadj> <Nc0_cm3> <dt> <icemicro> in.txt out.txt
!       in: ncol nlev / per level (bottom-up): p rho dz w T qv qc qr nr qi ni qs ns qg ng
!       out per level: T qv qc qr nr qi ni qs ns qg ng nc ODIAG(1:60); per column: precrt snowrt
program gsam_driver
  use micro_params
  use module_mp_GRAUPEL, only: GRAUPEL_INIT, M2005MICRO_GRAUPEL, ODIAG
  use satadj_mod
  use params, only: fac_cond
  implicit none
  character(len=512) :: mode, a1, a2, a3, a4, a5, a6, a7, fin, fout
  integer :: n, i, k, ncol, nlev, irain, ifix, isat, ice, j
  real :: tq(1), qq(1), qcq(1), pq(1), dt, nc0in
  real, allocatable :: P(:), RH(:), DZ(:), W(:), T(:), QV(:), QC(:), QR(:), NR(:), &
       QI(:), NI(:), QS(:), NS(:), QG(:), NG(:), NC(:), WV(:), &
       QCT(:), QIT(:), QST(:), QRT(:), NCT(:), NIT(:), NST(:), NRT(:), TT(:), QVT(:), &
       QGT(:), NGT(:), EFC(:), EFI(:), EFS(:), EFR(:), EFG(:), &
       QGS(:), QRS(:), QIS(:), QNS(:), QCS(:), PG(:), LC(:), MPR(:,:)
  real :: precrt, snowrt
  call get_command_argument(1, mode)
  if (trim(mode) == 'satadj') then
     call get_command_argument(2, a1); call get_command_argument(3, a2)
     call get_command_argument(4, a3); call get_command_argument(5, fin)
     call get_command_argument(6, fout)
     read(a2, *) thresh_mult; read(a3, *) maxiter_run
     open(10, file=trim(fin), status='old'); read(10, *) n
     open(11, file=trim(fout), status='replace')
     do i = 1, n
        read(10, *) tq(1), qq(1), qcq(1), pq(1)
        ! micro_proc: tmpqv = total water, tmptabs = T - (L/Cp)*qcl
        qq(1) = qq(1) + qcq(1)
        tq(1) = tq(1) - fac_cond*qcq(1)
        j = nhit_maxiter
        if (trim(a1) == 'stock') then
           call satadj_liquid_stock(1, tq, qq, qcq, pq)
        else
           call satadj_liquid_lego(1, tq, qq, qcq, pq)
        end if
        write(11, '(4(1x,es25.16e3))') tq(1), qq(1), qcq(1), real(nhit_maxiter - j)
     end do
     close(10); close(11)
     print *, 'SATADJ_OK n=', n, ' nhit_maxiter=', nhit_maxiter
     stop
  end if
  ! ------------------------------------------------------------- m2005 column
  call get_command_argument(2, a1); call get_command_argument(3, a2)
  call get_command_argument(4, a3); call get_command_argument(5, a4)
  call get_command_argument(6, a5); call get_command_argument(7, a6)
  call get_command_argument(8, fin); call get_command_argument(9, fout)
  read(a1, *) irain; read(a2, *) ifix; read(a3, *) isat; read(a4, *) nc0in
  read(a5, *) dt; read(a6, *) ice
  ! optional: tighten the pre-kernel satadj_liquid tolerance (stock = 1, 20)
  call get_environment_variable('SATADJ_THRESH_MULT', a7, status=j)
  if (j == 0) read(a7, *) thresh_mult
  call get_environment_variable('SATADJ_MAXITER', a7, status=j)
  if (j == 0) read(a7, *) maxiter_run
  dosb_warm_rain = (irain == 1)
  dofix_pgam = (ifix == 1)
  dopredictNc = .false.
  doicemicro = (ice == 1)
  dograupel = .true.
  Nc0 = nc0in
  nmicro_proc = nmicro_process_rates
  call GRAUPEL_INIT()
  open(10, file=trim(fin), status='old'); read(10, *) ncol, nlev
  allocate(P(nlev), RH(nlev), DZ(nlev), W(nlev), T(nlev), QV(nlev), QC(nlev), QR(nlev), &
       NR(nlev), QI(nlev), NI(nlev), QS(nlev), NS(nlev), QG(nlev), NG(nlev), NC(nlev), &
       WV(nlev), QCT(nlev), QIT(nlev), QST(nlev), QRT(nlev), NCT(nlev), NIT(nlev), NST(nlev), &
       NRT(nlev), TT(nlev), QVT(nlev), QGT(nlev), NGT(nlev), EFC(nlev), EFI(nlev), EFS(nlev), &
       EFR(nlev), EFG(nlev), QGS(nlev), QRS(nlev), QIS(nlev), QNS(nlev), QCS(nlev), PG(nlev), &
       LC(nlev), MPR(nlev, nmicro_proc))
  open(11, file=trim(fout), status='replace')
  do j = 1, ncol
     do k = 1, nlev
        read(10, *) P(k), RH(k), DZ(k), W(k), T(k), QV(k), QC(k), QR(k), NR(k), QI(k), &
             NI(k), QS(k), NS(k), QG(k), NG(k)
     end do
     NC = 0.; WV = 0.
     if (isat == 1) then
        ! micro_proc: satadj on (q_t, T_l) BEFORE the kernel; q_c re-diagnosed
        QV = QV + QC
        T = T - fac_cond*QC
        call satadj_liquid_lego(nlev, T, QV, QC, P)
     end if
     QCT = 0.; QIT = 0.; QST = 0.; QRT = 0.; NCT = 0.; NIT = 0.; NST = 0.; NRT = 0.
     TT = 0.; QVT = 0.; QGT = 0.; NGT = 0.; QGS = 0.; QRS = 0.; QIS = 0.; QNS = 0.; QCS = 0.
     MPR = 0.; ODIAG = 0.; precrt = 0.; snowrt = 0.
     call M2005MICRO_GRAUPEL(QCT, QIT, QST, QRT, NCT, NIT, NST, NRT, &
          QC, QI, QS, QR, NC, NI, NS, NR, TT, QVT, T, QV, P, RH, DZ, W, WV, &
          precrt, snowrt, EFC, EFI, EFS, EFR, dt, &
          1, 1, 1, 1, 1, nlev, 1, 1, 1, 1, 1, nlev, &
          QGT, NGT, QG, NG, EFG, QGS, QRS, QIS, QNS, QCS, PG, LC, MPR, .false.)
     do k = 1, nlev
        write(11, '(72(1x,es25.16e3))') T(k), QV(k), QC(k), QR(k), NR(k), QI(k), NI(k), &
             QS(k), NS(k), QG(k), NG(k), NC(k), ODIAG(k, 1:60)
     end do
     write(11, '(2(1x,es25.16e3))') precrt, snowrt
  end do
  close(10); close(11)
  print *, 'M2005_OK ncol=', ncol, ' nlev=', nlev, ' irain=', irain, ' fixpgam=', ifix, &
       ' satadj thresh_mult=', thresh_mult, ' maxiter=', maxiter_run, ' nhit_maxiter=', nhit_maxiter
end program gsam_driver
