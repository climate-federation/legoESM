! Standalone single-column driver for WRF v4.8.0 KF_eta_PARA (module_cu_kfeta.F).
! Does per column what KF_eta_CPS does (zero the tendencies, copy the 1-D
! profiles, call KF_eta_PARA) minus the W0AVG running mean, which the input
! supplies directly as W0AVG1D.  Levels are BOTTOM-UP (k=1 lowest), as in WRF.
! Input (stdin, free format):
!   ncol nlev
!   G CP R EP2 XLV0 XLV1 XLS0 XLS1 SVP1 SVP2 SVP3 SVPT0 DT DX CUDT trigger warm_rain f_qi f_qs
!   per column, nlev lines: T QV(mixing ratio) P DZ RHO U V W0AVG
! Output (stdout): keyed records per column.
PROGRAM kf_driver
  USE module_cu_kfeta
  USE module_wrf_error, ONLY: n_fatal
  USE kfdbg
  IMPLICIT NONE
  INTEGER :: ncol, nlev, ic, k, trig, iwarm, ifqi, ifqs, ishall, nf0
  REAL :: G, CP, R, EP2, XLV0, XLV1, XLS0, XLS1, SVP1, SVP2, SVP3, SVPT0, DT, DX, CUDT, DXSQ
  REAL, ALLOCATABLE :: T(:), QV(:), P(:), DZ(:), RHO(:), U(:), V(:), W0(:), TPH(:), TPV(:)
  REAL, ALLOCATABLE :: DQDT(:), DQIDT(:), DQCDT(:), DQRDT(:), DQSDT(:), DTDT(:)
  REAL, ALLOCATABLE :: cdp(:,:,:), csh(:,:,:), qck(:,:,:), qik(:,:,:)
  REAL, ALLOCATABLE :: udr(:,:,:), ddr(:,:,:), uer(:,:,:), der(:,:,:)
  REAL :: RAINCV(1,1), PRATEC(1,1), NCA(1,1), CUTOP(1,1), CUBOT(1,1), TIMECK(1,1)

  READ(*,*) ncol, nlev
  READ(*,*) G, CP, R, EP2, XLV0, XLV1, XLS0, XLS1, SVP1, SVP2, SVP3, SVPT0, DT, DX, CUDT, &
            trig, iwarm, ifqi, ifqs
  DXSQ = DX*DX
  ALLOCATE(T(nlev), QV(nlev), P(nlev), DZ(nlev), RHO(nlev), U(nlev), V(nlev), W0(nlev), TPH(nlev), TPV(nlev))
  ALLOCATE(DQDT(nlev), DQIDT(nlev), DQCDT(nlev), DQRDT(nlev), DQSDT(nlev), DTDT(nlev))
  ALLOCATE(cdp(1,nlev,1), csh(1,nlev,1), qck(1,nlev,1), qik(1,nlev,1))
  ALLOCATE(udr(1,nlev,1), ddr(1,nlev,1), uer(1,nlev,1), der(1,nlev,1))
  CALL kf_lutab(SVP1, SVP2, SVP3, SVPT0)
  WRITE(*,'(A,I0,1X,I0)') 'NCOL ', ncol, nlev
  DO ic = 1, ncol
    DO k = 1, nlev
      READ(*,*) T(k), QV(k), P(k), DZ(k), RHO(k), U(k), V(k), W0(k)
    END DO
    TPH = 0.; TPV = 0.
    DQDT = 0.; DQIDT = 0.; DQCDT = 0.; DQRDT = 0.; DQSDT = 0.; DTDT = 0.
    cdp = 0.; csh = 0.; qck = 0.; qik = 0.; udr = 0.; ddr = 0.; uer = 0.; der = 0.
    RAINCV = 0.; PRATEC = 0.; NCA = -100.; CUTOP = 1.; CUBOT = nlev + 1.; TIMECK = 0.
    ishall = 0
    nf0 = n_fatal
    CALL KF_eta_PARA(1, 1, U, V, T, QV, P, DZ, W0, TPH, TPV, trig, DT, DX, DXSQ, RHO, &
         XLV0, XLV1, XLS0, XLS1, CP, R, G, EP2, SVP1, SVP2, SVP3, SVPT0, &
         DQDT, DQIDT, DQCDT, DQRDT, DQSDT, DTDT, RAINCV, PRATEC, NCA, &
         ifqi == 1, ifqs == 1, iwarm == 1, CUTOP, CUBOT, ishall, CUDT, &
         1,1, 1,1, 1,nlev, 1,1, 1,1, 1,nlev, 1,1, 1,1, 1,nlev, &
         cdp, csh, qck, qik, udr, ddr, uer, der, TIMECK, 1)
    WRITE(*,'(A,I0)') 'COL ', ic
    CALL s('fired', REAL(d_fired)); CALL s('ishall', REAL(ishall)); CALL s('nfatal', REAL(n_fatal - nf0))
    CALL s('pratec', PRATEC(1,1)); CALL s('nca', NCA(1,1))
    CALL s('cutop', CUTOP(1,1)); CALL s('cubot', CUBOT(1,1))
    IF (d_fired == 1) THEN
      CALL s('ltop', REAL(d_ltop)); CALL s('let', REAL(d_let)); CALL s('abe', d_abe)
      CALL s('vmflcl', d_vmflcl); CALL s('ainc', d_ainc); CALL s('timec', d_timec)
      CALL s('ncount', REAL(d_ncount)); CALL s('fabe', d_fabe)
      CALL s('tder', d_tder); CALL s('trppt', d_trppt); CALL s('pptflx', d_pptflx); CALL s('cpr', d_cpr)
      CALL s('lfs', REAL(d_lfs)); CALL s('ldb', REAL(d_ldb)); CALL s('wlcl', d_wlcl)
      CALL s('rad', d_rad); CALL s('chmin', d_chmin); CALL s('cldhgt', d_cldhgt)
      CALL s('lcl', REAL(d_lcl)); CALL s('au0', d_au0); CALL s('nstep', REAL(d_nstep))
      CALL s('kpbl', REAL(d_kpblf)); CALL s('lc', REAL(d_lcf)); CALL s('fbfrc', d_fbfrc)
      CALL s('frc2', d_frc2)
      CALL a('umf', d_umf(1:nlev)); CALL a('uer', d_uer(1:nlev)); CALL a('udr', d_udr(1:nlev))
      CALL a('dmf', d_dmf(1:nlev)); CALL a('der', d_der(1:nlev)); CALL a('ddr', d_ddr(1:nlev))
      CALL a('tu', d_tu(1:nlev)); CALL a('qu', d_qu(1:nlev)); CALL a('qliq', d_qliq(1:nlev))
      CALL a('qice', d_qice(1:nlev)); CALL a('detlq', d_detlq(1:nlev))
      CALL a('detic', d_detic(1:nlev)); CALL a('pptliq', d_pptliq(1:nlev)); CALL a('pptice', d_pptice(1:nlev))
      CALL a('tg', d_tg(1:nlev)); CALL a('qg', d_qg(1:nlev)); CALL a('eqfrc', d_eqfrc(1:nlev))
      CALL a('ems', d_ems(1:nlev))
    END IF
    CALL a('dtdt', DTDT); CALL a('dqdt', DQDT); CALL a('dqcdt', DQCDT); CALL a('dqrdt', DQRDT)
    CALL a('dqidt', DQIDT); CALL a('dqsdt', DQSDT)
  END DO
CONTAINS
  SUBROUTINE s(name, val)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, INTENT(IN) :: val
    WRITE(*,'(A,1X,A,1X,ES25.17E3)') 'S', name, val
  END SUBROUTINE s
  SUBROUTINE a(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, INTENT(IN) :: arr(:)
    INTEGER :: kk
    WRITE(*,'(A,1X,A)', ADVANCE='NO') 'A', name
    DO kk = 1, SIZE(arr)
      WRITE(*,'(1X,ES25.17E3)', ADVANCE='NO') arr(kk)
    END DO
    WRITE(*,*)
  END SUBROUTINE a
END PROGRAM kf_driver
