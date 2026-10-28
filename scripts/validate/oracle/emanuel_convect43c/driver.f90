! Standalone driver for instrumented CONVECT 4.3c (convect43c_patched.f).
! Input (list-directed text): line 1 = XCPD XCPV XCL XRV XRD XLV0 XG XCOEFFR XCOEFFS IDDOFF ICOLD
!   line 2 = ncol nd nl delt ; then per column: cbmf_in, then nd rows "T Q QS U V P(mb)"
!   (surface-first), then nd+1 rows PH(mb).
! Output: per column header + scalars + profile table + nd x nd matrices.
program drv
  implicit none
  integer, parameter :: na = 70
  integer :: IDDOFF, ICOLD
  real :: XCPD, XCPV, XCL, XRV, XRD, XLV0, XG, XCOEFFR, XCOEFFS
  common /EMCONST/ XCPD, XCPV, XCL, XRV, XRD, XLV0, XG
  common /EMCONST/ XCOEFFR, XCOEFFS, IDDOFF, ICOLD
  include 'emdbg.inc'
  integer :: ncol, nd, nl, ic, k, iflag
  real :: delt, cbmf, precip, wd, tprime, qprime
  real, allocatable :: T(:), Q(:), QS(:), U(:), V(:), P(:), PH(:), TRA(:,:)
  real, allocatable :: FT(:), FQ(:), FU(:), FV(:), FTRA(:,:)
  character(len=512) :: fin, fout
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old')
  open(20, file=trim(fout), status='replace')
  read(10,*) XCPD, XCPV, XCL, XRV, XRD, XLV0, XG, XCOEFFR, XCOEFFS, IDDOFF, ICOLD
  read(10,*) ncol, nd, nl, delt
  if (nd + 1 > na) stop 'nd too large'
  allocate(T(nd), Q(nd), QS(nd), U(nd), V(nd), P(nd), PH(nd+1), TRA(nd,1))
  allocate(FT(nd), FQ(nd), FU(nd), FV(nd), FTRA(nd,1))
  write(20,'(A,I6,I6,I6)') 'NCOL', ncol, nd, nl
  do ic = 1, ncol
    read(10,*) cbmf
    do k = 1, nd
      read(10,*) T(k), Q(k), QS(k), U(k), V(k), P(k)
    end do
    do k = 1, nd + 1
      read(10,*) PH(k)
    end do
    TRA = 0.0
    DPLCL=0; DDTMA=0; DCBMF=0; DFRAC=0; DTVPPLCL=0; DTVAPLCL=0; DDTPBL=0
    DM=0; DMENT=0; DEP=0; DCLW=0; DTVP=0; DTP=0; DHP=0; DQENT=0; DELIJ=0; DSIJ=0
    DTV=0; DFT1=0; DFQ1=0; DFT2=0; DFQ2=0; DMP=0; DEVAP=0; DWATER=0; DQP=0
    DGZ=0; DH=0; DNK=0; DICB=0; DINB=0; DINB1=0; DIHMIN=0; DSTAGE=0
    call CONVECT(T, Q, QS, U, V, TRA, P, PH, nd, nl, 1, delt, iflag, FT, FQ, FU, FV, &
                 FTRA, precip, wd, tprime, qprime, cbmf)
    write(20,'(A,I6,7I6)') 'COL', ic, iflag, DNK, DICB, DINB, DINB1, DIHMIN, DSTAGE
    write(20,'(13ES25.16E3)') cbmf, precip, wd, tprime, qprime, DPLCL, DDTMA, DCBMF, &
         DFRAC, DTVPPLCL, DTVAPLCL, DDTPBL, 0.0
    do k = 1, nd
      write(20,'(25ES25.16E3)') FT(k), FQ(k), FU(k), FV(k), DFT1(k), DFQ1(k), DFT2(k), &
           DFQ2(k), DM(k), DEP(k), DCLW(k), DTVP(k), DTP(k), DHP(k), DTV(k), DGZ(k), &
           DH(k), DMP(k), DEVAP(k), DWATER(k), DQP(k), 0.0, 0.0, 0.0, 0.0
    end do
    do k = 1, nd
      write(20,'(70ES25.16E3)') DMENT(k,1:nd)
    end do
    do k = 1, nd
      write(20,'(70ES25.16E3)') DQENT(k,1:nd)
    end do
    do k = 1, nd
      write(20,'(70ES25.16E3)') DELIJ(k,1:nd)
    end do
    do k = 1, nd
      write(20,'(70ES25.16E3)') DSIJ(k,1:nd)
    end do
  end do
  close(20)
end program drv
