! Driver: calls the REAL OpenIFS VDFEXCU (arpifs/phys_ec/vdfexcu.F90) one column at a time.
! Harness settings (stated): PKHVFL=0 -> no stable-PBL-height search -> stable asymptotic
! mixing length = ZLMIN = 30 m everywhere; KINV=KLEV -> mixed-layer K, cloud-top K and
! entrainment branches OFF; LRDIFF_STRATO off; SPP off; PZ0MM=0 (so ZZH = mid-interface height);
! PRI = legoESM's own gradient Ri (same Ri on both sides).  Output PKM/PKH [m2/s] (post-cap,
! pre-ZCFNC1 scaling).
program ifs_driver
  use parkind1, only: jpim, jprb
  use yomcst, only: rg, rd, rcpd, retv, rlvtt
  use yoevdfs, only: rchba, rchbb, rchbc, rchbd, rchb23a, rchbbcd, rchbcd, rcheta, rchetb, rchbhdl, rcdhalf, rcdhpi2
  use yoegwd, only: tegwd
  use yoephy, only: tephy
  use spp_mod, only: tspp_config
  implicit none
  type(tegwd) :: ydegwd
  type(tephy) :: ydephy
  type(tspp_config) :: ydspp
  integer :: ncol, klev, i, k, nin
  real(jprb), allocatable :: pum1(:,:), pvm1(:,:), ptm1(:,:), pqm1(:,:), paphm1(:,:), pgeom1(:,:), pgeoh(:,:)
  real(jprb), allocatable :: pri(:,:), pcptgz(:,:), phrlw(:,:), phrsw(:,:), pcfm(:,:), pcfh(:,:), pkh(:,:), pkm(:,:)
  real(jprb) :: pz0mm(1), pgelat(1), pkmfl(1), pkhfl(1), pkqfl(1), pkhvfl(1), pzinv(1), pzcldbase(1), pgp2dspp(1,1), plam
  integer(jpim) :: kinv(1), kcbase(1), kctop(1), kpbltype(1)
  character(len=512) :: fin, fout
  real(jprb) :: c5(5)
  external vdfexcu
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=nin, file=trim(fin), status='old')
  read(nin,*) ncol, klev
  read(nin,*) c5
  rg = c5(1); rd = c5(2); rcpd = c5(3); retv = c5(4); rlvtt = c5(5)
  ! suvdfs.F90 values
  rcheta = 5._jprb; rchetb = 4._jprb; rchbhdl = 5._jprb
  rchba = 1._jprb; rchbb = 2._jprb/3._jprb; rchbc = 5._jprb; rchbd = 0.35_jprb
  rchb23a = (2._jprb/3._jprb)*rchba; rchbbcd = rchbb*rchbc/rchbd; rchbcd = rchbc/rchbd
  rcdhalf = 16._jprb; rcdhpi2 = 2._jprb*atan(1._jprb)
  allocate(pum1(1,klev), pvm1(1,klev), ptm1(1,klev), pqm1(1,klev), paphm1(1,0:klev), pgeom1(1,klev), pgeoh(1,0:klev))
  allocate(pri(1,klev), pcptgz(1,klev), phrlw(1,klev), phrsw(1,klev), pcfm(1,klev), pcfh(1,klev), pkh(1,klev), pkm(1,klev))
  open(20, file=trim(fout), status='replace')
  do i = 1, ncol
    read(nin,*) pum1(1,:); read(nin,*) pvm1(1,:); read(nin,*) ptm1(1,:); read(nin,*) pqm1(1,:)
    read(nin,*) paphm1(1,:); read(nin,*) pgeom1(1,:); read(nin,*) pgeoh(1,:); read(nin,*) pri(1,:)
    read(nin,*) pkmfl(1)
    pcptgz = rcpd*ptm1 + pgeom1
    phrlw = 0; phrsw = 0; pz0mm = 0; pgelat = 0; pkhfl = 0; pkqfl = 0; pkhvfl = 0
    pzinv = 0; pzcldbase = 0; pgp2dspp = 0; plam = 150._jprb
    kinv = klev; kcbase = 0; kctop = 0; kpbltype = 0
    pcfm = -999; pcfh = -999; pkh = -999; pkm = -999
    call vdfexcu(ydegwd, ydephy, ydspp, 1, 1, 1, klev, kinv, kcbase, kctop, plam, pz0mm, &
         phrlw, phrsw, pum1, pvm1, ptm1, pqm1, paphm1, pgeom1, pgeoh, pgelat, pcptgz, &
         pkmfl, pkhfl, pkqfl, pkhvfl, pgp2dspp, pcfm, pcfh, pzinv, pkh, pkm, pri, pzcldbase, kpbltype)
    write(20,'(*(es25.17e3))') (pkm(1,k), k=1,klev-1)
    write(20,'(*(es25.17e3))') (pkh(1,k), k=1,klev-1)
  end do
  close(20)
  print *, 'IFS_DRIVER_DONE', ncol
end program ifs_driver
