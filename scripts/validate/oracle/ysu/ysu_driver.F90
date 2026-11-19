! Standalone driver for WRF v4.8.0 phys/physics_mmm/bl_ysu.F90 :: bl_ysu_run
! (used by make_fixture.py for tests/atmosphere/oracle_fixtures/ysu.npz).
! Usage: ysu_driver <input.txt> <output.txt> <topdown 0|1>
! Input is bottom-up (k=1 lowest), written by make_fixture.py.
! bl_ysu.F90 only uses ccpp_kind_types (WRF phys/ccpp_kind_types.F, -DDOUBLE_PRECISION).
! Optional arguments deliberately omitted: uox/vox (-> 0, no ocean current),
! ctopo/ctopo2 (absent == WRF topo_wind=0 default, ad(1)=1+fric),
! a_u..frcurb (no BEP urban; flag_bep=.false.).  qc=qi=0, f_qc=f_qi=.false.,
! nmix=0, rthraten=0.
program ysu_driver
  use ccpp_kind_types, only: kind_phys
  use bl_ysu, only: bl_ysu_run
  implicit none
  integer :: ncol, nlev, i, k, td
  real(kind_phys) :: cst(11), sc(12)
  real(kind_phys), allocatable :: ux(:,:), vx(:,:), tx(:,:), qvx(:,:), qcx(:,:), qix(:,:), &
       p2d(:,:), p2di(:,:), pi2d(:,:), dz8w(:,:), rth(:,:), qmix(:,:,:), &
       utnp(:,:), vtnp(:,:), ttnp(:,:), qvtnp(:,:), qctnp(:,:), qitnp(:,:), qmixtnp(:,:,:), &
       exh(:,:), exm(:,:)
  real(kind_phys), allocatable :: psfc(:), znt(:), ust(:), hpbl(:), dusfc(:), dvsfc(:), dtsfc(:), &
       dqsfc(:), psim(:), psih(:), xland(:), hfx(:), qfx(:), wspd(:), br(:), wstar(:), delta(:), &
       u10(:), v10(:)
  integer, allocatable :: kpbl(:)
  character(len=512) :: fin, fout, ctd
  character(len=256) :: errmsg
  integer :: errflg

  call get_command_argument(1, fin); call get_command_argument(2, fout); call get_command_argument(3, ctd)
  read(ctd, *) td
  open(10, file=trim(fin), status='old')
  read(10, *) ncol, nlev
  read(10, *) cst
  allocate(ux(ncol,nlev), vx(ncol,nlev), tx(ncol,nlev), qvx(ncol,nlev), qcx(ncol,nlev), qix(ncol,nlev), &
       p2d(ncol,nlev), p2di(ncol,nlev+1), pi2d(ncol,nlev), dz8w(ncol,nlev), rth(ncol,nlev), &
       qmix(ncol,nlev,1), utnp(ncol,nlev), vtnp(ncol,nlev), ttnp(ncol,nlev), qvtnp(ncol,nlev), &
       qctnp(ncol,nlev), qitnp(ncol,nlev), qmixtnp(ncol,nlev,1), exh(ncol,nlev), exm(ncol,nlev))
  allocate(psfc(ncol), znt(ncol), ust(ncol), hpbl(ncol), dusfc(ncol), dvsfc(ncol), dtsfc(ncol), &
       dqsfc(ncol), psim(ncol), psih(ncol), xland(ncol), hfx(ncol), qfx(ncol), wspd(ncol), br(ncol), &
       wstar(ncol), delta(ncol), u10(ncol), v10(ncol), kpbl(ncol))
  do i = 1, ncol
    read(10, *) ux(i,:); read(10, *) vx(i,:); read(10, *) tx(i,:); read(10, *) qvx(i,:)
    read(10, *) p2d(i,:); read(10, *) pi2d(i,:); read(10, *) dz8w(i,:); read(10, *) p2di(i,:)
    read(10, *) sc
    psfc(i)=sc(1); znt(i)=sc(2); ust(i)=sc(3); hfx(i)=sc(4); qfx(i)=sc(5); wspd(i)=sc(6)
    br(i)=sc(7); psim(i)=sc(8); psih(i)=sc(9); xland(i)=sc(10); u10(i)=sc(11); v10(i)=sc(12)
  end do
  close(10)
  qcx = 0.; qix = 0.; rth = 0.; qmix = 0.

  call bl_ysu_run(ux=ux, vx=vx, tx=tx, qvx=qvx, qcx=qcx, qix=qix, nmix=0, qmix=qmix, &
       p2d=p2d, p2di=p2di, pi2d=pi2d, f_qc=.false., f_qi=.false., &
       utnp=utnp, vtnp=vtnp, ttnp=ttnp, qvtnp=qvtnp, qctnp=qctnp, qitnp=qitnp, qmixtnp=qmixtnp, &
       cp=cst(1), g=cst(2), rovcp=cst(3), rd=cst(4), rovg=cst(5), ep1=cst(6), ep2=cst(7), &
       karman=cst(8), xlv=cst(9), rv=cst(10), dz8w2d=dz8w, psfcpa=psfc, &
       znt=znt, ust=ust, hpbl=hpbl, dusfc=dusfc, dvsfc=dvsfc, dtsfc=dtsfc, dqsfc=dqsfc, &
       psim=psim, psih=psih, xland=xland, hfx=hfx, qfx=qfx, wspd=wspd, br=br, &
       dt=cst(11), kpbl1d=kpbl, exch_hx=exh, exch_mx=exm, wstar=wstar, delta=delta, &
       u10=u10, v10=v10, rthraten=rth, ysu_topdown_pblmix=(td == 1), flag_bep=.false., &
       its=1, ite=ncol, kte=nlev, kme=nlev+1, errmsg=errmsg, errflg=errflg)
  if (errflg /= 0) stop 'bl_ysu_run errflg'

  open(20, file=trim(fout), status='replace')
  write(20, '(a,i4,i4,a,i2,a,a)') 'YSU ', ncol, nlev, ' topdown=', td, ' ', trim(errmsg)
  do i = 1, ncol
    write(20, '(a,i4,i4)') 'COL', i, kpbl(i)
    write(20, '(8es25.16e3)') hpbl(i), wstar(i), delta(i), dusfc(i), dvsfc(i), dtsfc(i), dqsfc(i), 0.0_kind_phys
    write(20, '(*(es25.16e3))') (utnp(i,k), k=1,nlev)
    write(20, '(*(es25.16e3))') (vtnp(i,k), k=1,nlev)
    write(20, '(*(es25.16e3))') (ttnp(i,k), k=1,nlev)
    write(20, '(*(es25.16e3))') (qvtnp(i,k), k=1,nlev)
    write(20, '(*(es25.16e3))') (exh(i,k), k=1,nlev)
    write(20, '(*(es25.16e3))') (exm(i,k), k=1,nlev)
  end do
  close(20)
end program ysu_driver
