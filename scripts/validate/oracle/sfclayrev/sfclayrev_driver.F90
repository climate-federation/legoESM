! Standalone driver for WRF v4.8.0 phys/module_sf_sfclayrev.F :: sfclayrev
! (-> phys/physics_mmm/sf_sfclayrev.F90 :: sf_sfclayrev_run), used by
! make_fixture.py for tests/atmosphere/oracle_fixtures/sfclayrev.npz.
! Usage: sfclayrev_driver <input.txt> <output.txt> <nstep> <scm_force_flux 0|1>
! Calls sf_sfclayrev_init once, then sfclayrev nstep times on the same
! atmosphere, the inout surface-layer state (ust, mol, znt, hfx, qfx, qsfc, ...)
! carried between calls as WRF carries it between time steps.  pblh is held at
! its input value.  Namelist options = WRF defaults: isfflx=1, isftcflx=0,
! iz0tlnd=0, shalwater_z0=0, kim_tofd=.false.; EM_CORE optional arguments
! (isftcflx, iz0tlnd, scm_force_flux, ustm, ck, cka, cd, cda) present, as the
! WRF build passes them.  lakemask=0.  One level (kts=kte=1).
program sfclayrev_driver
  use ccpp_kind_types, only: kind_phys
  use sf_sfclayrev, only: sf_sfclayrev_init
  use module_sf_sfclayrev, only: sfclayrev
  implicit none
  integer :: ncol, nstep, sff, i, s, errflg
  real(kind_phys) :: cst(13), c(19)
  real(kind_phys), allocatable, dimension(:,:,:) :: u3d, v3d, t3d, qv3d, p3d, dz8w
  real(kind_phys), allocatable, dimension(:,:) :: psfc, chs, chs2, cqs2, cpm, znt, ust, pblh, &
       mavail, zol, mol, regime, psim, psih, fm, fh, xland, hfx, qfx, lh, tsk, flhc, flqc, qgh, &
       qsfc, rmol, u10, v10, th2, t2, q2, gz1oz0, wspd, br, dx, lakemask, var2d, water_depth, &
       ustm, ck, cka, cd, cda
  character(len=512) :: fin, fout, arg
  character(len=256) :: errmsg

  call get_command_argument(1, fin); call get_command_argument(2, fout)
  call get_command_argument(3, arg); read(arg, *) nstep
  call get_command_argument(4, arg); read(arg, *) sff
  open(10, file=trim(fin), status='old')
  read(10, *) ncol
  read(10, *) cst
  allocate(u3d(ncol,1,1), v3d(ncol,1,1), t3d(ncol,1,1), qv3d(ncol,1,1), p3d(ncol,1,1), dz8w(ncol,1,1))
  allocate(psfc(ncol,1), chs(ncol,1), chs2(ncol,1), cqs2(ncol,1), cpm(ncol,1), znt(ncol,1), &
       ust(ncol,1), pblh(ncol,1), mavail(ncol,1), zol(ncol,1), mol(ncol,1), regime(ncol,1), &
       psim(ncol,1), psih(ncol,1), fm(ncol,1), fh(ncol,1), xland(ncol,1), hfx(ncol,1), qfx(ncol,1), &
       lh(ncol,1), tsk(ncol,1), flhc(ncol,1), flqc(ncol,1), qgh(ncol,1), qsfc(ncol,1), rmol(ncol,1), &
       u10(ncol,1), v10(ncol,1), th2(ncol,1), t2(ncol,1), q2(ncol,1), gz1oz0(ncol,1), wspd(ncol,1), &
       br(ncol,1), dx(ncol,1), lakemask(ncol,1), var2d(ncol,1), water_depth(ncol,1), ustm(ncol,1), &
       ck(ncol,1), cka(ncol,1), cd(ncol,1), cda(ncol,1))
  do i = 1, ncol
    read(10, *) c
    u3d(i,1,1)=c(1); v3d(i,1,1)=c(2); t3d(i,1,1)=c(3); qv3d(i,1,1)=c(4); p3d(i,1,1)=c(5)
    dz8w(i,1,1)=c(6); psfc(i,1)=c(7); tsk(i,1)=c(8); xland(i,1)=c(9); mavail(i,1)=c(10)
    pblh(i,1)=c(11); dx(i,1)=c(12); ust(i,1)=c(13); mol(i,1)=c(14); znt(i,1)=c(15)
    hfx(i,1)=c(16); qfx(i,1)=c(17); qsfc(i,1)=c(18); ustm(i,1)=c(19)
  end do
  close(10)
  ! inout fields fully overwritten before use inside sf_sfclayrev_run
  chs=0.; chs2=0.; cqs2=0.; cpm=0.; zol=0.; regime=0.; psim=0.; psih=0.; fm=0.; fh=0.
  flhc=0.; flqc=0.; qgh=0.; rmol=0.; gz1oz0=0.; wspd=0.; br=0.
  lh=0.; u10=0.; v10=0.; th2=0.; t2=0.; q2=0.; ck=0.; cka=0.; cd=0.; cda=0.
  lakemask=0.; var2d=0.; water_depth=0.

  call sf_sfclayrev_init(errmsg, errflg)
  if (errflg /= 0) stop 'init errflg'
  open(20, file=trim(fout), status='replace')
  write(20, '(a,3i6)') 'SFCLAYREV', ncol, nstep, sff
  do s = 1, nstep
    call sfclayrev(u3d=u3d, v3d=v3d, t3d=t3d, qv3d=qv3d, p3d=p3d, dz8w=dz8w, &
         cp=cst(1), g=cst(2), rovcp=cst(3), r=cst(4), xlv=cst(5), psfc=psfc, chs=chs, chs2=chs2, &
         cqs2=cqs2, cpm=cpm, znt=znt, ust=ust, pblh=pblh, mavail=mavail, zol=zol, mol=mol, &
         regime=regime, psim=psim, psih=psih, fm=fm, fh=fh, xland=xland, hfx=hfx, qfx=qfx, lh=lh, &
         tsk=tsk, flhc=flhc, flqc=flqc, qgh=qgh, qsfc=qsfc, rmol=rmol, u10=u10, v10=v10, th2=th2, &
         t2=t2, q2=q2, gz1oz0=gz1oz0, wspd=wspd, br=br, isfflx=1, dx=dx, &
         svp1=cst(6), svp2=cst(7), svp3=cst(8), svpt0=cst(9), ep1=cst(10), ep2=cst(11), &
         karman=cst(12), p1000mb=cst(13), lakemask=lakemask, kim_tofd=.false., tofd_factor=0.003_kind_phys, &
         var2d=var2d, ids=1, ide=ncol, jds=1, jde=1, kds=1, kde=1, ims=1, ime=ncol, jms=1, jme=1, &
         kms=1, kme=1, its=1, ite=ncol, jts=1, jte=1, kts=1, kte=1, ustm=ustm, ck=ck, cka=cka, &
         cd=cd, cda=cda, isftcflx=0, iz0tlnd=0, shalwater_z0=0, water_depth=water_depth, &
         scm_force_flux=sff, errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) stop 'sfclayrev errflg'
    write(20, '(a,i4,a,a)') 'STEP', s, ' ', trim(errmsg)
    do i = 1, ncol
      write(20, '(*(es25.16e3))') ust(i,1), mol(i,1), znt(i,1), hfx(i,1), qfx(i,1), lh(i,1), &
           qsfc(i,1), br(i,1), fm(i,1), fh(i,1), psim(i,1), psih(i,1), wspd(i,1), u10(i,1), &
           v10(i,1), th2(i,1), t2(i,1), q2(i,1), zol(i,1), rmol(i,1), regime(i,1), gz1oz0(i,1), &
           chs(i,1), chs2(i,1), cqs2(i,1), cpm(i,1), flhc(i,1), flqc(i,1), qgh(i,1), ustm(i,1), &
           ck(i,1), cka(i,1), cd(i,1), cda(i,1)
    end do
  end do
  close(20)
end program sfclayrev_driver
