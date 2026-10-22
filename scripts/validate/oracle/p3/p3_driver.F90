! Standalone column driver for WRF v4.8.0 phys/module_mp_p3.F p3_main (P3 v4.5.2).
! Mirrors mp_p3_wrapper_wrf for mp_physics = 50 (P3_1CATEGORY: specified droplet
! number, one 2-moment ice category): model='WRF' (k=1 bottom), nCat=1,
! log_predictNc=.false. (nc passed in as 0 and set to nccnst/rho by p3_main),
! typeDiags_ON=.false., debug_on=.false., clbfact_dep=clbfact_sub=1, scpf_on=.false.,
! scpf_pfrac=scpf_resfact=0, ssat=0, p3_init(lookup_dir, 1, .false., 'WRF') as
! module_physics_init.F calls it for mp_physics=50.
! Harness glue (NOT oracle physics): th_old=th and qv_old=qv (no dynamical forcing
! over the step: the dT/dt and dqv/dt parts of P3's 'aaa' term are 0, its Bergeron
! part remains), uzpl=0, th = T*(1e5/p)**kappa with kappa = rd/cp of the patched
! module (passed in the header so T round-trips), it from the header.
!
! input  (argv1): header 'ncol nlev dt it kappa'; 'lookup_dir'; then ncol*nlev lines
!                 bottom-up: p dz T qv qc qr nr qi qirim ni birim (SI, numbers per kg)
! output (argv2): per column 'COL icol prt_liq prt_sol' then nlev lines:
!                 T qv qc nc qr nr qi qirim ni birim diag(1:36)
program p3_driver
  use microphy_p3, only: p3_init, p3_main
  implicit none
  integer, parameter :: ndiag = 36
  integer :: ncol, nlev, it, icol, k, stat
  real :: dt, kappa
  character(len=1024) :: fin, fout, lookdir
  real, allocatable :: p(:,:), dz(:,:), tt(:,:), qv(:,:), qc(:,:), nc(:,:), qr(:,:), nr(:,:), &
       qi(:,:,:), qir(:,:,:), ni(:,:,:), bir(:,:,:), th(:,:), th_old(:,:), qv_old(:,:),        &
       ssat(:,:), uz(:,:), ze(:,:), effc(:,:), effi(:,:,:), vmi(:,:,:), di(:,:,:),             &
       rhoi(:,:,:), d2(:,:), d3(:,:,:), scf(:,:)
  real :: prl(1), prs(1)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(11, file=trim(fin), status='old')
  read(11,*) ncol, nlev, dt, it, kappa
  read(11,'(a)') lookdir

  call p3_init(trim(lookdir), 1, .false., 'WRF', stat, .false.)
  if (stat /= 0) stop 'P3_INIT_FAILED'

  allocate(p(1,nlev), dz(1,nlev), tt(1,nlev), qv(1,nlev), qc(1,nlev), nc(1,nlev), qr(1,nlev), &
       nr(1,nlev), qi(1,nlev,1), qir(1,nlev,1), ni(1,nlev,1), bir(1,nlev,1), th(1,nlev),     &
       th_old(1,nlev), qv_old(1,nlev), ssat(1,nlev), uz(1,nlev), ze(1,nlev), effc(1,nlev),   &
       effi(1,nlev,1), vmi(1,nlev,1), di(1,nlev,1), rhoi(1,nlev,1), d2(1,1),                 &
       d3(1,nlev,ndiag),                                                                     &
       scf(1,nlev))

  open(12, file=trim(fout), status='replace')
  do icol = 1, ncol
     do k = 1, nlev
        read(11,*) p(1,k), dz(1,k), tt(1,k), qv(1,k), qc(1,k), qr(1,k), nr(1,k), qi(1,k,1), &
             qir(1,k,1), ni(1,k,1), bir(1,k,1)
     end do
     th = tt*(1.e5/p)**kappa
     th_old = th
     qv_old = qv
     nc = 0.
     ssat = 0.; uz = 0.; d2 = 0.; d3 = 0.
     call p3_main(qc=qc, nc=nc, qr=qr, nr=nr, th_old=th_old, th=th, qv_old=qv_old, qv=qv, &
          dt=dt, qitot=qi, qirim=qir, nitot=ni, birim=bir, ssat=ssat, uzpl=uz,           &
          pres=p, dzq=dz, it=it, prt_liq=prl, prt_sol=prs, its=1, ite=1, kts=1, kte=nlev,  &
          nCat=1, diag_ze=ze, diag_effc=effc, diag_effi=effi, diag_vmi=vmi, diag_di=di,     &
          diag_rhoi=rhoi, n_diag_2d=1, diag_2d=d2, n_diag_3d=ndiag, diag_3d=d3,             &
          log_predictNc=.false., typeDiags_ON=.false., model='WRF',                         &
          clbfact_dep=1., clbfact_sub=1., debug_on=.false.,                                 &
          scpf_on=.false., scpf_pfrac=0., scpf_resfact=0., SCF_out=scf)
     tt = th*(p*1.e-5)**kappa
     write(12,'(a,i6,2es26.17e3)') 'COL', icol, prl(1), prs(1)
     do k = 1, nlev
        write(12,'(60es26.17e3)') tt(1,k), qv(1,k), qc(1,k), nc(1,k), qr(1,k), nr(1,k), &
             qi(1,k,1), qir(1,k,1), ni(1,k,1), bir(1,k,1), d3(1,k,1:ndiag)
     end do
  end do
  close(12)
  print*, 'P3_DRIVER_DONE', ncol, nlev
end program p3_driver
