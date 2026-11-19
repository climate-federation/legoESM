! Standalone RTE+RRTMGP v1.8 oracle driver for the legoESM faithfulness audit.
! Reads identical inputs written by the Python harness (raw float64, Fortran order),
! runs RRTMGP gas optics (LW g128 / SW g112) and the RTE LW/SW solvers, and writes
! every intermediate (tau, ssa, sources, toa_src) plus fluxes.
!
! usage: oracle_driver <indir> <lw_gas_file> <sw_gas_file>
!   <indir>/dims.txt : ncol nlay ngas \n gas names (one per line)
!   <indir>/in.bin   : play(ncol,nlay) plev(ncol,nlay+1) tlay(ncol,nlay) tlev(ncol,nlay+1)
!                      tsfc(ncol) col_dry(ncol,nlay) vmr(ncol,nlay,ngas) emis(ncol)
!                      mu0(ncol) alb_dir(ncol) alb_dif(ncol) toa_flux(ncol,ngpt_sw)
program oracle_driver
  use mo_rte_kind,           only: wp, wl
  use mo_optical_props,      only: ty_optical_props_2str, ty_optical_props_1scl
  use mo_source_functions,   only: ty_source_func_lw
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  use mo_load_coefficients,  only: load_and_init
  use mo_rte_lw,             only: rte_lw
  use mo_rte_sw,             only: rte_sw
  use mo_fluxes,             only: ty_fluxes_broadband
  use mo_rte_solver_kernels, only: lw_solver_2stream, sw_solver_2stream
  implicit none

  character(len=512) :: indir, lwfile, swfile
  integer :: ncol, nlay, ngas, i, ngpt_lw, ngpt_sw, nbnd_lw, nbnd_sw, u, ib
  character(len=32), allocatable :: gnames(:)
  real(wp), allocatable :: play(:,:), plev(:,:), tlay(:,:), tlev(:,:), tsfc(:), col_dry(:,:)
  real(wp), allocatable :: vmr(:,:,:), emis(:), mu0(:), alb_dir(:), alb_dif(:), toa_ours(:,:)
  real(wp), allocatable :: emis_b(:,:), adir_b(:,:), adif_b(:,:), toa_src(:,:)
  real(wp), allocatable, target :: fup(:,:), fdn(:,:), fdir(:,:)
  real(wp), allocatable :: sp_up(:,:,:), sp_dn(:,:,:), sp_dir(:,:,:), emis_g(:,:), zinc(:,:)
  real(wp), allocatable :: mu0_lay(:,:), adir_g(:,:), adif_g(:,:), bb1(:,:), bb2(:,:), bb3(:,:)
  type(ty_gas_concs) :: gas
  type(ty_gas_optics_rrtmgp) :: kd_lw, kd_sw
  type(ty_optical_props_2str) :: op_lw, op_sw
  type(ty_source_func_lw) :: src, src_own
  type(ty_fluxes_broadband) :: flx
  logical :: top_at_1

  call get_command_argument(1, indir)
  call get_command_argument(2, lwfile)
  call get_command_argument(3, swfile)

  open(newunit=u, file=trim(indir)//'/dims.txt', status='old')
  read(u,*) ncol, nlay, ngas
  allocate(gnames(ngas))
  do i = 1, ngas
    read(u,'(a)') gnames(i)
    gnames(i) = adjustl(gnames(i))
  end do
  close(u)

  call stop_on_err(gas%init(gnames))
  call load_and_init(kd_lw, trim(lwfile), gas)
  call load_and_init(kd_sw, trim(swfile), gas)
  ngpt_lw = kd_lw%get_ngpt(); nbnd_lw = kd_lw%get_nband()
  ngpt_sw = kd_sw%get_ngpt(); nbnd_sw = kd_sw%get_nband()
  print *, 'ORACLE ncol nlay ngas ngpt_lw ngpt_sw', ncol, nlay, ngas, ngpt_lw, ngpt_sw
  print *, 'ORACLE press_min press_max temp_min temp_max', kd_lw%get_press_min(), kd_lw%get_press_max(), &
           kd_lw%get_temp_min(), kd_lw%get_temp_max()

  allocate(play(ncol,nlay), plev(ncol,nlay+1), tlay(ncol,nlay), tlev(ncol,nlay+1), tsfc(ncol), &
           col_dry(ncol,nlay), vmr(ncol,nlay,ngas), emis(ncol), mu0(ncol), alb_dir(ncol), alb_dif(ncol), &
           toa_ours(ncol,ngpt_sw))
  open(newunit=u, file=trim(indir)//'/in.bin', access='stream', form='unformatted', status='old')
  read(u) play, plev, tlay, tlev, tsfc, col_dry, vmr, emis, mu0, alb_dir, alb_dif, toa_ours
  close(u)
  do i = 1, ngas
    call stop_on_err(gas%set_vmr(trim(gnames(i)), vmr(:,:,i)))
  end do
  top_at_1 = play(1,1) < play(1,nlay)
  print *, 'ORACLE top_at_1', top_at_1

  ! ------------------------------ LW ------------------------------
  call stop_on_err(op_lw%alloc_2str(ncol, nlay, kd_lw))
  call stop_on_err(src%alloc(ncol, nlay, kd_lw))
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src, col_dry=col_dry, tlev=tlev))
  allocate(fup(ncol,nlay+1), fdn(ncol,nlay+1), fdir(ncol,nlay+1))
  allocate(emis_b(nbnd_lw,ncol))
  do i = 1, ncol
    emis_b(:,i) = emis(i)
  end do
  flx%flux_up => fup; flx%flux_dn => fdn
  allocate(sp_up(ncol,nlay+1,ngpt_lw), sp_dn(ncol,nlay+1,ngpt_lw), emis_g(ncol,ngpt_lw), zinc(ncol,ngpt_lw))
  do i = 1, ncol
    emis_g(i,:) = emis(i)
  end do
  zinc = 0._wp
  open(newunit=u, file=trim(indir)//'/out_lw.bin', access='stream', form='unformatted', status='replace')
  write(u) op_lw%tau, src%lay_source, src%lev_source, src%sfc_source
  ! NOTE: in v1.8 rte_lw(..., use_2stream=.true.) with ty_fluxes_broadband never fills the
  ! broadband arrays (lw_solver_2stream has no broadband path and the broadband type skips
  ! reduce()), so the two-stream fluxes are taken from the kernel and summed here.
  ! (a) two-stream LW solver (the algorithm legoESM implements), legoESM tlev + col_dry
  call lw_solver_2stream(ncol, nlay, ngpt_lw, logical(top_at_1, wl), op_lw%tau, op_lw%ssa, op_lw%g, &
                         src%lay_source, src%lev_source, emis_g, src%sfc_source, zinc, sp_up, sp_dn)
  write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
  ! (b) RRTMGP default LW: no-scattering, 1 Gauss angle (secant 1/0.6096748751)
  call stop_on_err(rte_lw(op_lw, top_at_1, src, emis_b, flx))
  write(u) fup, fdn
  ! (c) two-stream with RRTMGP's own pressure-weighted tlev (tlev not supplied)
  call stop_on_err(src_own%alloc(ncol, nlay, kd_lw))
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src_own, col_dry=col_dry))
  call lw_solver_2stream(ncol, nlay, ngpt_lw, logical(top_at_1, wl), op_lw%tau, op_lw%ssa, op_lw%g, &
                         src_own%lay_source, src_own%lev_source, emis_g, src_own%sfc_source, zinc, sp_up, sp_dn)
  write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
  ! (d) two-stream with RRTMGP's own col_dry (from plev, g=9.80665, its m_dry/m_h2o), legoESM tlev
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src_own, tlev=tlev))
  write(u) op_lw%tau
  call lw_solver_2stream(ncol, nlay, ngpt_lw, logical(top_at_1, wl), op_lw%tau, op_lw%ssa, op_lw%g, &
                         src_own%lay_source, src_own%lev_source, emis_g, src_own%sfc_source, zinc, sp_up, sp_dn)
  write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
  ! (e) spectral 2-stream fluxes, configuration (a)
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src, col_dry=col_dry, tlev=tlev))
  call lw_solver_2stream(ncol, nlay, ngpt_lw, logical(top_at_1, wl), op_lw%tau, op_lw%ssa, op_lw%g, &
                         src%lay_source, src%lev_source, emis_g, src%sfc_source, zinc, sp_up, sp_dn)
  write(u) sp_up, sp_dn
  close(u)

  ! ------------------------------ SW ------------------------------
  call stop_on_err(op_sw%alloc_2str(ncol, nlay, kd_sw))
  allocate(toa_src(ncol,ngpt_sw))
  call stop_on_err(kd_sw%gas_optics(play, plev, tlay, gas, op_sw, toa_src, col_dry=col_dry))
  allocate(adir_b(nbnd_sw,ncol), adif_b(nbnd_sw,ncol))
  do i = 1, ncol
    adir_b(:,i) = alb_dir(i); adif_b(:,i) = alb_dif(i)
  end do
  flx%flux_up => fup; flx%flux_dn => fdn; flx%flux_dn_dir => fdir
  open(newunit=u, file=trim(indir)//'/out_sw.bin', access='stream', form='unformatted', status='replace')
  write(u) op_sw%tau, op_sw%ssa, op_sw%g, toa_src
  ! same per-g-point incident flux as legoESM (S0 * legoESM spectral weights)
  call stop_on_err(rte_sw(op_sw, top_at_1, mu0, toa_ours, adir_b, adif_b, flx))
  write(u) fup, fdn, fdir
  ! RRTMGP's own toa_src (quiet+facular+sunspot at default indices), rescaled to the same TSI
  do i = 1, ncol
    toa_src(i,:) = toa_src(i,:) * sum(toa_ours(i,:)) / sum(toa_src(i,:))
  end do
  call stop_on_err(rte_sw(op_sw, top_at_1, mu0, toa_src, adir_b, adif_b, flx))
  write(u) fup, fdn, fdir
  ! spectral from kernel with legoESM toa
  deallocate(sp_up, sp_dn)
  allocate(sp_up(ncol,nlay+1,ngpt_sw), sp_dn(ncol,nlay+1,ngpt_sw), sp_dir(ncol,nlay+1,ngpt_sw))
  allocate(mu0_lay(ncol,nlay), adir_g(ncol,ngpt_sw), adif_g(ncol,ngpt_sw), bb1(ncol,nlay+1), bb2(ncol,nlay+1), &
           bb3(ncol,nlay+1))
  do i = 1, ncol
    mu0_lay(i,:) = mu0(i); adir_g(i,:) = alb_dir(i); adif_g(i,:) = alb_dif(i)
  end do
  deallocate(zinc); allocate(zinc(ncol,ngpt_sw)); zinc = 0._wp
  call sw_solver_2stream(ncol, nlay, ngpt_sw, logical(top_at_1, wl), op_sw%tau, op_sw%ssa, op_sw%g, mu0_lay, &
                         adir_g, adif_g, toa_ours, sp_up, sp_dn, sp_dir, logical(.false., wl), zinc, &
                         logical(.false., wl), bb1, bb2, bb3)
  write(u) sp_up, sp_dn, sp_dir
  close(u)
  print *, 'ORACLE_DONE'
contains
  subroutine stop_on_err(msg)
    character(len=*), intent(in) :: msg
    if (len_trim(msg) > 0) then
      print *, 'ORACLE_ERROR: ', trim(msg)
      stop 1
    end if
  end subroutine stop_on_err
end program oracle_driver
