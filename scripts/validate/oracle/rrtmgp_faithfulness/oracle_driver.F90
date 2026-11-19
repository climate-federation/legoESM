! Unpatched latest RTE+RRTMGP driver for the legoESM numerical-fidelity audit.
! Reads inputs written by audit.py (float64, Fortran order); writes every intermediate.
!
! usage: oracle_driver <dir> <lw_gas> <sw_gas> <lw_cloud> <sw_cloud> <ice_roughness>
!   dims.txt : ncol nlay ngas, then gas names
!   in.bin   : play plev tlay tlev tsfc col_dry vmr emis mu0 alb_dir alb_dif toa(ncol,ngpt_sw)
!   cloud.bin (optional): clwp[g/m2] rel[um] ciwp[g/m2] rei[um, table coordinate]
!   synth_dims.txt + synth.bin (optional): solver-only case, one g-point:
!     tau ssa g (n,l)  lev_source (n,l+1)  emis sfc_src mu0 alb_dir alb_dif inc_dir (n)
program oracle_driver
  use mo_rte_kind,            only: wp, wl
  use mo_optical_props,       only: ty_optical_props_2str, ty_optical_props_1scl
  use mo_source_functions,    only: ty_source_func_lw
  use mo_gas_concentrations,  only: ty_gas_concs
  use mo_gas_optics_rrtmgp,   only: ty_gas_optics_rrtmgp
  use mo_cloud_optics_rrtmgp, only: ty_cloud_optics_rrtmgp
  use mo_optics_utils_rrtmgp, only: load_gas_optics, load_cloud_optics
  use mo_rte_lw,              only: rte_lw
  use mo_rte_sw,              only: rte_sw
  use mo_fluxes,              only: ty_fluxes_broadband
  use mo_rte_solver_kernels,  only: lw_solver_2stream, sw_solver_2stream
  implicit none

  character(len=512) :: dir, lwfile, swfile, lwcld, swcld, arg
  integer :: ncol, nlay, ngas, i, ngpt_lw, ngpt_sw, nbnd_lw, nbnd_sw, u, icergh
  logical :: top_at_1, has_cloud, has_synth
  character(len=32), allocatable :: gnames(:)
  real(wp), allocatable :: play(:,:), plev(:,:), tlay(:,:), tlev(:,:), tsfc(:), col_dry(:,:)
  real(wp), allocatable :: vmr(:,:,:), emis(:), mu0(:), alb_dir(:), alb_dif(:), toa(:,:)
  real(wp), allocatable :: emis_b(:,:), adir_b(:,:), adif_b(:,:), toa_src(:,:)
  real(wp), allocatable, target :: fup(:,:), fdn(:,:), fdir(:,:)
  real(wp), allocatable :: sp_up(:,:,:), sp_dn(:,:,:), emis_g(:,:), zinc(:,:)
  real(wp), allocatable :: clwp(:,:), rel(:,:), ciwp(:,:), rei(:,:)
  type(ty_gas_concs) :: gas
  type(ty_gas_optics_rrtmgp) :: kd_lw, kd_sw
  type(ty_cloud_optics_rrtmgp) :: cop_lw, cop_sw
  type(ty_optical_props_2str) :: op_lw, op_sw, c2
  type(ty_optical_props_1scl) :: c1
  type(ty_source_func_lw) :: src, src_own
  type(ty_fluxes_broadband) :: flx

  call get_command_argument(1, dir)
  call get_command_argument(2, lwfile)
  call get_command_argument(3, swfile)
  call get_command_argument(4, lwcld)
  call get_command_argument(5, swcld)
  call get_command_argument(6, arg); read(arg,*) icergh

  inquire(file=trim(dir)//'/synth.bin', exist=has_synth)
  if (has_synth) then
    call synth()
    stop
  end if

  open(newunit=u, file=trim(dir)//'/dims.txt', status='old')
  read(u,*) ncol, nlay, ngas
  allocate(gnames(ngas))
  do i = 1, ngas
    read(u,'(a)') gnames(i)
    gnames(i) = adjustl(gnames(i))
  end do
  close(u)

  call stop_on_err(gas%init(gnames))
  call load_gas_optics(kd_lw, trim(lwfile), gas)
  call load_gas_optics(kd_sw, trim(swfile), gas)
  ngpt_lw = kd_lw%get_ngpt(); nbnd_lw = kd_lw%get_nband()
  ngpt_sw = kd_sw%get_ngpt(); nbnd_sw = kd_sw%get_nband()
  print *, 'ORACLE ncol nlay ngas ngpt_lw ngpt_sw', ncol, nlay, ngas, ngpt_lw, ngpt_sw
  print *, 'ORACLE press_min press_max temp_min temp_max', kd_lw%get_press_min(), &
           kd_lw%get_press_max(), kd_lw%get_temp_min(), kd_lw%get_temp_max()

  allocate(play(ncol,nlay), plev(ncol,nlay+1), tlay(ncol,nlay), tlev(ncol,nlay+1), tsfc(ncol), &
           col_dry(ncol,nlay), vmr(ncol,nlay,ngas), emis(ncol), mu0(ncol), alb_dir(ncol), &
           alb_dif(ncol), toa(ncol,ngpt_sw))
  open(newunit=u, file=trim(dir)//'/in.bin', access='stream', form='unformatted', status='old')
  read(u) play, plev, tlay, tlev, tsfc, col_dry, vmr, emis, mu0, alb_dir, alb_dif, toa
  close(u)
  do i = 1, ngas
    call stop_on_err(gas%set_vmr(trim(gnames(i)), vmr(:,:,i)))
  end do
  top_at_1 = play(1,1) < play(1,nlay)
  print *, 'ORACLE top_at_1', top_at_1

  ! ------------------------------ LW ------------------------------
  call stop_on_err(op_lw%alloc_2str(ncol, nlay, kd_lw))
  call stop_on_err(src%alloc(ncol, nlay, kd_lw))
  call stop_on_err(src_own%alloc(ncol, nlay, kd_lw))
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src, col_dry=col_dry, tlev=tlev))
  allocate(fup(ncol,nlay+1), fdn(ncol,nlay+1), fdir(ncol,nlay+1), emis_b(nbnd_lw,ncol))
  allocate(sp_up(ncol,nlay+1,ngpt_lw), sp_dn(ncol,nlay+1,ngpt_lw), emis_g(ncol,ngpt_lw), zinc(ncol,ngpt_lw))
  do i = 1, ncol
    emis_b(:,i) = emis(i); emis_g(i,:) = emis(i)
  end do
  zinc = 0._wp
  flx%flux_up => fup; flx%flux_dn => fdn
  open(newunit=u, file=trim(dir)//'/out_lw.bin', access='stream', form='unformatted', status='replace')
  write(u) op_lw%tau, src%lay_source, src%lev_source, src%sfc_source
  ! (a) two-stream solver, inputs as ours
  call two_stream(src)
  write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
  ! (b) upstream default: no scattering, one angle; (c) three angles
  call stop_on_err(rte_lw(op_lw, src, emis_b, flx))
  write(u) fup, fdn
  call stop_on_err(rte_lw(op_lw, src, emis_b, flx, n_gauss_angles=3))
  write(u) fup, fdn
  ! (d) two-stream, upstream's own level temperatures
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src_own, col_dry=col_dry))
  call two_stream(src_own)
  write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
  ! (e) two-stream, upstream's own dry-air column
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src_own, tlev=tlev))
  write(u) op_lw%tau
  call two_stream(src_own)
  write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
  close(u)

  ! ------------------------------ SW ------------------------------
  call stop_on_err(op_sw%alloc_2str(ncol, nlay, kd_sw))
  allocate(toa_src(ncol,ngpt_sw), adir_b(nbnd_sw,ncol), adif_b(nbnd_sw,ncol))
  call stop_on_err(kd_sw%gas_optics(play, plev, tlay, gas, op_sw, toa_src, col_dry=col_dry))
  do i = 1, ncol
    adir_b(:,i) = alb_dir(i); adif_b(:,i) = alb_dif(i)
  end do
  flx%flux_dn_dir => fdir
  open(newunit=u, file=trim(dir)//'/out_sw.bin', access='stream', form='unformatted', status='replace')
  write(u) op_sw%tau, op_sw%ssa, op_sw%g, toa_src
  call stop_on_err(rte_sw(op_sw, mu0, toa, adir_b, adif_b, flx))
  write(u) fup, fdn, fdir
  ! upstream's own spectral weights, rescaled to the same total irradiance
  do i = 1, ncol
    toa_src(i,:) = toa_src(i,:) * sum(toa(i,:)) / sum(toa_src(i,:))
  end do
  call stop_on_err(rte_sw(op_sw, mu0, toa_src, adir_b, adif_b, flx))
  write(u) fup, fdn, fdir
  close(u)

  ! ------------------------------ all-sky ------------------------------
  inquire(file=trim(dir)//'/cloud.bin', exist=has_cloud)
  if (has_cloud) then
    allocate(clwp(ncol,nlay), rel(ncol,nlay), ciwp(ncol,nlay), rei(ncol,nlay))
    open(newunit=u, file=trim(dir)//'/cloud.bin', access='stream', form='unformatted', status='old')
    read(u) clwp, rel, ciwp, rei
    close(u)
    call load_cloud_optics(cop_lw, trim(lwcld))
    call load_cloud_optics(cop_sw, trim(swcld))
    call stop_on_err(cop_lw%set_ice_roughness(icergh))
    call stop_on_err(cop_sw%set_ice_roughness(icergh))
    open(newunit=u, file=trim(dir)//'/out_cld.bin', access='stream', form='unformatted', status='replace')
    ! LW: absorption-only clouds added to gas, two-stream solver
    call stop_on_err(c1%alloc_1scl(ncol, nlay, cop_lw))
    call stop_on_err(cop_lw%cloud_optics(clwp, ciwp, rel, rei, c1))
    call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src, col_dry=col_dry, tlev=tlev))
    call stop_on_err(c1%increment(op_lw))
    call two_stream(src)
    write(u) sum(sp_up, dim=3), sum(sp_dn, dim=3)
    call stop_on_err(rte_lw(op_lw, src, emis_b, flx))
    write(u) fup, fdn
    ! SW: delta-scaled clouds added to gas, as examples/all-sky/rrtmgp_allsky.F90
    call stop_on_err(c2%alloc_2str(ncol, nlay, cop_sw))
    call stop_on_err(cop_sw%cloud_optics(clwp, ciwp, rel, rei, c2))
    call stop_on_err(c2%delta_scale())
    call stop_on_err(kd_sw%gas_optics(play, plev, tlay, gas, op_sw, toa_src, col_dry=col_dry))
    call stop_on_err(c2%increment(op_sw))
    call stop_on_err(rte_sw(op_sw, mu0, toa, adir_b, adif_b, flx))
    write(u) fup, fdn, fdir
    close(u)
  end if
  print *, 'ORACLE_DONE'

contains
  ! lw_solver_2stream passes lev_source to lw_source_2str without the g-point index
  ! (mo_rte_solver_kernels.F90:422), so with ngpt > 1 every g-point uses g-point 1's
  ! level source. One g-point per call gives what the routine intends, unpatched.
  subroutine two_stream(s)
    type(ty_source_func_lw), intent(in) :: s
    integer :: ig
    do ig = 1, ngpt_lw
      call lw_solver_2stream(ncol, nlay, 1, logical(top_at_1, wl), op_lw%tau(:,:,ig:ig), &
                             op_lw%ssa(:,:,ig:ig), op_lw%g(:,:,ig:ig), s%lay_source(:,:,ig:ig), &
                             s%lev_source(:,:,ig:ig), emis_g(:,ig:ig), s%sfc_source(:,ig:ig), &
                             zinc(:,ig:ig), sp_up(:,:,ig:ig), sp_dn(:,:,ig:ig))
    end do
  end subroutine two_stream

  subroutine synth()
    integer :: n, l, v, j
    real(wp), allocatable :: tau(:,:,:), ssa(:,:,:), g(:,:,:), lev(:,:,:), lay(:,:,:)
    real(wp), allocatable :: e(:,:), ss(:,:), m(:), ad(:,:), af(:,:), inc(:,:), mu(:,:), zi(:,:)
    real(wp), allocatable :: up(:,:,:), dn(:,:,:), dr(:,:,:), b1(:,:), b2(:,:), b3(:,:)
    open(newunit=v, file=trim(dir)//'/synth_dims.txt', status='old')
    read(v,*) n, l
    close(v)
    allocate(tau(n,l,1), ssa(n,l,1), g(n,l,1), lev(n,l+1,1), lay(n,l,1), e(n,1), ss(n,1), m(n), &
             ad(n,1), af(n,1), inc(n,1), mu(n,l), zi(n,1), up(n,l+1,1), dn(n,l+1,1), dr(n,l+1,1), &
             b1(n,l+1), b2(n,l+1), b3(n,l+1))
    open(newunit=v, file=trim(dir)//'/synth.bin', access='stream', form='unformatted', status='old')
    read(v) tau, ssa, g, lev, e, ss, m, ad, af, inc
    close(v)
    lay = 0._wp; zi = 0._wp
    do j = 1, l
      mu(:,j) = m
    end do
    open(newunit=v, file=trim(dir)//'/synth_out.bin', access='stream', form='unformatted', status='replace')
    call lw_solver_2stream(n, l, 1, logical(.false., wl), tau, ssa, g, lay, lev, e, ss, zi, up, dn)
    write(v) up, dn
    call sw_solver_2stream(n, l, 1, logical(.false., wl), tau, ssa, g, mu, ad, af, inc, up, dn, dr, &
                           logical(.false., wl), zi, logical(.false., wl), b1, b2, b3)
    write(v) up, dn, dr
    close(v)
    print *, 'ORACLE_SYNTH_DONE', n, l
  end subroutine synth

  subroutine stop_on_err(msg)
    character(len=*), intent(in) :: msg
    if (len_trim(msg) > 0) then
      print *, 'ORACLE_ERROR: ', trim(msg)
      stop 1
    end if
  end subroutine stop_on_err
end program oracle_driver
