! RTE+RRTMGP v1.8 gas optics with value checks OFF (rte_config_checks), for the
! out-of-table pressure rule of legoESM's RRTMGP core.  Upstream errors on a
! layer pressure outside [press_ref_min, press_ref_max] by default; with checks
! off its kernel clamps the index and extrapolates the weight
! (mo_gas_optics_rrtmgp_kernels.F90:115-117).  Unpatched upstream.
!
! usage: oracle_driver <dir> <lw_gas_file> <sw_gas_file>
!   <dir>/dims.txt : ncol nlay ngas, then gas names
!   <dir>/in.bin   : play(ncol,nlay) plev(ncol,nlay+1) tlay(ncol,nlay) tlev(ncol,nlay+1)
!                    tsfc(ncol) col_dry(ncol,nlay) vmr(ncol,nlay,ngas)   (float64, Fortran order)
!   writes <dir>/out.bin : tau_lw lay_source (ncol,nlay,ngpt_lw), tau_sw ssa_sw (ncol,nlay,ngpt_sw)
program oracle_driver
  use mo_rte_kind,           only: wp, wl
  use mo_rte_config,         only: rte_config_checks
  use mo_optical_props,      only: ty_optical_props_2str
  use mo_source_functions,   only: ty_source_func_lw
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  use mo_load_coefficients,  only: load_and_init
  implicit none

  character(len=512) :: dir, lwfile, swfile
  integer :: ncol, nlay, ngas, i, u
  character(len=32), allocatable :: gnames(:)
  real(wp), allocatable :: play(:,:), plev(:,:), tlay(:,:), tlev(:,:), tsfc(:), col_dry(:,:)
  real(wp), allocatable :: vmr(:,:,:), toa_src(:,:)
  type(ty_gas_concs) :: gas
  type(ty_gas_optics_rrtmgp) :: kd_lw, kd_sw
  type(ty_optical_props_2str) :: op_lw, op_sw
  type(ty_source_func_lw) :: src

  call get_command_argument(1, dir)
  call get_command_argument(2, lwfile)
  call get_command_argument(3, swfile)
  open(newunit=u, file=trim(dir)//'/dims.txt', status='old')
  read(u,*) ncol, nlay, ngas
  allocate(gnames(ngas))
  do i = 1, ngas
    read(u,'(a)') gnames(i)
    gnames(i) = adjustl(gnames(i))
  end do
  close(u)
  allocate(play(ncol,nlay), plev(ncol,nlay+1), tlay(ncol,nlay), tlev(ncol,nlay+1), tsfc(ncol), &
           col_dry(ncol,nlay), vmr(ncol,nlay,ngas))
  open(newunit=u, file=trim(dir)//'/in.bin', access='stream', form='unformatted', status='old')
  read(u) play, plev, tlay, tlev, tsfc, col_dry, vmr
  close(u)

  call rte_config_checks(logical(.false., wl))
  call stop_on_err(gas%init(gnames))
  do i = 1, ngas
    call stop_on_err(gas%set_vmr(trim(gnames(i)), vmr(:,:,i)))
  end do
  call load_and_init(kd_lw, trim(lwfile), gas)
  call load_and_init(kd_sw, trim(swfile), gas)
  print *, 'ORACLE press_min press_max', kd_lw%get_press_min(), kd_lw%get_press_max()

  call stop_on_err(op_lw%alloc_2str(ncol, nlay, kd_lw))
  call stop_on_err(src%alloc(ncol, nlay, kd_lw))
  call stop_on_err(kd_lw%gas_optics(play, plev, tlay, tsfc, gas, op_lw, src, col_dry=col_dry, tlev=tlev))
  call stop_on_err(op_sw%alloc_2str(ncol, nlay, kd_sw))
  allocate(toa_src(ncol, kd_sw%get_ngpt()))
  call stop_on_err(kd_sw%gas_optics(play, plev, tlay, gas, op_sw, toa_src, col_dry=col_dry))

  open(newunit=u, file=trim(dir)//'/out.bin', access='stream', form='unformatted', status='replace')
  write(u) op_lw%tau, src%lay_source, op_sw%tau, op_sw%ssa
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
