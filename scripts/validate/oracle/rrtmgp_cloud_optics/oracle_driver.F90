! RTE-RRTMGP v1.8 cloud-optics oracle driver for
! tests/atmosphere/oracle_fixtures/rrtmgp_cloud_optics.npz.
! Unmodified oracle code is called: mo_cloud_optics_rrtmgp%cloud_optics (LUT path)
! into a ty_optical_props_1scl (absorption only, the LW reference treatment) and a
! ty_optical_props_2str, and mo_optical_props%delta_scale (SW, as in
! examples/all-sky/rrtmgp_allsky.F90).  Adapted from the tier-1 audit driver
! (oracle_audit/radiation_cloud_optics_mcica/fortran/cloud_oracle.F90).
!
! usage: oracle_driver <lut.nc> <dir> <icergh>
!   <dir>/dims.txt : ncol nlay nbnd
!   <dir>/in.bin   : clwp reliq ciwp reice (ncol,nlay)  [g/m2, um, g/m2, um table coordinate]
!   writes <dir>/out.bin : tau_1scl, (tau, ssa, g) 2str, (tau, ssa, g) 2str delta-scaled
program oracle_driver
  use mo_rte_kind,            only: wp
  use mo_optical_props,       only: ty_optical_props_2str, ty_optical_props_1scl
  use mo_cloud_optics_rrtmgp, only: ty_cloud_optics_rrtmgp
  use mo_load_cloud_coefficients, only: load_cld_lutcoeff
  implicit none
  character(len=512) :: lutfile, dir, arg
  integer :: ncol, nlay, nbnd, u, icergh
  real(wp), allocatable :: clwp(:,:), ciwp(:,:), rel(:,:), rei(:,:)
  type(ty_cloud_optics_rrtmgp) :: cop
  type(ty_optical_props_2str) :: c2, c2ds
  type(ty_optical_props_1scl) :: c1

  call get_command_argument(1, lutfile)
  call get_command_argument(2, dir)
  call get_command_argument(3, arg); read(arg,*) icergh
  open(newunit=u, file=trim(dir)//'/dims.txt', status='old')
  read(u,*) ncol, nlay, nbnd
  close(u)
  allocate(clwp(ncol,nlay), ciwp(ncol,nlay), rel(ncol,nlay), rei(ncol,nlay))
  open(newunit=u, file=trim(dir)//'/in.bin', access='stream', form='unformatted', status='old')
  read(u) clwp, rel, ciwp, rei
  close(u)

  call load_cld_lutcoeff(cop, trim(lutfile))
  call stop_on_err(cop%set_ice_roughness(icergh))
  if (cop%get_nband() /= nbnd) call stop_on_err('band count mismatch')

  call stop_on_err(c1%alloc_1scl(ncol, nlay, cop))
  call stop_on_err(cop%cloud_optics(clwp, ciwp, rel, rei, c1))
  call stop_on_err(c2%alloc_2str(ncol, nlay, cop))
  call stop_on_err(cop%cloud_optics(clwp, ciwp, rel, rei, c2))
  call stop_on_err(c2ds%alloc_2str(ncol, nlay, cop))
  c2ds%tau = c2%tau; c2ds%ssa = c2%ssa; c2ds%g = c2%g
  call stop_on_err(c2ds%delta_scale())

  open(newunit=u, file=trim(dir)//'/out.bin', access='stream', form='unformatted', status='replace')
  write(u) c1%tau
  write(u) c2%tau, c2%ssa, c2%g
  write(u) c2ds%tau, c2ds%ssa, c2ds%g
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
