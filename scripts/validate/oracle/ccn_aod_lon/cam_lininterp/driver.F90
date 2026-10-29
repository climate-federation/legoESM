! Run CAM's prescribed-data horizontal interpolation exactly as
! tracer_data.F90:read_2d_trc / read_3d_trc call it (cam_cesm2_1_rel_60,
! a03b84b7, lines 1372-1375 and 1539-1543):
!   lininterp_init(file%lons, nlon, to_lons, ncols, 2, lon_wgts, zero, twopi)
!   lininterp_init(file%lats, nlat, to_lats, ncols, 1, lat_wgts)
!   lininterp(field, nlon, nlat, out, ncols, lon_wgts, lat_wgts)
! Input (stream, native endian): int32 nlon nlat nfld ncol; float64
! lons(nlon) lats(nlat) [radians, file axes, lons in [0, 2pi) increasing];
! to_lons(ncol) to_lats(ncol) [radians]; field(nlon, nlat, nfld).
! Output (stream): float64 out(ncol, nfld).
program cam_lininterp_driver
  use shr_kind_mod, only: r8 => shr_kind_r8
  use interpolate_data, only: lininterp_init, lininterp, interp_type, lininterp_finish
  implicit none
  integer :: nlon, nlat, nfld, ncol, f
  real(r8), allocatable :: lons(:), lats(:), to_lons(:), to_lats(:), fld(:,:,:), out(:,:)
  real(r8), parameter :: zero = 0._r8, twopi = 2._r8*3.14159265358979323846_r8
  type(interp_type) :: lon_wgts, lat_wgts
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old')
  read(10) nlon, nlat, nfld, ncol
  allocate(lons(nlon), lats(nlat), to_lons(ncol), to_lats(ncol), fld(nlon, nlat, nfld), out(ncol, nfld))
  read(10) lons, lats, to_lons, to_lats, fld
  close(10)

  call lininterp_init(lons, nlon, to_lons, ncol, 2, lon_wgts, zero, twopi)
  call lininterp_init(lats, nlat, to_lats, ncol, 1, lat_wgts)
  do f = 1, nfld
     call lininterp(fld(:,:,f), nlon, nlat, out(:,f), ncol, lon_wgts, lat_wgts)
  end do
  call lininterp_finish(lon_wgts)
  call lininterp_finish(lat_wgts)

  open(11, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(11) out
  close(11)
end program cam_lininterp_driver
