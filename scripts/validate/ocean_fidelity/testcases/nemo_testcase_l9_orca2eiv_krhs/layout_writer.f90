program l9_layout_writer
  implicit none
  integer, parameter :: wp=kind(1.0d0), jpi=3, jpj=2, jpk=2
  integer, parameter :: nx=8, ny=8, nz=2
  integer :: unit, n
  real(wp) :: field(jpi,jpj,jpk), surface(jpi,jpj)
  real(wp) :: eiv(nx,ny,nz)
  character(len=16) :: magic
  character(len=16), dimension(6) :: tags
  real(wp), dimension(6) :: values

  magic = 'NEMO_L2_RKTR3_1'
  open(newunit=unit, file='oracle_rktracer_stage3_kt00000001.bin', &
       access='stream', form='unformatted', status='replace', action='write')
  write(unit) magic, 1, 1, 3, 2, 3, 4, 1, jpi, jpj, jpk, storage_size(1.0_wp)
  do n=1,22
    field = real(n,wp)
    write(unit) field
  end do
  do n=1,3
    surface = real(22+n,wp)
    write(unit) surface
  end do
  close(unit)

  magic = 'NEMO_L3_EIV_1'
  tags = [character(len=16) :: 'BEFORE_U', 'BEFORE_V', 'AFTER_U', &
          'AFTER_V', 'DELTA_U', 'DELTA_V']
  values = [1.0_wp, 2.0_wp, 3.0_wp, 5.0_wp, 2.0_wp, 3.0_wp]
  open(newunit=unit, file='oracle_l3_eiv_kt00000001.bin', access='stream', &
       form='unformatted', status='replace', action='write')
  write(unit) magic
  write(unit) 1, 1, 2, 3, nx, ny, nz, 64, 6
  do n=1,6
    eiv = values(n)
    write(unit) tags(n)
    write(unit) eiv
  end do
  close(unit)
end program l9_layout_writer
