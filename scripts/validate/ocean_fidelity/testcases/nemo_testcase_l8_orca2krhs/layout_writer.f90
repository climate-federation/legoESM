program l8_layout_writer
  implicit none
  integer, parameter :: wp=kind(1.0d0), jpi=3, jpj=2, jpk=2
  integer :: unit, n
  real(wp) :: field(jpi,jpj,jpk), surface(jpi,jpj)
  character(len=16) :: magic
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
end program l8_layout_writer
