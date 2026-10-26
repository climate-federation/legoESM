! Prints the lego_sat stub's liquid/ice saturation specific humidity for (T,p) pairs
! read from argv1, so make_fixture.py can verify the stub against legoesm.thermo.
program qsat_check
  use lego_sat, only: qsl_lego, qsi_lego
  implicit none
  character(len=1024) :: fin
  real :: t, p
  integer :: ios
  call get_command_argument(1, fin)
  open(11, file=trim(fin), status='old')
  do
     read(11,*,iostat=ios) t, p
     if (ios /= 0) exit
     write(*,'(4es26.17e3)') t, p, qsl_lego(t,p), qsi_lego(t,p)
  end do
end program qsat_check
