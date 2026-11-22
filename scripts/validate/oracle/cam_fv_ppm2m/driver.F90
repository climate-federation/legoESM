! CAM-FV map1_ppm oracle: reads one remap problem per call from stdin and
! writes the remapped cell means to stdout.
! Line 1: km kn iv kord
! Line 2: source edges (km+1, top first); line 3: target edges (kn+1);
! line 4: source cell means (km).  Edges in any increasing coordinate.
program map1_ppm_driver
  use shr_kind_mod, only: r8 => shr_kind_r8
  use mapz_module, only: map1_ppm
  implicit none
  integer :: km, kn, iv, kord, k
  real(r8), allocatable :: pe1(:,:), pe2(:,:), q1(:,:,:), q2(:,:,:)
  read(*,*) km, kn, iv, kord
  allocate(pe1(1,km+1), pe2(1,kn+1), q1(1,1:1,km), q2(1,1:1,kn))
  read(*,*) pe1(1,:)
  read(*,*) pe2(1,:)
  read(*,*) q1(1,1,:)
  q2 = 0.0_r8
  call map1_ppm(km, pe1, q1, kn, pe2, q2, 0, 0, 1, 1, 1, 1, 1, 1, iv, kord)
  do k = 1, kn
    write(*,'(ES27.17E3)') q2(1,1,k)
  end do
end program map1_ppm_driver
