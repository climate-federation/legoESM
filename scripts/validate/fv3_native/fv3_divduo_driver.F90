! Phase-4c divergence_corner_duo one-step oracle driver.
!
! Reads the gridstruct (rarea_c, sin_sg, cos_sg, dxc, dyc) + state
! (u, v, ua, va) exported by the python side, runs the VERBATIM authoritative
! divergence_corner_duo (Mouallem/Xi-Chen symmetryclean sw_core.F90:2345),
! and dumps divg_d.  Certifies the DUO-GRID divergence branch the plain-FV3
! phase-4a extraction lacks.
program fv3_divduo_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use sw_core_extract_mod, only: divergence_corner_duo
  implicit none

  type(fv_grid_bounds_type) :: bd
  type(fv_grid_type) :: gs
  type(fv_flags_type) :: fl
  integer :: res, ng, ios, i, j, k, u_in, u_out
  real :: val
  character(len=32) :: name
  character(len=256) :: line
  real, allocatable, dimension(:, :) :: u, v, ua, va, divg_d

  open(newunit=u_in, file='divduo_input.txt', status='old', action='read')
  res = -1; ng = -1
  do
    read(u_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (line(1:1) /= '#') exit
    if (index(line, '# res ') == 1) read(line(7:), *) res
    if (index(line, '# ng ') == 1) read(line(6:), *) ng
  end do
  close(u_in)
  if (res <= 0 .or. ng <= 0) stop 'bad header'

  bd%is = 1; bd%ie = res; bd%js = 1; bd%je = res; bd%ng = ng
  bd%isd = 1 - ng; bd%ied = res + ng; bd%jsd = 1 - ng; bd%jed = res + ng
  fl%grid_type = 0; fl%npx = res + 1; fl%npy = res + 1

  allocate (gs%rarea_c(bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))
  allocate (gs%dxc(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%dyc(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%sin_sg(bd%isd:bd%ied, bd%jsd:bd%jed, 9))
  allocate (gs%cos_sg(bd%isd:bd%ied, bd%jsd:bd%jed, 9))
  allocate (u(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (v(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (ua(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (va(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (divg_d(bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))
  gs%rarea_c = -9.e9; gs%dxc = -9.e9; gs%dyc = -9.e9
  gs%sin_sg = -9.e9; gs%cos_sg = -9.e9
  u = -9.e9; v = -9.e9; ua = -9.e9; va = -9.e9

  open(newunit=u_in, file='divduo_input.txt', status='old', action='read')
  do
    read(u_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (line(1:1) == '#') cycle
    read(line, *, iostat=ios) name
    if (ios /= 0) cycle
    select case (trim(name))
    case ('SIN_SG', 'COS_SG')
      read(line, *) name, i, j, k, val
      if (trim(name) == 'SIN_SG') gs%sin_sg(i, j, k) = val
      if (trim(name) == 'COS_SG') gs%cos_sg(i, j, k) = val
    case default
      read(line, *) name, i, j, val
      select case (trim(name))
      case ('RAREA_C'); gs%rarea_c(i, j) = val
      case ('DXC');     gs%dxc(i, j) = val
      case ('DYC');     gs%dyc(i, j) = val
      case ('U');       u(i, j) = val
      case ('V');       v(i, j) = val
      case ('UA');      ua(i, j) = val
      case ('VA');      va(i, j) = val
      case default
        stop 'unknown record'
      end select
    end select
  end do
  close(u_in)

  call divergence_corner_duo(u, v, ua, va, divg_d, gs, fl, bd)

  open(newunit=u_out, file='divduo_output.txt', status='replace', &
       action='write')
  do j = bd%jsd, bd%jed + 1
    do i = bd%isd, bd%ied + 1
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DIVGD', i, j, divg_d(i, j)
    end do
  end do
  close(u_out)
  write(*, *) 'fv3_divduo_oracle: divergence_corner_duo dumped'
end program fv3_divduo_driver
