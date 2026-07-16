! Phase-4c duo d2a2c_vect -> divergence_corner_duo CHAIN oracle driver.
!
! Runs the authoritative sub-chain the duo c_sw actually executes: the DUO
! d2a2c_vect (dg%is_initialized) produces ua/va, which feed
! divergence_corner_duo -> divg_d.  This certifies divergence_corner_duo on
! the REAL duo ua/va (its own oracle used plain-c_sw ua/va), closing that
! scope caveat end-to-end for the divergence sub-pipeline.
program fv3_dchain_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use sw_core_extract_mod, only: divergence_corner_duo
  use d2a2c_duo_extract_mod, only: d2a2c_vect
  implicit none

  type(fv_grid_bounds_type) :: bd
  type(fv_grid_type) :: gs
  type(fv_flags_type) :: fl
  integer :: res, ng, ios, i, j, k, u_in, u_out
  real :: val
  character(len=32) :: name
  character(len=256) :: line
  real, allocatable, dimension(:, :) :: u, v, ua, va, uc, vc, ut, vt, divg_d

  open(newunit=u_in, file='dchain_input.txt', status='old', action='read')
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

  ! d2a2c_vect metrics
  allocate (gs%cosa_s(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%rsin2 (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%cosa_u(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%rsin_u(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%cosa_v(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%rsin_v(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%dxa(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%dya(bd%isd:bd%ied, bd%jsd:bd%jed))
  ! divergence_corner_duo metrics
  allocate (gs%rarea_c(bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))
  allocate (gs%dxc(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%dyc(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%sin_sg(bd%isd:bd%ied, bd%jsd:bd%jed, 9))
  allocate (gs%cos_sg(bd%isd:bd%ied, bd%jsd:bd%jed, 9))
  allocate (u (bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (v (bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (ua(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (va(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (uc(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (vc(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (ut(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (vt(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (divg_d(bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))
  gs%cosa_s = -9.e9; gs%rsin2 = -9.e9; gs%cosa_u = -9.e9; gs%rsin_u = -9.e9
  gs%cosa_v = -9.e9; gs%rsin_v = -9.e9; gs%dxa = -9.e9; gs%dya = -9.e9
  gs%rarea_c = -9.e9; gs%dxc = -9.e9; gs%dyc = -9.e9
  gs%sin_sg = -9.e9; gs%cos_sg = -9.e9; u = -9.e9; v = -9.e9

  open(newunit=u_in, file='dchain_input.txt', status='old', action='read')
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
      case ('COSA_S'); gs%cosa_s(i, j) = val
      case ('RSIN2');  gs%rsin2(i, j) = val
      case ('COSA_U'); gs%cosa_u(i, j) = val
      case ('RSIN_U'); gs%rsin_u(i, j) = val
      case ('COSA_V'); gs%cosa_v(i, j) = val
      case ('RSIN_V'); gs%rsin_v(i, j) = val
      case ('DXA');    gs%dxa(i, j) = val
      case ('DYA');    gs%dya(i, j) = val
      case ('RAREA_C'); gs%rarea_c(i, j) = val
      case ('DXC');    gs%dxc(i, j) = val
      case ('DYC');    gs%dyc(i, j) = val
      case ('U');      u(i, j) = val
      case ('V');      v(i, j) = val
      case default;    stop 'unknown record'
      end select
    end select
  end do
  close(u_in)

  gs%dg%is_initialized = .true.
  gs%grid_type = 0
  gs%bounded_domain = .false.

  ! CHAIN: duo d2a2c_vect -> ua/va -> divergence_corner_duo -> divg_d
  ! (divergence_corner_duo is already the DUO routine; no duogrid flag)
  call d2a2c_vect(u, v, ua, va, uc, vc, ut, vt, .true., gs, bd, &
                  res + 1, res + 1, .false., 0)
  call divergence_corner_duo(u, v, ua, va, divg_d, gs, fl, bd)

  open(newunit=u_out, file='dchain_output.txt', status='replace', &
       action='write')
  do j = bd%jsd, bd%jed + 1
    do i = bd%isd, bd%ied + 1
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DIVGD', i, j, divg_d(i, j)
    end do
  end do
  close(u_out)
  write(*, *) 'fv3_dchain_oracle: duo d2a2c->divergence_corner_duo dumped'
end program fv3_dchain_driver
