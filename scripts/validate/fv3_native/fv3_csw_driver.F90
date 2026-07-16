! Phase-4c duo c_sw one-step oracle driver.
!
! Reads the full gridstruct (metrics for c_sw + its d2a2c_vect/divergence
! calls) + state (delp, pt, w, u, v), sets flagstruct%duogrid=.true. and
! gs%dg%is_initialized=.true., runs the VERBATIM authoritative symmetryclean
! c_sw DUO branch (sw_core.F90:79-494), and dumps delpc, ptc, wc, uc, vc.
program fv3_csw_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use c_sw_duo_extract_mod, only: c_sw
  implicit none

  type(fv_grid_bounds_type) :: bd
  type(fv_grid_type) :: gs
  type(fv_flags_type) :: fl
  integer :: res, ng, ios, i, j, k, u_in, u_out
  real :: val
  character(len=32) :: name
  character(len=256) :: line
  real, allocatable, dimension(:, :) :: delp, pt, w, u, v
  real, allocatable, dimension(:, :) :: delpc, ptc, wc, uc, vc, ua, va, ut, vt
  real, allocatable, dimension(:, :) :: divg_d

  open(newunit=u_in, file='csw_input.txt', status='old', action='read')
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
  fl%grid_type = 0; fl%npx = res + 1; fl%npy = res + 1; fl%duogrid = .true.

  ! AA (isd:ied, jsd:jed)
  allocate (gs%cosa_s(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%rsin2 (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%rarea (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%dxa   (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (gs%dya   (bd%isd:bd%ied, bd%jsd:bd%jed))
  ! BA (isd:ied+1, jsd:jed)
  allocate (gs%cosa_u(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%sina_u(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%rsin_u(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%dy    (bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%dxc   (bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (gs%rdxc  (bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  ! AB (isd:ied, jsd:jed+1)
  allocate (gs%cosa_v(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%sina_v(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%rsin_v(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%dx    (bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%dyc   (bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (gs%rdyc  (bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  ! BB (isd:ied+1, jsd:jed+1)
  allocate (gs%rarea_c(bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))
  allocate (gs%fC     (bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))
  ! SG
  allocate (gs%sin_sg(bd%isd:bd%ied, bd%jsd:bd%jed, 9))
  allocate (gs%cos_sg(bd%isd:bd%ied, bd%jsd:bd%jed, 9))
  ! state
  allocate (delp(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (pt  (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (w   (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (u (bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (v (bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  ! outputs / workspace
  allocate (delpc(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (ptc  (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (wc   (bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (ua(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (va(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (ut(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (vt(bd%isd:bd%ied, bd%jsd:bd%jed))
  allocate (uc(bd%isd:bd%ied + 1, bd%jsd:bd%jed))
  allocate (vc(bd%isd:bd%ied, bd%jsd:bd%jed + 1))
  allocate (divg_d(bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1))

  gs%cosa_s = -9.e9; gs%rsin2 = -9.e9; gs%rarea = -9.e9; gs%dxa = -9.e9
  gs%dya = -9.e9; gs%cosa_u = -9.e9; gs%sina_u = -9.e9; gs%rsin_u = -9.e9
  gs%dy = -9.e9; gs%dxc = -9.e9; gs%rdxc = -9.e9; gs%cosa_v = -9.e9
  gs%sina_v = -9.e9; gs%rsin_v = -9.e9; gs%dx = -9.e9; gs%dyc = -9.e9
  gs%rdyc = -9.e9; gs%rarea_c = -9.e9; gs%fC = -9.e9; gs%sin_sg = -9.e9
  gs%cos_sg = -9.e9; delp = -9.e9; pt = -9.e9; w = -9.e9; u = -9.e9; v = -9.e9
  ! outputs / workspace init to a BIG sentinel (|.|>1e24) so cells c_sw does
  ! not write are excluded by the test's non-sentinel mask, matching the
  ! port's NaN-initialised unwritten cells (else the region masks disagree)
  delpc = 1.e30; ptc = 1.e30; wc = 1.e30; ua = 1.e30; va = 1.e30
  ut = 1.e30; vt = 1.e30; uc = 1.e30; vc = 1.e30; divg_d = 1.e30

  open(newunit=u_in, file='csw_input.txt', status='old', action='read')
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
      case ('RAREA');  gs%rarea(i, j) = val
      case ('DXA');    gs%dxa(i, j) = val
      case ('DYA');    gs%dya(i, j) = val
      case ('COSA_U'); gs%cosa_u(i, j) = val
      case ('SINA_U'); gs%sina_u(i, j) = val
      case ('RSIN_U'); gs%rsin_u(i, j) = val
      case ('DY');     gs%dy(i, j) = val
      case ('DXC');    gs%dxc(i, j) = val
      case ('RDXC');   gs%rdxc(i, j) = val
      case ('COSA_V'); gs%cosa_v(i, j) = val
      case ('SINA_V'); gs%sina_v(i, j) = val
      case ('RSIN_V'); gs%rsin_v(i, j) = val
      case ('DX');     gs%dx(i, j) = val
      case ('DYC');    gs%dyc(i, j) = val
      case ('RDYC');   gs%rdyc(i, j) = val
      case ('RAREA_C'); gs%rarea_c(i, j) = val
      case ('FC');     gs%fC(i, j) = val
      case ('DELP');   delp(i, j) = val
      case ('PT');     pt(i, j) = val
      case ('W');      w(i, j) = val
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
  gs%sw_corner = .true.; gs%se_corner = .true.
  gs%ne_corner = .true.; gs%nw_corner = .true.

  call c_sw(delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, &
            ut, vt, divg_d, 1, 112.5, .true., .true., bd, gs, fl)

  open(newunit=u_out, file='csw_output.txt', status='replace', &
       action='write')
  do j = bd%jsd, bd%jed
    do i = bd%isd, bd%ied
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DELPC', i, j, delpc(i, j)
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'PTC', i, j, ptc(i, j)
    end do
  end do
  do j = bd%jsd, bd%jed
    do i = bd%isd, bd%ied + 1
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UC', i, j, uc(i, j)
    end do
  end do
  do j = bd%jsd, bd%jed + 1
    do i = bd%isd, bd%ied
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VC', i, j, vc(i, j)
    end do
  end do
  close(u_out)
  write(*, *) 'fv3_csw_oracle: duo c_sw dumped'
end program fv3_csw_driver
