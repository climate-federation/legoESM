! Phase-4 c_sw one-step oracle driver.
!
! Reads the metric + state inputs exported by export_swcore_inputs.py
! (text records, Fortran-index keyed, halo-inclusive), runs the VERBATIM
! c_sw extraction ONCE on a single tile, and dumps every output field.
! The exporter supplies mpp-equivalent (kinked, cross-face) halos, so no
! communication is needed here.
!
! Input format (swcore_input_c<res>.txt):
!   header lines '# key value' for res, ng, dt2, nord
!   records: NAME i j value      (2-D fields)
!            NAME i j k value    (sin_sg/cos_sg, k = 1..9)
! Output (swcore_output_c<res>.txt): NAME i j value for
!   delpc, ptc, uc, vc, ut, vt, divg_d, ua, va, wc (+ echo of inputs' u,v).
!
! Build (gen_swcore_oracle.sh is the one command that does all of this):
!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -cpp \
!            -ffree-line-length-none shim + extract + driver
! PRODUCTION branch: no -DSW_DYNAMICS (ptc transports pt), no OVERLOAD_R4
! (sw_core big_number = 1.E30).
program fv3_swcore_oracle_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, &
                             fv_flags_type
  use sw_core_extract_mod
  implicit none

  type(fv_grid_bounds_type) :: bd
  type(fv_grid_type) :: gs
  type(fv_flags_type) :: fl

  integer :: res, ng, nord
  real :: dt2
  integer :: ios, i, j, k, u_in, u_out
  character(len=32) :: name
  character(len=256) :: line
  real :: val

  real, allocatable, dimension(:, :) :: delp, pt, u, v, w
  real, allocatable, dimension(:, :) :: uc, vc, ua, va, ut, vt
  real, allocatable, dimension(:, :) :: delpc, ptc, wc, divg_d

  ! ---- pass 1: header ----
  open(newunit=u_in, file='swcore_input.txt', status='old', action='read')
  res = -1; ng = -1; nord = 1; dt2 = -1.
  do
    read(u_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (line(1:1) /= '#') exit
    if (index(line, '# res ') == 1) read(line(7:), *) res
    if (index(line, '# ng ') == 1) read(line(6:), *) ng
    if (index(line, '# nord ') == 1) read(line(8:), *) nord
    if (index(line, '# dt2 ') == 1) read(line(7:), *) dt2
  end do
  close(u_in)
  if (res <= 0 .or. ng <= 0 .or. dt2 <= 0.) stop 'bad header'

  bd%is = 1;  bd%ie = res
  bd%js = 1;  bd%je = res
  bd%ng = ng
  bd%isd = 1 - ng; bd%ied = res + ng
  bd%jsd = 1 - ng; bd%jed = res + ng

  fl%grid_type = 0
  fl%npx = res + 1
  fl%npy = res + 1

  ! ---- allocate (halo-inclusive, upstream dims) ----
  allocate (gs%rarea (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (gs%dxa   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (gs%dya   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (gs%cosa_s(bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (gs%rsin2 (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (gs%dx    (bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (gs%dy    (bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (gs%dxc   (bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (gs%dyc   (bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (gs%rdxc  (bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (gs%rdyc  (bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (gs%cosa_u(bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (gs%sina_u(bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (gs%rsin_u(bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (gs%cosa_v(bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (gs%sina_v(bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (gs%rsin_v(bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (gs%rarea_c(bd%isd:bd%ied+1, bd%jsd:bd%jed+1))
  allocate (gs%fC    (bd%isd:bd%ied+1, bd%jsd:bd%jed+1))
  allocate (gs%sin_sg(bd%isd:bd%ied,   bd%jsd:bd%jed, 9))
  allocate (gs%cos_sg(bd%isd:bd%ied,   bd%jsd:bd%jed, 9))

  allocate (delp (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (pt   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (w    (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (u    (bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (vc   (bd%isd:bd%ied,   bd%jsd:bd%jed+1))
  allocate (v    (bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (uc   (bd%isd:bd%ied+1, bd%jsd:bd%jed))
  allocate (ua   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (va   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (ut   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (vt   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (delpc(bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (ptc  (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (wc   (bd%isd:bd%ied,   bd%jsd:bd%jed))
  allocate (divg_d(bd%isd:bd%ied+1, bd%jsd:bd%jed+1))

  ! poison everything the exporter must overwrite
  gs%rarea = -9.e9; gs%dxa = -9.e9; gs%dya = -9.e9
  gs%cosa_s = -9.e9; gs%rsin2 = -9.e9
  gs%dx = -9.e9; gs%dy = -9.e9; gs%dxc = -9.e9; gs%dyc = -9.e9
  gs%rdxc = -9.e9; gs%rdyc = -9.e9
  gs%cosa_u = -9.e9; gs%sina_u = -9.e9; gs%rsin_u = -9.e9
  gs%cosa_v = -9.e9; gs%sina_v = -9.e9; gs%rsin_v = -9.e9
  gs%rarea_c = -9.e9; gs%fC = -9.e9
  gs%sin_sg = -9.e9; gs%cos_sg = -9.e9
  delp = -9.e9; pt = -9.e9; u = -9.e9; v = -9.e9; w = 0.
  uc = 0.; vc = 0.; ua = 0.; va = 0.; ut = 0.; vt = 0.
  ! INTENT(OUT) args: pre-set so never-written deep-halo slots dump a
  ! recognizable sentinel instead of uninitialized memory
  delpc = -9.e9; ptc = -9.e9; wc = -9.e9; divg_d = -9.e9

  ! ---- pass 2: records ----
  open(newunit=u_in, file='swcore_input.txt', status='old', action='read')
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
      case ('RAREA');  gs%rarea(i, j) = val
      case ('DXA');    gs%dxa(i, j) = val
      case ('DYA');    gs%dya(i, j) = val
      case ('COSA_S'); gs%cosa_s(i, j) = val
      case ('RSIN2');  gs%rsin2(i, j) = val
      case ('DX');     gs%dx(i, j) = val
      case ('DY');     gs%dy(i, j) = val
      case ('DXC');    gs%dxc(i, j) = val
      case ('DYC');    gs%dyc(i, j) = val
      case ('RDXC');   gs%rdxc(i, j) = val
      case ('RDYC');   gs%rdyc(i, j) = val
      case ('COSA_U'); gs%cosa_u(i, j) = val
      case ('SINA_U'); gs%sina_u(i, j) = val
      case ('RSIN_U'); gs%rsin_u(i, j) = val
      case ('COSA_V'); gs%cosa_v(i, j) = val
      case ('SINA_V'); gs%sina_v(i, j) = val
      case ('RSIN_V'); gs%rsin_v(i, j) = val
      case ('RAREA_C'); gs%rarea_c(i, j) = val
      case ('FC');     gs%fC(i, j) = val
      case ('DELP');   delp(i, j) = val
      case ('PT');     pt(i, j) = val
      case ('U');      u(i, j) = val
      case ('V');      v(i, j) = val
      case default
        stop 'unknown record name'
      end select
    end select
  end do
  close(u_in)

  call c_sw(delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, &
            ut, vt, divg_d, nord, dt2, .true., .true., bd, gs, fl)

  open(newunit=u_out, file='swcore_output.txt', status='replace', action='write')
  write(u_out, '(A,I0)') '# res ', res
  do j = bd%jsd, bd%jed
    do i = bd%isd, bd%ied
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DELPC', i, j, delpc(i, j)
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'PTC', i, j, ptc(i, j)
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UA', i, j, ua(i, j)
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VA', i, j, va(i, j)
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UT', i, j, ut(i, j)
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VT', i, j, vt(i, j)
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
  do j = bd%jsd, bd%jed + 1
    do i = bd%isd, bd%ied + 1
      write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DIVG_D', i, j, divg_d(i, j)
    end do
  end do
  close(u_out)

  write(*, *) 'fv3_swcore_oracle: one c_sw step dumped'
end program fv3_swcore_oracle_driver
