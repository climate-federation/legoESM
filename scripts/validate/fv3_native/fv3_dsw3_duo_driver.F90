! Phase-4c duo d_sw3 chain-oracle driver.
!
! Same input protocol as the d_sw1 driver (dswcore_input.txt from the
! committed npz; fl%duogrid + dg%is_initialized; ut/vt=1e30 sentinels),
! runs VERBATIM d_sw1 (dsw1_duo_extract_mod) so ut/vt carry the
! certified defined-workspace values, then VERBATIM d_sw3
! (dsw3_duo_extract_mod) — the KE-flux stage.  On the duo lane d_sw3's
! vt/ut-based edge extrapolations are skipped (interior contravariant
! formulas everywhere), so the outputs must be ut/vt-independent — the
! python port never reads them, and any hidden Fortran-side read would
! break the bit-compare.  ubbtemp/vbbtemp/ubb/vbb are pre-initialised
! to 1e30 and fully overwritten on the B-grid compute ring (the pack
! step's NaN check enforces full token coverage).
program fv3_dsw3_duo_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, &
                             fv_flags_type, set_fill_corner_bounds
  use dsw1_duo_extract_mod, only: d_sw1
  use dsw3_duo_extract_mod, only: d_sw3
  implicit none

  type(fv_grid_bounds_type) :: bd
  type(fv_grid_type) :: gs
  type(fv_flags_type) :: fl

  integer :: res, nhalo
  real :: dt
  real :: dddmp_in, d4bg_in, dampv_in
  integer :: ios, i, j, k, u_in, u_out
  character(len=32) :: name
  character(len=256) :: line
  real :: val

  real, allocatable, dimension(:, :) :: delp, pt, w, u, v, uc, vc, ua, va
  real, allocatable, dimension(:, :) :: delpc, ptc, divg_d
  real, allocatable, dimension(:, :) :: xflux, yflux, cx, cy
  real, allocatable, dimension(:, :) :: crx_adv, cry_adv, xfx_adv, yfx_adv
  real, allocatable, dimension(:, :) :: z_rat, heat_source, diss_est
  real, allocatable :: q(:, :, :, :), q_con(:, :)
  real, allocatable :: allflux_x(:, :, :, :), allflux_y(:, :, :, :)
  real, allocatable :: ra_x(:, :), ra_y(:, :)
  real, allocatable :: ut(:, :), vt(:, :)
  real, allocatable :: ubbtemp(:, :), vbbtemp(:, :), ubb(:, :), vbb(:, :)

  ! ---- pass 1: header ----
  open(newunit=u_in, file='dswcore_input.txt', status='old', action='read')
  res = -1; nhalo = -1; dt = -1.
  ! damping knobs default to the oracle configuration; optional header
  ! overrides support config bisection without touching the extraction
  dddmp_in = 0.2; d4bg_in = 0.12; dampv_in = 0.2
  do
    read(u_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (line(1:1) /= '#') exit
    if (index(line, '# res ') == 1) read(line(7:), *) res
    if (index(line, '# ng ') == 1) read(line(6:), *) nhalo
    if (index(line, '# dt ') == 1) read(line(6:), *) dt
    if (index(line, '# dddmp ') == 1) read(line(9:), *) dddmp_in
    if (index(line, '# d4_bg ') == 1) read(line(9:), *) d4bg_in
    if (index(line, '# damp_v ') == 1) read(line(10:), *) dampv_in
  end do
  close(u_in)
  if (res <= 0 .or. nhalo <= 0 .or. dt <= 0.) stop 'bad header'

  bd%is = 1;  bd%ie = res
  bd%js = 1;  bd%je = res
  bd%ng = nhalo
  bd%isd = 1 - nhalo; bd%ied = res + nhalo
  bd%jsd = 1 - nhalo; bd%jed = res + nhalo
  call set_fill_corner_bounds(bd)

  fl%grid_type = 0
  fl%npx = res + 1
  fl%npy = res + 1
  fl%do_diss_est = .false.
  fl%do_f3d = .false.
  fl%lim_fac = 1.0

  call alloc_all()

  ! ---- pass 2: records ----
  open(newunit=u_in, file='dswcore_input.txt', status='old', action='read')
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
    case ('GRID_LON', 'GRID_LAT', 'AGRID_LON', 'AGRID_LAT')
      read(line, *) name, i, j, val
      if (trim(name) == 'GRID_LON')  gs%grid(i, j, 1) = val
      if (trim(name) == 'GRID_LAT')  gs%grid(i, j, 2) = val
      if (trim(name) == 'AGRID_LON') gs%agrid(i, j, 1) = val
      if (trim(name) == 'AGRID_LAT') gs%agrid(i, j, 2) = val
    case ('EDGE_S', 'EDGE_N', 'EDGE_W', 'EDGE_E')
      read(line, *) name, i, val
      if (trim(name) == 'EDGE_S') gs%edge_s(i) = val
      if (trim(name) == 'EDGE_N') gs%edge_n(i) = val
      if (trim(name) == 'EDGE_W') gs%edge_w(i) = val
      if (trim(name) == 'EDGE_E') gs%edge_e(i) = val
    case ('DA_MIN')
      read(line, *) name, val
      gs%da_min = val
    case ('DA_MIN_C')
      read(line, *) name, val
      gs%da_min_c = val
    case default
      read(line, *) name, i, j, val
      select case (trim(name))
      case ('RAREA');   gs%rarea(i, j) = val
      case ('AREA');    gs%area(i, j) = val
      case ('AREA_C');  gs%area_c(i, j) = val
      case ('RAREA_C'); gs%rarea_c(i, j) = val
      case ('DXA');     gs%dxa(i, j) = val
      case ('DYA');     gs%dya(i, j) = val
      case ('RDXA');    gs%rdxa(i, j) = val
      case ('RDYA');    gs%rdya(i, j) = val
      case ('COSA_S');  gs%cosa_s(i, j) = val
      case ('RSIN2');   gs%rsin2(i, j) = val
      case ('DX');      gs%dx(i, j) = val
      case ('DY');      gs%dy(i, j) = val
      case ('RDX');     gs%rdx(i, j) = val
      case ('RDY');     gs%rdy(i, j) = val
      case ('DXC');     gs%dxc(i, j) = val
      case ('DYC');     gs%dyc(i, j) = val
      case ('RDXC');    gs%rdxc(i, j) = val
      case ('RDYC');    gs%rdyc(i, j) = val
      case ('COSA_U');  gs%cosa_u(i, j) = val
      case ('SINA_U');  gs%sina_u(i, j) = val
      case ('RSIN_U');  gs%rsin_u(i, j) = val
      case ('COSA_V');  gs%cosa_v(i, j) = val
      case ('SINA_V');  gs%sina_v(i, j) = val
      case ('RSIN_V');  gs%rsin_v(i, j) = val
      case ('COSA');    gs%cosa(i, j) = val
      case ('SINA');    gs%sina(i, j) = val
      case ('RSINA');   gs%rsina(i, j) = val
      case ('FC');      gs%fC(i, j) = val
      case ('F0');      gs%f0(i, j) = val
      case ('DIVG_U');  gs%divg_u(i, j) = val
      case ('DIVG_V');  gs%divg_v(i, j) = val
      case ('DEL6_U');  gs%del6_u(i, j) = val
      case ('DEL6_V');  gs%del6_v(i, j) = val
      case ('DELP');    delp(i, j) = val
      case ('PT');      pt(i, j) = val
      case ('W');       w(i, j) = val
      case ('U');       u(i, j) = val
      case ('V');       v(i, j) = val
      case ('UC');      uc(i, j) = val
      case ('VC');      vc(i, j) = val
      case ('UA');      ua(i, j) = val
      case ('VA');      va(i, j) = val
      case ('DIVGD_IN'); divg_d(i, j) = val
      case default
        stop 'unknown record name'
      end select
    end select
  end do
  close(u_in)

  z_rat = 1.0
  q = 0.; q_con = 0.
  xflux = 0.; yflux = 0.; cx = 0.; cy = 0.

  fl%duogrid = .true.
  gs%dg%is_initialized = .true.
  ut = 1.e30
  vt = 1.e30
  call d_sw1(delp, pt, w, uc, vc, xflux, yflux, cx, cy,               &
             crx_adv, cry_adv, xfx_adv, yfx_adv, q_con,               &
             1, q, 1, 1, .false., dt,                                 &
             8, 6, 6, 6, 1, 0, dampv_in, 0.0, .true.,                 &
             gs, fl, bd, allflux_x, allflux_y, ra_x, ra_y, ut, vt)

  ubbtemp = 1.e30
  vbbtemp = 1.e30
  ubb = 1.e30
  vbb = 1.e30
  call d_sw3(u, v, uc, vc, dt, 6, gs, fl, bd, ut, vt,                 &
             ubbtemp, vbbtemp, ubb, vbb)

  call dump_all()
  write(*, *) 'fv3_dswcore_oracle: d_sw1 -> d_sw3 chain dumped'

contains

  subroutine alloc_all()
    integer :: isd, ied, jsd, jed
    isd = bd%isd; ied = bd%ied; jsd = bd%jsd; jed = bd%jed
    allocate (gs%rarea(isd:ied, jsd:jed), gs%area(isd:ied, jsd:jed))
    allocate (gs%dxa(isd:ied, jsd:jed), gs%dya(isd:ied, jsd:jed))
    allocate (gs%rdxa(isd:ied, jsd:jed), gs%rdya(isd:ied, jsd:jed))
    allocate (gs%cosa_s(isd:ied, jsd:jed), gs%rsin2(isd:ied, jsd:jed))
    allocate (gs%f0(isd:ied, jsd:jed))
    allocate (gs%dx(isd:ied, jsd:jed + 1), gs%rdx(isd:ied, jsd:jed + 1))
    allocate (gs%dy(isd:ied + 1, jsd:jed), gs%rdy(isd:ied + 1, jsd:jed))
    allocate (gs%dxc(isd:ied + 1, jsd:jed), gs%rdxc(isd:ied + 1, jsd:jed))
    allocate (gs%dyc(isd:ied, jsd:jed + 1), gs%rdyc(isd:ied, jsd:jed + 1))
    allocate (gs%cosa_u(isd:ied + 1, jsd:jed))
    allocate (gs%sina_u(isd:ied + 1, jsd:jed))
    allocate (gs%rsin_u(isd:ied + 1, jsd:jed))
    allocate (gs%cosa_v(isd:ied, jsd:jed + 1))
    allocate (gs%sina_v(isd:ied, jsd:jed + 1))
    allocate (gs%rsin_v(isd:ied, jsd:jed + 1))
    allocate (gs%divg_u(isd:ied, jsd:jed + 1))
    allocate (gs%del6_u(isd:ied, jsd:jed + 1))
    allocate (gs%divg_v(isd:ied + 1, jsd:jed))
    allocate (gs%del6_v(isd:ied + 1, jsd:jed))
    allocate (gs%rarea_c(isd:ied + 1, jsd:jed + 1))
    allocate (gs%area_c(isd:ied + 1, jsd:jed + 1))
    allocate (gs%fC(isd:ied + 1, jsd:jed + 1))
    allocate (gs%cosa(isd:ied + 1, jsd:jed + 1))
    allocate (gs%sina(isd:ied + 1, jsd:jed + 1))
    allocate (gs%rsina(isd:ied + 1, jsd:jed + 1))
    allocate (gs%grid(isd:ied + 1, jsd:jed + 1, 2))
    allocate (gs%agrid(isd:ied, jsd:jed, 2))
    allocate (gs%sin_sg(isd:ied, jsd:jed, 9))
    allocate (gs%cos_sg(isd:ied, jsd:jed, 9))
    allocate (gs%edge_s(fl%npx), gs%edge_n(fl%npx))
    allocate (gs%edge_w(fl%npy), gs%edge_e(fl%npy))
    allocate (gs%a11(isd:ied, jsd:jed), gs%a12(isd:ied, jsd:jed))
    allocate (gs%a21(isd:ied, jsd:jed), gs%a22(isd:ied, jsd:jed))
    allocate (gs%w00(isd:ied, jsd:jed))
    gs%a11 = 0.; gs%a12 = 0.; gs%a21 = 0.; gs%a22 = 0.; gs%w00 = 0.

    allocate (delp(isd:ied, jsd:jed), pt(isd:ied, jsd:jed))
    allocate (w(isd:ied, jsd:jed))
    allocate (ua(isd:ied, jsd:jed), va(isd:ied, jsd:jed))
    allocate (u(isd:ied, jsd:jed + 1), vc(isd:ied, jsd:jed + 1))
    allocate (v(isd:ied + 1, jsd:jed), uc(isd:ied + 1, jsd:jed))
    allocate (delpc(isd:ied, jsd:jed), ptc(isd:ied, jsd:jed))
    allocate (divg_d(isd:ied + 1, jsd:jed + 1))
    allocate (xflux(bd%is:bd%ie + 1, bd%js:bd%je))
    allocate (yflux(bd%is:bd%ie, bd%js:bd%je + 1))
    allocate (cx(bd%is:bd%ie + 1, jsd:jed), cy(isd:ied, bd%js:bd%je + 1))
    allocate (crx_adv(bd%is:bd%ie + 1, jsd:jed))
    allocate (xfx_adv(bd%is:bd%ie + 1, jsd:jed))
    allocate (cry_adv(isd:ied, bd%js:bd%je + 1))
    allocate (yfx_adv(isd:ied, bd%js:bd%je + 1))
    allocate (allflux_x(bd%is:bd%ie + 1, bd%js:bd%je, 1, 5))
    allocate (allflux_y(bd%is:bd%ie, bd%js:bd%je + 1, 1, 5))
    allocate (ra_x(bd%is:bd%ie, jsd:jed), ra_y(isd:ied, bd%js:bd%je))
    allocate (ut(isd:ied + 1, jsd:jed), vt(isd:ied, jsd:jed + 1))
    allflux_x = 1.e30; allflux_y = 1.e30
    ra_x = 1.e30; ra_y = 1.e30
    allocate (z_rat(isd:ied, jsd:jed))
    allocate (heat_source(bd%is:bd%ie, bd%js:bd%je))
    allocate (ubbtemp(bd%is:bd%ie + 1, bd%js:bd%je + 1))
    allocate (vbbtemp(bd%is:bd%ie + 1, bd%js:bd%je + 1))
    allocate (ubb(bd%is:bd%ie + 1, bd%js:bd%je + 1))
    allocate (vbb(bd%is:bd%ie + 1, bd%js:bd%je + 1))
    allocate (diss_est(bd%is:bd%ie, bd%js:bd%je))
    allocate (q(isd:ied, jsd:jed, 1, 1), q_con(isd:ied, jsd:jed))
    ! INOUT arrays fully poisoned; the exporter overwrites every slot
    delp = -9.e9; pt = -9.e9; w = 0.; u = -9.e9; v = -9.e9
    uc = -9.e9; vc = -9.e9; ua = -9.e9; va = -9.e9
    divg_d = -9.e9
  end subroutine alloc_all

  subroutine dump_all()
    ! d_sw3 chain outputs: the four B-grid compute-ring arrays
    ! (is:ie+1, js:je+1), fully written by the duo stage.
    integer :: u_out
    open(newunit=u_out, file='dsw3_output.txt', status='replace', &
         action='write')
    do j = bd%js, bd%je + 1
      do i = bd%is, bd%ie + 1
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UBBT', i, j, ubbtemp(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VBBT', i, j, vbbtemp(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UBB', i, j, ubb(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VBB', i, j, vbb(i, j)
      end do
    end do
    close(u_out)
  end subroutine dump_all

end program fv3_dsw3_duo_driver
