! Phase-4b d_sw one-step oracle driver.
!
! Reads the gridstruct + state inputs exported by export_dswcore_inputs.py
! (text records, Fortran-index keyed; the c_sw-updated uc/vc/ua/va arrive
! as EXPLICIT inputs so both oracle sides consume byte-identical values —
! re-deriving them through c_sw could flip PPM sign branches on last-bit
! differences), runs the VERBATIM d_sw extraction ONCE on a single tile,
! and dumps the outputs over their source-defined regions.
!
! CONVENTION: d_sw's final wind update is the vector-invariant
! CIRCULATION form (u <- u*dx + d(ke) + vorticity flux; sw_core final
! loops) — the rdx/rdy division happens downstream inside the D-grid
! pressure-gradient routines (dyn_core one_grad_p/nh_p_grad fold *rdx
! into the PG update).  Post-d_sw u/v magnitudes ~ u*dx (1e7-scale) are
! CORRECT, not blown.
!
! Configuration (production-representative; a2b divergence-damping ON
! via dddmp=0.2/nord=1, dissipation-estimate heating OFF):
!   hord_mt=hord_vt=hord_tm=hord_dp=6, hord_tr=8, nord=1, nord_v=1,
!   dddmp=0.2, d2_bg=0., d4_bg=0.12, damp_v=0.2, d_con=0., zvir=0.,
!   inline_q=F, nq=1 (dummy tracer, untouched), km=k=1, kgb=0.,
!   z_rat=1., hydrostatic=T, damp_w=nord_w=damp_t=nord_t=0.
!
! Build (gen_dswcore_oracle.sh): gfortran -O2 -fdefault-real-8
!   -fdefault-double-8 -cpp -ffree-line-length-none
!   shim + sw_core extract + dsw extract + driver  (fresh -J dir!)
program fv3_dswcore_oracle_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, &
                             fv_flags_type, set_fill_corner_bounds
  use dsw_extract_mod, only: d_sw
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

  call d_sw(delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, divg_d,     &
            xflux, yflux, cx, cy, crx_adv, cry_adv, xfx_adv, yfx_adv,  &
            q_con, z_rat, 0.0, heat_source, diss_est, 0.0, 1, 1, q,    &
            1, 1, .false., dt, 8, 6, 6, 6, 6, 1, 1, 0, 0,              &
            dddmp_in, 0.0, d4bg_in, dampv_in, 0.0, 0.0, 0.0, .true.,   &
            gs, fl, .false., bd)

  call dump_all()
  write(*, *) 'fv3_dswcore_oracle: one d_sw step dumped'

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
    allocate (z_rat(isd:ied, jsd:jed))
    allocate (heat_source(bd%is:bd%ie, bd%js:bd%je))
    allocate (diss_est(bd%is:bd%ie, bd%js:bd%je))
    allocate (q(isd:ied, jsd:jed, 1, 1), q_con(isd:ied, jsd:jed))
    ! INOUT arrays fully poisoned; the exporter overwrites every slot
    delp = -9.e9; pt = -9.e9; w = 0.; u = -9.e9; v = -9.e9
    uc = -9.e9; vc = -9.e9; ua = -9.e9; va = -9.e9
    divg_d = -9.e9
  end subroutine alloc_all

  subroutine dump_all()
    ! d_sw source-defined output regions (sw_core.F90 d_sw write loops;
    ! INTENT(OUT) / INOUT args dumped only where the source DEFINES them):
    !   delpc       : the higher-order damping block writes 1..res+1 in
    !                 both directions (sw_core.F90:1376, delpc=divg_d over
    !                 is..ie+1, js..je+1) -> (res+1)^2 = 169, not res^2.
    !   ptc         : NOT dumped — INTENT(OUT) scratch written only by the
    !                 nord==0 circulation branch (skipped at nord=1) and
    !                 the inline_q arm; genuinely undefined on this path.
    !   u/v (INOUT) : D-wind update loops  u i 1..res, j 1..res+1;
    !                                      v i 1..res+1, j 1..res
    !   uc/vc/ua/va : INOUT, updated on compute + consumed rings; dump
    !                 the full data domain (fully defined INPUTS).
    !   divg_d      : INOUT (real c_sw divergence in; damping updates the
    !                 B ring 1..res+1).
    !   crx/xfx_adv : i 1..res+1, j jsd..jed;  cry/yfx transposed.
    !   cx/xflux    : i 1..res+1 (cx j jsd..jed; xflux j 1..res); cy/yflux
    !                 transposed.
    !   heat_source/diss_est: production #ifndef SW_DYNAMICS zeroes them on
    !                 the compute domain is..ie, js..je (d_con=0/do_diss_est
    !                 =F leave them at 0) — a real defined output.
    integer :: isd, ied, jsd, jed
    isd = bd%isd; ied = bd%ied; jsd = bd%jsd; jed = bd%jed
    open(newunit=u_out, file='dswcore_output.txt', status='replace', &
         action='write')
    write(u_out, '(A,I0)') '# res ', res
    do j = 1, res + 1
      do i = 1, res + 1
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DELPC', i, j, delpc(i, j)
      end do
    end do
    do j = bd%js, bd%je
      do i = bd%is, bd%ie
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'HEAT', i, j, heat_source(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DISS', i, j, diss_est(i, j)
        ! delp/pt ARE updated on the production (inline_q=F) else arm
        ! (sw_core.F90:1050-1065), over the compute cells is..ie, js..je
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DELPO', i, j, delp(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'PTO', i, j, pt(i, j)
      end do
    end do
    do j = jsd, jed + 1
      do i = isd, ied
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UOUT', i, j, u(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VCOUT', i, j, vc(i, j)
      end do
    end do
    do j = jsd, jed
      do i = isd, ied + 1
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VOUT', i, j, v(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UCOUT', i, j, uc(i, j)
      end do
    end do
    do j = jsd, jed
      do i = isd, ied
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'UAOUT', i, j, ua(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'VAOUT', i, j, va(i, j)
      end do
    end do
    do j = jsd, jed + 1
      do i = isd, ied + 1
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'DIVGD', i, j, divg_d(i, j)
      end do
    end do
    do j = jsd, jed
      do i = 1, res + 1
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'CRX', i, j, crx_adv(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'XFX', i, j, xfx_adv(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'CX', i, j, cx(i, j)
      end do
    end do
    do j = 1, res + 1
      do i = isd, ied
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'CRY', i, j, cry_adv(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'YFX', i, j, yfx_adv(i, j)
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'CY', i, j, cy(i, j)
      end do
    end do
    do j = 1, res
      do i = 1, res + 1
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'XFLUX', i, j, xflux(i, j)
      end do
    end do
    do j = 1, res + 1
      do i = 1, res
        write(u_out, '(A,1X,I5,1X,I5,1X,ES26.17E3)') 'YFLUX', i, j, yflux(i, j)
      end do
    end do
    close(u_out)
  end subroutine dump_all

end program fv3_dswcore_oracle_driver
