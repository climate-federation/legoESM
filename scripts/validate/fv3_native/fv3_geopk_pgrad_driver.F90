! Multilevel hydrostatic pressure-chain oracle driver (brick geopk_pgrad).
!
! Chain, ONE acoustic substep, ONE face, DRY HYDROSTATIC DUO:
!   geopk(CG=.true. , computehalo=.false.)  dyn_core.F90:2660-2790 @ :533
!     -> p_grad_c                           dyn_core.F90:2073-2132 @ :629
!     -> geopk(CG=.false., computehalo=.true.)                     @ :1401
!       -> one_grad_p                       dyn_core.F90:2347-2480 @ :1531
!
! SCOPE: ROUTINE-TRANSLATION CERTIFICATE on ONE face.  Every inter-stage
! and cross-face quantity (delpc/ptc/uc/vc from the certified c_sw, the
! D-grid delp/pt/u/v, hs, divg2) arrives as EXPLICIT serialized input;
! re-deriving an upstream stage inside Fortran can flip last-bit
! branches.  This certifies NEITHER the six-face exchange cadence NOR
! the ext_scalar schedule — those stay with the existing system gates.
!
! Build lane: NO -DSW_DYNAMICS, NO -DUSE_COND (first brick outside the
! SW lane), hydrostatic=.true., beta<=0, a2b_ord=4, duogrid, d_ext>0.
!
! Sentinel contract:
!   1.e30  = never-written OUTPUT region.  Dumped and compared, so the
!            write window is itself certified (geopk's pk/gz/pe/peln/pkz
!            are intent(OUT) upstream and only partially written; the
!            extract's intent shims make the round-trip defined).
!   -9.e9  = INPUT slot the exporter is expected to overwrite, or a
!            DELIBERATE poison for a dead-branch input (see the two
!            p_grad_c/one_grad_p re-runs below).
!
! Two DELIBERATE poison re-runs prove the hydrostatic branch never reads
! the non-hydrostatic-only inputs: p_grad_c is re-run with delpc=-9.e9
! and one_grad_p with delp=-9.e9, each from restored pristine state.
! The *_DPPOISON dumps must be BITWISE equal to the clean ones.
!
! gs%rdxc is allocated at the DUMMY-declared shape (isd:ied+1,jsd:jed+1)
! from dyn_core:2084 while only (isd:ied+1, jsd:jed) is serialized: row
! j=jed+1 stays -9.e9 and is never read (the uc loop runs j=js..je), a
! LIVE proof of the read window.  See UNCERTAIN U3 — fv_arrays.F90 is
! absent from this tree so the production allocation is unverified.
!
! The LOGEXP transcendental-attribution probe (spec 5.4) is evaluated
! FIRST, before any chain stage, so a gfortran-vs-libm last-bit
! disagreement on log/exp is ATTRIBUTED to the environment rather than
! mistaken for a port defect.
program fv3_geopk_pgrad_driver
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, &
                             fv_flags_type, set_fill_corner_bounds
  use geopk_shim_mod, only: set_dyncore_scalars
  use geopk_pgrad_extract_mod, only: geopk, p_grad_c, one_grad_p
  implicit none

  integer, parameter :: NPROBE = 64
  real, parameter :: SENTINEL = 1.e30
  real, parameter :: POISON = -9.e9

  type(fv_grid_bounds_type) :: bd
  type(fv_grid_type) :: gs
  type(fv_flags_type) :: fl

  integer :: res, nhalo, km, npx, npy
  real :: dt, dt2, ptop, akap, cpair, dext
  integer :: ios, i, j, k, u_in, u_out
  character(len=32) :: name
  character(len=256) :: line
  character(len=64) :: outfile
  real :: val

  real, allocatable :: delpc(:, :, :), ptc(:, :, :)
  real, allocatable :: delp(:, :, :), pt(:, :, :), q_con(:, :, :)
  real, allocatable :: uc(:, :, :), vc(:, :, :), u(:, :, :), v(:, :, :)
  real, allocatable :: hs(:, :), divg2(:, :)
  real, allocatable :: pkc(:, :, :), gz(:, :, :)
  real, allocatable :: pe(:, :, :), peln(:, :, :), pkz(:, :, :)
  ! pristine copies for the poison re-runs
  real, allocatable :: uc0(:, :, :), vc0(:, :, :), u0(:, :, :), v0(:, :, :)
  real, allocatable :: delpc0(:, :, :), delp0(:, :, :)
  real, allocatable :: pkc0(:, :, :), gz0(:, :, :)
  real, allocatable :: logexp_probe(:), logexp_out(:)

  ! ---- pass 1: header ----
  open(newunit=u_in, file='geopk_pgrad_input.txt', status='old', &
       action='read')
  res = -1; nhalo = -1; km = -1
  ! every scalar has a hard-coded default and is overwritten by the
  ! header if present; res/ng/km are MANDATORY.  ptop/akap/cpair are
  ! header-settable so the oracle is INSENSITIVE to the (unverifiable,
  ! see UNCERTAIN U1) production FMS constants.
  dt = 225.0; dt2 = 112.5
  ptop = 100.0; akap = 2./7.; cpair = 1004.64; dext = 0.02
  do
    read(u_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (line(1:1) /= '#') exit
    if (index(line, '# res ') == 1) read(line(7:), *) res
    if (index(line, '# ng ') == 1) read(line(6:), *) nhalo
    if (index(line, '# km ') == 1) read(line(6:), *) km
    if (index(line, '# dt ') == 1) read(line(6:), *) dt
    if (index(line, '# dt2 ') == 1) read(line(7:), *) dt2
    if (index(line, '# ptop ') == 1) read(line(8:), *) ptop
    if (index(line, '# akap ') == 1) read(line(8:), *) akap
    if (index(line, '# cpair ') == 1) read(line(9:), *) cpair
    if (index(line, '# dext ') == 1) read(line(8:), *) dext
  end do
  close(u_in)
  if (res <= 0 .or. nhalo <= 0 .or. km <= 0) stop 'bad header'
  if (ptop <= 0. .or. akap <= 0. .or. cpair <= 0.) stop 'bad header scalars'

  bd%is = 1;  bd%ie = res
  bd%js = 1;  bd%je = res
  bd%ng = nhalo
  bd%isd = 1 - nhalo; bd%ied = res + nhalo
  bd%jsd = 1 - nhalo; bd%jed = res + nhalo
  ! fv_mp_mod module-scope globals (fv3_swcore_shim.F90:24-29).  The duo
  ! a2b_ord4 branch never calls fill_corners, but set them anyway so a
  ! stray call cannot index -huge(1).
  call set_fill_corner_bounds(bd)
  call set_dyncore_scalars(ptop, akap, cpair)

  npx = res + 1
  npy = res + 1

  call alloc_all()

  ! ---- pass 2: records ----
  open(newunit=u_in, file='geopk_pgrad_input.txt', status='old', &
       action='read')
  do
    read(u_in, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (line(1:1) == '#') cycle
    read(line, *, iostat=ios) name
    if (ios /= 0) cycle
    select case (trim(name))
    case ('DA_MIN_C')
      read(line, *) name, val
      gs%da_min_c = val
    case ('EDGE_W', 'EDGE_E', 'EDGE_S', 'EDGE_N', 'LOGEXP_PROBE')
      read(line, *) name, i, val
      select case (trim(name))
      case ('EDGE_W'); gs%edge_w(i) = val
      case ('EDGE_E'); gs%edge_e(i) = val
      case ('EDGE_S'); gs%edge_s(i) = val
      case ('EDGE_N'); gs%edge_n(i) = val
      case ('LOGEXP_PROBE'); logexp_probe(i) = val
      end select
    case ('DELPC', 'PTC', 'UC', 'VC', 'DELP', 'PT', 'U', 'V', 'Q_CON')
      read(line, *) name, i, j, k, val
      select case (trim(name))
      case ('DELPC'); delpc(i, j, k) = val
      case ('PTC');   ptc(i, j, k) = val
      case ('UC');    uc(i, j, k) = val
      case ('VC');    vc(i, j, k) = val
      case ('DELP');  delp(i, j, k) = val
      case ('PT');    pt(i, j, k) = val
      case ('U');     u(i, j, k) = val
      case ('V');     v(i, j, k) = val
      case ('Q_CON'); q_con(i, j, k) = val
      end select
    case ('RDXC', 'RDYC', 'RDX', 'RDY', 'DXA', 'DYA', 'GRID_LON', &
          'GRID_LAT', 'AGRID_LON', 'AGRID_LAT', 'HS', 'DIVG2')
      read(line, *) name, i, j, val
      select case (trim(name))
      case ('RDXC');      gs%rdxc(i, j) = val
      case ('RDYC');      gs%rdyc(i, j) = val
      case ('RDX');       gs%rdx(i, j) = val
      case ('RDY');       gs%rdy(i, j) = val
      case ('DXA');       gs%dxa(i, j) = val
      case ('DYA');       gs%dya(i, j) = val
      case ('GRID_LON');  gs%grid(i, j, 1) = val
      case ('GRID_LAT');  gs%grid(i, j, 2) = val
      case ('AGRID_LON'); gs%agrid(i, j, 1) = val
      case ('AGRID_LAT'); gs%agrid(i, j, 2) = val
      case ('HS');        hs(i, j) = val
      case ('DIVG2');     divg2(i, j) = val
      end select
    case default
      stop 'unknown record'
    end select
  end do
  close(u_in)

  write(outfile, '(A,I0,A)') 'geopk_pgrad_output_km', km, '.txt'
  open(newunit=u_out, file=trim(outfile), status='replace', action='write')

  ! ---- step 0: transcendental attribution probe (spec 5.4) ----
  ! Runs FIRST: if gfortran's libm disagrees with NumPy on this host,
  ! the failure is an ENVIRONMENT result, not a port defect.
  do i = 1, NPROBE
    logexp_out(i) = exp(akap*log(logexp_probe(i)))
  end do
  call dump1('LOGEXP_OUT', logexp_out, 1)

  ! ---- flag selection (explicit; fv3_d2a2c_driver.F90:88-90 precedent) ----
  gs%grid_type = 0
  gs%bounded_domain = .false.
  gs%sw_corner = .true.; gs%se_corner = .true.
  gs%ne_corner = .true.; gs%nw_corner = .true.
  ! TWO DIFFERENT duo flags: dyn_core:534 feeds geopk's `duogrid` dummy
  ! from gridstruct%dg%is_initialized at the C site, dyn_core:1402 feeds
  ! it from flagstruct%duogrid at the D site.  Both .true. here, but the
  ! driver READS THEM FROM THE TWO DIFFERENT MEMBERS so a config where
  ! they disagree stays representable (UNCERTAIN U5).  Do NOT unify.
  gs%dg%is_initialized = .true.
  fl%duogrid = .true.
  fl%grid_type = 0
  fl%npx = npx
  fl%npy = npy

  ! pristine copies for the poison re-runs
  uc0 = uc; vc0 = vc; u0 = u; v0 = v
  delpc0 = delpc; delp0 = delp

  ! ---- step 1: geopk, C-grid site (dyn_core.F90:533) ----
  pkc = SENTINEL; gz = SENTINEL
  pe = SENTINEL; peln = SENTINEL; pkz = SENTINEL
  call geopk(ptop, pe, peln, delpc, pkc, gz, hs, ptc, q_con, pkz, km, &
             akap, .true., gs%bounded_domain, gs%dg%is_initialized, &
             .false., npx, npy, 4, bd)
  call dump3('PKC_C', pkc, bd%isd, bd%jsd, 1)
  call dump3('GZ_C', gz, bd%isd, bd%jsd, 1)
  call dump3('PE_C', pe, bd%is - 1, 1, bd%js - 1)
  call dump3('PELN_C', peln, bd%is, 1, bd%js)
  call dump3('PKZ_C', pkz, bd%is, bd%js, 1)

  ! ---- step 2: p_grad_c (dyn_core.F90:629) ----
  call p_grad_c(dt2, km, delpc, pkc, gz, uc, vc, bd, gs%rdxc, gs%rdyc, &
                .true.)
  call dump3('UC_PGC', uc, bd%isd, bd%jsd, 1)
  call dump3('VC_PGC', vc, bd%isd, bd%jsd, 1)

  ! ---- step 2b: delpc-poison control (spec 5.2) ----
  uc = uc0; vc = vc0
  delpc = POISON
  call p_grad_c(dt2, km, delpc, pkc, gz, uc, vc, bd, gs%rdxc, gs%rdyc, &
                .true.)
  call dump3('UC_PGC_DPPOISON', uc, bd%isd, bd%jsd, 1)
  call dump3('VC_PGC_DPPOISON', vc, bd%isd, bd%jsd, 1)
  delpc = delpc0

  ! ---- step 3: geopk, D-grid site (dyn_core.F90:1401) ----
  pkc = SENTINEL; gz = SENTINEL
  pe = SENTINEL; peln = SENTINEL; pkz = SENTINEL
  call geopk(ptop, pe, peln, delp, pkc, gz, hs, pt, q_con, pkz, km, &
             akap, .false., gs%bounded_domain, fl%duogrid, &
             .true., npx, npy, 4, bd)
  call dump3('PKC_D', pkc, bd%isd, bd%jsd, 1)
  call dump3('GZ_D', gz, bd%isd, bd%jsd, 1)
  call dump3('PE_D', pe, bd%is - 1, 1, bd%js - 1)
  call dump3('PELN_D', peln, bd%is, 1, bd%js)
  call dump3('PKZ_D', pkz, bd%is, bd%js, 1)

  ! ---- step 4: one_grad_p (dyn_core.F90:1531) ----
  ! one_grad_p MUTATES pk and gz (a2b_ord4 replace=.true. at :2399/:2409)
  pkc0 = pkc; gz0 = gz
  call one_grad_p(u, v, pkc, gz, divg2, delp, dt, bd%ng, gs, bd, npx, &
                  npy, km, ptop, .true., 4, dext)
  call dump3('U_OGP', u, bd%isd, bd%jsd, 1)
  call dump3('V_OGP', v, bd%isd, bd%jsd, 1)
  call dump3('PK_OGP', pkc, bd%isd, bd%jsd, 1)
  call dump3('GZ_OGP', gz, bd%isd, bd%jsd, 1)

  ! ---- step 4b: delp-poison control (spec 5.2) ----
  u = u0; v = v0
  pkc = pkc0; gz = gz0
  delp = POISON
  call one_grad_p(u, v, pkc, gz, divg2, delp, dt, bd%ng, gs, bd, npx, &
                  npy, km, ptop, .true., 4, dext)
  call dump3('U_OGP_DPPOISON', u, bd%isd, bd%jsd, 1)
  call dump3('V_OGP_DPPOISON', v, bd%isd, bd%jsd, 1)
  delp = delp0

  close(u_out)
  write(*, *) 'fv3_geopk_pgrad_driver: chain dumped, km =', km

contains

  subroutine alloc_all()
    integer :: isd, ied, jsd, jed
    isd = bd%isd; ied = bd%ied; jsd = bd%jsd; jed = bd%jed

    ! gridstruct members at their DUMMY-declared shapes (see the header
    ! note on rdxc; dyn_core:2084/2085 and :2466/:2473)
    allocate (gs%rdxc(isd:ied + 1, jsd:jed + 1))
    allocate (gs%rdyc(isd:ied, jsd:jed))
    allocate (gs%rdx(isd:ied, jsd:jed + 1))
    allocate (gs%rdy(isd:ied + 1, jsd:jed))
    allocate (gs%dxa(isd:ied, jsd:jed), gs%dya(isd:ied, jsd:jed))
    allocate (gs%grid(isd:ied + 1, jsd:jed + 1, 2))
    allocate (gs%agrid(isd:ied, jsd:jed, 2))
    allocate (gs%edge_s(npx), gs%edge_n(npx))
    allocate (gs%edge_w(npy), gs%edge_e(npy))
    gs%rdxc = POISON; gs%rdyc = POISON; gs%rdx = POISON; gs%rdy = POISON
    gs%dxa = POISON; gs%dya = POISON
    gs%grid = POISON; gs%agrid = POISON
    gs%edge_s = POISON; gs%edge_n = POISON
    gs%edge_w = POISON; gs%edge_e = POISON

    allocate (delpc(isd:ied, jsd:jed, km), ptc(isd:ied, jsd:jed, km))
    allocate (delp(isd:ied, jsd:jed, km), pt(isd:ied, jsd:jed, km))
    allocate (q_con(isd:ied, jsd:jed, km))
    allocate (hs(isd:ied, jsd:jed))
    allocate (uc(isd:ied + 1, jsd:jed, km))
    allocate (vc(isd:ied, jsd:jed + 1, km))
    allocate (u(isd:ied, jsd:jed + 1, km))
    allocate (v(isd:ied + 1, jsd:jed, km))
    allocate (divg2(bd%is:bd%ie + 1, bd%js:bd%je + 1))
    delpc = POISON; ptc = POISON; delp = POISON; pt = POISON
    q_con = POISON; hs = POISON
    uc = POISON; vc = POISON; u = POISON; v = POISON
    divg2 = POISON

    allocate (pkc(isd:ied, jsd:jed, km + 1))
    allocate (gz(isd:ied, jsd:jed, km + 1))
    allocate (pe(bd%is - 1:bd%ie + 1, km + 1, bd%js - 1:bd%je + 1))
    allocate (peln(bd%is:bd%ie, km + 1, bd%js:bd%je))
    allocate (pkz(bd%is:bd%ie, bd%js:bd%je, km))
    pkc = SENTINEL; gz = SENTINEL
    pe = SENTINEL; peln = SENTINEL; pkz = SENTINEL

    ! explicit bounds (not mold=) so the pristine copies are bound-for-
    ! bound identical on any gfortran
    allocate (uc0(isd:ied + 1, jsd:jed, km))
    allocate (vc0(isd:ied, jsd:jed + 1, km))
    allocate (u0(isd:ied, jsd:jed + 1, km))
    allocate (v0(isd:ied + 1, jsd:jed, km))
    allocate (delpc0(isd:ied, jsd:jed, km))
    allocate (delp0(isd:ied, jsd:jed, km))
    allocate (pkc0(isd:ied, jsd:jed, km + 1))
    allocate (gz0(isd:ied, jsd:jed, km + 1))
    uc0 = POISON; vc0 = POISON; u0 = POISON; v0 = POISON
    delpc0 = POISON; delp0 = POISON
    pkc0 = SENTINEL; gz0 = SENTINEL

    allocate (logexp_probe(NPROBE), logexp_out(NPROBE))
    logexp_probe = POISON; logexp_out = SENTINEL
  end subroutine alloc_all

  ! Family dump format (fv3_dsw5_duo_driver.F90:305-330), extended with
  ! the k index.  The three written indices are the array's OWN axis
  ! order: for pe/peln that is (i, k, j) — the Fortran order is
  ! LOAD-BEARING for the compare and is preserved end to end.
  subroutine dump3(nm, a, i0, j0, k0)
    character(len=*), intent(in) :: nm
    real, intent(in) :: a(:, :, :)
    integer, intent(in) :: i0, j0, k0
    integer :: ii, jj, kk
    do kk = 1, size(a, 3)
      do jj = 1, size(a, 2)
        do ii = 1, size(a, 1)
          write(u_out, '(A,1X,I5,1X,I5,1X,I5,1X,ES26.17E3)') nm, &
            ii + i0 - 1, jj + j0 - 1, kk + k0 - 1, a(ii, jj, kk)
        end do
      end do
    end do
  end subroutine dump3

  subroutine dump1(nm, a, i0)
    character(len=*), intent(in) :: nm
    real, intent(in) :: a(:)
    integer, intent(in) :: i0
    integer :: ii
    do ii = 1, size(a)
      write(u_out, '(A,1X,I5,1X,ES26.17E3)') nm, ii + i0 - 1, a(ii)
    end do
  end subroutine dump1

end program fv3_geopk_pgrad_driver
