! Composed duo halo-exchange oracle driver (ext chain certificate).
!
! Runs the REAL, pinned-tree FV3 duo initialization (fv_control_init ->
! duogrid_init -> init_grid -> grid_utils_init, exactly the runtime path
! of the C48 parity deck) on 6 MPI ranks (layout 1,1 per tile -- the
! one-rank-per-face configuration the port models), builds analytic input
! fields on the compute domain, and then
!
!   1. calls the REAL linked duogrid_mod::ext_scalar (A and B stagger)
!      and duogrid_mod::ext_vector (D and C stagger) -- the COMPOSED
!      authority, zero copies, zero shims;
!   2. calls the staged verbatim copies (fv3_extchain_extract.F90) on an
!      identical input, dumping every intermediate stage
!      (post-mpp / post-cube_rmp rings / geographic lattice / projection);
!   3. records max|staged - composed| per family (must be exactly 0.0
!      for the stage dumps to be trusted);
!   4. repeats everything under TWO different halo-sentinel prefills --
!      any output slot that differs between the two variants depends on
!      an UNDEFINED (never-written) halo location in the oracle itself.
!
! Input fields (per vertical level k = 1..5; levels are independent in
! the ext machinery, so one call tests five fields):
!   scalars: k=1 constant, k=2 lat (linear-in-lat), k=3 sin(lon)cos(lat),
!            k=4 J&W-like T (240 + 30 cos 2lat + 10 sin lon cos lat),
!            k=5 Gaussian bump centred ON A CUBE VERTEX (tile-1 NE corner).
!   vectors: covariant winds from a streamfunction psi (test_cases.F90
!            defOnGrid pattern: u = -dpsi/dyc etc.), psi per level:
!            solid-body a=0, solid-body a=45deg, jet sin^3(lat),
!            vertex-centred vortex, multi-harmonic smooth.
!
! Outputs (per rank): extchain_t<tile>.dat (raw stream f64) +
! extchain_t<tile>.mf (manifest: name ni nj nk byte-offset + NOTE lines
! with every resolved flag and the certification values).
!
! Build: see scripts/cluster/fv3_native/extchain_oracle.sbatch (compiles
! this file + fv3_extchain_extract.F90 against the pinned build_hydro
! objects and the real libFMS_r8; mpirun -np 6).
program fv3_extchain_oracle_driver
  use fms_mod, only: fms_init, fms_end
  use mpp_mod, only: mpp_pe, mpp_npes, mpp_error, FATAL, mpp_sum, &
                     mpp_sync
  use mpp_domains_mod, only: mpp_get_tile_id
  use fv_arrays_mod, only: fv_atmos_type, fv_grid_bounds_type, &
                           duogrid_type
  use fv_control_mod, only: fv_control_init
  use duogrid_mod, only: ext_scalar, ext_vector
  use fv_grid_utils_mod, only: great_circle_dist
  use lib_grid_mod, only: R_GRID, RADIUS
  use extchain_dump_mod, only: extchain_dump_open, extchain_dump_close, &
                               extchain_dump2, extchain_dump3, &
                               extchain_mf_note
  use extchain_extract_mod, only: ext_scalar_3d_staged, ext_vector_staged
  implicit none

  type(fv_atmos_type), allocatable, target :: Atm(:)
  logical, allocatable :: grids_on_this_pe(:)
  integer :: this_grid, p_split
  integer :: tile_id(1), tile
  integer :: is, ie, js, je, isd, ied, jsd, jed, ng, npz
  integer :: iv
  real :: sent
  character(len=8) :: vtag
  character(len=64) :: fname_dat, fname_mf
  character(len=256) :: note
  real(kind=R_GRID) :: c0(2)          ! cube-vertex lon/lat (tile-1 NE)
  ! two sentinel prefills: any output slot that changes between them
  ! reads an undefined halo location inside the oracle chain
  real, parameter :: sentinels(2) = (/0.0, 3125.0625/)

  call fms_init()
  p_split = 1
  call fv_control_init(Atm, 1920.0, this_grid, grids_on_this_pe, p_split)

  if (mpp_npes() /= 6) call mpp_error(FATAL, &
      'extchain driver requires exactly 6 PEs (one per tile)')
  if (Atm(this_grid)%gridstruct%dg%layout(1) /= 1 .or. &
      Atm(this_grid)%gridstruct%dg%layout(2) /= 1) call mpp_error(FATAL, &
      'extchain driver requires layout 1,1 (fill_corners_domain_decomp '// &
      'must be a no-op)')
  if (.not. Atm(this_grid)%gridstruct%dg%is_initialized) &
      call mpp_error(FATAL, 'duogrid not initialized -- deck must set '// &
      'duogrid=.true.')

  tile_id = mpp_get_tile_id(Atm(this_grid)%domain)
  tile = tile_id(1)

  is = Atm(this_grid)%bd%is;  ie = Atm(this_grid)%bd%ie
  js = Atm(this_grid)%bd%js;  je = Atm(this_grid)%bd%je
  isd = Atm(this_grid)%bd%isd; ied = Atm(this_grid)%bd%ied
  jsd = Atm(this_grid)%bd%jsd; jed = Atm(this_grid)%bd%jed
  ng = Atm(this_grid)%bd%ng
  npz = Atm(this_grid)%flagstruct%npz

  ! cube-vertex bump centre: NE corner of tile 1 == a true cube vertex
  ! (layout 1,1).  Broadcast by allreduce-sum: only tile 1 contributes.
  c0 = 0.0_R_GRID
  if (tile == 1) then
    c0(1) = Atm(this_grid)%gridstruct%grid(ie + 1, je + 1, 1)
    c0(2) = Atm(this_grid)%gridstruct%grid(ie + 1, je + 1, 2)
  end if
  call mpp_sum(c0, 2)

  write (fname_dat, '(A,I1,A)') 'extchain_t', tile, '.dat'
  write (fname_mf, '(A,I1,A)') 'extchain_t', tile, '.mf'
  call extchain_dump_open(fname_dat, fname_mf)

  write (note, '(A,7(1X,I6))') 'NOTE bounds is ie js je isd ied ng =', &
      is, ie, js, je, isd, ied, ng
  call extchain_mf_note(note)
  write (note, '(A,5(1X,I6))') 'NOTE npx npz tile k2e_nord dgng =', &
      Atm(this_grid)%flagstruct%npx, npz, tile, &
      Atm(this_grid)%gridstruct%dg%k2e_nord, &
      Atm(this_grid)%gridstruct%dg%bd%ng
  call extchain_mf_note(note)
  write (note, '(A,L2,1X,L2,1X,I3,3(1X,ES24.16E3))') &
      'NOTE duogrid do_schmidt grid_type stretch tlon tlat =', &
      Atm(this_grid)%flagstruct%duogrid, &
      Atm(this_grid)%flagstruct%do_schmidt, &
      Atm(this_grid)%flagstruct%grid_type, &
      Atm(this_grid)%flagstruct%stretch_fac, &
      Atm(this_grid)%flagstruct%target_lon, &
      Atm(this_grid)%flagstruct%target_lat
  call extchain_mf_note(note)
  write (note, '(A,2(1X,ES24.16E3))') 'NOTE vertex_c0 =', c0(1), c0(2)
  call extchain_mf_note(note)

  ! coordinates (for the face-map instrument control + input sha)
  call extchain_dump2('AG_LON', &
      real(Atm(this_grid)%gridstruct%agrid(isd:ied, jsd:jed, 1)))
  call extchain_dump2('AG_LAT', &
      real(Atm(this_grid)%gridstruct%agrid(isd:ied, jsd:jed, 2)))
  call extchain_dump2('GR_LON', &
      real(Atm(this_grid)%gridstruct%grid(isd:ied + 1, jsd:jed + 1, 1)))
  call extchain_dump2('GR_LAT', &
      real(Atm(this_grid)%gridstruct%grid(isd:ied + 1, jsd:jed + 1, 2)))

  ! ------------------------------------------------------------------
  ! RUNTIME GRIDSTRUCT METRIC FAMILIES over their full allocated
  ! (padded) extents -- the per-family authority for the halo-metric
  ! diagnosis.  fv_grid_tools.F90:749-835 builds grid FROM dg%b_pt on
  ! the duo lane with every mpp/fill_corners/get_symmetry step
  ! skipped, so whatever init_grid_utils derived from it here IS what
  ! dyn_core's stencils consume at halo cells.  Bounds are the
  ! fv_arrays.F90 allocations (:1311-1443).  rsina is the ONE
  ! compute-B-only array (:1346, "Why is the size different?") and is
  ! dumped at that extent; cosa AND sina are PADDED B-plane arrays
  ! (:1347/:1344, isd:ied+1 x jsd:jed+1) and are dumped padded.
  call extchain_dump2('M_DX', &
      real(Atm(this_grid)%gridstruct%dx(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_DY', &
      real(Atm(this_grid)%gridstruct%dy(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_DXA', &
      real(Atm(this_grid)%gridstruct%dxa(isd:ied, jsd:jed)))
  call extchain_dump2('M_DYA', &
      real(Atm(this_grid)%gridstruct%dya(isd:ied, jsd:jed)))
  call extchain_dump2('M_DXC', &
      real(Atm(this_grid)%gridstruct%dxc(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_DYC', &
      real(Atm(this_grid)%gridstruct%dyc(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_AREA', &
      real(Atm(this_grid)%gridstruct%area(isd:ied, jsd:jed)))
  call extchain_dump2('M_AREAC', &
      real(Atm(this_grid)%gridstruct%area_c(isd:ied + 1, jsd:jed + 1)))
  call extchain_dump2('M_COSA_U', &
      real(Atm(this_grid)%gridstruct%cosa_u(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_SINA_U', &
      real(Atm(this_grid)%gridstruct%sina_u(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_RSIN_U', &
      real(Atm(this_grid)%gridstruct%rsin_u(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_COSA_V', &
      real(Atm(this_grid)%gridstruct%cosa_v(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_SINA_V', &
      real(Atm(this_grid)%gridstruct%sina_v(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_RSIN_V', &
      real(Atm(this_grid)%gridstruct%rsin_v(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_COSA_S', &
      real(Atm(this_grid)%gridstruct%cosa_s(isd:ied, jsd:jed)))
  call extchain_dump2('M_RSIN2', &
      real(Atm(this_grid)%gridstruct%rsin2(isd:ied, jsd:jed)))
  call extchain_dump2('M_RSINA', &
      real(Atm(this_grid)%gridstruct%rsina(is:ie + 1, js:je + 1)))
  call extchain_dump2('M_COSA', &
      real(Atm(this_grid)%gridstruct%cosa(isd:ied + 1, jsd:jed + 1)))
  call extchain_dump2('M_DIVG_U', &
      real(Atm(this_grid)%gridstruct%divg_u(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_DIVG_V', &
      real(Atm(this_grid)%gridstruct%divg_v(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_DEL6_U', &
      real(Atm(this_grid)%gridstruct%del6_u(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_DEL6_V', &
      real(Atm(this_grid)%gridstruct%del6_v(isd:ied + 1, jsd:jed)))
  ! reciprocal / remaining static families (codex retro-review 2026-08-11
  ! finding 5: the dump set above was NOT "all runtime gridstruct metric
  ! families" -- these are consumed directly, e.g. rdxc/rdyc in
  ! p_grad_c, rdx/rdy in d_sw's KE ranges, rarea in vorticity, and f0
  ! in d_sw5's absolute vorticity).  Bounds per fv_arrays.F90
  ! :1313/:1317/:1321/:1324/:1328/:1331/:1335/:1338/:1344/:1399.
  call extchain_dump2('M_RAREA', &
      real(Atm(this_grid)%gridstruct%rarea(isd:ied, jsd:jed)))
  call extchain_dump2('M_RAREAC', &
      real(Atm(this_grid)%gridstruct%rarea_c(isd:ied + 1, jsd:jed + 1)))
  call extchain_dump2('M_RDX', &
      real(Atm(this_grid)%gridstruct%rdx(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_RDY', &
      real(Atm(this_grid)%gridstruct%rdy(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_RDXA', &
      real(Atm(this_grid)%gridstruct%rdxa(isd:ied, jsd:jed)))
  call extchain_dump2('M_RDYA', &
      real(Atm(this_grid)%gridstruct%rdya(isd:ied, jsd:jed)))
  call extchain_dump2('M_RDXC', &
      real(Atm(this_grid)%gridstruct%rdxc(isd:ied + 1, jsd:jed)))
  call extchain_dump2('M_RDYC', &
      real(Atm(this_grid)%gridstruct%rdyc(isd:ied, jsd:jed + 1)))
  call extchain_dump2('M_SINA', &
      real(Atm(this_grid)%gridstruct%sina(isd:ied + 1, jsd:jed + 1)))
  call extchain_dump2('M_F0', &
      real(Atm(this_grid)%gridstruct%f0(isd:ied, jsd:jed)))
  call extchain_dump3('M_SIN_SG', &
      real(Atm(this_grid)%gridstruct%sin_sg(isd:ied, jsd:jed, 1:9)))
  call extchain_dump3('M_COS_SG', &
      real(Atm(this_grid)%gridstruct%cos_sg(isd:ied, jsd:jed, 1:9)))

  do iv = 1, 2
    sent = sentinels(iv)
    write (vtag, '(A,I1)') '_v', iv
    call run_scalar_battery(sent, vtag)
    call run_vector_battery(sent, vtag)
  end do

  call extchain_dump_close()
  call mpp_sync()
  if (mpp_pe() == 0) write (*, '(A)') 'EXTCHAIN_DRIVER_DONE'
  call fms_end()

contains

  real function scalar_field(k, lon, lat)
    integer, intent(in) :: k
    real(kind=R_GRID), intent(in) :: lon, lat
    real(kind=R_GRID) :: d, q(2)
    select case (mod(k - 1, 5) + 1)
    case (1)
      scalar_field = 7.25
    case (2)
      scalar_field = real(lat)
    case (3)
      scalar_field = real(sin(lon)*cos(lat))
    case (4)
      scalar_field = real(240.0_R_GRID + 30.0_R_GRID*cos(2.0_R_GRID*lat) &
                          + 10.0_R_GRID*sin(lon)*cos(lat))
    case (5)
      q(1) = lon; q(2) = lat
      d = great_circle_dist(q, c0)
      scalar_field = real(25.0_R_GRID*exp(-(d/0.15_R_GRID)**2))
    end select
  end function scalar_field

  real(kind=R_GRID) function psi_field(k, lon, lat)
    integer, intent(in) :: k
    real(kind=R_GRID), intent(in) :: lon, lat
    real(kind=R_GRID), parameter :: u0 = 38.61068276698372_R_GRID
    real(kind=R_GRID), parameter :: a45 = 0.7853981633974483_R_GRID
    real(kind=R_GRID) :: d, q(2)
    select case (mod(k - 1, 5) + 1)
    case (1)
      psi_field = -u0*RADIUS*sin(lat)
    case (2)
      psi_field = -u0*RADIUS*(sin(lat)*cos(a45) - &
                              cos(lon)*cos(lat)*sin(a45))
    case (3)
      psi_field = -60.0_R_GRID*RADIUS*sin(lat)**3
    case (4)
      q(1) = lon; q(2) = lat
      d = great_circle_dist(q, c0)
      psi_field = 20.0_R_GRID*RADIUS*exp(-(d/0.2_R_GRID)**2)
    case (5)
      psi_field = -RADIUS*(15.0_R_GRID*sin(lat) + &
                           5.0_R_GRID*sin(2.0_R_GRID*lat) + &
                           4.0_R_GRID*cos(lon)*cos(lat))
    end select
  end function psi_field

  subroutine run_scalar_battery(sv, tagv)
    real, intent(in) :: sv
    character(len=*), intent(in) :: tagv
    real, allocatable :: a3(:, :, :), stg(:, :, :), cmp(:, :, :)
    real, allocatable :: b3(:, :, :), stgb(:, :, :), cmpb(:, :, :)
    real :: cert
    integer :: i, j, k
    character(len=256) :: note2

    ! ------------------------- A stagger (0,0) -------------------------
    allocate (a3(isd:ied, jsd:jed, npz))
    a3 = sv
    do k = 1, npz
      do j = js, je
        do i = is, ie
          a3(i, j, k) = scalar_field(k, &
              Atm(this_grid)%gridstruct%agrid(i, j, 1), &
              Atm(this_grid)%gridstruct%agrid(i, j, 2))
        end do
      end do
    end do
    call extchain_dump3('IN_A'//trim(tagv), a3)
    allocate (stg(isd:ied, jsd:jed, npz), cmp(isd:ied, jsd:jed, npz))
    stg = a3
    cmp = a3
    call ext_scalar_3d_staged(stg, Atm(this_grid)%gridstruct%dg, &
                              Atm(this_grid)%bd, Atm(this_grid)%domain, &
                              0, 0, 'A'//trim(tagv))
    call extchain_dump3('FIN_A'//trim(tagv), stg)
    call ext_scalar(cmp, Atm(this_grid)%gridstruct%dg, Atm(this_grid)%bd, &
                    Atm(this_grid)%domain, 0, 0)
    call extchain_dump3('CFIN_A'//trim(tagv), cmp)
    cert = maxval(abs(stg - cmp))
    write (note2, '(A,A,1X,ES24.16E3)') 'CERT A', trim(tagv), cert
    call extchain_mf_note(note2)
    deallocate (a3, stg, cmp)

    ! ------------------------- B stagger (1,1) -------------------------
    allocate (b3(isd:ied + 1, jsd:jed + 1, npz))
    b3 = sv
    do k = 1, npz
      do j = js, je + 1
        do i = is, ie + 1
          b3(i, j, k) = scalar_field(k, &
              Atm(this_grid)%gridstruct%grid(i, j, 1), &
              Atm(this_grid)%gridstruct%grid(i, j, 2))
        end do
      end do
    end do
    call extchain_dump3('IN_B'//trim(tagv), b3)
    allocate (stgb(isd:ied + 1, jsd:jed + 1, npz))
    allocate (cmpb(isd:ied + 1, jsd:jed + 1, npz))
    stgb = b3
    cmpb = b3
    call ext_scalar_3d_staged(stgb, Atm(this_grid)%gridstruct%dg, &
                              Atm(this_grid)%bd, Atm(this_grid)%domain, &
                              1, 1, 'B'//trim(tagv))
    call extchain_dump3('FIN_B'//trim(tagv), stgb)
    call ext_scalar(cmpb, Atm(this_grid)%gridstruct%dg, Atm(this_grid)%bd, &
                    Atm(this_grid)%domain, 1, 1)
    call extchain_dump3('CFIN_B'//trim(tagv), cmpb)
    cert = maxval(abs(stgb - cmpb))
    write (note2, '(A,A,1X,ES24.16E3)') 'CERT B', trim(tagv), cert
    call extchain_mf_note(note2)
    deallocate (b3, stgb, cmpb)
  end subroutine run_scalar_battery

  subroutine run_vector_battery(sv, tagv)
    real, intent(in) :: sv
    character(len=*), intent(in) :: tagv
    real(kind=R_GRID), allocatable :: psi(:, :, :), psib(:, :, :)
    real, allocatable :: u(:, :, :), v(:, :, :)
    real, allocatable :: us(:, :, :), vs(:, :, :)
    real, allocatable :: uc(:, :, :), vc(:, :, :)
    real, allocatable :: ucs(:, :, :), vcs(:, :, :)
    real, allocatable :: ucm(:, :, :), vcm(:, :, :)
    real, allocatable :: um(:, :, :), vm(:, :, :)
    real :: certu, certv
    integer :: i, j, k
    character(len=256) :: note2

    ! streamfunction at cell centres / corners over the full data domain
    ! (analytic evaluation -- every agrid/grid slot holds real kinked
    ! coordinates on the mpp state)
    allocate (psi(isd:ied, jsd:jed, npz))
    allocate (psib(isd:ied + 1, jsd:jed + 1, npz))
    do k = 1, npz
      do j = jsd, jed
        do i = isd, ied
          psi(i, j, k) = psi_field(k, &
              Atm(this_grid)%gridstruct%agrid(i, j, 1), &
              Atm(this_grid)%gridstruct%agrid(i, j, 2))
        end do
      end do
      do j = jsd, jed + 1
        do i = isd, ied + 1
          psib(i, j, k) = psi_field(k, &
              Atm(this_grid)%gridstruct%grid(i, j, 1), &
              Atm(this_grid)%gridstruct%grid(i, j, 2))
        end do
      end do
    end do

    ! ------------------------- D stagger (0,1,1,0) ----------------------
    ! test_cases.F90 defOnGrid=2 pattern: u = -dpsi/dyc, v = +dpsi/dxc
    allocate (u(isd:ied, jsd:jed + 1, npz), v(isd:ied + 1, jsd:jed, npz))
    u = sv
    v = sv
    do k = 1, npz
      do j = js, je + 1
        do i = is, ie
          u(i, j, k) = real(-(psi(i, j, k) - psi(i, j - 1, k)) &
                            /Atm(this_grid)%gridstruct%dyc(i, j))
        end do
      end do
      do j = js, je
        do i = is, ie + 1
          v(i, j, k) = real((psi(i, j, k) - psi(i - 1, j, k)) &
                            /Atm(this_grid)%gridstruct%dxc(i, j))
        end do
      end do
    end do
    call extchain_dump3('IN_DU'//trim(tagv), u)
    call extchain_dump3('IN_DV'//trim(tagv), v)
    allocate (us(isd:ied, jsd:jed + 1, npz), vs(isd:ied + 1, jsd:jed, npz))
    allocate (um(isd:ied, jsd:jed + 1, npz), vm(isd:ied + 1, jsd:jed, npz))
    us = u
    vs = v
    um = u
    vm = v
    call ext_vector_staged(us, vs, Atm(this_grid)%gridstruct%dg, &
                           Atm(this_grid)%bd, Atm(this_grid)%domain, &
                           Atm(this_grid)%gridstruct, &
                           Atm(this_grid)%flagstruct, 0, 1, 1, 0, &
                           'D'//trim(tagv))
    call extchain_dump3('FIN_DU'//trim(tagv), us)
    call extchain_dump3('FIN_DV'//trim(tagv), vs)
    call ext_vector(um, vm, Atm(this_grid)%gridstruct%dg, &
                    Atm(this_grid)%bd, Atm(this_grid)%domain, &
                    Atm(this_grid)%gridstruct, &
                    Atm(this_grid)%flagstruct, 0, 1, 1, 0)
    call extchain_dump3('CFIN_DU'//trim(tagv), um)
    call extchain_dump3('CFIN_DV'//trim(tagv), vm)
    certu = maxval(abs(us - um))
    certv = maxval(abs(vs - vm))
    write (note2, '(A,A,2(1X,ES24.16E3))') 'CERT D', trim(tagv), &
        certu, certv
    call extchain_mf_note(note2)
    deallocate (u, v, us, vs, um, vm)

    ! ------------------------- C stagger (1,0,0,1) ----------------------
    ! test_cases.F90 defOnGrid=0/1 pattern: uc = -dpsi_b/dy, vc = +dpsi_b/dx
    allocate (uc(isd:ied + 1, jsd:jed, npz), vc(isd:ied, jsd:jed + 1, npz))
    uc = sv
    vc = sv
    do k = 1, npz
      do j = js, je
        do i = is, ie + 1
          uc(i, j, k) = real(-(psib(i, j + 1, k) - psib(i, j, k)) &
                             /Atm(this_grid)%gridstruct%dy(i, j))
        end do
      end do
      do j = js, je + 1
        do i = is, ie
          vc(i, j, k) = real((psib(i + 1, j, k) - psib(i, j, k)) &
                             /Atm(this_grid)%gridstruct%dx(i, j))
        end do
      end do
    end do
    call extchain_dump3('IN_CU'//trim(tagv), uc)
    call extchain_dump3('IN_CV'//trim(tagv), vc)
    allocate (ucs(isd:ied + 1, jsd:jed, npz))
    allocate (vcs(isd:ied, jsd:jed + 1, npz))
    allocate (ucm(isd:ied + 1, jsd:jed, npz))
    allocate (vcm(isd:ied, jsd:jed + 1, npz))
    ucs = uc
    vcs = vc
    ucm = uc
    vcm = vc
    call ext_vector_staged(ucs, vcs, Atm(this_grid)%gridstruct%dg, &
                           Atm(this_grid)%bd, Atm(this_grid)%domain, &
                           Atm(this_grid)%gridstruct, &
                           Atm(this_grid)%flagstruct, 1, 0, 0, 1, &
                           'C'//trim(tagv))
    call extchain_dump3('FIN_CU'//trim(tagv), ucs)
    call extchain_dump3('FIN_CV'//trim(tagv), vcs)
    call ext_vector(ucm, vcm, Atm(this_grid)%gridstruct%dg, &
                    Atm(this_grid)%bd, Atm(this_grid)%domain, &
                    Atm(this_grid)%gridstruct, &
                    Atm(this_grid)%flagstruct, 1, 0, 0, 1)
    call extchain_dump3('CFIN_CU'//trim(tagv), ucm)
    call extchain_dump3('CFIN_CV'//trim(tagv), vcm)
    certu = maxval(abs(ucs - ucm))
    certv = maxval(abs(vcs - vcm))
    write (note2, '(A,A,2(1X,ES24.16E3))') 'CERT C', trim(tagv), &
        certu, certv
    call extchain_mf_note(note2)
    deallocate (uc, vc, ucs, vcs, ucm, vcm, psi, psib)
  end subroutine run_vector_battery

end program fv3_extchain_oracle_driver
