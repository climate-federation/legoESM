! FMS mpp_get_boundary sentinel oracle — BGRID_NE + CGRID_NE cube-seam
! buffer semantics (codex vertex-hunt protocol).
!
! Runs the REAL FMS (NOAA-GFDL/FMS) on the genuine 6-tile cubed-sphere
! mosaic (contact table verbatim from FMS test_domains_utility_mod) with
! index-coded fields, and prints every boundary-buffer slot delivered to
! every tile.  This is THE oracle for the legoESM ports
! `average_shared_edge_bgrid` / `average_shared_edge_cgrid`
! (`_bgrid_edge_partner` / `_cgrid_edge_partner` pairing + vector sign),
! matching the dyn_core.F90:968-1020 (ubb/vbbtemp, BGRID_NE) and
! dyn_core.F90:853-900 (allflux, CGRID_NE) call shapes.
!
! Field coding (decodable from any buffer value v = s*(2e6*t + 1e6*c +
! 1e3*i + j)): sign s = vector-rotation sign applied by FMS; t = source
! tile 1..6; c = 0 fieldx / 1 fieldy; (i,j) = source GLOBAL B/C node
! index.  A NINETY/MINUS_NINETY contact delivers the OTHER component
! (c flips) with FMS's sign; ZERO delivers the same component unsigned.
!
! Build: gfortran + libFMS (see fv3_recon/fms_sentinel.sbatch); run with
! mpirun -np 6.  n=12 per face, layout 1x1 per tile, halo 3.
program fms_sentinel_boundary
  use mpp_mod,         only: mpp_init, mpp_exit, mpp_pe, mpp_npes, &
                             mpp_error, FATAL, stdout
  use mpp_domains_mod, only: mpp_domains_init, mpp_define_mosaic, &
                             mpp_get_boundary, mpp_get_compute_domain, &
                             domain2d, BGRID_NE, CGRID_NE, &
                             mpp_domains_exit
  implicit none

  integer, parameter :: n = 12, halo = 3
  type(domain2d) :: domain
  integer :: pe, npes, tile
  integer :: isc, iec, jsc, jec
  integer :: i, j
  real(8), allocatable :: fbx(:,:), fby(:,:)      ! BGRID (n+1, n+1)
  real(8), allocatable :: fcx(:,:), fcy(:,:)      ! CGRID x:(n+1,n) y:(n,n+1)
  real(8), allocatable :: wbx(:), ebx(:), sby(:), nby(:)
  integer :: out

  call mpp_init()
  pe = mpp_pe(); npes = mpp_npes()
  if (npes /= 6) call mpp_error(FATAL, "sentinel needs exactly 6 PEs")
  call mpp_domains_init()

  call build_cube(domain)
  tile = pe + 1
  call mpp_get_compute_domain(domain, isc, iec, jsc, jec)
  out = stdout()

  ! ---------------- BGRID_NE (dyn_core ubb/vbbtemp shape) -------------
  allocate (fbx(isc:iec+1, jsc:jec+1), fby(isc:iec+1, jsc:jec+1))
  do j = jsc, jec + 1
     do i = isc, iec + 1
        fbx(i, j) = code(tile, 0, i, j)
        fby(i, j) = code(tile, 1, i, j)
     end do
  end do
  allocate (wbx(jec - jsc + 2), ebx(jec - jsc + 2))
  allocate (sby(iec - isc + 2), nby(iec - isc + 2))
  wbx = -9.9d9; ebx = -9.9d9; sby = -9.9d9; nby = -9.9d9
  call mpp_get_boundary(fbx, fby, domain, &
                        wbufferx=wbx, ebufferx=ebx, &
                        sbuffery=sby, nbuffery=nby, gridtype=BGRID_NE)
  do j = 1, size(wbx)
     write (out, '(a,i2,a,i3,f16.1)') 'BG t', tile, ' W ', j, wbx(j)
     write (out, '(a,i2,a,i3,f16.1)') 'BG t', tile, ' E ', j, ebx(j)
  end do
  do i = 1, size(sby)
     write (out, '(a,i2,a,i3,f16.1)') 'BG t', tile, ' S ', i, sby(i)
     write (out, '(a,i2,a,i3,f16.1)') 'BG t', tile, ' N ', i, nby(i)
  end do
  deallocate (wbx, ebx, sby, nby)

  ! ---------------- CGRID_NE (dyn_core allflux shape) -----------------
  allocate (fcx(isc:iec+1, jsc:jec), fcy(isc:iec, jsc:jec+1))
  do j = jsc, jec
     do i = isc, iec + 1
        fcx(i, j) = code(tile, 0, i, j)
     end do
  end do
  do j = jsc, jec + 1
     do i = isc, iec
        fcy(i, j) = code(tile, 1, i, j)
     end do
  end do
  allocate (wbx(jec - jsc + 1), ebx(jec - jsc + 1))
  allocate (sby(iec - isc + 1), nby(iec - isc + 1))
  wbx = -9.9d9; ebx = -9.9d9; sby = -9.9d9; nby = -9.9d9
  call mpp_get_boundary(fcx, fcy, domain, &
                        wbufferx=wbx, ebufferx=ebx, &
                        sbuffery=sby, nbuffery=nby, gridtype=CGRID_NE)
  do j = 1, size(wbx)
     write (out, '(a,i2,a,i3,f16.1)') 'CG t', tile, ' W ', j, wbx(j)
     write (out, '(a,i2,a,i3,f16.1)') 'CG t', tile, ' E ', j, ebx(j)
  end do
  do i = 1, size(sby)
     write (out, '(a,i2,a,i3,f16.1)') 'CG t', tile, ' S ', i, sby(i)
     write (out, '(a,i2,a,i3,f16.1)') 'CG t', tile, ' N ', i, nby(i)
  end do

  write (out, '(a,i2)') 'SENTINEL_DONE t', tile
  call mpp_domains_exit()
  call mpp_exit()

contains

  pure real(8) function code(t, c, ii, jj)
    integer, intent(in) :: t, c, ii, jj
    code = 2.0d6*t + 1.0d6*c + 1.0d3*ii + jj
  end function code

  subroutine build_cube(dom)
    ! verbatim contact table: FMS test_domains_utility_mod
    ! define_cubic_mosaic, ni=nj=n, layout 1x1 per tile, 6 PEs
    type(domain2d), intent(inout) :: dom
    integer :: ni(6), nj(6), global_indices(4, 6), layout(2, 6)
    integer :: pe_start(6), pe_end(6)
    integer, dimension(12) :: istart1, iend1, jstart1, jend1, tile1
    integer, dimension(12) :: istart2, iend2, jstart2, jend2, tile2
    integer :: k

    ni = n; nj = n
    do k = 1, 6
       global_indices(:, k) = (/1, n, 1, n/)
       layout(:, k) = (/1, 1/)
       pe_start(k) = k - 1; pe_end(k) = k - 1
    end do

    tile1(1) = 1; tile2(1) = 2
    istart1(1) = n; iend1(1) = n; jstart1(1) = 1; jend1(1) = n
    istart2(1) = 1; iend2(1) = 1; jstart2(1) = 1; jend2(1) = n
    tile1(2) = 1; tile2(2) = 3
    istart1(2) = 1; iend1(2) = n; jstart1(2) = n; jend1(2) = n
    istart2(2) = 1; iend2(2) = 1; jstart2(2) = n; jend2(2) = 1
    tile1(3) = 1; tile2(3) = 5
    istart1(3) = 1; iend1(3) = 1; jstart1(3) = 1; jend1(3) = n
    istart2(3) = n; iend2(3) = 1; jstart2(3) = n; jend2(3) = n
    tile1(4) = 1; tile2(4) = 6
    istart1(4) = 1; iend1(4) = n; jstart1(4) = 1; jend1(4) = 1
    istart2(4) = 1; iend2(4) = n; jstart2(4) = n; jend2(4) = n
    tile1(5) = 2; tile2(5) = 3
    istart1(5) = 1; iend1(5) = n; jstart1(5) = n; jend1(5) = n
    istart2(5) = 1; iend2(5) = n; jstart2(5) = 1; jend2(5) = 1
    tile1(6) = 2; tile2(6) = 4
    istart1(6) = n; iend1(6) = n; jstart1(6) = 1; jend1(6) = n
    istart2(6) = n; iend2(6) = 1; jstart2(6) = 1; jend2(6) = 1
    tile1(7) = 2; tile2(7) = 6
    istart1(7) = 1; iend1(7) = n; jstart1(7) = 1; jend1(7) = 1
    istart2(7) = n; iend2(7) = n; jstart2(7) = n; jend2(7) = 1
    tile1(8) = 3; tile2(8) = 4
    istart1(8) = n; iend1(8) = n; jstart1(8) = 1; jend1(8) = n
    istart2(8) = 1; iend2(8) = 1; jstart2(8) = 1; jend2(8) = n
    tile1(9) = 3; tile2(9) = 5
    istart1(9) = 1; iend1(9) = n; jstart1(9) = n; jend1(9) = n
    istart2(9) = 1; iend2(9) = 1; jstart2(9) = n; jend2(9) = 1
    tile1(10) = 4; tile2(10) = 5
    istart1(10) = 1; iend1(10) = n; jstart1(10) = n; jend1(10) = n
    istart2(10) = 1; iend2(10) = n; jstart2(10) = 1; jend2(10) = 1
    tile1(11) = 4; tile2(11) = 6
    istart1(11) = n; iend1(11) = n; jstart1(11) = 1; jend1(11) = n
    istart2(11) = n; iend2(11) = 1; jstart2(11) = 1; jend2(11) = 1
    tile1(12) = 5; tile2(12) = 6
    istart1(12) = n; iend1(12) = n; jstart1(12) = 1; jend1(12) = n
    istart2(12) = 1; iend2(12) = 1; jstart2(12) = 1; jend2(12) = n

    call mpp_define_mosaic(global_indices, layout, dom, 6, 12, &
         tile1, tile2, istart1, iend1, jstart1, jend1, &
         istart2, iend2, jstart2, jend2, pe_start, pe_end, &
         symmetry=.true., whalo=halo, ehalo=halo, shalo=halo, &
         nhalo=halo, name='sentinel_cube')
  end subroutine build_cube

end program fms_sentinel_boundary
