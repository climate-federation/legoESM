! Ext-projection oracle driver: builds the duo a2stag ext bases from
! python-exported ext A/B lattices (dg%a_pt/b_pt analogs), then runs
! the verbatim cubed_a2d_halo + cubed_a2c_halo on python-exported
! analytic geographic winds and dumps every basis + projected array.
!
! Scope (disclosed): the shim's fill_corner_region is an identity —
! inputs are fully defined analytic fields on the whole ng=4 data
! domain, so this oracle certifies the BASES (a2stag_metrics) and the
! PROJECTIONS (edge averaging + es/ew variant selection), not the
! corner fill (certified separately against smooth fields).
!
! stdin (text): n ng, then records
!   APT i j lon lat        (A ext points, fort i,j in isd..ied)
!   BPT i j lon lat        (B ext points, isd..ied+1)
!   UG  i j val / VG i j val  (geographic winds at A ext points)
! stdout: records
!   VLON i j k val / VLAT i j k val
!   EW k i j m val / ES k i j m val
!   UD i j val / VD i j val / UC i j val / VC i j val
program fv3_extproj_driver
  use extproj_shim_mod
  use extproj_extract_mod
  implicit none

  integer :: n, ng, isd, ied, jsd, jed, i, j, k, m, ios
  character(len=16) :: tag
  type(duogrid_type) :: dg
  type(fv_grid_type) :: gridstruct
  real, allocatable :: ug(:, :, :), vg(:, :, :)
  real, allocatable :: uc(:, :, :), vc(:, :, :)
  real, allocatable :: ud(:, :, :), vd(:, :, :)
  real(kind=R_GRID) :: a, b

  read (*, *) n, ng
  dg%bd%is = 1;  dg%bd%ie = n
  dg%bd%js = 1;  dg%bd%je = n
  dg%bd%ng = ng
  dg%bd%isd = 1 - ng; dg%bd%ied = n + ng
  dg%bd%jsd = 1 - ng; dg%bd%jed = n + ng
  isd = dg%bd%isd; ied = dg%bd%ied
  jsd = dg%bd%jsd; jed = dg%bd%jed

  allocate (dg%a_pt(2, isd:ied, jsd:jed))
  allocate (dg%b_pt(2, isd:ied + 1, jsd:jed + 1))
  allocate (dg%vlon_ext(isd:ied, jsd:jed, 3))
  allocate (dg%vlat_ext(isd:ied, jsd:jed, 3))
  allocate (dg%ew_ext(3, isd:ied + 1, jsd:jed, 2))
  allocate (dg%es_ext(3, isd:ied, jsd:jed + 1, 2))
  dg%vlon_ext = -9.9e9; dg%vlat_ext = -9.9e9
  dg%ew_ext = -9.9e9; dg%es_ext = -9.9e9
  allocate (ug(isd:ied, jsd:jed, 1), vg(isd:ied, jsd:jed, 1))
  ug = -9.9e9; vg = -9.9e9

  do
    read (*, *, iostat=ios) tag, i, j, a, b
    if (ios /= 0) exit
    select case (trim(tag))
    case ("APT"); dg%a_pt(1, i, j) = a; dg%a_pt(2, i, j) = b
    case ("BPT"); dg%b_pt(1, i, j) = a; dg%b_pt(2, i, j) = b
    case ("UG");  ug(i, j, 1) = a      ! b carries a checksum dummy
    case ("VG");  vg(i, j, 1) = a
    case default; stop "unknown record"
    end select
  end do

  call a2stag_metrics(gridstruct, dg%bd, dg)

  do k = 1, 3
    do j = jsd, jed
      do i = isd, ied
        write (*, '(A,3I6,ES26.17E3)') "VLON ", i, j, k, dg%vlon_ext(i, j, k)
        write (*, '(A,3I6,ES26.17E3)') "VLAT ", i, j, k, dg%vlat_ext(i, j, k)
      end do
    end do
  end do
  do m = 1, 2
    do j = jsd, jed
      do i = isd + 1, ied            ! a2stag populated range
        write (*, '(A,4I6,ES26.17E3)') "EW ", 1, i, j, m, dg%ew_ext(1, i, j, m)
        write (*, '(A,4I6,ES26.17E3)') "EW ", 2, i, j, m, dg%ew_ext(2, i, j, m)
        write (*, '(A,4I6,ES26.17E3)') "EW ", 3, i, j, m, dg%ew_ext(3, i, j, m)
      end do
    end do
    do j = jsd + 1, jed
      do i = isd, ied
        write (*, '(A,4I6,ES26.17E3)') "ES ", 1, i, j, m, dg%es_ext(1, i, j, m)
        write (*, '(A,4I6,ES26.17E3)') "ES ", 2, i, j, m, dg%es_ext(2, i, j, m)
        write (*, '(A,4I6,ES26.17E3)') "ES ", 3, i, j, m, dg%es_ext(3, i, j, m)
      end do
    end do
  end do

  allocate (ud(isd:ied, jsd:jed + 1, 1), vd(isd:ied + 1, jsd:jed, 1))
  ud = -9.9e9; vd = -9.9e9
  call cubed_a2d_halo(1, ug, vg, ud, vd, dg)
  do j = jsd + 1, jed                ! upstream written range
    do i = isd, ied
      write (*, '(A,2I6,ES26.17E3)') "UD ", i, j, ud(i, j, 1)
    end do
  end do
  do j = jsd, jed
    do i = isd + 1, ied
      write (*, '(A,2I6,ES26.17E3)') "VD ", i, j, vd(i, j, 1)
    end do
  end do

  allocate (uc(isd:ied + 1, jsd:jed, 1), vc(isd:ied, jsd:jed + 1, 1))
  uc = -9.9e9; vc = -9.9e9
  call cubed_a2c_halo(1, ug, vg, uc, vc, dg)
  do j = jsd, jed
    do i = isd + 1, ied
      write (*, '(A,2I6,ES26.17E3)') "UC ", i, j, uc(i, j, 1)
    end do
  end do
  do j = jsd + 1, jed
    do i = isd, ied
      write (*, '(A,2I6,ES26.17E3)') "VC ", i, j, vc(i, j, 1)
    end do
  end do

end program fv3_extproj_driver
