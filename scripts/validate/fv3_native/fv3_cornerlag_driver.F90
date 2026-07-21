! Corner-Lagrange oracle driver: builds the compute_lagrange_coeff
! weight tables from python-exported ext A points, then runs the
! verbatim fill_corner_region_2d on python-exported fields at each of
! the four staggerings and dumps the full arrays after the fill.
!
! Bounds split mirrors upstream: bd = the FIELD bounds (ng), while
! dg%a_pt lives on the ONE-RING-WIDER lattice (the dg%bd ng+1 analog of
! duogrid_alloc) — compute_lagrange_coeff(bd, dg) indexes a_pt one past
! the field lattice at staggered targets.
!
! stdin: n ng, then records (always: TAG i j a b)
!   APT i j lon lat        (A ext points over isd-1..ied+1 square)
!   F00/F11/F01/F10 i j v 0
!   COEF 0 0 0 0           (build the weight tables; once, after APT)
!   RUN istag jstag 0 0    (fill + dump that stagger's field)
! stdout: OUT istag jstag i j val
program fv3_cornerlag_driver
  use cornerlag_shim_mod
  use cornerlag_extract_mod
  implicit none

  integer :: n, ng, isd, ied, jsd, jed, i, j, ios, istag, jstag
  character(len=16) :: tag
  type(duogrid_type) :: dg
  real, allocatable :: f00(:, :), f11(:, :), f01(:, :), f10(:, :)
  real(kind=R_GRID) :: a, b
  integer :: ii, jj

  read (*, *) n, ng
  dg%bd%is = 1;  dg%bd%ie = n
  dg%bd%js = 1;  dg%bd%je = n
  dg%bd%ng = ng
  dg%bd%isd = 1 - ng; dg%bd%ied = n + ng
  dg%bd%jsd = 1 - ng; dg%bd%jed = n + ng
  isd = dg%bd%isd; ied = dg%bd%ied
  jsd = dg%bd%jsd; jed = dg%bd%jed

  ! a_pt one ring wider than the field lattice (dg%bd analog)
  allocate (dg%a_pt(2, isd - 1:ied + 1, jsd - 1:jed + 1))
  allocate (dg%xp(1:interporder + 1, dg%bd%ie - interporder:ied + 1, &
                  jsd:jed + 1, 4))
  allocate (dg%xm(1:interporder + 1, isd:dg%bd%is + interporder, &
                  jsd:jed + 1, 4))
  allocate (dg%yp(1:interporder + 1, isd:ied + 1, &
                  dg%bd%je - interporder:jed + 1, 4))
  allocate (dg%ym(1:interporder + 1, isd:ied + 1, &
                  jsd:dg%bd%js + interporder, 4))
  dg%xp = -9.9e9; dg%xm = -9.9e9; dg%yp = -9.9e9; dg%ym = -9.9e9
  allocate (f00(isd:ied, jsd:jed), f11(isd:ied + 1, jsd:jed + 1))
  allocate (f01(isd:ied, jsd:jed + 1), f10(isd:ied + 1, jsd:jed))
  f00 = -9.9e9; f11 = -9.9e9; f01 = -9.9e9; f10 = -9.9e9

  do
    read (*, *, iostat=ios) tag, i, j, a, b
    if (ios /= 0) exit
    select case (trim(tag))
    case ("APT"); dg%a_pt(1, i, j) = a; dg%a_pt(2, i, j) = b
    case ("F00"); f00(i, j) = a
    case ("F11"); f11(i, j) = a
    case ("F01"); f01(i, j) = a
    case ("F10"); f10(i, j) = a
    case ("COEF")
      call compute_lagrange_coeff(dg%xp, dg%xm, dg%yp, dg%ym, dg%bd, dg)
    case ("RUN")
      istag = i; jstag = j
      if (istag == 0 .and. jstag == 0) then
        call fill_corner_region(f00, dg%bd, dg, 0, 0)
        do jj = jsd, jed
          do ii = isd, ied
            write (*, '(A,4I6,ES26.17E3)') "OUT ", 0, 0, ii, jj, f00(ii, jj)
          end do
        end do
      else if (istag == 1 .and. jstag == 1) then
        call fill_corner_region(f11, dg%bd, dg, 1, 1)
        do jj = jsd, jed + 1
          do ii = isd, ied + 1
            write (*, '(A,4I6,ES26.17E3)') "OUT ", 1, 1, ii, jj, f11(ii, jj)
          end do
        end do
      else if (istag == 0 .and. jstag == 1) then
        call fill_corner_region(f01, dg%bd, dg, 0, 1)
        do jj = jsd, jed + 1
          do ii = isd, ied
            write (*, '(A,4I6,ES26.17E3)') "OUT ", 0, 1, ii, jj, f01(ii, jj)
          end do
        end do
      else
        call fill_corner_region(f10, dg%bd, dg, 1, 0)
        do jj = jsd, jed
          do ii = isd, ied + 1
            write (*, '(A,4I6,ES26.17E3)') "OUT ", 1, 0, ii, jj, f10(ii, jj)
          end do
        end do
      end if
    case default; stop "unknown record"
    end select
  end do

end program fv3_cornerlag_driver
