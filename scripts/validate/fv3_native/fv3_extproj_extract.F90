! VERBATIM extract — authoritative Zenodo 8327578 symmetryclean.
! Blocks (do not edit bodies):
!   tools/fv_duogrid.F90:2590-2763   cubed_a2c_halo + cubed_a2d_halo
!   tools/fv_duogrid.F90:2846-2992   unit_vect_latlon_ext + a2stag_metrics
!   model/fv_grid_utils.F90:1639-1665,1781-1791,1880-1893,1996-2036 helpers
module extproj_extract_mod
  use extproj_shim_mod, only: R_GRID, f_p, duogrid_type, fv_grid_bounds_type, fv_grid_type, fill_corner_region
  implicit none
  public
contains
 subroutine latlon2xyz(p, e, id)
!
! Routine to map (lon, lat) to (x,y,z)
!
 real(kind=R_GRID), intent(in) :: p(2)
 real(kind=R_GRID), intent(out):: e(3)
 integer, optional, intent(in):: id   ! id=0 do nothing; id=1, right_hand

 integer n
 real (f_p):: q(2)
 real (f_p):: e1, e2, e3

    do n=1,2
       q(n) = p(n)
    enddo

    e1 = cos(q(2)) * cos(q(1))
    e2 = cos(q(2)) * sin(q(1))
    e3 = sin(q(2))
!-----------------------------------
! Truncate to the desired precision:
!-----------------------------------
    e(1) = e1
    e(2) = e2
    e(3) = e3

 end subroutine latlon2xyz
 subroutine vect_cross(e, p1, p2)
 real(kind=R_GRID), intent(in) :: p1(3), p2(3)
 real(kind=R_GRID), intent(out):: e(3)
!
! Perform cross products of 3D vectors: e = P1 X P2
!
      e(1) = p1(2)*p2(3) - p1(3)*p2(2)
      e(2) = p1(3)*p2(1) - p1(1)*p2(3)
      e(3) = p1(1)*p2(2) - p1(2)*p2(1)

 end subroutine vect_cross
 subroutine normalize_vect(e)
!                              Make e an unit vector
 real(kind=R_GRID), intent(inout):: e(3)
 real(f_p):: pdot
 integer k

    pdot = e(1)**2 + e(2)**2 + e(3)**2
    pdot = sqrt( pdot )

    do k=1,3
       e(k) = e(k) / pdot
    enddo

 end subroutine normalize_vect
 subroutine mid_pt3_cart(p1, p2, e)
       real(kind=R_GRID), intent(IN)  :: p1(3), p2(3)
       real(kind=R_GRID), intent(OUT) :: e(3)
!
       real (f_p):: q1(3), q2(3)
       real (f_p):: dd, e1, e2, e3
       integer k

       do k=1,3
          q1(k) = p1(k)
          q2(k) = p2(k)
       enddo

       e1 = q1(1) + q2(1)
       e2 = q1(2) + q2(2)
       e3 = q1(3) + q2(3)

       dd = sqrt( e1**2 + e2**2 + e3**2 )
       e1 = e1 / dd
       e2 = e2 / dd
       e3 = e3 / dd

       e(1) = e1
       e(2) = e2
       e(3) = e3

 end subroutine mid_pt3_cart
 subroutine mid_pt_cart(p1, p2, e3)
    real(kind=R_GRID), intent(IN)  :: p1(2), p2(2)
    real(kind=R_GRID), intent(OUT) :: e3(3)
!-------------------------------------
    real(kind=R_GRID) e1(3), e2(3)

    call latlon2xyz(p1, e1)
    call latlon2xyz(p2, e2)
    call mid_pt3_cart(e1, e2, e3)

 end subroutine mid_pt_cart
  subroutine cubed_a2c_halo(npz, uatemp, vatemp, uc, vc, dg)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HERE WE USE THE BD OF ng=4 domain, so isd,ied,jsd,jed are offsetted by one
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Purpose; Transform wind on A grid to C grid

    type(duogrid_type), intent(in), target :: dg
    integer, intent(in):: npz
    real, intent(inout), dimension(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed, npz):: uatemp, vatemp
    real, intent(inout):: uc(dg%bd%isd:dg%bd%ied + 1, dg%bd%jsd:dg%bd%jed, npz)
    real, intent(inout):: vc(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed + 1, npz)
! local:
    real v3(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)
    real ue(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges
    real ve(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges

    integer i, j, k

    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: ew, es
    real(kind=R_GRID), pointer, dimension(:, :, :)   :: vlon, vlat

    integer :: isd, ied, jsd, jed

    isd = dg%bd%isd
    ied = dg%bd%ied
    jsd = dg%bd%jsd
    jed = dg%bd%jed

    vlon => dg%vlon_ext
    vlat => dg%vlat_ext
    ew => dg%ew_ext
    es => dg%es_ext

! check if this ok
    call fill_corner_region(vatemp, dg%bd, dg, 0, 0)
    call fill_corner_region(uatemp, dg%bd, dg, 0, 0)

    do k = 1, npz
! Compute 3D wind on A grid
      do j = jsd, jed
        do i = isd, ied
          v3(1, i, j) = uatemp(i, j, k)*vlon(i, j, 1) + vatemp(i, j, k)*vlat(i, j, 1)
          v3(2, i, j) = uatemp(i, j, k)*vlon(i, j, 2) + vatemp(i, j, k)*vlat(i, j, 2)
          v3(3, i, j) = uatemp(i, j, k)*vlon(i, j, 3) + vatemp(i, j, k)*vlat(i, j, 3)
        end do
      end do

! A --> C
! Interpolate to cell edges
      do j = jsd, jed
        do i = isd + 1, ied
          ue(1, i, j) = 0.5*(v3(1, i - 1, j) + v3(1, i, j))
          ue(2, i, j) = 0.5*(v3(2, i - 1, j) + v3(2, i, j))
          ue(3, i, j) = 0.5*(v3(3, i - 1, j) + v3(3, i, j))
        end do
      end do

      do j = jsd + 1, jed
        do i = isd, ied
          ve(1, i, j) = 0.5*(v3(1, i, j - 1) + v3(1, i, j))
          ve(2, i, j) = 0.5*(v3(2, i, j - 1) + v3(2, i, j))
          ve(3, i, j) = 0.5*(v3(3, i, j - 1) + v3(3, i, j))
        end do
      end do

      do j = jsd, jed
        do i = isd + 1, ied
          uc(i, j, k) = ue(1, i, j)*ew(1, i, j, 1) + &
                        ue(2, i, j)*ew(2, i, j, 1) + &
                        ue(3, i, j)*ew(3, i, j, 1)
        end do
      end do
      do j = jsd + 1, jed
        do i = isd, ied
          vc(i, j, k) = ve(1, i, j)*es(1, i, j, 2) + &
                        ve(2, i, j)*es(2, i, j, 2) + &
                        ve(3, i, j)*es(3, i, j, 2)
        end do
      end do

    end do         ! k-loop

  end subroutine cubed_a2c_halo

  subroutine cubed_a2d_halo(npz, uatemp, vatemp, ud, vd, dg)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HERE WE USE THE BD OF ng=4 domain, so isd,ied,jsd,jed are offsetted by one
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Purpose; Transform wind on A grid to D grid

    type(duogrid_type), intent(IN), target :: dg
    integer, intent(in):: npz
    real, intent(inout), dimension(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed, npz):: uatemp, vatemp
    real, intent(inout):: ud(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed + 1, npz)
    real, intent(inout):: vd(dg%bd%isd:dg%bd%ied + 1, dg%bd%jsd:dg%bd%jed, npz)
! local:
    real v3(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)
    real ue(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges
    real ve(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges

    integer i, j, k

    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: ew, es
    real(kind=R_GRID), pointer, dimension(:, :, :) :: vlon, vlat

    integer :: isd, ied, jsd, jed

    isd = dg%bd%isd
    ied = dg%bd%ied
    jsd = dg%bd%jsd
    jed = dg%bd%jed

    ud = -999.
    vd = -888.

    vlon => dg%vlon_ext
    vlat => dg%vlat_ext
    ew => dg%ew_ext
    es => dg%es_ext

! check if this ok
    call fill_corner_region(vatemp, dg%bd, dg, 0, 0)
    call fill_corner_region(uatemp, dg%bd, dg, 0, 0)

    do k = 1, npz
! Compute 3D wind on A grid
      do j = jsd, jed
        do i = isd, ied
          v3(1, i, j) = uatemp(i, j, k)*vlon(i, j, 1) + vatemp(i, j, k)*vlat(i, j, 1)
          v3(2, i, j) = uatemp(i, j, k)*vlon(i, j, 2) + vatemp(i, j, k)*vlat(i, j, 2)
          v3(3, i, j) = uatemp(i, j, k)*vlon(i, j, 3) + vatemp(i, j, k)*vlat(i, j, 3)
        end do
      end do

! A --> D
! Interpolate to cell edges
      do j = jsd + 1, jed
        do i = isd, ied
          ue(1, i, j) = 0.5*(v3(1, i, j - 1) + v3(1, i, j))
          ue(2, i, j) = 0.5*(v3(2, i, j - 1) + v3(2, i, j))
          ue(3, i, j) = 0.5*(v3(3, i, j - 1) + v3(3, i, j))
        end do
      end do

      do j = jsd, jed
        do i = isd + 1, ied
          ve(1, i, j) = 0.5*(v3(1, i - 1, j) + v3(1, i, j))
          ve(2, i, j) = 0.5*(v3(2, i - 1, j) + v3(2, i, j))
          ve(3, i, j) = 0.5*(v3(3, i - 1, j) + v3(3, i, j))
        end do
      end do

      do j = jsd + 1, jed
        do i = isd, ied
          ud(i, j, k) = ue(1, i, j)*es(1, i, j, 1) + &
                        ue(2, i, j)*es(2, i, j, 1) + &
                        ue(3, i, j)*es(3, i, j, 1)
        end do
      end do
      do j = jsd, jed
        do i = isd + 1, ied
          vd(i, j, k) = ve(1, i, j)*ew(1, i, j, 2) + &
                        ve(2, i, j)*ew(2, i, j, 2) + &
                        ve(3, i, j)*ew(3, i, j, 2)
        end do
      end do

    end do         ! k-loop

  end subroutine cubed_a2d_halo
  subroutine unit_vect_latlon_ext(pp, elon, elat)
    real(kind=R_GRID), intent(IN)  :: pp(2)
    real(kind=R_GRID), intent(OUT) :: elon(3), elat(3)

    real(selected_real_kind(20)):: lon, lat
    real(selected_real_kind(20)):: sin_lon, cos_lon, sin_lat, cos_lat

    lon = pp(1)
    lat = pp(2)

    sin_lon = sin(lon)
    cos_lon = cos(lon)
    sin_lat = sin(lat)
    cos_lat = cos(lat)

    elon(1) = -sin_lon
    elon(2) = cos_lon
    elon(3) = 0.d0

    elat(1) = -sin_lat*cos_lon
    elat(2) = -sin_lat*sin_lon
    elat(3) = cos_lat

  end subroutine unit_vect_latlon_ext

  subroutine a2stag_metrics(gridstruct, bd, dg)

    type(fv_grid_bounds_type), intent(IN) :: bd
    type(duogrid_type), intent(INOUT), target :: dg
    type(fv_grid_type), intent(IN), target :: gridstruct
! local:

    integer i, j, k

    real(kind=R_GRID), dimension(bd%isd:bd%ied, bd%jsd:bd%jed, 3)   :: vlon, vlat
    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: ew, es

    real(kind=R_GRID), pointer, dimension(:, :, :) :: aagrid_local
    real(kind=R_GRID), pointer, dimension(:, :, :) :: bbgrid_local

    real(kind=R_GRID) p1(3), p2(3), p3(3), pp(3)
    real(kind=R_GRID) grid3(3, bd%isd:bd%ied + 1, bd%jsd:bd%jed + 1)

    integer :: isd, ied, jsd, jed
    integer :: l, ll

    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed

    allocate (ew(3, isd:ied + 1, jsd:jed, 2))
    allocate (es(3, isd:ied, jsd:jed + 1, 2))

    allocate (aagrid_local(isd:ied, jsd:jed, 2))
    allocate (bbgrid_local(isd:ied + 1, jsd:jed + 1, 2))

    do j = jsd, jed
      do i = isd, ied
        do l = 1, 2
          aagrid_local(i, j, l) = dg%a_pt(l, i, j)
        end do
      end do
    end do

    do j = jsd, jed + 1
      do i = isd, ied + 1
        do l = 1, 2
          bbgrid_local(i, j, l) = dg%b_pt(l, i, j)
        end do
      end do
    end do

    do j = jsd, jed + 1
      do i = isd, ied + 1
        call latlon2xyz(bbgrid_local(i, j, 1:2), grid3(1, i, j))
      end do
    end do

    do j = jsd, jed
      do i = isd + 1, ied
        call mid_pt_cart(bbgrid_local(i, j, 1:2), bbgrid_local(i, j + 1, 1:2), pp)
        call latlon2xyz(aagrid_local(i - 1, j, 1:2), p3)
        call latlon2xyz(aagrid_local(i, j, 1:2), p1)
        call vect_cross(p2, p3, p1)
        call vect_cross(ew(1:3, i, j, 1), p2, pp)
        call normalize_vect(ew(1:3, i, j, 1))
!---
        call vect_cross(p1, grid3(1, i, j), grid3(1, i, j + 1))
        call vect_cross(ew(1:3, i, j, 2), p1, pp)
        call normalize_vect(ew(1:3, i, j, 2))
      end do
    end do

    do j = jsd + 1, jed
      do i = isd, ied
        call mid_pt_cart(bbgrid_local(i, j, 1:2), bbgrid_local(i + 1, j, 1:2), pp)
        call latlon2xyz(aagrid_local(i, j, 1:2), p1)
        call latlon2xyz(aagrid_local(i, j - 1, 1:2), p3)
        call vect_cross(p2, p3, p1)
        call vect_cross(es(1:3, i, j, 2), p2, pp)
        call normalize_vect(es(1:3, i, j, 2))
!---
        call vect_cross(p3, grid3(1, i, j), grid3(1, i + 1, j))
        call vect_cross(es(1:3, i, j, 1), p3, pp)
        call normalize_vect(es(1:3, i, j, 1))
      end do
    end do

! Compute 3D wind on A grid
    do j = jsd, jed
      do i = isd, ied
        call unit_vect_latlon_ext(aagrid_local(i, j, 1:2), vlon(i, j, 1:3), vlat(i, j, 1:3))
      end do
    end do

! Populate in dg
    do l = 1, 3
      do j = jsd, jed
        do i = isd + 1, ied
          dg%ew_ext(l, i, j, 1) = ew(l, i, j, 1)
          dg%ew_ext(l, i, j, 2) = ew(l, i, j, 2)
        end do
      end do

      do j = jsd + 1, jed
        do i = isd, ied
          dg%es_ext(l, i, j, 1) = es(l, i, j, 1)
          dg%es_ext(l, i, j, 2) = es(l, i, j, 2)
        end do
      end do

      do j = jsd, jed
        do i = isd, ied
          dg%vlon_ext(i, j, l) = vlon(i, j, l)
          dg%vlat_ext(i, j, l) = vlat(i, j, l)
        end do
      end do
    end do

    deallocate (ew)
    deallocate (es)

    deallocate (aagrid_local)
    deallocate (bbgrid_local)

  end subroutine a2stag_metrics
end module extproj_extract_mod
