! VERBATIM extract — authoritative Zenodo 8327578 symmetryclean.
! Blocks (do not edit bodies):
!   tools/fv_duogrid.F90:1719-1903   fill_corner_region_2d
!   tools/fv_duogrid.F90:2159-2272   compute_lagrange_coeff
!   tools/fv_duogrid.F90:2369-2449   lagrange_poly_interp_2d
!   model/fv_grid_utils.F90:2040-2062 great_circle_dist
module cornerlag_extract_mod
  use cornerlag_shim_mod, only: R_GRID, f_p, duogrid_type, &
    fv_grid_bounds_type, interporder, timing_on, timing_off, &
    mpp_pe, mpp_root_pe
  implicit none
  public
  interface lagrange_poly_interp
    module procedure lagrange_poly_interp_2d
  end interface lagrange_poly_interp
  interface fill_corner_region
    module procedure fill_corner_region_2d
  end interface fill_corner_region
contains
 real function great_circle_dist( q1, q2, radius )
      real(kind=R_GRID), intent(IN)           :: q1(2), q2(2)
      real(kind=R_GRID), intent(IN), optional :: radius

      real (f_p):: p1(2), p2(2)
      real (f_p):: beta
      integer n

      do n=1,2
         p1(n) = q1(n)
         p2(n) = q2(n)
      enddo

      beta = asin( sqrt( sin((p1(2)-p2(2))/2.)**2 + cos(p1(2))*cos(p2(2))*   &
                         sin((p1(1)-p2(1))/2.)**2 ) ) * 2.

      if ( present(radius) ) then
           great_circle_dist = radius * beta
      else
           great_circle_dist = beta   ! Returns the angle
      endif

  end function great_circle_dist
  subroutine fill_corner_region_2d(vel, bd, dg, istag, jstag)
    type(fv_grid_bounds_type), intent(IN) :: bd
    type(duogrid_type), intent(in) :: dg
    integer, intent(IN) :: istag, jstag
    real, dimension(bd%isd:, bd%jsd:) :: vel

    integer :: i, j, is, ie, js, je, isd, ied, jsd, jed
    !integer :: interporder = 3 !order is interporder+1

    real, dimension(bd%isd:bd%ied + istag, bd%jsd:bd%jed + jstag) :: veltemp
    real, dimension(bd%isd:bd%ied + istag, bd%jsd:bd%jed + jstag) :: veltempp
    is = bd%is
    ie = bd%ie
    js = bd%js
    je = bd%je
    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed

    !lastpoint in compute domain
    ie = ie + istag
    je = je + jstag

    ! NE corner
!     |   1--2--3
!     |   |     |
!     |   4  5  6
!     |   |     |
!     |   7--8--9
!     -------------

    if (dg%rmp_ne) then
      call lagrange_poly_interp(vel, ie + 1, je + 2, bd, dg, 'X+', istag, jstag, interporder) !4
      call lagrange_poly_interp(vel, ie + 1, je + 3, bd, dg, 'X+', istag, jstag, interporder) !1
      call lagrange_poly_interp(vel, ie + 2, je + 3, bd, dg, 'X+', istag, jstag, interporder) !2
      call lagrange_poly_interp(vel, ie + 2, je + 1, bd, dg, 'Y+', istag, jstag, interporder) !8
      call lagrange_poly_interp(vel, ie + 3, je + 1, bd, dg, 'Y+', istag, jstag, interporder) !9
      call lagrange_poly_interp(vel, ie + 3, je + 2, bd, dg, 'Y+', istag, jstag, interporder) !6

      veltemp = vel
      veltempp = vel

      !diag
      i = ie + 1
      j = je + 1
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X+', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y+', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))

      i = ie + 3
      j = je + 3
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X+', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y+', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))

      i = ie + 2
      j = je + 2
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X+', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y+', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
    end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! NW
!        1--2--3 |
!        |     | |
!        4  5  6 |
!        |     | |
!        7--8--9 |
!        ---------

    if (dg%rmp_nw) then
      call lagrange_poly_interp(vel, is - 1, je + 2, bd, dg, 'X-', istag, jstag, interporder) !6
      call lagrange_poly_interp(vel, is - 1, je + 3, bd, dg, 'X-', istag, jstag, interporder) !3
      call lagrange_poly_interp(vel, is - 2, je + 3, bd, dg, 'X-', istag, jstag, interporder) !2
      call lagrange_poly_interp(vel, is - 2, je + 1, bd, dg, 'Y+', istag, jstag, interporder) !8
      call lagrange_poly_interp(vel, is - 3, je + 1, bd, dg, 'Y+', istag, jstag, interporder) !7
      call lagrange_poly_interp(vel, is - 3, je + 2, bd, dg, 'Y+', istag, jstag, interporder) !4

      veltemp = vel
      veltempp = vel

      !diag
      i = is - 1
      j = je + 1
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X-', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y+', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))

      i = is - 3
      j = je + 3
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X-', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y+', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))

      i = is - 2
      j = je + 2
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X-', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y+', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
    end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! SE corner
!     --------------
!     |   1--2--3
!     |   |     |
!     |   4  5  6
!     |   |     |
!     |   7--8--9
!
    if (dg%rmp_se) then
      call lagrange_poly_interp(vel, ie + 1, js - 2, bd, dg, 'X+', istag, jstag, interporder) !4
      call lagrange_poly_interp(vel, ie + 1, js - 3, bd, dg, 'X+', istag, jstag, interporder) !7
      call lagrange_poly_interp(vel, ie + 2, js - 3, bd, dg, 'X+', istag, jstag, interporder) !8
      call lagrange_poly_interp(vel, ie + 2, js - 1, bd, dg, 'Y-', istag, jstag, interporder) !2
      call lagrange_poly_interp(vel, ie + 3, js - 1, bd, dg, 'Y-', istag, jstag, interporder) !3
      call lagrange_poly_interp(vel, ie + 3, js - 2, bd, dg, 'Y-', istag, jstag, interporder) !6

      veltemp = vel
      veltempp = vel

      !diag
      i = ie + 1
      j = js - 1
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X+', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y-', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
      i = ie + 3
      j = js - 3
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X+', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y-', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
      i = ie + 2
      j = js - 2
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X+', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y-', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
    end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! SW corner
!     -------------
!        1--2--3  |
!        |     |  |
!        4  5  6  |
!        |     |  |
!        7--8--9  |
!

    if (dg%rmp_sw) then
      call lagrange_poly_interp(vel, is - 1, js - 2, bd, dg, 'X-', istag, jstag, interporder) !6
      call lagrange_poly_interp(vel, is - 1, js - 3, bd, dg, 'X-', istag, jstag, interporder) !9
      call lagrange_poly_interp(vel, is - 2, js - 3, bd, dg, 'X-', istag, jstag, interporder) !8
      call lagrange_poly_interp(vel, is - 2, js - 1, bd, dg, 'Y-', istag, jstag, interporder) !2
      call lagrange_poly_interp(vel, is - 3, js - 1, bd, dg, 'Y-', istag, jstag, interporder) !1
      call lagrange_poly_interp(vel, is - 3, js - 2, bd, dg, 'Y-', istag, jstag, interporder) !4

      veltemp = vel
      veltempp = vel

      !diag
      i = is - 1
      j = js - 1
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X-', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y-', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
      i = is - 3
      j = js - 3
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X-', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y-', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
      i = is - 2
      j = js - 2
      call lagrange_poly_interp(veltemp, i, j, bd, dg, 'X-', istag, jstag, interporder)
      call lagrange_poly_interp(veltempp, i, j, bd, dg, 'Y-', istag, jstag, interporder)
      vel(i, j) = 0.5*(veltemp(i, j) + veltempp(i, j))
    end if

  end subroutine fill_corner_region_2d
  subroutine compute_lagrange_coeff(xp, xm, yp, ym, bd, dg)

    type(fv_grid_bounds_type), intent(IN) :: bd
    type(duogrid_type), intent(inout) :: dg
    real(kind=R_GRID), intent(out) :: xp(1:interporder + 1, bd%ie - interporder:bd%ied + 1, bd%jsd:bd%jed + 1, 4)
    real(kind=R_GRID), intent(out) :: xm(1:interporder + 1, bd%isd:bd%is + interporder, bd%jsd:bd%jed + 1, 4)
    real(kind=R_GRID), intent(out) :: yp(1:interporder + 1, bd%isd:bd%ied + 1, bd%je - interporder:bd%jed + 1, 4)
    real(kind=R_GRID), intent(out) :: ym(1:interporder + 1, bd%isd:bd%ied + 1, bd%jsd:bd%js + interporder, 4)

    !local
    integer ii, iii, it, jstag, istag
    integer is, js, ie, je, i, j
    real(kind=R_GRID) :: ss, S

    do istag = 0, 1
    do jstag = 0, 1
    do i = bd%ie - interporder, bd%ied + istag
    do j = bd%jsd, bd%jed + jstag
      is = bd%is
      js = bd%js
      !  ie = bd%ie + istag
      !  je = bd%je + jstag
      do ii = bd%ie - interporder, bd%ie
        ss = 1.0
        do iii = bd%ie - interporder, bd%ie
          if (ii < iii) it = -1
          if (ii > iii) it = 1
          if (ii .eq. iii) cycle
          if (j > bd%je - 1) then
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i - istag, j), dg%a_pt(:, iii, j))/ &
                        great_circle_dist(dg%a_pt(:, ii, j), dg%a_pt(:, iii, j)))
          else
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i - istag, j - jstag), dg%a_pt(:, iii, j - jstag))/ &
                        great_circle_dist(dg%a_pt(:, ii, j - jstag), dg%a_pt(:, iii, j - jstag)))
          end if
        end do
        xp(bd%ie - ii + 1, i, j, 2*istag + jstag + 1) = ss
      end do

    end do
    end do

    do i = bd%isd, bd%is + interporder
    do j = bd%jsd, bd%jed + jstag

      do ii = is + interporder, is, -1
        ss = 1.0
        do iii = is + interporder, is, -1
          if (ii < iii) it = 1
          if (ii > iii) it = -1
          if (ii .eq. iii) cycle
          if (j > bd%je - 1) then
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i, j), dg%a_pt(:, iii, j))/ &
                        great_circle_dist(dg%a_pt(:, ii, j), dg%a_pt(:, iii, j)))
          else
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i, j - jstag), dg%a_pt(:, iii, j - jstag))/ &
                        great_circle_dist(dg%a_pt(:, ii, j - jstag), dg%a_pt(:, iii, j - jstag)))
          end if

        end do
        xm(is + interporder - ii + 1, i, j, 2*istag + jstag + 1) = ss
      end do
    end do
    end do

    do i = bd%isd, bd%ied + istag
    do j = bd%je - interporder, bd%jed + jstag
      do ii = bd%je - interporder, bd%je
        ss = 1.0
        do iii = bd%je - interporder, bd%je
          if (ii < iii) it = -1
          if (ii > iii) it = 1
          if (ii .eq. iii) cycle
          if (i > bd%ie - 1) then
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i, j - jstag), dg%a_pt(:, i, iii))/ &
                        great_circle_dist(dg%a_pt(:, i, ii), dg%a_pt(:, i, iii)))
          else
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i - istag, j - jstag), dg%a_pt(:, i - istag, iii))/ &
                        great_circle_dist(dg%a_pt(:, i - istag, ii), dg%a_pt(:, i - istag, iii)))

          end if

        end do
        yp(bd%je - ii + 1, i, j, 2*istag + jstag + 1) = ss
      end do
    end do
    end do

    do i = bd%isd, bd%ied + istag
    do j = bd%jsd, bd%js + interporder
      do ii = js + interporder, js, -1
        ss = 1.0
        do iii = js + interporder, js, -1
          if (ii < iii) it = 1
          if (ii > iii) it = -1
          if (ii .eq. iii) cycle
          if (i > bd%ie - 1) then
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i, j), dg%a_pt(:, i, iii))/ &
                        great_circle_dist(dg%a_pt(:, i, ii), dg%a_pt(:, i, iii)))
          else
            ss = ss*it*(great_circle_dist(dg%a_pt(:, i - istag, j), dg%a_pt(:, i - istag, iii))/ &
                        great_circle_dist(dg%a_pt(:, i - istag, ii), dg%a_pt(:, i - istag, iii)))
          end if

        end do
        ym(js + interporder - ii + 1, i, j, 2*istag + jstag + 1) = ss
      end do
    end do
    end do
    end do
    end do

    if (mpp_pe() == mpp_root_pe()) print *, "DUO lag coeff: DONE!"
  end subroutine compute_lagrange_coeff
  subroutine lagrange_poly_interp_2d(field, i, j, bd, dg, direction, istag, jstag, order, corner)

    integer, intent(in) :: i, j, istag, jstag, order
    type(fv_grid_bounds_type), intent(IN) :: bd
    real, dimension(bd%isd:, bd%jsd:) :: field
    type(duogrid_type), target, intent(in) :: dg
    character(len=*), intent(in) :: direction
    logical, optional, intent(in) :: corner

    !local
    integer ii, iii, it
    integer is, js, ie, je
    real(kind=R_GRID) :: ss, S
    !real(kind=R_GRID) :: ss_store(5, 1:interporder+1, bd%isd:bd%ied+1, bd%jsd:bd%jed+1, 0:1,0:1)

    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: xp
    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: xm
    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: yp
    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: ym

    xp => dg%xp
    xm => dg%xm
    yp => dg%yp
    ym => dg%ym

    S = 0. ! here we sum all the polynomial*solution

    is = bd%is
    js = bd%js
    ie = bd%ie + istag
    je = bd%je + jstag
    it = 1

    call timing_on('lag')
    !call compute_lagrange_poly_interp_2d(ss_store, i, j, bd, dg, direction, istag, jstag, order, corner)

!TO check: compute weight coefficients a priori?!

    if (direction == 'X+') then

      do ii = bd%ie - interporder, bd%ie
        S = S + xp(bd%ie - ii + 1, i, j, 2*istag + jstag + 1)*field(ii + istag, j)
      end do

      field(i, j) = S

    end if

    if (direction == 'X-') then

      do ii = is + order, is, -1
        S = S + xm(is + order - ii + 1, i, j, 2*istag + jstag + 1)*field(ii, j)
      end do

      field(i, j) = S

    end if

    if (direction == 'Y+') then

      do ii = bd%je - order, bd%je
        S = S + yp(bd%je - ii + 1, i, j, 2*istag + jstag + 1)*field(i, ii + jstag)
      end do

      field(i, j) = S

    end if

    if (direction == 'Y-') then

      do ii = js + order, js, -1
        S = S + ym(js + order - ii + 1, i, j, 2*istag + jstag + 1)*field(i, ii)
      end do

      field(i, j) = S

    end if
    nullify (xp, xm, yp, ym)

    call timing_off('lag')
  end subroutine lagrange_poly_interp_2d
end module cornerlag_extract_mod
