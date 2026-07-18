! VERBATIM extract — BOUNDED-conventions gridstruct oracle (task #10).
! Blocks (do not edit bodies):
!   SYMMETRYCLEAN model/fv_grid_utils.F90 helpers:
!     984-996 inner_prod, 1628-1665 latlon2xyz2/latlon2xyz,
!     1781-1791 vect_cross, 1795-1845 get_center_vect,
!     1848-1863 get_unit_vect2, 1880-1893 normalize_vect,
!     1981-1992 mid_pt_sphere, 1996-2036 mid_pt3_cart/mid_pt_cart,
!     2040-2062 great_circle_dist, 2700-2725 cell_center2,
!     2749-2790 get_area, 2838-2942 spherical_angle/cos_angle,
!     3326-3340 get_latlon_vector
!   SYMMETRYCLEAN model/fv_grid_utils.F90:224-770 init angle/metric
!     section (as init_grid_utils_metrics; Atm% derefs bridged by
!     the shim type, bodies verbatim)
!   PLAIN-CLONE tools/fv_grid_tools.F90:725-1000 metric block
!     (as init_grid_metrics_tools; DOCUMENTED SUBSTITUTION — the
!     duo archive ships no fv_grid_tools; bounded machinery
!     predates duogrid) + 2397-2587 grid_area
module bounded_gs_extract_mod
  use bounded_gs_shim_mod
  implicit none
  public
contains
  real function inner_prod(v1, v2)
       real(kind=R_GRID),intent(in):: v1(3), v2(3)
       real (f_p) :: vp1(3), vp2(3), prod16
       integer k

         do k=1,3
            vp1(k) = real(v1(k),kind=f_p)
            vp2(k) = real(v2(k),kind=f_p)
         enddo
         prod16 = vp1(1)*vp2(1) + vp1(2)*vp2(2) + vp1(3)*vp2(3)
         inner_prod = prod16

  end function inner_prod
 subroutine latlon2xyz2(lon, lat, p3)
 real(kind=R_GRID), intent(in):: lon, lat
 real(kind=R_GRID), intent(out):: p3(3)
 real(kind=R_GRID) e(2)

    e(1) = lon;    e(2) = lat
    call latlon2xyz(e, p3)

 end subroutine latlon2xyz2


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
 subroutine get_center_vect( npx, npy, pp, u1, u2, bd )
   type(shim_bd_type), intent(IN) :: bd
    integer, intent(in):: npx, npy
    real(kind=R_GRID), intent(in) :: pp(3,bd%isd:bd%ied+1,bd%jsd:bd%jed+1)
    real(kind=R_GRID), intent(out):: u1(3,bd%isd:bd%ied,  bd%jsd:bd%jed)
    real(kind=R_GRID), intent(out):: u2(3,bd%isd:bd%ied,  bd%jsd:bd%jed)
! Local:
    integer i,j,k
    real(kind=R_GRID) p1(3), p2(3), pc(3), p3(3)

    integer :: isd, ied, jsd, jed

      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

    do j=jsd,jed
       do i=isd,ied
        if ( (i<1       .and. j<1  )     .or. (i>(npx-1) .and. j<1) .or.  &
             (i>(npx-1) .and. j>(npy-1)) .or. (i<1       .and. j>(npy-1))) then
             u1(1:3,i,j) = 0.d0
             u2(1:3,i,j) = 0.d0
        else
#ifdef OLD_VECT
          do k=1,3
             u1(k,i,j) = pp(k,i+1,j)+pp(k,i+1,j+1) - pp(k,i,j)-pp(k,i,j+1)
             u2(k,i,j) = pp(k,i,j+1)+pp(k,i+1,j+1) - pp(k,i,j)-pp(k,i+1,j)
          enddo
          call normalize_vect( u1(1,i,j) )
          call normalize_vect( u2(1,i,j) )
#else
          call cell_center3(pp(1,i,j), pp(1,i+1,j), pp(1,i,j+1), pp(1,i+1,j+1), pc)
! e1:
          call mid_pt3_cart(pp(1,i,j),   pp(1,i,j+1),   p1)
          call mid_pt3_cart(pp(1,i+1,j), pp(1,i+1,j+1), p2)
          call vect_cross(p3, p2, p1)
          call vect_cross(u1(1,i,j), pc, p3)
          call normalize_vect( u1(1,i,j) )
! e2:
          call mid_pt3_cart(pp(1,i,j),   pp(1,i+1,j),   p1)
          call mid_pt3_cart(pp(1,i,j+1), pp(1,i+1,j+1), p2)
          call vect_cross(p3, p2, p1)
          call vect_cross(u2(1,i,j), pc, p3)
          call normalize_vect( u2(1,i,j) )
#endif
        endif
       enddo
    enddo

 end subroutine get_center_vect
 subroutine get_unit_vect2( e1, e2, uc )
   real(kind=R_GRID), intent(in) :: e1(2), e2(2)
   real(kind=R_GRID), intent(out):: uc(3) ! unit vector e1--->e2
! Local:
   real(kind=R_GRID), dimension(3):: pc, p1, p2, p3

! RIGHT_HAND system:
   call latlon2xyz(e1, p1)
   call latlon2xyz(e2, p2)

   call mid_pt3_cart(p1, p2,  pc)
   call vect_cross(p3, p2, p1)
   call vect_cross(uc, pc, p3)
   call normalize_vect( uc )

 end subroutine get_unit_vect2
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
 subroutine mid_pt_sphere(p1, p2, pm)
      real(kind=R_GRID) , intent(IN)  :: p1(2), p2(2)
      real(kind=R_GRID) , intent(OUT) :: pm(2)
!------------------------------------------
      real(kind=R_GRID) e1(3), e2(3), e3(3)

      call latlon2xyz(p1, e1)
      call latlon2xyz(p2, e2)
      call mid_pt3_cart(e1, e2, e3)
      call cart_to_latlon(1, e3, pm(1), pm(2))

 end subroutine mid_pt_sphere
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
 subroutine cell_center2(q1, q2, q3, q4, e2)
      real(kind=R_GRID) , intent(in ) :: q1(2), q2(2), q3(2), q4(2)
      real(kind=R_GRID) , intent(out) :: e2(2)
! Local
      real(kind=R_GRID) p1(3), p2(3), p3(3), p4(3)
      real(kind=R_GRID) ec(3)
      real(kind=R_GRID) dd
      integer k

      call latlon2xyz(q1, p1)
      call latlon2xyz(q2, p2)
      call latlon2xyz(q3, p3)
      call latlon2xyz(q4, p4)

      do k=1,3
         ec(k) = p1(k) + p2(k) + p3(k) + p4(k)
      enddo
      dd = sqrt( ec(1)**2 + ec(2)**2 + ec(3)**2 )

      do k=1,3
         ec(k) = ec(k) / dd
      enddo

      call cart_to_latlon(1, ec, e2(1), e2(2))

 end subroutine cell_center2
 real(kind=R_GRID) function get_area(p1, p4, p2, p3, radius)
!-----------------------------------------------
 real(kind=R_GRID), intent(in), dimension(2):: p1, p2, p3, p4
 real(kind=R_GRID), intent(in), optional:: radius
!-----------------------------------------------
 real(kind=R_GRID) e1(3), e2(3), e3(3)
 real(kind=R_GRID) ang1, ang2, ang3, ang4

! S-W: 1
       call latlon2xyz(p1, e1)   ! p1
       call latlon2xyz(p2, e2)   ! p2
       call latlon2xyz(p4, e3)   ! p4
       ang1 = spherical_angle(e1, e2, e3)
!----
! S-E: 2
!----
       call latlon2xyz(p2, e1)
       call latlon2xyz(p3, e2)
       call latlon2xyz(p1, e3)
       ang2 = spherical_angle(e1, e2, e3)
!----
! N-E: 3
!----
       call latlon2xyz(p3, e1)
       call latlon2xyz(p4, e2)
       call latlon2xyz(p2, e3)
       ang3 = spherical_angle(e1, e2, e3)
!----
! N-W: 4
!----
       call latlon2xyz(p4, e1)
       call latlon2xyz(p3, e2)
       call latlon2xyz(p1, e3)
       ang4 = spherical_angle(e1, e2, e3)

       if ( present(radius) ) then
            get_area = (ang1 + ang2 + ang3 + ang4 - 2.*pi) * radius**2
       else
            get_area = ang1 + ang2 + ang3 + ang4 - 2.*pi
       endif

 end function get_area
 real(kind=R_GRID) function spherical_angle(p1, p2, p3)

!           p3
!         /
!        /
!       p1 ---> angle
!         \
!          \
!           p2

 real(kind=R_GRID) p1(3), p2(3), p3(3)

 real (f_p):: e1(3), e2(3), e3(3)
 real (f_p):: px, py, pz
 real (f_p):: qx, qy, qz
 real (f_p):: angle, ddd
 integer n

  do n=1,3
     e1(n) = p1(n)
     e2(n) = p2(n)
     e3(n) = p3(n)
  enddo

!-------------------------------------------------------------------
! Page 41, Silverman's book on Vector Algebra; spherical trigonmetry
!-------------------------------------------------------------------
! Vector P:
   px = e1(2)*e2(3) - e1(3)*e2(2)
   py = e1(3)*e2(1) - e1(1)*e2(3)
   pz = e1(1)*e2(2) - e1(2)*e2(1)
! Vector Q:
   qx = e1(2)*e3(3) - e1(3)*e3(2)
   qy = e1(3)*e3(1) - e1(1)*e3(3)
   qz = e1(1)*e3(2) - e1(2)*e3(1)

   ddd = (px*px+py*py+pz*pz)*(qx*qx+qy*qy+qz*qz)

   if ( ddd <= 0.0d0 ) then
        angle = 0.d0
   else
        ddd = (px*qx+py*qy+pz*qz) / sqrt(ddd)
        if ( abs(ddd)>1.d0) then
             angle = 2.d0*atan(1.0)    ! 0.5*pi
           !FIX (lmh) to correctly handle co-linear points (angle near pi or 0)
           if (ddd < 0.d0) then
              angle = 4.d0*atan(1.0d0) !should be pi
           else
              angle = 0.d0
           end if
        else
             angle = acos( ddd )
        endif
   endif

   spherical_angle = angle

 end function spherical_angle


 real(kind=R_GRID) function cos_angle(p1, p2, p3)
! As spherical_angle, but returns the cos(angle)
!       p3
!       ^
!       |
!       |
!       p1 ---> p2
!
 real(kind=R_GRID), intent(in):: p1(3), p2(3), p3(3)

 real (f_p):: e1(3), e2(3), e3(3)
 real (f_p):: px, py, pz
 real (f_p):: qx, qy, qz
 real (f_p):: angle, ddd
 integer n

  do n=1,3
     e1(n) = p1(n)
     e2(n) = p2(n)
     e3(n) = p3(n)
  enddo

!-------------------------------------------------------------------
! Page 41, Silverman's book on Vector Algebra; spherical trigonmetry
!-------------------------------------------------------------------
! Vector P:= e1 X e2
   px = e1(2)*e2(3) - e1(3)*e2(2)
   py = e1(3)*e2(1) - e1(1)*e2(3)
   pz = e1(1)*e2(2) - e1(2)*e2(1)

! Vector Q: e1 X e3
   qx = e1(2)*e3(3) - e1(3)*e3(2)
   qy = e1(3)*e3(1) - e1(1)*e3(3)
   qz = e1(1)*e3(2) - e1(2)*e3(1)

! ddd = sqrt[ (P*P) (Q*Q) ]
   ddd = sqrt( (px**2+py**2+pz**2)*(qx**2+qy**2+qz**2) )
   if ( ddd > 0.d0 ) then
        angle = (px*qx+py*qy+pz*qz) / ddd
   else
        angle = 1.d0
   endif
   cos_angle = angle

 end function cos_angle
 subroutine get_latlon_vector(pp, elon, elat)
 real(kind=R_GRID), intent(IN)  :: pp(2)
 real(kind=R_GRID), intent(OUT) :: elon(3), elat(3)

         elon(1) = -SIN(pp(1))
         elon(2) =  COS(pp(1))
         elon(3) =  0.0
         elat(1) = -SIN(pp(2))*COS(pp(1))
         elat(2) = -SIN(pp(2))*SIN(pp(1))
!!! RIGHT_HAND
         elat(3) =  COS(pp(2))
! Left-hand system needed to be consistent with rest of the codes
!        elat(3) = -COS(pp(2))

 end subroutine get_latlon_vector

   subroutine grid_utils_init(Atm, npx, npy, npz, non_ortho, grid_type, c2l_order)
! Initialize 2D memory and geometrical factors
      type(shim_atm_type), intent(inout), target :: Atm
      logical, intent(in):: non_ortho
      integer, intent(in):: npx, npy, npz
      integer, intent(in):: grid_type, c2l_order
!
! Super (composite) grid:

!     9---4---8
!     |       |
!     1   5   3
!     |       |
!     6---2---7

      real(kind=R_GRID) grid3(3,Atm%bd%isd:Atm%bd%ied+1,Atm%bd%jsd:Atm%bd%jed+1)
      real(kind=R_GRID) p1(3), p2(3), p3(3), p4(3), pp(3), ex(3), ey(3), e1(3), e2(3)
      real(kind=R_GRID) pp1(2), pp2(2), pp3(2)
      real(kind=R_GRID) sin2, tmp1, tmp2
      integer i, j, k, n, ip

      integer :: is,  ie,  js,  je
      integer :: isd, ied, jsd, jed

      !Local pointers
      real(kind=R_GRID), pointer, dimension(:,:,:) :: agrid, grid
      real(kind=R_GRID), pointer, dimension(:,:) :: area, area_c
      real(kind=R_GRID), pointer, dimension(:,:) :: sina, cosa, dx, dy, dxc, dyc, dxa, dya
      real, pointer, dimension(:,:) :: del6_u, del6_v
      real, pointer, dimension(:,:) :: divg_u, divg_v
      real, pointer, dimension(:,:) :: cosa_u, cosa_v, cosa_s
      real, pointer, dimension(:,:) :: sina_u, sina_v
      real, pointer, dimension(:,:) :: rsin_u, rsin_v
      real, pointer, dimension(:,:) :: rsina, rsin2
      real, pointer, dimension(:,:,:) :: sin_sg, cos_sg
      real(kind=R_GRID), pointer, dimension(:,:,:) :: ee1, ee2, ec1, ec2
      real(kind=R_GRID), pointer, dimension(:,:,:,:) :: ew, es
      real(kind=R_GRID), pointer, dimension(:,:,:) :: en1, en2
!     real(kind=R_GRID), pointer, dimension(:,:) :: eww, ess
      logical, pointer :: sw_corner, se_corner, ne_corner, nw_corner

      is  = Atm%bd%is
      ie  = Atm%bd%ie
      js  = Atm%bd%js
      je  = Atm%bd%je
      isd = Atm%bd%isd
      ied = Atm%bd%ied
      jsd = Atm%bd%jsd
      jed = Atm%bd%jed

!--- pointers to higher-order precision quantities
      agrid => Atm%gridstruct%agrid_64
      grid  => Atm%gridstruct%grid_64
      area    => Atm%gridstruct%area_64
      area_c  => Atm%gridstruct%area_c_64
      dx     => Atm%gridstruct%dx_64
      dy     => Atm%gridstruct%dy_64
      dxc    => Atm%gridstruct%dxc_64
      dyc    => Atm%gridstruct%dyc_64
      dxa    => Atm%gridstruct%dxa_64
      dya    => Atm%gridstruct%dya_64
      sina   => Atm%gridstruct%sina_64
      cosa   => Atm%gridstruct%cosa_64

      divg_u => Atm%gridstruct%divg_u
      divg_v => Atm%gridstruct%divg_v

      del6_u => Atm%gridstruct%del6_u
      del6_v => Atm%gridstruct%del6_v

      cosa_u => Atm%gridstruct%cosa_u
      cosa_v => Atm%gridstruct%cosa_v
      cosa_s => Atm%gridstruct%cosa_s
      sina_u => Atm%gridstruct%sina_u
      sina_v => Atm%gridstruct%sina_v
      rsin_u => Atm%gridstruct%rsin_u
      rsin_v => Atm%gridstruct%rsin_v
      rsina => Atm%gridstruct%rsina
      rsin2 => Atm%gridstruct%rsin2
      ee1 => Atm%gridstruct%ee1
      ee2 => Atm%gridstruct%ee2
      ec1 => Atm%gridstruct%ec1
      ec2 => Atm%gridstruct%ec2
      ew => Atm%gridstruct%ew
      es => Atm%gridstruct%es
      sin_sg => Atm%gridstruct%sin_sg
      cos_sg => Atm%gridstruct%cos_sg
      en1 => Atm%gridstruct%en1
      en2 => Atm%gridstruct%en2
!     eww => Atm%gridstruct%eww
!     ess => Atm%gridstruct%ess

      sw_corner                     => Atm%gridstruct%sw_corner
      se_corner                     => Atm%gridstruct%se_corner
      ne_corner                     => Atm%gridstruct%ne_corner
      nw_corner                     => Atm%gridstruct%nw_corner

      if ( (Atm%flagstruct%do_schmidt .or. Atm%flagstruct%do_cube_transform) .and. abs(Atm%flagstruct%stretch_fac-1.) > 1.E-5 ) then
           Atm%gridstruct%stretched_grid = .true.
           symm_grid = .false.
      else
      Atm%gridstruct%stretched_grid = .false.
           symm_grid = .true.
      endif

      if ( npz == 1 ) then
           Atm%ak(1) = 0.
           Atm%ak(2) = 0.
           Atm%bk(1) = 0.
           Atm%bk(2) = 1.
           Atm%ptop  = 0.
           Atm%ks    = 0
      elseif ( .not. Atm%flagstruct%hybrid_z ) then
! Initialize (ak,bk) for cold start; overwritten with restart file
           if (.not. Atm%flagstruct%external_eta) then
              call set_eta(npz, Atm%ks, Atm%ptop, Atm%ak, Atm%bk, Atm%flagstruct%npz_type)
              if ( is_master() ) then
                 write(*,*) 'Grid_init', npz, Atm%ks, Atm%ptop
                 tmp1 = Atm%ak(Atm%ks+1)
                 do k=Atm%ks+1,npz
                    tmp1 = max(tmp1, (Atm%ak(k)-Atm%ak(k+1))/max(1.E-9, (Atm%bk(k+1)-Atm%bk(k))) )
                 enddo
                 write(*,*) 'Hybrid Sigma-P: minimum allowable surface pressure (hpa)=', tmp1/100.
                 if ( tmp1 > 420.E2 ) write(*,*) 'Warning: the chosen setting in set_eta can cause instability'
              endif
           endif
      endif

! NCEP analysis available from amip-Interp (allocate if needed)
#ifndef DYCORE_SOLO
      if (.not. allocated(sst_ncep)) allocate (sst_ncep(i_sst,j_sst))
      if (.not. allocated(sst_anom)) allocate (sst_anom(i_sst,j_sst))
#endif


      cos_sg(:,:,:) =  big_number
      sin_sg(:,:,:) = tiny_number

      sw_corner = .false.
      se_corner = .false.
      ne_corner = .false.
      nw_corner = .false.

      if (grid_type < 3 .and. .not. Atm%gridstruct%bounded_domain) then
         if (       is==1 .and.  js==1 )      sw_corner = .true.
         if ( (ie+1)==npx .and.  js==1 )      se_corner = .true.
         if ( (ie+1)==npx .and. (je+1)==npy ) ne_corner = .true.
         if (       is==1 .and. (je+1)==npy ) nw_corner = .true.
      endif

  if ( sw_corner ) then
       tmp1 = great_circle_dist(grid(1,1,1:2), agrid(1,1,1:2))
       tmp2 = great_circle_dist(grid(1,1,1:2), agrid(2,2,1:2))
       write(*,*) 'Corner interpolation coefficient=', tmp2/(tmp2-tmp1)
  endif

  if (grid_type < 3) then
!xxx if ( .not. Atm%neststruct%nested ) then
     if ( .not. Atm%gridstruct%bounded_domain ) then
     call fill_corners(grid(:,:,1), npx, npy, FILL=XDir, BGRID=.true.)
     call fill_corners(grid(:,:,2), npx, npy, FILL=XDir, BGRID=.true.)
     end if

     do j=jsd,jed+1
        do i=isd,ied+1
           call latlon2xyz(grid(i,j,1:2), grid3(1,i,j))
        enddo
     enddo


     call get_center_vect( npx, npy, grid3, ec1, ec2, Atm%bd )

! Fill arbitrary values in the non-existing corner regions:
     if (.not. Atm%gridstruct%bounded_domain) then
     do k=1,3
        call fill_ghost(ec1(k,:,:), npx, npy, big_number, Atm%bd)
        call fill_ghost(ec2(k,:,:), npx, npy, big_number, Atm%bd)
     enddo
     end if


     do j=jsd,jed
        do i=isd+1,ied
        if ( ( (i<1   .and. j<1  ) .or. (i>npx .and. j<1  ) .or.  &
             (i>npx .and. j>(npy-1)) .or. (i<1   .and. j>(npy-1)) )  .and. .not. Atm%gridstruct%bounded_domain) then
             ew(1:3,i,j,1:2) = 0.
        else
           call mid_pt_cart( grid(i,j,1:2), grid(i,j+1,1:2), pp)
           if (i==1 .and. .not. Atm%gridstruct%bounded_domain) then
              call latlon2xyz( agrid(i,j,1:2), p1)
              call vect_cross(p2, pp, p1)
           elseif(i==npx .and. .not. Atm%gridstruct%bounded_domain) then
              call latlon2xyz( agrid(i-1,j,1:2), p1)
              call vect_cross(p2, p1, pp)
           else
              call latlon2xyz( agrid(i-1,j,1:2), p3)
              call latlon2xyz( agrid(i,  j,1:2), p1)
              call vect_cross(p2, p3, p1)
           endif
           call vect_cross(ew(1:3,i,j,1), p2, pp)
           call normalize_vect(ew(1:3,i,j,1))
!---
           call vect_cross(p1, grid3(1,i,j), grid3(1,i,j+1))
           call vect_cross(ew(1:3,i,j,2), p1, pp)
           call normalize_vect(ew(1:3,i,j,2))
        endif
        enddo
     enddo

     do j=jsd+1,jed
        do i=isd,ied
        if ( ( (i<1   .and. j<1  ) .or. (i>(npx-1) .and. j<1  ) .or.  &
               (i>(npx-1) .and. j>npy) .or. (i<1   .and. j>npy) ) .and. .not. Atm%gridstruct%bounded_domain) then
             es(1:3,i,j,1:2) = 0.
        else
           call mid_pt_cart(grid(i,j,1:2), grid(i+1,j,1:2), pp)
           if (j==1 .and. .not. Atm%gridstruct%bounded_domain) then
              call latlon2xyz( agrid(i,j,1:2), p1)
              call vect_cross(p2, pp, p1)
           elseif (j==npy .and. .not. Atm%gridstruct%bounded_domain) then
              call latlon2xyz( agrid(i,j-1,1:2), p1)
              call vect_cross(p2, p1, pp)
           else
              call latlon2xyz( agrid(i,j  ,1:2), p1)
              call latlon2xyz( agrid(i,j-1,1:2), p3)
              call vect_cross(p2, p3, p1)
           endif
           call vect_cross(es(1:3,i,j,2), p2, pp)
           call normalize_vect(es(1:3,i,j,2))
!---
           call vect_cross(p3, grid3(1,i,j), grid3(1,i+1,j))
           call vect_cross(es(1:3,i,j,1), p3, pp)
           call normalize_vect(es(1:3,i,j,1))
        endif
        enddo
     enddo

!     9---4---8
!     |       |
!     1   5   3
!     |       |
!     6---2---7

      do j=jsd,jed
         do i=isd,ied
! Testing using spherical formular: exact if coordinate lines are along great circles
! SW corner:
            cos_sg(i,j,6) = cos_angle( grid3(1,i,j), grid3(1,i+1,j), grid3(1,i,j+1) )
! SE corner:
            cos_sg(i,j,7) = -cos_angle( grid3(1,i+1,j), grid3(1,i,j), grid3(1,i+1,j+1) )
! NE corner:
            cos_sg(i,j,8) = cos_angle( grid3(1,i+1,j+1), grid3(1,i+1,j), grid3(1,i,j+1) )
! NW corner:
            cos_sg(i,j,9) = -cos_angle( grid3(1,i,j+1), grid3(1,i,j), grid3(1,i+1,j+1) )
! Mid-points by averaging:
!!!         cos_sg(i,j,1) = 0.5*( cos_sg(i,j,6) + cos_sg(i,j,9) )
!!!         cos_sg(i,j,2) = 0.5*( cos_sg(i,j,6) + cos_sg(i,j,7) )
!!!         cos_sg(i,j,3) = 0.5*( cos_sg(i,j,7) + cos_sg(i,j,8) )
!!!         cos_sg(i,j,4) = 0.5*( cos_sg(i,j,8) + cos_sg(i,j,9) )
!!!!!       cos_sg(i,j,5) = 0.25*(cos_sg(i,j,6)+cos_sg(i,j,7)+cos_sg(i,j,8)+cos_sg(i,j,9))
! No averaging -----
            call latlon2xyz(agrid(i,j,1:2), p3)   ! righ-hand system consistent with grid3
               call mid_pt3_cart(grid3(1,i,j), grid3(1,i,j+1), p1)
            cos_sg(i,j,1) = cos_angle( p1, p3, grid3(1,i,j+1) )
               call mid_pt3_cart(grid3(1,i,j), grid3(1,i+1,j), p1)
            cos_sg(i,j,2) = cos_angle( p1, grid3(1,i+1,j), p3 )
               call mid_pt3_cart(grid3(1,i+1,j), grid3(1,i+1,j+1), p1)
            cos_sg(i,j,3) = cos_angle( p1, p3, grid3(1,i+1,j) )
               call mid_pt3_cart(grid3(1,i,j+1), grid3(1,i+1,j+1), p1)
            cos_sg(i,j,4) = cos_angle( p1, grid3(1,i,j+1), p3 )
! Center point:
! Using center_vect: [ec1, ec2]
            cos_sg(i,j,5) = inner_prod( ec1(1:3,i,j), ec2(1:3,i,j) )
         enddo
      enddo

      do ip=1,9
         do j=jsd,jed
            do i=isd,ied
               sin_sg(i,j,ip) = min(1.0, sqrt( max(0., 1.-cos_sg(i,j,ip)**2) ) )
            enddo
         enddo
      enddo

! -------------------------------
! For transport operation
! -------------------------------
!xxx  if (.not. Atm%neststruct%nested) then
      if (.not. Atm%gridstruct%bounded_domain) then
      if ( sw_corner ) then
           do i=-2,0
              sin_sg(0,i,3) = sin_sg(i,1,2)
              sin_sg(i,0,4) = sin_sg(1,i,1)
           enddo
      endif
      if ( nw_corner ) then
           do i=npy,npy+2
              sin_sg(0,i,3) = sin_sg(npy-i,npy-1,4)
           enddo
           do i=-2,0
              sin_sg(i,npy,2) = sin_sg(1,npx+i,1)
           enddo
      endif
      if ( se_corner ) then
           do j=-2,0
              sin_sg(npx,j,1) = sin_sg(npx-j,1,2)
           enddo
           do i=npx,npx+2
              sin_sg(i,0,4) = sin_sg(npx-1,npx-i,3)
           enddo
      endif
      if ( ne_corner ) then
           do i=npy,npy+2
              sin_sg(npx,i,1) = sin_sg(i,npy-1,4)
              sin_sg(i,npy,2) = sin_sg(npx-1,i,3)
           enddo
        endif
     endif

! For AAM correction:
     do j=js,je
        do i=is,ie+1
           pp1(:) = grid(i  ,j ,1:2)
           pp2(:) = grid(i,j+1 ,1:2)
           call mid_pt_sphere(pp1, pp2, pp3)
           call get_unit_vect2(pp1, pp2, e2)
           call get_latlon_vector(pp3, ex, ey)
           Atm%gridstruct%l2c_v(i,j) = cos(pp3(2)) * inner_prod(e2, ex)
        enddo
     enddo
     do j=js,je+1
        do i=is,ie
           pp1(:) = grid(i,  j,1:2)
           pp2(:) = grid(i+1,j,1:2)
           call mid_pt_sphere(pp1, pp2, pp3)
           call get_unit_vect2(pp1, pp2, e1)
           call get_latlon_vector(pp3, ex, ey)
           Atm%gridstruct%l2c_u(i,j) = cos(pp3(2)) * inner_prod(e1, ex)
        enddo
     enddo

   else
     cos_sg(:,:,:) = 0.
     sin_sg(:,:,:) = 1.

     ec1(1,:,:)=1.
     ec1(2,:,:)=0.
     ec1(3,:,:)=0.

     ec2(1,:,:)=0.
     ec2(2,:,:)=1.
     ec2(3,:,:)=0.

     ew(1,:,:,1)=1.
     ew(2,:,:,1)=0.
     ew(3,:,:,1)=0.

     ew(1,:,:,2)=0.
     ew(2,:,:,2)=1.
     ew(3,:,:,2)=0.

     es(1,:,:,1)=1.
     es(2,:,:,1)=0.
     es(3,:,:,1)=0.

     es(1,:,:,2)=0.
     es(2,:,:,2)=1.
     es(3,:,:,2)=0.
  endif

   if ( non_ortho ) then
           cosa_u = big_number
           cosa_v = big_number
           cosa_s = big_number
           sina_u = big_number
           sina_v = big_number
           rsin_u = big_number
           rsin_v = big_number
           rsina  = big_number
           rsin2  = big_number
           cosa = big_number
           sina = big_number

        do j=js,je+1
           do i=is,ie+1
! unit vect in X-dir: ee1
              if (i==1 .and. .not. Atm%gridstruct%bounded_domain) then
                  call vect_cross(pp, grid3(1,i,  j), grid3(1,i+1,j))
              elseif(i==npx .and. .not. Atm%gridstruct%bounded_domain) then
                  call vect_cross(pp, grid3(1,i-1,j), grid3(1,i,  j))
              else
                  call vect_cross(pp, grid3(1,i-1,j), grid3(1,i+1,j))
              endif
              call vect_cross(ee1(1:3,i,j), pp, grid3(1:3,i,j))
              call normalize_vect( ee1(1:3,i,j) )

! unit vect in Y-dir: ee2
              if (j==1 .and. .not. Atm%gridstruct%bounded_domain) then
                  call vect_cross(pp, grid3(1:3,i,j  ), grid3(1:3,i,j+1))
              elseif(j==npy .and. .not. Atm%gridstruct%bounded_domain) then
                  call vect_cross(pp, grid3(1:3,i,j-1), grid3(1:3,i,j  ))
              else
                  call vect_cross(pp, grid3(1:3,i,j-1), grid3(1:3,i,j+1))
              endif
              call vect_cross(ee2(1:3,i,j), pp, grid3(1:3,i,j))
              call normalize_vect( ee2(1:3,i,j) )

! symmetrical grid
#ifdef TEST_FP
              tmp1 = inner_prod(ee1(1:3,i,j), ee2(1:3,i,j))
              cosa(i,j) = sign(min(1., abs(tmp1)), tmp1)
              sina(i,j) = sqrt(max(0.,1. -cosa(i,j)**2))
#else
              cosa(i,j) = 0.5*(cos_sg(i-1,j-1,8)+cos_sg(i,j,6))
              sina(i,j) = 0.5*(sin_sg(i-1,j-1,8)+sin_sg(i,j,6))
#endif
           enddo
        enddo

!     9---4---8
!     |       |
!     1   5   3
!     |       |
!     6---2---7
      do j=jsd,jed
         do i=isd+1,ied
            cosa_u(i,j) = 0.5*(cos_sg(i-1,j,3)+cos_sg(i,j,1))
            sina_u(i,j) = 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))
!           rsin_u(i,j) =  1. / sina_u(i,j)**2
            rsin_u(i,j) =  1. / max(tiny_number, sina_u(i,j)**2)
         enddo
      enddo
      do j=jsd+1,jed
         do i=isd,ied
            cosa_v(i,j) = 0.5*(cos_sg(i,j-1,4)+cos_sg(i,j,2))
            sina_v(i,j) = 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
!           rsin_v(i,j) =  1. / sina_v(i,j)**2
            rsin_v(i,j) =  1. / max(tiny_number, sina_v(i,j)**2)
         enddo
      enddo

      do j=jsd,jed
         do i=isd,ied
            cosa_s(i,j) = cos_sg(i,j,5)
!           rsin2(i,j) = 1. / sin_sg(i,j,5)**2
            rsin2(i,j) = 1. / max(tiny_number, sin_sg(i,j,5)**2)
         enddo
      enddo
! Force the model to fail if incorrect corner values are to be used:
      if (.not. Atm%gridstruct%bounded_domain) then
         call fill_ghost(cosa_s, npx, npy,  big_number, Atm%bd)
      end if
!------------------------------------
! Set special sin values at edges:
!------------------------------------
      do j=js,je+1
         do i=is,ie+1
            if ( i==npx .and. j==npy .and. .not. Atm%gridstruct%bounded_domain) then
            else if ( ( i==1 .or. i==npx .or. j==1 .or. j==npy ) .and. .not. Atm%gridstruct%bounded_domain ) then
                 rsina(i,j) = big_number
            else
!                rsina(i,j) = 1. / sina(i,j)**2
                 rsina(i,j) = 1. / max(tiny_number, sina(i,j)**2)
            endif
         enddo
      enddo

      do j=jsd,jed
         do i=is,ie+1
            if ( (i==1 .or. i==npx)  .and. .not. Atm%gridstruct%bounded_domain ) then
!                rsin_u(i,j) = 1. / sina_u(i,j)
                 rsin_u(i,j) = 1. / sign(max(tiny_number,abs(sina_u(i,j))), sina_u(i,j))
            endif
         enddo
      enddo

      do j=js,je+1
         do i=isd,ied
            if ( (j==1 .or. j==npy) .and. .not. Atm%gridstruct%bounded_domain ) then
!                rsin_v(i,j) = 1. / sina_v(i,j)
                 rsin_v(i,j) = 1. / sign(max(tiny_number,abs(sina_v(i,j))), sina_v(i,j))
            endif
         enddo
      enddo

      !EXPLANATION HERE: calling fill_ghost overwrites **SOME** of the sin_sg
      !values along the outward-facing edge of a tile in the corners, which is incorrect.
      !What we will do is call fill_ghost and then fill in the appropriate values

      if (.not. Atm%gridstruct%bounded_domain) then
     do k=1,9
        call fill_ghost(sin_sg(:,:,k), npx, npy, tiny_number, Atm%bd)  ! this will cause NAN if used
        call fill_ghost(cos_sg(:,:,k), npx, npy, big_number, Atm%bd)
     enddo
     end if

! -------------------------------
! For transport operation
! -------------------------------
      if ( sw_corner ) then
           do i=0,-2,-1
              sin_sg(0,i,3) = sin_sg(i,1,2)
              sin_sg(i,0,4) = sin_sg(1,i,1)
              cos_sg(0,i,3) = cos_sg(i,1,2)
              cos_sg(i,0,4) = cos_sg(1,i,1)
!!!           cos_sg(0,i,7) = cos_sg(i,1,6)
!!!           cos_sg(0,i,8) = cos_sg(i,1,7)
!!!           cos_sg(i,0,8) = cos_sg(1,i,9)
!!!           cos_sg(i,0,9) = cos_sg(1,i,6)
           enddo
!!!        cos_sg(0,0,8) = 0.5*(cos_sg(0,1,7)+cos_sg(1,0,9))

      endif
      if ( nw_corner ) then
           do i=npy,npy+2
              sin_sg(0,i,3) = sin_sg(npy-i,npy-1,4)
              cos_sg(0,i,3) = cos_sg(npy-i,npy-1,4)
!!!           cos_sg(0,i,7) = cos_sg(npy-i,npy-1,8)
!!!           cos_sg(0,i,8) = cos_sg(npy-i,npy-1,9)
           enddo
           do i=0,-2,-1
              sin_sg(i,npy,2) = sin_sg(1,npy-i,1)
              cos_sg(i,npy,2) = cos_sg(1,npy-i,1)
!!!           cos_sg(i,npy,6) = cos_sg(1,npy-i,9)
!!!           cos_sg(i,npy,7) = cos_sg(1,npy-i,6)
           enddo
!!!        cos_sg(0,npy,7) = 0.5*(cos_sg(1,npy,6)+cos_sg(0,npy-1,8))
      endif
      if ( se_corner ) then
           do j=0,-2,-1
              sin_sg(npx,j,1) = sin_sg(npx-j,1,2)
              cos_sg(npx,j,1) = cos_sg(npx-j,1,2)
!!!           cos_sg(npx,j,6) = cos_sg(npx-j,1,7)
!!!           cos_sg(npx,j,9) = cos_sg(npx-j,1,6)
           enddo
           do i=npx,npx+2
              sin_sg(i,0,4) = sin_sg(npx-1,npx-i,3)
              cos_sg(i,0,4) = cos_sg(npx-1,npx-i,3)
!!!           cos_sg(i,0,9) = cos_sg(npx-1,npx-i,8)
!!!           cos_sg(i,0,8) = cos_sg(npx-1,npx-i,7)
           enddo
!!!        cos_sg(npx,0,9) = 0.5*(cos_sg(npx,1,6)+cos_sg(npx-1,0,8))
      endif
      if ( ne_corner ) then
         do i=0,2
            sin_sg(npx,npy+i,1) = sin_sg(npx+i,npy-1,4)
            sin_sg(npx+i,npy,2) = sin_sg(npx-1,npy+i,3)
            cos_sg(npx,npy+i,1) = cos_sg(npx+i,npy-1,4)
!!!         cos_sg(npx,npy+i,6) = cos_sg(npx+i,npy-1,9)
!!!         cos_sg(npx,npy+i,9) = cos_sg(npx+i,npy-1,8)
            cos_sg(npx+i,npy,2) = cos_sg(npx-1,npy+i,3)
!!!         cos_sg(npx+i,npy,6) = cos_sg(npx-1,npy+i,7)
!!!         cos_sg(npx+i,npy,7) = cos_sg(npx-1,npy+i,8)
         end do
!!!      cos_sg(npx,npy,6) = 0.5*(cos_sg(npx-1,npy,7)+cos_sg(npx,npy-1,9))
      endif

   else
           sina = 1.
           cosa = 0.
           rsina  = 1.
           rsin2  = 1.
           sina_u = 1.
           sina_v = 1.
           cosa_u = 0.
           cosa_v = 0.
           cosa_s = 0.
           rsin_u = 1.
           rsin_v = 1.
   endif

   if ( grid_type < 3 ) then

#ifdef USE_NORM_VECT
!-------------------------------------------------------------
! Make normal vect at face edges after consines are computed:
!-------------------------------------------------------------
! for old d2a2c_vect routines
      if (.not. Atm%gridstruct%bounded_domain) then
         do j=js-1,je+1
            if ( is==1 ) then
               i=1
               call vect_cross(ew(1,i,j,1), grid3(1,i,j+1), grid3(1,i,j))
               call normalize_vect( ew(1,i,j,1) )
            endif
            if ( (ie+1)==npx ) then
               i=npx
               call vect_cross(ew(1,i,j,1), grid3(1,i,j+1), grid3(1,i,j))
               call normalize_vect( ew(1,i,j,1) )
            endif
         enddo

         if ( js==1 ) then
            j=1
            do i=is-1,ie+1
               call vect_cross(es(1,i,j,2), grid3(1,i,j),grid3(1,i+1,j))
               call normalize_vect( es(1,i,j,2) )
            enddo
         endif
         if ( (je+1)==npy ) then
            j=npy
            do i=is-1,ie+1
               call vect_cross(es(1,i,j,2), grid3(1,i,j),grid3(1,i+1,j))
               call normalize_vect( es(1,i,j,2) )
            enddo
         endif
      endif
#endif

! For omega computation:
! Unit vectors:
     do j=js,je+1
        do i=is,ie
           call vect_cross(en1(1:3,i,j), grid3(1,i,j), grid3(1,i+1,j))
           call normalize_vect( en1(1:3,i,j) )
        enddo
     enddo
     do j=js,je
        do i=is,ie+1
           call vect_cross(en2(1:3,i,j), grid3(1,i,j+1), grid3(1,i,j))
           call normalize_vect( en2(1:3,i,j) )
        enddo
     enddo
!-------------------------------------------------------------
! Make unit vectors for the coordinate extension:
!-------------------------------------------------------------
  endif

  do j=jsd,jed+1
     if ((j==1 .OR. j==npy) .and. .not. Atm%gridstruct%bounded_domain) then
        do i=isd,ied
           divg_u(i,j) = 0.5*(sin_sg(i,j,2)+sin_sg(i,j-1,4))*dyc(i,j)/dx(i,j)
           del6_u(i,j) = 0.5*(sin_sg(i,j,2)+sin_sg(i,j-1,4))*dx(i,j)/dyc(i,j)
        enddo
     else
        do i=isd,ied
           divg_u(i,j) = sina_v(i,j)*dyc(i,j)/dx(i,j)
           del6_u(i,j) = sina_v(i,j)*dx(i,j)/dyc(i,j)
        enddo
     end if
  enddo
  do j=jsd,jed
     do i=isd,ied+1
        divg_v(i,j) = sina_u(i,j)*dxc(i,j)/dy(i,j)
        del6_v(i,j) = sina_u(i,j)*dy(i,j)/dxc(i,j)
     enddo
     if (is == 1 .and. .not. Atm%gridstruct%bounded_domain) then
         divg_v(is,j) = 0.5*(sin_sg(1,j,1)+sin_sg(0,j,3))*dxc(is,j)/dy(is,j)
         del6_v(is,j) = 0.5*(sin_sg(1,j,1)+sin_sg(0,j,3))*dy(is,j)/dxc(is,j)
     endif
     if (ie+1 == npx .and. .not. Atm%gridstruct%bounded_domain) then
         divg_v(ie+1,j) = 0.5*(sin_sg(npx,j,1)+sin_sg(npx-1,j,3))*dxc(ie+1,j)/dy(ie+1,j)
         del6_v(ie+1,j) = 0.5*(sin_sg(npx,j,1)+sin_sg(npx-1,j,3))*dy(ie+1,j)/dxc(ie+1,j)
     endif
  enddo

! Initialize cubed_sphere to lat-lon transformation:
     call init_cubed_to_latlon( Atm%gridstruct, Atm%flagstruct%hydrostatic, agrid, grid_type, c2l_order, Atm%bd )

     call global_mx(area, Atm%ng, Atm%gridstruct%da_min, Atm%gridstruct%da_max, Atm%bd)
     if( is_master() ) write(*,*) 'da_max/da_min=', Atm%gridstruct%da_max/Atm%gridstruct%da_min

     call global_mx_c(area_c(is:ie,js:je), is, ie, js, je, Atm%gridstruct%da_min_c, Atm%gridstruct%da_max_c)

     if( is_master() ) write(*,*) 'da_max_c, da_min_c, da_max_c/da_min_c=', Atm%gridstruct%da_max_c, Atm%gridstruct%da_min_c, Atm%gridstruct%da_max_c/Atm%gridstruct%da_min_c

!------------------------------------------------
! Initialization for interpolation at face edges
!------------------------------------------------
! A->B scalar:
     if (grid_type < 3 .and. .not. Atm%gridstruct%bounded_domain ) then
        call mpp_update_domains(divg_v, divg_u, Atm%domain, flags=SCALAR_PAIR,      &
                                gridtype=CGRID_NE_PARAM, complete=.true.)
        call mpp_update_domains(del6_v, del6_u, Atm%domain, flags=SCALAR_PAIR,      &
                                gridtype=CGRID_NE_PARAM, complete=.true.)
        call edge_factors (Atm%gridstruct%edge_s, Atm%gridstruct%edge_n, Atm%gridstruct%edge_w, &
             Atm%gridstruct%edge_e, non_ortho, grid, agrid, npx, npy, Atm%bd)
        call efactor_a2c_v(Atm%gridstruct%edge_vect_s, Atm%gridstruct%edge_vect_n, &
             Atm%gridstruct%edge_vect_w, Atm%gridstruct%edge_vect_e, &
             non_ortho, grid, agrid, npx, npy, Atm%gridstruct%bounded_domain, Atm%bd)
!       call extend_cube_s(non_ortho, grid, agrid, npx, npy, .false., Atm%neststruct%nested)
!       call van2d_init(grid, agrid, npx, npy)
     else

        Atm%gridstruct%edge_s = big_number
        Atm%gridstruct%edge_n = big_number
        Atm%gridstruct%edge_w = big_number
        Atm%gridstruct%edge_e = big_number

        Atm%gridstruct%edge_vect_s = big_number
        Atm%gridstruct%edge_vect_n = big_number
        Atm%gridstruct%edge_vect_w = big_number
        Atm%gridstruct%edge_vect_e = big_number

     endif

!32-bit versions of the data
      Atm%gridstruct%grid   = Atm%gridstruct%grid_64
      Atm%gridstruct%agrid  = Atm%gridstruct%agrid_64
      Atm%gridstruct%area   = Atm%gridstruct%area_64
      Atm%gridstruct%area_c = Atm%gridstruct%area_c_64
      Atm%gridstruct%dx     = Atm%gridstruct%dx_64
      Atm%gridstruct%dy     = Atm%gridstruct%dy_64
      Atm%gridstruct%dxa    = Atm%gridstruct%dxa_64
      Atm%gridstruct%dya    = Atm%gridstruct%dya_64
      Atm%gridstruct%dxc    = Atm%gridstruct%dxc_64
      Atm%gridstruct%dyc    = Atm%gridstruct%dyc_64
      Atm%gridstruct%cosa   = Atm%gridstruct%cosa_64
      Atm%gridstruct%sina   = Atm%gridstruct%sina_64

!--- deallocate the higher-order gridstruct arrays
!rab      deallocate ( Atm%gridstruct%grid_64 )
!rab      deallocate ( Atm%gridstruct%agrid_64 )
!rab      deallocate ( Atm%gridstruct%area_64 )
      deallocate ( Atm%gridstruct%area_c_64 )
!rab      deallocate ( Atm%gridstruct%dx_64 )
!rab      deallocate ( Atm%gridstruct%dy_64 )
      deallocate ( Atm%gridstruct%dxa_64 )
      deallocate ( Atm%gridstruct%dya_64 )
      deallocate ( Atm%gridstruct%dxc_64 )
      deallocate ( Atm%gridstruct%dyc_64 )
      deallocate ( Atm%gridstruct%cosa_64 )
      deallocate ( Atm%gridstruct%sina_64 )

      nullify(agrid)
      nullify(grid)
      nullify(area)
      nullify(area_c)
      nullify(dx)
      nullify(dy)
      nullify(dxc)
      nullify(dyc)
      nullify(dxa)
      nullify(dya)
      nullify(sina)
      nullify(cosa)
      nullify(divg_u)
      nullify(divg_v)

      nullify(del6_u)
      nullify(del6_v)

      nullify(cosa_u)
      nullify(cosa_v)
      nullify(cosa_s)
      nullify(sina_u)
      nullify(sina_v)
      nullify(rsin_u)
      nullify(rsin_v)
      nullify(rsina)
      nullify(rsin2)
      nullify(ee1)
      nullify(ee2)
      nullify(ec1)
      nullify(ec2)
      nullify(ew)
      nullify(es)
      nullify(sin_sg)
      nullify(cos_sg)
      nullify(en1)
      nullify(en2)
      nullify(sw_corner)
      nullify(se_corner)
      nullify(ne_corner)
      nullify(nw_corner)

  end subroutine grid_utils_init

 subroutine efactor_a2c_v(edge_vect_s, edge_vect_n, edge_vect_w, edge_vect_e, non_ortho, grid, agrid, npx, npy, bounded_domain, bd)
!
! Initialization of interpolation factors at face edges
! for interpolating vectors from A to C grid
!
 type(shim_bd_type), intent(IN) :: bd
 real(kind=R_GRID),    intent(INOUT), dimension(bd%isd:bd%ied) :: edge_vect_s, edge_vect_n
 real(kind=R_GRID),    intent(INOUT), dimension(bd%jsd:bd%jed) :: edge_vect_w, edge_vect_e
 logical, intent(in):: non_ortho, bounded_domain
 real(kind=R_GRID),    intent(in)::  grid(bd%isd:bd%ied+1,bd%jsd:bd%jed+1,2)
 real(kind=R_GRID),    intent(in):: agrid(bd%isd:bd%ied  ,bd%jsd:bd%jed  ,2)
 integer, intent(in):: npx, npy

 real(kind=R_GRID) px(2,bd%isd:bd%ied+1),  py(2,bd%jsd:bd%jed+1)
 real(kind=R_GRID) p1(2,bd%isd:bd%ied+1),  p2(2,bd%jsd:bd%jed+1)       ! mid-point
 real(kind=R_GRID) d1, d2
 integer i, j
 integer im2, jm2

 integer :: is,  ie,  js,  je
 integer :: isd, ied, jsd, jed

 is  = bd%is
 ie  = bd%ie
 js  = bd%js
 je  = bd%je
 isd = bd%isd
 ied = bd%ied
 jsd = bd%jsd
 jed = bd%jed


  if ( .not. non_ortho ) then
     edge_vect_s = 0.
     edge_vect_n = 0.
     edge_vect_w = 0.
     edge_vect_e = 0.
  else
     edge_vect_s = big_number
     edge_vect_n = big_number
     edge_vect_w = big_number
     edge_vect_e = big_number

     if ( npx /= npy .and. .not. (bounded_domain)) call mpp_error(FATAL, 'efactor_a2c_v: npx /= npy')
     if ( (npx/2)*2 == npx ) call mpp_error(FATAL, 'efactor_a2c_v: npx/npy is not an odd number')

     im2 = (npx-1)/2
     jm2 = (npy-1)/2

 if ( is==1 ) then
    i=1
    do j=js-2,je+2
       call mid_pt_sphere(agrid(i-1,j,1:2), agrid(i,j,  1:2), py(1,j))
       call mid_pt_sphere( grid(i,  j,1:2),  grid(i,j+1,1:2), p2(1,j))
    enddo

! west edge:
!------------------------------------------------------------------
! v_sw(j) = (1.-edge_vect_w(j)) * p(j) + edge_vect_w(j) * p(j+1)
!------------------------------------------------------------------
    do j=js-1,je+1
       if ( j<=jm2 ) then
            d1 = great_circle_dist( py(1,j  ), p2(1,j) )
            d2 = great_circle_dist( py(1,j+1), p2(1,j) )
            edge_vect_w(j) = d1 / ( d1 + d2 )
       else
            d2 = great_circle_dist( py(1,j-1), p2(1,j) )
            d1 = great_circle_dist( py(1,j  ), p2(1,j) )
            edge_vect_w(j) = d1 / ( d2 + d1 )
       endif
    enddo
    if ( js==1 ) then
         edge_vect_w(0) = edge_vect_w(1)
    endif
    if ( (je+1)==npy ) then
         edge_vect_w(npy) = edge_vect_w(je)
    endif
    do j=js-1,je+1
!      if ( is_master() ) write(*,*) j, edge_vect_w(j)
    enddo
 endif

 if ( (ie+1)==npx ) then
    i=npx
    do j=jsd,jed
       call mid_pt_sphere(agrid(i-1,j,1:2), agrid(i,j,  1:2), py(1,j))
       call mid_pt_sphere( grid(i,  j,1:2),  grid(i,j+1,1:2), p2(1,j))
    enddo

    do j=js-1,je+1
       if ( j<=jm2 ) then
            d1 = great_circle_dist( py(1,j  ), p2(1,j) )
            d2 = great_circle_dist( py(1,j+1), p2(1,j) )
            edge_vect_e(j) = d1 / ( d1 + d2 )
       else
            d2 = great_circle_dist( py(1,j-1), p2(1,j) )
            d1 = great_circle_dist( py(1,j  ), p2(1,j) )
            edge_vect_e(j) = d1 / ( d2 + d1 )
       endif
    enddo
    if ( js==1 ) then
         edge_vect_e(0) = edge_vect_e(1)
    endif
    if ( (je+1)==npy ) then
         edge_vect_e(npy) = edge_vect_e(je)
    endif
    do j=js-1,je+1
!      if ( is_master() ) write(*,*) j, edge_vect_e(j)
    enddo
 endif

 if ( js==1 ) then
    j=1
    do i=isd,ied
       call mid_pt_sphere(agrid(i,j-1,1:2), agrid(i,  j,1:2), px(1,i))
       call mid_pt_sphere( grid(i,j,  1:2),  grid(i+1,j,1:2), p1(1,i))
    enddo
! south_west edge:
!------------------------------------------------------------------
! v_s(i) = (1.-edge_vect_s(i)) * p(i) + edge_vect_s(i) * p(i+1)
!------------------------------------------------------------------
    do i=is-1,ie+1
       if ( i<=im2 ) then
            d1 = great_circle_dist( px(1,i  ), p1(1,i) )
            d2 = great_circle_dist( px(1,i+1), p1(1,i) )
            edge_vect_s(i) = d1 / ( d1 + d2 )
       else
            d2 = great_circle_dist( px(1,i-1), p1(1,i) )
            d1 = great_circle_dist( px(1,i  ), p1(1,i) )
            edge_vect_s(i) = d1 / ( d2 + d1 )
       endif
    enddo
    if ( is==1 ) then
         edge_vect_s(0) = edge_vect_s(1)
    endif
    if ( (ie+1)==npx ) then
         edge_vect_s(npx) = edge_vect_s(ie)
    endif
    do i=is-1,ie+1
!      if ( is_master() ) write(*,*) i, edge_vect_s(i)
    enddo
 endif


 if ( (je+1)==npy ) then
! v_n(i) = (1.-edge_vect_n(i)) * p(i) + edge_vect_n(i) * p(i+1)
    j=npy
    do i=isd,ied
       call mid_pt_sphere(agrid(i,j-1,1:2), agrid(i,  j,1:2), px(1,i))
       call mid_pt_sphere( grid(i,j,  1:2),  grid(i+1,j,1:2), p1(1,i))
    enddo

    do i=is-1,ie+1
       if ( i<=im2 ) then
            d1 = great_circle_dist( px(1,i  ), p1(1,i) )
            d2 = great_circle_dist( px(1,i+1), p1(1,i) )
            edge_vect_n(i) = d1 / ( d1 + d2 )
       else
            d2 = great_circle_dist( px(1,i-1), p1(1,i) )
            d1 = great_circle_dist( px(1,i  ), p1(1,i) )
            edge_vect_n(i) = d1 / ( d2 + d1 )
       endif
    enddo
    if ( is==1 ) then
         edge_vect_n(0) = edge_vect_n(1)
    endif
    if ( (ie+1)==npx ) then
         edge_vect_n(npx) = edge_vect_n(ie)
    endif
    do i=is-1,ie+1
!      if ( is_master() ) write(*,*) i, edge_vect_n(i)
    enddo
 endif

 endif

 end subroutine efactor_a2c_v


  ! ---- PLAIN-CLONE tools/fv_grid_tools.F90:725-866 metric loops ----
  ! (DOCUMENTED SUBSTITUTION: duo archive ships no fv_grid_tools)
  subroutine tools_metrics(Atm, npx, npy)
    type(shim_atm_type), intent(inout), target :: Atm
    integer, intent(in) :: npx, npy
    real(kind=R_GRID), pointer, dimension(:, :, :) :: grid, agrid
    real(kind=R_GRID), pointer, dimension(:, :) :: dx, dy, dxa, dya
    real(kind=R_GRID), pointer, dimension(:, :) :: dxc, dyc
    real(kind=R_GRID) :: p1(2), p2(2), p3(2), p4(2)
    integer :: i, j, istart, iend, jstart, jend
    integer :: is, ie, js, je, isd, ied, jsd, jed
    logical :: stretched_grid, cubed_sphere
    is = Atm%bd%is;  ie = Atm%bd%ie
    js = Atm%bd%js;  je = Atm%bd%je
    isd = Atm%bd%isd; ied = Atm%bd%ied
    jsd = Atm%bd%jsd; jed = Atm%bd%jed
    grid => Atm%gridstruct%grid;   agrid => Atm%gridstruct%agrid
    dx => Atm%gridstruct%dx;       dy => Atm%gridstruct%dy
    dxa => Atm%gridstruct%dxa;     dya => Atm%gridstruct%dya
    dxc => Atm%gridstruct%dxc;     dyc => Atm%gridstruct%dyc
    stretched_grid = Atm%gridstruct%stretched_grid
    cubed_sphere = .true.
             call mpp_update_domains( grid, Atm%domain, position=CORNER)
             if (.not. (Atm%gridstruct%bounded_domain)) then
                call fill_corners(grid(:,:,1), npx, npy, FILL=XDir, BGRID=.true.)
                call fill_corners(grid(:,:,2), npx, npy, FILL=XDir, BGRID=.true.)
             endif

             !--- dx and dy
             if( .not. Atm%gridstruct%bounded_domain) then
                istart=is
                iend=ie
                jstart=js
                jend=je
             else
                istart=isd
                iend=ied
                jstart=jsd
                jend=jed
             endif

             do j = jstart, jend+1
             do i = istart, iend
                p1(1) = grid(i  ,j,1)
                p1(2) = grid(i  ,j,2)
                p2(1) = grid(i+1,j,1)
                p2(2) = grid(i+1,j,2)
                dx(i,j) = great_circle_dist( p2, p1, radius )
             enddo
             enddo
             if( stretched_grid .or. Atm%gridstruct%bounded_domain ) then
                do j = jstart, jend
                do i = istart, iend+1
                   p1(1) = grid(i,j,  1)
                   p1(2) = grid(i,j,  2)
                   p2(1) = grid(i,j+1,1)
                   p2(2) = grid(i,j+1,2)
                   dy(i,j) = great_circle_dist( p2, p1, radius )
                enddo
                enddo
             else
      ! [EXCISED: .not.bounded-only mosaic/buffer code — dead at bounded=T; see plain fv_grid_tools for the arm]
      ! [EXCISED: .not.bounded-only mosaic/buffer code — dead at bounded=T; see plain fv_grid_tools for the arm]
             endif

      ! [EXCISED BLOCK: mpp_get_boundary west/east symmetry fixes.
      !  The CALL is unconditional upstream (fv_grid_tools.F90:768) but
      !  the buffer assignments that consume its output are
      !  bounded-dead, so the live call has no observable effect on the
      !  bounded lane; the serial harness omits the no-effect call.
      !  (codex bounded-r2 finding 3 label correction)]

             call mpp_update_domains( dy, dx, Atm%domain, flags=SCALAR_PAIR,      &
                  gridtype=CGRID_NE_PARAM, complete=.true.)
             if (cubed_sphere .and. (.not. (Atm%gridstruct%bounded_domain))) then
                call fill_corners(dx, dy, npx, npy, DGRID=.true.)
             endif

      ! [SUBSTITUTED — LIVE under bounded (codex bounded-r2 finding 3):
      !  upstream orders the SAME four agrid corners via the
      !  sorted_inta reproducible-summation tables
      !  (fv_grid_tools.F90:785,799-803); this harness uses plain
      !  ordering.  ULP-class only — the C48 da_min_c reproduces the
      !  Zenodo log to 1.5e-12 rel with plain ordering.]

             agrid(:,:,:) = -1.e25

          !--- compute agrid (use same indices as for dx/dy above)

             do j=jstart,jend
             do i=istart,iend
                ! [SUBSTITUTED — LIVE under bounded: the non-stretched
                !  arm orders these four corners via sorted_inta
                !  (fv_grid_tools.F90:799-803); plain ordering here,
                !  ULP-class only (C48 log match 1.5e-12 rel)]
                call cell_center2(grid(i,j,  1:2), grid(i+1,j,  1:2),   &
                                  grid(i,j+1,1:2), grid(i+1,j+1,1:2),   &
                                  agrid(i,j,1:2) )
             enddo
             enddo

             call mpp_update_domains( agrid, Atm%domain, position=CENTER, complete=.true. )
             if (.not. (Atm%gridstruct%bounded_domain)) then
                call fill_corners(agrid(:,:,1), npx, npy, XDir, AGRID=.true.)
                call fill_corners(agrid(:,:,2), npx, npy, YDir, AGRID=.true.)
             endif

             do j=jsd,jed
             do i=isd,ied
                call mid_pt_sphere(grid(i,  j,1:2), grid(i,  j+1,1:2), p1)
                call mid_pt_sphere(grid(i+1,j,1:2), grid(i+1,j+1,1:2), p2)
                dxa(i,j) = great_circle_dist( p2, p1, radius )
                !
                call mid_pt_sphere(grid(i,j  ,1:2), grid(i+1,j  ,1:2), p1)
                call mid_pt_sphere(grid(i,j+1,1:2), grid(i+1,j+1,1:2), p2)
                dya(i,j) = great_circle_dist( p2, p1, radius )
             enddo
             enddo
!      call mpp_update_domains( dxa, dya, Atm%domain, flags=SCALAR_PAIR, gridtype=AGRID_PARAM)
             if (cubed_sphere  .and. (.not. (Atm%gridstruct%bounded_domain))) then
                call fill_corners(dxa, dya, npx, npy, AGRID=.true.)
             endif


      ! [EXCISED: end of the grid-file/nested selector structure]


!       do j=js,je
!          do i=is,ie+1
       do j=jsd,jed
          do i=isd+1,ied
             dxc(i,j) = great_circle_dist(agrid(i,j,:), agrid(i-1,j,:), radius)
          enddo
!xxxxxx
      !Are the following 2 lines appropriate for the regional domain?
!xxxxxx
          dxc(isd,j)   = dxc(isd+1,j)
          dxc(ied+1,j) = dxc(ied,j)
       enddo

!       do j=js,je+1
!          do i=is,ie
       do j=jsd+1,jed
          do i=isd,ied
             dyc(i,j) = great_circle_dist(agrid(i,j,:), agrid(i,j-1,:), radius)
          enddo
       enddo
!xxxxxx
      !Are the following 2 lines appropriate for the regional domain?
!xxxxxx
       do i=isd,ied
          dyc(i,jsd)   = dyc(i,jsd+1)
          dyc(i,jed+1) = dyc(i,jed)
       end do


      ! [SUBSTITUTED: sorted_intb tables — consumed only by the
      !  .not.bounded area_c corner path, dead at bounded=T]

  end subroutine tools_metrics

  ! ---- grid_area BOUNDED branches only (fv_grid_tools.F90:2397-2587;
  ! the stretched_grid .or. bounded_domain arms verbatim: area = plain
  ! A-quads over is-ng..ie+ng, area_c = plain B-quads; every
  ! .not.bounded corner-triangle/edge special is skipped upstream at
  ! bounded=T and therefore not extracted) ----
  subroutine grid_area_bounded(Atm, ndims)
    type(shim_atm_type), intent(inout), target :: Atm
    integer, intent(in) :: ndims
    real(kind=R_GRID) :: p_lL(ndims), p_uL(ndims)
    real(kind=R_GRID) :: p_lR(ndims), p_uR(ndims)
    integer :: i, j, n, nh
    integer :: is, ie, js, je
    real(kind=R_GRID), pointer, dimension(:, :, :) :: grid, agrid
    real(kind=R_GRID), pointer, dimension(:, :) :: area, area_c
    is = Atm%bd%is; ie = Atm%bd%ie
    js = Atm%bd%js; je = Atm%bd%je
    grid => Atm%gridstruct%grid; agrid => Atm%gridstruct%agrid
    area => Atm%gridstruct%area; area_c => Atm%gridstruct%area_c
    nh = Atm%bd%ng
    do j = js - nh, je + nh
      do i = is - nh, ie + nh
        do n = 1, ndims
          p_lL(n) = grid(i, j, n)
          p_uL(n) = grid(i, j + 1, n)
          p_lR(n) = grid(i + 1, j, n)
          p_uR(n) = grid(i + 1, j + 1, n)
        end do
        area(i, j) = get_area(p_lL, p_uL, p_lR, p_uR, radius)
      end do
    end do
    ! upstream bounded arm: nh = ng-1, area_c = 1.e30 poison-init, quad
    ! loop over the inner frame only (fv_grid_tools.F90:2490-2515)
    area_c = 1.d30
    do j = js - nh + 1, je + nh
      do i = is - nh + 1, ie + nh
        do n = 1, ndims
          p_lL(n) = agrid(i - 1, j - 1, n)
          p_lR(n) = agrid(i, j - 1, n)
          p_uL(n) = agrid(i - 1, j, n)
          p_uR(n) = agrid(i, j, n)
        end do
        area_c(i, j) = get_area(p_lL, p_uL, p_lR, p_uR, radius)
      end do
    end do
    ! "Handling outermost ends for area_c" (fv_grid_tools.F90:949-973,
    ! bounded arm verbatim; single tile: is==1, ie==npx-1, js==1,
    ! je==npy-1 all true) — replicate the frame so rarea_c is real over
    ! the full node domain
    call bounded_area_c_ends(Atm)
  end subroutine grid_area_bounded

  ! ---- reciprocal metrics (fv_grid_tools.F90:984-1014 verbatim
  ! ranges; computed once after grid_area, before grid_utils_init) ----
  subroutine tools_reciprocals(Atm)
    type(shim_atm_type), intent(inout), target :: Atm
    integer :: i, j, isd, ied, jsd, jed
    isd = Atm%bd%isd; ied = Atm%bd%ied
    jsd = Atm%bd%jsd; jed = Atm%bd%jed
    do j = jsd, jed + 1
      do i = isd, ied
        Atm%gridstruct%rdx(i, j) = 1.0/Atm%gridstruct%dx(i, j)
      end do
    end do
    do j = jsd, jed
      do i = isd, ied + 1
        Atm%gridstruct%rdy(i, j) = 1.0/Atm%gridstruct%dy(i, j)
      end do
    end do
    do j = jsd, jed
      do i = isd, ied + 1
        Atm%gridstruct%rdxc(i, j) = 1.0/Atm%gridstruct%dxc(i, j)
      end do
    end do
    do j = jsd, jed + 1
      do i = isd, ied
        Atm%gridstruct%rdyc(i, j) = 1.0/Atm%gridstruct%dyc(i, j)
      end do
    end do
    do j = jsd, jed
      do i = isd, ied
        Atm%gridstruct%rarea(i, j) = 1.0/Atm%gridstruct%area(i, j)
        Atm%gridstruct%rdxa(i, j) = 1./Atm%gridstruct%dxa(i, j)
        Atm%gridstruct%rdya(i, j) = 1./Atm%gridstruct%dya(i, j)
      end do
    end do
    do j = jsd, jed + 1
      do i = isd, ied + 1
        Atm%gridstruct%rarea_c(i, j) = 1.0/Atm%gridstruct%area_c(i, j)
      end do
    end do
  end subroutine tools_reciprocals

  subroutine bounded_area_c_ends(Atm)
    type(shim_atm_type), intent(inout), target :: Atm
    real(kind=R_GRID), pointer, dimension(:, :) :: area_c
    integer :: i, j, isd, ied, jsd, jed
    isd = Atm%bd%isd; ied = Atm%bd%ied
    jsd = Atm%bd%jsd; jed = Atm%bd%jed
    area_c => Atm%gridstruct%area_c
    do j = jsd, jed
      area_c(isd, j) = area_c(isd + 1, j)
    end do
    area_c(isd, jsd) = area_c(isd + 1, jsd + 1)
    area_c(isd, jed + 1) = area_c(isd + 1, jed)
    do j = jsd, jed
      area_c(ied + 1, j) = area_c(ied, j)
    end do
    area_c(ied + 1, jsd) = area_c(ied, jsd + 1)
    area_c(ied + 1, jed + 1) = area_c(ied, jed)
    do i = isd, ied
      area_c(i, jsd) = area_c(i, jsd + 1)
    end do
    do i = isd, ied
      area_c(i, jed + 1) = area_c(i, jed)
    end do
  end subroutine bounded_area_c_ends

  ! ---- helpers round 2 (fv_grid_utils 1739-1777, 2728-2745) ----
 subroutine cart_to_latlon(np, q, xs, ys)
! vector version of cart_to_latlon1
  integer, intent(in):: np
  real(kind=R_GRID), intent(inout):: q(3,np)
  real(kind=R_GRID), intent(inout):: xs(np), ys(np)
! local
  real(kind=R_GRID), parameter:: esl=1.d-10
  real (f_p):: p(3)
  real (f_p):: dist, lat, lon
  integer i,k

  do i=1,np
     do k=1,3
        p(k) = q(k,i)
     enddo
     dist = sqrt(p(1)**2 + p(2)**2 + p(3)**2)
     do k=1,3
        p(k) = p(k) / dist
     enddo

     if ( (abs(p(1))+abs(p(2)))  < esl ) then
          lon = real(0.,kind=f_p)
     else
          lon = atan2( p(2), p(1) )   ! range [-pi,pi]
     endif

     if ( lon < 0.) lon = real(2.,kind=f_p)*pi + lon
! RIGHT_HAND system:
     lat = asin(p(3))

     xs(i) = lon
     ys(i) = lat
! q Normalized:
     do k=1,3
        q(k,i) = p(k)
     enddo
  enddo

 end  subroutine cart_to_latlon

 subroutine cell_center3(p1, p2, p3, p4, ec)
! Get center position of a cell
         real(kind=R_GRID) , intent(IN)  :: p1(3), p2(3), p3(3), p4(3)
         real(kind=R_GRID) , intent(OUT) :: ec(3)
! Local
         real (kind=R_GRID)dd
         integer k

         do k=1,3
            ec(k) = p1(k) + p2(k) + p3(k) + p4(k)
         enddo
         dd = sqrt( ec(1)**2 + ec(2)**2 + ec(3)**2 )

         do k=1,3
            ec(k) = ec(k) / dd
         enddo

 end subroutine cell_center3

end module bounded_gs_extract_mod
