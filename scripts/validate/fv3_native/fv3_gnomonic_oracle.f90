! Standalone extraction of GFDL FV3 cubed-sphere grid generation for use as a
! source-derived test oracle (legoESM phase-1 FV3-native grid work).
!
! Source: GFDL_atmos_cubed_sphere @ 6f658bd00ffa52158c1c0f32244b6dd59a0360d7
!   (https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere, cloned 2026-07-14)
!   model/fv_grid_utils.F90: gnomonic_grids, gnomonic_ed, gnomonic_angl,
!     gnomonic_dist, symm_ed, latlon2xyz2, latlon2xyz, mirror_latlon,
!     cart_to_latlon, vect_cross, cell_center2
!   tools/fv_grid_tools.F90: mirror_grid, rot_3d, spherical_to_cartesian,
!     cartesian_to_spherical, torad
!
! Algorithm bodies are copied verbatim; the DOCUMENTED substitutions are:
!   - module plumbing and is_master() printing removed;
!   - R_GRID inlined as selected_real_kind(15) (upstream: r8_kind from
!     platform_mod via fv_arrays.F90:39);
!   - f_p pinned to selected_real_kind(20), i.e. the ENABLE_QUAD_PRECISION
!     build branch of fv_grid_utils.F90:49-55 (the GFDL production choice);
!   - pi computed as 4*atan(1) in double precision — bit-identical to the
!     correctly-rounded FMS constant pi_8 imported upstream;
!   - spherical_to_cartesian / cartesian_to_spherical use the DEFAULT
!     (RIGHT_HAND undefined) branches, i.e. z = -r*sin(lat) and
!     lat = acos(z/r) - pi/2.  This is the convention the mirror_grid
!     pole-forcing lines assume (face 3 pinned to +pi/2 = north, face 6 to
!     -pi/2 = south): selecting the RIGHT_HAND branch instead yields polar
!     tiles whose unforced nodes sit in the OPPOSITE hemisphere from the
!     forced ones — an internally inconsistent grid.  The grid-generation
!     pipeline's own cart_to_latlon (lat = asin(z)) is a separate,
!     unconditional routine and is unaffected;
!   - mirror_latlon's scalar cart_to_latlon call is passed sp(1:1)/sp(2:2)
!     slices (Fortran-legal adaptation of the upstream scalar actual args);
!   - `radius` used by mirror_grid is set to 1 (it enters only as the radial
!     coordinate fed through rot_3d's spherical->cartesian->spherical
!     round-trip, where it cancels exactly).
!
! The driver writes, for grid_type=0 (gnomonic_ed, the FV3 default
! fv_arrays.F90 grid_type=0) at C1, C8, C36:
!   fv3_gnomonic_ed_c<N>.txt        face-1 panel corners (i j lon lat)
!   fv3_gnomonic_ed_faces_c<N>.txt  all 6 mirrored faces (f i j lon lat)
!   fv3_gnomonic_ed_agrid_c<N>.txt  all 6 faces' cell_center2 A-grid centres
! consumed by scripts/validate/fv3_native/gen_oracle.sh ->
! tests/grids/fixtures/fv3_gnomonic_ed_oracle.npz.
!
! Build + run (takes < 1 s):
!   gfortran -O2 -o fv3_gnomonic_oracle fv3_gnomonic_oracle.f90
!   ./fv3_gnomonic_oracle
module fv3_grid_oracle_mod
  implicit none
  private
  public :: gnomonic_grids, mirror_grid, cell_center2, R_GRID

  integer, parameter :: R_GRID = selected_real_kind(15)   ! r8_kind
  ! FV3 ENABLE_QUAD_PRECISION: higher precision (kind=16) for grid factors
  integer, parameter :: f_p = selected_real_kind(20)
  real(kind=R_GRID), parameter :: pi = 4.0d0 * atan(1.0d0)
  real(kind=R_GRID), parameter :: torad = pi / 180.0d0
  ! mirror_grid feeds `radius` through rot_3d's spherical round-trip, where
  ! it cancels; any positive value gives identical lon/lat.
  real(kind=R_GRID), parameter :: radius = 1.0d0

contains

 subroutine gnomonic_grids(grid_type, im, lon, lat)
 integer, intent(in):: im, grid_type
 real(kind=R_GRID), intent(out):: lon(im+1,im+1)
 real(kind=R_GRID), intent(out):: lat(im+1,im+1)
 integer i, j

  if(grid_type==0) call gnomonic_ed(  im, lon, lat)
  if(grid_type==1) call gnomonic_dist(im, lon, lat)
  if(grid_type==2) call gnomonic_angl(im, lon, lat)


  if(grid_type<3) then
     call symm_ed(im, lon, lat)
     do j=1,im+1
        do i=1,im+1
           lon(i,j) = lon(i,j) - pi
        enddo
     enddo
  endif

 end subroutine gnomonic_grids

 subroutine gnomonic_ed(im, lamda, theta)
!-----------------------------------------------------
! Equal distance along the 4 edges of the cubed sphere
!-----------------------------------------------------
! Properties:
!            * defined by intersections of great circles
!            * max(dx,dy; global) / min(dx,dy; global) = sqrt(2) = 1.4142
!            * Max(aspect ratio) = 1.06089
!            * the N-S coordinate curves are const longitude on the 4 faces with equator

 integer, intent(in):: im
 real(kind=R_GRID), intent(out):: lamda(im+1,im+1)
 real(kind=R_GRID), intent(out):: theta(im+1,im+1)

! Local:
 real(kind=R_GRID) pp(3,im+1,im+1)
 real(f_p):: rsq3, alpha, delx, dely
 integer i, j, k

  rsq3 = 1.d0/sqrt(3.d0)
 alpha = asin( rsq3 )

! Ranges:
! lamda = [0.75*pi, 1.25*pi]
! theta = [-alpha, alpha]

    dely = 2.d0*alpha / real(im,kind=f_p)

! Define East-West edges:
 do j=1,im+1
    lamda(1,   j) = 0.75d0*pi                  ! West edge
    lamda(im+1,j) = 1.25d0*pi                  ! East edge
    theta(1,   j) = -alpha + dely*real(j-1,kind=f_p)  ! West edge
    theta(im+1,j) = theta(1,j)               ! East edge
 enddo

! Get North-South edges by symmetry:

 do i=2,im
    call mirror_latlon(lamda(1,1), theta(1,1), lamda(im+1,im+1), theta(im+1,im+1), &
                       lamda(1,i), theta(1,i), lamda(i,1),       theta(i,      1) )
    lamda(i,im+1) =  lamda(i,1)
    theta(i,im+1) = -theta(i,1)
 enddo

! Set 4 corners:
    call latlon2xyz2(lamda(1    ,  1), theta(1,      1), pp(1,   1,   1))
    call latlon2xyz2(lamda(im+1,   1), theta(im+1,   1), pp(1,im+1,   1))
    call latlon2xyz2(lamda(1,   im+1), theta(1,   im+1), pp(1,   1,im+1))
    call latlon2xyz2(lamda(im+1,im+1), theta(im+1,im+1), pp(1,im+1,im+1))

! Map edges on the sphere back to cube:
! Intersections at x=-rsq3

 i=1
 do j=2,im
    call latlon2xyz2(lamda(i,j), theta(i,j), pp(1,i,j))
    pp(2,i,j) = -pp(2,i,j)*rsq3/pp(1,i,j)
    pp(3,i,j) = -pp(3,i,j)*rsq3/pp(1,i,j)
 enddo

 j=1
 do i=2,im
    call latlon2xyz2(lamda(i,j), theta(i,j), pp(1,i,1))
    pp(2,i,1) = -pp(2,i,1)*rsq3/pp(1,i,1)
    pp(3,i,1) = -pp(3,i,1)*rsq3/pp(1,i,1)
 enddo

 do j=1,im+1
    do i=1,im+1
       pp(1,i,j) = -rsq3
    enddo
 enddo

 do j=2,im+1
    do i=2,im+1
! Copy y-z face of the cube along j=1
       pp(2,i,j) = pp(2,i,1)
! Copy along i=1
       pp(3,i,j) = pp(3,1,j)
    enddo
 enddo

 call cart_to_latlon( (im+1)*(im+1), pp, lamda, theta)

 end subroutine gnomonic_ed

 subroutine gnomonic_angl(im, lamda, theta)
! This is the commonly known equi-angular grid
 integer im
 real(kind=R_GRID) lamda(im+1,im+1)
 real(kind=R_GRID) theta(im+1,im+1)
 real(kind=R_GRID) p(3,im+1,im+1)
! Local
 real(kind=R_GRID) rsq3
 integer j,k
 real(kind=R_GRID) dp

 dp = 0.5d0*pi/real(im,kind=R_GRID)

 rsq3 = 1.d0/sqrt(3.d0)
 do k=1,im+1
    do j=1,im+1
       p(1,j,k) =-rsq3               ! constant
       p(2,j,k) =-rsq3*tan(-0.25d0*pi+(j-1)*dp)
       p(3,j,k) = rsq3*tan(-0.25d0*pi+(k-1)*dp)
    enddo
 enddo

 call cart_to_latlon( (im+1)*(im+1), p, lamda, theta)

 end subroutine gnomonic_angl

 subroutine gnomonic_dist(im, lamda, theta)
! This is the commonly known equi-distance grid
 integer im
 real(kind=R_GRID) lamda(im+1,im+1)
 real(kind=R_GRID) theta(im+1,im+1)
 real(kind=R_GRID) p(3,im+1,im+1)
! Local
 real(kind=R_GRID) rsq3, xf, y0, z0
 real(kind=R_GRID) dy, dz
 integer j,k

! Face-2

 rsq3 = 1.d0/sqrt(3.d0)
 xf = -rsq3
 y0 =  rsq3;  dy = -2.d0*rsq3/im
 z0 = -rsq3;  dz =  2.d0*rsq3/im

 do k=1,im+1
    do j=1,im+1
       p(1,j,k) = xf
       p(2,j,k) = y0 + (j-1)*dy
       p(3,j,k) = z0 + (k-1)*dz
    enddo
 enddo
 call cart_to_latlon( (im+1)*(im+1), p, lamda, theta)

 end subroutine gnomonic_dist

 subroutine symm_ed(im, lamda, theta)
! Make grid symmetrical to i=im/2+1
 integer im
 real(kind=R_GRID) lamda(im+1,im+1)
 real(kind=R_GRID) theta(im+1,im+1)
 integer i,j,ip,jp
 real(kind=R_GRID) avg

 do j=2,im+1
    do i=2,im
       lamda(i,j) = lamda(i,1)
    enddo
 enddo

 do j=1,im+1
    do i=1,im/2
       ip = im + 2 - i
       avg = 0.5d0*(lamda(i,j)-lamda(ip,j))
       lamda(i, j) = avg + pi
       lamda(ip,j) = pi - avg
       avg = 0.5d0*(theta(i,j)+theta(ip,j))
       theta(i, j) = avg
       theta(ip,j) = avg
    enddo
 enddo

! Make grid symmetrical to j=im/2+1
 do j=1,im/2
       jp = im + 2 - j
    do i=2,im
       avg = 0.5d0*(lamda(i,j)+lamda(i,jp))
       lamda(i, j) = avg
       lamda(i,jp) = avg
       avg = 0.5d0*(theta(i,j)-theta(i,jp))
       theta(i, j) =  avg
       theta(i,jp) = -avg
    enddo
 enddo

 end subroutine symm_ed

 subroutine latlon2xyz2(lon, lat, p3)
 real(kind=R_GRID), intent(in):: lon, lat
 real(kind=R_GRID), intent(out):: p3(3)
 real(kind=R_GRID) e(2)

    e(1) = lon;    e(2) = lat
    call latlon2xyz(e, p3)

 end subroutine latlon2xyz2

 subroutine latlon2xyz(p, e)
!
! Routine to map (lon, lat) to (x,y,z)
!
 real(kind=R_GRID), intent(in) :: p(2)
 real(kind=R_GRID), intent(out):: e(3)

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

 subroutine mirror_latlon(lon1, lat1, lon2, lat2, lon0, lat0, lon3, lat3)
!
! Given the "mirror" as defined by (lon1, lat1), (lon2, lat2), and center
! of the sphere, compute the mirror image of (lon0, lat0) as  (lon3, lat3)

 real(kind=R_GRID), intent(in):: lon1, lat1, lon2, lat2, lon0, lat0
 real(kind=R_GRID), intent(out):: lon3, lat3
!
 real(kind=R_GRID) p0(3), p1(3), p2(3), nb(3), pp(3), sp(2)
 real(kind=R_GRID) pdot
 integer k

 call latlon2xyz2(lon0, lat0, p0)
 call latlon2xyz2(lon1, lat1, p1)
 call latlon2xyz2(lon2, lat2, p2)
 call vect_cross(nb, p1, p2)

 pdot = sqrt(nb(1)**2+nb(2)**2+nb(3)**2)
 do k=1,3
    nb(k) = nb(k) / pdot
 enddo

 pdot = p0(1)*nb(1) + p0(2)*nb(2) + p0(3)*nb(3)
 do k=1,3
    pp(k) = p0(k) - 2.d0*pdot*nb(k)
 enddo

 call cart_to_latlon(1, pp, sp(1:1), sp(2:2))
 lon3 = sp(1)
 lat3 = sp(2)

 end subroutine  mirror_latlon

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

      call cart_to_latlon(1, ec, e2(1:1), e2(2:2))

 end subroutine cell_center2

      subroutine mirror_grid(grid_global,ng,npx,npy,ndims,nregions)
         integer, intent(IN)    :: ng,npx,npy,ndims,nregions
         real(kind=R_GRID)   , intent(INOUT) :: grid_global(1-ng:npx  +ng,1-ng:npy  +ng,ndims,1:nregions)
         integer :: i,j,n,n1,n2,nreg
         real(kind=R_GRID) :: x1,y1,z1, x2,y2,z2, ang
!
!    Mirror Across the 0-longitude
!
         nreg = 1
         do j=1,ceiling(npy/2.)
            do i=1,ceiling(npx/2.)

            x1 = 0.25d0 * (ABS(grid_global(i        ,j        ,1,nreg)) + &
                           ABS(grid_global(npx-(i-1),j        ,1,nreg)) + &
                           ABS(grid_global(i        ,npy-(j-1),1,nreg)) + &
                           ABS(grid_global(npx-(i-1),npy-(j-1),1,nreg)))
            grid_global(i        ,j        ,1,nreg) = SIGN(x1,grid_global(i        ,j        ,1,nreg))
            grid_global(npx-(i-1),j        ,1,nreg) = SIGN(x1,grid_global(npx-(i-1),j        ,1,nreg))
            grid_global(i        ,npy-(j-1),1,nreg) = SIGN(x1,grid_global(i        ,npy-(j-1),1,nreg))
            grid_global(npx-(i-1),npy-(j-1),1,nreg) = SIGN(x1,grid_global(npx-(i-1),npy-(j-1),1,nreg))

            y1 = 0.25d0 * (ABS(grid_global(i        ,j        ,2,nreg)) + &
                           ABS(grid_global(npx-(i-1),j        ,2,nreg)) + &
                           ABS(grid_global(i        ,npy-(j-1),2,nreg)) + &
                           ABS(grid_global(npx-(i-1),npy-(j-1),2,nreg)))
            grid_global(i        ,j        ,2,nreg) = SIGN(y1,grid_global(i        ,j        ,2,nreg))
            grid_global(npx-(i-1),j        ,2,nreg) = SIGN(y1,grid_global(npx-(i-1),j        ,2,nreg))
            grid_global(i        ,npy-(j-1),2,nreg) = SIGN(y1,grid_global(i        ,npy-(j-1),2,nreg))
            grid_global(npx-(i-1),npy-(j-1),2,nreg) = SIGN(y1,grid_global(npx-(i-1),npy-(j-1),2,nreg))

           ! force dateline/greenwich-meridion consitency
            if (mod(npx,2) /= 0) then
              if ( (i==1+(npx-1)/2.0d0) ) then
                 grid_global(i,j        ,1,nreg) = 0.0d0
                 grid_global(i,npy-(j-1),1,nreg) = 0.0d0
              endif
            endif

            enddo
         enddo

         do nreg=2,nregions
           do j=1,npy
             do i=1,npx

               x1 = grid_global(i,j,1,1)
               y1 = grid_global(i,j,2,1)
               z1 = radius

               if (nreg == 2) then
                  ang = -90.d0
                  call rot_3d( 3, x1, y1, z1, ang, x2, y2, z2, 1, 1)  ! rotate about the z-axis
               elseif (nreg == 3) then
                  ang = -90.d0
                  call rot_3d( 3, x1, y1, z1, ang, x2, y2, z2, 1, 1)  ! rotate about the z-axis
                  ang = 90.d0
                  call rot_3d( 1, x2, y2, z2, ang, x1, y1, z1, 1, 1)  ! rotate about the x-axis
                  x2=x1
                  y2=y1
                  z2=z1

           ! force North Pole and dateline/greenwich-meridion consitency
                  if (mod(npx,2) /= 0) then
                     if ( (i==1+(npx-1)/2.0d0) .and. (i==j) ) then
                        x2 = 0.0d0
                        y2 = pi/2.0d0
                     endif
                     if ( (j==1+(npy-1)/2.0d0) .and. (i < 1+(npx-1)/2.0d0) ) then
                        x2 = 0.0d0
                     endif
                     if ( (j==1+(npy-1)/2.0d0) .and. (i > 1+(npx-1)/2.0d0) ) then
                        x2 = pi
                     endif
                  endif

               elseif (nreg == 4) then
                  ang = -180.d0
                  call rot_3d( 3, x1, y1, z1, ang, x2, y2, z2, 1, 1)  ! rotate about the z-axis
                  ang = 90.d0
                  call rot_3d( 1, x2, y2, z2, ang, x1, y1, z1, 1, 1)  ! rotate about the x-axis
                  x2=x1
                  y2=y1
                  z2=z1

               ! force dateline/greenwich-meridion consitency
                  if (mod(npx,2) /= 0) then
                    if ( (j==1+(npy-1)/2.0d0) ) then
                       x2 = pi
                    endif
                  endif

               elseif (nreg == 5) then
                  ang = 90.d0
                  call rot_3d( 3, x1, y1, z1, ang, x2, y2, z2, 1, 1)  ! rotate about the z-axis
                  ang = 90.d0
                  call rot_3d( 2, x2, y2, z2, ang, x1, y1, z1, 1, 1)  ! rotate about the y-axis
                  x2=x1
                  y2=y1
                  z2=z1
               elseif (nreg == 6) then
                  ang = 90.d0
                  call rot_3d( 2, x1, y1, z1, ang, x2, y2, z2, 1, 1)  ! rotate about the y-axis
                  ang = 0.d0
                  call rot_3d( 3, x2, y2, z2, ang, x1, y1, z1, 1, 1)  ! rotate about the z-axis
                  x2=x1
                  y2=y1
                  z2=z1

           ! force South Pole and dateline/greenwich-meridion consitency
                  if (mod(npx,2) /= 0) then
                     if ( (i==1+(npx-1)/2.0d0) .and. (i==j) ) then
                        x2 = 0.0d0
                        y2 = -pi/2.0d0
                     endif
                     if ( (i==1+(npx-1)/2.0d0) .and. (j > 1+(npy-1)/2.0d0) ) then
                        x2 = 0.0d0
                     endif
                     if ( (i==1+(npx-1)/2.0d0) .and. (j < 1+(npy-1)/2.0d0) ) then
                        x2 = pi
                     endif
                  endif

               endif

               grid_global(i,j,1,nreg) = x2
               grid_global(i,j,2,nreg) = y2

              enddo
            enddo
          enddo

  end subroutine mirror_grid

      subroutine rot_3d(axis, x1in, y1in, z1in, angle, x2out, y2out, z2out, degrees, convert)

         integer, intent(IN) :: axis         ! axis of rotation 1=x, 2=y, 3=z
         real(kind=R_GRID) , intent(IN)    :: x1in, y1in, z1in
         real(kind=R_GRID) , intent(INOUT) :: angle        ! angle to rotate in radians
         real(kind=R_GRID) , intent(OUT)   :: x2out, y2out, z2out
         integer, intent(IN), optional :: degrees ! if present convert angle
                                                  ! from degrees to radians
         integer, intent(IN), optional :: convert ! if present convert input point
                                                  ! from spherical to cartesian, rotate,
                                                  ! and convert back

         real(kind=R_GRID)  :: c, s
         real(kind=R_GRID)  :: x1,y1,z1, x2,y2,z2

         if ( present(convert) ) then
           call spherical_to_cartesian(x1in, y1in, z1in, x1, y1, z1)
         else
           x1=x1in
           y1=y1in
           z1=z1in
         endif

         if ( present(degrees) ) then
            angle = angle*torad
         endif

         c = COS(angle)
         s = SIN(angle)

         SELECT CASE(axis)

            CASE(1)
               x2 =  x1
               y2 =  c*y1 + s*z1
               z2 = -s*y1 + c*z1
            CASE(2)
               x2 = c*x1 - s*z1
               y2 = y1
               z2 = s*x1 + c*z1
            CASE(3)
               x2 =  c*x1 + s*y1
               y2 = -s*x1 + c*y1
               z2 = z1
            CASE DEFAULT
              write(*,*) "Invalid axis: must be 1 for X, 2 for Y, 3 for Z."

         END SELECT

         if ( present(convert) ) then
           call cartesian_to_spherical(x2, y2, z2, x2out, y2out, z2out)
         else
           x2out=x2
           y2out=y2
           z2out=z2
         endif

      end subroutine rot_3d

 subroutine spherical_to_cartesian(lon, lat, r, x, y, z)
         real(kind=R_GRID) , intent(IN)  :: lon, lat, r
         real(kind=R_GRID) , intent(OUT) :: x, y, z

         x = r * COS(lon) * cos(lat)
         y = r * SIN(lon) * cos(lat)
! default build branch (RIGHT_HAND undefined) — see header
         z = -r * sin(lat)
 end subroutine spherical_to_cartesian

      subroutine cartesian_to_spherical(x, y, z, lon, lat, r)
      real(kind=R_GRID) , intent(IN)  :: x, y, z
      real(kind=R_GRID) , intent(OUT) :: lon, lat, r

      r = SQRT(x*x + y*y + z*z)
      if ( (abs(x) + abs(y)) < 1.E-10 ) then       ! poles:
           lon = 0.
      else
           lon = ATAN2(y,x)    ! range: [-pi,pi]
      endif

! default build branch (RIGHT_HAND undefined) — see header
      lat = ACOS(z/r) - pi/2.

      end subroutine cartesian_to_spherical

end module fv3_grid_oracle_mod


program fv3_gnomonic_oracle
  use fv3_grid_oracle_mod, only: gnomonic_grids, mirror_grid, cell_center2, &
                                 R_GRID
  implicit none
  integer, parameter :: sizes(3) = (/ 1, 8, 36 /)
  integer :: s, im, i, j, f, unit_no
  real(kind=R_GRID), allocatable :: lon(:,:), lat(:,:)
  real(kind=R_GRID), allocatable :: grid_global(:,:,:,:)
  real(kind=R_GRID) :: q1(2), q2(2), q3(2), q4(2), e2(2)
  character(len=64) :: fname

  do s = 1, size(sizes)
     im = sizes(s)
     allocate(lon(im+1, im+1), lat(im+1, im+1))
     call gnomonic_grids(0, im, lon, lat)

     ! --- face-1 panel corners (as produced by gnomonic_grids) ---
     write(fname, '(A,I0,A)') 'fv3_gnomonic_ed_c', im, '.txt'
     open(newunit=unit_no, file=trim(fname), status='replace', action='write')
     write(unit_no, '(A)') '# FV3 gnomonic_grids(grid_type=0) face-panel corners: i j lon lat [rad]'
     write(unit_no, '(A,I0)') '# im = ', im
     do j = 1, im+1
        do i = 1, im+1
           write(unit_no, '(I4,1X,I4,1X,ES26.17E3,1X,ES26.17E3)') &
                i, j, lon(i,j), lat(i,j)
        enddo
     enddo
     close(unit_no)

     ! --- all 6 faces via mirror_grid (fv_grid_tools init_grid pipeline) ---
     allocate(grid_global(im+1, im+1, 2, 6))
     do j = 1, im+1
        do i = 1, im+1
           grid_global(i, j, 1, 1) = lon(i, j)
           grid_global(i, j, 2, 1) = lat(i, j)
        enddo
     enddo
     call mirror_grid(grid_global, 0, im+1, im+1, 2, 6)
     write(fname, '(A,I0,A)') 'fv3_gnomonic_ed_faces_c', im, '.txt'
     open(newunit=unit_no, file=trim(fname), status='replace', action='write')
     write(unit_no, '(A)') '# FV3 mirror_grid 6-face gnomonic_ed corners: face i j lon lat [rad]'
     write(unit_no, '(A,I0)') '# im = ', im
     do f = 1, 6
        do j = 1, im+1
           do i = 1, im+1
              write(unit_no, '(I2,1X,I4,1X,I4,1X,ES26.17E3,1X,ES26.17E3)') &
                   f, i, j, grid_global(i,j,1,f), grid_global(i,j,2,f)
           enddo
        enddo
     enddo
     close(unit_no)

     ! --- A-grid cell centres per face via cell_center2 (FV3 agrid) ---
     write(fname, '(A,I0,A)') 'fv3_gnomonic_ed_agrid_c', im, '.txt'
     open(newunit=unit_no, file=trim(fname), status='replace', action='write')
     write(unit_no, '(A)') '# FV3 cell_center2 A-grid centres: face i j lon lat [rad]'
     write(unit_no, '(A,I0)') '# im = ', im
     do f = 1, 6
        do j = 1, im
           do i = 1, im
              ! FV3 fv_grid_tools.F90:1258 ordering (SW, SE, NW, NE);
              ! cell_center2 is a normalized xyz sum, so order-invariant.
              q1(1) = grid_global(i,  j,  1,f); q1(2) = grid_global(i,  j,  2,f)
              q2(1) = grid_global(i+1,j,  1,f); q2(2) = grid_global(i+1,j,  2,f)
              q3(1) = grid_global(i,  j+1,1,f); q3(2) = grid_global(i,  j+1,2,f)
              q4(1) = grid_global(i+1,j+1,1,f); q4(2) = grid_global(i+1,j+1,2,f)
              call cell_center2(q1, q2, q3, q4, e2)
              write(unit_no, '(I2,1X,I4,1X,I4,1X,ES26.17E3,1X,ES26.17E3)') &
                   f, i, j, e2(1), e2(2)
           enddo
        enddo
     enddo
     close(unit_no)

     deallocate(grid_global)
     deallocate(lon, lat)
  enddo

  write(*,*) 'fv3_gnomonic_oracle: wrote C1/C8/C36 panel, 6-face, agrid files'
end program fv3_gnomonic_oracle
