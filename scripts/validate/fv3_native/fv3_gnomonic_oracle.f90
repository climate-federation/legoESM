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
!     round-trip, where it cancels exactly);
!   - metric outputs are UNIT-SPHERE (earth_radius = 1): R enters every
!     upstream formula only as a final multiplicative factor (R for
!     distances, R**2 for areas), so consumers rescale exactly;
!   - MUST be compiled with -fdefault-real-8: great_circle_dist returns
!     default `real` upstream (FV3 production builds use r8 default reals);
!   - dy is computed by direct great_circle_dist instead of upstream
!     get_symmetry (an MPI transpose-copy of dx used for bit
!     reproducibility; values agree to fp roundoff by grid symmetry);
!   - the sorted_index (sorted_inta/sorted_intb) vertex orderings are
!     replaced by the stretched-branch natural orderings — the spherical
!     excess/area sums are relabel-invariant, so this affects values only
!     at fp-roundoff level (~1e-16 relative);
!   - cross-face halo grid/agrid lines (filled by mpp_update_domains +
!     fill_corners upstream) are reconstructed geometrically in the
!     driver: shared cube-edge nodes are bit-equal between faces by the
!     mirror construction, so side pairing/orientation is recovered by
!     exact node matching (see fill_halos).
!
! The metric driver replicates the fv_grid_tools init_grid + grid_area
! WRITE ORDER for the generated (non-stretched, non-bounded) cubed sphere:
! interior agrid-quad area_c -> cube-corner get_area_tri -> the x2
! half-dual edge overwrites in W,E,S,N code order.  The dumped fields are
! therefore the FINAL post-overwrite FV3 state, whatever formula "wins"
! at each node — including the cube corners, where the W/E/S/N edge
! blocks overwrite the get_area_tri values (last writer: S/N blocks).
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
  public :: get_area, get_area_tri, great_circle_dist, mid_pt_sphere
  public :: cos_angle, inner_prod, normalize_vect, cell_center3
  public :: latlon2xyz, mid_pt3_cart, vect_cross

  integer, parameter :: R_GRID = selected_real_kind(15)   ! r8_kind
  ! FV3 ENABLE_QUAD_PRECISION: higher precision (kind=16) for grid factors
  integer, parameter :: f_p = selected_real_kind(20)
  real(kind=R_GRID), parameter :: pi = 4.0d0 * atan(1.0d0)
  real(kind=R_GRID), parameter :: torad = pi / 180.0d0
  real(kind=R_GRID), parameter :: todeg = 180.0d0 / pi
  ! mirror_grid feeds `radius` through rot_3d's spherical round-trip, where
  ! it cancels; any positive value gives identical lon/lat.
  real(kind=R_GRID), parameter :: radius = 1.0d0
  ! Metric substitution (documented in the header): upstream areas/distances
  ! scale by the module-global Earth radius; the oracle emits UNIT-SPHERE
  ! metrics (earth_radius = 1) because R enters every formula only as a final
  ! multiplicative factor — consumers scale by R (distances) / R**2 (areas)
  ! exactly.
  real(kind=R_GRID), parameter :: earth_radius = 1.0d0

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

 subroutine mid_pt_sphere(p1, p2, pm)
      real(kind=R_GRID) , intent(IN)  :: p1(2), p2(2)
      real(kind=R_GRID) , intent(OUT) :: pm(2)
!------------------------------------------
      real(kind=R_GRID) e1(3), e2(3), e3(3)

      call latlon2xyz(p1, e1)
      call latlon2xyz(p2, e2)
      call mid_pt3_cart(e1, e2, e3)
      call cart_to_latlon(1, e3, pm(1:1), pm(2:2))

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

 real(kind=R_GRID) function get_angle(ndims, p1, p2, p3, rad) result (angle)
!     get_angle :: get angle between 3 points on a sphere in lat/lon coords or
!                  xyz coords (determined by ndims argument 2=lat/lon, 3=xyz)
!                  [angle is returned in degrees]

         integer, intent(IN) :: ndims         ! 2=lat/lon, 3=xyz
         real(kind=R_GRID) , intent(IN)   :: p1(ndims)
         real(kind=R_GRID) , intent(IN)   :: p2(ndims)
         real(kind=R_GRID) , intent(IN)   :: p3(ndims)
         integer, intent(in), optional:: rad

         real(kind=R_GRID)  :: e1(3), e2(3), e3(3)

         if (ndims == 2) then
            call spherical_to_cartesian(p2(1), p2(2), real(1.,kind=R_GRID), e1(1), e1(2), e1(3))
            call spherical_to_cartesian(p1(1), p1(2), real(1.,kind=R_GRID), e2(1), e2(2), e2(3))
            call spherical_to_cartesian(p3(1), p3(2), real(1.,kind=R_GRID), e3(1), e3(2), e3(3))
         else
            e1 = p2; e2 = p1; e3 = p3
         endif

! High precision version:
         if ( present(rad) ) then
           angle = spherical_angle(e1, e2, e3)
         else
           angle = todeg * spherical_angle(e1, e2, e3)
         endif

      end function get_angle

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

      real(kind=R_GRID)  function get_area_tri(ndims, p_1, p_2, p_3) &
                        result (myarea)

!     get_area_tri :: get the surface area of a cell defined as a triangle
!                  on the sphere. Area is computed as the spherical excess
!                  [area units are based on the units of radius]


      integer, intent(IN)    :: ndims          ! 2=lat/lon, 3=xyz
      real(kind=R_GRID) , intent(IN)    :: p_1(ndims) !
      real(kind=R_GRID) , intent(IN)    :: p_2(ndims) !
      real(kind=R_GRID) , intent(IN)    :: p_3(ndims) !

      real(kind=R_GRID)  :: angA, angB, angC

        if ( ndims==3 ) then
            angA = spherical_angle(p_1, p_2, p_3)
            angB = spherical_angle(p_2, p_3, p_1)
            angC = spherical_angle(p_3, p_1, p_2)
        else
            angA = get_angle(ndims, p_1, p_2, p_3, 1)
            angB = get_angle(ndims, p_2, p_3, p_1, 1)
            angC = get_angle(ndims, p_3, p_1, p_2, 1)
        endif

        myarea = (angA+angB+angC - pi) * earth_radius**2

      end function get_area_tri

end module fv3_grid_oracle_mod


program fv3_gnomonic_oracle
  use fv3_grid_oracle_mod, only: gnomonic_grids, mirror_grid, cell_center2, &
                                 R_GRID, get_area, get_area_tri,           &
                                 great_circle_dist, mid_pt_sphere,         &
                                 cos_angle, inner_prod, normalize_vect,    &
                                 cell_center3, latlon2xyz, mid_pt3_cart,   &
                                 vect_cross
  implicit none
  integer, parameter :: sizes(3) = (/ 1, 8, 36 /)
  integer :: s, im, i, j, f, unit_no
  real(kind=R_GRID), allocatable :: lon(:,:), lat(:,:)
  real(kind=R_GRID), allocatable :: grid_global(:,:,:,:)
  real(kind=R_GRID) :: q1(2), q2(2), q3(2), q4(2), e2(2)
  character(len=64) :: fname
  ! --- metric-oracle work arrays (unit sphere) ---
  real(kind=R_GRID), allocatable :: agrid6(:,:,:,:)      ! (im,im,2,6)
  real(kind=R_GRID), allocatable :: gridh(:,:,:,:)       ! (0:npx+1,0:npx+1,2,6)
  real(kind=R_GRID), allocatable :: agridh(:,:,:,:)      ! (0:im+1,0:im+1,2,6)
  real(kind=R_GRID), allocatable :: area(:,:,:)          ! (im,im,6)
  real(kind=R_GRID), allocatable :: dxm(:,:,:), dym(:,:,:)   ! (im,im+1,6)/(im+1,im,6)
  real(kind=R_GRID), allocatable :: dxa(:,:,:), dya(:,:,:)   ! (im,im,6)
  real(kind=R_GRID), allocatable :: dxc(:,:,:), dyc(:,:,:)   ! (im+1,im,6)/(im,im+1,6)
  real(kind=R_GRID), allocatable :: area_c(:,:,:)        ! (im+1,im+1,6)
  real(kind=R_GRID) :: p1(2), p2(2), p3(2), p4(2), pm1(2), pm2(2)
  real(kind=R_GRID) :: closure, mult
  integer :: npx

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
     allocate(agrid6(im, im, 2, 6))
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
              agrid6(i, j, 1, f) = e2(1)
              agrid6(i, j, 2, f) = e2(2)
              write(unit_no, '(I2,1X,I4,1X,I4,1X,ES26.17E3,1X,ES26.17E3)') &
                   f, i, j, e2(1), e2(2)
           enddo
        enddo
     enddo
     close(unit_no)

     ! ------------------------------------------------------------------
     ! Metric oracle (unit sphere), replicating fv_grid_tools init_grid +
     ! grid_area for the generated (non-stretched, non-bounded) cubed
     ! sphere, INCLUDING the code-order overwrites:
     !   1. dx/dy corner-to-corner edge lengths
     !   2. dxa/dya through-cell mid-point distances
     !   3. dxc/dyc interior agrid distances, panel edges = 2*(mid->centre)
     !   4. grid_area: area (corner quads), area_c (agrid quads, using
     !      cross-face halo agrid), cube-corner triangles (get_area_tri)
     !   5. init_grid ×2 edge overwrites of area_c in W,E,S,N order using
     !      halo grid/agrid — replicated with the SAME write order, so the
     !      dumped fields are the FINAL post-overwrite FV3 state.
     ! Cross-face halo lines are reconstructed geometrically (shared cube
     ! edges match bit-exactly between faces by the mirror construction).
     ! ------------------------------------------------------------------
     npx = im + 1
     allocate(gridh(0:npx+1, 0:npx+1, 2, 6))
     allocate(agridh(0:im+1, 0:im+1, 2, 6))
     call fill_halos(im, grid_global, agrid6, gridh, agridh)

     allocate(area(im, im, 6))
     allocate(dxm(im, im+1, 6), dym(im+1, im, 6))
     allocate(dxa(im, im, 6), dya(im, im, 6))
     allocate(dxc(im+1, im, 6), dyc(im, im+1, 6))
     allocate(area_c(im+1, im+1, 6))

     do f = 1, 6
        ! --- dx(i,j): great_circle_dist(grid(i,j), grid(i+1,j)), j=1..npx
        do j = 1, npx
           do i = 1, im
              p1(1) = gridh(i,  j, 1, f); p1(2) = gridh(i,  j, 2, f)
              p2(1) = gridh(i+1,j, 1, f); p2(2) = gridh(i+1,j, 2, f)
              dxm(i, j, f) = great_circle_dist(p2, p1)
           enddo
        enddo
        ! --- dy(i,j): great_circle_dist(grid(i,j), grid(i,j+1)), i=1..npx
        do j = 1, im
           do i = 1, npx
              p1(1) = gridh(i, j,  1, f); p1(2) = gridh(i, j,  2, f)
              p2(1) = gridh(i, j+1,1, f); p2(2) = gridh(i, j+1,2, f)
              dym(i, j, f) = great_circle_dist(p2, p1)
           enddo
        enddo
        ! --- dxa/dya: mid-point-to-mid-point through the cell
        do j = 1, im
           do i = 1, im
              p1(1) = gridh(i,  j, 1, f); p1(2) = gridh(i,  j, 2, f)
              p2(1) = gridh(i,  j+1,1,f); p2(2) = gridh(i,  j+1,2,f)
              call mid_pt_sphere(p1, p2, pm1)
              p1(1) = gridh(i+1,j, 1, f); p1(2) = gridh(i+1,j, 2, f)
              p2(1) = gridh(i+1,j+1,1,f); p2(2) = gridh(i+1,j+1,2,f)
              call mid_pt_sphere(p1, p2, pm2)
              dxa(i, j, f) = great_circle_dist(pm2, pm1)
              p1(1) = gridh(i,  j, 1, f); p1(2) = gridh(i,  j, 2, f)
              p2(1) = gridh(i+1,j, 1, f); p2(2) = gridh(i+1,j, 2, f)
              call mid_pt_sphere(p1, p2, pm1)
              p1(1) = gridh(i,  j+1,1,f); p1(2) = gridh(i,  j+1,2,f)
              p2(1) = gridh(i+1,j+1,1,f); p2(2) = gridh(i+1,j+1,2,f)
              call mid_pt_sphere(p1, p2, pm2)
              dya(i, j, f) = great_circle_dist(pm2, pm1)
           enddo
        enddo
        ! --- dxc: interior agrid-to-agrid; panel edges 2*(mid->centre)
        do j = 1, im
           do i = 2, im
              p1(1) = agridh(i,  j, 1, f); p1(2) = agridh(i,  j, 2, f)
              p2(1) = agridh(i-1,j, 1, f); p2(2) = agridh(i-1,j, 2, f)
              dxc(i, j, f) = great_circle_dist(p1, p2)
           enddo
           ! i = 1 (west edge): dxc = 2*gcd(mid(grid(1,j),grid(1,j+1)), agrid(1,j))
           p1(1) = gridh(1, j,  1, f); p1(2) = gridh(1, j,  2, f)
           p2(1) = gridh(1, j+1,1, f); p2(2) = gridh(1, j+1,2, f)
           call mid_pt_sphere(p1, p2, pm1)
           p2(1) = agridh(1, j, 1, f); p2(2) = agridh(1, j, 2, f)
           dxc(1, j, f) = 2.0d0 * great_circle_dist(pm1, p2)
           ! i = npx (east edge)
           p1(1) = agridh(im, j, 1, f); p1(2) = agridh(im, j, 2, f)
           p2(1) = gridh(npx, j,  1, f); p2(2) = gridh(npx, j,  2, f)
           p3(1) = gridh(npx, j+1,1, f); p3(2) = gridh(npx, j+1,2, f)
           call mid_pt_sphere(p2, p3, pm1)
           dxc(npx, j, f) = 2.0d0 * great_circle_dist(p1, pm1)
        enddo
        ! --- dyc analog
        do i = 1, im
           do j = 2, im
              p1(1) = agridh(i, j,  1, f); p1(2) = agridh(i, j,  2, f)
              p2(1) = agridh(i, j-1,1, f); p2(2) = agridh(i, j-1,2, f)
              dyc(i, j, f) = great_circle_dist(p1, p2)
           enddo
           p1(1) = gridh(i,  1, 1, f); p1(2) = gridh(i,  1, 2, f)
           p2(1) = gridh(i+1,1, 1, f); p2(2) = gridh(i+1,1, 2, f)
           call mid_pt_sphere(p1, p2, pm1)
           p2(1) = agridh(i, 1, 1, f); p2(2) = agridh(i, 1, 2, f)
           dyc(i, 1, f) = 2.0d0 * great_circle_dist(pm1, p2)
           p1(1) = agridh(i, im, 1, f); p1(2) = agridh(i, im, 2, f)
           p2(1) = gridh(i,  npx,1, f); p2(2) = gridh(i,  npx,2, f)
           p3(1) = gridh(i+1,npx,1, f); p3(2) = gridh(i+1,npx,2, f)
           call mid_pt_sphere(p2, p3, pm1)
           dyc(i, npx, f) = 2.0d0 * great_circle_dist(p1, pm1)
        enddo
        ! --- area: spherical-excess corner quads,
        !     get_area(p_lL, p_uL, p_lR, p_uR)
        do j = 1, im
           do i = 1, im
              p1(1) = gridh(i,  j,  1, f); p1(2) = gridh(i,  j,  2, f)  ! lL
              p2(1) = gridh(i,  j+1,1, f); p2(2) = gridh(i,  j+1,2, f)  ! uL
              p3(1) = gridh(i+1,j,  1, f); p3(2) = gridh(i+1,j,  2, f)  ! lR
              p4(1) = gridh(i+1,j+1,1, f); p4(2) = gridh(i+1,j+1,2, f)  ! uR
              area(i, j, f) = get_area(p1, p2, p3, p4)
           enddo
        enddo
        ! --- area_c step 1 (grid_area): agrid quads everywhere (halo agrid
        !     at panel borders), get_area(p_lL, p_uL, p_lR, p_uR)
        do j = 1, npx
           do i = 1, npx
              p1(1) = agridh(i-1,j-1,1, f); p1(2) = agridh(i-1,j-1,2, f)  ! lL
              p2(1) = agridh(i-1,j,  1, f); p2(2) = agridh(i-1,j,  2, f)  ! uL
              p3(1) = agridh(i,  j-1,1, f); p3(2) = agridh(i,  j-1,2, f)  ! lR
              p4(1) = agridh(i,  j,  1, f); p4(2) = agridh(i,  j,  2, f)  ! uR
              area_c(i, j, f) = get_area(p1, p2, p3, p4)
           enddo
        enddo
        ! --- area_c step 2 (grid_area corners): triangular dual cells,
        !     3 surrounding cell centres (stretched-branch point sets)
        p1(1) = agridh(0,1,1,f); p1(2) = agridh(0,1,2,f)
        p2(1) = agridh(1,1,1,f); p2(2) = agridh(1,1,2,f)
        p3(1) = agridh(1,0,1,f); p3(2) = agridh(1,0,2,f)
        area_c(1, 1, f) = get_area_tri(2, p1, p2, p3)
        p1(1) = agridh(im,1,1,f);   p1(2) = agridh(im,1,2,f)
        p2(1) = agridh(im,0,1,f);   p2(2) = agridh(im,0,2,f)
        p3(1) = agridh(im+1,1,1,f); p3(2) = agridh(im+1,1,2,f)
        area_c(npx, 1, f) = get_area_tri(2, p1, p2, p3)
        p1(1) = agridh(im,im,1,f);   p1(2) = agridh(im,im,2,f)
        p2(1) = agridh(im+1,im,1,f); p2(2) = agridh(im+1,im,2,f)
        p3(1) = agridh(im,im+1,1,f); p3(2) = agridh(im,im+1,2,f)
        area_c(npx, npx, f) = get_area_tri(2, p1, p2, p3)
        p1(1) = agridh(1,im,1,f);   p1(2) = agridh(1,im,2,f)
        p2(1) = agridh(1,im+1,1,f); p2(2) = agridh(1,im+1,2,f)
        p3(1) = agridh(0,im,1,f);   p3(2) = agridh(0,im,2,f)
        area_c(1, npx, f) = get_area_tri(2, p1, p2, p3)
        ! --- area_c step 3 (init_grid): ×2 half-dual overwrites, W,E,S,N
        !     code order — replicated verbatim (arg slots of
        !     get_area(p1, p4, p2, p3) as in the source)
        ! West (i=1), j = 1..npx:
        do j = 1, npx
           p1(1) = gridh(1, j-1,1, f); p1(2) = gridh(1, j-1,2, f)
           p2(1) = gridh(1, j,  1, f); p2(2) = gridh(1, j,  2, f)
           call mid_pt_sphere(p1, p2, pm1)
           p1(1) = gridh(1, j+1,1, f); p1(2) = gridh(1, j+1,2, f)
           call mid_pt_sphere(p2, p1, pm2)
           p3(1) = agridh(1, j-1,1, f); p3(2) = agridh(1, j-1,2, f)
           p4(1) = agridh(1, j,  1, f); p4(2) = agridh(1, j,  2, f)
           area_c(1, j, f) = 2.0d0 * get_area(pm1, pm2, p3, p4)
        enddo
        ! East (i=npx):
        do j = 1, npx
           p1(1) = agridh(im, j-1,1, f); p1(2) = agridh(im, j-1,2, f)
           p2(1) = gridh(npx, j-1,1, f); p2(2) = gridh(npx, j-1,2, f)
           p3(1) = gridh(npx, j,  1, f); p3(2) = gridh(npx, j,  2, f)
           call mid_pt_sphere(p2, p3, pm1)
           p2(1) = gridh(npx, j+1,1, f); p2(2) = gridh(npx, j+1,2, f)
           call mid_pt_sphere(p3, p2, pm2)
           p4(1) = agridh(im, j, 1, f); p4(2) = agridh(im, j, 2, f)
           area_c(npx, j, f) = 2.0d0 * get_area(p1, p4, pm1, pm2)
        enddo
        ! South (j=1):
        do i = 1, npx
           p1(1) = gridh(i-1,1, 1, f); p1(2) = gridh(i-1,1, 2, f)
           p2(1) = gridh(i,  1, 1, f); p2(2) = gridh(i,  1, 2, f)
           call mid_pt_sphere(p1, p2, pm1)
           p1(1) = gridh(i+1,1, 1, f); p1(2) = gridh(i+1,1, 2, f)
           call mid_pt_sphere(p2, p1, pm2)
           p3(1) = agridh(i,  1, 1, f); p3(2) = agridh(i,  1, 2, f)
           p4(1) = agridh(i-1,1, 1, f); p4(2) = agridh(i-1,1, 2, f)
           area_c(i, 1, f) = 2.0d0 * get_area(pm1, p4, pm2, p3)
        enddo
        ! North (j=npx):
        do i = 1, npx
           p1(1) = agridh(i-1,im,1, f); p1(2) = agridh(i-1,im,2, f)
           p2(1) = agridh(i,  im,1, f); p2(2) = agridh(i,  im,2, f)
           p3(1) = gridh(i,  npx,1, f); p3(2) = gridh(i,  npx,2, f)
           p4(1) = gridh(i+1,npx,1, f); p4(2) = gridh(i+1,npx,2, f)
           call mid_pt_sphere(p3, p4, pm1)
           p4(1) = gridh(i-1,npx,1, f); p4(2) = gridh(i-1,npx,2, f)
           call mid_pt_sphere(p4, p3, pm2)
           area_c(i, npx, f) = 2.0d0 * get_area(p1, pm2, p2, pm1)
        enddo
     enddo

     ! closure diagnostics (unit sphere): cell areas tile exactly once;
     ! area_c nodes are shared by 2 faces on cube edges, 3 at vertices.
     closure = sum(area)
     write(*,'(A,I0,A,ES24.16)') 'C', im, &
          ' area closure  sum/4pi - 1 = ', closure / (4.0d0*pi_val()) - 1.0d0
     closure = 0.0d0
     do f = 1, 6
        do j = 1, npx
           do i = 1, npx
              mult = 1.0d0
              if (i == 1 .or. i == npx) mult = mult * 2.0d0
              if (j == 1 .or. j == npx) mult = mult * 2.0d0
              if (mult > 3.0d0) mult = 3.0d0   ! cube vertex: 3 faces
              closure = closure + area_c(i, j, f) / mult
           enddo
        enddo
     enddo
     write(*,'(A,I0,A,ES24.16)') 'C', im, &
          ' area_c closure sum/4pi - 1 = ', closure / (4.0d0*pi_val()) - 1.0d0

     ! ------------------------------------------------------------------
     ! Angle/tangent oracle (grid_utils_init, fv_grid_utils.F90:240-561):
     ! cos_sg/sin_sg 9-position per cell (corners 6-9 via cos_angle at the
     ! cell corner nodes with the upstream sign pattern; edge mid-points
     ! 1-4 via mid_pt3_cart + agrid centre — the "No averaging" exact
     ! branch; centre 5 via inner_prod(ec1,ec2) from get_center_vect's
     ! cell-local construction), then the derived staggered families:
     !   cosa_u/sina_u/rsin_u (u-faces), cosa_v/sina_v/rsin_v (v-faces),
     !   cosa_s/rsin2 (centres), cosa/sina (B-nodes, 0.5*(sg8+sg6)).
     ! sg is also evaluated on SIDE-halo cells (cross-face, via the
     ! reconstructed halo lines) because panel-edge staggered values
     ! average with the neighbour face's cells, exactly as the mpp-filled
     ! upstream loops do.  The FOUR cube-vertex B-nodes per face are
     ! emitted as -9999: upstream computes them from fill_corners
     ! XDir-mirrored ghost geometry (and rsina there is big_number
     ! poison) — that convention bundle belongs to the phase-4 native
     ! solver port and is deliberately out of the phase-2B pin.
     ! ------------------------------------------------------------------
     call angle_oracle(im, gridh, agridh)

     write(fname, '(A,I0,A)') 'fv3_metrics_c', im, '.txt'
     open(newunit=unit_no, file=trim(fname), status='replace', action='write')
     write(unit_no, '(A)') '# FV3 init_grid/grid_area metrics (unit sphere): fieldid f i j value'
     write(unit_no, '(A)') '# 1=area 2=dx 3=dy 4=dxa 5=dya 6=dxc 7=dyc 8=area_c'
     write(unit_no, '(A,I0)') '# im = ', im
     do f = 1, 6
        do j = 1, im
           do i = 1, im
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 1,f,i,j,area(i,j,f)
           enddo
        enddo
        do j = 1, npx
           do i = 1, im
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 2,f,i,j,dxm(i,j,f)
           enddo
        enddo
        do j = 1, im
           do i = 1, npx
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 3,f,i,j,dym(i,j,f)
           enddo
        enddo
        do j = 1, im
           do i = 1, im
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 4,f,i,j,dxa(i,j,f)
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 5,f,i,j,dya(i,j,f)
           enddo
        enddo
        do j = 1, im
           do i = 1, npx
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 6,f,i,j,dxc(i,j,f)
           enddo
        enddo
        do j = 1, npx
           do i = 1, im
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 7,f,i,j,dyc(i,j,f)
           enddo
        enddo
        do j = 1, npx
           do i = 1, npx
              write(unit_no,'(I1,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 8,f,i,j,area_c(i,j,f)
           enddo
        enddo
     enddo
     close(unit_no)

     deallocate(area, dxm, dym, dxa, dya, dxc, dyc, area_c)
     deallocate(gridh, agridh, agrid6)
     deallocate(grid_global)
     deallocate(lon, lat)
  enddo

  write(*,*) 'fv3_gnomonic_oracle: wrote C1/C8/C36 panel, 6-face, agrid, metric files'

contains

  function pi_val() result(v)
    real(kind=R_GRID) :: v
    v = 4.0d0 * atan(1.0d0)
  end function pi_val

  !--------------------------------------------------------------------
  ! Reconstruct 1-deep cross-face halo lines for grid corners and agrid
  ! centres.  Shared cube-edge corner NODES are bit-equal between faces
  ! (mirror construction), so side pairing is found by matching the great-
  ! circle midpoint of each side's two END nodes (the cube-edge midpoint,
  ! unique to each of the 12 cube edges); orientation by matching the
  ! first node.  Halo grid line = neighbour's second node line inward;
  ! halo agrid line = neighbour's first cell line inward.
  ! Diagonal halo entries are left at a poison value — the replicated
  ! FV3 formulas never read them.
  !--------------------------------------------------------------------
  subroutine fill_halos(im, grid6, agrid6, gridh, agridh)
    integer, intent(in) :: im
    real(kind=R_GRID), intent(in)  :: grid6(im+1, im+1, 2, 6)
    real(kind=R_GRID), intent(in)  :: agrid6(im, im, 2, 6)
    real(kind=R_GRID), intent(out) :: gridh(0:im+2, 0:im+2, 2, 6)
    real(kind=R_GRID), intent(out) :: agridh(0:im+1, 0:im+1, 2, 6)

    integer :: npx, f, g, s, sp, k, n
    logical :: rev, found
    real(kind=R_GRID) :: e1(3), e2(3), m_f(3), m_g(3), d
    real(kind=R_GRID) :: pa(2), pb(2)
    real(kind=R_GRID) :: nodeline(im+1, 2), cellline(im, 2)

    npx = im + 1
    gridh  = -1.0d25
    agridh = -1.0d25
    do f = 1, 6
       gridh(1:npx, 1:npx, :, f) = grid6(:, :, :, f)
       agridh(1:im, 1:im, :, f)  = agrid6(:, :, :, f)
    enddo

    do f = 1, 6
       do s = 1, 4   ! 1=S(j=1), 2=N(j=npx), 3=W(i=1), 4=E(i=npx)
          call side_midpoint(im, grid6, f, s, m_f)
          found = .false.
          do g = 1, 6
             if (g == f) cycle
             do sp = 1, 4
                call side_midpoint(im, grid6, g, sp, m_g)
                d = (m_f(1)-m_g(1))**2 + (m_f(2)-m_g(2))**2 + (m_f(3)-m_g(3))**2
                if (d < 1.0d-20) then
                   found = .true.
                   exit
                endif
             enddo
             if (found) exit
          enddo
          if (.not. found) then
             write(*,*) 'fill_halos: no neighbour for face', f, 'side', s
             stop 1
          endif
          ! orientation: does f-side node 1 equal g-side node 1?
          call side_node(im, grid6, f, s, 1, pa)
          call side_node(im, grid6, g, sp, 1, pb)
          call latlon2xyz_local(pa, e1)
          call latlon2xyz_local(pb, e2)
          d = (e1(1)-e2(1))**2 + (e1(2)-e2(2))**2 + (e1(3)-e2(3))**2
          rev = (d > 1.0d-20)

          ! neighbour's inward node line (one step off the shared edge)
          do k = 1, npx
             n = k
             if (rev) n = npx + 1 - k
             call side_inner_node(im, grid6, g, sp, n, pa)
             nodeline(k, :) = pa
          enddo
          ! neighbour's first cell line
          do k = 1, im
             n = k
             if (rev) n = im + 1 - k
             call side_inner_cell(im, agrid6, g, sp, n, pa)
             cellline(k, :) = pa
          enddo

          select case (s)
          case (1)  ! S: fill (k, 0)
             do k = 1, npx
                gridh(k, 0, :, f) = nodeline(k, :)
             enddo
             do k = 1, im
                agridh(k, 0, :, f) = cellline(k, :)
             enddo
          case (2)  ! N: fill (k, npx+1) nodes / (k, im+1) cells
             do k = 1, npx
                gridh(k, npx+1, :, f) = nodeline(k, :)
             enddo
             do k = 1, im
                agridh(k, im+1, :, f) = cellline(k, :)
             enddo
          case (3)  ! W
             do k = 1, npx
                gridh(0, k, :, f) = nodeline(k, :)
             enddo
             do k = 1, im
                agridh(0, k, :, f) = cellline(k, :)
             enddo
          case (4)  ! E
             do k = 1, npx
                gridh(npx+1, k, :, f) = nodeline(k, :)
             enddo
             do k = 1, im
                agridh(im+1, k, :, f) = cellline(k, :)
             enddo
          end select
       enddo
    enddo
  end subroutine fill_halos

  subroutine side_node(im, grid6, f, s, k, p)
    integer, intent(in) :: im, f, s, k
    real(kind=R_GRID), intent(in) :: grid6(im+1, im+1, 2, 6)
    real(kind=R_GRID), intent(out) :: p(2)
    integer :: npx
    npx = im + 1
    select case (s)
    case (1); p = grid6(k, 1, :, f)
    case (2); p = grid6(k, npx, :, f)
    case (3); p = grid6(1, k, :, f)
    case (4); p = grid6(npx, k, :, f)
    end select
  end subroutine side_node

  subroutine side_inner_node(im, grid6, f, s, k, p)
    ! node one step INWARD from side s (the neighbour-face line that fills
    ! the requesting face's halo)
    integer, intent(in) :: im, f, s, k
    real(kind=R_GRID), intent(in) :: grid6(im+1, im+1, 2, 6)
    real(kind=R_GRID), intent(out) :: p(2)
    integer :: npx
    npx = im + 1
    select case (s)
    case (1); p = grid6(k, 2, :, f)
    case (2); p = grid6(k, npx-1, :, f)
    case (3); p = grid6(2, k, :, f)
    case (4); p = grid6(npx-1, k, :, f)
    end select
  end subroutine side_inner_node

  subroutine side_inner_cell(im, agrid6, f, s, k, p)
    integer, intent(in) :: im, f, s, k
    real(kind=R_GRID), intent(in) :: agrid6(im, im, 2, 6)
    real(kind=R_GRID), intent(out) :: p(2)
    select case (s)
    case (1); p = agrid6(k, 1, :, f)
    case (2); p = agrid6(k, im, :, f)
    case (3); p = agrid6(1, k, :, f)
    case (4); p = agrid6(im, k, :, f)
    end select
  end subroutine side_inner_cell

  subroutine side_midpoint(im, grid6, f, s, m)
    ! cube-edge midpoint: normalized xyz sum of the side's two END nodes
    integer, intent(in) :: im, f, s
    real(kind=R_GRID), intent(in) :: grid6(im+1, im+1, 2, 6)
    real(kind=R_GRID), intent(out) :: m(3)
    real(kind=R_GRID) :: pa(2), pb(2), ea(3), eb(3), dd
    call side_node(im, grid6, f, s, 1, pa)
    call side_node(im, grid6, f, s, im+1, pb)
    call latlon2xyz_local(pa, ea)
    call latlon2xyz_local(pb, eb)
    m = ea + eb
    dd = sqrt(m(1)**2 + m(2)**2 + m(3)**2)
    m = m / dd
  end subroutine side_midpoint

  subroutine latlon2xyz_local(p, e)
    real(kind=R_GRID), intent(in)  :: p(2)
    real(kind=R_GRID), intent(out) :: e(3)
    e(1) = cos(p(2)) * cos(p(1))
    e(2) = cos(p(2)) * sin(p(1))
    e(3) = sin(p(2))
  end subroutine latlon2xyz_local

  !--------------------------------------------------------------------
  ! Per-cell 9-position cos_sg via the exact grid_utils_init formulas
  ! (fv_grid_utils.F90:327-357).  Nodes: A=(i,j) SW, B=(i+1,j) SE,
  ! C=(i,j+1) NW, D=(i+1,j+1) NE; centre from the halo agrid.
  !--------------------------------------------------------------------
  subroutine cell_sg(a, b, c, d, ctr, sg)
    real(kind=R_GRID), intent(in) :: a(2), b(2), c(2), d(2), ctr(2)
    real(kind=R_GRID), intent(out) :: sg(9)
    real(kind=R_GRID) :: g_a(3), g_b(3), g_c(3), g_d(3), p3c(3)
    real(kind=R_GRID) :: p1(3), p2(3), pc(3), pv(3), u1(3), u2(3)

    call latlon2xyz(a, g_a)
    call latlon2xyz(b, g_b)
    call latlon2xyz(c, g_c)
    call latlon2xyz(d, g_d)
    call latlon2xyz(ctr, p3c)

    ! corners (upstream sign pattern)
    sg(6) =  cos_angle( g_a, g_b, g_c )
    sg(7) = -cos_angle( g_b, g_a, g_d )
    sg(8) =  cos_angle( g_d, g_b, g_c )
    sg(9) = -cos_angle( g_c, g_a, g_d )
    ! edge mid-points ("No averaging" exact branch)
    call mid_pt3_cart(g_a, g_c, p1)
    sg(1) = cos_angle( p1, p3c, g_c )
    call mid_pt3_cart(g_a, g_b, p1)
    sg(2) = cos_angle( p1, g_b, p3c )
    call mid_pt3_cart(g_b, g_d, p1)
    sg(3) = cos_angle( p1, p3c, g_b )
    call mid_pt3_cart(g_c, g_d, p1)
    sg(4) = cos_angle( p1, g_c, p3c )
    ! centre via get_center_vect's cell-local ec1/ec2
    call cell_center3(g_a, g_b, g_c, g_d, pc)
    call mid_pt3_cart(g_a, g_c, p1)
    call mid_pt3_cart(g_b, g_d, p2)
    call vect_cross(pv, p2, p1)
    call vect_cross(u1, pc, pv)
    call normalize_vect(u1)
    call mid_pt3_cart(g_a, g_b, p1)
    call mid_pt3_cart(g_c, g_d, p2)
    call vect_cross(pv, p2, p1)
    call vect_cross(u2, pc, pv)
    call normalize_vect(u2)
    sg(5) = inner_prod(u1, u2)
  end subroutine cell_sg

  subroutine angle_oracle(im, gridh, agridh)
    integer, intent(in) :: im
    real(kind=R_GRID), intent(in) :: gridh(0:im+2, 0:im+2, 2, 6)
    real(kind=R_GRID), intent(in) :: agridh(0:im+1, 0:im+1, 2, 6)

    integer :: npx, f, i, j, ip, u
    character(len=64) :: fname
    real(kind=R_GRID), parameter :: tiny_number = 1.0d-8
    real(kind=R_GRID), parameter :: sentinel = -9999.0d0
    ! sg over cells incl. the SIDE halo ring (diagonal halo cells unused)
    real(kind=R_GRID), allocatable :: csg(:,:,:), ssg(:,:,:)
    real(kind=R_GRID) :: sg(9)
    real(kind=R_GRID) :: cu, su, cv, sv, cb, sb, val

    npx = im + 1
    allocate(csg(9, 0:im+1, 0:im+1))
    allocate(ssg(9, 0:im+1, 0:im+1))

    write(fname, '(A,I0,A)') 'fv3_angles_c', im, '.txt'
    open(newunit=u, file=trim(fname), status='replace', action='write')
    write(u, '(A)') '# FV3 grid_utils_init angle fields: fieldid f i j value'
    write(u, '(A)') '# 10+ip=cos_sg(ip) 20=cosa_u 21=sina_u 22=rsin_u 23=cosa_v'
    write(u, '(A)') '# 24=sina_v 25=rsin_v 26=cosa_s 27=rsin2 28=cosa 29=sina 30=rsina'
    write(u, '(A)') '# cube-vertex B-nodes / edge rsina = -9999 sentinel (phase-4 bundle)'
    write(u, '(A,I0)') '# im = ', im

    do f = 1, 6
       csg = sentinel
       ssg = sentinel
       ! interior + side-halo cells (skip the 4 diagonal halo cells)
       do j = 0, im+1
          do i = 0, im+1
             if ( (i==0 .or. i==im+1) .and. (j==0 .or. j==im+1) ) cycle
             call cell_sg(gridh(i,  j,  :, f), gridh(i+1,j,  :, f),   &
                          gridh(i,  j+1,:, f), gridh(i+1,j+1,:, f),   &
                          agridh(i, j, :, f), sg)
             csg(:, i, j) = sg
             do ip = 1, 9
                ssg(ip, i, j) = min(1.0d0, sqrt(max(0.0d0, 1.0d0 - sg(ip)**2)))
             enddo
          enddo
       enddo

       ! interior sg dump
       do j = 1, im
          do i = 1, im
             do ip = 1, 9
                write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') &
                     10+ip, f, i, j, csg(ip, i, j)
             enddo
          enddo
       enddo
       ! cosa_u / sina_u / rsin_u on u-faces (i=1..npx, j=1..im)
       do j = 1, im
          do i = 1, npx
             cu = 0.5d0 * (csg(3, i-1, j) + csg(1, i, j))
             su = 0.5d0 * (ssg(3, i-1, j) + ssg(1, i, j))
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 20, f, i, j, cu
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 21, f, i, j, su
             if (i == 1 .or. i == npx) then
                val = 1.0d0 / sign(max(tiny_number, abs(su)), su)
             else
                val = 1.0d0 / max(tiny_number, su**2)
             endif
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 22, f, i, j, val
          enddo
       enddo
       ! cosa_v / sina_v / rsin_v on v-faces (i=1..im, j=1..npx)
       do j = 1, npx
          do i = 1, im
             cv = 0.5d0 * (csg(4, i, j-1) + csg(2, i, j))
             sv = 0.5d0 * (ssg(4, i, j-1) + ssg(2, i, j))
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 23, f, i, j, cv
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 24, f, i, j, sv
             if (j == 1 .or. j == npx) then
                val = 1.0d0 / sign(max(tiny_number, abs(sv)), sv)
             else
                val = 1.0d0 / max(tiny_number, sv**2)
             endif
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 25, f, i, j, val
          enddo
       enddo
       ! cosa_s / rsin2 at centres
       do j = 1, im
          do i = 1, im
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') &
                  26, f, i, j, csg(5, i, j)
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') &
                  27, f, i, j, 1.0d0 / max(tiny_number, ssg(5, i, j)**2)
          enddo
       enddo
       ! cosa / sina / rsina at B-nodes (sentinel at the 4 cube vertices;
       ! rsina additionally sentinel on all panel edges — upstream poison)
       do j = 1, npx
          do i = 1, npx
             if ( (i==1 .or. i==npx) .and. (j==1 .or. j==npx) ) then
                cb = sentinel
                sb = sentinel
             else
                cb = 0.5d0 * (csg(8, i-1, j-1) + csg(6, i, j))
                sb = 0.5d0 * (ssg(8, i-1, j-1) + ssg(6, i, j))
             endif
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 28, f, i, j, cb
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 29, f, i, j, sb
             if (i==1 .or. i==npx .or. j==1 .or. j==npx) then
                val = sentinel
             else
                val = 1.0d0 / max(tiny_number, sb**2)
             endif
             write(u,'(I2,1X,I2,1X,I4,1X,I4,1X,ES26.17E3)') 30, f, i, j, val
          enddo
       enddo
    enddo
    close(u)
    deallocate(csg, ssg)
  end subroutine angle_oracle

end program fv3_gnomonic_oracle
