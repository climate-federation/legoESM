! VERBATIM extraction from GFDL_atmos_cubed_sphere @ 6f658bd0 model/sw_core.F90
! for the phase-4 c_sw one-step oracle: c_sw, divergence_corner, d2a2c_vect,
! fill3_4corners, fill2_4corners, fill_4corners. Only the module wrapper and
! type shims (fv3_swcore_shim.F90) are new; subroutine bodies are unmodified.
module sw_core_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, &
                             fv_flags_type
  implicit none
  ! sw_core.F90 module-level constants (verbatim; big_number is the
  ! non-OVERLOAD_R4 production branch value 1.E30 — codex r1 P1-1)
  real, parameter:: r3 = 1./3.
  real, parameter:: t11=27./28., t12=-13./28., t13=3./7., t14=6./7., t15=3./28.
  real, parameter:: s11=11./14., s13=-13./14., s14=4./7., s15=3./14.
  real, parameter:: near_zero = 1.E-9
  real, parameter:: big_number = 1.E30
  real, parameter:: p1 =  7./12.
  real, parameter:: p2 = -1./12.
  real, parameter:: a1 =  0.5625
  real, parameter:: a2 = -0.0625
  real, parameter:: c1 = -2./14.
  real, parameter:: c2 = 11./14.
  real, parameter:: c3 =  5./14.
contains
   subroutine c_sw(delpc, delp, ptc, pt, u,v, w, uc,vc, ua,va, wc,  &
                   ut, vt, divg_d, nord, dt2, hydrostatic, dord4, &
                   bd, gridstruct, flagstruct)

      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed+1) :: u, vc
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ) :: v, uc
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed  ) :: delp,  pt,  ua, va, ut, vt
      real, intent(INOUT), dimension(bd%isd:      ,  bd%jsd:        ) :: w
      real, intent(OUT  ), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed  ) :: delpc, ptc, wc
      real, intent(OUT  ), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) :: divg_d
      integer, intent(IN) :: nord
      real,    intent(IN) :: dt2
      logical, intent(IN) :: hydrostatic
      logical, intent(IN) :: dord4
      type(fv_grid_type),  intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct

! Local:
      logical:: sw_corner, se_corner, ne_corner, nw_corner
      real, dimension(bd%is-1:bd%ie+1,bd%js-1:bd%je+1):: vort, ke
      real, dimension(bd%is-1:bd%ie+2,bd%js-1:bd%je+1):: fx, fx1, fx2
      real, dimension(bd%is-1:bd%ie+1,bd%js-1:bd%je+2):: fy, fy1, fy2
      real :: dt4
      integer :: i,j, is2, ie1
      integer iep1, jep1

      integer :: is,  ie,  js,  je
      integer :: isd, ied, jsd, jed
      integer :: npx, npy
      logical :: bounded_domain

      real, pointer, dimension(:,:,:) :: sin_sg, cos_sg
      real, pointer, dimension(:,:)   :: cosa_u, cosa_v
      real, pointer, dimension(:,:)   :: sina_u, sina_v

      real, pointer, dimension(:,:) :: dx, dy, dxc, dyc

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je
      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

      npx = flagstruct%npx
      npy = flagstruct%npy
      bounded_domain = gridstruct%bounded_domain

      sin_sg  => gridstruct%sin_sg
      cos_sg  => gridstruct%cos_sg
      cosa_u  => gridstruct%cosa_u
      cosa_v  => gridstruct%cosa_v
      sina_u  => gridstruct%sina_u
      sina_v  => gridstruct%sina_v
      dx      => gridstruct%dx
      dy      => gridstruct%dy
      dxc     => gridstruct%dxc
      dyc     => gridstruct%dyc

      sw_corner = gridstruct%sw_corner
      se_corner = gridstruct%se_corner
      nw_corner = gridstruct%nw_corner
      ne_corner = gridstruct%ne_corner

      iep1 = ie+1; jep1 = je+1

      call d2a2c_vect(u, v, ua, va, uc, vc, ut, vt, dord4, gridstruct, bd, &
                      npx, npy, bounded_domain, flagstruct%grid_type)

      if( nord > 0 ) then
         if (bounded_domain) then
            call divergence_corner_nest(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
         else
            call divergence_corner(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
         endif
      endif

      do j=js-1,jep1
         do i=is-1,iep1+1
            if (ut(i,j) > 0.) then
                ut(i,j) = dt2*ut(i,j)*dy(i,j)*sin_sg(i-1,j,3)
            else
                ut(i,j) = dt2*ut(i,j)*dy(i,j)*sin_sg(i,j,1)
            end if
         enddo
      enddo
      do j=js-1,je+2
         do i=is-1,iep1
            if (vt(i,j) > 0.) then
                vt(i,j) = dt2*vt(i,j)*dx(i,j)*sin_sg(i,j-1,4)
            else
                vt(i,j) = dt2*vt(i,j)*dx(i,j)*sin_sg(i,j,  2)
            end if
         enddo
      enddo

!----------------
! Transport delp:
!----------------
! Xdir:
      if (flagstruct%grid_type < 3 .and. .not. bounded_domain) call fill2_4corners(delp, pt, 1, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)

      if ( hydrostatic ) then
#ifdef SW_DYNAMICS
           do j=js-1,jep1
              do i=is-1,ie+2
                 if ( ut(i,j) > 0. ) then
                      fx1(i,j) = delp(i-1,j)
                 else
                      fx1(i,j) = delp(i,j)
                 endif
                 fx1(i,j) =  ut(i,j)*fx1(i,j)
              enddo
           enddo
#else
           do j=js-1,jep1
              do i=is-1,ie+2
                 if ( ut(i,j) > 0. ) then
                      fx1(i,j) = delp(i-1,j)
                       fx(i,j) =   pt(i-1,j)
                 else
                      fx1(i,j) = delp(i,j)
                       fx(i,j) =   pt(i,j)
                 endif
                 fx1(i,j) =  ut(i,j)*fx1(i,j)
                  fx(i,j) = fx1(i,j)* fx(i,j)
              enddo
           enddo
#endif
      else
           if (flagstruct%grid_type < 3)   &
               call fill_4corners(w, 1, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
           do j=js-1,je+1
              do i=is-1,ie+2
                 if ( ut(i,j) > 0. ) then
                      fx1(i,j) = delp(i-1,j)
                       fx(i,j) =   pt(i-1,j)
                      fx2(i,j) =    w(i-1,j)
                 else
                      fx1(i,j) = delp(i,j)
                       fx(i,j) =   pt(i,j)
                      fx2(i,j) =    w(i,j)
                 endif
                 fx1(i,j) =  ut(i,j)*fx1(i,j)
                  fx(i,j) = fx1(i,j)* fx(i,j)
                 fx2(i,j) = fx1(i,j)*fx2(i,j)
              enddo
           enddo
      endif

! Ydir:
      if (flagstruct%grid_type < 3 .and. .not. bounded_domain) call fill2_4corners(delp, pt, 2, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
      if ( hydrostatic ) then
           do j=js-1,jep1+1
              do i=is-1,iep1
                 if ( vt(i,j) > 0. ) then
                      fy1(i,j) = delp(i,j-1)
                       fy(i,j) =   pt(i,j-1)
                 else
                      fy1(i,j) = delp(i,j)
                       fy(i,j) =   pt(i,j)
                 endif
                 fy1(i,j) =  vt(i,j)*fy1(i,j)
                  fy(i,j) = fy1(i,j)* fy(i,j)
              enddo
           enddo
           do j=js-1,jep1
              do i=is-1,iep1
                 delpc(i,j) = delp(i,j) + (fx1(i,j)-fx1(i+1,j)+fy1(i,j)-fy1(i,j+1))*gridstruct%rarea(i,j)
#ifdef SW_DYNAMICS
                   ptc(i,j) = pt(i,j)
#else
                   ptc(i,j) = (pt(i,j)*delp(i,j) +   &
                              (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))*gridstruct%rarea(i,j))/delpc(i,j)
#endif
              enddo
           enddo
      else
           if (flagstruct%grid_type < 3) call fill_4corners(w, 2, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
           do j=js-1,je+2
              do i=is-1,ie+1
                 if ( vt(i,j) > 0. ) then
                      fy1(i,j) = delp(i,j-1)
                       fy(i,j) =   pt(i,j-1)
                      fy2(i,j) =    w(i,j-1)
                 else
                      fy1(i,j) = delp(i,j)
                       fy(i,j) =   pt(i,j)
                      fy2(i,j) =    w(i,j)
                 endif
                 fy1(i,j) =  vt(i,j)*fy1(i,j)
                  fy(i,j) = fy1(i,j)* fy(i,j)
                 fy2(i,j) = fy1(i,j)*fy2(i,j)
              enddo
           enddo
           do j=js-1,je+1
              do i=is-1,ie+1
                 delpc(i,j) = delp(i,j) + (fx1(i,j)-fx1(i+1,j)+fy1(i,j)-fy1(i,j+1))*gridstruct%rarea(i,j)
                   ptc(i,j) = (pt(i,j)*delp(i,j) +   &
                              (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))*gridstruct%rarea(i,j))/delpc(i,j)
                    wc(i,j) = (w(i,j)*delp(i,j) + (fx2(i,j)-fx2(i+1,j) +    &
                               fy2(i,j)-fy2(i,j+1))*gridstruct%rarea(i,j))/delpc(i,j)
              enddo
           enddo
      endif

!------------
! Compute KE:
!------------

!Since uc = u*, i.e. the covariant wind perpendicular to the face edge, if we want to compute kinetic energy we will need the true coordinate-parallel covariant wind, computed through u = uc*sina + v*cosa.
!Use the alpha for the cell KE is being computed in.
!!! TO DO:
!!! Need separate versions for nesting/single-tile
!!!   and for cubed-sphere
      if (bounded_domain .or. flagstruct%grid_type >=3 ) then
         do j=js-1,jep1
         do i=is-1,iep1
            if ( ua(i,j) > 0. ) then
               ke(i,j) = uc(i,j)
            else
               ke(i,j) = uc(i+1,j)
            endif
         enddo
         enddo
         do j=js-1,jep1
         do i=is-1,iep1
            if ( va(i,j) > 0. ) then
               vort(i,j) = vc(i,j)
            else
               vort(i,j) = vc(i,j+1)
            endif
         enddo
         enddo
      else
         do j=js-1,jep1
         do i=is-1,iep1
            if ( ua(i,j) > 0. ) then
               if ( i==1 ) then
                  ke(1,j) = uc(1,  j)*sin_sg(1,j,1)+v(1,j)*cos_sg(1,j,1)
               elseif ( i==npx  ) then
                  ke(i,j) = uc(npx,j)*sin_sg(npx,j,1)+v(npx,j)*cos_sg(npx,j,1)
               else
                  ke(i,j) = uc(i,j)
               endif
            else
               if ( i==0   ) then
                  ke(0,j) = uc(1,  j)*sin_sg(0,j,3)+v(1,j)*cos_sg(0,j,3)
               elseif ( i==(npx-1)   ) then
                  ke(i,j) = uc(npx,j)*sin_sg(npx-1,j,3)+v(npx,j)*cos_sg(npx-1,j,3)
               else
                  ke(i,j) = uc(i+1,j)
               endif
            endif
         enddo
         enddo
         do j=js-1,jep1
            do i=is-1,iep1
               if ( va(i,j) > 0. ) then
                  if ( j==1   ) then
                     vort(i,1) = vc(i,  1)*sin_sg(i,1,2)+u(i,  1)*cos_sg(i,1,2)
                  elseif ( j==npy   ) then
                     vort(i,j) = vc(i,npy)*sin_sg(i,npy,2)+u(i,npy)*cos_sg(i,npy,2)
                  else
                     vort(i,j) = vc(i,j)
                  endif
               else
                  if ( j==0   ) then
                     vort(i,0) = vc(i,  1)*sin_sg(i,0,4)+u(i,  1)*cos_sg(i,0,4)
                  elseif ( j==(npy-1)  ) then
                     vort(i,j) = vc(i,npy)*sin_sg(i,npy-1,4)+u(i,npy)*cos_sg(i,npy-1,4)
                  else
                     vort(i,j) = vc(i,j+1)
                  endif
               endif
            enddo
         enddo
      endif

      dt4 = 0.5*dt2
      do j=js-1,jep1
         do i=is-1,iep1
            ke(i,j) = dt4*(ua(i,j)*ke(i,j) + va(i,j)*vort(i,j))
         enddo
      enddo

!------------------------------
! Compute circulation on C grid
!------------------------------
! To consider using true co-variant winds at face edges?
      do j=js-1,je+1
         do i=is,ie+1
            fx(i,j) = uc(i,j) * dxc(i,j)
         enddo
      enddo

      do j=js,je+1
         do i=is-1,ie+1
            fy(i,j) = vc(i,j) * dyc(i,j)
         enddo
      enddo

      do j=js,je+1
         do i=is,ie+1
            vort(i,j) =  fx(i,j-1) - fx(i,j) - fy(i-1,j) + fy(i,j)
         enddo
      enddo

! Remove the extra term at the corners:
      if ( sw_corner ) vort(1,    1) = vort(1,    1) + fy(0,   1)
      if ( se_corner ) vort(npx  ,1) = vort(npx,  1) - fy(npx, 1)
      if ( ne_corner ) vort(npx,npy) = vort(npx,npy) - fy(npx,npy)
      if ( nw_corner ) vort(1,  npy) = vort(1,  npy) + fy(0,  npy)

!----------------------------
! Compute absolute vorticity
!----------------------------
      do j=js,je+1
         do i=is,ie+1
            vort(i,j) = gridstruct%fC(i,j) + gridstruct%rarea_c(i,j) * vort(i,j)
         enddo
      enddo

!----------------------------------
! Transport absolute vorticity:
!----------------------------------
!To go from v to contravariant v at the edges, we divide by sin_sg;
! but we then must multiply by sin_sg to get the proper flux.
! These cancel, leaving us with fy1 = dt2*v at the edges.
! (For the same reason we only divide by sin instead of sin**2 in the interior)

!! TO DO: separate versions for nesting/single-tile and cubed-sphere
      if (bounded_domain .or. flagstruct%grid_type >= 3) then
         do j=js,je
            do i=is,iep1
               fy1(i,j) = dt2*(v(i,j)-uc(i,j)*cosa_u(i,j))/sina_u(i,j)
               if ( fy1(i,j) > 0. ) then
                  fy(i,j) = vort(i,j)
               else
                  fy(i,j) = vort(i,j+1)
               endif
            enddo
         enddo
         do j=js,jep1
            do i=is,ie
               fx1(i,j) = dt2*(u(i,j)-vc(i,j)*cosa_v(i,j))/sina_v(i,j)
               if ( fx1(i,j) > 0. ) then
                  fx(i,j) = vort(i,j)
               else
                  fx(i,j) = vort(i+1,j)
               endif
            enddo
         enddo
      else
         do j=js,je
!DEC$ VECTOR ALWAYS
            do i=is,iep1
               if ( ( i==1 .or. i==npx ) ) then
                  fy1(i,j) = dt2*v(i,j)
               else
                  fy1(i,j) = dt2*(v(i,j)-uc(i,j)*cosa_u(i,j))/sina_u(i,j)
               endif
               if ( fy1(i,j) > 0. ) then
                  fy(i,j) = vort(i,j)
               else
                  fy(i,j) = vort(i,j+1)
               endif
            enddo
         enddo
         do j=js,jep1
            if ( ( j==1 .or. j==npy ) ) then
!DEC$ VECTOR ALWAYS
               do i=is,ie
                  fx1(i,j) = dt2*u(i,j)
                  if ( fx1(i,j) > 0. ) then
                     fx(i,j) = vort(i,j)
                  else
                     fx(i,j) = vort(i+1,j)
                  endif
               enddo
            else
!DEC$ VECTOR ALWAYS
               do i=is,ie
                  fx1(i,j) = dt2*(u(i,j)-vc(i,j)*cosa_v(i,j))/sina_v(i,j)
                  if ( fx1(i,j) > 0. ) then
                     fx(i,j) = vort(i,j)
                  else
                     fx(i,j) = vort(i+1,j)
                  endif
               enddo
            endif
         enddo
      endif

! Update time-centered winds on the C-Grid
      do j=js,je
         do i=is,iep1
            uc(i,j) = uc(i,j) + fy1(i,j)*fy(i,j) + gridstruct%rdxc(i,j)*(ke(i-1,j)-ke(i,j))
         enddo
      enddo
      do j=js,jep1
         do i=is,ie
            vc(i,j) = vc(i,j) - fx1(i,j)*fx(i,j) + gridstruct%rdyc(i,j)*(ke(i,j-1)-ke(i,j))
         enddo
      enddo

   end subroutine c_sw
 subroutine divergence_corner(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
 type(fv_grid_bounds_type), intent(IN) :: bd
 real, intent(in),  dimension(bd%isd:bd%ied,  bd%jsd:bd%jed+1):: u
 real, intent(in),  dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v
 real, intent(in),  dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: ua, va
 real, intent(out), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed+1):: divg_d
 type(fv_grid_type), intent(IN), target :: gridstruct
 type(fv_flags_type), intent(IN), target :: flagstruct
! local
 real uf(bd%is-2:bd%ie+2,bd%js-1:bd%je+2)
 real vf(bd%is-1:bd%ie+2,bd%js-2:bd%je+2)
 integer i,j
 integer is2, ie1

 real, pointer, dimension(:,:,:) :: sin_sg, cos_sg
 real, pointer, dimension(:,:) ::  dxc,dyc

      integer :: is,  ie,  js,  je
      integer :: npx, npy
      logical :: bounded_domain

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je

      npx = flagstruct%npx
      npy = flagstruct%npy
      bounded_domain = gridstruct%bounded_domain

      sin_sg     => gridstruct%sin_sg
      cos_sg     => gridstruct%cos_sg
      dxc        => gridstruct%dxc
      dyc        => gridstruct%dyc

 if (bounded_domain) then
    is2 = is;        ie1 = ie+1
 else
    is2 = max(2,is); ie1 = min(npx-1,ie+1)
 end if

    if (flagstruct%grid_type > 3) then
        do j=js-1,je+2
           do i=is-2,ie+2
              uf(i,j) = u(i,j)*dyc(i,j)
           enddo
        enddo
        do j=js-2,je+2
           do i=is-1,ie+2
              vf(i,j) = v(i,j)*dxc(i,j)
           enddo
        enddo
        do j=js-1,je+2
           do i=is-1,ie+2
              divg_d(i,j) = gridstruct%rarea_c(i,j)*(vf(i,j-1)-vf(i,j)+uf(i-1,j)-uf(i,j))
           enddo
        enddo
  else
!     9---4---8
!     |       |
!     1   5   3
!     |       |
!     6---2---7
    do j=js,je+1
       if ( j==1 .or. j==npy ) then
         do i=is-1,ie+1
            uf(i,j) = u(i,j)*dyc(i,j)*0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
         enddo
       else
         do i=is-1,ie+1
            uf(i,j) = (u(i,j)-0.25*(va(i,j-1)+va(i,j))*(cos_sg(i,j-1,4)+cos_sg(i,j,2)))   &
                                        * dyc(i,j)*0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
         enddo
       endif
    enddo

    do j=js-1,je+1
       do i=is2,ie1
          vf(i,j) = (v(i,j) - 0.25*(ua(i-1,j)+ua(i,j))*(cos_sg(i-1,j,3)+cos_sg(i,j,1)))  &
                                         *dxc(i,j)*0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))
       enddo
       if (  is   ==  1 ) vf(1,  j) = v(1,  j)*dxc(1,  j)*0.5*(sin_sg(0,j,3)+sin_sg(1,j,1))
       if ( (ie+1)==npx ) vf(npx,j) = v(npx,j)*dxc(npx,j)*0.5*(sin_sg(npx-1,j,3)+sin_sg(npx,j,1))
    enddo

    do j=js,je+1
       do i=is,ie+1
          divg_d(i,j) = vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j)
       enddo
    enddo

! Remove the extra term at the corners:
    if (gridstruct%sw_corner) divg_d(1,    1) = divg_d(1,    1) - vf(1,    0)
    if (gridstruct%se_corner) divg_d(npx,  1) = divg_d(npx,  1) - vf(npx,  0)
    if (gridstruct%ne_corner) divg_d(npx,npy) = divg_d(npx,npy) + vf(npx,npy)
    if (gridstruct%nw_corner) divg_d(1,  npy) = divg_d(1,  npy) + vf(1,  npy)

    do j=js,je+1
       do i=is,ie+1
          divg_d(i,j) = gridstruct%rarea_c(i,j)*divg_d(i,j)
       enddo
    enddo

  endif

 end subroutine divergence_corner

 subroutine divergence_corner_nest(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
 type(fv_grid_bounds_type), intent(IN) :: bd
 real, intent(in),  dimension(bd%isd:bd%ied,  bd%jsd:bd%jed+1):: u
 real, intent(in),  dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed):: v
 real, intent(in),  dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: ua, va
 real, intent(out), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed+1):: divg_d
 type(fv_grid_type), intent(IN), target :: gridstruct
 type(fv_flags_type), intent(IN), target :: flagstruct

! local
 real uf(bd%isd:bd%ied,bd%jsd:bd%jed+1)
 real vf(bd%isd:bd%ied+1,bd%jsd:bd%jed)
 integer i,j


  real, pointer, dimension(:,:) :: rarea_c

  real, pointer, dimension(:,:,:) :: sin_sg, cos_sg
  real, pointer, dimension(:,:)   :: cosa_u, cosa_v
  real, pointer, dimension(:,:)   :: sina_u, sina_v
  real, pointer, dimension(:,:) ::  dxc,dyc

      integer :: isd, ied, jsd, jed
      integer :: npx, npy

      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

      npx = flagstruct%npx
      npy = flagstruct%npy

      rarea_c    => gridstruct%rarea_c
      sin_sg     => gridstruct%sin_sg
      cos_sg     => gridstruct%cos_sg
      cosa_u     => gridstruct%cosa_u
      cosa_v     => gridstruct%cosa_v
      sina_u     => gridstruct%sina_u
      sina_v     => gridstruct%sina_v
      dxc        => gridstruct%dxc
      dyc        => gridstruct%dyc

 divg_d = 1.e25

    if (flagstruct%grid_type > 3) then
        do j=jsd,jed
           do i=isd,ied
              uf(i,j) = u(i,j)*dyc(i,j)
           enddo
        enddo
        do j=jsd,jed
           do i=isd,ied
              vf(i,j) = v(i,j)*dxc(i,j)
           enddo
        enddo
        do j=jsd+1,jed
           do i=isd+1,ied
              divg_d(i,j) = rarea_c(i,j)*(vf(i,j-1)-vf(i,j)+uf(i-1,j)-uf(i,j))
           enddo
        enddo
    else

       do j=jsd+1,jed
          do i=isd,ied
            uf(i,j) = (u(i,j)-0.25*(va(i,j-1)+va(i,j))*(cos_sg(i,j-1,4)+cos_sg(i,j,2)))   &
                                        * dyc(i,j)*0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
          enddo
       enddo

       do j=jsd,jed
          do i=isd+1,ied
             vf(i,j) = (v(i,j) - 0.25*(ua(i-1,j)+ua(i,j))*(cos_sg(i-1,j,3)+cos_sg(i,j,1)))  &
                  *dxc(i,j)*0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))
          enddo
       enddo

       do j=jsd+1,jed
          do i=isd+1,ied
             divg_d(i,j) = (vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j))*rarea_c(i,j)
          enddo
       enddo

    endif


end subroutine divergence_corner_nest

 subroutine divergence_corner_duo(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
 type(fv_grid_bounds_type), intent(IN) :: bd
 real, intent(in),  dimension(bd%isd:bd%ied,  bd%jsd:bd%jed+1):: u
 real, intent(in),  dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed):: v
 real, intent(in),  dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: ua, va
 real, intent(out), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed+1):: divg_d
 type(fv_grid_type), intent(IN), target :: gridstruct
 type(fv_flags_type), intent(IN), target :: flagstruct

! local
 real uf(bd%isd:bd%ied,bd%jsd:bd%jed+1)
 real vf(bd%isd:bd%ied+1,bd%jsd:bd%jed)
 integer i,j


  real, pointer, dimension(:,:) :: rarea_c

  real, pointer, dimension(:,:,:) :: sin_sg, cos_sg
  real, pointer, dimension(:,:)   :: cosa_u, cosa_v
  real, pointer, dimension(:,:)   :: sina_u, sina_v
  real, pointer, dimension(:,:) ::  dxc,dyc

      integer :: is, ie, js, je
      integer :: isd, ied, jsd, jed
      integer :: npx, npy

      is = bd%is
      ie = bd%ie
      js = bd%js
      je = bd%je
      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

      npx = flagstruct%npx
      npy = flagstruct%npy

      rarea_c    => gridstruct%rarea_c
      sin_sg     => gridstruct%sin_sg
      cos_sg     => gridstruct%cos_sg
      cosa_u     => gridstruct%cosa_u
      cosa_v     => gridstruct%cosa_v
      sina_u     => gridstruct%sina_u
      sina_v     => gridstruct%sina_v
      dxc        => gridstruct%dxc
      dyc        => gridstruct%dyc

 divg_d = 1.e25

    if (flagstruct%grid_type > 3) then
        do j=jsd,jed
           do i=isd,ied
              uf(i,j) = u(i,j)*dyc(i,j)
           enddo
        enddo
        do j=jsd,jed
           do i=isd,ied
              vf(i,j) = v(i,j)*dxc(i,j)
           enddo
        enddo
        do j=jsd+1,jed
           do i=isd+1,ied
              divg_d(i,j) = rarea_c(i,j)*(vf(i,j-1)-vf(i,j)+uf(i-1,j)-uf(i,j))
           enddo
        enddo
    else

       do j=jsd+1,jed
          do i=isd,ied
            uf(i,j) = (u(i,j)-0.25*(va(i,j-1)+va(i,j))*(cos_sg(i,j-1,4)+cos_sg(i,j,2)))   &
                                        * dyc(i,j)*0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
          enddo
       enddo

       do j=jsd,jed
          do i=isd+1,ied
             vf(i,j) = (v(i,j) - 0.25*(ua(i-1,j)+ua(i,j))*(cos_sg(i-1,j,3)+cos_sg(i,j,1)))  &
                  *dxc(i,j)*0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))
          enddo
       enddo

       do j=jsd+1,jed
          do i=isd+1,ied
             divg_d(i,j) = (vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j))*rarea_c(i,j)

       if (  is   ==  1 .and. i==is) divg_d(i,j)=0.
       if ( (ie+1)==npx .and. i==ie+1)  divg_d(i,j)= 0.
       if (   js==1   .and. j==1)  divg_d(i,j)=0.
       if ( je+1==npx .and. j==je+1)  divg_d(i,j)=0.

!improve rosby a bit more +100days but decrease mag very little
       if (  is   ==  1 .and. i==is+1) divg_d(i,j)=0.25*divg_d(i,j)
       if ( (ie+1)==npx .and. i==ie)  divg_d(i,j)= 0.25*divg_d(i,j)
       if (   js==1   .and. j==1+1)  divg_d(i,j)=0.25*divg_d(i,j)
       if ( je+1==npx .and. j==je)  divg_d(i,j)=0.25*divg_d(i,j)
          enddo
       enddo

    endif


end subroutine divergence_corner_duo


 subroutine d2a2c_vect(u, v, ua, va, uc, vc, ut, vt, dord4, gridstruct, &
                       bd, npx, npy, bounded_domain, grid_type)
  type(fv_grid_bounds_type), intent(IN) :: bd
  logical, intent(in):: dord4
  real, intent(in) ::  u(bd%isd:bd%ied,bd%jsd:bd%jed+1)
  real, intent(in) ::  v(bd%isd:bd%ied+1,bd%jsd:bd%jed)
  real, intent(out), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: uc
  real, intent(out), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: vc
  real, intent(out), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed  ):: ua, va, ut, vt
  integer, intent(IN) :: npx, npy, grid_type
  logical, intent(IN) :: bounded_domain
  type(fv_grid_type), intent(IN), target :: gridstruct
! Local
  real, dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: utmp, vtmp
  integer npt, i, j, ifirst, ilast, id
  integer :: is,  ie,  js,  je
  integer :: isd, ied, jsd, jed

  real, pointer, dimension(:,:,:) :: sin_sg
  real, pointer, dimension(:,:)   :: cosa_u, cosa_v, cosa_s
  real, pointer, dimension(:,:)   :: rsin_u, rsin_v, rsin2
  real, pointer, dimension(:,:)   :: dxa,dya

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je
      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

      sin_sg    => gridstruct%sin_sg
      cosa_u    => gridstruct%cosa_u
      cosa_v    => gridstruct%cosa_v
      cosa_s    => gridstruct%cosa_s
      rsin_u    => gridstruct%rsin_u
      rsin_v    => gridstruct%rsin_v
      rsin2     => gridstruct%rsin2
      dxa       => gridstruct%dxa
      dya       => gridstruct%dya

  if ( dord4 ) then
       id = 1
  else
       id = 0
  endif

  if (grid_type < 3 .and. .not. bounded_domain) then
     npt = 4
  else
     npt = -2
  endif

! Initialize the non-existing corner regions
  utmp(:,:) = big_number
  vtmp(:,:) = big_number

 if ( bounded_domain) then

     do j=jsd+1,jed-1
        do i=isd,ied
           utmp(i,j) = a2*(u(i,j-1)+u(i,j+2)) + a1*(u(i,j)+u(i,j+1))
        enddo
     enddo
     do i=isd,ied
        j = jsd
        utmp(i,j) = 0.5*(u(i,j)+u(i,j+1))
        j = jed
        utmp(i,j) = 0.5*(u(i,j)+u(i,j+1))
     end do

     do j=jsd,jed
        do i=isd+1,ied-1
           vtmp(i,j) = a2*(v(i-1,j)+v(i+2,j)) + a1*(v(i,j)+v(i+1,j))
        enddo
        i = isd
        vtmp(i,j) = 0.5*(v(i,j)+v(i+1,j))
        i = ied
        vtmp(i,j) = 0.5*(v(i,j)+v(i+1,j))
     enddo

     do j=jsd,jed
        do i=isd,ied
           ua(i,j) = (utmp(i,j)-vtmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
           va(i,j) = (vtmp(i,j)-utmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
        enddo
     enddo

 else
     !----------
     ! Interior:
     !----------
     do j=max(npt,js-1),min(npy-npt,je+1)
        do i=max(npt,isd),min(npx-npt,ied)
           utmp(i,j) = a2*(u(i,j-1)+u(i,j+2)) + a1*(u(i,j)+u(i,j+1))
        enddo
     enddo
     do j=max(npt,jsd),min(npy-npt,jed)
        do i=max(npt,is-1),min(npx-npt,ie+1)
           vtmp(i,j) = a2*(v(i-1,j)+v(i+2,j)) + a1*(v(i,j)+v(i+1,j))
        enddo
     enddo

     !----------
     ! edges:
     !----------
     if (grid_type < 3) then

        if ( js==1 .or. jsd<npt) then
           do j=jsd,npt-1
              do i=isd,ied
                 utmp(i,j) = 0.5*(u(i,j) + u(i,j+1))
                 vtmp(i,j) = 0.5*(v(i,j) + v(i+1,j))
              enddo
           enddo
        endif
        if ( (je+1)==npy .or. jed>=(npy-npt)) then
           do j=npy-npt+1,jed
              do i=isd,ied
                 utmp(i,j) = 0.5*(u(i,j) + u(i,j+1))
                 vtmp(i,j) = 0.5*(v(i,j) + v(i+1,j))
              enddo
           enddo
        endif

        if ( is==1 .or. isd<npt ) then
           do j=max(npt,jsd),min(npy-npt,jed)
              do i=isd,npt-1
                 utmp(i,j) = 0.5*(u(i,j) + u(i,j+1))
                 vtmp(i,j) = 0.5*(v(i,j) + v(i+1,j))
              enddo
           enddo
        endif
        if ( (ie+1)==npx .or. ied>=(npx-npt)) then
           do j=max(npt,jsd),min(npy-npt,jed)
              do i=npx-npt+1,ied
                 utmp(i,j) = 0.5*(u(i,j) + u(i,j+1))
                 vtmp(i,j) = 0.5*(v(i,j) + v(i+1,j))
              enddo
           enddo
        endif

     endif

! Contra-variant components at cell center:
     do j=js-1-id,je+1+id
        do i=is-1-id,ie+1+id
           ua(i,j) = (utmp(i,j)-vtmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
           va(i,j) = (vtmp(i,j)-utmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
        enddo
     enddo

 end if

! A -> C
!--------------
! Fix the edges
!--------------
! Xdir:
     if( gridstruct%sw_corner ) then
         do i=-2,0
            utmp(i,0) = -vtmp(0,1-i)
         enddo
     endif
     if( gridstruct%se_corner ) then
         do i=0,2
            utmp(npx+i,0) = vtmp(npx,i+1)
         enddo
     endif
     if( gridstruct%ne_corner ) then
         do i=0,2
            utmp(npx+i,npy) = -vtmp(npx,je-i)
         enddo
     endif
     if( gridstruct%nw_corner ) then
         do i=-2,0
            utmp(i,npy) = vtmp(0,je+i)
         enddo
     endif

  if (grid_type < 3 .and. .not. bounded_domain) then
     ifirst = max(3,    is-1)
     ilast  = min(npx-2,ie+2)
  else
     ifirst = is-1
     ilast  = ie+2
  endif
!---------------------------------------------
! 4th order interpolation for interior points:
!---------------------------------------------
     do j=js-1,je+1
        do i=ifirst,ilast
           uc(i,j) = a2*(utmp(i-2,j)+utmp(i+1,j)) + a1*(utmp(i-1,j)+utmp(i,j))
           ut(i,j) = (uc(i,j) - v(i,j)*cosa_u(i,j))*rsin_u(i,j)
        enddo
     enddo

 if (grid_type < 3) then
! Xdir:
     if( gridstruct%sw_corner ) then
         ua(-1,0) = -va(0,2)
         ua( 0,0) = -va(0,1)
     endif
     if( gridstruct%se_corner ) then
         ua(npx,  0) = va(npx,1)
         ua(npx+1,0) = va(npx,2)
     endif
     if( gridstruct%ne_corner ) then
         ua(npx,  npy) = -va(npx,npy-1)
         ua(npx+1,npy) = -va(npx,npy-2)
     endif
     if( gridstruct%nw_corner ) then
         ua(-1,npy) = va(0,npy-2)
         ua( 0,npy) = va(0,npy-1)
     endif

     if( is==1 .and. .not. bounded_domain  ) then
        do j=js-1,je+1
           uc(0,j) = c1*utmp(-2,j) + c2*utmp(-1,j) + c3*utmp(0,j)
           ut(1,j) = edge_interpolate4(ua(-1:2,j), dxa(-1:2,j))
           !Want to use the UPSTREAM value
           if (ut(1,j) > 0.) then
              uc(1,j) = ut(1,j)*sin_sg(0,j,3)
           else
              uc(1,j) = ut(1,j)*sin_sg(1,j,1)
           end if
           uc(2,j) = c1*utmp(3,j) + c2*utmp(2,j) + c3*utmp(1,j)
           ut(0,j) = (uc(0,j) - v(0,j)*cosa_u(0,j))*rsin_u(0,j)
           ut(2,j) = (uc(2,j) - v(2,j)*cosa_u(2,j))*rsin_u(2,j)
        enddo
     endif

     if( (ie+1)==npx  .and. .not. bounded_domain ) then
        do j=js-1,je+1
           uc(npx-1,j) = c1*utmp(npx-3,j)+c2*utmp(npx-2,j)+c3*utmp(npx-1,j)
           ut(npx,  j) = edge_interpolate4(ua(npx-2:npx+1,j), dxa(npx-2:npx+1,j))
           if (ut(npx,j) > 0.) then
               uc(npx,j) = ut(npx,j)*sin_sg(npx-1,j,3)
           else
               uc(npx,j) = ut(npx,j)*sin_sg(npx,j,1)
           end if
           uc(npx+1,j) = c3*utmp(npx,j) + c2*utmp(npx+1,j) + c1*utmp(npx+2,j)
           ut(npx-1,j) = (uc(npx-1,j)-v(npx-1,j)*cosa_u(npx-1,j))*rsin_u(npx-1,j)
           ut(npx+1,j) = (uc(npx+1,j)-v(npx+1,j)*cosa_u(npx+1,j))*rsin_u(npx+1,j)
        enddo
     endif

 endif

!------
! Ydir:
!------
     if( gridstruct%sw_corner ) then
         do j=-2,0
            vtmp(0,j) = -utmp(1-j,0)
         enddo
     endif
     if( gridstruct%nw_corner ) then
         do j=0,2
            vtmp(0,npy+j) = utmp(j+1,npy)
         enddo
     endif
     if( gridstruct%se_corner ) then
         do j=-2,0
            vtmp(npx,j) = utmp(ie+j,0)
         enddo
     endif
     if( gridstruct%ne_corner ) then
         do j=0,2
            vtmp(npx,npy+j) = -utmp(ie-j,npy)
         enddo
     endif
     if( gridstruct%sw_corner ) then
         va(0,-1) = -ua(2,0)
         va(0, 0) = -ua(1,0)
     endif
     if( gridstruct%se_corner ) then
         va(npx, 0) = ua(npx-1,0)
         va(npx,-1) = ua(npx-2,0)
     endif
     if( gridstruct%ne_corner ) then
         va(npx,npy  ) = -ua(npx-1,npy)
         va(npx,npy+1) = -ua(npx-2,npy)
     endif
     if( gridstruct%nw_corner ) then
         va(0,npy)   = ua(1,npy)
         va(0,npy+1) = ua(2,npy)
     endif

 if (grid_type < 3) then

     do j=js-1,je+2
      if ( j==1 .and. .not. bounded_domain  ) then
        do i=is-1,ie+1
           vt(i,j) = edge_interpolate4(va(i,-1:2), dya(i,-1:2))
           if (vt(i,j) > 0.) then
              vc(i,j) = vt(i,j)*sin_sg(i,j-1,4)
           else
              vc(i,j) = vt(i,j)*sin_sg(i,j,2)
           end if
        enddo
      elseif ( j==0 .or. j==(npy-1) .and. .not. bounded_domain  ) then
        do i=is-1,ie+1
           vc(i,j) = c1*vtmp(i,j-2) + c2*vtmp(i,j-1) + c3*vtmp(i,j)
           vt(i,j) = (vc(i,j) - u(i,j)*cosa_v(i,j))*rsin_v(i,j)
        enddo
      elseif ( j==2 .or. j==(npy+1)  .and. .not. bounded_domain ) then
        do i=is-1,ie+1
           vc(i,j) = c1*vtmp(i,j+1) + c2*vtmp(i,j) + c3*vtmp(i,j-1)
           vt(i,j) = (vc(i,j) - u(i,j)*cosa_v(i,j))*rsin_v(i,j)
        enddo
      elseif ( j==npy .and. .not. bounded_domain  ) then
        do i=is-1,ie+1
           vt(i,j) = edge_interpolate4(va(i,j-2:j+1), dya(i,j-2:j+1))
           if (vt(i,j) > 0.) then
              vc(i,j) = vt(i,j)*sin_sg(i,j-1,4)
           else
              vc(i,j) = vt(i,j)*sin_sg(i,j,2)
           end if
        enddo
      else
! 4th order interpolation for interior points:
        do i=is-1,ie+1
           vc(i,j) = a2*(vtmp(i,j-2)+vtmp(i,j+1)) + a1*(vtmp(i,j-1)+vtmp(i,j))
           vt(i,j) = (vc(i,j) - u(i,j)*cosa_v(i,j))*rsin_v(i,j)
        enddo
      endif
     enddo
 else
! 4th order interpolation:
       do j=js-1,je+2
          do i=is-1,ie+1
             vc(i,j) = a2*(vtmp(i,j-2)+vtmp(i,j+1)) + a1*(vtmp(i,j-1)+vtmp(i,j))
             vt(i,j) = vc(i,j)
          enddo
       enddo
 endif

 end subroutine d2a2c_vect
 subroutine fill3_4corners(q1, q2, q3, dir, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
  type(fv_grid_bounds_type), intent(IN) :: bd
! This routine fill the 4 corners of the scalar fileds only as needed by c_core
  integer, intent(in):: dir                ! 1: x-dir; 2: y-dir
  real, intent(inout):: q1(bd%isd:bd%ied,bd%jsd:bd%jed)
  real, intent(inout):: q2(bd%isd:bd%ied,bd%jsd:bd%jed)
  real, intent(inout):: q3(bd%isd:bd%ied,bd%jsd:bd%jed)
  logical, intent(IN) :: sw_corner, se_corner, ne_corner, nw_corner
  integer, intent(IN) :: npx, npy
  integer i,j

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

  select case(dir)
  case(1)
      if ( sw_corner ) then
          q1(-1,0) = q1(0,2); q1(0,0) = q1(0,1); q1(0,-1) = q1(-1,1)
          q2(-1,0) = q2(0,2); q2(0,0) = q2(0,1); q2(0,-1) = q2(-1,1)
          q3(-1,0) = q3(0,2); q3(0,0) = q3(0,1); q3(0,-1) = q3(-1,1)
      endif
      if ( se_corner ) then
          q1(npx+1,0) = q1(npx,2); q1(npx,0) = q1(npx,1); q1(npx,-1) = q1(npx+1,1)
          q2(npx+1,0) = q2(npx,2); q2(npx,0) = q2(npx,1); q2(npx,-1) = q2(npx+1,1)
          q3(npx+1,0) = q3(npx,2); q3(npx,0) = q3(npx,1); q3(npx,-1) = q3(npx+1,1)
      endif
      if ( ne_corner ) then
          q1(npx,npy) = q1(npx,npy-1); q1(npx+1,npy) = q1(npx,npy-2); q1(npx,npy+1) = q1(npx+1,npy-1)
          q2(npx,npy) = q2(npx,npy-1); q2(npx+1,npy) = q2(npx,npy-2); q2(npx,npy+1) = q2(npx+1,npy-1)
          q3(npx,npy) = q3(npx,npy-1); q3(npx+1,npy) = q3(npx,npy-2); q3(npx,npy+1) = q3(npx+1,npy-1)
      endif
      if ( nw_corner ) then
          q1(0,npy) = q1(0,npy-1); q1(-1,npy) = q1(0,npy-2); q1(0,npy+1) = q1(-1,npy-1)
          q2(0,npy) = q2(0,npy-1); q2(-1,npy) = q2(0,npy-2); q2(0,npy+1) = q2(-1,npy-1)
          q3(0,npy) = q3(0,npy-1); q3(-1,npy) = q3(0,npy-2); q3(0,npy+1) = q3(-1,npy-1)
      endif

  case(2)
      if ( sw_corner ) then
          q1(0,0) = q1(1,0); q1(0,-1) = q1(2,0); q1(-1,0) = q1(1,-1)
          q2(0,0) = q2(1,0); q2(0,-1) = q2(2,0); q2(-1,0) = q2(1,-1)
          q3(0,0) = q3(1,0); q3(0,-1) = q3(2,0); q3(-1,0) = q3(1,-1)
      endif
      if ( se_corner ) then
          q1(npx,0) = q1(npx-1,0); q1(npx,-1) = q1(npx-2,0); q1(npx+1,0) = q1(npx-1,-1)
          q2(npx,0) = q2(npx-1,0); q2(npx,-1) = q2(npx-2,0); q2(npx+1,0) = q2(npx-1,-1)
          q3(npx,0) = q3(npx-1,0); q3(npx,-1) = q3(npx-2,0); q3(npx+1,0) = q3(npx-1,-1)
      endif
      if ( ne_corner ) then
          q1(npx,npy) = q1(npx-1,npy); q1(npx,npy+1) = q1(npx-2,npy); q1(npx+1,npy) = q1(npx-1,npy+1)
          q2(npx,npy) = q2(npx-1,npy); q2(npx,npy+1) = q2(npx-2,npy); q2(npx+1,npy) = q2(npx-1,npy+1)
          q3(npx,npy) = q3(npx-1,npy); q3(npx,npy+1) = q3(npx-2,npy); q3(npx+1,npy) = q3(npx-1,npy+1)
      endif
      if ( nw_corner ) then
          q1(0,npy) = q1(1,npy); q1(0,npy+1) = q1(2,npy); q1(-1,npy) = q1(1,npy+1)
          q2(0,npy) = q2(1,npy); q2(0,npy+1) = q2(2,npy); q2(-1,npy) = q2(1,npy+1)
          q3(0,npy) = q3(1,npy); q3(0,npy+1) = q3(2,npy); q3(-1,npy) = q3(1,npy+1)
      endif

  end select
 end subroutine fill3_4corners


 subroutine fill2_4corners(q1, q2, dir, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
  type(fv_grid_bounds_type), intent(IN) :: bd
! This routine fill the 4 corners of the scalar fileds only as needed by c_core
  integer, intent(in):: dir                ! 1: x-dir; 2: y-dir
  real, intent(inout):: q1(bd%isd:bd%ied,bd%jsd:bd%jed)
  real, intent(inout):: q2(bd%isd:bd%ied,bd%jsd:bd%jed)
  logical, intent(IN) :: sw_corner, se_corner, ne_corner, nw_corner
  integer, intent(IN) :: npx, npy

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

  select case(dir)
  case(1)
      if ( sw_corner ) then
          q1(-1,0) = q1(0,2);    q1(0,0) = q1(0,1)
          q2(-1,0) = q2(0,2);    q2(0,0) = q2(0,1)
      endif
      if ( se_corner ) then
          q1(npx+1,0) = q1(npx,2); q1(npx,0) = q1(npx,1)
          q2(npx+1,0) = q2(npx,2); q2(npx,0) = q2(npx,1)
      endif
      if ( nw_corner ) then
          q1(0,npy) = q1(0,npy-1); q1(-1,npy) = q1(0,npy-2)
          q2(0,npy) = q2(0,npy-1); q2(-1,npy) = q2(0,npy-2)
      endif
      if ( ne_corner ) then
          q1(npx,npy) = q1(npx,npy-1); q1(npx+1,npy) = q1(npx,npy-2)
          q2(npx,npy) = q2(npx,npy-1); q2(npx+1,npy) = q2(npx,npy-2)
      endif

  case(2)
      if ( sw_corner ) then
          q1(0,0) = q1(1,0); q1(0,-1) = q1(2,0)
          q2(0,0) = q2(1,0); q2(0,-1) = q2(2,0)
      endif
      if ( se_corner ) then
          q1(npx,0) = q1(npx-1,0); q1(npx,-1) = q1(npx-2,0)
          q2(npx,0) = q2(npx-1,0); q2(npx,-1) = q2(npx-2,0)
      endif
      if ( nw_corner ) then
          q1(0,npy) = q1(1,npy); q1(0,npy+1) = q1(2,npy)
          q2(0,npy) = q2(1,npy); q2(0,npy+1) = q2(2,npy)
      endif
      if ( ne_corner ) then
          q1(npx,npy) = q1(npx-1,npy); q1(npx,npy+1) = q1(npx-2,npy)
          q2(npx,npy) = q2(npx-1,npy); q2(npx,npy+1) = q2(npx-2,npy)
      endif

  end select

 end subroutine fill2_4corners

 subroutine fill_4corners(q, dir, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
  type(fv_grid_bounds_type), intent(IN) :: bd
! This routine fill the 4 corners of the scalar fileds only as needed by c_core
  integer, intent(in):: dir                ! 1: x-dir; 2: y-dir
  real, intent(inout):: q(bd%isd:bd%ied,bd%jsd:bd%jed)
  logical, intent(IN) :: sw_corner, se_corner, ne_corner, nw_corner
  integer, intent(IN) :: npx, npy

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

  select case(dir)
  case(1)
      if ( sw_corner ) then
          q(-1,0) = q(0,2)
          q( 0,0) = q(0,1)
      endif
      if ( se_corner ) then
          q(npx+1,0) = q(npx,2)
          q(npx,  0) = q(npx,1)
      endif
      if ( nw_corner ) then
          q( 0,npy) = q(0,npy-1)
          q(-1,npy) = q(0,npy-2)
      endif
      if ( ne_corner ) then
          q(npx,  npy) = q(npx,npy-1)
          q(npx+1,npy) = q(npx,npy-2)
      endif

  case(2)
      if ( sw_corner ) then
          q(0, 0) = q(1,0)
          q(0,-1) = q(2,0)
      endif
      if ( se_corner ) then
          q(npx, 0) = q(npx-1,0)
          q(npx,-1) = q(npx-2,0)
      endif
      if ( nw_corner ) then
          q(0,npy  ) = q(1,npy)
          q(0,npy+1) = q(2,npy)
      endif
      if ( ne_corner ) then
          q(npx,npy  ) = q(npx-1,npy)
          q(npx,npy+1) = q(npx-2,npy)
      endif

  end select

 end subroutine fill_4corners

 real function edge_interpolate4(ua, dxa)

   real, intent(in) :: ua(4)
   real, intent(in) :: dxa(4)
   real:: t1, t2

   t1 = dxa(1) + dxa(2)
   t2 = dxa(3) + dxa(4)
   edge_interpolate4 = 0.5*( ((t1+dxa(2))*ua(2)-dxa(2)*ua(1)) / t1 + &
                             ((t2+dxa(3))*ua(3)-dxa(3)*ua(4)) / t2 )

 end function edge_interpolate4

end module sw_core_extract_mod
