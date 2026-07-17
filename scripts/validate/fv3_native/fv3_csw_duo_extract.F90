! Phase-4c duo c_sw one-step oracle module.
!
! VERBATIM authoritative symmetryclean sw_core.F90:79-494 c_sw (the version
! carrying the ``flagstruct%duogrid`` branches).  Kept in a SEPARATE module
! so the phase-4a-certified PLAIN c_sw in sw_core_extract_mod (the 6f658bd0
! tree, no duogrid) is untouched.  c_sw calls the DUO d2a2c_vect (from
! d2a2c_duo_extract_mod, dg-initialized branch) + divergence_corner{,_nest,
! _duo} + fill{2,}_4corners (from sw_core_extract_mod).
module c_sw_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use sw_core_extract_mod, only: divergence_corner, divergence_corner_nest, &
                                 divergence_corner_duo, fill2_4corners, &
                                 fill_4corners
  use d2a2c_duo_extract_mod, only: d2a2c_vect
  implicit none
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
      integer :: i,j
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

      !call d2a2c_vect(u, v, ua, va, uc, vc, ut, vt, dord4, gridstruct, bd, &
      !                npx, npy, bounded_domain, flagstruct%grid_type)
      call d2a2c_vect(u, v, ua, va, uc, vc, ut, vt, dord4, gridstruct, bd, &
                      npx, npy, .false., flagstruct%grid_type)

      if( nord > 0 ) then
         if (bounded_domain .and.  ( .not. flagstruct%duogrid )) then
            call divergence_corner_nest(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
         elseif (flagstruct%duogrid) then
            call divergence_corner_duo(u, v, ua, va, divg_d, gridstruct, flagstruct, bd)
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
      if (flagstruct%grid_type < 3 .and. .not. bounded_domain .and. (.not. flagstruct%duogrid)) call fill2_4corners(delp, pt, 1, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
      !if (flagstruct%grid_type < 3 .and. .not. bounded_domain) call fill2_4corners(delp, pt, 1, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)

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
           if (flagstruct%grid_type < 3 .and. (.not. flagstruct%duogrid ))   &
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
      if (flagstruct%grid_type < 3 .and. .not. bounded_domain   .and. (.not. flagstruct%duogrid )) call fill2_4corners(delp, pt, 2, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
      !if (flagstruct%grid_type < 3 .and. .not. bounded_domain  ) call fill2_4corners(delp, pt, 2, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
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
           if (flagstruct%grid_type < 3    .and. .not.(flagstruct%duogrid)   ) call fill_4corners(w, 2, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)
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
      if (bounded_domain .or. flagstruct%grid_type >=3 .or. (flagstruct%duogrid) ) then
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
if (.not. flagstruct%duogrid) then
! Remove the extra term at the corners:
      if ( sw_corner ) vort(1,    1) = vort(1,    1) + fy(0,   1)
      if ( se_corner ) vort(npx  ,1) = vort(npx,  1) - fy(npx, 1)
      if ( ne_corner ) vort(npx,npy) = vort(npx,npy) - fy(npx,npy)
      if ( nw_corner ) vort(1,  npy) = vort(1,  npy) + fy(0,  npy)
endif
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
      if (bounded_domain .or. flagstruct%grid_type >= 3 .or. (flagstruct%duogrid)) then
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

end module c_sw_duo_extract_mod
