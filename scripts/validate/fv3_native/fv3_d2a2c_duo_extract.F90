! Phase-4c duo d2a2c_vect one-step oracle module.
!
! VERBATIM authoritative symmetryclean sw_core.F90:3361-3706 d2a2c_vect (the
! version carrying the ``gridstruct%dg%is_initialized`` duo branches:
! interior 4th-order formulas over the full domain, all panel-edge /
! cube-corner special-cases guarded by ``.not. dg%is_initialized``).  Kept
! in a SEPARATE module so the phase-4a-certified PLAIN d2a2c_vect in
! sw_core_extract_mod is untouched (that one is the 6f658bd0 tree, no dg).
! Constants + edge_interpolate4 are reused from sw_core_extract_mod.
module d2a2c_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type
  use sw_core_extract_mod, only: a1, a2, c1, c2, c3, big_number, &
                                 edge_interpolate4
  implicit none
contains

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

 if ( bounded_domain   .or. (gridstruct%dg%is_initialized) ) then

        do i=isd,ied
     do j=jsd+1,jed-1
           utmp(i,j) = a2*(u(i,j-1)+u(i,j+2)) + a1*(u(i,j)+u(i,j+1))
        enddo
     enddo
     do i=isd,ied
        j = jsd
        utmp(i,j) = 0.5*(u(i,j)+u(i,j+1))
        utmp(i,j) = 0.5*(u(i,j+1)+u(i,j+1))
        !utmp(i,j) = u(i,jsd+1) 
        j = jed
        utmp(i,j) = 0.5*(u(i,j)+u(i,j+1))
        utmp(i,j) = 0.5*(u(i,j+1)+u(i,j+1))
        !utmp(i,j) = u(i,jed-1) 
     end do

     do j=jsd,jed
        do i=isd+1,ied-1
           vtmp(i,j) = a2*(v(i-1,j)+v(i+2,j)) + a1*(v(i,j)+v(i+1,j))
        enddo
        i = isd
        vtmp(i,j) = 0.5*(v(i,j)+v(i+1,j))
        vtmp(i,j) = 0.5*(v(i+1,j)+v(i+1,j))
        i = ied
        vtmp(i,j) = 0.5*(v(i,j)+v(i+1,j))
        vtmp(i,j) = 0.5*(v(i+1,j)+v(i+1,j))
     enddo

        do i=isd,ied
     do j=jsd,jed
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
     if (grid_type < 3 .and. (.not. gridstruct%dg%is_initialized)) then

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
     if( gridstruct%sw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do i=-2,0
            utmp(i,0) = -vtmp(0,1-i)
         enddo
     endif
     if( gridstruct%se_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do i=0,2
            utmp(npx+i,0) = vtmp(npx,i+1)
         enddo
     endif
     if( gridstruct%ne_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do i=0,2
            utmp(npx+i,npy) = -vtmp(npx,je-i)
         enddo
     endif
     if( gridstruct%nw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do i=-2,0
            utmp(i,npy) = vtmp(0,je+i)
         enddo
     endif

  if (grid_type < 3 .and. .not. bounded_domain .and. (.not. gridstruct%dg%is_initialized)) then
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
     if( gridstruct%sw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         ua(-1,0) = -va(0,2)
         ua( 0,0) = -va(0,1)
     endif
     if( gridstruct%se_corner .and. (.not. gridstruct%dg%is_initialized)) then
         ua(npx,  0) = va(npx,1)
         ua(npx+1,0) = va(npx,2)
     endif
     if( gridstruct%ne_corner .and. (.not. gridstruct%dg%is_initialized)) then
         ua(npx,  npy) = -va(npx,npy-1)
         ua(npx+1,npy) = -va(npx,npy-2)
     endif
     if( gridstruct%nw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         ua(-1,npy) = va(0,npy-2)
         ua( 0,npy) = va(0,npy-1)
     endif

     if( is==1 .and. .not. bounded_domain  .and. (.not. gridstruct%dg%is_initialized)) then
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

     if( (ie+1)==npx  .and. .not. bounded_domain .and. (.not. gridstruct%dg%is_initialized)) then
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
     if( gridstruct%sw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do j=-2,0
            vtmp(0,j) = -utmp(1-j,0)
         enddo
     endif
     if( gridstruct%nw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do j=0,2
            vtmp(0,npy+j) = utmp(j+1,npy)
         enddo
     endif
     if( gridstruct%se_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do j=-2,0
            vtmp(npx,j) = utmp(ie+j,0)
         enddo
     endif
     if( gridstruct%ne_corner .and. (.not. gridstruct%dg%is_initialized)) then
         do j=0,2
            vtmp(npx,npy+j) = -utmp(ie-j,npy)
         enddo
     endif
     if( gridstruct%sw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         va(0,-1) = -ua(2,0)
         va(0, 0) = -ua(1,0)
     endif
     if( gridstruct%se_corner .and. (.not. gridstruct%dg%is_initialized)) then
         va(npx, 0) = ua(npx-1,0)
         va(npx,-1) = ua(npx-2,0)
     endif
     if( gridstruct%ne_corner .and. (.not. gridstruct%dg%is_initialized)) then
         va(npx,npy  ) = -ua(npx-1,npy)
         va(npx,npy+1) = -ua(npx-2,npy)
     endif
     if( gridstruct%nw_corner .and. (.not. gridstruct%dg%is_initialized)) then
         va(0,npy)   = ua(1,npy)
         va(0,npy+1) = ua(2,npy)
     endif

 if (grid_type < 3) then

     do j=js-1,je+2
      if ( j==1 .and. .not. bounded_domain  .and. (.not. gridstruct%dg%is_initialized)) then
        do i=is-1,ie+1
           vt(i,j) = edge_interpolate4(va(i,-1:2), dya(i,-1:2))
           if (vt(i,j) > 0.) then
              vc(i,j) = vt(i,j)*sin_sg(i,j-1,4)
           else
              vc(i,j) = vt(i,j)*sin_sg(i,j,2)
           end if
        enddo
      elseif ( (j==0 .or. j==(npy-1)) .and. .not. bounded_domain .and. (.not. gridstruct%dg%is_initialized) ) then
        do i=is-1,ie+1
           vc(i,j) = c1*vtmp(i,j-2) + c2*vtmp(i,j-1) + c3*vtmp(i,j)
           vt(i,j) = (vc(i,j) - u(i,j)*cosa_v(i,j))*rsin_v(i,j)
        enddo
      elseif ( (j==2 .or. j==(npy+1))  .and. .not. bounded_domain .and. (.not. gridstruct%dg%is_initialized)) then
        do i=is-1,ie+1
           vc(i,j) = c1*vtmp(i,j+1) + c2*vtmp(i,j) + c3*vtmp(i,j-1)
           vt(i,j) = (vc(i,j) - u(i,j)*cosa_v(i,j))*rsin_v(i,j)
        enddo
      elseif ( j==npy .and. .not. bounded_domain .and. (.not. gridstruct%dg%is_initialized) ) then
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

end module d2a2c_duo_extract_mod
