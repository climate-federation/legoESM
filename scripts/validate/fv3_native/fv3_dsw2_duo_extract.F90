! Phase-4c duo d_sw2 extract — VERBATIM authoritative symmetryclean
! sw_core.F90:1000-1199 d_sw2 (the post-averaging delp/pt/w update stage:
! consumes allflux slots 1/2/4 and applies the flux divergences; on the
! hydrostatic non-SW_DYNAMICS inline_q=F lane it reduces to the delp+pt
! else-arm) plus sw_core.F90:2008-2121 del6_vt_flux VERBATIM (referenced
! by the non-hydrostatic damp_w branch; needed at link, and the verbatim
! body doubles as the d_sw3 vorticity-damping reference).
! Authoritative-block SHAs (sha256 of the sw_core.F90 line ranges):
!   d_sw2 (1000-1199)        e1c22c848026d80dff6cce2fd6d936f9809f20339c61c5478e20f312b53b425f
!   del6_vt_flux (2008-2121) a0c6676195fbac993b5457a3c2187b14ae4e658edbc6808b1b9888bde3f56d4d
! The oracle test pins the sha256 of THIS extract file (drift guard);
! the single intended deviation is the dw/ptc intent shim below.
module dsw2_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use tpcore_duo_extract_mod, only: copy_corners
  implicit none
contains

   subroutine d_sw2( delp, ptc, pt, w, q_con, &
                    kgb, heat_source,    &
                    nq, q, k, km, inline_q,  &
                    dt,    &
                    nord_w, damp_w, &
                    hydrostatic, gridstruct, flagstruct, bd, &
                    dw, allflux_x,allflux_y)

      integer, intent(IN):: nord_w ! vertical velocity
      integer, intent(IN):: nq, k, km
      real   , intent(IN):: dt
      real,    intent(in):: damp_w, kgb
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: delp, pt
      real, intent(INOUT), dimension(bd%isd:      ,  bd%jsd:      ):: w, q_con

      real, intent(INOUT):: q(bd%isd:bd%ied,bd%jsd:bd%jed,km,nq)
      real, intent(INOUT),   dimension(bd%is:bd%ie,bd%js:bd%je):: heat_source

      logical, intent(IN):: hydrostatic
      logical, intent(IN):: inline_q

      real,intent(IN) ::   allflux_x(bd%is:bd%ie+1,bd%js:bd%je,km,4+nq  )  ! 1-D X-direction Fluxes
      real,intent(IN) ::   allflux_y(bd%is:bd%ie  ,bd%js:bd%je+1,km,4+nq)  ! 1-D Y-direction Fluxes

! SHIM DEVIATION (the ONLY edit vs the authoritative block): upstream
! declares dw/ptc intent(OUT) — undefined on entry, so the driver's 1e30
! sentinel init (proving they stay untouched on the hydrostatic lane)
! would be one compiler's stack behavior.  intent(inout) makes the
! sentinel round-trip defined semantics.  Same rationale as the ut/vt
! shim in fv3_dsw1_duo_extract.F90.
      real,intent(inout) :: dw(bd%is:bd%ie,bd%js:bd%je) !  w damping to be applied in dsw5
      real, intent(inout),   dimension(bd%isd:bd%ied,  bd%jsd:bd%jed)  ::  ptc

      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
      real :: fx2(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real :: fy2(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
!---
      real :: wk(bd%isd:bd%ied,bd%jsd:bd%jed) !  work array
      real ::   fx(bd%is:bd%ie+1,bd%js:bd%je  )  ! 1-D X-direction Fluxes
      real ::   fy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! 1-D Y-direction Fluxes
      real :: gx(bd%is:bd%ie+1,bd%js:bd%je  )
      real :: gy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! work Y-dir flux array

      real ::  damp4, dd8
      integer :: i,j, iq

      real, pointer, dimension(:,:) :: rarea

      integer :: is,  ie,  js,  je
      integer :: npx, npy

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je

      npx      = flagstruct%npx
      npy      = flagstruct%npy

      rarea     => gridstruct%rarea

        do j=js,je
           do i=is,ie+1
              fx(i,j)=allflux_x(i,j,k,1)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              fy(i,j)=allflux_y(i,j,k,1)
           enddo
        enddo

#ifndef SW_DYNAMICS
        do j=js,je
           do i=is,ie
              heat_source(i,j) = 0.
           enddo
        enddo

        if ( .not. hydrostatic ) then
            if ( damp_w>1.E-5 ) then
                 dd8 = kgb*abs(dt)
                 damp4 = (damp_w*gridstruct%da_min_c)**(nord_w+1)
                 call del6_vt_flux(nord_w, npx, npy, damp4, w, wk, fx2, fy2, gridstruct, bd)
                do j=js,je
                   do i=is,ie
                      dw(i,j) = (fx2(i,j)-fx2(i+1,j)+fy2(i,j)-fy2(i,j+1))*rarea(i,j)
! 0.5 * [ (w+dw)**2 - w**2 ] = w*dw + 0.5*dw*dw
!                   heat_source(i,j) = -d_con*dw(i,j)*(w(i,j)+0.5*dw(i,j))
                    heat_source(i,j) = dd8 - dw(i,j)*(w(i,j)+0.5*dw(i,j))
                   enddo
                enddo
            endif

        do j=js,je
           do i=is,ie+1
              gx(i,j)=allflux_x(i,j,k,2)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              gy(i,j)=allflux_y(i,j,k,2)
           enddo
        enddo

            do j=js,je
               do i=is,ie
                  w(i,j) = delp(i,j)*w(i,j) + (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j)
               enddo
            enddo
        endif


#ifdef USE_COND

        do j=js,je
           do i=is,ie+1
              gx(i,j)=allflux_x(i,j,k,3)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              gy(i,j)=allflux_y(i,j,k,3)
           enddo
        enddo

            do j=js,je
               do i=is,ie
                  q_con(i,j) = delp(i,j)*q_con(i,j) + (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j)
               enddo
            enddo
#endif

        do j=js,je
           do i=is,ie+1
              gx(i,j)=allflux_x(i,j,k,4)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              gy(i,j)=allflux_y(i,j,k,4)
           enddo
        enddo

#endif

     if ( inline_q ) then
        do j=js,je
           do i=is,ie
                wk(i,j) = delp(i,j)
              delp(i,j) = wk(i,j) + (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))*rarea(i,j)
#ifdef SW_DYNAMICS
              ptc(i,j) = pt(i,j)
#else
              pt(i,j) = (pt(i,j)*wk(i,j) +               &
                        (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j))/delp(i,j)
#endif
           enddo
        enddo
        do iq=1,nq

        do j=js,je
           do i=is,ie+1
              gx(i,j)=allflux_x(i,j,k,4+iq)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              gy(i,j)=allflux_y(i,j,k,4+iq)
           enddo
        enddo

           do j=js,je
              do i=is,ie
                 q(i,j,k,iq) = (q(i,j,k,iq)*wk(i,j) +               &
                         (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j))/delp(i,j)
              enddo
           enddo
        enddo

     else
        do j=js,je
           do i=is,ie
#ifndef SW_DYNAMICS
              pt(i,j) = pt(i,j)*delp(i,j) +               &
                         (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j)
#endif
              delp(i,j) = delp(i,j) +                     &
                         (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))*rarea(i,j)
#ifndef SW_DYNAMICS
              pt(i,j) = pt(i,j) / delp(i,j)

#endif
           enddo
        enddo
     endif

 end subroutine d_sw2

 subroutine del6_vt_flux(nord, npx, npy, damp, q, d2, fx2, fy2, gridstruct, bd)
! Del-nord damping for the relative vorticity
! nord must be <= 2
!------------------
! nord = 0:   del-2
! nord = 1:   del-4
! nord = 2:   del-6
!------------------
!This does the same operation as tp_core::deln_flux except that it does not
!add diffusive fluxes into the regular fluxes
   integer, intent(in):: nord, npx, npy
   real, intent(in):: damp
   type(fv_grid_bounds_type), intent(IN) :: bd
   real, intent(in):: q(bd%isd:bd%ied, bd%jsd:bd%jed)  ! rel. vorticity ghosted on input
   type(fv_grid_type), intent(IN), target :: gridstruct
! Work arrays:
   real, intent(out):: d2(bd%isd:bd%ied, bd%jsd:bd%jed)
   real, intent(out):: fx2(bd%isd:bd%ied+1,bd%jsd:bd%jed), fy2(bd%isd:bd%ied,bd%jsd:bd%jed+1)
   integer i,j, nt, n, i1, i2, j1, j2

   logical :: bounded_domain

#ifdef USE_SG
  real, pointer, dimension(:,:,:) :: sin_sg
  real, pointer, dimension(:,:) ::  rdxc, rdyc, dx,dy
#endif

   integer :: is,  ie,  js,  je

#ifdef USE_SG
   sin_sg   => gridstruct%sin_sg
   rdxc     => gridstruct%rdxc
   rdyc     => gridstruct%rdyc
   dx       => gridstruct%dx
   dy       => gridstruct%dy
#endif
   bounded_domain = gridstruct%bounded_domain

   is  = bd%is
   ie  = bd%ie
   js  = bd%js
   je  = bd%je

   i1 = is-1-nord;    i2 = ie+1+nord
   j1 = js-1-nord;    j2 = je+1+nord

   do j=j1, j2
      do i=i1, i2
         d2(i,j) = damp*q(i,j)
      enddo
   enddo

   if( nord>0 .and. (.not. bounded_domain .or. .not. gridstruct%dg%is_initialized)) call copy_corners(d2, npx, npy, 1, bounded_domain, gridstruct%dg%is_initialized, bd, gridstruct%sw_corner,    &
                   gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)
   do j=js-nord,je+nord
      do i=is-nord,ie+nord+1
#ifdef USE_SG
         fx2(i,j) = 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))*dy(i,j)*(d2(i-1,j)-d2(i,j))*rdxc(i,j)
#else
         fx2(i,j) = gridstruct%del6_v(i,j)*(d2(i-1,j)-d2(i,j))
#endif
      enddo
   enddo

   if( nord>0 .and. (.not. bounded_domain .or. .not. gridstruct%dg%is_initialized)) call copy_corners(d2, npx, npy, 2, bounded_domain, gridstruct%dg%is_initialized, bd, gridstruct%sw_corner,   &
                   gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)
   do j=js-nord,je+nord+1
      do i=is-nord,ie+nord
#ifdef USE_SG
         fy2(i,j) = 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))*dx(i,j)*(d2(i,j-1)-d2(i,j))*rdyc(i,j)
#else
         fy2(i,j) = gridstruct%del6_u(i,j)*(d2(i,j-1)-d2(i,j))
#endif
      enddo
   enddo

   if ( nord>0 ) then
   do n=1, nord
      nt = nord-n
      do j=js-nt-1,je+nt+1
         do i=is-nt-1,ie+nt+1
            d2(i,j) = (fx2(i,j)-fx2(i+1,j)+fy2(i,j)-fy2(i,j+1))*gridstruct%rarea(i,j)
         enddo
      enddo

      if (.not. bounded_domain .or. .not. (gridstruct%dg%is_initialized)) call copy_corners(d2, npx, npy, 1, bounded_domain, gridstruct%dg%is_initialized, bd, gridstruct%sw_corner,    &
         gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)

      do j=js-nt,je+nt
         do i=is-nt,ie+nt+1
#ifdef USE_SG
            fx2(i,j) = 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))*dy(i,j)*(d2(i,j)-d2(i-1,j))*rdxc(i,j)
#else
            fx2(i,j) = gridstruct%del6_v(i,j)*(d2(i,j)-d2(i-1,j))
#endif
         enddo
      enddo

      if (.not. bounded_domain .or. .not. (gridstruct%dg%is_initialized)) call copy_corners(d2, npx, npy, 2, bounded_domain, gridstruct%dg%is_initialized, bd, &
      gridstruct%sw_corner, gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)

      do j=js-nt,je+nt+1
         do i=is-nt,ie+nt
#ifdef USE_SG
            fy2(i,j) = 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))*dx(i,j)*(d2(i,j)-d2(i,j-1))*rdyc(i,j)
#else
            fy2(i,j) = gridstruct%del6_u(i,j)*(d2(i,j)-d2(i,j-1))
#endif
         enddo
      enddo
   enddo
   endif

 end subroutine del6_vt_flux

end module dsw2_duo_extract_mod
