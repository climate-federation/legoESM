! VERBATIM extraction from GFDL_atmos_cubed_sphere @ 6f658bd0 for the
! phase-4b d_sw one-step oracle.  Subroutine bodies are unmodified; only
! the module wrappers and use-line swaps onto swcore_shim_mod are new.
! Build: gfortran -O2 -fdefault-real-8 -fdefault-double-8 -cpp
!        -ffree-line-length-none  (production: no SW_DYNAMICS/WAVE_FORM)

!***********************************************************************
!*                   GNU Lesser General Public License
!*
!* This file is part of the FV3 dynamical core.
!*
!* The FV3 dynamical core is free software: you can redistribute it
!* and/or modify it under the terms of the
!* GNU Lesser General Public License as published by the
!* Free Software Foundation, either version 3 of the License, or
!* (at your option) any later version.
!*
!* The FV3 dynamical core is distributed in the hope that it will be
!* useful, but WITHOUT ANY WARRANTY; without even the implied warranty
!* of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!* See the GNU General Public License for more details.
!*
!* You should have received a copy of the GNU Lesser General Public
!* License along with the FV3 dynamical core.
!* If not, see <http://www.gnu.org/licenses/>.
!***********************************************************************

module tp_core_mod
!BOP
!
! !MODULE: tp_core --- A collection of routines to support FV transport
!
   use swcore_shim_mod,   only: big_number
   use swcore_shim_mod,   only: fv_grid_type, fv_grid_bounds_type

 implicit none

 private
 public fv_tp_2d, pert_ppm, copy_corners

 real, parameter:: ppm_fac = 1.5   ! nonlinear scheme limiter: between 1 and 2
 real, parameter:: r3 = 1./3.
 real, parameter:: near_zero = 1.E-25
 real, parameter:: ppm_limiter = 2.0
 real, parameter:: r12 = 1./12.

#ifdef WAVE_FORM
! Suresh & Huynh scheme 2.2 (purtabation form)
! The wave-form is more diffusive than scheme 2.1
 real, parameter:: b1 =   0.0375
 real, parameter:: b2 =  -7./30.
 real, parameter:: b3 =  -23./120.
 real, parameter:: b4 =  13./30.
 real, parameter:: b5 = -11./240.
#else
! scheme 2.1: perturbation form
 real, parameter:: b1 =   1./30.
 real, parameter:: b2 = -13./60.
 real, parameter:: b3 = -13./60.
 real, parameter:: b4 =  0.45
 real, parameter:: b5 = -0.05
#endif
 real, parameter:: t11 = 27./28., t12 = -13./28., t13=3./7.
 real, parameter:: s11 = 11./14., s14 = 4./7.,    s15=3./14.
!----------------------------------------------------
! volume-conserving cubic with 2nd drv=0 at end point:
!----------------------------------------------------
! Non-monotonic
  real, parameter:: c1 = -2./14.
  real, parameter:: c2 = 11./14.
  real, parameter:: c3 =  5./14.
!----------------------
! PPM volume mean form:
!----------------------
  real, parameter:: p1 =  7./12.     ! 0.58333333
  real, parameter:: p2 = -1./12.
!   q(i+0.5) = p1*(q(i-1)+q(i)) + p2*(q(i-2)+q(i+1))
! integer:: is, ie, js, je, isd, ied, jsd, jed

  !List of schemes for tracer setup
  integer, public, parameter :: tp_mono_schemes(1) = (/8/)
  integer, public, parameter :: tp_PD_schemes(5) = (/-5, 7, 9, 12, 13/)
  integer, public, parameter :: tp_unlim_schemes(8) = (/1, 2, 3, 4, 5, 6, 10, 11/)
  integer, public, parameter :: tp_valid_schemes(14) = (/-5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13/)
!
!EOP
!-----------------------------------------------------------------------

contains

 subroutine fv_tp_2d(q, crx, cry, npx, npy, hord, fx, fy, xfx, yfx,  &
                     gridstruct, bd, ra_x, ra_y, lim_fac, mfx, mfy, &
                     mass, nord, damp_c, damp_smag, damp_Km)
   type(fv_grid_bounds_type), intent(IN) :: bd
   integer, intent(in):: npx, npy
   integer, intent(in)::hord

   real, intent(in)::  crx(bd%is:bd%ie+1,bd%jsd:bd%jed)  !
   real, intent(in)::  xfx(bd%is:bd%ie+1,bd%jsd:bd%jed)  !
   real, intent(in)::  cry(bd%isd:bd%ied,bd%js:bd%je+1 )  !
   real, intent(in)::  yfx(bd%isd:bd%ied,bd%js:bd%je+1 )  !
   real, intent(in):: ra_x(bd%is:bd%ie,bd%jsd:bd%jed)
   real, intent(in):: ra_y(bd%isd:bd%ied,bd%js:bd%je)
   real, intent(inout):: q(bd%isd:bd%ied,bd%jsd:bd%jed)  ! transported scalar
   real, intent(out)::fx(bd%is:bd%ie+1 ,bd%js:bd%je)    ! Flux in x ( E )
   real, intent(out)::fy(bd%is:bd%ie,   bd%js:bd%je+1 )    ! Flux in y ( N )

   type(fv_grid_type), intent(IN), target :: gridstruct

   real, intent(in):: lim_fac
! optional Arguments:
   real, OPTIONAL, intent(in):: mfx(bd%is:bd%ie+1,bd%js:bd%je  )  ! Mass Flux X-Dir
   real, OPTIONAL, intent(in):: mfy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! Mass Flux Y-Dir
   real, OPTIONAL, intent(in):: mass(bd%isd:bd%ied,bd%jsd:bd%jed)
   real, OPTIONAL, intent(in):: damp_c
   integer, OPTIONAL, intent(in):: nord
   real, OPTIONAL, intent(in) :: damp_smag ! additional 2nd-order flux
   real, OPTIONAL, intent(in) :: damp_Km(bd%isd:bd%ied,bd%jsd:bd%jed) ! variable diffusion coeff for scalars
                                                                      ! First try adapts cell-centered eddy diffusivities
! Local:
   integer ord_ou, ord_in
   real q_i(bd%isd:bd%ied,bd%js:bd%je)
   real q_j(bd%is:bd%ie,bd%jsd:bd%jed)
   real   fx2(bd%is:bd%ie+1,bd%jsd:bd%jed)
   real   fy2(bd%isd:bd%ied,bd%js:bd%je+1)
   real   fyy(bd%isd:bd%ied,bd%js:bd%je+1)
   real   fx1(bd%is:bd%ie+1)
   real   damp
   integer i, j

   integer:: is, ie, js, je, isd, ied, jsd, jed

   is  = bd%is
   ie  = bd%ie
   js  = bd%js
   je  = bd%je
   isd = bd%isd
   ied = bd%ied
   jsd = bd%jsd
   jed = bd%jed

   if ( hord == 10 ) then
        ord_in = 8
   else
        ord_in = hord
   endif
   ord_ou = hord

   if (.not. gridstruct%bounded_domain) &
      call copy_corners(q, npx, npy, 2, gridstruct%bounded_domain, bd, &
                         gridstruct%sw_corner, gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)

   call yppm(fy2, q, cry, ord_in, isd,ied,isd,ied, js,je,jsd,jed, npx,npy, gridstruct%dya, &
             gridstruct%bounded_domain, gridstruct%grid_type, lim_fac)

   do j=js,je+1
      do i=isd,ied
         fyy(i,j) = yfx(i,j) * fy2(i,j)
      enddo
   enddo
   do j=js,je
      do i=isd,ied
         q_i(i,j) = (q(i,j)*gridstruct%area(i,j) + fyy(i,j)-fyy(i,j+1))/ra_y(i,j)
      enddo
   enddo

   call xppm(fx, q_i, crx(is,js), ord_ou, is,ie,isd,ied, js,je,jsd,jed, npx,npy, &
             gridstruct%dxa, gridstruct%bounded_domain, gridstruct%grid_type, lim_fac)

   if (.not. gridstruct%bounded_domain) &
     call copy_corners(q, npx, npy, 1, gridstruct%bounded_domain, bd, &
                       gridstruct%sw_corner, gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)

   call xppm(fx2, q, crx, ord_in, is,ie,isd,ied, jsd,jed,jsd,jed, npx,npy, gridstruct%dxa, &
             gridstruct%bounded_domain, gridstruct%grid_type, lim_fac)

   do j=jsd,jed
      do i=is,ie+1
         fx1(i) =  xfx(i,j) * fx2(i,j)
      enddo
      do i=is,ie
         q_j(i,j) = (q(i,j)*gridstruct%area(i,j) + fx1(i)-fx1(i+1))/ra_x(i,j)
      enddo
   enddo

   call yppm(fy, q_j, cry, ord_ou, is,ie,isd,ied, js,je,jsd,jed, npx, npy, gridstruct%dya, &
             gridstruct%bounded_domain, gridstruct%grid_type, lim_fac)

!----------------
! Flux averaging:
!----------------

   if ( present(mfx) .and. present(mfy) ) then
!---------------------------------
! For transport of pt and tracers
!---------------------------------
      do j=js,je
         do i=is,ie+1
            fx(i,j) = 0.5*(fx(i,j) + fx2(i,j)) * mfx(i,j)
         enddo
      enddo
      do j=js,je+1
         do i=is,ie
            fy(i,j) = 0.5*(fy(i,j) + fy2(i,j)) * mfy(i,j)
         enddo
      enddo
      if ( present(nord) .and. present(damp_c) .and. present(mass) ) then
        if ( damp_c > 1.e-4 ) then
           damp = (damp_c * gridstruct%da_min)**(nord+1)
           call deln_flux(nord, is,ie,js,je, npx, npy, damp, q, fx, fy, gridstruct, bd, mass )
        endif
     endif
     if (present(damp_smag) .and. present(damp_Km) .and. present(mass)) then
        if (damp_smag > 1.e-3) then
           damp = damp_smag * gridstruct%da_min !2nd order
           call deln_flux(0, is,ie,js,je, npx, npy, damp, q, fx, fy, gridstruct, bd, mass, damp_Km=damp_Km )
        endif
     endif
   else
!---------------------------------
! For transport of delp, vorticity
!---------------------------------
      do j=js,je
         do i=is,ie+1
            fx(i,j) = 0.5*(fx(i,j) + fx2(i,j)) * xfx(i,j)
         enddo
      enddo
      do j=js,je+1
         do i=is,ie
            fy(i,j) = 0.5*(fy(i,j) + fy2(i,j)) * yfx(i,j)
         enddo
      enddo
      if ( present(nord) .and. present(damp_c) ) then
           if ( damp_c > 1.E-4 ) then
                damp = (damp_c * gridstruct%da_min)**(nord+1)
                call deln_flux(nord, is,ie,js,je, npx, npy, damp, q, fx, fy, gridstruct, bd)
           endif
      endif
      if (present(damp_smag) .and. present(damp_Km)) then
         if (damp_smag > 1.e-3) then
            damp = damp_smag * gridstruct%da_min !2nd order
            call deln_flux(0, is,ie,js,je, npx, npy, damp, q, fx, fy, gridstruct, bd, damp_Km=damp_Km)
         endif
      endif
   endif

 end subroutine fv_tp_2d

 !Weird arguments are because this routine is called in a lot of
 !places outside of tp_core, sometimes very deeply nested in the call tree.
 subroutine copy_corners(q, npx, npy, dir, bounded_domain, bd, &
                         sw_corner, se_corner, nw_corner, ne_corner)
 type(fv_grid_bounds_type), intent(IN) :: bd
 integer, intent(in):: npx, npy, dir
 real, intent(inout):: q(bd%isd:bd%ied,bd%jsd:bd%jed)
 logical, intent(IN) :: bounded_domain, sw_corner, se_corner, nw_corner, ne_corner
 integer  i,j, ng

 ng = bd%ng

 if (bounded_domain) return

 if ( dir == 1 ) then
! XDir:
    if ( sw_corner ) then
         do j=1-ng,0
            do i=1-ng,0
               q(i,j) = q(j,1-i)
            enddo
         enddo
    endif
    if ( se_corner ) then
         do j=1-ng,0
            do i=npx,npx+ng-1
               q(i,j) = q(npy-j,i-npx+1)
            enddo
         enddo
    endif
    if ( ne_corner ) then
         do j=npy,npy+ng-1
            do i=npx,npx+ng-1
               q(i,j) = q(j,2*npx-1-i)
            enddo
         enddo
    endif
    if ( nw_corner ) then
         do j=npy,npy+ng-1
            do i=1-ng,0
               q(i,j) = q(npy-j,i-1+npx)
            enddo
         enddo
    endif

 elseif ( dir == 2 ) then
! YDir:

    if ( sw_corner ) then
         do j=1-ng,0
            do i=1-ng,0
               q(i,j) = q(1-j,i)
            enddo
         enddo
    endif
    if ( se_corner ) then
         do j=1-ng,0
            do i=npx,npx+ng-1
               q(i,j) = q(npy+j-1,npx-i)
            enddo
         enddo
    endif
    if ( ne_corner ) then
         do j=npy,npy+ng-1
            do i=npx,npx+ng-1
               q(i,j) = q(2*npy-1-j,i)
            enddo
         enddo
    endif
    if ( nw_corner ) then
         do j=npy,npy+ng-1
            do i=1-ng,0
               q(i,j) = q(j+1-npx,npy-i)
            enddo
         enddo
    endif

 endif

 end subroutine copy_corners

 subroutine xppm(flux, q, c, iord, is,ie,isd,ied, jfirst,jlast,jsd,jed, npx, npy, dxa, bounded_domain, grid_type, lim_fac)
 integer, INTENT(IN) :: is, ie, isd, ied, jsd, jed
 integer, INTENT(IN) :: jfirst, jlast  ! compute domain
 integer, INTENT(IN) :: iord
 integer, INTENT(IN) :: npx, npy
 real   , INTENT(IN) :: q(isd:ied,jfirst:jlast)
 real   , INTENT(IN) :: c(is:ie+1,jfirst:jlast) ! Courant   N (like FLUX)
 real   , intent(IN) :: dxa(isd:ied,jsd:jed)
 logical, intent(IN) :: bounded_domain
 integer, intent(IN) :: grid_type
 real   , intent(IN) :: lim_fac
! !OUTPUT PARAMETERS:
 real  , INTENT(OUT) :: flux(is:ie+1,jfirst:jlast) !  Flux
! Local
 real, dimension(is-1:ie+1):: bl, br, b0, a4, da1
 real:: q1(isd:ied)
 real, dimension(is:ie+1):: fx0, fx1, xt1
 logical, dimension(is-1:ie+1):: ext5, ext6, smt5, smt6
 logical, dimension(is:ie+1):: hi5, hi6
 real  al(is-1:ie+2)
 real  dm(is-2:ie+2)
 real  dq(is-3:ie+2)
 integer:: i, j, ie3, is1, ie1, mord
 real:: x0, x1, xt, qtmp, pmp_1, lac_1, pmp_2, lac_2

 if ( .not. bounded_domain .and. grid_type<3 ) then
    is1 = max(3,is-1);  ie3 = min(npx-2,ie+2)
                        ie1 = min(npx-3,ie+1)
 else
    is1 = is-1;         ie3 = ie+2
                        ie1 = ie+1
 end if

 mord = abs(iord)

 do 666 j=jfirst,jlast

    do i=isd, ied
       q1(i) = q(i,j)
    enddo

 if ( iord < 7 ) then
! ord = 2: perfectly linear ppm scheme
! Diffusivity: ord2 < ord5 < ord3 < ord4 < ord6

   do i=is1, ie3
      al(i) = p1*(q1(i-1)+q1(i)) + p2*(q1(i-2)+q1(i+1))
   enddo

   if ( .not.bounded_domain .and. grid_type<3 ) then
     if ( is==1 ) then
       al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0)
       al(1) = 0.5*(((2.*dxa(0,j)+dxa(-1,j))*q1(0)-dxa(0,j)*q1(-1))/(dxa(-1,j)+dxa(0,j)) &
             +      ((2.*dxa(1,j)+dxa( 2,j))*q1(1)-dxa(1,j)*q1( 2))/(dxa(1, j)+dxa(2,j)))
       al(2) = c3*q1(1) + c2*q1(2) +c1*q1(3)
     endif
     if ( (ie+1)==npx ) then
       al(npx-1) = c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1)
       al(npx) = 0.5*(((2.*dxa(npx-1,j)+dxa(npx-2,j))*q1(npx-1)-dxa(npx-1,j)*q1(npx-2))/(dxa(npx-2,j)+dxa(npx-1,j)) &
               +      ((2.*dxa(npx,  j)+dxa(npx+1,j))*q1(npx  )-dxa(npx,  j)*q1(npx+1))/(dxa(npx,  j)+dxa(npx+1,j)))
       al(npx+1) = c3*q1(npx) + c2*q1(npx+1) + c1*q1(npx+2)
     endif
   endif

   if ( iord<0 ) then
       do i=is-1, ie+2
          al(i) = max(0., al(i))
       enddo
   endif

   if ( mord==1 ) then  ! perfectly linear scheme
        do i=is-1,ie+1
           bl(i) = al(i)   - q1(i)
           br(i) = al(i+1) - q1(i)
           b0(i) = bl(i) + br(i)
           smt5(i) = abs(lim_fac*b0(i)) < abs(bl(i)-br(i))
        enddo
!DEC$ VECTOR ALWAYS
      do i=is,ie+1
         if ( c(i,j) > 0. ) then
             fx1(i) = (1.-c(i,j))*(br(i-1) - c(i,j)*b0(i-1))
             flux(i,j) = q1(i-1)
         else
             fx1(i) = (1.+c(i,j))*(bl(i) + c(i,j)*b0(i))
             flux(i,j) = q1(i)
         endif
         if (smt5(i-1).or.smt5(i)) flux(i,j) = flux(i,j) + fx1(i)
      enddo

   elseif ( mord==2 ) then  ! perfectly linear scheme

!DEC$ VECTOR ALWAYS
      do i=is,ie+1
         xt = c(i,j)
         if ( xt > 0. ) then
              qtmp = q1(i-1)
              flux(i,j) = qtmp + (1.-xt)*(al(i)-qtmp-xt*(al(i-1)+al(i)-(qtmp+qtmp)))
         else
              qtmp = q1(i)
              flux(i,j) = qtmp + (1.+xt)*(al(i)-qtmp+xt*(al(i)+al(i+1)-(qtmp+qtmp)))
         endif
!        x0 = sign(dim(xt, 0.), 1.)
!        x1 = sign(dim(0., xt), 1.)
!        flux(i,j) = x0*(q1(i-1)+(1.-xt)*(al(i)-qtmp-xt*(al(i-1)+al(i)-(qtmp+qtmp))))     &
!                  + x1*(q1(i)  +(1.+xt)*(al(i)-qtmp+xt*(al(i)+al(i+1)-(qtmp+qtmp))))
      enddo

   elseif ( mord==3 ) then

        do i=is-1,ie+1
           bl(i) = al(i)   - q1(i)
           br(i) = al(i+1) - q1(i)
           b0(i) = bl(i) + br(i)
              x0 = abs(b0(i))
              xt = abs(bl(i)-br(i))
           smt5(i) =    x0 < xt
           smt6(i) = 3.*x0 < xt
        enddo
        do i=is,ie+1
           xt1(i) = c(i,j)
           if ( xt1(i) > 0. ) then
               if ( smt5(i-1) .or. smt6(i) ) then
                    flux(i,j) = q1(i-1) + (1.-xt1(i))*(br(i-1) - xt1(i)*b0(i-1))
               else
                    flux(i,j) = q1(i-1)
               endif
           else
               if ( smt6(i-1) .or. smt5(i) ) then
                    flux(i,j) = q1(i) + (1.+xt1(i))*(bl(i) + xt1(i)*b0(i))
               else
                    flux(i,j) = q1(i)
               endif
           endif
        enddo

   elseif ( mord==4 ) then

        do i=is-1,ie+1
           bl(i) = al(i)   - q1(i)
           br(i) = al(i+1) - q1(i)
           b0(i) = bl(i) + br(i)
              x0 = abs(b0(i))
              xt = abs(bl(i)-br(i))
           smt5(i) =    x0 < xt
           smt6(i) = 3.*x0 < xt
        enddo
        do i=is,ie+1
           xt1(i) = c(i,j)
           hi5(i) = smt5(i-1) .and. smt5(i)   ! more diffusive
           hi6(i) = smt6(i-1) .or.  smt6(i)
           hi5(i) = hi5(i) .or. hi6(i)
        enddo
!DEC$ VECTOR ALWAYS
        do i=is,ie+1
! Low-order only if (ext6(i-1).and.ext6(i)) .AND. ext5(i1).or.ext5(i)()
          if ( xt1(i) > 0. ) then
               fx1(i) = (1.-xt1(i))*(br(i-1) - xt1(i)*b0(i-1))
               flux(i,j) = q1(i-1)
           else
               fx1(i) = (1.+xt1(i))*(bl(i) + xt1(i)*b0(i))
               flux(i,j) = q1(i)
           endif
           if ( hi5(i) ) flux(i,j) = flux(i,j) + fx1(i)
        enddo

   else

      if ( iord==5 ) then
        do i=is-1,ie+1
           bl(i) = al(i)   - q1(i)
           br(i) = al(i+1) - q1(i)
           b0(i) = bl(i) + br(i)
           smt5(i) = bl(i)*br(i) < 0.
        enddo
      else
        if ( iord==-5 ) then
          do i=is-1,ie+1
             bl(i) = al(i)   - q1(i)
             br(i) = al(i+1) - q1(i)
             b0(i) = bl(i) + br(i)
             smt5(i) = bl(i)*br(i) < 0.
             da1(i) = br(i) - bl(i)
             a4(i) = -3.*b0(i)
          enddo
          do i=is-1,ie+1
             if( abs(da1(i)) < -a4(i) ) then
             if( q1(i)+0.25/a4(i)*da1(i)**2+a4(i)*r12 < 0. ) then
               if( .not. smt5(i) ) then
                  br(i) = 0.
                  bl(i) = 0.
                  b0(i) = 0.
               elseif( da1(i) > 0. ) then
                  br(i) = -2.*bl(i)
                  b0(i) =    -bl(i)
               else
                  bl(i) = -2.*br(i)
                  b0(i) =    -br(i)
               endif
             endif
             endif
          enddo
        else
          do i=is-1,ie+1
             bl(i) = al(i)   - q1(i)
             br(i) = al(i+1) - q1(i)
             b0(i) = bl(i) + br(i)
             smt5(i) = 3.*abs(b0(i)) < abs(bl(i)-br(i))
          enddo
        endif

!WMP
! fix edge issues
        if ( (.not. bounded_domain) .and. grid_type < 3) then
           if( is==1 ) then
              smt5(0) = bl(0)*br(0) < 0.
              smt5(1) = bl(1)*br(1) < 0.
           endif
           if( (ie+1)==npx ) then
              smt5(npx-1) = bl(npx-1)*br(npx-1) < 0.
              smt5(npx ) = bl(npx )*br(npx ) < 0.
           endif
        endif
      endif

!DEC$ VECTOR ALWAYS
      do i=is,ie+1
         if ( c(i,j) > 0. ) then
              fx1(i) = (1.-c(i,j))*(br(i-1) - c(i,j)*b0(i-1))
              flux(i,j) = q1(i-1)
         else
              fx1(i) = (1.+c(i,j))*(bl(i) + c(i,j)*b0(i))
              flux(i,j) = q1(i)
         endif
         if (smt5(i-1).or.smt5(i)) flux(i,j) = flux(i,j) + fx1(i)
      enddo

   endif
   goto 666

 else

! Monotonic constraints:
! ord = 8: PPM with Lin's PPM fast monotone constraint
! ord = 10: PPM with Lin's modification of Huynh 2nd constraint
! ord = 13: positive definite constraint

    do i=is-2,ie+2
          xt = 0.25*(q1(i+1) - q1(i-1))
       dm(i) = sign(min(abs(xt), max(q1(i-1), q1(i), q1(i+1)) - q1(i),  &
                         q1(i) - min(q1(i-1), q1(i), q1(i+1))), xt)
    enddo
    do i=is1,ie1+1
       al(i) = 0.5*(q1(i-1)+q1(i)) + r3*(dm(i-1)-dm(i))
    enddo

    if ( iord==8 ) then
       do i=is1, ie1
          xt = 2.*dm(i)
          bl(i) = -sign(min(abs(xt), abs(al(i  )-q1(i))), xt)
          br(i) =  sign(min(abs(xt), abs(al(i+1)-q1(i))), xt)
       enddo
    elseif ( iord==10 ) then
       do i=is1-2, ie1+1
          dq(i) = 2.*(q1(i+1) - q1(i))
       enddo
       do i=is1, ie1
          bl(i) = al(i  ) - q1(i)
          br(i) = al(i+1) - q1(i)
          if ( abs(dm(i-1))+abs(dm(i))+abs(dm(i+1)) < near_zero ) then
                   bl(i) = 0.
                   br(i) = 0.
          elseif( abs(3.*(bl(i)+br(i))) > abs(bl(i)-br(i)) ) then
                   pmp_2 = dq(i-1)
                   lac_2 = pmp_2 - 0.75*dq(i-2)
                   br(i) = min( max(0., pmp_2, lac_2), max(br(i), min(0., pmp_2, lac_2)) )
                   pmp_1 = -dq(i)
                   lac_1 = pmp_1 + 0.75*dq(i+1)
                   bl(i) = min( max(0., pmp_1, lac_1), max(bl(i), min(0., pmp_1, lac_1)) )
          endif
       enddo
    elseif ( iord==11 ) then
! This is emulation of 2nd van Leer scheme using PPM codes
       do i=is1, ie1
          xt = ppm_fac*dm(i)
          bl(i) = -sign(min(abs(xt), abs(al(i  )-q1(i))), xt)
          br(i) =  sign(min(abs(xt), abs(al(i+1)-q1(i))), xt)
       enddo
    elseif ( iord==7 .or. iord==12 ) then  ! positive definite (Lin & Rood 1996)
       do i=is1, ie1
          bl(i) = al(i)   - q1(i)
          br(i) = al(i+1) - q1(i)
          a4(i) = -3.*(bl(i) + br(i))
           da1(i) = br(i) - bl(i)
          ext5(i) = br(i)*bl(i) > 0.
          ext6(i) = abs(da1(i)) < -a4(i)
       enddo
       do i=is1, ie1
          if( ext6(i) ) then
            if( q1(i)+0.25/a4(i)*da1(i)**2+a4(i)*r12 < 0. ) then
                if( ext5(i) ) then
                   br(i) = 0.
                   bl(i) = 0.
                elseif( da1(i) > 0. ) then
                   br(i) = -2.*bl(i)
                else
                   bl(i) = -2.*br(i)
                endif
            endif
          endif
       enddo
    else
       do i=is1, ie1
          bl(i) = al(i  ) - q1(i)
          br(i) = al(i+1) - q1(i)
       enddo
    endif
! Positive definite constraint:
    if(iord==9 .or. iord==13) call pert_ppm(ie1-is1+1, q1(is1), bl(is1), br(is1), 0)

    if (.not. bounded_domain .and. grid_type<3) then
      if ( is==1 ) then
         bl(0) = s14*dm(-1) + s11*(q1(-1)-q1(0))

         xt = 0.5*(((2.*dxa(0,j)+dxa(-1,j))*q1(0)-dxa(0,j)*q1(-1))/(dxa(-1,j)+dxa(0,j)) &
            +      ((2.*dxa(1,j)+dxa( 2,j))*q1(1)-dxa(1,j)*q1( 2))/(dxa(1, j)+dxa(2,j)))
!        if ( iord==8 .or. iord==10 ) then
            xt = max(xt, min(q1(-1),q1(0),q1(1),q1(2)))
            xt = min(xt, max(q1(-1),q1(0),q1(1),q1(2)))
!        endif
         br(0) = xt - q1(0)
         bl(1) = xt - q1(1)
         xt = s15*q1(1) + s11*q1(2) - s14*dm(2)
         br(1) = xt - q1(1)
         bl(2) = xt - q1(2)

         br(2) = al(3) - q1(2)
         call pert_ppm(3, q1(0), bl(0), br(0), 1)
      endif
      if ( (ie+1)==npx ) then
         bl(npx-2) = al(npx-2) - q1(npx-2)

         xt = s15*q1(npx-1) + s11*q1(npx-2) + s14*dm(npx-2)
         br(npx-2) = xt - q1(npx-2)
         bl(npx-1) = xt - q1(npx-1)

         xt = 0.5*(((2.*dxa(npx-1,j)+dxa(npx-2,j))*q1(npx-1)-dxa(npx-1,j)*q1(npx-2))/(dxa(npx-2,j)+dxa(npx-1,j)) &
            +      ((2.*dxa(npx,  j)+dxa(npx+1,j))*q1(npx  )-dxa(npx,  j)*q1(npx+1))/(dxa(npx,  j)+dxa(npx+1,j)))
!        if ( iord==8 .or. iord==10 ) then
            xt = max(xt, min(q1(npx-2),q1(npx-1),q1(npx),q1(npx+1)))
            xt = min(xt, max(q1(npx-2),q1(npx-1),q1(npx),q1(npx+1)))
!        endif
         br(npx-1) = xt - q1(npx-1)
         bl(npx  ) = xt - q1(npx  )

         br(npx) = s11*(q1(npx+1)-q1(npx)) - s14*dm(npx+1)
         call pert_ppm(3, q1(npx-2), bl(npx-2), br(npx-2), 1)
      endif
    endif

  endif

  if ( iord==7 ) then
      do i=is-1,ie+1
           b0(i) = bl(i) + br(i)
         smt5(i) = bl(i) * br(i) < 0.
      enddo
      do i=is,ie+1
         if ( c(i,j) > 0. ) then
              fx1(i) = (1.-c(i,j))*(br(i-1) - c(i,j)*b0(i-1))
              flux(i,j) = q1(i-1)
         else
              fx1(i) = (1.+c(i,j))*(bl(i) + c(i,j)*b0(i))
              flux(i,j) = q1(i)
         endif
         if ( smt5(i-1).or.smt5(i) ) flux(i,j) = flux(i,j) + fx1(i)
      enddo
  else
      do i=is,ie+1
         if( c(i,j)>0. ) then
             flux(i,j) = q1(i-1) + (1.-c(i,j))*(br(i-1)-c(i,j)*(bl(i-1)+br(i-1)))
         else
             flux(i,j) = q1(i  ) + (1.+c(i,j))*(bl(i  )+c(i,j)*(bl(i)+br(i)))
         endif
      enddo
  endif

666   continue

 end subroutine xppm


 subroutine yppm(flux, q, c, jord, ifirst,ilast, isd,ied, js,je,jsd,jed, npx, npy, dya, bounded_domain, grid_type, lim_fac)
 integer, INTENT(IN) :: ifirst,ilast    ! Compute domain
 integer, INTENT(IN) :: isd,ied, js,je,jsd,jed
 integer, INTENT(IN) :: jord
 integer, INTENT(IN) :: npx, npy
 real   , INTENT(IN) :: q(ifirst:ilast,jsd:jed)
 real   , intent(in) :: c(isd:ied,js:je+1 )  ! Courant number
 real   , INTENT(OUT):: flux(ifirst:ilast,js:je+1)   !  Flux
 real   , intent(IN) :: dya(isd:ied,jsd:jed)
 logical, intent(IN) :: bounded_domain
 integer, intent(IN) :: grid_type
 real   , intent(IN) :: lim_fac
! Local:
 real:: dm(ifirst:ilast,js-2:je+2)
 real:: al(ifirst:ilast,js-1:je+2)
 real, dimension(ifirst:ilast,js-1:je+1):: bl, br, b0
 real:: dq(ifirst:ilast,js-3:je+2)
 real,    dimension(ifirst:ilast):: fx0, fx1, xt1, a4
 logical, dimension(ifirst:ilast,js-1:je+1):: smt5, smt6
 logical, dimension(ifirst:ilast):: hi5, hi6
 real:: x0, xt, qtmp, pmp_1, lac_1, pmp_2, lac_2
 integer:: i, j, js1, je3, je1, mord

   if ( .not.bounded_domain .and. grid_type < 3 ) then
! Cubed-sphere:
      js1 = max(3,js-1); je3 = min(npy-2,je+2)
                         je1 = min(npy-3,je+1)
   else
! Bounded_domain grid OR Doubly periodic domain:
      js1 = js-1;        je3 = je+2
                         je1 = je+1
   endif

 mord = abs(jord)

if ( jord < 7 ) then

   do j=js1, je3
      do i=ifirst,ilast
         al(i,j) = p1*(q(i,j-1)+q(i,j)) + p2*(q(i,j-2)+q(i,j+1))
      enddo
   enddo

   if ( .not. bounded_domain .and. grid_type<3 ) then
      if( js==1 ) then
        do i=ifirst,ilast
           al(i,0) = c1*q(i,-2) + c2*q(i,-1) + c3*q(i,0)
           al(i,1) = 0.5*(((2.*dya(i,0)+dya(i,-1))*q(i,0)-dya(i,0)*q(i,-1))/(dya(i,-1)+dya(i,0))   &
                   +      ((2.*dya(i,1)+dya(i,2))*q(i,1)-dya(i,1)*q(i,2))/(dya(i,1)+dya(i,2)))
           al(i,2) = c3*q(i,1) + c2*q(i,2) + c1*q(i,3)
        enddo
      endif
      if( (je+1)==npy ) then
        do i=ifirst,ilast
         al(i,npy-1) = c1*q(i,npy-3) + c2*q(i,npy-2) + c3*q(i,npy-1)
         al(i,npy) = 0.5*(((2.*dya(i,npy-1)+dya(i,npy-2))*q(i,npy-1)-dya(i,npy-1)*q(i,npy-2))/(dya(i,npy-2)+dya(i,npy-1))  &
                   +      ((2.*dya(i,npy)+dya(i,npy+1))*q(i,npy)-dya(i,npy)*q(i,npy+1))/(dya(i,npy)+dya(i,npy+1)))
         al(i,npy+1) = c3*q(i,npy) + c2*q(i,npy+1) + c1*q(i,npy+2)
        enddo
      endif
   endif

   if ( jord<0 ) then
      do j=js-1, je+2
         do i=ifirst,ilast
            al(i,j) = max(0., al(i,j))
         enddo
      enddo
   endif

   if ( mord==1 ) then
       do j=js-1,je+1
          do i=ifirst,ilast
             bl(i,j) = al(i,j  ) - q(i,j)
             br(i,j) = al(i,j+1) - q(i,j)
             b0(i,j) = bl(i,j) + br(i,j)
             smt5(i,j) = abs(lim_fac*b0(i,j)) < abs(bl(i,j)-br(i,j))
          enddo
       enddo
       do j=js,je+1
!DEC$ VECTOR ALWAYS
          do i=ifirst,ilast
             if ( c(i,j) > 0. ) then
                  fx1(i) = (1.-c(i,j))*(br(i,j-1) - c(i,j)*b0(i,j-1))
                  flux(i,j) = q(i,j-1)
             else
                  fx1(i) = (1.+c(i,j))*(bl(i,j) + c(i,j)*b0(i,j))
                  flux(i,j) = q(i,j)
             endif
             if (smt5(i,j-1).or.smt5(i,j)) flux(i,j) = flux(i,j) + fx1(i)
          enddo
       enddo

   elseif ( mord==2 ) then   ! Perfectly linear scheme
! Diffusivity: ord2 < ord5 < ord3 < ord4 < ord6  < ord7

      do j=js,je+1
!DEC$ VECTOR ALWAYS
         do i=ifirst,ilast
            xt = c(i,j)
            if ( xt > 0. ) then
                 qtmp = q(i,j-1)
                 flux(i,j) = qtmp + (1.-xt)*(al(i,j)-qtmp-xt*(al(i,j-1)+al(i,j)-(qtmp+qtmp)))
            else
                 qtmp = q(i,j)
                 flux(i,j) = qtmp + (1.+xt)*(al(i,j)-qtmp+xt*(al(i,j)+al(i,j+1)-(qtmp+qtmp)))
            endif
         enddo
      enddo

   elseif ( mord==3 ) then

        do j=js-1,je+1
           do i=ifirst,ilast
              bl(i,j) = al(i,j  ) - q(i,j)
              br(i,j) = al(i,j+1) - q(i,j)
              b0(i,j) = bl(i,j) + br(i,j)
                   x0 = abs(b0(i,j))
                   xt = abs(bl(i,j)-br(i,j))
              smt5(i,j) =    x0 < xt
              smt6(i,j) = 3.*x0 < xt
           enddo
        enddo
        do j=js,je+1
           do i=ifirst,ilast
              xt1(i) = c(i,j)
           enddo
           do i=ifirst,ilast
              if ( xt1(i) > 0. ) then
                   if( smt5(i,j-1) .or. smt6(i,j) ) then
                       flux(i,j) = q(i,j-1) + (1.-xt1(i))*(br(i,j-1) - xt1(i)*b0(i,j-1))
                   else
                       flux(i,j) = q(i,j-1)
                   endif
              else
                   if( smt6(i,j-1) .or. smt5(i,j) ) then
                       flux(i,j) = q(i,j) + (1.+xt1(i))*(bl(i,j) + xt1(i)*b0(i,j))
                   else
                       flux(i,j) = q(i,j)
                   endif
              endif
           enddo
        enddo

   elseif ( mord==4 ) then

        do j=js-1,je+1
           do i=ifirst,ilast
              bl(i,j) = al(i,j  ) - q(i,j)
              br(i,j) = al(i,j+1) - q(i,j)
              b0(i,j) = bl(i,j) + br(i,j)
                   x0 = abs(b0(i,j))
                   xt = abs(bl(i,j)-br(i,j))
              smt5(i,j) =    x0 < xt
              smt6(i,j) = 3.*x0 < xt
           enddo
        enddo
        do j=js,je+1
           do i=ifirst,ilast
              xt1(i) = c(i,j)
              hi5(i) = smt5(i,j-1) .and. smt5(i,j)
              hi6(i) = smt6(i,j-1) .or.  smt6(i,j)
              hi5(i) = hi5(i) .or. hi6(i)
           enddo
!DEC$ VECTOR ALWAYS
           do i=ifirst,ilast
                if ( xt1(i) > 0. ) then
                     fx1(i) = (1.-xt1(i))*(br(i,j-1) - xt1(i)*b0(i,j-1))
                     flux(i,j) = q(i,j-1)
                else
                     fx1(i) = (1.+xt1(i))*(bl(i,j) + xt1(i)*b0(i,j))
                     flux(i,j) = q(i,j)
                endif
                if ( hi5(i) ) flux(i,j) = flux(i,j) + fx1(i)
           enddo
        enddo

   else  ! mord=5,6
       if ( jord==5 ) then
          do j=js-1,je+1
             do i=ifirst,ilast
                bl(i,j) = al(i,j  ) - q(i,j)
                br(i,j) = al(i,j+1) - q(i,j)
                b0(i,j) = bl(i,j) + br(i,j)
                smt5(i,j) = bl(i,j)*br(i,j) < 0.
             enddo
          enddo
       else
          if ( jord==-5 ) then
             do j=js-1,je+1
                do i=ifirst,ilast
                   bl(i,j) = al(i,j  ) - q(i,j)
                   br(i,j) = al(i,j+1) - q(i,j)
                   b0(i,j) = bl(i,j) + br(i,j)
                   xt1(i) = br(i,j) - bl(i,j)
                    a4(i) = -3.*b0(i,j)
                   smt5(i,j) = bl(i,j)*br(i,j) < 0.
                enddo
                do i=ifirst,ilast
                   if( abs(xt1(i)) < -a4(i) ) then
                     if( q(i,j)+0.25/a4(i)*xt1(i)**2+a4(i)*r12 < 0. ) then
                       if( .not. smt5(i,j) ) then
                           br(i,j) = 0.
                           bl(i,j) = 0.
                           b0(i,j) = 0.
                       elseif( xt1(i) > 0. ) then
                           br(i,j) = -2.*bl(i,j)
                           b0(i,j) =    -bl(i,j)
                       else
                           bl(i,j) = -2.*br(i,j)
                           b0(i,j) =    -br(i,j)
                       endif
                     endif
                   endif
                enddo
             enddo
          else
             do j=js-1,je+1
                do i=ifirst,ilast
                   bl(i,j) = al(i,j  ) - q(i,j)
                   br(i,j) = al(i,j+1) - q(i,j)
                   b0(i,j) = bl(i,j) + br(i,j)
                   smt5(i,j) = 3.*abs(b0(i,j)) < abs(bl(i,j)-br(i,j))
                enddo
             enddo
          endif

!WMP
! fix edge issues
          if ( (.not. bounded_domain) .and. grid_type < 3) then
             if( js==1 ) then
                do i=ifirst,ilast
                   smt5(i,0) = bl(i,0)*br(i,0) < 0.
                   smt5(i,1) = bl(i,1)*br(i,1) < 0.
                enddo
             endif
             if( (je+1)==npy ) then
                do i=ifirst,ilast
                   smt5(i,npy-1) = bl(i,npy-1)*br(i,npy-1) < 0.
                   smt5(i,npy ) = bl(i,npy )*br(i,npy ) < 0.
                enddo
             endif
          endif
       endif

       do j=js,je+1
!DEC$ VECTOR ALWAYS
          do i=ifirst,ilast
             if ( c(i,j) > 0. ) then
                  fx1(i) = (1.-c(i,j))*(br(i,j-1) - c(i,j)*b0(i,j-1))
                  flux(i,j) = q(i,j-1)
             else
                  fx1(i) = (1.+c(i,j))*(bl(i,j) + c(i,j)*b0(i,j))
                  flux(i,j) = q(i,j)
             endif
             if (smt5(i,j-1).or.smt5(i,j)) flux(i,j) = flux(i,j) + fx1(i)
          enddo
       enddo

   endif
   return

else
! Monotonic constraints:
! ord = 8: PPM with Lin's PPM fast monotone constraint
! ord > 8: PPM with Lin's modification of Huynh 2nd constraint

  do j=js-2,je+2
     do i=ifirst,ilast
             xt = 0.25*(q(i,j+1) - q(i,j-1))
        dm(i,j) = sign(min(abs(xt), max(q(i,j-1), q(i,j), q(i,j+1)) - q(i,j),   &
                           q(i,j) - min(q(i,j-1), q(i,j), q(i,j+1))), xt)
     enddo
  enddo
  do j=js1,je1+1
     do i=ifirst,ilast
        al(i,j) = 0.5*(q(i,j-1)+q(i,j)) + r3*(dm(i,j-1) - dm(i,j))
     enddo
  enddo

  if ( jord==8 ) then
       do j=js1,je1
          do i=ifirst,ilast
             xt = 2.*dm(i,j)
             bl(i,j) = -sign(min(abs(xt), abs(al(i,j)-q(i,j))),   xt)
             br(i,j) =  sign(min(abs(xt), abs(al(i,j+1)-q(i,j))), xt)
          enddo
       enddo
  elseif ( jord==10 ) then
       do j=js1-2,je1+1
          do i=ifirst,ilast
             dq(i,j) = 2.*(q(i,j+1) - q(i,j))
          enddo
       enddo
       do j=js1,je1
          do i=ifirst,ilast
             bl(i,j) = al(i,j  ) - q(i,j)
             br(i,j) = al(i,j+1) - q(i,j)
             if ( abs(dm(i,j-1))+abs(dm(i,j))+abs(dm(i,j+1)) < near_zero ) then
                  bl(i,j) = 0.
                  br(i,j) = 0.
             elseif( abs(3.*(bl(i,j)+br(i,j))) > abs(bl(i,j)-br(i,j)) ) then
                  pmp_2 = dq(i,j-1)
                  lac_2 = pmp_2 - 0.75*dq(i,j-2)
                  br(i,j) = min(max(0.,pmp_2,lac_2), max(br(i,j), min(0.,pmp_2,lac_2)))
                  pmp_1 = -dq(i,j)
                  lac_1 = pmp_1 + 0.75*dq(i,j+1)
                  bl(i,j) = min(max(0.,pmp_1,lac_1), max(bl(i,j), min(0.,pmp_1,lac_1)))
             endif
          enddo
       enddo
  elseif ( jord==11 ) then
       do j=js1,je1
          do i=ifirst,ilast
             xt = ppm_fac*dm(i,j)
             bl(i,j) = -sign(min(abs(xt), abs(al(i,j)-q(i,j))),   xt)
             br(i,j) =  sign(min(abs(xt), abs(al(i,j+1)-q(i,j))), xt)
          enddo
       enddo
  elseif ( jord==7 .or. jord==12 ) then
       do j=js1,je1
          do i=ifirst,ilast
             bl(i,j) = al(i,j  ) - q(i,j)
             br(i,j) = al(i,j+1) - q(i,j)
              xt1(i) = br(i,j) - bl(i,j)
               a4(i) = -3.*(br(i,j) + bl(i,j))
              hi5(i) = bl(i,j)*br(i,j) > 0.
              hi6(i) = abs(xt1(i)) < -a4(i)
          enddo
          do i=ifirst,ilast
             if( hi6(i) ) then
                 if( q(i,j)+0.25/a4(i)*xt1(i)**2+a4(i)*r12 < 0. ) then
                    if( hi5(i) ) then
                        br(i,j) = 0.
                        bl(i,j) = 0.
                    elseif( xt1(i) > 0. ) then
                        br(i,j) = -2.*bl(i,j)
                    else
                        bl(i,j) = -2.*br(i,j)
                    endif
                 endif
             endif
          enddo
       enddo
  else
       do j=js1,je1
          do i=ifirst,ilast
             bl(i,j) = al(i,j  ) - q(i,j)
             br(i,j) = al(i,j+1) - q(i,j)
          enddo
       enddo
  endif
  if ( jord==9 .or. jord==13 ) then
! Positive definite constraint:
     do j=js1,je1
        call pert_ppm(ilast-ifirst+1, q(ifirst,j), bl(ifirst,j), br(ifirst,j), 0)
     enddo
  endif

  if (.not. bounded_domain .and. grid_type<3) then
    if( js==1 ) then
      do i=ifirst,ilast
         bl(i,0) = s14*dm(i,-1) + s11*(q(i,-1)-q(i,0))

         xt = 0.5*(((2.*dya(i,0)+dya(i,-1))*q(i,0)-dya(i,0)*q(i,-1))/(dya(i,-1)+dya(i,0))   &
            +      ((2.*dya(i,1)+dya(i,2))*q(i,1)-dya(i,1)*q(i,2))/(dya(i,1)+dya(i,2)))
!        if ( jord==8 .or. jord==10 ) then
            xt = max(xt, min(q(i,-1),q(i,0),q(i,1),q(i,2)))
            xt = min(xt, max(q(i,-1),q(i,0),q(i,1),q(i,2)))
!        endif
         br(i,0) = xt - q(i,0)
         bl(i,1) = xt - q(i,1)

         xt = s15*q(i,1) + s11*q(i,2) - s14*dm(i,2)
         br(i,1) = xt - q(i,1)
         bl(i,2) = xt - q(i,2)

         br(i,2) = al(i,3) - q(i,2)
      enddo
      call pert_ppm(3*(ilast-ifirst+1), q(ifirst,0), bl(ifirst,0), br(ifirst,0), 1)
    endif
    if( (je+1)==npy ) then
      do i=ifirst,ilast
         bl(i,npy-2) = al(i,npy-2) - q(i,npy-2)

         xt = s15*q(i,npy-1) + s11*q(i,npy-2) + s14*dm(i,npy-2)
         br(i,npy-2) = xt - q(i,npy-2)
         bl(i,npy-1) = xt - q(i,npy-1)

         xt = 0.5*(((2.*dya(i,npy-1)+dya(i,npy-2))*q(i,npy-1)-dya(i,npy-1)*q(i,npy-2))/(dya(i,npy-2)+dya(i,npy-1))  &
            +      ((2.*dya(i,npy)+dya(i,npy+1))*q(i,npy)-dya(i,npy)*q(i,npy+1))/(dya(i,npy)+dya(i,npy+1)))
!        if ( jord==8 .or. jord==10 ) then
            xt = max(xt, min(q(i,npy-2),q(i,npy-1),q(i,npy),q(i,npy+1)))
            xt = min(xt, max(q(i,npy-2),q(i,npy-1),q(i,npy),q(i,npy+1)))
!        endif
         br(i,npy-1) = xt - q(i,npy-1)
         bl(i,npy  ) = xt - q(i,npy)

         br(i,npy) = s11*(q(i,npy+1)-q(i,npy)) - s14*dm(i,npy+1)
     enddo
     call pert_ppm(3*(ilast-ifirst+1), q(ifirst,npy-2), bl(ifirst,npy-2), br(ifirst,npy-2), 1)
    endif
 end if

endif

  if ( jord==7 ) then
      do j=js-1,je+1
         do i=ifirst,ilast
              b0(i,j) = bl(i,j) + br(i,j)
            smt5(i,j) = bl(i,j) * br(i,j) < 0.
         enddo
      enddo
      do j=js,je+1
         do i=ifirst,ilast
            if ( c(i,j) > 0. ) then
                 fx1(i) = (1.-c(i,j))*(br(i,j-1) - c(i,j)*b0(i,j-1))
                 flux(i,j) = q(i,j-1)
            else
                 fx1(i) = (1.+c(i,j))*(bl(i,j) + c(i,j)*b0(i,j))
                 flux(i,j) = q(i,j)
            endif
            if ( smt5(i,j-1).or.smt5(i,j) ) flux(i,j) = flux(i,j) + fx1(i)
         enddo
      enddo
  else
      do j=js,je+1
         do i=ifirst,ilast
            if( c(i,j)>0. ) then
                flux(i,j) = q(i,j-1) + (1.-c(i,j))*(br(i,j-1)-c(i,j)*(bl(i,j-1)+br(i,j-1)))
            else
                flux(i,j) = q(i,j  ) + (1.+c(i,j))*(bl(i,j  )+c(i,j)*(bl(i,j)+br(i,j)))
            endif
         enddo
      enddo
  endif

 end subroutine yppm



 subroutine mp_ghost_ew(im, jm, km, nq, ifirst, ilast, jfirst, jlast, &
                              kfirst, klast, ng_w, ng_e, ng_s, ng_n, q_ghst, q)
!
! !INPUT PARAMETERS:
      integer, intent(in):: im, jm, km, nq
      integer, intent(in):: ifirst, ilast
      integer, intent(in):: jfirst, jlast
      integer, intent(in):: kfirst, klast
      integer, intent(in):: ng_e      ! eastern  zones to ghost
      integer, intent(in):: ng_w      ! western  zones to ghost
      integer, intent(in):: ng_s      ! southern zones to ghost
      integer, intent(in):: ng_n      ! northern zones to ghost
      real, intent(inout):: q_ghst(ifirst-ng_w:ilast+ng_e,jfirst-ng_s:jlast+ng_n,kfirst:klast,nq)
      real, optional, intent(in):: q(ifirst:ilast,jfirst:jlast,kfirst:klast,nq)
!
! !DESCRIPTION:
!
!     Ghost 4d east/west
!
! !REVISION HISTORY:
!    2005.08.22   Putman
!
!EOP
!------------------------------------------------------------------------------
!BOC
      integer :: i,j,k,n

      if (present(q)) then
         q_ghst(ifirst:ilast,jfirst:jlast,kfirst:klast,1:nq) = &
              q(ifirst:ilast,jfirst:jlast,kfirst:klast,1:nq)
      endif

!      Assume Periodicity in X-dir and not overlapping
      do n=1,nq
         do k=kfirst,klast
            do j=jfirst-ng_s,jlast+ng_n
               do i=1, ng_w
                  q_ghst(ifirst-i,j,k,n) = q_ghst(ilast-i+1,j,k,n)
               enddo
               do i=1, ng_e
                  q_ghst(ilast+i,j,k,n) = q_ghst(ifirst+i-1,j,k,n)
               enddo
            enddo
         enddo
      enddo

 end subroutine mp_ghost_ew



 subroutine pert_ppm(im, a0, al, ar, iv)
 integer, intent(in):: im
 integer, intent(in):: iv
 real, intent(in)   :: a0(im)
 real, intent(inout):: al(im), ar(im)
! Local:
 real a4, da1, da2, a6da, fmin
 integer i

!-----------------------------------
! Optimized PPM in perturbation form:
!-----------------------------------

 if ( iv==0 ) then
! Positive definite constraint
    do i=1,im
     if ( a0(i) <= 0. ) then
          al(i) = 0.
          ar(i) = 0.
     else
        a4 = -3.*(ar(i) + al(i))
       da1 =      ar(i) - al(i)
      if( abs(da1) < -a4 ) then
         fmin = a0(i) + 0.25/a4*da1**2 + a4*r12
         if( fmin < 0. ) then
             if( ar(i)>0. .and. al(i)>0. ) then
                 ar(i) = 0.
                 al(i) = 0.
             elseif( da1 > 0. ) then
                 ar(i) = -2.*al(i)
             else
                 al(i) = -2.*ar(i)
             endif
         endif
      endif
     endif
    enddo
 else
! Standard PPM constraint
    do i=1,im
       if ( al(i)*ar(i) < 0. ) then
            da1 = al(i) - ar(i)
            da2 = da1**2
            a6da = 3.*(al(i)+ar(i))*da1
! abs(a6da) > da2 --> 3.*abs(al+ar) > abs(al-ar)
            if( a6da < -da2 ) then
                ar(i) = -2.*al(i)
            elseif( a6da > da2 ) then
                al(i) = -2.*ar(i)
            endif
       else
! effect of dm=0 included here
            al(i) = 0.
            ar(i) = 0.
       endif
  enddo
 endif

 end subroutine pert_ppm

!TODO lmh 25may18: Need to ensure copy_corners is just ignored if not a global domain
 subroutine deln_flux(nord,is,ie,js,je, npx, npy, damp, q, fx, fy, gridstruct, bd, mass, damp_Km )
! Del-n damping for the cell-mean values (A grid)
!------------------
! nord = 0:   del-2
! nord = 1:   del-4
! nord = 2:   del-6
! nord = 3:   del-8 --> requires more ghosting than current
!------------------
   type(fv_grid_bounds_type), intent(IN) :: bd
   integer, intent(in):: nord            ! del-n
   integer, intent(in):: is,ie,js,je, npx, npy
   real, intent(in):: damp
   real, intent(in):: q(bd%isd:bd%ied, bd%jsd:bd%jed)  ! q ghosted on input
   type(fv_grid_type), intent(IN), target :: gridstruct
   real, optional, intent(in):: mass(bd%isd:bd%ied, bd%jsd:bd%jed)  ! q ghosted on input
   real, OPTIONAL, intent(in) :: damp_Km(bd%isd:bd%ied,bd%jsd:bd%jed) ! variable diffusion coeff for scalars
! diffusive fluxes:
   real, intent(inout):: fx(bd%is:bd%ie+1,bd%js:bd%je), fy(bd%is:bd%ie,bd%js:bd%je+1)
! local:
   real fx2(bd%isd:bd%ied+1,bd%jsd:bd%jed), fy2(bd%isd:bd%ied,bd%jsd:bd%jed+1)
   real d2(bd%isd:bd%ied,bd%jsd:bd%jed)
   real damp2
   integer i,j, n, nt, i1, i2, j1, j2

#ifdef USE_SG
   real, pointer, dimension(:,:)   :: dx, dy, rdxc, rdyc
   real, pointer, dimension(:,:,:) :: sin_sg
   dx       => gridstruct%dx
   dy       => gridstruct%dy
   rdxc     => gridstruct%rdxc
   rdyc     => gridstruct%rdyc
   sin_sg   => gridstruct%sin_sg
#endif

   i1 = is-1-nord;    i2 = ie+1+nord
   j1 = js-1-nord;    j2 = je+1+nord

   if ( .not. present(mass) .and. .not. present(damp_Km) ) then
     do j=j1, j2
        do i=i1,i2
           d2(i,j) = damp*q(i,j)
        enddo
     enddo
   else
     do j=j1, j2
        do i=i1,i2
           d2(i,j) = q(i,j)
        enddo
     enddo
   endif

   if( nord>0 ) call copy_corners(d2, npx, npy, 1, gridstruct%bounded_domain, bd, &
      gridstruct%sw_corner, gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)

   do j=js-nord,je+nord
      do i=is-nord,ie+nord+1
#ifdef USE_SG
         fx2(i,j) = 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))*dy(i,j)*(d2(i-1,j)-d2(i,j))*rdxc(i,j)
#else
         fx2(i,j) = gridstruct%del6_v(i,j)*(d2(i-1,j)-d2(i,j))
#endif
      enddo
   enddo

   if( nord>0 ) call copy_corners(d2, npx, npy, 2, gridstruct%bounded_domain, bd, &
      gridstruct%sw_corner, gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)
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

!----------
! high-order
!----------

   do n=1, nord

      nt = nord-n

      do j=js-nt-1,je+nt+1
         do i=is-nt-1,ie+nt+1
            d2(i,j) = (fx2(i,j)-fx2(i+1,j)+fy2(i,j)-fy2(i,j+1))*gridstruct%rarea(i,j)
         enddo
      enddo

      call copy_corners(d2, npx, npy, 1, gridstruct%bounded_domain, bd, &
           gridstruct%sw_corner, gridstruct%se_corner, gridstruct%nw_corner, gridstruct%ne_corner)
      do j=js-nt,je+nt
         do i=is-nt,ie+nt+1
#ifdef USE_SG
            fx2(i,j) = 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))*dy(i,j)*(d2(i,j)-d2(i-1,j))*rdxc(i,j)
#else
            fx2(i,j) = gridstruct%del6_v(i,j)*(d2(i,j)-d2(i-1,j))
#endif
         enddo
      enddo

      call copy_corners(d2, npx, npy, 2, gridstruct%bounded_domain, bd, &
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

!---------------------------------------------
! Add the diffusive fluxes to the flux arrays:
!---------------------------------------------

   if ( present(mass) ) then
      ! Apply mass weighting to diffusive fluxes:
      if (present(damp_Km)) then
        do j=js,je
           do i=is,ie+1
              damp2 = 0.25*damp*(damp_km(i-1,j)+damp_km(i,j))
              fx(i,j) = fx(i,j) + damp2*(mass(i-1,j)+mass(i,j))*fx2(i,j)
           enddo
        enddo
        do j=js,je+1
           do i=is,ie
              damp2 = 0.25*damp*(damp_km(i,j-1)+damp_km(i,j))
              fy(i,j) = fy(i,j) + damp2*(mass(i,j-1)+mass(i,j))*fy2(i,j)
           enddo
        enddo
      else
        damp2 = 0.5*damp
        do j=js,je
           do i=is,ie+1
              fx(i,j) = fx(i,j) + damp2*(mass(i-1,j)+mass(i,j))*fx2(i,j)
           enddo
        enddo
        do j=js,je+1
           do i=is,ie
              fy(i,j) = fy(i,j) + damp2*(mass(i,j-1)+mass(i,j))*fy2(i,j)
           enddo
        enddo
      endif
   else
      if (present(damp_Km)) then
        do j=js,je
           do i=is,ie+1
              damp2 = 0.25*damp*(damp_km(i-1,j)+damp_km(i,j))
              fx(i,j) = fx(i,j) + damp2*fx2(i,j)
           enddo
        enddo
        do j=js,je+1
           do i=is,ie
              damp2 = 0.25*damp*(damp_km(i,j-1)+damp_km(i,j))
              fy(i,j) = fy(i,j) + damp2*fy2(i,j)
           enddo
        enddo
      else
      
        do j=js,je
           do i=is,ie+1
              fx(i,j) = fx(i,j) + fx2(i,j)
           enddo
        enddo
        do j=js,je+1
           do i=is,ie
              fy(i,j) = fy(i,j) + fy2(i,j)
           enddo
        enddo
     endif
   endif

 end subroutine deln_flux


end module tp_core_mod

!***********************************************************************
!*                   GNU Lesser General Public License
!*
!* This file is part of the FV3 dynamical core.
!*
!* The FV3 dynamical core is free software: you can redistribute it
!* and/or modify it under the terms of the
!* GNU Lesser General Public License as published by the
!* Free Software Foundation, either version 3 of the License, or
!* (at your option) any later version.
!*
!* The FV3 dynamical core is distributed in the hope that it will be
!* useful, but WITHOUT ANY WARRANTY; without even the implied warranty
!* of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!* See the GNU General Public License for more details.
!*
!* You should have received a copy of the GNU Lesser General Public
!* License along with the FV3 dynamical core.
!* If not, see <http://www.gnu.org/licenses/>.
!***********************************************************************

module a2b_edge_mod

  use swcore_shim_mod, only: great_circle_dist

  use swcore_shim_mod, only: fv_grid_type, fv_grid_bounds_type, R_GRID

  implicit none

  real, parameter:: r3 = 1./3.
!----------------------------
! 4-pt Lagrange interpolation
!----------------------------
  real, parameter:: a1 =  0.5625  !  9/16
  real, parameter:: a2 = -0.0625  ! -1/16
!----------------------
! PPM volume mean form:
!----------------------
  real, parameter:: b1 =  7./12.     ! 0.58333333
  real, parameter:: b2 = -1./12.

  private
  public :: a2b_ord2, a2b_ord4

contains

  subroutine a2b_ord4(qin, qout, gridstruct, npx, npy, is, ie, js, je, ng, replace)
  integer, intent(IN):: npx, npy, is, ie, js, je, ng
  real, intent(INOUT)::  qin(is-ng:ie+ng,js-ng:je+ng)   ! A-grid field
  real, intent(INOUT):: qout(is-ng:ie+ng,js-ng:je+ng)   ! Output  B-grid field
  type(fv_grid_type), intent(IN), target :: gridstruct
  logical, optional, intent(IN):: replace
! local: compact 4-pt cubic
  real, parameter:: c1 =  2./3.
  real, parameter:: c2 = -1./6.
! Parabolic spline
! real, parameter:: c1 =  0.75
! real, parameter:: c2 = -0.25

  real qx(is:ie+1,js-ng:je+ng)
  real qy(is-ng:ie+ng,js:je+1)
  real qxx(is-ng:ie+ng,js-ng:je+ng)
  real qyy(is-ng:ie+ng,js-ng:je+ng)
  real g_in, g_ou
  real:: p0(2)
  real:: q1(is-1:ie+1), q2(js-1:je+1)
  integer:: i, j, is1, js1, is2, js2, ie1, je1

  real, pointer, dimension(:,:,:) :: grid, agrid
  real, pointer, dimension(:,:)   :: dxa, dya
  real(kind=R_GRID), pointer, dimension(:) :: edge_w, edge_e, edge_s, edge_n

  edge_w => gridstruct%edge_w
  edge_e => gridstruct%edge_e
  edge_s => gridstruct%edge_s
  edge_n => gridstruct%edge_n

  grid => gridstruct%grid
  agrid => gridstruct%agrid
  dxa => gridstruct%dxa
  dya => gridstruct%dya

  if (gridstruct%grid_type < 3) then

    is1 = max(1,is-1)
    js1 = max(1,js-1)
    is2 = max(2,is)
    js2 = max(2,js)

    ie1 = min(npx-1,ie+1)
    je1 = min(npy-1,je+1)

! Corners:
! 3-way extrapolation
    if (gridstruct%bounded_domain) then

    do j=js-2,je+2
       do i=is,ie+1
          qx(i,j) = b2*(qin(i-2,j)+qin(i+1,j)) + b1*(qin(i-1,j)+qin(i,j))
       enddo
    enddo


    else

    if ( gridstruct%sw_corner ) then
          p0(1:2) = grid(1,1,1:2)
        qout(1,1) = (extrap_corner(p0, agrid(1,1,1:2), agrid( 2, 2,1:2), qin(1,1), qin( 2, 2)) + &
                     extrap_corner(p0, agrid(0,1,1:2), agrid(-1, 2,1:2), qin(0,1), qin(-1, 2)) + &
                     extrap_corner(p0, agrid(1,0,1:2), agrid( 2,-1,1:2), qin(1,0), qin( 2,-1)))*r3

    endif
    if ( gridstruct%se_corner ) then
            p0(1:2) = grid(npx,1,1:2)
        qout(npx,1) = (extrap_corner(p0, agrid(npx-1,1,1:2), agrid(npx-2, 2,1:2), qin(npx-1,1), qin(npx-2, 2)) + &
                       extrap_corner(p0, agrid(npx-1,0,1:2), agrid(npx-2,-1,1:2), qin(npx-1,0), qin(npx-2,-1)) + &
                       extrap_corner(p0, agrid(npx  ,1,1:2), agrid(npx+1, 2,1:2), qin(npx  ,1), qin(npx+1, 2)))*r3
    endif
    if ( gridstruct%ne_corner ) then
              p0(1:2) = grid(npx,npy,1:2)
        qout(npx,npy) = (extrap_corner(p0, agrid(npx-1,npy-1,1:2), agrid(npx-2,npy-2,1:2), qin(npx-1,npy-1), qin(npx-2,npy-2)) + &
                         extrap_corner(p0, agrid(npx  ,npy-1,1:2), agrid(npx+1,npy-2,1:2), qin(npx  ,npy-1), qin(npx+1,npy-2)) + &
                         extrap_corner(p0, agrid(npx-1,npy  ,1:2), agrid(npx-2,npy+1,1:2), qin(npx-1,npy  ), qin(npx-2,npy+1)))*r3
    endif
    if ( gridstruct%nw_corner ) then
            p0(1:2) = grid(1,npy,1:2)
        qout(1,npy) = (extrap_corner(p0, agrid(1,npy-1,1:2), agrid( 2,npy-2,1:2), qin(1,npy-1), qin( 2,npy-2)) + &
                       extrap_corner(p0, agrid(0,npy-1,1:2), agrid(-1,npy-2,1:2), qin(0,npy-1), qin(-1,npy-2)) + &
                       extrap_corner(p0, agrid(1,npy,  1:2), agrid( 2,npy+1,1:2), qin(1,npy  ), qin( 2,npy+1)))*r3
    endif

!------------
! X-Interior:
!------------
    do j=max(1,js-2),min(npy-1,je+2)
       do i=max(3,is), min(npx-2,ie+1)
          qx(i,j) = b2*(qin(i-2,j)+qin(i+1,j)) + b1*(qin(i-1,j)+qin(i,j))
       enddo
    enddo

    ! *** West Edges:
    if ( is==1 ) then
       do j=js1, je1
          q2(j) = (qin(0,j)*dxa(1,j) + qin(1,j)*dxa(0,j))/(dxa(0,j) + dxa(1,j))
       enddo
       do j=js2, je1
          qout(1,j) = edge_w(j)*q2(j-1) + (1.-edge_w(j))*q2(j)
       enddo
!
       do j=max(1,js-2),min(npy-1,je+2)
             g_in = dxa(2,j) / dxa(1,j)
             g_ou = dxa(-1,j) / dxa(0,j)
          qx(1,j) = 0.5*( ((2.+g_in)*qin(1,j)-qin( 2,j))/(1.+g_in) +          &
                          ((2.+g_ou)*qin(0,j)-qin(-1,j))/(1.+g_ou) )
          qx(2,j) = ( 3.*(g_in*qin(1,j)+qin(2,j))-(g_in*qx(1,j)+qx(3,j)) ) / (2.+2.*g_in)
       enddo
    endif

    ! East Edges:
    if ( (ie+1)==npx ) then
       do j=js1, je1
          q2(j) = (qin(npx-1,j)*dxa(npx,j) + qin(npx,j)*dxa(npx-1,j))/(dxa(npx-1,j) + dxa(npx,j))
       enddo
       do j=js2, je1
          qout(npx,j) = edge_e(j)*q2(j-1) + (1.-edge_e(j))*q2(j)
       enddo
!
       do j=max(1,js-2),min(npy-1,je+2)
              g_in = dxa(npx-2,j) / dxa(npx-1,j)
              g_ou = dxa(npx+1,j) / dxa(npx,j)
          qx(npx,j) = 0.5*( ((2.+g_in)*qin(npx-1,j)-qin(npx-2,j))/(1.+g_in) +          &
                            ((2.+g_ou)*qin(npx,  j)-qin(npx+1,j))/(1.+g_ou) )
          qx(npx-1,j) = (3.*(qin(npx-2,j)+g_in*qin(npx-1,j)) - (g_in*qx(npx,j)+qx(npx-2,j)))/(2.+2.*g_in)
       enddo
    endif

    end if
!------------
! Y-Interior:
!------------

    if (gridstruct%bounded_domain) then


    do j=js,je+1
       do i=is-2,ie+2
          qy(i,j) = b2*(qin(i,j-2)+qin(i,j+1)) + b1*(qin(i,j-1) + qin(i,j))
       enddo
    enddo

    else

    do j=max(3,js),min(npy-2,je+1)
       do i=max(1,is-2), min(npx-1,ie+2)
          qy(i,j) = b2*(qin(i,j-2)+qin(i,j+1)) + b1*(qin(i,j-1) + qin(i,j))
       enddo
    enddo

    ! South Edges:
    if ( js==1 ) then
       do i=is1, ie1
          q1(i) = (qin(i,0)*dya(i,1) + qin(i,1)*dya(i,0))/(dya(i,0) + dya(i,1))
       enddo
       do i=is2, ie1
          qout(i,1) = edge_s(i)*q1(i-1) + (1.-edge_s(i))*q1(i)
       enddo
!
       do i=max(1,is-2),min(npx-1,ie+2)
             g_in = dya(i,2) / dya(i,1)
             g_ou = dya(i,-1) / dya(i,0)
          qy(i,1) = 0.5*( ((2.+g_in)*qin(i,1)-qin(i,2))/(1.+g_in) +          &
                          ((2.+g_ou)*qin(i,0)-qin(i,-1))/(1.+g_ou) )
          qy(i,2) = (3.*(g_in*qin(i,1)+qin(i,2)) - (g_in*qy(i,1)+qy(i,3)))/(2.+2.*g_in)
       enddo
    endif

    ! North Edges:
    if ( (je+1)==npy ) then
       do i=is1, ie1
          q1(i) = (qin(i,npy-1)*dya(i,npy) + qin(i,npy)*dya(i,npy-1))/(dya(i,npy-1)+dya(i,npy))
       enddo
       do i=is2, ie1
          qout(i,npy) = edge_n(i)*q1(i-1) + (1.-edge_n(i))*q1(i)
       enddo
!
       do i=max(1,is-2),min(npx-1,ie+2)
              g_in = dya(i,npy-2) / dya(i,npy-1)
              g_ou = dya(i,npy+1) / dya(i,npy)
          qy(i,npy) = 0.5*( ((2.+g_in)*qin(i,npy-1)-qin(i,npy-2))/(1.+g_in) +          &
                            ((2.+g_ou)*qin(i,npy  )-qin(i,npy+1))/(1.+g_ou) )
          qy(i,npy-1) = (3.*(qin(i,npy-2)+g_in*qin(i,npy-1)) - (g_in*qy(i,npy)+qy(i,npy-2)))/(2.+2.*g_in)
       enddo
    endif

    end if
!--------------------------------------

    if (gridstruct%bounded_domain) then

    do j=js, je+1
       do i=is,ie+1
          qxx(i,j) = a2*(qx(i,j-2)+qx(i,j+1)) + a1*(qx(i,j-1)+qx(i,j))
       enddo
    enddo

    do j=js,je+1
       do i=is,ie+1
          qyy(i,j) = a2*(qy(i-2,j)+qy(i+1,j)) + a1*(qy(i-1,j)+qy(i,j))
       enddo

       do i=is,ie+1
          qout(i,j) = 0.5*(qxx(i,j) + qyy(i,j))   ! averaging
       enddo
    enddo



    else

    do j=max(3,js),min(npy-2,je+1)
       do i=max(2,is),min(npx-1,ie+1)
          qxx(i,j) = a2*(qx(i,j-2)+qx(i,j+1)) + a1*(qx(i,j-1)+qx(i,j))
       enddo
    enddo

    if ( js==1 ) then
       do i=max(2,is),min(npx-1,ie+1)
          qxx(i,2) = c1*(qx(i,1)+qx(i,2))+c2*(qout(i,1)+qxx(i,3))
       enddo
    endif
    if ( (je+1)==npy ) then
       do i=max(2,is),min(npx-1,ie+1)
          qxx(i,npy-1) = c1*(qx(i,npy-2)+qx(i,npy-1))+c2*(qout(i,npy)+qxx(i,npy-2))
       enddo
    endif


    do j=max(2,js),min(npy-1,je+1)
       do i=max(3,is),min(npx-2,ie+1)
          qyy(i,j) = a2*(qy(i-2,j)+qy(i+1,j)) + a1*(qy(i-1,j)+qy(i,j))
       enddo
       if ( is==1 ) qyy(2,j) = c1*(qy(1,j)+qy(2,j))+c2*(qout(1,j)+qyy(3,j))
       if((ie+1)==npx) qyy(npx-1,j) = c1*(qy(npx-2,j)+qy(npx-1,j))+c2*(qout(npx,j)+qyy(npx-2,j))

       do i=max(2,is),min(npx-1,ie+1)
          qout(i,j) = 0.5*(qxx(i,j) + qyy(i,j))   ! averaging
       enddo
    enddo

    end if

 else  ! grid_type>=3
!------------------------
! Doubly periodic domain:
!------------------------
! X-sweep: PPM
    do j=js-2,je+2
       do i=is,ie+1
          qx(i,j) = b1*(qin(i-1,j)+qin(i,j)) + b2*(qin(i-2,j)+qin(i+1,j))
       enddo
    enddo
! Y-sweep: PPM
    do j=js,je+1
       do i=is-2,ie+2
          qy(i,j) = b1*(qin(i,j-1)+qin(i,j)) + b2*(qin(i,j-2)+qin(i,j+1))
       enddo
    enddo

    do j=js,je+1
       do i=is,ie+1
          qout(i,j) = 0.5*( a1*(qx(i,j-1)+qx(i,j  ) + qy(i-1,j)+qy(i,  j)) +  &
                            a2*(qx(i,j-2)+qx(i,j+1) + qy(i-2,j)+qy(i+1,j)) )
       enddo
    enddo
 endif

    if ( present(replace) ) then
       if ( replace ) then
          do j=js,je+1
          do i=is,ie+1
             qin(i,j) = qout(i,j)
          enddo
          enddo
       endif
    endif

  end subroutine a2b_ord4

  subroutine a2b_ord2(qin, qout, gridstruct, npx, npy, is, ie, js, je, ng, replace)
    integer, intent(IN   ) :: npx, npy, is, ie, js, je, ng
    real   , intent(INOUT) ::  qin(is-ng:ie+ng,js-ng:je+ng)   ! A-grid field
    real   , intent(  OUT) :: qout(is-ng:ie+ng,js-ng:je+ng)   ! Output  B-grid field
    type(fv_grid_type), intent(IN), target :: gridstruct
    logical, optional, intent(IN) ::  replace
    ! local:
    real q1(npx), q2(npy)
    integer :: i,j
    integer :: is1, js1, is2, js2, ie1, je1

    real, pointer, dimension(:,:,:) :: grid, agrid
    real, pointer, dimension(:,:)   :: dxa, dya

  real(kind=R_GRID), pointer, dimension(:) :: edge_w, edge_e, edge_s, edge_n

  edge_w => gridstruct%edge_w
  edge_e => gridstruct%edge_e
  edge_s => gridstruct%edge_s
  edge_n => gridstruct%edge_n

    grid => gridstruct%grid
    agrid => gridstruct%agrid
    dxa => gridstruct%dxa
    dya => gridstruct%dya

    if (gridstruct%grid_type < 3) then

       if (gridstruct%bounded_domain) then

          do j=js,je+1
             do i=is,ie+1
                qout(i,j) = 0.25*(qin(i-1,j-1)+qin(i,j-1)+qin(i-1,j)+qin(i,j))
             enddo
          enddo

       else

    is1 = max(1,is-1)
    js1 = max(1,js-1)
    is2 = max(2,is)
    js2 = max(2,js)

    ie1 = min(npx-1,ie+1)
    je1 = min(npy-1,je+1)

    do j=js2,je1
       do i=is2,ie1
          qout(i,j) = 0.25*(qin(i-1,j-1)+qin(i,j-1)+qin(i-1,j)+qin(i,j))
       enddo
    enddo

! Fix the 4 Corners:
    if ( gridstruct%sw_corner ) qout(1,    1) = r3*(qin(1,        1)+qin(1,      0)+qin(0,      1))
    if ( gridstruct%se_corner ) qout(npx,  1) = r3*(qin(npx-1,    1)+qin(npx-1,  0)+qin(npx,    1))
    if ( gridstruct%ne_corner ) qout(npx,npy) = r3*(qin(npx-1,npy-1)+qin(npx,npy-1)+qin(npx-1,npy))
    if ( gridstruct%nw_corner ) qout(1,  npy) = r3*(qin(1,    npy-1)+qin(0,  npy-1)+qin(1,    npy))

    ! *** West Edges:
    if ( is==1 ) then
       do j=js1, je1
          q2(j) = 0.5*(qin(0,j) + qin(1,j))
       enddo
       do j=js2, je1
          qout(1,j) = edge_w(j)*q2(j-1) + (1.-edge_w(j))*q2(j)
       enddo
    endif

    ! East Edges:
    if ( (ie+1)==npx ) then
       do j=js1, je1
          q2(j) = 0.5*(qin(npx-1,j) + qin(npx,j))
       enddo
       do j=js2, je1
          qout(npx,j) = edge_e(j)*q2(j-1) + (1.-edge_e(j))*q2(j)
       enddo
    endif

    ! South Edges:
    if ( js==1 ) then
       do i=is1, ie1
          q1(i) = 0.5*(qin(i,0) + qin(i,1))
       enddo
       do i=is2, ie1
          qout(i,1) = edge_s(i)*q1(i-1) + (1.-edge_s(i))*q1(i)
       enddo
    endif

    ! North Edges:
    if ( (je+1)==npy ) then
       do i=is1, ie1
          q1(i) = 0.5*(qin(i,npy-1) + qin(i,npy))
       enddo
       do i=is2, ie1
          qout(i,npy) = edge_n(i)*q1(i-1) + (1.-edge_n(i))*q1(i)
       enddo
    endif

 end if

 else

    do j=js,je+1
       do i=is,ie+1
          qout(i,j) = 0.25*(qin(i-1,j-1)+qin(i,j-1)+qin(i-1,j)+qin(i,j))
       enddo
    enddo

 endif


    if ( present(replace) ) then
       if ( replace ) then
          do j=js,je+1
             do i=is,ie+1
                qin(i,j) = qout(i,j)
             enddo
          enddo
       endif
    endif

  end subroutine a2b_ord2

  real function extrap_corner ( p0, p1, p2, q1, q2 )
    real, intent(in ), dimension(2):: p0, p1, p2
    real, intent(in ):: q1, q2
    real:: x1, x2

    x1 = great_circle_dist( real(p1,kind=R_GRID), real(p0,kind=R_GRID) )
    x2 = great_circle_dist( real(p2,kind=R_GRID), real(p0,kind=R_GRID) )

    extrap_corner = q1 + x1/(x2-x1) * (q1-q2)

  end function extrap_corner

end module a2b_edge_mod

module dsw_extract_mod
  ! VERBATIM sw_core.F90 slices: d_sw (494-1606), del6_vt_flux (1608-1737),
  ! smag_corner (1937-2024), xtp_u (2154-2521), ytp_v (2524-2998), plus the
  ! shared c_sw-tree pieces reused from sw_core_extract_mod and the pure
  ! index-shuffle fill_corners bodies from tools/fv_mp_mod.F90 (r8 variants;
  ! module bounds provided by swcore_shim_mod::set_fill_corner_bounds).
  use swcore_shim_mod
  use sw_core_extract_mod, only: d2a2c_vect, fill2_4corners, fill_4corners
  use tp_core_mod, only: fv_tp_2d, copy_corners, pert_ppm
  use a2b_edge_mod, only: a2b_ord4
  implicit none
  ! sw_core.F90 module-level constants (verbatim, production branch)
  real, parameter:: r3 = 1./3.
  real, parameter:: t11=27./28., t12=-13./28., t13=3./7., t14=6./7., t15=3./28.
  real, parameter:: s11=11./14., s13=-13./14., s14=4./7., s15=3./14.
  real, parameter:: near_zero = 1.E-9
  real, parameter:: big_number_sw = 1.E8
  real, parameter:: p1 =  7./12.
  real, parameter:: p2 = -1./12.
  real, parameter:: a1 =  0.5625
  real, parameter:: a2 = -0.0625
  real, parameter:: c1 = -2./14.
  real, parameter:: c2 = 11./14.
  real, parameter:: c3 =  5./14.

  interface fill_corners
     module procedure fill_corners_2d_r8
     module procedure fill_corners_xy_2d_r8
  end interface
  interface fill_corners_dgrid
     module procedure fill_corners_dgrid_r8
  end interface
  interface fill_corners_cgrid
     module procedure fill_corners_cgrid_r8
  end interface
  interface fill_corners_agrid
     module procedure fill_corners_agrid_r8
  end interface

contains
   subroutine d_sw(delpc, delp,  ptc,   pt, u,  v, w, uc,vc, &
                   ua, va, divg_d, xflux, yflux, cx, cy,              &
                   crx_adv, cry_adv,  xfx_adv, yfx_adv, q_con, z_rat, kgb, heat_source,    &
                   diss_est, zvir, sphum, nq, q, k, km, inline_q,  &
                   dt, hord_tr, hord_mt, hord_vt, hord_tm, hord_dp, nord,   &
                   nord_v, nord_w, nord_t, dddmp, d2_bg, d4_bg, damp_v, damp_w, &
                   damp_t, d_con, hydrostatic, gridstruct, flagstruct, use_cond, bd)

      integer, intent(IN):: hord_tr, hord_mt, hord_vt, hord_tm, hord_dp
      integer, intent(IN):: nord   ! nord=1 divergence damping; (del-4) or 3 (del-8)
      integer, intent(IN):: nord_v ! vorticity damping
      integer, intent(IN):: nord_w ! vertical velocity
      integer, intent(IN):: nord_t ! pt
      integer, intent(IN):: sphum, nq, k, km
      real   , intent(IN):: dt, dddmp, d2_bg, d4_bg, d_con
      real   , intent(IN):: zvir
      real   , intent(IN):: damp_v, damp_w, damp_t, kgb
      logical, intent(IN):: use_cond
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT):: divg_d(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) ! divergence
      real, intent(IN), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: z_rat
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: delp, pt, ua, va
      real, intent(INOUT), dimension(bd%isd:      ,  bd%jsd:      ):: w, q_con
      real, intent(INOUT), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: u, vc
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v, uc
      real, intent(INOUT):: q(bd%isd:bd%ied,bd%jsd:bd%jed,km,nq)
      real, intent(OUT),   dimension(bd%isd:bd%ied,  bd%jsd:bd%jed)  :: delpc, ptc
      real, intent(OUT),   dimension(bd%is:bd%ie,bd%js:bd%je):: heat_source
      real, intent(OUT),   dimension(bd%is:bd%ie,bd%js:bd%je):: diss_est
! The flux capacitors:
      real, intent(INOUT):: xflux(bd%is:bd%ie+1,bd%js:bd%je  )
      real, intent(INOUT):: yflux(bd%is:bd%ie  ,bd%js:bd%je+1)
!------------------------
      real, intent(INOUT)::    cx(bd%is:bd%ie+1,bd%jsd:bd%jed  )
      real, intent(INOUT)::    cy(bd%isd:bd%ied,bd%js:bd%je+1)
      logical, intent(IN):: hydrostatic
      logical, intent(IN):: inline_q
      real, intent(OUT), dimension(bd%is:bd%ie+1,bd%jsd:bd%jed):: crx_adv, xfx_adv
      real, intent(OUT), dimension(bd%isd:bd%ied,bd%js:bd%je+1):: cry_adv, yfx_adv
      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
      logical:: sw_corner, se_corner, ne_corner, nw_corner
      real :: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real :: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
!---
      real :: fx2(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real :: fy2(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
      real :: fx3(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real :: fy3(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
      real :: dw(bd%is:bd%ie,bd%js:bd%je) !  work array
!---
      real, dimension(bd%is:bd%ie+1,bd%js:bd%je+1):: ub, vb
      real :: wk(bd%isd:bd%ied,bd%jsd:bd%jed) !  work array
      real :: smag_q(bd%isd:bd%ied,bd%jsd:bd%jed)
      real :: ke(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) !  needs this for corner_comm
      real :: vort(bd%isd:bd%ied,bd%jsd:bd%jed)     ! Vorticity
      real ::   fx(bd%is:bd%ie+1,bd%js:bd%je  )  ! 1-D X-direction Fluxes
      real ::   fy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! 1-D Y-direction Fluxes
      real :: ra_x(bd%is:bd%ie,bd%jsd:bd%jed)
      real :: ra_y(bd%isd:bd%ied,bd%js:bd%je)
      real :: gx(bd%is:bd%ie+1,bd%js:bd%je  )
      real :: gy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! work Y-dir flux array
      logical :: fill_c

      real :: dt2, dt4, dt5, dt6
      real :: damp, damp2, damp4, dd8, u2, v2, du2, dv2
      real :: u_lon, tmp
      integer :: i,j, is2, ie1, js2, je1, n, nt, n2, iq
      logical :: prevent_diss_cooling

      real, pointer, dimension(:,:) :: area, area_c, rarea

      real, pointer, dimension(:,:,:) :: sin_sg
      real, pointer, dimension(:,:)   :: cosa_u, cosa_v, cosa_s
      real, pointer, dimension(:,:)   :: sina_u, sina_v
      real, pointer, dimension(:,:)   :: rsin_u, rsin_v, rsina
      real, pointer, dimension(:,:)   :: f0, rsin2, divg_u, divg_v

      real, pointer, dimension(:,:) ::  cosa, dx, dy, dxc, dyc, rdxa, rdya, rdx, rdy

      integer :: is,  ie,  js,  je
      integer :: isd, ied, jsd, jed
      integer :: npx, npy, ng
      logical :: bounded_domain

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je
      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed
      ng  = bd%ng

      npx      = flagstruct%npx
      npy      = flagstruct%npy
      bounded_domain = gridstruct%bounded_domain

      area      => gridstruct%area
      rarea     => gridstruct%rarea
      sin_sg    => gridstruct%sin_sg
      cosa_u    => gridstruct%cosa_u
      cosa_v    => gridstruct%cosa_v
      cosa_s    => gridstruct%cosa_s
      sina_u    => gridstruct%sina_u
      sina_v    => gridstruct%sina_v
      rsin_u    => gridstruct%rsin_u
      rsin_v    => gridstruct%rsin_v
      rsina     => gridstruct%rsina
      f0        => gridstruct%f0
      rsin2     => gridstruct%rsin2
      divg_u    => gridstruct%divg_u
      divg_v    => gridstruct%divg_v
      cosa      => gridstruct%cosa
      dx        => gridstruct%dx
      dy        => gridstruct%dy
      dxc       => gridstruct%dxc
      dyc       => gridstruct%dyc
      rdxa      => gridstruct%rdxa
      rdya      => gridstruct%rdya
      rdx       => gridstruct%rdx
      rdy       => gridstruct%rdy

      sw_corner = gridstruct%sw_corner
      se_corner = gridstruct%se_corner
      nw_corner = gridstruct%nw_corner
      ne_corner = gridstruct%ne_corner

      prevent_diss_cooling = flagstruct%prevent_diss_cooling

#ifdef SW_DYNAMICS
      if ( test_case == 1 ) then
        do j=jsd,jed
           do i=is,ie+1
              xfx_adv(i,j) = dt * uc(i,j) / sina_u(i,j)
              if (xfx_adv(i,j) > 0.) then
                  crx_adv(i,j) = xfx_adv(i,j) * rdxa(i-1,j)
              else
                  crx_adv(i,j) = xfx_adv(i,j) * rdxa(i,j)
              endif
              xfx_adv(i,j) = dy(i,j)*xfx_adv(i,j)*sina_u(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=isd,ied
              yfx_adv(i,j) = dt * vc(i,j) / sina_v(i,j)
              if (yfx_adv(i,j) > 0.) then
                 cry_adv(i,j) = yfx_adv(i,j) * rdya(i,j-1)
              else
                 cry_adv(i,j) = yfx_adv(i,j) * rdya(i,j)
              endif
              yfx_adv(i,j) = dx(i,j)*yfx_adv(i,j)*sina_v(i,j)
           enddo
        enddo
      else
#endif
     if ( flagstruct%grid_type < 3 ) then

!!! TO DO: separate versions for nesting and for cubed-sphere
        if (bounded_domain) then
           do j=jsd,jed
              do i=is,ie+1
                 ut(i,j) = ( uc(i,j) - 0.25 * cosa_u(i,j) *     &
                      (vc(i-1,j)+vc(i,j)+vc(i-1,j+1)+vc(i,j+1)))*rsin_u(i,j)
              enddo
           enddo
           do j=js,je+1
              do i=isd,ied
                 vt(i,j) = ( vc(i,j) - 0.25 * cosa_v(i,j) *     &
                      (uc(i,j-1)+uc(i+1,j-1)+uc(i,j)+uc(i+1,j)))*rsin_v(i,j)
              enddo
           enddo
        else
           do j=jsd,jed
              if( j/=0 .and. j/=1 .and. j/=(npy-1) .and. j/=npy) then
                 do i=is-1,ie+2
                    ut(i,j) = ( uc(i,j) - 0.25 * cosa_u(i,j) *     &
                         (vc(i-1,j)+vc(i,j)+vc(i-1,j+1)+vc(i,j+1)))*rsin_u(i,j)
                 enddo
              endif
           enddo
           do j=js-1,je+2

              if( j/=1 .and. j/=npy ) then

                 do i=isd,ied
                    vt(i,j) = ( vc(i,j) - 0.25 * cosa_v(i,j) *     &
                         (uc(i,j-1)+uc(i+1,j-1)+uc(i,j)+uc(i+1,j)))*rsin_v(i,j)
                 enddo
              endif
           enddo
        endif

      if (.not. bounded_domain) then
! West edge:
       if ( is==1 ) then
          do j=jsd,jed
             if ( uc(1,j)*dt > 0. ) then
                ut(1,j) = uc(1,j) / sin_sg(0,j,3)
             else
                ut(1,j) = uc(1,j) / sin_sg(1,j,1)
             endif
          enddo
          do j=max(3,js), min(npy-2,je+1)
             vt(0,j) = vc(0,j) - 0.25*cosa_v(0,j)*   &
                  (ut(0,j-1)+ut(1,j-1)+ut(0,j)+ut(1,j))
             vt(1,j) = vc(1,j) - 0.25*cosa_v(1,j)*   &
                  (ut(1,j-1)+ut(2,j-1)+ut(1,j)+ut(2,j))
          enddo
       endif   ! West face

! East edge:
       if ( (ie+1)==npx ) then
          do j=jsd,jed
             if ( uc(npx,j)*dt > 0. ) then
                ut(npx,j) = uc(npx,j) / sin_sg(npx-1,j,3)
             else
                ut(npx,j) = uc(npx,j) / sin_sg(npx,j,1)
             endif
          enddo

           do j=max(3,js), min(npy-2,je+1)
              vt(npx-1,j) = vc(npx-1,j) - 0.25*cosa_v(npx-1,j)*   &
                           (ut(npx-1,j-1)+ut(npx,j-1)+ut(npx-1,j)+ut(npx,j))
              vt(npx,j) = vc(npx,j) - 0.25*cosa_v(npx,j)*   &
                         (ut(npx,j-1)+ut(npx+1,j-1)+ut(npx,j)+ut(npx+1,j))
           enddo
       endif

! South (Bottom) edge:
       if ( js==1 ) then

           do i=isd,ied
              if ( vc(i,1)*dt > 0. ) then
                   vt(i,1) = vc(i,1) / sin_sg(i,0,4)
              else
                   vt(i,1) = vc(i,1) / sin_sg(i,1,2)
              endif
           enddo

           do i=max(3,is),min(npx-2,ie+1)
              ut(i,0) = uc(i,0) - 0.25*cosa_u(i,0)*   &
                       (vt(i-1,0)+vt(i,0)+vt(i-1,1)+vt(i,1))
              ut(i,1) = uc(i,1) - 0.25*cosa_u(i,1)*   &
                       (vt(i-1,1)+vt(i,1)+vt(i-1,2)+vt(i,2))
           enddo
       endif

! North edge:
       if ( (je+1)==npy ) then
           do i=isd,ied
              if ( vc(i,npy)*dt > 0. ) then
                   vt(i,npy) = vc(i,npy) / sin_sg(i,npy-1,4)
              else
                   vt(i,npy) = vc(i,npy) / sin_sg(i,npy,2)
              endif
           enddo
           do i=max(3,is),min(npx-2,ie+1)
              ut(i,npy-1) = uc(i,npy-1) - 0.25*cosa_u(i,npy-1)*   &
                           (vt(i-1,npy-1)+vt(i,npy-1)+vt(i-1,npy)+vt(i,npy))
              ut(i,npy) = uc(i,npy) - 0.25*cosa_u(i,npy)*   &
                         (vt(i-1,npy)+vt(i,npy)+vt(i-1,npy+1)+vt(i,npy+1))
           enddo
       endif

! The following code solves a 2x2 system to get the interior parallel-to-edge uc,vc values
! near the corners (ex: for the sw corner ut(2,1) and vt(1,2) are solved for simultaneously).
! It then computes the halo uc, vc values so as to be consistent with the computations on
! the facing panel.

       !The system solved is:
       !  ut(2,1) = uc(2,1) - avg(vt)*cosa_u(2,1)
       !  vt(1,2) = vc(1,2) - avg(ut)*cosa_v(1,2)
       ! in which avg(vt) includes vt(1,2) and avg(ut) includes ut(2,1)

        if( sw_corner ) then
            damp = 1. / (1.-0.0625*cosa_u(2,0)*cosa_v(1,0))
            ut(2,0) = (uc(2,0)-0.25*cosa_u(2,0)*(vt(1,1)+vt(2,1)+vt(2,0) +vc(1,0) -   &
                      0.25*cosa_v(1,0)*(ut(1,0)+ut(1,-1)+ut(2,-1))) ) * damp
            damp = 1. / (1.-0.0625*cosa_u(0,1)*cosa_v(0,2))
            vt(0,2) = (vc(0,2)-0.25*cosa_v(0,2)*(ut(1,1)+ut(1,2)+ut(0,2)+uc(0,1) -   &
                      0.25*cosa_u(0,1)*(vt(0,1)+vt(-1,1)+vt(-1,2))) ) * damp

            damp = 1. / (1.-0.0625*cosa_u(2,1)*cosa_v(1,2))
            ut(2,1) = (uc(2,1)-0.25*cosa_u(2,1)*(vt(1,1)+vt(2,1)+vt(2,2)+vc(1,2) -   &
                      0.25*cosa_v(1,2)*(ut(1,1)+ut(1,2)+ut(2,2))) ) * damp

            vt(1,2) = (vc(1,2)-0.25*cosa_v(1,2)*(ut(1,1)+ut(1,2)+ut(2,2)+uc(2,1) -   &
                      0.25*cosa_u(2,1)*(vt(1,1)+vt(2,1)+vt(2,2))) ) * damp
        endif

        if( se_corner ) then
            damp = 1. / (1. - 0.0625*cosa_u(npx-1,0)*cosa_v(npx-1,0))
            ut(npx-1,0) = ( uc(npx-1,0)-0.25*cosa_u(npx-1,0)*(   &
                            vt(npx-1,1)+vt(npx-2,1)+vt(npx-2,0)+vc(npx-1,0) -   &
                      0.25*cosa_v(npx-1,0)*(ut(npx,0)+ut(npx,-1)+ut(npx-1,-1))) ) * damp
            damp = 1. / (1. - 0.0625*cosa_u(npx+1,1)*cosa_v(npx,2))
            vt(npx,  2) = ( vc(npx,2)-0.25*cosa_v(npx,2)*(  &
                            ut(npx,1)+ut(npx,2)+ut(npx+1,2)+uc(npx+1,1) -   &
                      0.25*cosa_u(npx+1,1)*(vt(npx,1)+vt(npx+1,1)+vt(npx+1,2))) ) * damp

            damp = 1. / (1. - 0.0625*cosa_u(npx-1,1)*cosa_v(npx-1,2))
            ut(npx-1,1) = ( uc(npx-1,1)-0.25*cosa_u(npx-1,1)*(  &
                            vt(npx-1,1)+vt(npx-2,1)+vt(npx-2,2)+vc(npx-1,2) -   &
                      0.25*cosa_v(npx-1,2)*(ut(npx,1)+ut(npx,2)+ut(npx-1,2))) ) * damp
            vt(npx-1,2) = ( vc(npx-1,2)-0.25*cosa_v(npx-1,2)*(  &
                            ut(npx,1)+ut(npx,2)+ut(npx-1,2)+uc(npx-1,1) -   &
                      0.25*cosa_u(npx-1,1)*(vt(npx-1,1)+vt(npx-2,1)+vt(npx-2,2))) ) * damp
        endif

        if( ne_corner ) then
            damp = 1. / (1. - 0.0625*cosa_u(npx-1,npy)*cosa_v(npx-1,npy+1))
            ut(npx-1,npy) = ( uc(npx-1,npy)-0.25*cosa_u(npx-1,npy)*(   &
                              vt(npx-1,npy)+vt(npx-2,npy)+vt(npx-2,npy+1)+vc(npx-1,npy+1) -   &
                0.25*cosa_v(npx-1,npy+1)*(ut(npx,npy)+ut(npx,npy+1)+ut(npx-1,npy+1))) ) * damp
            damp = 1. / (1. - 0.0625*cosa_u(npx+1,npy-1)*cosa_v(npx,npy-1))
            vt(npx,  npy-1) = ( vc(npx,npy-1)-0.25*cosa_v(npx,npy-1)*(   &
                                ut(npx,npy-1)+ut(npx,npy-2)+ut(npx+1,npy-2)+uc(npx+1,npy-1) -   &
                0.25*cosa_u(npx+1,npy-1)*(vt(npx,npy)+vt(npx+1,npy)+vt(npx+1,npy-1))) ) * damp

            damp = 1. / (1. - 0.0625*cosa_u(npx-1,npy-1)*cosa_v(npx-1,npy-1))
            ut(npx-1,npy-1) = ( uc(npx-1,npy-1)-0.25*cosa_u(npx-1,npy-1)*(  &
                                vt(npx-1,npy)+vt(npx-2,npy)+vt(npx-2,npy-1)+vc(npx-1,npy-1) -  &
                0.25*cosa_v(npx-1,npy-1)*(ut(npx,npy-1)+ut(npx,npy-2)+ut(npx-1,npy-2))) ) * damp
            vt(npx-1,npy-1) = ( vc(npx-1,npy-1)-0.25*cosa_v(npx-1,npy-1)*(  &
                                ut(npx,npy-1)+ut(npx,npy-2)+ut(npx-1,npy-2)+uc(npx-1,npy-1) -  &
                0.25*cosa_u(npx-1,npy-1)*(vt(npx-1,npy)+vt(npx-2,npy)+vt(npx-2,npy-1))) ) * damp
        endif

        if( nw_corner ) then
            damp = 1. / (1. - 0.0625*cosa_u(2,npy)*cosa_v(1,npy+1))
            ut(2,npy) = ( uc(2,npy)-0.25*cosa_u(2,npy)*(   &
                          vt(1,npy)+vt(2,npy)+vt(2,npy+1)+vc(1,npy+1) -   &
                      0.25*cosa_v(1,npy+1)*(ut(1,npy)+ut(1,npy+1)+ut(2,npy+1))) ) * damp
            damp = 1. / (1. - 0.0625*cosa_u(0,npy-1)*cosa_v(0,npy-1))
            vt(0,npy-1) = ( vc(0,npy-1)-0.25*cosa_v(0,npy-1)*(  &
                            ut(1,npy-1)+ut(1,npy-2)+ut(0,npy-2)+uc(0,npy-1) -   &
                      0.25*cosa_u(0,npy-1)*(vt(0,npy)+vt(-1,npy)+vt(-1,npy-1))) ) * damp

            damp = 1. / (1. - 0.0625*cosa_u(2,npy-1)*cosa_v(1,npy-1))
            ut(2,npy-1) = ( uc(2,npy-1)-0.25*cosa_u(2,npy-1)*(  &
                            vt(1,npy)+vt(2,npy)+vt(2,npy-1)+vc(1,npy-1) -   &
                      0.25*cosa_v(1,npy-1)*(ut(1,npy-1)+ut(1,npy-2)+ut(2,npy-2))) ) * damp

            vt(1,npy-1) = ( vc(1,npy-1)-0.25*cosa_v(1,npy-1)*(  &
                            ut(1,npy-1)+ut(1,npy-2)+ut(2,npy-2)+uc(2,npy-1) -   &
                      0.25*cosa_u(2,npy-1)*(vt(1,npy)+vt(2,npy)+vt(2,npy-1))) ) * damp
        endif

       end if !.not. bounded_domain

     else
! flagstruct%grid_type >= 3
        do j=jsd,jed
           do i=is,ie+1
              ut(i,j) =  uc(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=isd,ied
              vt(i,j) = vc(i,j)
           enddo
        enddo
     endif      ! end grid_type choices

        do j=jsd,jed
           do i=is,ie+1
              xfx_adv(i,j) = dt*ut(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=isd,ied
              yfx_adv(i,j) = dt*vt(i,j)
           enddo
        enddo

! Explanation of the following code:
!    xfx_adv = dt*ut*dy
!    crx_adv = dt*ut/dx

        do j=jsd,jed
!DEC$ VECTOR ALWAYS
           do i=is,ie+1
              if ( xfx_adv(i,j) > 0. ) then
                   crx_adv(i,j) = xfx_adv(i,j) * rdxa(i-1,j)
                   xfx_adv(i,j) = dy(i,j)*xfx_adv(i,j)*sin_sg(i-1,j,3)
              else
                   crx_adv(i,j) = xfx_adv(i,j) * rdxa(i,j)
                   xfx_adv(i,j) = dy(i,j)*xfx_adv(i,j)*sin_sg(i,j,1)
              end if
           enddo
        enddo
        do j=js,je+1
!DEC$ VECTOR ALWAYS
           do i=isd,ied
              if ( yfx_adv(i,j) > 0. ) then
                   cry_adv(i,j) = yfx_adv(i,j) * rdya(i,j-1)
                   yfx_adv(i,j) = dx(i,j)*yfx_adv(i,j)*sin_sg(i,j-1,4)
              else
                   cry_adv(i,j) = yfx_adv(i,j) * rdya(i,j)
                   yfx_adv(i,j) = dx(i,j)*yfx_adv(i,j)*sin_sg(i,j,2)
              endif
           enddo
        enddo

#ifdef SW_DYNAMICS
      endif
#endif

      do j=jsd,jed
         do i=is,ie
            ra_x(i,j) = area(i,j) + xfx_adv(i,j) - xfx_adv(i+1,j)
         enddo
      enddo
      do j=js,je
         do i=isd,ied
            ra_y(i,j) = area(i,j) + yfx_adv(i,j) - yfx_adv(i,j+1)
         enddo
      enddo

      call fv_tp_2d(delp, crx_adv, cry_adv, npx, npy, hord_dp, fx, fy,  &
                    xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, nord=nord_v, damp_c=damp_v)

! <<< Save the mass fluxes to the "Flux Capacitor" for tracer transport >>>
        do j=jsd,jed
            do i=is,ie+1
              cx(i,j) = cx(i,j) + crx_adv(i,j)
           enddo
        enddo
        do j=js,je
           do i=is,ie+1
              xflux(i,j) = xflux(i,j) + fx(i,j)
           enddo
        enddo
        do j=js,je+1
           do i=isd,ied
              cy(i,j) = cy(i,j) + cry_adv(i,j)
           enddo
           do i=is,ie
              yflux(i,j) = yflux(i,j) + fy(i,j)
           enddo
        enddo

#ifndef SW_DYNAMICS
        do j=js,je
           do i=is,ie
              heat_source(i,j) = 0.
              diss_est(i,j) = 0.
           enddo
        enddo

        if ( .not. hydrostatic ) then
            if ( damp_w>1.E-5 ) then
               dd8 = kgb*abs(dt)
               damp4 = (damp_w*gridstruct%da_min_c)**(nord_w+1)
               call del6_vt_flux(nord_w, npx, npy, damp4, w, wk, fx2, fy2, gridstruct, bd)
               if (prevent_diss_cooling) then
                  do j=js,je
                  do i=is,ie
                    dw(i,j) = (fx2(i,j)-fx2(i+1,j)+fy2(i,j)-fy2(i,j+1))*rarea(i,j)
                    ! 0.5 * [ (w+dw)**2 - w**2 ] = w*dw + 0.5*dw*dw
                    !limiter to prevent "dissipative cooling"
                    !physically `tmp` is negative.
                    tmp = dw(i,j)*(w(i,j)+0.5*dw(i,j))
                    heat_source(i,j) = dd8 - min(0.,tmp)
                    if ( flagstruct%do_diss_est ) then
                       diss_est(i,j) = dd8 - tmp
                    endif
                 enddo
                 enddo
              else
                 do j=js,je
                 do i=is,ie
                   dw(i,j) = (fx2(i,j)-fx2(i+1,j)+fy2(i,j)-fy2(i,j+1))*rarea(i,j)
                   ! 0.5 * [ (w+dw)**2 - w**2 ] = w*dw + 0.5*dw*dw
                   heat_source(i,j) = dd8 - dw(i,j)*(w(i,j)+0.5*dw(i,j))
                   tmp = dw(i,j)*(w(i,j)+0.5*dw(i,j))
                   if ( flagstruct%do_diss_est ) then
                      diss_est(i,j) = heat_source(i,j)
                   endif
                enddo
                enddo
              endif
           endif
           call fv_tp_2d(w, crx_adv,cry_adv, npx, npy, hord_vt, gx, gy, xfx_adv, yfx_adv, &
                          gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, mfx=fx, mfy=fy)
           do j=js,je
              do i=is,ie
                 w(i,j) = delp(i,j)*w(i,j) + (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j)
              enddo
           enddo
        endif

        if (use_cond) then
           call fv_tp_2d(q_con, crx_adv,cry_adv, npx, npy, hord_dp, gx, gy,  &
                xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, mfx=fx, mfy=fy, mass=delp, nord=nord_t, damp_c=damp_t)
           do j=js,je
              do i=is,ie
                 q_con(i,j) = delp(i,j)*q_con(i,j) + (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j)
              enddo
           enddo
        endif

!    if ( inline_q .and. zvir>0.01 ) then
!       do j=jsd,jed
!          do i=isd,ied
!             pt(i,j) = pt(i,j)/(1.+zvir*q(i,j,k,sphum))
!          enddo
!       enddo
!    endif
#if defined(GFS_PHYS) || defined(DCMIP)
        call fv_tp_2d(pt, crx_adv,cry_adv, npx, npy, hord_tm, gx, gy,  &
                      xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, &
                      mfx=fx, mfy=fy, mass=delp, nord=nord_v, damp_c=damp_v) !SHiELD
#else
        call fv_tp_2d(pt, crx_adv,cry_adv, npx, npy, hord_tm, gx, gy,  &
                      xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, &
                      mfx=fx, mfy=fy, mass=delp, nord=nord_t, damp_c=damp_t) !AM4
#endif
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
           call fv_tp_2d(q(isd,jsd,k,iq), crx_adv,cry_adv, npx, npy, hord_tr, gx, gy,  &
                         xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, &
                         mfx=fx, mfy=fy, mass=delp, nord=nord_t, damp_c=damp_t)
           do j=js,je
              do i=is,ie
                 q(i,j,k,iq) = (q(i,j,k,iq)*wk(i,j) +               &
                         (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j))/delp(i,j)
              enddo
           enddo
        enddo
!     if ( zvir>0.01 ) then
!       do j=js,je
!          do i=is,ie
!             pt(i,j) = pt(i,j)*(1.+zvir*q(i,j,k,sphum))
!          enddo
!       enddo
!     endif

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

#ifdef SW_DYNAMICS
      if (test_case > 1) then
#endif

!----------------------
! Kinetic Energy Fluxes
!----------------------
! Compute B grid contra-variant components for KE:

      dt5 = 0.5 *dt
      dt4 = 0.25*dt

      if (bounded_domain) then
         is2 = is;        ie1 = ie+1
         js2 = js;        je1 = je+1
      else
         is2 = max(2,is); ie1 = min(npx-1,ie+1)
         js2 = max(2,js); je1 = min(npy-1,je+1)
      end if

      if (flagstruct%grid_type < 3) then

         if (bounded_domain) then
            do j=js2,je1
               do i=is2,ie1
                  vb(i,j) = dt5*(vc(i-1,j)+vc(i,j)-(uc(i,j-1)+uc(i,j))*cosa(i,j))*rsina(i,j)
               enddo
            enddo
         else
            if ( js==1 ) then
               do i=is,ie+1
                  vb(i,1) = dt5*(vt(i-1,1)+vt(i,1))       ! corner values are incorrect
               enddo
            endif

            do j=js2,je1
               do i=is2,ie1
                  vb(i,j) = dt5*(vc(i-1,j)+vc(i,j)-(uc(i,j-1)+uc(i,j))*cosa(i,j))*rsina(i,j)
               enddo

               if ( is==1 ) then
                  ! 2-pt extrapolation from both sides:
                  vb(1,j) = dt4*(-vt(-1,j) + 3.*(vt(0,j)+vt(1,j)) - vt(2,j))
               endif
               if ( (ie+1)==npx ) then
                  ! 2-pt extrapolation from both sides:
                  vb(npx,j) = dt4*(-vt(npx-2,j) + 3.*(vt(npx-1,j)+vt(npx,j)) - vt(npx+1,j))
               endif
            enddo

            if ( (je+1)==npy ) then
               do i=is,ie+1
                  vb(i,npy) = dt5*(vt(i-1,npy)+vt(i,npy)) ! corner values are incorrect
               enddo
            endif
         endif

      else
         do j=js,je+1
            do i=is,ie+1
               vb(i,j) = dt5*(vc(i-1,j)+vc(i,j))
            enddo
         enddo
      endif

      call ytp_v(is,ie,js,je,isd,ied,jsd,jed, vb, u, v, ub, hord_mt, gridstruct%dy, gridstruct%rdy, &
                 npx, npy, flagstruct%grid_type, bounded_domain, flagstruct%lim_fac)

      do j=js,je+1
         do i=is,ie+1
            ke(i,j) = vb(i,j)*ub(i,j)
         enddo
      enddo

      if (flagstruct%grid_type < 3) then

         if (bounded_domain) then

            do j=js,je+1

                  do i=is2,ie1
                     ub(i,j) = dt5*(uc(i,j-1)+uc(i,j)-(vc(i-1,j)+vc(i,j))*cosa(i,j))*rsina(i,j)
                  enddo

            enddo


         else
            if ( is==1 ) then
               do j=js,je+1
                  ub(1,j) = dt5*(ut(1,j-1)+ut(1,j))       ! corner values are incorrect
               enddo
            endif

            do j=js,je+1
               if ( (j==1 .or. j==npy) ) then
                  do i=is2,ie1
                     ! 2-pt extrapolation from both sides:
                     ub(i,j) = dt4*(-ut(i,j-2) + 3.*(ut(i,j-1)+ut(i,j)) - ut(i,j+1))
                  enddo
               else
                  do i=is2,ie1
                     ub(i,j) = dt5*(uc(i,j-1)+uc(i,j)-(vc(i-1,j)+vc(i,j))*cosa(i,j))*rsina(i,j)
                  enddo
               endif
            enddo

            if ( (ie+1)==npx ) then
               do j=js,je+1
                  ub(npx,j) = dt5*(ut(npx,j-1)+ut(npx,j))       ! corner values are incorrect
               enddo
            endif
         endif

      else
         do j=js,je+1
            do i=is,ie+1
               ub(i,j) = dt5*(uc(i,j-1)+uc(i,j))
            enddo
         enddo
      endif

      call xtp_u(is,ie,js,je, isd,ied,jsd,jed, ub, u, v, vb, hord_mt, gridstruct%dx, gridstruct%rdx, &
                 npx, npy, flagstruct%grid_type, bounded_domain, flagstruct%lim_fac)

      do j=js,je+1
         do i=is,ie+1
            ke(i,j) = 0.5*(ke(i,j) + ub(i,j)*vb(i,j))
         enddo
      enddo

!-----------------------------------------
! Fix KE at the 4 corners of the face:
!-----------------------------------------
    if (.not. bounded_domain) then
      dt6 = dt / 6.
      if ( sw_corner ) then
           ke(1,1) = dt6*( (ut(1,1) + ut(1,0)) * u(1,1) +  &
                           (vt(1,1) + vt(0,1)) * v(1,1) +  &
                           (ut(1,1) + vt(1,1)) * u(0,1) )
      endif
      if ( se_corner ) then
           i = npx
           ke(i,1) = dt6*( (ut(i,1) + ut(i,  0)) * u(i-1,1) + &
                           (vt(i,1) + vt(i-1,1)) * v(i,  1) + &
                           (ut(i,1) - vt(i-1,1)) * u(i,  1) )
      endif
      if ( ne_corner ) then
           i = npx;      j = npy
           ke(i,j) = dt6*( (ut(i,j  ) + ut(i,j-1)) * u(i-1,j) +  &
                           (vt(i,j  ) + vt(i-1,j)) * v(i,j-1) +  &
                           (ut(i,j-1) + vt(i-1,j)) * u(i,j  )  )
      endif
      if ( nw_corner ) then
           j = npy
           ke(1,j) = dt6*( (ut(1,  j) + ut(1,j-1)) * u(1,j  ) +  &
                           (vt(1,  j) + vt(0,  j)) * v(1,j-1) +  &
                           (ut(1,j-1) - vt(1,  j)) * u(0,j  )  )
      endif
    end if

! Compute vorticity:
       do j=jsd,jed+1
          do i=isd,ied
             vt(i,j) = u(i,j)*dx(i,j)
          enddo
       enddo
       do j=jsd,jed
          do i=isd,ied+1
             ut(i,j) = v(i,j)*dy(i,j)
          enddo
       enddo

! wk is "volume-mean" relative vorticity
       do j=jsd,jed
          do i=isd,ied
             wk(i,j) = rarea(i,j)*(vt(i,j)-vt(i,j+1)-ut(i,j)+ut(i+1,j))
          enddo
       enddo

     if ( .not. hydrostatic ) then
        if( flagstruct%do_f3d ) then
#ifdef ROT3
            dt2 = 2.*dt
            do j=js,je
               do i=is,ie
                  w(i,j) = w(i,j)/delp(i,j) + dt2*gridstruct%w00(i,j) *  &
                         ( gridstruct%a11(i,j)*(u(i,j)+u(i,j+1)) +       &
                           gridstruct%a12(i,j)*(v(i,j)+v(i+1,j)) )
               enddo
            enddo
#endif
        else
            do j=js,je
               do i=is,ie
                  w(i,j) = w(i,j)/delp(i,j)
               enddo
            enddo
        endif
        if ( damp_w>1.E-5 ) then
          do j=js,je
             do i=is,ie
                w(i,j) = w(i,j) + dw(i,j)
             enddo
          enddo
        endif

     endif
     if (use_cond) then
        do j=js,je
        do i=is,ie
           q_con(i,j) = q_con(i,j)/delp(i,j)
        enddo
        enddo
     endif

!-----------------------------
! Compute divergence damping
!-----------------------------
!  damp = dddmp * da_min_c

   if ( nord==0 ) then
!         area ~ dxb*dyb*sin(alpha)

      if (bounded_domain) then

         do j=js,je+1
            do i=is-1,ie+1
               ptc(i,j) = (u(i,j)-0.5*(va(i,j-1)+va(i,j))*cosa_v(i,j))   &
                    *dyc(i,j)*sina_v(i,j)
            enddo
         enddo

         do j=js-1,je+1
            do i=is2,ie1
               vort(i,j) = (v(i,j) - 0.5*(ua(i-1,j)+ua(i,j))*cosa_u(i,j))  &
                    *dxc(i,j)*sina_u(i,j)
            enddo
         enddo

      else
         do j=js,je+1

            if ( (j==1 .or. j==npy)  ) then
               do i=is-1,ie+1
                  if (vc(i,j) > 0) then
                     ptc(i,j) = u(i,j)*dyc(i,j)*sin_sg(i,j-1,4)
                  else
                     ptc(i,j) = u(i,j)*dyc(i,j)*sin_sg(i,j,2)
                  end if
               enddo
            else
               do i=is-1,ie+1
                  ptc(i,j) = (u(i,j)-0.5*(va(i,j-1)+va(i,j))*cosa_v(i,j))   &
                       *dyc(i,j)*sina_v(i,j)
               enddo
            endif
         enddo

         do j=js-1,je+1
            do i=is2,ie1
               vort(i,j) = (v(i,j) - 0.5*(ua(i-1,j)+ua(i,j))*cosa_u(i,j))  &
                    *dxc(i,j)*sina_u(i,j)
            enddo
            if ( is ==  1 ) then
               if (uc(1,j) > 0) then
                  vort(1,  j) = v(1,  j)*dxc(1,  j)*sin_sg(0,j,3)
               else
                  vort(1,  j) = v(1,  j)*dxc(1,  j)*sin_sg(1,j,1)
               end if
            end if
            if ( (ie+1)==npx ) then
               if (uc(npx,j) > 0) then
                  vort(npx,j) = v(npx,j)*dxc(npx,j)* &
                       sin_sg(npx-1,j,3)
               else
                  vort(npx,j) = v(npx,j)*dxc(npx,j)* &
                       sin_sg(npx,j,1)
               end if
            end if
         enddo
      endif

      do j=js,je+1
         do i=is,ie+1
            delpc(i,j) = vort(i,j-1) - vort(i,j) + ptc(i-1,j) - ptc(i,j)
         enddo
      enddo

! Remove the extra term at the corners:
      if (sw_corner) delpc(1,    1) = delpc(1,    1) - vort(1,    0)
      if (se_corner) delpc(npx,  1) = delpc(npx,  1) - vort(npx,  0)
      if (ne_corner) delpc(npx,npy) = delpc(npx,npy) + vort(npx,npy)
      if (nw_corner) delpc(1,  npy) = delpc(1,  npy) + vort(1,  npy)

      do j=js,je+1
         do i=is,ie+1
            delpc(i,j) = gridstruct%rarea_c(i,j)*delpc(i,j)
                damp = gridstruct%da_min_c*max(d2_bg, min(0.20, dddmp*abs(delpc(i,j)*dt)))
                vort(i,j) = damp*delpc(i,j)
                ke(i,j) = ke(i,j) + vort(i,j)
         enddo
      enddo
   else
!--------------------------
! Higher order divg damping
!--------------------------
     do j=js,je+1
        do i=is,ie+1
! Save divergence for external mode filter
           delpc(i,j) = divg_d(i,j)
        enddo
     enddo

     n2 = nord + 1    ! N > 1
     do n=1,nord
        nt = nord-n

        fill_c = (nt/=0) .and. (flagstruct%grid_type<3) .and.               &
                 ( sw_corner .or. se_corner .or. ne_corner .or. nw_corner ) &
                  .and. .not. bounded_domain

        if ( fill_c ) call fill_corners(divg_d, npx, npy, FILL=XDir, BGRID=.true.)
        do j=js-nt,je+1+nt
           do i=is-1-nt,ie+1+nt
              vc(i,j) = (divg_d(i+1,j)-divg_d(i,j))*divg_u(i,j)
           enddo
        enddo

        if ( fill_c ) call fill_corners(divg_d, npx, npy, FILL=YDir, BGRID=.true.)
        do j=js-1-nt,je+1+nt
           do i=is-nt,ie+1+nt
              uc(i,j) = (divg_d(i,j+1)-divg_d(i,j))*divg_v(i,j)
           enddo
        enddo

        if ( fill_c ) call fill_corners(vc, uc, npx, npy, VECTOR=.true., DGRID=.true.)
        do j=js-nt,je+1+nt
           do i=is-nt,ie+1+nt
              divg_d(i,j) = uc(i,j-1) - uc(i,j) + vc(i-1,j) - vc(i,j)
           enddo
        enddo

! Remove the extra term at the corners:
        if (sw_corner) divg_d(1,    1) = divg_d(1,    1) - uc(1,    0)
        if (se_corner) divg_d(npx,  1) = divg_d(npx,  1) - uc(npx,  0)
        if (ne_corner) divg_d(npx,npy) = divg_d(npx,npy) + uc(npx,npy)
        if (nw_corner) divg_d(1,  npy) = divg_d(1,  npy) + uc(1,  npy)

     if ( .not. gridstruct%stretched_grid ) then
        do j=js-nt,je+1+nt
           do i=is-nt,ie+1+nt
              divg_d(i,j) = divg_d(i,j)*gridstruct%rarea_c(i,j)
           enddo
        enddo
     endif

     enddo ! n-loop

     if ( dddmp<1.E-5) then
          vort(:,:) = 0.
     else
      if ( flagstruct%grid_type < 3 ) then
! Interpolate relative vort to cell corners
          call a2b_ord4(wk, vort, gridstruct, npx, npy, is, ie, js, je, ng, .false.)
          do j=js,je+1
             do i=is,ie+1
! The following is an approxi form of Smagorinsky diffusion
                vort(i,j) = abs(dt)*sqrt(delpc(i,j)**2 + vort(i,j)**2)
             enddo
          enddo
      else  ! Correct form: works only for doubly preiodic domain
          call smag_corner(abs(dt), u, v, ua, va, vort, bd, npx, npy, gridstruct, ng)
      endif
     endif

     if (gridstruct%stretched_grid ) then
! Stretched grid with variable damping ~ area
         dd8 = gridstruct%da_min * d4_bg**n2
     else
         dd8 = ( gridstruct%da_min_c*d4_bg )**n2
     endif

     do j=js,je+1
        do i=is,ie+1
           damp2 =  gridstruct%da_min_c*max(d2_bg, min(0.20, dddmp*vort(i,j)))  ! del-2
           vort(i,j) = damp2*delpc(i,j) + dd8*divg_d(i,j)
             ke(i,j) = ke(i,j) + vort(i,j)
        enddo
     enddo

   endif

   if ( d_con > 1.e-5 .or. flagstruct%do_diss_est) then
      do j=js,je+1
         do i=is,ie
            ub(i,j) = vort(i,j) - vort(i+1,j)
         enddo
      enddo
      do j=js,je
         do i=is,ie+1
            vb(i,j) = vort(i,j) - vort(i,j+1)
         enddo
      enddo
   endif

! Vorticity transport
   if ( hydrostatic ) then
    do j=jsd,jed
       do i=isd,ied
          vort(i,j) = wk(i,j) + f0(i,j)
       enddo
    enddo
   else
    if ( flagstruct%do_f3d ) then
       do j=jsd,jed
       do i=isd,ied
          vort(i,j) = wk(i,j) + f0(i,j)*z_rat(i,j)
       enddo
       enddo
    else
       do j=jsd,jed
       do i=isd,ied
          vort(i,j) = wk(i,j) + f0(i,j)
       enddo
       enddo
    endif
   endif

    call fv_tp_2d(vort, crx_adv, cry_adv, npx, npy, hord_vt, fx, fy, &
                  xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac)
    do j=js,je+1
       do i=is,ie
          u(i,j) = vt(i,j) + ke(i,j) - ke(i+1,j) + fy(i,j)
       enddo
    enddo
    do j=js,je
       do i=is,ie+1
          v(i,j) = ut(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)
       enddo
    enddo

!--------------------------------------------------------
! damping applied to relative vorticity (wk):
   if ( damp_v>1.E-5 ) then
        damp4 = (damp_v*gridstruct%da_min_c)**(nord_v+1)
        call del6_vt_flux(nord_v, npx, npy, damp4, wk, vort, ut, vt, gridstruct, bd)
   elseif (flagstruct%do_diss_est) then
        ut=0.
        vt=0.
   endif

   !estimate dissipation for dissipative heating
   ! or dissipation estimate diagnostic
   if ( d_con > 1.e-5 .or. flagstruct%do_diss_est ) then
      do j=js,je+1
         do i=is,ie
            ub(i,j) = (ub(i,j) + vt(i,j))*rdx(i,j)
            fy(i,j) =  u(i,j)*rdx(i,j)
            gy(i,j) = fy(i,j)*ub(i,j)
         enddo
      enddo
      do j=js,je
         do i=is,ie+1
            vb(i,j) = (vb(i,j) - ut(i,j))*rdy(i,j)
            fx(i,j) =  v(i,j)*rdy(i,j)
            gx(i,j) = fx(i,j)*vb(i,j)
         enddo
      enddo
!----------------------------------
! Heating due to damping:
!----------------------------------
      damp = 0.25*d_con
      if (prevent_diss_cooling) then
         do j=js,je
         do i=is,ie
            u2 = fy(i,j) + fy(i,j+1)
           du2 = ub(i,j) + ub(i,j+1)
            v2 = fx(i,j) + fx(i+1,j)
           dv2 = vb(i,j) + vb(i+1,j)
! Total energy conserving:
! Convert lost KE due to divergence damping to "heat"
           tmp = rsin2(i,j)*((ub(i,j)**2 + ub(i,j+1)**2 + vb(i,j)**2 + vb(i+1,j)**2)  &
                              + 2.*(gy(i,j)+gy(i,j+1)+gx(i,j)+gx(i+1,j))   &
                              - cosa_s(i,j)*(u2*dv2 + v2*du2 + du2*dv2))
           if (d_con > 1.e-5) then
              !limiter to prevent dissipative cooling
              ! again this quantity should physically be negative
              heat_source(i,j) = delp(i,j)*(heat_source(i,j) - damp*min(0.,tmp) )
           endif
           if (flagstruct%do_diss_est) then
             diss_est(i,j) = diss_est(i,j)-tmp
           endif
         enddo
         enddo
      else
         do j=js,je
         do i=is,ie
            u2 = fy(i,j) + fy(i,j+1)
           du2 = ub(i,j) + ub(i,j+1)
            v2 = fx(i,j) + fx(i+1,j)
           dv2 = vb(i,j) + vb(i+1,j)
! Total energy conserving:
! Convert lost KE due to divergence damping to "heat"
         heat_source(i,j) = delp(i,j)*(heat_source(i,j) - damp*rsin2(i,j)*( &
                  (ub(i,j)**2 + ub(i,j+1)**2 + vb(i,j)**2 + vb(i+1,j)**2)  &
                              + 2.*(gy(i,j)+gy(i,j+1)+gx(i,j)+gx(i+1,j))   &
                              - cosa_s(i,j)*(u2*dv2 + v2*du2 + du2*dv2)) )
           if (flagstruct%do_diss_est) then
             diss_est(i,j) = diss_est(i,j)-rsin2(i,j)*( &
                  (ub(i,j)**2 + ub(i,j+1)**2 + vb(i,j)**2 + vb(i+1,j)**2)  &
                                + 2.*(gy(i,j)+gy(i,j+1)+gx(i,j)+gx(i+1,j))   &
                               - cosa_s(i,j)*(u2*dv2 + v2*du2 + du2*dv2))
          endif
         enddo
         enddo
      endif !prevent_diss_cooling
   endif !  d_con > 1.e-5 .or. flagstruct%do_diss_est 

! Add diffusive fluxes to the momentum equation:
   if ( damp_v>1.E-5 ) then
      do j=js,je+1
         do i=is,ie
            u(i,j) = u(i,j) + vt(i,j)
         enddo
      enddo
      do j=js,je
         do i=is,ie+1
            v(i,j) = v(i,j) - ut(i,j)
         enddo
      enddo
   endif

#ifdef SW_DYNAMICS
      endif ! test_case
#endif

 end subroutine d_sw

 subroutine del6_vt_flux(nord, npx, npy, damp, q, d2, fx2, fy2, gridstruct, bd, damp_Km)
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
   real, intent(inout):: q(bd%isd:bd%ied, bd%jsd:bd%jed)  ! rel. vorticity ghosted on input
   type(fv_grid_type), intent(IN), target :: gridstruct
   real, OPTIONAL, intent(in) :: damp_Km(bd%isd:bd%ied,bd%jsd:bd%jed) ! variable diffusion coeff for scalars
                                                                      ! First try adapts cell-centered eddy diffusivities
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

   if( nord>0 .and. .not. bounded_domain) call copy_corners(d2, npx, npy, 1, bounded_domain, bd, gridstruct%sw_corner,    &
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

   if( nord>0 .and. .not. bounded_domain) call copy_corners(d2, npx, npy, 2, bounded_domain, bd, gridstruct%sw_corner,   &
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

      if (.not. bounded_domain) call copy_corners(d2, npx, npy, 1, bounded_domain, bd, gridstruct%sw_corner,    &
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

      if (.not. bounded_domain) call copy_corners(d2, npx, npy, 2, bounded_domain, bd, &
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

   if (present(damp_Km)) then !Coefficient multiplied in earlier
      do j=js,je
         do i=is,ie+1
            fx2(i,j) = fx2(i,j)*0.5*damp_km(i,j)
         enddo
      enddo
      do j=js,je+1
         do i=is,ie
            fy2(i,j) = fy2(i,j)*0.5*damp_km(i,j)
         enddo
      enddo

   endif

 end subroutine del6_vt_flux

 subroutine smag_corner(dt, u, v, ua, va, smag_c, bd, npx, npy, gridstruct, ng)
! Compute the Tension_Shear strain at cell corners for Smagorinsky diffusion
!!!  work only if (grid_type==4)
 type(fv_grid_bounds_type), intent(IN) :: bd
 real, intent(in):: dt
 integer, intent(IN) :: npx, npy, ng
 real, intent(in),  dimension(bd%isd:bd%ied,  bd%jsd:bd%jed+1):: u
 real, intent(in),  dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v
 real, intent(in),  dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: ua, va
 real, intent(out), dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: smag_c
 type(fv_grid_type), intent(IN), target :: gridstruct
! local
 real:: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
 real:: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
 real:: wk(bd%isd:bd%ied,bd%jsd:bd%jed) !  work array
 real:: sh(bd%isd:bd%ied,bd%jsd:bd%jed)
 integer i,j
 integer is2, ie1

 real, pointer, dimension(:,:) :: dxc, dyc, dx, dy, rarea, rarea_c

 integer :: is,  ie,  js,  je
 integer :: isd, ied, jsd, jed

 is  = bd%is
 ie  = bd%ie
 js  = bd%js
 je  = bd%je

 isd  = bd%isd
 ied  = bd%ied
 jsd  = bd%jsd
 jed  = bd%jed

 dxc => gridstruct%dxc
 dyc => gridstruct%dyc
 dx  => gridstruct%dx
 dy  => gridstruct%dy
 rarea   => gridstruct%rarea
 rarea_c => gridstruct%rarea_c

  is2 = max(2,is); ie1 = min(npx-1,ie+1)

! Smag = sqrt [ T**2 + S**2 ]:  unit = 1/s
! where T = du/dx - dv/dy;   S = du/dy + dv/dx
! Compute tension strain at corners:
       do j=js,je+1
          do i=is-1,ie+1
             ut(i,j) = u(i,j)*dyc(i,j)
          enddo
       enddo
       do j=js-1,je+1
          do i=is,ie+1
             vt(i,j) = v(i,j)*dxc(i,j)
          enddo
       enddo
       do j=js,je+1
          do i=is,ie+1
             smag_c(i,j) = rarea_c(i,j)*(vt(i,j-1)-vt(i,j)-ut(i-1,j)+ut(i,j))
          enddo
       enddo
! Fix the corners?? if grid_type /= 4

! Compute shear strain:
       do j=jsd,jed+1
          do i=isd,ied
             vt(i,j) = u(i,j)*dx(i,j)
          enddo
       enddo
       do j=jsd,jed
          do i=isd,ied+1
             ut(i,j) = v(i,j)*dy(i,j)
          enddo
       enddo

       do j=jsd,jed
          do i=isd,ied
             wk(i,j) = rarea(i,j)*(vt(i,j)-vt(i,j+1)+ut(i,j)-ut(i+1,j))
          enddo
       enddo
       call a2b_ord4(wk, sh, gridstruct, npx, npy, is, ie, js, je, ng, .false.)
       do j=js,je+1
          do i=is,ie+1
             smag_c(i,j) = dt*sqrt( sh(i,j)**2 + smag_c(i,j)**2 )
          enddo
       enddo

 end subroutine smag_corner

 subroutine xtp_u(is,ie,js,je,isd,ied,jsd,jed,c, u, v, flux, iord, dx, rdx, npx, npy, grid_type, bounded_domain, lim_fac)

 integer, intent(in):: is,ie,js,je, isd,ied,jsd,jed
 real, INTENT(IN)::   u(isd:ied,jsd:jed+1)
 real, INTENT(IN)::   v(isd:ied+1,jsd:jed)
 real, INTENT(IN)::   c(is:ie+1,js:je+1)
 real, INTENT(out):: flux(is:ie+1,js:je+1)
 real, INTENT(IN) ::   dx(isd:ied,  jsd:jed+1)
 real, INTENT(IN) ::  rdx(isd:ied,  jsd:jed+1)
 integer, INTENT(IN) :: iord, npx, npy, grid_type
 logical, INTENT(IN) :: bounded_domain
 real, INTENT(IN) ::  lim_fac
! Local
 real, dimension(is-1:ie+1):: bl, br, b0
 logical, dimension(is-1:ie+1):: smt5, smt6
 logical, dimension(is:ie+1):: hi5, hi6
 real:: fx0(is:ie+1)
 real al(is-1:ie+2), dm(is-2:ie+2)
 real dq(is-3:ie+2)
 real dl, dr, xt, pmp, lac, cfl
 real pmp_1, lac_1, pmp_2, lac_2
 real x0, x1, x0L, x0R
 integer i, j
 integer is3, ie3
 integer is2, ie2

 if ( bounded_domain .or. grid_type>3 ) then
    is3 = is-1        ; ie3 = ie+1
 else
    is3 = max(3,is-1) ; ie3 = min(npx-3,ie+1)
 end if


 if ( iord < 8 ) then
! Diffusivity: ord2 < ord5 < ord3 < ord4 < ord6

     do j=js,je+1

        do i=is3,ie3+1
           al(i) = p1*(u(i-1,j)+u(i,j)) + p2*(u(i-2,j)+u(i+1,j))
        enddo
        do i=is3,ie3
           bl(i) = al(i  ) - u(i,j)
           br(i) = al(i+1) - u(i,j)
        enddo

      if ( (.not.bounded_domain) .and. grid_type < 3) then
        if ( is==1 ) then
             xt = c3*u(1,j) + c2*u(2,j) + c1*u(3,j)
             br(1) = xt - u(1,j)
             bl(2) = xt - u(2,j)
             br(2) = al(3) - u(2,j)
             if( j==1 .or. j==npy ) then
                 bl(0) = 0.   ! out
                 br(0) = 0.   ! edge
                 bl(1) = 0.   ! edge
                 br(1) = 0.   ! in
             else
                 bl(0) = c1*u(-2,j) + c2*u(-1,j) + c3*u(0,j) - u(0,j)
             xt = 0.5*( ((2.*dx(0,j)+dx(-1,j))*(u(0,j))-dx(0,j)*u(-1,j))/(dx(0,j)+dx(-1,j))  &
                +       ((2.*dx(1,j)+dx( 2,j))*(u(1,j))-dx(1,j)*u( 2,j))/(dx(1,j)+dx( 2,j)) )
                 br(0) = xt - u(0,j)
                 bl(1) = xt - u(1,j)
             endif
!       call pert_ppm(1, u(2,j), bl(2), br(2), -1)
        endif
        if ( (ie+1)==npx ) then
             bl(npx-2) = al(npx-2) - u(npx-2,j)
             xt = c1*u(npx-3,j) + c2*u(npx-2,j) + c3*u(npx-1,j)
             br(npx-2) = xt - u(npx-2,j)
             bl(npx-1) = xt - u(npx-1,j)
             if( j==1 .or. j==npy ) then
                 bl(npx-1) = 0.  ! in
                 br(npx-1) = 0.  ! edge
                 bl(npx  ) = 0.  ! edge
                 br(npx  ) = 0.  ! out
             else
             xt = 0.5*( ( (2.*dx(npx-1,j)+dx(npx-2,j))*u(npx-1,j)-dx(npx-1,j)*u(npx-2,j))/(dx(npx-1,j)+dx(npx-2,j)) &
                +       ( (2.*dx(npx  ,j)+dx(npx+1,j))*u(npx  ,j)-dx(npx  ,j)*u(npx+1,j))/(dx(npx  ,j)+dx(npx+1,j)) )
                 br(npx-1) = xt - u(npx-1,j)
                 bl(npx  ) = xt - u(npx  ,j)
                 br(npx) = c3*u(npx,j) + c2*u(npx+1,j) + c1*u(npx+2,j) - u(npx,j)
             endif
!       call pert_ppm(1, u(npx-2,j), bl(npx-2), br(npx-2), -1)
        endif
      endif

     do i=is-1,ie+1
        b0(i) = bl(i) + br(i)
     enddo

    if ( iord==1 ) then

      do i=is-1, ie+1
         smt5(i) = abs(lim_fac*b0(i)) < abs(bl(i)-br(i))
      enddo
!DEC$ VECTOR ALWAYS
      do i=is,ie+1
         if( c(i,j)>0. ) then
             cfl = c(i,j)*rdx(i-1,j)
             fx0(i) = (1.-cfl)*(br(i-1)-cfl*b0(i-1))
             flux(i,j) = u(i-1,j)
         else
             cfl = c(i,j)*rdx(i,j)
             fx0(i) = (1.+cfl)*(bl(i)+cfl*b0(i))
             flux(i,j) = u(i,j)
         endif
         if (smt5(i-1).or.smt5(i)) flux(i,j) = flux(i,j) + fx0(i)
      enddo

     elseif ( iord==2 ) then   ! Perfectly linear

!DEC$ VECTOR ALWAYS
        do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdx(i-1,j)
               flux(i,j) = u(i-1,j) + (1.-cfl)*(br(i-1)-cfl*b0(i-1))
           else
               cfl = c(i,j)*rdx(i,j)
               flux(i,j) = u(i,j) + (1.+cfl)*(bl(i)+cfl*b0(i))
           endif
        enddo

     elseif ( iord==3 ) then

          do i=is-1, ie+1
             x0 = abs(b0(i))
             x1 = abs(bl(i)-br(i))
             smt5(i) =    x0 < x1
             smt6(i) = 3.*x0 < x1
          enddo
          do i=is, ie+1
             fx0(i) = 0.
             hi5(i) = smt5(i-1) .and. smt5(i)
             hi6(i) = smt6(i-1) .or.  smt6(i)
          enddo
          do i=is, ie+1
             if( c(i,j)>0. ) then
                 cfl = c(i,j)*rdx(i-1,j)
                 if ( hi6(i) ) then
                    fx0(i) = br(i-1) - cfl*b0(i-1)
                 elseif( hi5(i) ) then
                    fx0(i) = sign(min(abs(bl(i-1)),abs(br(i-1))), br(i-1))
                 endif
                 flux(i,j) = u(i-1,j) + (1.-cfl)*fx0(i)
             else
                 cfl = c(i,j)*rdx(i,j)
                 if ( hi6(i) ) then
                    fx0(i) = bl(i) + cfl*b0(i)
                 elseif( hi5(i) ) then
                    fx0(i) = sign(min(abs(bl(i)),abs(br(i))), bl(i))
                 endif
                 flux(i,j) = u(i,j) + (1.+cfl)*fx0(i)
             endif
          enddo

     elseif ( iord==4 ) then

          do i=is-1, ie+1
             x0 = abs(b0(i))
             x1 = abs(bl(i)-br(i))
             smt5(i) =    x0 < x1
             smt6(i) = 3.*x0 < x1
          enddo
          do i=is, ie+1
             hi5(i) = smt5(i-1) .and. smt5(i)
             hi6(i) = smt6(i-1) .or.  smt6(i)
             hi5(i) = hi5(i) .or. hi6(i)
          enddo
!DEC$ VECTOR ALWAYS
          do i=is,ie+1
             if( c(i,j)>0. ) then
                 cfl = c(i,j)*rdx(i-1,j)
                 fx0(i) = (1.-cfl)*(br(i-1)-cfl*b0(i-1))
                 flux(i,j) = u(i-1,j)
             else
                 cfl = c(i,j)*rdx(i,j)
                 fx0(i) = (1.+cfl)*(bl(i)+cfl*b0(i))
                 flux(i,j) = u(i,j)
             endif
             if ( hi5(i) ) flux(i,j) = flux(i,j) + fx0(i)
          enddo

     else    !  iord=5,6,7

        if ( iord==5 ) then
           do i=is-1, ie+1
              smt5(i) = bl(i)*br(i) < 0.
           enddo
        else

           do i=is-1, ie+1
              smt5(i) = 3.*abs(b0(i)) < abs(bl(i)-br(i))
           enddo
!WMP
! fix edge issues
           if ( (.not. bounded_domain) .and. grid_type < 3) then
              if( is==1 ) then
                 smt5(0) = bl(0)*br(0) < 0.
                 smt5(1) = bl(1)*br(1) < 0.
              endif
              if( (ie+1)==npx ) then
                 smt5(npx-1) = bl(npx-1)*br(npx-1) < 0.
                 smt5(npx ) = bl(npx )*br(npx ) < 0.
              endif
           endif
        endif

!DEC$ VECTOR ALWAYS
        do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdx(i-1,j)
               fx0(i) = (1.-cfl)*(br(i-1)-cfl*b0(i-1))
               flux(i,j) = u(i-1,j)
           else
               cfl = c(i,j)*rdx(i,j)
               fx0(i) = (1.+cfl)*(bl(i)+cfl*b0(i))
               flux(i,j) = u(i,j)
           endif
           if (smt5(i-1).or.smt5(i)) flux(i,j) = flux(i,j) + fx0(i)
        enddo

     endif
   enddo

 else
 ! iord = 8, 9, 10, 11

   do j=js,je+1
        do i=is-2,ie+2
           xt = 0.25*(u(i+1,j) - u(i-1,j))
           dm(i) = sign(min(abs(xt), max(u(i-1,j), u(i,j), u(i+1,j)) - u(i,j),  &
                            u(i,j) - min(u(i-1,j), u(i,j), u(i+1,j))), xt)
        enddo
        do i=is-3,ie+2
           dq(i) = u(i+1,j) - u(i,j)
        enddo

      if (grid_type < 3) then

          do i=is3,ie3+1
             al(i) = 0.5*(u(i-1,j)+u(i,j)) + r3*(dm(i-1) - dm(i))
          enddo

! Perturbation form:
         if( iord==8 ) then
             do i=is3,ie3
                xt = 2.*dm(i)
                bl(i) = -sign(min(abs(xt), abs(al(i  )-u(i,j))), xt)
                br(i) =  sign(min(abs(xt), abs(al(i+1)-u(i,j))), xt)
             enddo
         elseif( iord==9 ) then
             do i=is3,ie3
              pmp_1 = -2.*dq(i)
              lac_1 = pmp_1 + 1.5*dq(i+1)
              bl(i) = min(max(0., pmp_1, lac_1), max(al(i  )-u(i,j), min(0., pmp_1, lac_1)))
              pmp_2 = 2.*dq(i-1)
              lac_2 = pmp_2 - 1.5*dq(i-2)
              br(i) = min(max(0., pmp_2, lac_2), max(al(i+1)-u(i,j), min(0., pmp_2, lac_2)))
           enddo
         elseif( iord==10 ) then
           do i=is3,ie3
              bl(i) = al(i  ) - u(i,j)
              br(i) = al(i+1) - u(i,j)
!             if ( abs(dm(i-1))+abs(dm(i))+abs(dm(i+1)) < near_zero ) then
              if ( abs(dm(i)) < near_zero ) then
                if ( abs(dm(i-1))+abs(dm(i+1)) < near_zero ) then
! 2-delta-x structure detected within 3 cells
                   bl(i) = 0.
                   br(i) = 0.
                endif
              elseif( abs(3.*(bl(i)+br(i))) > abs(bl(i)-br(i)) ) then
                   pmp_1 = -2.*dq(i)
                   lac_1 = pmp_1 + 1.5*dq(i+1)
                   bl(i) = min(max(0., pmp_1, lac_1), max(bl(i), min(0., pmp_1, lac_1)))
                   pmp_2 = 2.*dq(i-1)
                   lac_2 = pmp_2 - 1.5*dq(i-2)
                   br(i) = min(max(0., pmp_2, lac_2), max(br(i), min(0., pmp_2, lac_2)))
              endif
             enddo
         else
! un-limited: 11
             do i=is3,ie3
                bl(i) = al(i  ) - u(i,j)
                br(i) = al(i+1) - u(i,j)
             enddo
         endif

!--------------
! fix the edges
!--------------
!!! TO DO: separate versions for bounded_domain and for cubed-sphere
           if ( is==1 .and. .not. bounded_domain) then
              br(2) = al(3) - u(2,j)
              xt = s15*u(1,j) + s11*u(2,j) - s14*dm(2)
              bl(2) = xt - u(2,j)
              br(1) = xt - u(1,j)
              if( j==1 .or. j==npy ) then
                 bl(0) = 0.   ! out
                 br(0) = 0.   ! edge
                 bl(1) = 0.   ! edge
                 br(1) = 0.   ! in
              else
                 bl(0) = s14*dm(-1) - s11*dq(-1)
                 x0L = 0.5*((2.*dx(0,j)+dx(-1,j))*(u(0,j))   &
                      - dx(0,j)*(u(-1,j)))/(dx(0,j)+dx(-1,j))
                 x0R = 0.5*((2.*dx(1,j)+dx(2,j))*(u(1,j))   &
                      - dx(1,j)*(u(2,j)))/(dx(1,j)+dx(2,j))
                 xt = x0L + x0R
                 br(0) = xt - u(0,j)
                 bl(1) = xt - u(1,j)
              endif
              call pert_ppm(1, u(2,j), bl(2), br(2), -1)
           endif

           if ( (ie+1)==npx  .and. .not. bounded_domain) then
              bl(npx-2) = al(npx-2) - u(npx-2,j)
              xt = s15*u(npx-1,j) + s11*u(npx-2,j) + s14*dm(npx-2)
              br(npx-2) = xt - u(npx-2,j)
              bl(npx-1) = xt - u(npx-1,j)
              if( j==1 .or. j==npy ) then
                 bl(npx-1) = 0.   ! in
                 br(npx-1) = 0.   ! edge
                 bl(npx  ) = 0.   ! edge
                 br(npx  ) = 0.   ! out
              else
                 br(npx) = s11*dq(npx) - s14*dm(npx+1)
                 x0L = 0.5*( (2.*dx(npx-1,j)+dx(npx-2,j))*(u(npx-1,j))  &
                      - dx(npx-1,j)*(u(npx-2,j)))/(dx(npx-1,j)+dx(npx-2,j))
                 x0R = 0.5*( (2.*dx(npx,j)+dx(npx+1,j))*(u(npx,j))  &
                      - dx(npx,j)*(u(npx+1,j)))/(dx(npx,j)+dx(npx+1,j))
                 xt = x0L + x0R
                 br(npx-1) = xt - u(npx-1,j)
                 bl(npx  ) = xt - u(npx  ,j)
              endif
              call pert_ppm(1, u(npx-2,j), bl(npx-2), br(npx-2), -1)
           endif

      else
! Other grids:
              do i=is-1,ie+2
                 al(i) = 0.5*(u(i-1,j)+u(i,j)) + r3*(dm(i-1) - dm(i))
              enddo

              do i=is-1,ie+1
                 pmp = -2.*dq(i)
                 lac = pmp + 1.5*dq(i+1)
                 bl(i) = min(max(0., pmp, lac), max(al(i  )-u(i,j), min(0.,pmp, lac)))
                 pmp = 2.*dq(i-1)
                 lac = pmp - 1.5*dq(i-2)
                 br(i) = min(max(0., pmp, lac), max(al(i+1)-u(i,j), min(0.,pmp, lac)))
              enddo
        endif

        do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdx(i-1,j)
               flux(i,j) = u(i-1,j) + (1.-cfl)*(br(i-1)-cfl*(bl(i-1)+br(i-1)))
           else
               cfl = c(i,j)*rdx(i,j)
               flux(i,j) = u(i,  j) + (1.+cfl)*(bl(i  )+cfl*(bl(i  )+br(i  )))
           endif
        enddo
     enddo

 endif

 end subroutine xtp_u

 subroutine ytp_v(is,ie,js,je,isd,ied,jsd,jed, c, u, v, flux, jord, dy, rdy, npx, npy, grid_type, bounded_domain, lim_fac)
 integer, intent(in):: is,ie,js,je, isd,ied,jsd,jed
 integer, intent(IN):: jord
 real, INTENT(IN)  ::   u(isd:ied,jsd:jed+1)
 real, INTENT(IN)  ::   v(isd:ied+1,jsd:jed)
 real, INTENT(IN) ::    c(is:ie+1,js:je+1)   !  Courant   N (like FLUX)
 real, INTENT(OUT):: flux(is:ie+1,js:je+1)
 real, INTENT(IN) ::   dy(isd:ied+1,jsd:jed)
 real, INTENT(IN) ::  rdy(isd:ied+1,jsd:jed)
 integer, INTENT(IN) :: npx, npy, grid_type
 logical, INTENT(IN) :: bounded_domain
 real, INTENT(IN) ::  lim_fac
! Local:
 logical, dimension(is:ie+1,js-1:je+1):: smt5, smt6
 logical, dimension(is:ie+1):: hi5, hi6
 real:: fx0(is:ie+1)
 real dm(is:ie+1,js-2:je+2)
 real al(is:ie+1,js-1:je+2)
 real, dimension(is:ie+1,js-1:je+1):: bl, br, b0
 real dq(is:ie+1,js-3:je+2)
 real xt, dl, dr, pmp, lac, cfl
 real pmp_1, lac_1, pmp_2, lac_2
 real x0, x1, x0R, x0L
 integer i, j, is1, ie1, js3, je3

 if ( bounded_domain .or. grid_type>3 ) then
    js3 = js-1;        je3 = je+1
 else
    js3 = max(3,js-1); je3 = min(npy-3,je+1)
 end if

 if ( jord<8 ) then
! Diffusivity: ord2 < ord5 < ord3 < ord4 < ord6

   do j=js3,je3+1
      do i=is,ie+1
         al(i,j) = p1*(v(i,j-1)+v(i,j)) + p2*(v(i,j-2)+v(i,j+1))
      enddo
   enddo
   do j=js3,je3
      do i=is,ie+1
          bl(i,j) = al(i,j  ) - v(i,j)
          br(i,j) = al(i,j+1) - v(i,j)
      enddo
   enddo

   if ( (.not.bounded_domain) .and. grid_type < 3) then
     if( js==1 ) then
       do i=is,ie+1
          bl(i,0) = c1*v(i,-2) + c2*v(i,-1) + c3*v(i,0) - v(i,0)
          xt = 0.5*( ((2.*dy(i,0)+dy(i,-1))*v(i,0)-dy(i,0)*v(i,-1))/(dy(i,0)+dy(i,-1)) &
             +       ((2.*dy(i,1)+dy(i, 2))*v(i,1)-dy(i,1)*v(i, 2))/(dy(i,1)+dy(i, 2)) )
          br(i,0) = xt - v(i,0)
          bl(i,1) = xt - v(i,1)
          xt = c3*v(i,1) + c2*v(i,2) + c1*v(i,3)
          br(i,1) = xt - v(i,1)
          bl(i,2) = xt - v(i,2)
          br(i,2) = al(i,3) - v(i,2)
       enddo
       if ( is==1 ) then
            bl(1,0) = 0.  ! out
            br(1,0) = 0.  ! edge
            bl(1,1) = 0.  ! edge
            br(1,1) = 0.  ! in
       endif
       if ( (ie+1)==npx ) then
            bl(npx,0) = 0.   ! out
            br(npx,0) = 0.   ! edge
            bl(npx,1) = 0.   ! edge
            br(npx,1) = 0.   ! in
       endif
!      j=2
!      call pert_ppm(ie-is+2, v(is,j), bl(is,j), br(is,j), -1)
   endif
   if( (je+1)==npy ) then
       do i=is,ie+1
          bl(i,npy-2) = al(i,npy-2) - v(i,npy-2)
          xt = c1*v(i,npy-3) + c2*v(i,npy-2) + c3*v(i,npy-1)
          br(i,npy-2) = xt - v(i,npy-2)
          bl(i,npy-1) = xt - v(i,npy-1)
          xt = 0.5*( ((2.*dy(i,npy-1)+dy(i,npy-2))*v(i,npy-1)-dy(i,npy-1)*v(i,npy-2))/(dy(i,npy-1)+dy(i,npy-2)) &
             +       ((2.*dy(i,npy  )+dy(i,npy+1))*v(i,npy  )-dy(i,npy  )*v(i,npy+1))/(dy(i,npy  )+dy(i,npy+1)) )
          br(i,npy-1) = xt - v(i,npy-1)
          bl(i,npy  ) = xt - v(i,npy)
          br(i,npy) = c3*v(i,npy)+ c2*v(i,npy+1) + c1*v(i,npy+2) - v(i,npy)
       enddo
       if ( is==1 ) then
            bl(1,npy-1) = 0.  ! in
            br(1,npy-1) = 0.  ! edge
            bl(1,npy  ) = 0.  ! edge
            br(1,npy  ) = 0.  ! out
       endif
       if ( (ie+1)==npx ) then
            bl(npx,npy-1) = 0.  ! in
            br(npx,npy-1) = 0.  ! edge
            bl(npx,npy  ) = 0.  ! edge
            br(npx,npy  ) = 0.  ! out
       endif
!      j=npy-2
!      call pert_ppm(ie-is+2, v(is,j), bl(is,j), br(is,j), -1)
     endif
   endif

   do j=js-1,je+1
      do i=is,ie+1
         b0(i,j) = bl(i,j) + br(i,j)
      enddo
   enddo

   if ( jord==1 ) then    ! Perfectly linear

     do j=js-1,je+1
        do i=is,ie+1
           smt5(i,j) = abs(lim_fac*b0(i,j)) < abs(bl(i,j)-br(i,j))
        enddo
     enddo
     do j=js,je+1
!DEC$ VECTOR ALWAYS
        do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdy(i,j-1)
               fx0(i) = (1.-cfl)*(br(i,j-1)-cfl*b0(i,j-1))
               flux(i,j) = v(i,j-1)
           else
               cfl = c(i,j)*rdy(i,j)
               fx0(i) = (1.+cfl)*(bl(i,j)+cfl*b0(i,j))
               flux(i,j) = v(i,j)
           endif
           if (smt5(i,j-1).or.smt5(i,j)) flux(i,j) = flux(i,j) + fx0(i)
        enddo
     enddo

   elseif ( jord==2 ) then    ! Perfectly linear
      do j=js,je+1
!DEC$ VECTOR ALWAYS
         do i=is,ie+1
            if( c(i,j)>0. ) then
               cfl = c(i,j)*rdy(i,j-1)
               flux(i,j) = v(i,j-1) + (1.-cfl)*(br(i,j-1)-cfl*b0(i,j-1))
            else
               cfl = c(i,j)*rdy(i,j)
               flux(i,j) = v(i,j) + (1.+cfl)*(bl(i,j)+cfl*b0(i,j))
            endif
          enddo
      enddo

   elseif ( jord==3 ) then

       do j=js-1,je+1
          do i=is,ie+1
             x0 = abs(b0(i,j))
             x1 = abs(bl(i,j)-br(i,j))
             smt5(i,j) =    x0 < x1
             smt6(i,j) = 3.*x0 < x1
          enddo
       enddo
       do j=js,je+1
          do i=is,ie+1
             fx0(i) = 0.
             hi5(i) = smt5(i,j-1) .and. smt5(i,j)
             hi6(i) = smt6(i,j-1) .or.  smt6(i,j)
          enddo
          do i=is,ie+1
             if( c(i,j)>0. ) then
                 cfl = c(i,j)*rdy(i,j-1)
                 if ( hi6(i) ) then
                    fx0(i) = br(i,j-1) - cfl*b0(i,j-1)
                 elseif ( hi5(i) ) then  ! piece-wise linear
                    fx0(i) = sign(min(abs(bl(i,j-1)),abs(br(i,j-1))), br(i,j-1))
                 endif
                 flux(i,j) = v(i,j-1) + (1.-cfl)*fx0(i)
             else
                 cfl = c(i,j)*rdy(i,j)
                 if ( hi6(i) ) then
                    fx0(i) = bl(i,j) + cfl*b0(i,j)
                 elseif ( hi5(i) ) then  ! piece-wise linear
                    fx0(i) = sign(min(abs(bl(i,j)),abs(br(i,j))), bl(i,j))
                 endif
                 flux(i,j) = v(i,j) + (1.+cfl)*fx0(i)
             endif
          enddo
       enddo

   elseif ( jord==4 ) then

       do j=js-1,je+1
          do i=is,ie+1
             x0 = abs(b0(i,j))
             x1 = abs(bl(i,j)-br(i,j))
             smt5(i,j) =    x0 < x1
             smt6(i,j) = 3.*x0 < x1
          enddo
       enddo
       do j=js,je+1
          do i=is,ie+1
             fx0(i) = 0.
             hi5(i) = smt5(i,j-1) .and. smt5(i,j)
             hi6(i) = smt6(i,j-1) .or.  smt6(i,j)
             hi5(i) = hi5(i) .or. hi6(i)
          enddo
!DEC$ VECTOR ALWAYS
          do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdy(i,j-1)
               fx0(i) = (1.-cfl)*(br(i,j-1)-cfl*b0(i,j-1))
               flux(i,j) = v(i,j-1)
           else
               cfl = c(i,j)*rdy(i,j)
               fx0(i) = (1.+cfl)*(bl(i,j)+cfl*b0(i,j))
               flux(i,j) = v(i,j)
           endif
           if ( hi5(i) ) flux(i,j) = flux(i,j) + fx0(i)
          enddo
       enddo

   else   ! jord = 5,6,7
     if ( jord==5 ) then

        do j=js-1,je+1
           do i=is,ie+1
              smt5(i,j) = bl(i,j)*br(i,j) < 0.
           enddo
        enddo
        do j=js,je+1
!DEC$ VECTOR ALWAYS
        do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdy(i,j-1)
               fx0(i) = (1.-cfl)*(br(i,j-1)-cfl*b0(i,j-1))
               flux(i,j) = v(i,j-1)
           else
               cfl = c(i,j)*rdy(i,j)
               fx0(i) = (1.+cfl)*(bl(i,j)+cfl*b0(i,j))
               flux(i,j) = v(i,j)
           endif
           if (smt5(i,j-1).or.smt5(i,j)) flux(i,j) = flux(i,j) + fx0(i)
        enddo
        enddo
     else
! hord=6
        do j=js-1,je+1
           do i=is,ie+1
              smt6(i,j) = 3.*abs(b0(i,j)) < abs(bl(i,j)-br(i,j))
           enddo
        enddo

!WMP
! fix edge issues
        if ( (.not.bounded_domain) .and. grid_type < 3) then
           if( js==1 ) then
              do i=is,ie+1
                 smt6(i,0) = bl(i,0)*br(i,0) < 0.
                 smt6(i,1) = bl(i,1)*br(i,1) < 0.
              enddo
           endif
           if( (je+1)==npy ) then
              do i=is,ie+1
                 smt6(i,npy-1) = bl(i,npy-1)*br(i,npy-1) < 0.
                 smt6(i,npy ) = bl(i,npy )*br(i,npy ) < 0.
              enddo
           endif
        endif


        do j=js,je+1
!DEC$ VECTOR ALWAYS
        do i=is,ie+1
           if( c(i,j)>0. ) then
               cfl = c(i,j)*rdy(i,j-1)
               fx0(i) = (1.-cfl)*(br(i,j-1)-cfl*b0(i,j-1))
               flux(i,j) = v(i,j-1)
           else
               cfl = c(i,j)*rdy(i,j)
               fx0(i) = (1.+cfl)*(bl(i,j)+cfl*b0(i,j))
               flux(i,j) = v(i,j)
           endif
           if (smt6(i,j-1).or.smt6(i,j)) flux(i,j) = flux(i,j) + fx0(i)
        enddo
        enddo
     endif

   endif

 else
! jord= 8, 9, 10

   do j=js-2,je+2
      do i=is,ie+1
         xt = 0.25*(v(i,j+1) - v(i,j-1))
         dm(i,j) = sign(min(abs(xt), max(v(i,j-1), v(i,j), v(i,j+1)) - v(i,j),   &
                            v(i,j) - min(v(i,j-1), v(i,j), v(i,j+1))), xt)
      enddo
   enddo

   do j=js-3,je+2
      do i=is,ie+1
         dq(i,j) = v(i,j+1) - v(i,j)
      enddo
   enddo

   if (grid_type < 3) then
      do j=js3,je3+1
         do i=is,ie+1
            al(i,j) = 0.5*(v(i,j-1)+v(i,j)) + r3*(dm(i,j-1)-dm(i,j))
         enddo
      enddo

      if ( jord==8 ) then
        do j=js3,je3
           do i=is,ie+1
              xt =  2.*dm(i,j)
              bl(i,j) = -sign(min(abs(xt), abs(al(i,j)-v(i,j))),   xt)
              br(i,j) =  sign(min(abs(xt), abs(al(i,j+1)-v(i,j))), xt)
           enddo
        enddo
      elseif ( jord==9 ) then
        do j=js3,je3
           do i=is,ie+1
              pmp_1 = -2.*dq(i,j)
              lac_1 = pmp_1 + 1.5*dq(i,j+1)
            bl(i,j) = min(max(0., pmp_1, lac_1), max(al(i,j)-v(i,j), min(0., pmp_1, lac_1)))
              pmp_2 = 2.*dq(i,j-1)
              lac_2 = pmp_2 - 1.5*dq(i,j-2)
            br(i,j) = min(max(0., pmp_2, lac_2), max(al(i,j+1)-v(i,j), min(0., pmp_2, lac_2)))
         enddo
      enddo
    elseif ( jord==10 ) then
      do j=js3,je3
         do i=is,ie+1
            bl(i,j) = al(i,j  ) - v(i,j)
            br(i,j) = al(i,j+1) - v(i,j)
!           if ( abs(dm(i,j-1))+abs(dm(i,j))+abs(dm(i,j+1)) < near_zero ) then
            if ( abs(dm(i,j)) < near_zero ) then
              if ( abs(dm(i,j-1))+abs(dm(i,j+1)) < near_zero ) then
                 bl(i,j) = 0.
                 br(i,j) = 0.
              endif
            elseif( abs(3.*(bl(i,j)+br(i,j))) > abs(bl(i,j)-br(i,j)) ) then
                  pmp_1 = -2.*dq(i,j)
                  lac_1 = pmp_1 + 1.5*dq(i,j+1)
                bl(i,j) = min(max(0., pmp_1, lac_1), max(bl(i,j), min(0., pmp_1, lac_1)))
                  pmp_2 = 2.*dq(i,j-1)
                  lac_2 = pmp_2 - 1.5*dq(i,j-2)
                br(i,j) = min(max(0., pmp_2, lac_2), max(br(i,j), min(0., pmp_2, lac_2)))
            endif
           enddo
        enddo
      else
! Unlimited:
        do j=js3,je3
           do i=is,ie+1
              bl(i,j) = al(i,j  ) - v(i,j)
              br(i,j) = al(i,j+1) - v(i,j)
           enddo
        enddo
      endif

!--------------
! fix the edges
!--------------
      if( js==1 .and. .not. bounded_domain) then
         do i=is,ie+1
            br(i,2) = al(i,3) - v(i,2)
            xt = s15*v(i,1) + s11*v(i,2) - s14*dm(i,2)
            br(i,1) = xt - v(i,1)
            bl(i,2) = xt - v(i,2)

            bl(i,0) = s14*dm(i,-1) - s11*dq(i,-1)

#ifdef ONE_SIDE
            xt =  t14*v(i,1) +  t12*v(i,2) + t15*v(i,3)
            bl(i,1) = 2.*xt - v(i,1)
            xt =  t14*v(i,0) +  t12*v(i,-1) + t15*v(i,-2)
            br(i,0) = 2.*xt - v(i,0)
#else
            x0L = 0.5*( (2.*dy(i,0)+dy(i,-1))*(v(i,0))   &
               - dy(i,0)*(v(i,-1)))/(dy(i,0)+dy(i,-1))
            x0R = 0.5*( (2.*dy(i,1)+dy(i,2))*(v(i,1))   &
               - dy(i,1)*(v(i,2)))/(dy(i,1)+dy(i,2))
            xt = x0L + x0R

             bl(i,1) = xt - v(i,1)
             br(i,0) = xt - v(i,0)
#endif
         enddo
         if ( is==1 ) then
               bl(1,0) = 0.   ! out
               br(1,0) = 0.   ! edge
               bl(1,1) = 0.   ! edge
               br(1,1) = 0.   ! in
         endif
         if ( (ie+1)==npx ) then
               bl(npx,0) = 0.   ! out
               br(npx,0) = 0.   ! edge
               bl(npx,1) = 0.   ! edge
               br(npx,1) = 0.   ! in
         endif
         j=2
         call pert_ppm(ie-is+2, v(is,j), bl(is,j), br(is,j), -1)
      endif
      if( (je+1)==npy  .and. .not. bounded_domain) then
         do i=is,ie+1
            bl(i,npy-2) = al(i,npy-2) - v(i,npy-2)
            xt = s15*v(i,npy-1) + s11*v(i,npy-2) + s14*dm(i,npy-2)
            br(i,npy-2) = xt - v(i,npy-2)
            bl(i,npy-1) = xt - v(i,npy-1)
            br(i,npy) = s11*dq(i,npy) - s14*dm(i,npy+1)
#ifdef ONE_SIDE
            xt = t14*v(i,npy-1) + t12*v(i,npy-2) + t15*v(i,npy-3)
            br(i,npy-1) = 2.*xt - v(i,npy-1)
            xt = t14*v(i,npy) + t12*v(i,npy+1) + t15*v(i,npy+2)
            bl(i,npy  ) = 2.*xt - v(i,npy)
#else
            x0L= 0.5*((2.*dy(i,npy-1)+dy(i,npy-2))*(v(i,npy-1)) -  &
                 dy(i,npy-1)*(v(i,npy-2)))/(dy(i,npy-1)+dy(i,npy-2))
            x0R= 0.5*((2.*dy(i,npy)+dy(i,npy+1))*(v(i,npy)) -  &
                 dy(i,npy)*(v(i,npy+1)))/(dy(i,npy)+dy(i,npy+1))
            xt = x0L + x0R

            br(i,npy-1) = xt - v(i,npy-1)
            bl(i,npy  ) = xt - v(i,npy)
#endif
         enddo
         if ( is==1 ) then
               bl(1,npy-1) = 0.   ! in
               br(1,npy-1) = 0.   ! edge
               bl(1,npy  ) = 0.   ! edge
               br(1,npy  ) = 0.   ! out
         endif
         if ( (ie+1)==npx ) then
               bl(npx,npy-1) = 0.   ! in
               br(npx,npy-1) = 0.   ! edge
               bl(npx,npy  ) = 0.   ! edge
               br(npx,npy  ) = 0.   ! out
         endif
         j=npy-2
         call pert_ppm(ie-is+2, v(is,j), bl(is,j), br(is,j), -1)
      endif

   else

      do j=js-1,je+2
         do i=is,ie+1
            al(i,j) = 0.5*(v(i,j-1)+v(i,j)) + r3*(dm(i,j-1)-dm(i,j))
         enddo
      enddo

      do j=js-1,je+1
         do i=is,ie+1
            pmp = 2.*dq(i,j-1)
            lac = pmp - 1.5*dq(i,j-2)
            br(i,j) = min(max(0.,pmp,lac), max(al(i,j+1)-v(i,j), min(0.,pmp,lac)))
            pmp = -2.*dq(i,j)
            lac = pmp + 1.5*dq(i,j+1)
            bl(i,j) = min(max(0.,pmp,lac), max(al(i,j)-v(i,j), min(0.,pmp,lac)))
         enddo
      enddo

   endif

   do j=js,je+1
      do i=is,ie+1
         if(c(i,j)>0.) then
            cfl = c(i,j)*rdy(i,j-1)
            flux(i,j) = v(i,j-1) + (1.-cfl)*(br(i,j-1)-cfl*(bl(i,j-1)+br(i,j-1)))
         else
            cfl = c(i,j)*rdy(i,j)
            flux(i,j) = v(i,j  ) + (1.+cfl)*(bl(i,j  )+cfl*(bl(i,j  )+br(i,j  )))
         endif
      enddo
   enddo

 endif

end subroutine ytp_v

      subroutine fill_corners_2d_r8(q, npx, npy, FILL, AGRID, BGRID)
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: q
         integer, intent(IN):: npx,npy
         integer, intent(IN):: FILL  ! X-Dir or Y-Dir
         logical, OPTIONAL, intent(IN) :: AGRID, BGRID
         integer :: i,j

         if (present(BGRID)) then
            if (BGRID) then
              select case (FILL)
              case (XDir)
                 do j=1,ng
                    do i=1,ng
                     if ((is==    1) .and. (js==    1)) q(1-i  ,1-j  ) = q(1-j  ,i+1    )  !SW Corner
                     if ((is==    1) .and. (je==npy-1)) q(1-i  ,npy+j) = q(1-j  ,npy-i  )  !NW Corner
                     if ((ie==npx-1) .and. (js==    1)) q(npx+i,1-j  ) = q(npx+j,i+1    )  !SE Corner
                     if ((ie==npx-1) .and. (je==npy-1)) q(npx+i,npy+j) = q(npx+j,npy-i  )  !NE Corner
                    enddo
                 enddo
              case (YDir)
                 do j=1,ng
                    do i=1,ng
                     if ((is==    1) .and. (js==    1)) q(1-j  ,1-i  ) = q(i+1  ,1-j    )  !SW Corner
                     if ((is==    1) .and. (je==npy-1)) q(1-j  ,npy+i) = q(i+1  ,npy+j  )  !NW Corner
                     if ((ie==npx-1) .and. (js==    1)) q(npx+j,1-i  ) = q(npx-i,1-j    )  !SE Corner
                     if ((ie==npx-1) .and. (je==npy-1)) q(npx+j,npy+i) = q(npx-i,npy+j  )  !NE Corner
                    enddo
                 enddo
              case default
                 do j=1,ng
                    do i=1,ng
                     if ((is==    1) .and. (js==    1)) q(1-i  ,1-j  ) = q(1-j  ,i+1    )  !SW Corner
                     if ((is==    1) .and. (je==npy-1)) q(1-i  ,npy+j) = q(1-j  ,npy-i  )  !NW Corner
                     if ((ie==npx-1) .and. (js==    1)) q(npx+i,1-j  ) = q(npx+j,i+1    )  !SE Corner
                     if ((ie==npx-1) .and. (je==npy-1)) q(npx+i,npy+j) = q(npx+j,npy-i  )  !NE Corner
                    enddo
                 enddo
              end select
            endif
          elseif (present(AGRID)) then
            if (AGRID) then
              select case (FILL)
              case (XDir)
                 do j=1,ng
                    do i=1,ng
                       if ((is==    1) .and. (js==    1)) q(1-i    ,1-j    ) = q(1-j    ,i        )  !SW Corner
                       if ((is==    1) .and. (je==npy-1)) q(1-i    ,npy-1+j) = q(1-j    ,npy-1-i+1)  !NW Corner
                       if ((ie==npx-1) .and. (js==    1)) q(npx-1+i,1-j    ) = q(npx-1+j,i        )  !SE Corner
                       if ((ie==npx-1) .and. (je==npy-1)) q(npx-1+i,npy-1+j) = q(npx-1+j,npy-1-i+1)  !NE Corner
                    enddo
                 enddo
              case (YDir)
                 do j=1,ng
                    do i=1,ng
                       if ((is==    1) .and. (js==    1)) q(1-j    ,1-i    ) = q(i        ,1-j    )  !SW Corner
                       if ((is==    1) .and. (je==npy-1)) q(1-j    ,npy-1+i) = q(i        ,npy-1+j)  !NW Corner
                       if ((ie==npx-1) .and. (js==    1)) q(npx-1+j,1-i    ) = q(npx-1-i+1,1-j    )  !SE Corner
                       if ((ie==npx-1) .and. (je==npy-1)) q(npx-1+j,npy-1+i) = q(npx-1-i+1,npy-1+j)  !NE Corner
                    enddo
                 enddo
              case default
                 do j=1,ng
                    do i=1,ng
                       if ((is==    1) .and. (js==    1)) q(1-j    ,1-i    ) = q(i        ,1-j    )  !SW Corner
                       if ((is==    1) .and. (je==npy-1)) q(1-j    ,npy-1+i) = q(i        ,npy-1+j)  !NW Corner
                       if ((ie==npx-1) .and. (js==    1)) q(npx-1+j,1-i    ) = q(npx-1-i+1,1-j    )  !SE Corner
                       if ((ie==npx-1) .and. (je==npy-1)) q(npx-1+j,npy-1+i) = q(npx-1-i+1,npy-1+j)  !NE Corner
                   enddo
                 enddo
              end select
            endif
          endif

      end subroutine fill_corners_2d_r8

      subroutine fill_corners_xy_2d_r8(x, y, npx, npy, DGRID, AGRID, CGRID, VECTOR)
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: x !(isd:ied  ,jsd:jed+1)
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: y !(isd:ied+1,jsd:jed  )
         integer, intent(IN):: npx,npy
         logical, OPTIONAL, intent(IN) :: DGRID, AGRID, CGRID, VECTOR
         integer :: i,j

         real(kind=8) :: mySign

         mySign = 1.0
         if (present(VECTOR)) then
            if (VECTOR) mySign = -1.0
         endif

         if (present(DGRID)) then
            call fill_corners_dgrid(x, y, npx, npy, mySign)
         elseif (present(CGRID)) then
            call fill_corners_cgrid(x, y, npx, npy, mySign)
         elseif (present(AGRID)) then
            call fill_corners_agrid(x, y, npx, npy, mySign)
         else
            call fill_corners_agrid(x, y, npx, npy, mySign)
         endif

      end subroutine fill_corners_xy_2d_r8

      subroutine fill_corners_dgrid_r8(x, y, npx, npy, mySign)
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: x
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: y
         integer, intent(IN):: npx,npy
         real(kind=8), intent(IN) :: mySign
         integer :: i,j

               do j=1,ng
                  do i=1,ng
                   !   if ((is  ==  1) .and. (js  ==  1)) x(1-i    ,1-j  ) =        y(j+1  ,1-i    )  !SW Corner
                   !   if ((is  ==  1) .and. (je+1==npy)) x(1-i    ,npy+j) = mySign*y(j+1  ,npy-1+i)  !NW Corner
                   !   if ((ie+1==npx) .and. (js  ==  1)) x(npx-1+i,1-j  ) = mySign*y(npx-j,1-i    )  !SE Corner
                   !   if ((ie+1==npx) .and. (je+1==npy)) x(npx-1+i,npy+j) =        y(npx-j,npy-1+i)  !NE Corner
                      if ((is  ==  1) .and. (js  ==  1)) x(1-i    ,1-j  ) = mySign*y(1-j  ,i    )  !SW Corner
                      if ((is  ==  1) .and. (je+1==npy)) x(1-i    ,npy+j) =        y(1-j  ,npy-i)  !NW Corner
                      if ((ie+1==npx) .and. (js  ==  1)) x(npx-1+i,1-j  ) =        y(npx+j,i    )  !SE Corner
                      if ((ie+1==npx) .and. (je+1==npy)) x(npx-1+i,npy+j) = mySign*y(npx+j,npy-i)  !NE Corner
                  enddo
               enddo
               do j=1,ng
                  do i=1,ng
                   !  if ((is  ==  1) .and. (js  ==  1)) y(1-i    ,1-j    ) =        x(1-j    ,i+1  )  !SW Corner
                   !  if ((is  ==  1) .and. (je+1==npy)) y(1-i    ,npy-1+j) = mySign*x(1-j    ,npy-i)  !NW Corner
                   !  if ((ie+1==npx) .and. (js  ==  1)) y(npx+i  ,1-j    ) = mySign*x(npx-1+j,i+1  )  !SE Corner
                   !  if ((ie+1==npx) .and. (je+1==npy)) y(npx+i  ,npy-1+j) =        x(npx-1+j,npy-i)  !NE Corner
                     if ((is  ==  1) .and. (js  ==  1)) y(1-i    ,1-j    ) = mySign*x(j      ,1-i  )  !SW Corner
                     if ((is  ==  1) .and. (je+1==npy)) y(1-i    ,npy-1+j) =        x(j      ,npy+i)  !NW Corner
                     if ((ie+1==npx) .and. (js  ==  1)) y(npx+i  ,1-j    ) =        x(npx-j  ,1-i  )  !SE Corner
                     if ((ie+1==npx) .and. (je+1==npy)) y(npx+i  ,npy-1+j) = mySign*x(npx-j  ,npy+i)  !NE Corner
                  enddo
               enddo

      end subroutine fill_corners_dgrid_r8

      subroutine fill_corners_cgrid_r8(x, y, npx, npy, mySign)
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: x
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: y
         integer, intent(IN):: npx,npy
         real(kind=8), intent(IN) :: mySign
         integer :: i,j

                  do j=1,ng
                     do i=1,ng
                        if ((is  ==  1) .and. (js  ==  1)) x(1-i    ,1-j    ) =        y(j      ,1-i  )  !SW Corner
                        if ((is  ==  1) .and. (je+1==npy)) x(1-i    ,npy-1+j) = mySign*y(j      ,npy+i)  !NW Corner
                        if ((ie+1==npx) .and. (js  ==  1)) x(npx+i  ,1-j    ) = mySign*y(npx-j  ,1-i  )  !SE Corner
                        if ((ie+1==npx) .and. (je+1==npy)) x(npx+i  ,npy-1+j) =        y(npx-j  ,npy+i)  !NE Corner
                     enddo
                  enddo
                  do j=1,ng
                     do i=1,ng
                        if ((is  ==  1) .and. (js  ==  1)) y(1-i    ,1-j  ) =        x(1-j  ,i    )  !SW Corner
                        if ((is  ==  1) .and. (je+1==npy)) y(1-i    ,npy+j) = mySign*x(1-j  ,npy-i)  !NW Corner
                        if ((ie+1==npx) .and. (js  ==  1)) y(npx-1+i,1-j  ) = mySign*x(npx+j,i    )  !SE Corner
                        if ((ie+1==npx) .and. (je+1==npy)) y(npx-1+i,npy+j) =        x(npx+j,npy-i)  !NE Corner
                     enddo
                  enddo

      end subroutine fill_corners_cgrid_r8

      subroutine fill_corners_agrid_r8(x, y, npx, npy, mySign)
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: x
         real(kind=8), DIMENSION(isd:,jsd:), intent(INOUT):: y
         integer, intent(IN):: npx,npy
         real(kind=8), intent(IN) :: mySign
         integer :: i,j

                 do j=1,ng
                    do i=1,ng
                       if ((is==    1) .and. (js==    1)) x(1-i    ,1-j    ) = mySign*y(1-j    ,i        )  !SW Corner
                       if ((is==    1) .and. (je==npy-1)) x(1-i    ,npy-1+j) =        y(1-j    ,npy-1-i+1)  !NW Corner
                       if ((ie==npx-1) .and. (js==    1)) x(npx-1+i,1-j    ) =        y(npx-1+j,i        )  !SE Corner
                       if ((ie==npx-1) .and. (je==npy-1)) x(npx-1+i,npy-1+j) = mySign*y(npx-1+j,npy-1-i+1)  !NE Corner
                    enddo
                 enddo
                 do j=1,ng
                    do i=1,ng
                       if ((is==    1) .and. (js==    1)) y(1-j    ,1-i    ) = mySign*x(i        ,1-j    )  !SW Corner
                       if ((is==    1) .and. (je==npy-1)) y(1-j    ,npy-1+i) =        x(i        ,npy-1+j)  !NW Corner
                       if ((ie==npx-1) .and. (js==    1)) y(npx-1+j,1-i    ) =        x(npx-1-i+1,1-j    )  !SE Corner
                       if ((ie==npx-1) .and. (je==npy-1)) y(npx-1+j,npy-1+i) = mySign*x(npx-1-i+1,npy-1+j)  !NE Corner
                    enddo
                 enddo

      end subroutine fill_corners_agrid_r8


end module dsw_extract_mod
