! Phase-4c duo d_sw3 extract — VERBATIM authoritative symmetryclean
! sw_core.F90:1201-1388 d_sw3 (the KE-flux stage: B-grid contravariant
! vb/ub + the ytp_v/xtp_u advective fluxes + the ubbtemp/vbbtemp/ubb/vbb
! outputs consumed by d_sw5's KE assembly) plus sw_core.F90:2540-3353
! xtp_u + ytp_v VERBATIM (the symmetryclean variants carry
! gridstruct%dg%is_initialized duo gates absent from the plain tree —
! note d_sw3 passes bounded_domain=.false. to both as a LITERAL, an
! upstream quirk preserved here).
! Authoritative-block SHAs (sha256 of the sw_core.F90 line ranges) are
! recorded by gen_dsw3_duo_oracle.py; the oracle test pins the sha256
! of THIS extract file (drift guard).
module dsw3_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use tpcore_duo_extract_mod, only: pert_ppm
  implicit none
! VERBATIM sw_core.F90:36-63 module parameters (xtp_u/ytp_v consume
! s11/s14/s15, t11..t15, c1..c3, p1/p2, a1/a2, r3, near_zero,
! big_number; no OVERLOAD_R4 -> big_number = 1.E30):
  real, parameter:: r3 = 1./3.
  real, parameter:: t11=27./28., t12=-13./28., t13=3./7., t14=6./7., t15=3./28.
  real, parameter:: s11=11./14., s13=-13./14., s14=4./7., s15=3./14.
  real, parameter:: near_zero = 1.E-9     ! for KE limiter
#ifdef OVERLOAD_R4
  real, parameter:: big_number = 1.E8
#else
  real, parameter:: big_number = 1.E30
#endif
!----------------------
! PPM volume mean form:
!----------------------
  real, parameter:: p1 =  7./12.     ! 0.58333333
  real, parameter:: p2 = -1./12.
!----------------------------
! 4-pt Lagrange interpolation
!----------------------------
  real, parameter:: a1 =  0.5625
  real, parameter:: a2 = -0.0625
!----------------------------------------------
! volume-conserving cubic with 2nd drv=0 at end point:
  real, parameter:: c1 = -2./14.
  real, parameter:: c2 = 11./14.
  real, parameter:: c3 =  5./14.
! 3-pt off-center intp formular:
! real, parameter:: c1 = -0.125
! real, parameter:: c2 =  0.75
! real, parameter:: c3 =  0.375
contains

   subroutine d_sw3(u, v, uc, vc, &
                   dt, hord_mt,   &
                   gridstruct, flagstruct, bd, ut, vt, ubbtemp, vbbtemp, ubb, vbb)

      integer, intent(IN):: hord_mt
      real   , intent(IN):: dt
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: u, vc
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v, uc


      real,intent(IN) :: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real,intent(IN) :: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)

      real,intent(INOUT), dimension(bd%is:bd%ie+1,bd%js:bd%je+1):: ubbtemp,vbbtemp, ubb, vbb !out for the next k
      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
      real, dimension(bd%is:bd%ie+1,bd%js:bd%je+1):: ub, vb

      real :: dt4, dt5
      integer :: i,j, is2, ie1, js2, je1

      real, pointer, dimension(:,:) :: rsina
      real, pointer, dimension(:,:) ::  cosa

      integer :: is,  ie,  js,  je
      integer :: isd, ied, jsd, jed
      integer :: npx, npy
      logical :: bounded_domain

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je
      isd = bd%isd
      ied = bd%ied
      jsd = bd%jsd
      jed = bd%jed

      npx      = flagstruct%npx
      npy      = flagstruct%npy
      bounded_domain = gridstruct%bounded_domain

      rsina     => gridstruct%rsina
      cosa      => gridstruct%cosa

#ifdef SW_DYNAMICS
      if (test_case > 1) then
#endif

!----------------------
! Kinetic Energy Fluxes
!----------------------
! Compute B grid contra-variant components for KE:

      dt5 = 0.5 *dt
      dt4 = 0.25*dt

      if (bounded_domain .or. (flagstruct%duogrid)) then
         is2 = is;        ie1 = ie+1
         js2 = js;        je1 = je+1
      else
         is2 = max(2,is); ie1 = min(npx-1,ie+1)
         js2 = max(2,js); je1 = min(npy-1,je+1)
      end if

      if (flagstruct%grid_type < 3) then

         if (bounded_domain .or. (flagstruct%duogrid)) then
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

      !call ytp_v(is,ie,js,je,isd,ied,jsd,jed, vb, u, v, ub, hord_mt, gridstruct%dy, gridstruct%rdy, &
      !           npx, npy, flagstruct%grid_type, gridstruct, bounded_domain, flagstruct%lim_fac)
      call ytp_v(is,ie,js,je,isd,ied,jsd,jed, vb, u, v, ub, hord_mt, gridstruct%dy, gridstruct%rdy, &
                 npx, npy, flagstruct%grid_type, gridstruct, .false., flagstruct%lim_fac)

         do j=js,je+1
            do i=is,ie+1
             ubbtemp(i,j)=ub(i,j)
             vbbtemp(i,j)=vb(i,j)
            enddo
         enddo

      if (flagstruct%grid_type < 3) then

         if (bounded_domain .or. flagstruct%duogrid) then

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

      !call xtp_u(is,ie,js,je, isd,ied,jsd,jed, ub, u, v, vb, hord_mt, gridstruct%dx, gridstruct%rdx, &
      !           npx, npy, flagstruct%grid_type, gridstruct, bounded_domain, flagstruct%lim_fac)
      call xtp_u(is,ie,js,je, isd,ied,jsd,jed, ub, u, v, vb, hord_mt, gridstruct%dx, gridstruct%rdx, &
                 npx, npy, flagstruct%grid_type, gridstruct, .false., flagstruct%lim_fac)

do j=js,je+1
   do i=is,ie+1
    ubb(i,j)=ub(i,j)
    vbb(i,j)=vb(i,j)
   enddo
enddo
!

#ifdef SW_DYNAMICS
      endif ! test_case
#endif

 end subroutine d_sw3

 subroutine xtp_u(is,ie,js,je,isd,ied,jsd,jed,c, u, v, flux, iord, dx, rdx, npx, npy, grid_type, gridstruct, bounded_domain, lim_fac)

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
 type(fv_grid_type), intent(IN), target :: gridstruct
! Local
 real, dimension(is-1:ie+1):: bl, br, b0
 logical, dimension(is-1:ie+1):: smt5, smt6
 logical, dimension(is:ie+1):: hi5, hi6
 real:: fx0(is:ie+1)
 real al(is-1:ie+2), dm(is-2:ie+2)
 real dq(is-3:ie+2)
 real xt, pmp, lac, cfl
 real pmp_1, lac_1, pmp_2, lac_2
 real x0, x1, x0L, x0R
 integer i, j
 integer is3, ie3

 if ( bounded_domain .or. grid_type>3 .or. gridstruct%dg%is_initialized ) then
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

      if ( (.not.bounded_domain .or. gridstruct%dg%is_initialized) .and. grid_type < 3) then
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
           if ( is==1 .and. (.not. bounded_domain .or. .not. gridstruct%dg%is_initialized) ) then
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

           if ( (ie+1)==npx  .and. (.not. bounded_domain .or. gridstruct%dg%is_initialized)) then
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


 subroutine ytp_v(is,ie,js,je,isd,ied,jsd,jed, c, u, v, flux, jord, dy, rdy, npx, npy, grid_type, gridstruct, bounded_domain, lim_fac)
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
 type(fv_grid_type), intent(IN), target :: gridstruct !target?
! Local:
 logical, dimension(is:ie+1,js-1:je+1):: smt5, smt6
 logical, dimension(is:ie+1):: hi5, hi6
 real:: fx0(is:ie+1)
 real dm(is:ie+1,js-2:je+2)
 real al(is:ie+1,js-1:je+2)
 real, dimension(is:ie+1,js-1:je+1):: bl, br, b0
 real dq(is:ie+1,js-3:je+2)
 real xt, pmp, lac, cfl
 real pmp_1, lac_1, pmp_2, lac_2
 real x0, x1, x0R, x0L
 integer i, j, js3, je3

 if ( bounded_domain .or. grid_type>3 .or. gridstruct%dg%is_initialized ) then
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

   if ( (.not.bounded_domain .or. .not. gridstruct%dg%is_initialized) .and. grid_type < 3) then
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
      if( js==1 .and. (.not. bounded_domain .or. .not. gridstruct%dg%is_initialized)) then
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
      if( (je+1)==npy  .and. (.not. bounded_domain .or. .not. gridstruct%dg%is_initialized)) then
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

end module dsw3_duo_extract_mod
