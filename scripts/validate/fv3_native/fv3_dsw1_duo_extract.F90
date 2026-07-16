! Phase-4c duo d_sw1 extract — VERBATIM authoritative symmetryclean
! sw_core.F90:500-998 d_sw1 (the duo D-grid transport stage: ut/vt,
! crx/cry/xfx/yfx, ra_x/ra_y, delp/pt/w fv_tp_2d transport, allflux
! outputs for the inter-panel averaging).  Calls the SYMMETRYCLEAN
! fv_tp_2d (tpcore_duo_extract_mod), NOT the phase-4b plain one.
! SHA pinned in the oracle test: 4954a058bf23eb0d4165db383cbe3b953286c5417da461c4b3c7e88890a436c7
module dsw1_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use tpcore_duo_extract_mod, only: fv_tp_2d
  implicit none
contains

   subroutine d_sw1( delp, pt, w, uc,vc, &
                    xflux, yflux, cx, cy,              &
                   crx_adv, cry_adv,  xfx_adv, yfx_adv, q_con,     &
                    nq, q, k, km, inline_q,  &
                   dt, hord_tr, hord_vt, hord_tm, hord_dp,   &
                   nord_v, nord_t, damp_v, &
                   damp_t, hydrostatic, gridstruct, flagstruct, bd, &
                   allflux_x, allflux_y, ra_x, ra_y, ut, vt)

      integer, intent(IN):: hord_tr, hord_vt, hord_tm, hord_dp
      integer, intent(IN):: nord_v ! vorticity damping
      integer, intent(IN):: nord_t ! pt
      integer, intent(IN):: nq, k, km
      real   , intent(IN):: dt
      real,    intent(in):: damp_v, damp_t
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: delp, pt
      real, intent(INOUT), dimension(bd%isd:      ,  bd%jsd:      ):: w, q_con
      real, intent(INOUT), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: vc
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: uc
      real, intent(INOUT):: q(bd%isd:bd%ied,bd%jsd:bd%jed,km,nq)
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
      real,intent(out) ::   allflux_x(bd%is:bd%ie+1,bd%js:bd%je,km,4+nq  )  ! 1-D X-direction Fluxes
      real,intent(out) ::   allflux_y(bd%is:bd%ie  ,bd%js:bd%je+1,km,4+nq)  ! 1-D Y-direction Fluxes
      real,intent(out) :: ra_x(bd%is:bd%ie,bd%jsd:bd%jed)
      real,intent(out) :: ra_y(bd%isd:bd%ied,bd%js:bd%je)
      real,intent(out) :: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real,intent(out) :: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
      logical:: sw_corner, se_corner, ne_corner, nw_corner
      real ::   fx(bd%is:bd%ie+1,bd%js:bd%je  )  ! 1-D X-direction Fluxes
      real ::   fy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! 1-D Y-direction Fluxes
      real :: gx(bd%is:bd%ie+1,bd%js:bd%je  )
      real :: gy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! work Y-dir flux array

      real :: damp
      integer :: i,j, iq
      real, pointer, dimension(:,:) :: area
      real, pointer, dimension(:,:,:) :: sin_sg
      real, pointer, dimension(:,:)   :: cosa_u, cosa_v
      real, pointer, dimension(:,:)   :: sina_u, sina_v
      real, pointer, dimension(:,:)   :: rsin_u, rsin_v

      real, pointer, dimension(:,:) :: dx, dy, rdxa, rdya

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

      area      => gridstruct%area
      sin_sg    => gridstruct%sin_sg
      cosa_u    => gridstruct%cosa_u
      cosa_v    => gridstruct%cosa_v
      sina_u    => gridstruct%sina_u
      sina_v    => gridstruct%sina_v
      rsin_u    => gridstruct%rsin_u
      rsin_v    => gridstruct%rsin_v
      dx        => gridstruct%dx
      dy        => gridstruct%dy
      rdxa      => gridstruct%rdxa
      rdya      => gridstruct%rdya

      sw_corner = gridstruct%sw_corner
      se_corner = gridstruct%se_corner
      nw_corner = gridstruct%nw_corner
      ne_corner = gridstruct%ne_corner

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
        if (bounded_domain .or. (flagstruct%duogrid)) then
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

      if (.not. bounded_domain .or. .not. (flagstruct%duogrid)) then
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

!endif
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


        do j=js,je
           do i=is,ie+1
              allflux_x(i,j,k,1)=fx(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              allflux_y(i,j,k,1)=fy(i,j)
           enddo
        enddo

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
       if ( .not. hydrostatic ) then
            call fv_tp_2d(w, crx_adv,cry_adv, npx, npy, hord_vt, gx, gy, xfx_adv, yfx_adv, &
                          gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, mfx=fx, mfy=fy)

        do j=js,je
           do i=is,ie+1
              allflux_x(i,j,k,2)=gx(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              allflux_y(i,j,k,2)=gy(i,j)
           enddo
        enddo

        endif

#ifdef USE_COND
           call fv_tp_2d(q_con, crx_adv,cry_adv, npx, npy, hord_dp, gx, gy,  &
                xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, mfx=fx, mfy=fy, mass=delp, nord=nord_t, damp_c=damp_t)

        do j=js,je
           do i=is,ie+1
              allflux_x(i,j,k,3)=gx(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              allflux_y(i,j,k,3)=gy(i,j)
           enddo
        enddo

#endif

        call fv_tp_2d(pt, crx_adv,cry_adv, npx, npy, hord_tm, gx, gy,  &
                      xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, &
                      mfx=fx, mfy=fy, mass=delp, nord=nord_v, damp_c=damp_v)

        do j=js,je
           do i=is,ie+1
              allflux_x(i,j,k,4)=gx(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              allflux_y(i,j,k,4)=gy(i,j)
           enddo
        enddo

#endif

     if ( inline_q ) then
        do iq=1,nq
           call fv_tp_2d(q(isd,jsd,k,iq), crx_adv,cry_adv, npx, npy, hord_tr, gx, gy,  &
                         xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac, &
                         mfx=fx, mfy=fy, mass=delp, nord=nord_t, damp_c=damp_t)

        do j=js,je
           do i=is,ie+1
              allflux_x(i,j,k,4+iq)=gx(i,j)
           enddo
        enddo

        do j=js,je+1
           do i=is,ie
              allflux_y(i,j,k,4+iq)=gy(i,j)
           enddo
        enddo

        enddo
     endif

 end subroutine d_sw1

end module dsw1_duo_extract_mod
