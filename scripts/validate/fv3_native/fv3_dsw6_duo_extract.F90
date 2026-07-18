! Phase-4c duo d_sw6 extract — VERBATIM authoritative symmetryclean
! sw_core.F90:1871-2006 d_sw6 (the final circulation-form wind update
! u = vt + ke - ke(i+1) + fy, v = ut + ke - ke(j+1) - fx, plus the
! damp_v del6 vorticity damping whose diffusive fluxes add to u/v;
! d_con=0 skips the heating block) plus sw_core.F90:36-63 module
! parameters.  del6_vt_flux comes from dsw2_duo_extract_mod (verbatim
! there, byte-identical to sw_core.F90:2008-2121).
! Authoritative-block SHAs recorded by gen_dsw6_duo_oracle.py; the
! oracle test pins the sha256 of THIS extract file (drift guard).
module dsw6_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  use dsw2_duo_extract_mod, only: del6_vt_flux
  implicit none
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


   subroutine d_sw6( delp, u,  v, &
                    heat_source,    &
                    nord_v,  damp_v, &
                    d_con,  gridstruct, flagstruct, bd, ut, vt, ub, vb, ke, wk, fx, fy)

      integer, intent(IN):: nord_v ! vorticity damping
      real   , intent(IN):: d_con
      real,    intent(in):: damp_v
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: delp
      real, intent(INOUT), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: u
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v
      real, intent(INOUT),   dimension(bd%is:bd%ie,bd%js:bd%je):: heat_source

      real,intent(INOUT) :: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real,intent(INOUT) :: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
      real,intent(INOUT) ::   fx(bd%is:bd%ie+1,bd%js:bd%je  )  ! 1-D vorticity X-direction Flux
      real,intent(INOUT) ::   fy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! 1-D vorticity Y-direction Flux
      real,intent(IN) :: ke(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) !  needs this for corner_comm
      real,intent(IN) :: wk(bd%isd:bd%ied,bd%jsd:bd%jed) !  work array

      real,intent(INOUT), dimension(bd%is:bd%ie+1,bd%js:bd%je+1):: ub, vb
      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
!---
      real :: vort(bd%isd:bd%ied,bd%jsd:bd%jed)     ! Vorticity
      real :: gx(bd%is:bd%ie+1,bd%js:bd%je  )
      real :: gy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! work Y-dir flux array

      real :: damp, damp4, u2, v2, du2, dv2
      integer :: i,j
      integer :: is,  ie,  js,  je
      integer :: isd,  ied,  jsd,  jed
      integer :: npx, npy
      logical :: bounded_domain

      real, pointer, dimension(:,:)   :: rsin2, cosa_s
      real, pointer, dimension(:,:) :: rdx, rdy

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je

      isd  = bd%isd
      ied  = bd%ied
      jsd  = bd%jsd
      jed  = bd%jed

      npx      = flagstruct%npx
      npy      = flagstruct%npy
      bounded_domain = gridstruct%bounded_domain

      cosa_s    => gridstruct%cosa_s
      rsin2     => gridstruct%rsin2
      rdx       => gridstruct%rdx
      rdy       => gridstruct%rdy

#ifdef SW_DYNAMICS
      if (test_case > 1) then
#endif

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
   endif

   if ( d_con > 1.e-5 ) then
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
         enddo
      enddo
   endif

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
     endif
#endif

 end subroutine d_sw6

end module dsw6_duo_extract_mod
