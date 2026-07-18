! Phase-4c duo d_sw4 extract — VERBATIM authoritative symmetryclean
! sw_core.F90:1390-1472 d_sw4 (the 4-corner KE fix; its guard
! .not.bounded .or. .not.duogrid is always true on the global cube, so
! the corner formulas fire on the duo lane, reading only duo-written
! ut/vt cells + u/v — all defined, no intent shim needed: every dumped
! array is INTENT(INOUT)).
! Authoritative-block SHA (sha256 of sw_core.F90 lines 1390-1472) is
! recorded by gen_dsw4_duo_oracle.py; the oracle test pins the sha256
! of THIS extract file (drift guard).
module dsw4_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type
  implicit none
contains

   subroutine d_sw4( u,  v, dt, gridstruct, flagstruct, bd, &
                     ut, vt, ke)

      real, intent(IN):: dt
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(INOUT), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: u
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v

      real,intent(INOUT) :: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real,intent(INOUT) :: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)
      real,intent(INOUT) :: ke(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) !  needs this for corner_comm

      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
      logical:: sw_corner, se_corner, ne_corner, nw_corner
!---
      real :: dt6

      integer :: i,j
      integer :: is,  ie,  js,  je
      integer :: isd,  ied,  jsd,  jed
      integer :: npx, npy
      logical :: bounded_domain

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

      sw_corner = gridstruct%sw_corner
      se_corner = gridstruct%se_corner
      nw_corner = gridstruct%nw_corner
      ne_corner = gridstruct%ne_corner

#ifdef SW_DYNAMICS
      if (test_case > 1) then
#endif

!-----------------------------------------
! Fix KE at the 4 corners of the face:
!-----------------------------------------
    if (.not. bounded_domain .or. (.not. flagstruct%duogrid)) then
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

#ifdef SW_DYNAMICS
     endif
#endif

 end subroutine d_sw4

end module dsw4_duo_extract_mod
