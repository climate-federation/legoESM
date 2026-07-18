! Phase-4c duo d_sw5 extract — VERBATIM authoritative symmetryclean:
!   a2b_edge.F90:50-330 a2b_ord4 + 455-465 extrap_corner (the
!     symmetryclean a2b carries dg%is_initialized gates at 98/185/241 —
!     duo takes the interior branch, no edge_w/e/s/n reconstruction)
!   sw_core.F90:1474-1869 d_sw5 (vorticity prep + divergence damping +
!     Smagorinsky + KE increment + vorticity-flux transport)
!   sw_core.F90:2451-2537 smag_corner (dead on the grid_type<3 lane,
!     needed at link)
!   sw_core.F90:36-63 module parameters (verbatim)
! INTENT SHIMS (documented deviations, same rationale as d_sw1/d_sw2):
!   crx_adv/cry_adv/xfx_adv/yfx_adv are declared intent(OUT) upstream
!   yet only READ (fv_tp_2d consumes them; dyn_core relies on the
!   caller-retained d_sw1 values — undefined-on-entry per the standard)
!   -> intent(inout); ptc is intent(OUT) but UNWRITTEN on the nord>0
!   branch -> intent(inout) so the 1e30 sentinel round-trip is defined.
! Authoritative-block SHAs recorded by gen_dsw5_duo_oracle.py; the
! oracle test pins the sha256 of THIS extract file (drift guard).
module a2b_edge_duo_mod
  use swcore_shim_mod, only: great_circle_dist
  use swcore_shim_mod, only: fv_grid_type, fv_grid_bounds_type, R_GRID
  implicit none
! VERBATIM a2b_edge.F90:33-43 module parameters:
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
    if (gridstruct%bounded_domain  .or. (gridstruct%dg%is_initialized)) then

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

    if (gridstruct%bounded_domain   .or. (gridstruct%dg%is_initialized) ) then


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

    if (gridstruct%bounded_domain   .or. (gridstruct%dg%is_initialized) ) then

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

  real function extrap_corner ( p0, p1, p2, q1, q2 )
    real, intent(in ), dimension(2):: p0, p1, p2
    real, intent(in ):: q1, q2
    real:: x1, x2

    x1 = great_circle_dist( real(p1,kind=R_GRID), real(p0,kind=R_GRID) )
    x2 = great_circle_dist( real(p2,kind=R_GRID), real(p0,kind=R_GRID) )

    extrap_corner = q1 + x1/(x2-x1) * (q1-q2)

  end function extrap_corner

end module a2b_edge_duo_mod

module dsw5_duo_extract_mod
  use swcore_shim_mod, only: fv_grid_bounds_type, fv_grid_type, &
                             fv_flags_type, XDir, YDir
  use tpcore_duo_extract_mod, only: fv_tp_2d
  use a2b_edge_duo_mod, only: a2b_ord4
  implicit none
  ! fill_corners is DEAD on the oracle lane (nord=1 -> nt=0 makes
  ! fill_c false, and duo excludes it anyway); these stubs satisfy the
  ! link and fail LOUD if a config change ever reaches them.
  interface fill_corners
     module procedure fill_corners_2d_stub
     module procedure fill_corners_vector_stub
  end interface fill_corners
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

   subroutine d_sw5(delpc, delp,  ptc, u,  v, w, uc,vc, &
                   ua, va, divg_d,              &
                   crx_adv, cry_adv,  xfx_adv, yfx_adv, q_con, z_rat,     &
                   dt, hord_vt, nord,   &
                    dddmp, d2_bg, d4_bg, damp_w, &
                    d_con, hydrostatic, gridstruct, flagstruct, bd,  &
                   dw,ra_x,ra_y,ut,vt,ub,vb,ke,wk,vortfluxx,vortfluxy)

      integer, intent(IN):: hord_vt
      integer, intent(IN):: nord   ! nord=1 divergence damping; (del-4) or 3 (del-8)
      real   , intent(IN):: dt, dddmp, d2_bg, d4_bg, d_con
      real,    intent(in)::  damp_w
      type(fv_grid_bounds_type), intent(IN) :: bd
      real, intent(inout):: divg_d(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) ! divergence
      real, intent(IN), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: z_rat
      real, intent(INOUT), dimension(bd%isd:bd%ied,  bd%jsd:bd%jed):: delp, ua, va
      real, intent(INOUT), dimension(bd%isd:      ,  bd%jsd:      ):: w, q_con
      real, intent(INOUT), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1):: u, vc
      real, intent(INOUT), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v, uc
! SHIM DEVIATION: ptc is intent(OUT) upstream but UNWRITTEN on the
! nord>0 branch — intent(inout) makes the 1e30 sentinel round-trip
! defined.  delpc rides along (fully written on the compute B-ring).
      real, intent(inout),   dimension(bd%isd:bd%ied,  bd%jsd:bd%jed)  :: delpc, ptc

      logical, intent(IN):: hydrostatic
! SHIM DEVIATION: upstream declares these intent(OUT) yet d_sw5 only
! READS them (fv_tp_2d inputs; dyn_core relies on the caller-retained
! d_sw1 values — undefined on entry per the standard).  intent(inout)
! makes the pass-through defined semantics.
      real, intent(inout), dimension(bd%is:bd%ie+1,bd%jsd:bd%jed):: crx_adv, xfx_adv
      real, intent(inout), dimension(bd%isd:bd%ied,bd%js:bd%je+1):: cry_adv, yfx_adv

      real,intent(out) ::   vortfluxx(bd%is:bd%ie+1,bd%js:bd%je  )  ! 1-D X-direction Fluxes
      real,intent(out) ::   vortfluxy(bd%is:bd%ie  ,bd%js:bd%je+1)  ! 1-D Y-direction Fluxes

      real,intent(INOUT) :: ut(bd%isd:bd%ied+1,bd%jsd:bd%jed)
      real,intent(INOUT) :: vt(bd%isd:bd%ied,  bd%jsd:bd%jed+1)

      real, intent(IN) :: dw(bd%is:bd%ie,bd%js:bd%je) !  w dammping from dsw2
      real,intent(IN) :: ra_x(bd%is:bd%ie,bd%jsd:bd%jed)
      real,intent(IN) :: ra_y(bd%isd:bd%ied,bd%js:bd%je)
      real,intent(INOUT) :: ke(bd%isd:bd%ied+1,bd%jsd:bd%jed+1) !  needs this for corner_comm
      real,intent(out) :: wk(bd%isd:bd%ied,bd%jsd:bd%jed) !  work array

      real,intent(INOUT), dimension(bd%is:bd%ie+1,bd%js:bd%je+1):: ub, vb
      type(fv_grid_type), intent(IN), target :: gridstruct
      type(fv_flags_type), intent(IN), target :: flagstruct
! Local:
      logical:: sw_corner, se_corner, ne_corner, nw_corner
!---
      logical :: fill_c
      real ::   vort(bd%isd:bd%ied  ,bd%jsd:bd%jed)  ! 1-D Y-direction Fluxes


      real :: damp, damp2, dd8
      integer :: i,j, is2, ie1, js2, je1, n, nt, n2

      real, pointer, dimension(:,:) :: rarea
      real, pointer, dimension(:,:,:) :: sin_sg
      real, pointer, dimension(:,:)   :: cosa_u, cosa_v
      real, pointer, dimension(:,:)   :: sina_u, sina_v
      real, pointer, dimension(:,:)   :: f0, divg_u, divg_v
      real, pointer, dimension(:,:) :: dx, dy, dxc, dyc

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

      rarea     => gridstruct%rarea
      sin_sg    => gridstruct%sin_sg
      cosa_u    => gridstruct%cosa_u
      cosa_v    => gridstruct%cosa_v
      sina_u    => gridstruct%sina_u
      sina_v    => gridstruct%sina_v
      f0        => gridstruct%f0
      divg_u    => gridstruct%divg_u
      divg_v    => gridstruct%divg_v
      dx        => gridstruct%dx
      dy        => gridstruct%dy
      dxc       => gridstruct%dxc
      dyc       => gridstruct%dyc

      sw_corner = gridstruct%sw_corner
      se_corner = gridstruct%se_corner
      nw_corner = gridstruct%nw_corner
      ne_corner = gridstruct%ne_corner

      if (bounded_domain .or. flagstruct%duogrid) then
         is2 = is;        ie1 = ie+1
         js2 = js;        je1 = je+1
      else
         is2 = max(2,is); ie1 = min(npx-1,ie+1)
         js2 = max(2,js); je1 = min(npy-1,je+1)
      end if

#ifdef SW_DYNAMICS
      if (test_case > 1) then
#endif
!
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
#ifdef USE_COND
     do j=js,je
        do i=is,ie
           q_con(i,j) = q_con(i,j)/delp(i,j)
        enddo
     enddo
#endif

!-----------------------------
! Compute divergence damping
!-----------------------------
!  damp = dddmp * da_min_c

   if ( nord==0 ) then
!         area ~ dxb*dyb*sin(alpha)

      if (bounded_domain .or. (flagstruct%duogrid)) then

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

if (.not. (flagstruct%duogrid)) then
! Remove the extra term at the corners:
      if (sw_corner) delpc(1,    1) = delpc(1,    1) - vort(1,    0)
      if (se_corner) delpc(npx,  1) = delpc(npx,  1) - vort(npx,  0)
      if (ne_corner) delpc(npx,npy) = delpc(npx,npy) + vort(npx,npy)
      if (nw_corner) delpc(1,  npy) = delpc(1,  npy) + vort(1,  npy)
endif

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
                  .and. .not. (bounded_domain .or. flagstruct%duogrid)

!recheck what to do here and below. if we need to use fill_corner_region or lag_interp
!seems the original fill_corners give lower errors
        if ( fill_c ) call fill_corners(divg_d, npx, npy, FILL=XDir, BGRID=.true.)
!if ((flagstruct%duogrid))  call fill_corner_region(divg_d, bd, gridstruct%dg, 1,1)
        do j=js-nt,je+1+nt
           do i=is-1-nt,ie+1+nt
              vc(i,j) = (divg_d(i+1,j)-divg_d(i,j))*divg_u(i,j)
           enddo
        enddo

        if ( fill_c ) call fill_corners(divg_d, npx, npy, FILL=YDir, BGRID=.true.)
!if ( (flagstruct%duogrid))  call fill_corner_region(divg_d, bd, gridstruct%dg, 1,1)
        do j=js-1-nt,je+1+nt
           do i=is-nt,ie+1+nt
              uc(i,j) = (divg_d(i,j+1)-divg_d(i,j))*divg_v(i,j)
           enddo
        enddo

        if ( fill_c ) call fill_corners(vc, uc, npx, npy, VECTOR=.true., DGRID=.true.)
!if ( (flagstruct%duogrid))  call fill_corner_region(uc, bd, gridstruct%dg, 1,0)
!if ( (flagstruct%duogrid))  call fill_corner_region(vc, bd, gridstruct%dg, 0,1)
        do j=js-nt,je+1+nt
           do i=is-nt,ie+1+nt
              divg_d(i,j) = uc(i,j-1) - uc(i,j) + vc(i-1,j) - vc(i,j)
           enddo
        enddo

if (.not. (flagstruct%duogrid)) then
! Remove the extra term at the corners:
        if (sw_corner) divg_d(1,    1) = divg_d(1,    1) - uc(1,    0)
        if (se_corner) divg_d(npx,  1) = divg_d(npx,  1) - uc(npx,  0)
        if (ne_corner) divg_d(npx,npy) = divg_d(npx,npy) + uc(npx,npy)
        if (nw_corner) divg_d(1,  npy) = divg_d(1,  npy) + uc(1,  npy)
endif

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
          call smag_corner(abs(dt), u, v, vort, bd, npx, npy, gridstruct, ng)
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

   if ( d_con > 1.e-5 ) then
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

   ! This vort is not needed as 'out' but the fluxes are instead
    call fv_tp_2d(vort, crx_adv, cry_adv, npx, npy, hord_vt, vortfluxx, vortfluxy, &
                  xfx_adv,yfx_adv, gridstruct, bd, ra_x, ra_y, flagstruct%lim_fac)
!

#ifdef SW_DYNAMICS
     endif
#endif

 end subroutine d_sw5

 subroutine smag_corner(dt, u, v, smag_c, bd, npx, npy, gridstruct, ng)
! Compute the Tension_Shear strain at cell corners for Smagorinsky diffusion
!!!  work only if (grid_type==4)
 type(fv_grid_bounds_type), intent(IN) :: bd
 real, intent(in):: dt
 integer, intent(IN) :: npx, npy, ng
 real, intent(in),  dimension(bd%isd:bd%ied,  bd%jsd:bd%jed+1):: u
 real, intent(in),  dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ):: v
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

  subroutine fill_corners_2d_stub(q, npx, npy, FILL, AGRID, BGRID)
    real, intent(inout) :: q(:, :)
    integer, intent(in) :: npx, npy, FILL
    logical, optional, intent(in) :: AGRID, BGRID
    stop 'fill_corners: dead on the nord=1 duo oracle lane'
  end subroutine fill_corners_2d_stub

  subroutine fill_corners_vector_stub(x, y, npx, npy, VECTOR, DGRID)
    real, intent(inout) :: x(:, :), y(:, :)
    integer, intent(in) :: npx, npy
    logical, optional, intent(in) :: VECTOR, DGRID
    stop 'fill_corners: dead on the nord=1 duo oracle lane'
  end subroutine fill_corners_vector_stub

end module dsw5_duo_extract_mod
