! VERBATIM extract — authoritative Zenodo 8327578 symmetryclean.
! Blocks (do not edit bodies):
!   model/dyn_core.F90:2660-2790   geopk
!   model/dyn_core.F90:2073-2132   p_grad_c
!   model/dyn_core.F90:2347-2480   one_grad_p
!   model/a2b_edge.F90:33-43       module parameters (re-declared, see below)
!   model/a2b_edge.F90:332-453     a2b_ord2   (DEAD on this lane, needed at link)
!
! a2b_ord4 is NOT re-extracted here: it is already CERTIFIED inside
! fv3_dsw5_duo_extract.F90's ``a2b_edge_duo_mod`` and is used from there
! (`use a2b_edge_duo_mod, only: a2b_ord4`).  a2b_ord2 is required only so
! the ``a2b_ord==4`` ELSE arms at dyn_core:2401/2411/2460 resolve at link;
! it is never called on this lane (a2b_ord=4).  Rather than MUTATE the
! already-certified extract (fv3_d2a2c_duo_extract.F90:1-13 precedent), it
! goes into a NEW module here, re-declaring the a2b_edge.F90:33-43 module
! parameters verbatim exactly as a2b_edge_duo_mod does.  See UNCERTAIN U8:
! a2b_ord is a namelist value that was not read from the Zenodo run, so
! "a2b_ord2 is dead" is a LANE ASSUMPTION recorded in input_lineage, not a
! proven property of the production configuration.
!
! Module parameters copied from dyn_core: NONE.  geopk / p_grad_c /
! one_grad_p reference no dyn_core `parameter` (`near0` at :135 is used
! only by the beta dispatch in the CALLER, not inside the three bodies).
! Everything else they read is a dummy or one of the three module scalars
! in fv3_geopk_pgrad_shim.F90.  Do NOT copy sw_core.F90:36-63 here.
!
! INTENT SHIMS (the intended deviations; geopk only — p_grad_c and
! one_grad_p need none, every array they touch is already
! intent(in)/intent(inout) upstream):
!   pk   intent(OUT) :2670 -> intent(inout): written only on
!        ifirst..ilast x jfirst..jlast; the CG pass leaves the outer halo
!        UNWRITTEN, so the 1e30 sentinel round-trip is undefined per the
!        standard unless the dummy is inout.
!   gz   intent(OUT) :2670 -> intent(inout): same box restriction.
!   pe   intent(OUT) :2671 -> intent(inout): guarded writes
!        (j>(js-2) .and. j<(je+2); max(ifirst,is-1)..min(ilast,ie+1)).
!   peln intent(out) :2672 -> intent(inout): written only on [is,ie]x[js,je]
!        (and not at all under -DSW_DYNAMICS).
!   pkz  intent(out) :2673 -> intent(inout): ENTIRELY unwritten when
!        CG=.true. (guard at :2781).
! No other body character changes.
!
! q_con stays intent(IN), dimension(bd%isd:,bd%jsd:,1:) (assumed-shape).
! The driver passes a zeroed explicit-shape array; without -DUSE_COND it
! is unreferenced.  It is NOT dropped: the argument position and the
! assumed-shape interface are part of the certificate.
!
! BUILD LANE ASSUMPTION (UNCERTAIN U11): compiled with NEITHER
! -DSW_DYNAMICS NOR -DUSE_COND, hydrostatic=.true., beta<=0, a2b_ord=4,
! duogrid, d_ext>0.  The actual build flags of the Zenodo duo run were
! not read; verify against its build log before citing this fixture as
! production fidelity.
!
! Authoritative-block SHAs recorded by gen_geopk_pgrad_oracle.py; the
! oracle test pins the sha256 of THIS extract file (drift guard) plus a
! per-subroutine block hash.

module a2b_ord2_geopk_mod
  use swcore_shim_mod, only: fv_grid_type, R_GRID
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

! VERBATIM a2b_edge.F90:332-453
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

       if (gridstruct%bounded_domain  .or. (gridstruct%dg%is_initialized) ) then

          do j=js-2,je+1+2
             do i=is-2,ie+1+2
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

end module a2b_ord2_geopk_mod


module geopk_pgrad_extract_mod
  use swcore_shim_mod,  only: fv_grid_type, fv_grid_bounds_type
  use a2b_edge_duo_mod, only: a2b_ord4        ! CERTIFIED (fv3_dsw5_duo_extract.F90)
  use a2b_ord2_geopk_mod, only: a2b_ord2      ! dead on this lane, needed at link
  use geopk_shim_mod,   only: cp_air, ptk, peln1
  implicit none

contains

! VERBATIM dyn_core.F90:2660-2790 (intent shims on pk/gz/pe/peln/pkz — see header)
 subroutine geopk(ptop, pe, peln, delp, pk, gz, hs, pt, q_con, pkz, km, akap, CG, bounded_domain, duogrid, computehalo, npx, npy, a2b_ord, bd)

   integer, intent(IN) :: km, npx, npy, a2b_ord
   real   , intent(IN) :: akap, ptop
   type(fv_grid_bounds_type), intent(IN) :: bd
   real   , intent(IN) :: hs(bd%isd:bd%ied,bd%jsd:bd%jed)
   real, intent(IN), dimension(bd%isd:bd%ied,bd%jsd:bd%jed,km):: pt, delp
   real, intent(IN), dimension(bd%isd:,bd%jsd:,1:):: q_con
   logical, intent(IN) :: CG, bounded_domain, computehalo, duogrid
   ! !OUTPUT PARAMETERS
   real, intent(INOUT), dimension(bd%isd:bd%ied,bd%jsd:bd%jed,km+1):: gz, pk   ! SHIM: intent(OUT) upstream
   real, intent(INOUT) :: pe(bd%is-1:bd%ie+1,km+1,bd%js-1:bd%je+1)             ! SHIM: intent(OUT) upstream
   real, intent(inout) :: peln(bd%is:bd%ie,km+1,bd%js:bd%je)          ! ln(pe) ! SHIM: intent(out) upstream
   real, intent(inout) :: pkz(bd%is:bd%ie,bd%js:bd%je,km)                      ! SHIM: intent(out) upstream
   ! !DESCRIPTION:
   !    Calculates geopotential and pressure to the kappa.
   ! Local:
   real peg(bd%isd:bd%ied,km+1)
   real pkg(bd%isd:bd%ied,km+1)
   real p1d(bd%isd:bd%ied)
   real logp(bd%isd:bd%ied)
   integer i, j, k
   integer ifirst, ilast
   integer jfirst, jlast

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

   if ( (.not. CG .and. a2b_ord==4) .or. ((bounded_domain .or. duogrid) .and. .not. CG) ) then   ! D-Grid
      ifirst = is-2; ilast = ie+2
      jfirst = js-2; jlast = je+2
   else
      ifirst = is-1; ilast = ie+1
      jfirst = js-1; jlast = je+1
   endif

   if ((bounded_domain .or. duogrid) .and. computehalo) then
      if (is == 1)     ifirst = isd
      if (ie == npx-1) ilast  = ied
      if (js == 1)     jfirst = jsd
      if (je == npy-1) jlast  = jed
   end if

!$OMP parallel do default(none) shared(jfirst,jlast,ifirst,ilast,pk,km,gz,hs,ptop,ptk, &
!$OMP                                  js,je,is,ie,peln,peln1,pe,delp,akap,pt,CG,pkz,q_con) &
!$OMP                          private(peg, pkg, p1d, logp)
   do 2000 j=jfirst,jlast

      do i=ifirst, ilast
         p1d(i) = ptop
         pk(i,j,1) = ptk
         gz(i,j,km+1) = hs(i,j)
#ifdef USE_COND
         peg(i,1) = ptop
         pkg(i,1) = ptk
#endif
      enddo

#ifndef SW_DYNAMICS
      if( j>=js .and. j<=je) then
         do i=is,ie
            peln(i,1,j) = peln1
         enddo
      endif
#endif

      if( j>(js-2) .and. j<(je+2) ) then
         do i=max(ifirst,is-1), min(ilast,ie+1)
            pe(i,1,j) = ptop
         enddo
      endif

      ! Top down
      do k=2,km+1
         do i=ifirst, ilast
            p1d(i)  = p1d(i) + delp(i,j,k-1)
            logp(i) = log(p1d(i))
            pk(i,j,k) = exp( akap*logp(i) )
#ifdef USE_COND
            peg(i,k) = peg(i,k-1) + delp(i,j,k-1)*(1.-q_con(i,j,k-1))
            pkg(i,k) = exp( akap*log(peg(i,k)) )
#endif
         enddo

         if( j>(js-2) .and. j<(je+2) ) then
            do i=max(ifirst,is-1), min(ilast,ie+1)
               pe(i,k,j) = p1d(i)
            enddo
            if( j>=js .and. j<=je) then
               do i=is,ie
                  peln(i,k,j) = logp(i)
               enddo
            endif
         endif

      enddo

      ! Bottom up
      do k=km,1,-1
         do i=ifirst, ilast
#ifdef SW_DYNAMICS
            gz(i,j,k) = gz(i,j,k+1) + pt(i,j,k)*(pk(i,j,k+1)-pk(i,j,k))
#else
#ifdef USE_COND
            gz(i,j,k) = gz(i,j,k+1) + cp_air*pt(i,j,k)*(pkg(i,k+1)-pkg(i,k))
#else
            gz(i,j,k) = gz(i,j,k+1) + cp_air*pt(i,j,k)*(pk(i,j,k+1)-pk(i,j,k))
#endif
#endif
         enddo
      enddo

      if ( .not. CG .and. j .ge. js .and. j .le. je ) then
         do k=1,km
            do i=is,ie
               pkz(i,j,k) = (pk(i,j,k+1)-pk(i,j,k))/(akap*(peln(i,k+1,j)-peln(i,k,j)))
            enddo
         enddo
      endif

2000  continue
 end subroutine geopk


! VERBATIM dyn_core.F90:2073-2132 (no intent shims required)
subroutine p_grad_c(dt2, npz, delpc, pkc, gz, uc, vc, bd, rdxc, rdyc, hydrostatic)

integer, intent(in):: npz
real,    intent(in):: dt2
type(fv_grid_bounds_type), intent(IN) :: bd
real, intent(in), dimension(bd%isd:, bd%jsd: ,:  ):: delpc
! pkc is pe**cappa     if hydrostatic
! pkc is full pressure if non-hydrostatic
real, intent(in), dimension(bd%isd:bd%ied, bd%jsd:bd%jed ,npz+1):: pkc, gz
real, intent(inout):: uc(bd%isd:bd%ied+1,bd%jsd:bd%jed  ,npz)
real, intent(inout):: vc(bd%isd:bd%ied  ,bd%jsd:bd%jed+1,npz)
real, intent(IN) :: rdxc(bd%isd:bd%ied+1,bd%jsd:bd%jed+1)
real, intent(IN) :: rdyc(bd%isd:bd%ied  ,bd%jsd:bd%jed)
logical, intent(in):: hydrostatic
! Local:
real:: wk(bd%is-1:bd%ie+1,bd%js-1:bd%je+1)
integer:: i,j,k

integer :: is,  ie,  js,  je

      is  = bd%is
      ie  = bd%ie
      js  = bd%js
      je  = bd%je

!$OMP parallel do default(none) shared(is,ie,js,je,npz,hydrostatic,pkc,delpc,uc,dt2,rdxc,gz,vc,rdyc) &
!$OMP                          private(wk)
do k=1,npz

   if ( hydrostatic ) then
      do j=js-1,je+1
         do i=is-1,ie+1
            wk(i,j) = pkc(i,j,k+1) - pkc(i,j,k)
         enddo
      enddo
   else
      do j=js-1,je+1
         do i=is-1,ie+1
            wk(i,j) = delpc(i,j,k)
         enddo
      enddo
   endif

   do j=js,je
      do i=is,ie+1
         uc(i,j,k) = uc(i,j,k) + dt2*rdxc(i,j) / (wk(i-1,j)+wk(i,j)) *   &
              ( (gz(i-1,j,k+1)-gz(i,j,k  ))*(pkc(i,j,k+1)-pkc(i-1,j,k))  &
              + (gz(i-1,j,k) - gz(i,j,k+1))*(pkc(i-1,j,k+1)-pkc(i,j,k)) )
      enddo
   enddo
   do j=js,je+1
      do i=is,ie
         vc(i,j,k) = vc(i,j,k) + dt2*rdyc(i,j) / (wk(i,j-1)+wk(i,j)) *   &
              ( (gz(i,j-1,k+1)-gz(i,j,k  ))*(pkc(i,j,k+1)-pkc(i,j-1,k))  &
              + (gz(i,j-1,k) - gz(i,j,k+1))*(pkc(i,j-1,k+1)-pkc(i,j,k)) )
      enddo
   enddo
enddo

end subroutine p_grad_c


! VERBATIM dyn_core.F90:2347-2480 (no intent shims required)
subroutine one_grad_p(u, v, pk, gz, divg2, delp, dt, ng, gridstruct, bd, npx, npy, npz,  &
   ptop, hydrostatic, a2b_ord, d_ext)

integer, intent(IN) :: ng, npx, npy, npz, a2b_ord
real,    intent(IN) :: dt, ptop, d_ext
logical, intent(in) :: hydrostatic
type(fv_grid_bounds_type), intent(IN) :: bd
real,    intent(in) :: divg2(bd%is:bd%ie+1,bd%js:bd%je+1)
real, intent(inout) ::    pk(bd%isd:bd%ied,  bd%jsd:bd%jed  ,npz+1)
real, intent(inout) ::    gz(bd%isd:bd%ied,  bd%jsd:bd%jed  ,npz+1)
real, intent(inout) ::  delp(bd%isd:bd%ied,  bd%jsd:bd%jed  ,npz)
real, intent(inout) ::     u(bd%isd:bd%ied  ,bd%jsd:bd%jed+1,npz)
real, intent(inout) ::     v(bd%isd:bd%ied+1,bd%jsd:bd%jed  ,npz)
type(fv_grid_type), intent(INOUT), target :: gridstruct
! Local:
real, dimension(bd%isd:bd%ied,bd%jsd:bd%jed):: wk
real:: wk1(bd%is:bd%ie+1,bd%js:bd%je+1)
real:: wk2(bd%is:bd%ie,bd%js:bd%je+1)
real top_value
integer i,j,k

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

if ( hydrostatic ) then
   ! pk is pe**kappa if hydrostatic
   top_value = ptk
else
   ! pk is full pressure if non-hydrostatic
   top_value = ptop
endif

!$OMP parallel do default(none) shared(is,ie,js,je,pk,top_value)
do j=js,je+1
   do i=is,ie+1
      pk(i,j,1) = top_value
   enddo
enddo

!$OMP parallel do default(none) shared(npz,isd,jsd,pk,gridstruct,npx,npy,is,ie,js,je,ng,a2b_ord) &
!$OMP                          private(wk)
do k=2,npz+1
   if ( a2b_ord==4 ) then
      call a2b_ord4(pk(isd,jsd,k), wk, gridstruct, npx, npy, is, ie, js, je, ng, .true.)
   else
      call a2b_ord2(pk(isd,jsd,k), wk, gridstruct, npx, npy, is, ie, js, je, ng, .true.)
   endif
enddo

!$OMP parallel do default(none) shared(npz,isd,jsd,gz,gridstruct,npx,npy,is,ie,js,je,ng,a2b_ord) &
!$OMP                          private(wk)
do k=1,npz+1
   if ( a2b_ord==4 ) then
      call a2b_ord4( gz(isd,jsd,k), wk, gridstruct, npx, npy, is, ie, js, je, ng, .true.)
   else
      call a2b_ord2( gz(isd,jsd,k), wk, gridstruct, npx, npy, is, ie, js, je, ng, .true.)
   endif
enddo

if ( d_ext > 0. ) then

   !$OMP parallel do default(none) shared(is,ie,js,je,wk2,divg2)
   do j=js,je+1
      do i=is,ie
         wk2(i,j) = divg2(i,j)-divg2(i+1,j)
      enddo
   enddo

   !$OMP parallel do default(none) shared(is,ie,js,je,wk1,divg2)
   do j=js,je
      do i=is,ie+1
         wk1(i,j) = divg2(i,j)-divg2(i,j+1)
      enddo
   enddo

else

   !$OMP parallel do default(none) shared(is,ie,js,je,wk1,wk2)
   do j=js,je+1
      do i=is,ie
         wk2(i,j) = 0.
      enddo
      do i=is,ie+1
         wk1(i,j) = 0.
      enddo
   enddo

endif

!$OMP parallel do default(none) shared(is,ie,js,je,npz,pk,delp,hydrostatic,a2b_ord,gridstruct, &
!$OMP                                  npx,npy,isd,jsd,ng,u,v,wk2,dt,gz,wk1) &
!$OMP                          private(wk)
do k=1,npz

   if ( hydrostatic ) then
      do j=js,je+1
         do i=is,ie+1
            wk(i,j) = pk(i,j,k+1) - pk(i,j,k)
         enddo
      enddo
   else
      if ( a2b_ord==4 ) then
         call a2b_ord4(delp(isd,jsd,k), wk, gridstruct, npx, npy, is, ie, js, je, ng)
      else
         call a2b_ord2(delp(isd,jsd,k), wk, gridstruct, npx, npy, is, ie, js, je, ng)
      endif
   endif

   do j=js,je+1
      do i=is,ie
         u(i,j,k) = gridstruct%rdx(i,j)*(wk2(i,j)+u(i,j,k) + dt/(wk(i,j)+wk(i+1,j)) * &
                                 ((gz(i,j,k+1)-gz(i+1,j,k))*(pk(i+1,j,k+1)-pk(i,j,k)) &
                                + (gz(i,j,k)-gz(i+1,j,k+1))*(pk(i,j,k+1)-pk(i+1,j,k))))
      enddo
   enddo
   do j=js,je
      do i=is,ie+1
         v(i,j,k) = gridstruct%rdy(i,j)*(wk1(i,j)+v(i,j,k) + dt/(wk(i,j)+wk(i,j+1)) * &
                                 ((gz(i,j,k+1)-gz(i,j+1,k))*(pk(i,j+1,k+1)-pk(i,j,k)) &
                                + (gz(i,j,k)-gz(i,j+1,k+1))*(pk(i,j,k+1)-pk(i,j+1,k))))
      enddo
   enddo
enddo    ! end k-loop

end subroutine one_grad_p

end module geopk_pgrad_extract_mod
