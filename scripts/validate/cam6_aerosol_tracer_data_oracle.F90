! CAM6 prescribed-aerosol mapping oracle (tracer_data lininterp path), for
! tests/atmosphere/hydrostatic/unit/test_cam6_aerosol_climatology_oracle.py.
! VERBATIM: interpolate_data.F90 lininterp_init (52-303), lininterp2d1d (400-437), lininterp3d2d (438-476);
! intp_util.F90 findplb (13-51); tracer_data.F90 vert_interp (2291-2361), find_times cyclical bracket
! (1070-1077) and the interpolate_trcdata fact1/fact2 block (1631-1648), datain/ps sums (1666-1670, 1724-1728).
! Shim: endrun stops with status 2; pcols/pver are module variables set from the case.
! Build: gfortran -O0 -ffree-line-length-none -fno-fast-math -J <scratch> cam6_aerosol_tracer_data_oracle.F90 -o o
!        ./o < cam6_aerosol_tracer_data_oracle_cases.txt > cam6_aerosol_tracer_data_oracle_expected.txt
module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod

module oracle_shim
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: iulog = 6
  integer :: pcols, pver
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'ENDRUN ', msg
    stop 2
  end subroutine endrun
end module oracle_shim

module interpolate_data
  use oracle_shim
  implicit none
  type interp_type
     real(r8), pointer :: wgts(:)
     real(r8), pointer :: wgtn(:)
     integer, pointer  :: jjm(:)
     integer, pointer  :: jjp(:)
  end type interp_type
  integer, parameter :: extrap_method_zero = 0
  integer, parameter :: extrap_method_bndry = 1
  integer, parameter :: extrap_method_cycle = 2
contains
  subroutine lininterp_init(yin, nin, yout, nout, extrap_method, interp_wgts, &
       cyclicmin, cyclicmax)
!
! Description:
!   Initialize a variable of type(interp_type) with weights for linear interpolation.
!       this variable can then be used in calls to lininterp1d and lininterp2d.
!   yin is a 1d array of length nin of locations to interpolate from - this array must
!       be monotonic but can be increasing or decreasing
!   yout is a 1d array of length nout of locations to interpolate to, this array need
!       not be ordered
!   extrap_method determines how to handle yout points beyond the bounds of yin
!       if 0 set values outside output grid to 0
!       if 1 set to boundary value
!       if 2 set to cyclic boundaries
!         optional values cyclicmin and cyclicmax can be used to set the bounds of the
!         cyclic mapping - these default to 0 and 360.
!

    integer, intent(in) :: nin
    integer, intent(in) :: nout
    real(r8), intent(in) :: yin(:)           ! input mesh
    real(r8), intent(in) :: yout(:)         ! output mesh
    integer, intent(in) :: extrap_method       ! if 0 set values outside output grid to 0
                                               ! if 1 set to boundary value
                                               ! if 2 set to cyclic boundaries
    real(r8), intent(in), optional :: cyclicmin, cyclicmax

    type (interp_type), intent(out) :: interp_wgts
    real(r8) :: cmin, cmax
    real(r8) :: extrap
    real(r8) :: dyinwrap
    real(r8) :: ratio
    real(r8) :: avgdyin
    integer :: i, j, icount
    integer :: jj
    real(r8), pointer :: wgts(:)
    real(r8), pointer :: wgtn(:)
    integer, pointer :: jjm(:)
    integer, pointer :: jjp(:)
    logical :: increasing
    !
    ! Check validity of input coordinate arrays: must be monotonically increasing,
    ! and have a total of at least 2 elements
    !
    if (nin.lt.2) then
       call endrun('LININTERP: Must have at least 2 input points for interpolation')
    end if
    if(present(cyclicmin)) then
       cmin=cyclicmin
    else
       cmin=0_r8
    end if
    if(present(cyclicmax)) then
       cmax=cyclicmax
    else
       cmax=360_r8
    end if
    if(cmax<=cmin) then
       call endrun('LININTERP: cyclic min value must be < max value')
    end if
    increasing=.true.
    icount = 0
    do j=1,nin-1
       if (yin(j).gt.yin(j+1)) icount = icount + 1
    end do
    if(icount.eq.nin-1) then
       increasing = .false.
       icount=0
    endif
    if (icount.gt.0) then
       call endrun('LININTERP: Non-monotonic input coordinate array found')
    end if
    allocate(interp_wgts%jjm(nout), &
         interp_wgts%jjp(nout), &
         interp_wgts%wgts(nout), &
         interp_wgts%wgtn(nout))

    jjm => interp_wgts%jjm
    jjp => interp_wgts%jjp
    wgts =>  interp_wgts%wgts
    wgtn =>  interp_wgts%wgtn

    !
    ! Initialize index arrays for later checking
    !
    jjm = 0
    jjp = 0

    extrap = 0._r8
    select case (extrap_method)
    case (extrap_method_zero)
       !
       ! For values which extend beyond boundaries, set weights
       ! such that values will be 0.
       !
       do j=1,nout
          if(increasing) then
             if (yout(j).lt.yin(1)) then
                jjm(j) = 1
                jjp(j) = 1
                wgts(j) = 0._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             else if (yout(j).gt.yin(nin)) then
                jjm(j) = nin
                jjp(j) = nin
                wgts(j) = 0._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             end if
          else
             if (yout(j).gt.yin(1)) then
                jjm(j) = 1
                jjp(j) = 1
                wgts(j) = 0._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             else if (yout(j).lt.yin(nin)) then
                jjm(j) = nin
                jjp(j) = nin
                wgts(j) = 0._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             end if
          end if
       end do
    case (extrap_method_bndry)
       !
       ! For values which extend beyond boundaries, set weights
       ! such that values will just be copied.
       !
       do j=1,nout
          if(increasing) then
             if (yout(j).le.yin(1)) then
                jjm(j) = 1
                jjp(j) = 1
                wgts(j) = 1._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             else if (yout(j).gt.yin(nin)) then
                jjm(j) = nin
                jjp(j) = nin
                wgts(j) = 1._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             end if
          else
             if (yout(j).gt.yin(1)) then
                jjm(j) = 1
                jjp(j) = 1
                wgts(j) = 1._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             else if (yout(j).le.yin(nin)) then
                jjm(j) = nin
                jjp(j) = nin
                wgts(j) = 1._r8
                wgtn(j) = 0._r8
                extrap = extrap + 1._r8
             end if
          end if
       end do
    case (extrap_method_cycle)
       !
       ! For values which extend beyond boundaries, set weights
       ! for circular boundaries
       !
       dyinwrap = yin(1) + (cmax-cmin) - yin(nin)
       avgdyin = abs(yin(nin)-yin(1))/(nin-1._r8)
       ratio = dyinwrap/avgdyin
       if (ratio < 0.9_r8 .or. ratio > 1.1_r8) then
          write(iulog,*) 'Lininterp: Bad dyinwrap value =',dyinwrap,&
               ' avg=', avgdyin, yin(1),yin(nin)
          call endrun('interpolate_data')
       end if

       do j=1,nout
          if(increasing) then
             if (yout(j) <= yin(1)) then
                jjm(j) = nin
                jjp(j) = 1
                wgts(j) = (yin(1)-yout(j))/dyinwrap
                wgtn(j) = (yout(j)+(cmax-cmin) - yin(nin))/dyinwrap
             else if (yout(j) > yin(nin)) then
                jjm(j) = nin
                jjp(j) = 1
                wgts(j) = (yin(1)+(cmax-cmin)-yout(j))/dyinwrap
                wgtn(j) = (yout(j)-yin(nin))/dyinwrap
             end if
          else
             if (yout(j) > yin(1)) then
                jjm(j) = nin
                jjp(j) = 1
                wgts(j) = (yin(1)-yout(j))/dyinwrap
                wgtn(j) = (yout(j)+(cmax-cmin) - yin(nin))/dyinwrap
             else if (yout(j) <= yin(nin)) then
                jjm(j) = nin
                jjp(j) = 1
                wgts(j) = (yin(1)+(cmax-cmin)-yout(j))/dyinwrap
                wgtn(j) = (yout(j)+(cmax-cmin)-yin(nin))/dyinwrap
             end if

          endif
       end do
    end select

    !
    ! Loop though output indices finding input indices and weights
    !
    if(increasing) then
       do j=1,nout
          do jj=1,nin-1
             if (yout(j).gt.yin(jj) .and. yout(j).le.yin(jj+1)) then
                jjm(j) = jj
                jjp(j) = jj + 1
                wgts(j) = (yin(jj+1)-yout(j))/(yin(jj+1)-yin(jj))
                wgtn(j) = (yout(j)-yin(jj))/(yin(jj+1)-yin(jj))
                exit
             end if
          end do
       end do
    else
       do j=1,nout
          do jj=1,nin-1
             if (yout(j).le.yin(jj) .and. yout(j).gt.yin(jj+1)) then
                jjm(j) = jj
                jjp(j) = jj + 1
                wgts(j) = (yin(jj+1)-yout(j))/(yin(jj+1)-yin(jj))
                wgtn(j) = (yout(j)-yin(jj))/(yin(jj+1)-yin(jj))
                exit
             end if
          end do
       end do
    end if

    !
    ! Check that interp/extrap points have been found for all outputs
    !
    icount = 0
    do j=1,nout
       if (jjm(j).eq.0 .or. jjp(j).eq.0) icount = icount + 1
       ratio=wgts(j)+wgtn(j)
       if((ratio<0.9_r8.or.ratio>1.1_r8).and.extrap_method.ne.0) then
          write(iulog,*) j, wgts(j),wgtn(j),jjm(j),jjp(j), increasing,extrap_method
          call endrun('Bad weight computed in LININTERP_init')
       end if
    end do
    if (icount.gt.0) then
       call endrun('LININTERP: Point found without interp indices')
    end if

  end subroutine lininterp_init

  subroutine lininterp2d1d(arrin, n1, n2, arrout, m1, wgt1, wgt2, fldname)
    implicit none
    !-----------------------------------------------------------------------
    !
    ! Arguments
    !
    integer, intent(in) :: n1, n2, m1
    real(r8), intent(in) :: arrin(n1,n2)    ! input array of values to interpolate
    type(interp_type), intent(in) :: wgt1, wgt2
    real(r8), intent(out) :: arrout(m1) ! interpolated array
    character(len=*), intent(in), optional :: fldname(:)
    !
    ! locals
    !
    integer i                ! indices
    integer, pointer :: iim(:), jjm(:)
    integer, pointer :: iip(:), jjp(:)

    real(r8), pointer :: wgts(:), wgte(:)
    real(r8), pointer :: wgtn(:), wgtw(:)

    jjm => wgt2%jjm
    jjp => wgt2%jjp
    wgts => wgt2%wgts
    wgtn => wgt2%wgtn

    iim => wgt1%jjm
    iip => wgt1%jjp
    wgtw => wgt1%wgts
    wgte => wgt1%wgtn

    do i=1,m1
       arrout(i) = arrin(iim(i),jjm(i))*wgtw(i)*wgts(i)+arrin(iip(i),jjm(i))*wgte(i)*wgts(i) + &
                   arrin(iim(i),jjp(i))*wgtw(i)*wgtn(i)+arrin(iip(i),jjp(i))*wgte(i)*wgtn(i)
    end do


  end subroutine lininterp2d1d

  subroutine lininterp3d2d(arrin, n1, n2, n3, arrout, m1, len1, wgt1, wgt2)
    implicit none
    !-----------------------------------------------------------------------
    !
    ! Arguments
    !
    integer, intent(in) :: n1, n2, n3, m1, len1   ! m1 is to len1 as ncols is to pcols
    real(r8), intent(in) :: arrin(n1,n2,n3)    ! input array of values to interpolate
    type(interp_type), intent(in) :: wgt1, wgt2
    real(r8), intent(out) :: arrout(len1, n3) ! interpolated array

    !
    ! locals
    !
    integer i, k               ! indices
    integer, pointer :: iim(:), jjm(:)
    integer, pointer :: iip(:), jjp(:)

    real(r8), pointer :: wgts(:), wgte(:)
    real(r8), pointer :: wgtn(:), wgtw(:)

    jjm => wgt2%jjm
    jjp => wgt2%jjp
    wgts => wgt2%wgts
    wgtn => wgt2%wgtn

    iim => wgt1%jjm
    iip => wgt1%jjp
    wgtw => wgt1%wgts
    wgte => wgt1%wgtn

    do k=1,n3
       do i=1,m1
          arrout(i,k) = arrin(iim(i),jjm(i),k)*wgtw(i)*wgts(i)+arrin(iip(i),jjm(i),k)*wgte(i)*wgts(i) + &
               arrin(iim(i),jjp(i),k)*wgtw(i)*wgtn(i)+arrin(iip(i),jjp(i),k)*wgte(i)*wgtn(i)
       end do
    end do

  end subroutine lininterp3d2d

end module interpolate_data

module intp_util
  use oracle_shim
  implicit none
contains
subroutine findplb( x, nx, xval, index )

   !----------------------------------------------------------------------- 
   ! Purpose: 
   ! "find periodic lower bound"
   ! Search the input array for the lower bound of the interval that
   ! contains the input value.  The returned index satifies:
   ! x(index) .le. xval .lt. x(index+1)
   ! Assume the array represents values in one cycle of a periodic coordinate.
   ! So, if xval .lt. x(1), or xval .ge. x(nx), then the index returned is nx.
   !
   ! Author: B. Eaton
   !----------------------------------------------------------------------- 

   use shr_kind_mod, only: r8 => shr_kind_r8

   integer, intent(in) ::   nx         ! size of x
   real(r8), intent(in) ::  x(nx)      ! strictly increasing array
   real(r8), intent(in) ::  xval       ! value to be searched for in x
   
   integer, intent(out) ::  index

   ! Local variables:
   integer i
   !-----------------------------------------------------------------------

   if ( xval .lt. x(1) .or. xval .ge. x(nx) ) then
      index = nx
      return
   end if

   do i = 2, nx
      if ( xval .lt. x(i) ) then
         index = i-1
         return
      end if
   end do

end subroutine findplb

end module intp_util

module trc_shim
  use oracle_shim
  implicit none
contains
  subroutine vert_interp( ncol, levsiz, pin, pmid, datain, dataout )
    !--------------------------------------------------------------------------
    !
    ! Interpolate data from current time-interpolated values to model levels
    !--------------------------------------------------------------------------
    implicit none
    ! Arguments
    !
    integer,  intent(in)  :: ncol                ! number of atmospheric columns
    integer,  intent(in)  :: levsiz
    real(r8), intent(in)  :: pin(pcols,levsiz)
    real(r8), intent(in)  :: pmid(pcols,pver)          ! level pressures
    real(r8), intent(in)  :: datain(pcols,levsiz)
    real(r8), intent(out) :: dataout(pcols,pver)

    !
    ! local storage
    !

    integer ::  i                   ! longitude index
    integer ::  k, kk, kkstart      ! level indices
    integer ::  kupper(pcols)       ! Level indices for interpolation
    real(r8) :: dpu                ! upper level pressure difference
    real(r8) :: dpl                ! lower level pressure difference



    !--------------------------------------------------------------------------
    !
    ! Initialize index array
    !
    do i=1,ncol
       kupper(i) = 1
    end do

    do k=1,pver
       !
       ! Top level we need to start looking is the top level for the previous k
       ! for all column points
       !
       kkstart = levsiz
       do i=1,ncol
          kkstart = min0(kkstart,kupper(i))
       end do
       !
       ! Store level indices for interpolation
       !
       do kk=kkstart,levsiz-1
          do i=1,ncol
             if (pin(i,kk).lt.pmid(i,k) .and. pmid(i,k).le.pin(i,kk+1)) then
                kupper(i) = kk
             end if
          end do
       end do
       ! interpolate or extrapolate...
       do i=1,ncol
          if (pmid(i,k) .lt. pin(i,1)) then
             dataout(i,k) = datain(i,1)*pmid(i,k)/pin(i,1)
          else if (pmid(i,k) .gt. pin(i,levsiz)) then
             dataout(i,k) = datain(i,levsiz)
          else
             dpu = pmid(i,k) - pin(i,kupper(i))
             dpl = pin(i,kupper(i)+1) - pmid(i,k)
             dataout(i,k) = (datain(i,kupper(i) )*dpl + &
                  datain(i,kupper(i)+1)*dpu)/(dpl + dpu)
          end if
       end do
    end do


  end subroutine vert_interp

end module trc_shim

program oracle
  use oracle_shim
  use interpolate_data
  use intp_util
  use trc_shim
  implicit none
  real(r8), parameter :: pi = 3.14159265358979323846_r8, d2r = pi/180._r8   ! shr_const_pi, mo_constants d2r
  real(r8), parameter :: zero=0_r8, twopi=2_r8*pi                              ! tracer_data.F90:1488
  type trfile_shim
     logical :: cyclical = .true., fixed = .false.
     integer :: interp_recs = 2, cyc_ndx_beg = 1, cyc_ndx_end
     real(r8) :: datatimem, datatimep, curr_mod_time, one_yr
  end type trfile_shim
  type(trfile_shim) :: file
  integer :: nlon, nlat, nlev, nrec, ncol, ntime, it, r, i, k, n, np1, cyc_tsize, nm, np
  real(r8), allocatable :: lons(:), lats(:), all_data_times(:), hyam(:), hybm(:), to_lats(:), to_lons(:)
  real(r8), allocatable :: fld(:,:,:,:), psf(:,:,:), fcol(:,:,:), pscol(:,:), pmid(:,:), pin(:,:), datain(:,:), dataout(:,:), ps(:)
  real(r8) :: p0, time, fact1, fact2, deltat
  type(interp_type) :: lon_wgts, lat_wgts

  read(*,*) nlon, nlat, nlev, nrec, ncol, pver
  pcols = ncol
  allocate(lons(nlon), lats(nlat), all_data_times(nrec), hyam(nlev), hybm(nlev), to_lats(ncol), to_lons(ncol))
  allocate(fld(nlon,nlat,nlev,nrec), psf(nlon,nlat,nrec), fcol(ncol,nlev,nrec), pscol(ncol,nrec))
  allocate(pmid(ncol,pver), pin(ncol,nlev), datain(ncol,nlev), dataout(ncol,pver), ps(ncol))
  read(*,*) lons; read(*,*) lats; read(*,*) all_data_times; read(*,*) file%one_yr
  read(*,*) hyam; read(*,*) hybm; read(*,*) p0
  read(*,*) to_lats; read(*,*) to_lons
  read(*,*) fld; read(*,*) psf
  lons = lons * d2r                                             ! tracer_data.F90:306
  lats = lats * d2r                                             ! tracer_data.F90:318
  call lininterp_init(lons, nlon, to_lons(1:ncol), ncol, 2, lon_wgts, zero, twopi)   ! :1539
  call lininterp_init(lats, nlat, to_lats(1:ncol), ncol, 1, lat_wgts)               ! :1540
  do r = 1, nrec
     call lininterp3d2d(fld(:,:,:,r), nlon, nlat, nlev, fcol(:,:,r), ncol, ncol, lon_wgts, lat_wgts)
     call lininterp2d1d(psf(:,:,r), nlon, nlat, pscol(:,r), ncol, lon_wgts, lat_wgts)
  end do
  do k = 1, nlev
     write(*,'(*(es25.17e3))') (fcol(i,k,1), i=1,ncol)
  end do
  write(*,'(*(es25.17e3))') (pscol(i,1), i=1,ncol)
  file%cyc_ndx_end = nrec
  read(*,*) ntime
  do it = 1, ntime
     read(*,*) time
     read(*,*) pmid
     file%curr_mod_time = time
     cyc_tsize = file%cyc_ndx_end - file%cyc_ndx_beg + 1

          call findplb(all_data_times(file%cyc_ndx_beg:file%cyc_ndx_end),cyc_tsize, time, n )

          if (n == cyc_tsize) then
             np1 = 1
          else
             np1 = n+1
          endif

     file%datatimem = all_data_times(n  +file%cyc_ndx_beg-1)
     file%datatimep = all_data_times(np1+file%cyc_ndx_beg-1)
     nm = n; np = np1

       deltat = file%datatimep - file%datatimem

       if ( file%cyclical .and. (deltat < 0._r8) ) then
          deltat = deltat+file%one_yr
          if ( file%datatimep >= file%curr_mod_time ) then
             fact1 = (file%datatimep - file%curr_mod_time)/deltat
          else
             fact1 = (file%datatimep+file%one_yr - file%curr_mod_time)/deltat
          endif
       else
             fact1 = (file%datatimep - file%curr_mod_time)/deltat
       endif

       fact2 = 1._r8-fact1
     if (fact2 == 0) then
        ps(:ncol) = fact1*pscol(:ncol,nm)
        datain(:ncol,:) = fact1*fcol(:ncol,:,nm)
     else
        ps(:ncol) = fact1*pscol(:ncol,nm) + fact2*pscol(:ncol,np)
        datain(:ncol,:) = fact1*fcol(:ncol,:,nm) + fact2*fcol(:ncol,:,np)
     end if
     do i = 1,ncol
        do k = 1,nlev
           pin(i,k) = p0*hyam(k) + ps(i)*hybm(k)
        enddo
     enddo
     call vert_interp(ncol, nlev, pin, pmid, datain, dataout)
     write(*,'(2i6,es25.17e3)') n, np1, fact1
     do k = 1, pver
        write(*,'(*(es25.17e3))') (dataout(i,k), i=1,ncol)
     end do
  end do
end program oracle
