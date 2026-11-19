! CAM6 microp_aero droplet-activation inputs oracle for
! tests/atmosphere/hydrostatic/unit/test_cam6_dropmixnuc_oracle.py (stage S9a).
! VERBATIM CAM6 microp_aero.F90 lines 517 (CLUBB tke), 523-524 and 527-553 (wsub, wsubi), 586-597 (liquid
! partition, use_preexisting_ice = .true. branch input); the rest is a driver.  eddy_scheme = 'CLUBB_SGS'.
! Build: gfortran -O0 -ffree-line-length-none -fno-fast-math cam6_microp_aero_oracle.F90 -o o
!        ./o < cam6_microp_aero_oracle_cases.txt > cam6_microp_aero_oracle_expected.txt
! Input: top_lev (1-based); then per column wp2 (pverp), q_liq (pver), q_ice (pver).
! Output per column: wsub (pver), then cldliqf (pver), es25.17.
module shr_kind_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
end module shr_kind_mod
program microp_aero_oracle
  use shr_kind_mod
  implicit none
  integer, parameter :: pcols = 3, ncol = 3, pver = 12, pverp = pver + 1
  integer, parameter :: cldliq_idx = 1, cldice_idx = 2
  real(r8), parameter :: qsmall = 1.e-18_r8                      ! microp_aero.F90:70
  logical, parameter :: use_preexisting_ice = .true.             ! namelist_defaults_cam.xml:1274 (cam6)
  character(len=16) :: eddy_scheme = 'CLUBB_SGS'
  type state_t
    real(r8) :: q(pcols, pver, 2)
  end type
  type(state_t) :: state1
  real(r8) :: wp2(pcols, pverp), tke(pcols, pverp), kvh(pcols, pverp)
  real(r8) :: wsub(pcols, pver), wsubi(pcols, pver), cldn(pcols, pver), cldo(pcols, pver)
  real(r8) :: lcldn(pcols, pver), lcldo(pcols, pver), cldliqf(pcols, pver)
  real(r8) :: qcld, dum
  integer :: top_lev, i, k
  read(*,*) top_lev
  do i = 1, ncol
    read(*,*) wp2(i, :)
    read(*,*) state1%q(i, :, cldliq_idx)
    read(*,*) state1%q(i, :, cldice_idx)
  end do
  kvh = 0._r8; cldn = 0._r8; cldo = 0._r8
      tke(:ncol,:) = (3._r8/2._r8)*wp2(:ncol,:)

   ! Set minimum values above top_lev.
   wsub(:ncol,:top_lev-1)  = 0.20_r8

   do k = top_lev, pver
      do i = 1, ncol

         select case (trim(eddy_scheme))
         case ('diag_TKE', 'CLUBB_SGS')
            wsub(i,k) = sqrt(0.5_r8*(tke(i,k) + tke(i,k+1))*(2._r8/3._r8))
            wsub(i,k) = min(wsub(i,k),10._r8)
         case default 
            ! get sub-grid vertical velocity from diff coef.
            ! following morrison et al. 2005, JAS
            ! assume mixing length of 30 m
            dum = (kvh(i,k) + kvh(i,k+1))/2._r8/30._r8
            ! use maximum sub-grid vertical vel of 10 m/s
            dum = min(dum, 10._r8)
            ! set wsub to value at current vertical level
            wsub(i,k)  = dum
         end select

         wsubi(i,k) = max(0.001_r8, wsub(i,k))
         if (.not. use_preexisting_ice) then
            wsubi(i,k) = min(wsubi(i,k), 0.2_r8)
         endif

         wsub(i,k)  = max(0.20_r8, wsub(i,k))

      end do
   end do

      lcldo = 0._r8
      cldliqf = 0._r8
      do k = top_lev, pver
         do i = 1, ncol
            qcld = state1%q(i,k,cldliq_idx) + state1%q(i,k,cldice_idx)
            if (qcld > qsmall) then
               lcldn(i,k)   = cldn(i,k)*state1%q(i,k,cldliq_idx)/qcld
               lcldo(i,k)   = cldo(i,k)*state1%q(i,k,cldliq_idx)/qcld
               cldliqf(i,k) = state1%q(i,k,cldliq_idx)/qcld
            end if
         end do
      end do

  do i = 1, ncol
    write(*, '(12es25.17)') wsub(i, :)
    write(*, '(12es25.17)') cldliqf(i, :)
  end do
end program microp_aero_oracle
