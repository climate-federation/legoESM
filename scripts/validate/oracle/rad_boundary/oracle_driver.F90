! Oracle driver for tests/atmosphere/oracle_fixtures/rad_boundary.npz.
!
! Runs CESM2.1 share code UNMODIFIED (cime5.6.47 share/util/shr_orb_mod.F90:
! shr_orb_params, shr_orb_decl, shr_orb_cosz) and the coupler open-ocean albedo
! block INCLUDEd verbatim (drivers/mct/main/seq_flux_mct.F90 L881-894, extracted
! by build.sh).  seq_flux_mct_albdif = 0.06 is the namelist default
! (namelist_definition_drv.xml).
!
! stdin : line 1 = eccen obliq_deg mvelp_deg ; line 2 = n ;
!         then n lines "calday lat lon" (lat/lon in rad, calday 1.0 = Jan 1 00Z)
! stdout: line 1 = obliqr lambm0 mvelpp ; then n lines "delta eccf cosz anidr anidf"
module cesm_ocnalb
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  real(r8), parameter :: seq_flux_mct_albdif = 0.06_r8   ! namelist_definition_drv.xml default
contains
  subroutine ocnalb(cosz, anidr, avsdr, anidf, avsdf)
    real(r8), intent(in) :: cosz
    real(r8), intent(out) :: anidr, avsdr, anidf, avsdf
include 'ocnalb.inc'
  end subroutine ocnalb
end module cesm_ocnalb

program rad_boundary_oracle
  use shr_kind_mod, only: r8 => shr_kind_r8
  use shr_orb_mod, only: shr_orb_params, shr_orb_decl, shr_orb_cosz, SHR_ORB_UNDEF_INT
  use cesm_ocnalb, only: ocnalb
  implicit none
  real(r8) :: eccen, obliq, mvelp, obliqr, lambm0, mvelpp
  real(r8) :: calday, lat, lon, delta, eccf, cosz, anidr, avsdr, anidf, avsdf
  integer :: n, i
  read(*,*) eccen, obliq, mvelp
  read(*,*) n
  call shr_orb_params(SHR_ORB_UNDEF_INT, eccen, obliq, mvelp, obliqr, lambm0, mvelpp, .false.)
  write(*,'(3ES25.17)') obliqr, lambm0, mvelpp
  do i = 1, n
     read(*,*) calday, lat, lon
     call shr_orb_decl(calday, eccen, mvelpp, lambm0, obliqr, delta, eccf)
     cosz = shr_orb_cosz(calday, lat, lon, delta)
     call ocnalb(cosz, anidr, avsdr, anidf, avsdf)
     write(*,'(5ES25.17)') delta, eccf, cosz, anidr, anidf
  end do
end program rad_boundary_oracle
