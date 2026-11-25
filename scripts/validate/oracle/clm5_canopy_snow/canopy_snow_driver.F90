! CLM5.0 canopy snow (snowveg_flag = 'ON_RAD') oracle driver.  The physics is the
! verbatim CTSM release-clm5.0.34 line ranges #include'd below (see build.sh);
! this file only declares their variables as one-element arrays (p = c = g = 1)
! and reads/writes one case per line.
! Input columns: snocan liqcan forc_snow forc_rain forc_t forc_wind elai esai
!                dewmx dtime t_veg evap_minus_tran
! Output columns: snocan_after_hydrology qflx_prec_grnd_snow fwet fcansno
!                 snocan_after_evap liqcan_after_evap qflx_snocanfall snounload
program canopy_snow_driver
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: tfrz = 273.15_r8
  integer, parameter :: iulog = 6
  logical, parameter :: snowveg_on = .false., snowveg_onrad = .true.
  logical, parameter :: use_clm5_fpi = .true.
  real(r8), parameter :: interception_fraction = 1.0_r8
  real(r8), parameter :: maximum_leaf_wetted_fraction = 0.05_r8
  integer :: p, c, g, n, i, frac_veg_nosno(1)
  real(r8) :: dtime, h2ocanmx, fpi, snocanmx, liqcanmx, fpisnow, xliqrun, xsnorun, xrun
  real(r8) :: vegt, dewmxi
  real(r8), dimension(1) :: snocan, liqcan, h2ocan, forc_snow, forc_rain, forc_t, forc_wind, &
       elai, esai, dewmx, qflx_candrip, qflx_snocanfall, qflx_liqcanfall, qflx_snowindunload, &
       qflx_snotempunload, snounload, qflx_through_snow, qflx_through_rain, qflx_prec_intr, &
       fracsnow, fracrain, qflx_prec_grnd_snow, qflx_prec_grnd_rain, fwet, fcansno, fdry, &
       t_veg, qflx_tran_veg, qflx_evap_veg
  real(r8) :: inp(12), s_hyd, grnd
  character(len=512) :: fin, fout
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=fin, status='old'); open(11, file=fout, status='replace')
  read(10, *) n
  p = 1; c = 1; g = 1
  do i = 1, n
     read(10, *) inp
     snocan(1) = inp(1); liqcan(1) = inp(2); forc_snow(1) = inp(3); forc_rain(1) = inp(4)
     forc_t(1) = inp(5); forc_wind(1) = inp(6); elai(1) = inp(7); esai(1) = inp(8)
     dewmx(1) = inp(9); dtime = inp(10); t_veg(1) = inp(11)
     qflx_tran_veg(1) = 0._r8; qflx_evap_veg(1) = inp(12)
     h2ocan(1) = snocan(1) + liqcan(1)
     frac_veg_nosno(1) = merge(1, 0, elai(1) + esai(1) > 0._r8)
     ! ---- CanopyHydrology: reset, interception, unloading ----
#include "ch_reset.inc"
#include "ch_intercept.inc"
#include "ch_unload.inc"
     s_hyd = snocan(1); grnd = qflx_prec_grnd_snow(1)
     ! ---- FracWet ----
#include "ch_fracwet.inc"
     ! ---- CanopyFluxes: wet-canopy evaporation ----
#include "cf_evap.inc"
     write(11, '(8es26.17e3)') s_hyd, grnd, fwet(1), fcansno(1), snocan(1), liqcan(1), &
          qflx_snocanfall(1), snounload(1)
  end do
  close(10); close(11)
end program canopy_snow_driver
