program burial_oracle
  ! Build: gfortran -ffree-line-length-none ctsm_burial_oracle.F90 -o b && ./b < ctsm_burial_oracle_cases.txt
  ! Feeds tests/land/unit/test_canopy_snow_albedo.py::BURIAL.
  ! Executes VERBATIM CTSM 5.1 SatellitePhenologyMod.F90 lines 173-178 and 185-188
  ! (snow burial of leaf/stem area) on one patch.  Only edit: patch%itype(p) -> ityp.
  ! noveg = 0, nbrdlf_dcd_brl_shrub = 11 (CTSM pftconMod, 16-PFT order).
  ! Input row: ityp htop hbot snow_depth frac_sno tlai tsai ; output row: elai esai
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: noveg = 0, nbrdlf_dcd_brl_shrub = 11
  integer :: p, c, ityp, ncase, k
  real(r8) :: htop(1), hbot(1), snow_depth(1), frac_sno(1), tlai(1), tsai(1), elai(1), esai(1), ol, fb
  p = 1; c = 1
  read(*,*) ncase
  do k = 1, ncase
    read(*,*) ityp, htop(1), hbot(1), snow_depth(1), frac_sno(1), tlai(1), tsai(1)
         if (ityp > noveg .and. ityp <= nbrdlf_dcd_brl_shrub ) then
            ol = min( max(snow_depth(c)-hbot(p), 0._r8), htop(p)-hbot(p))
            fb = 1._r8 - ol / max(1.e-06_r8, htop(p)-hbot(p))
         else
            fb = 1._r8 - (max(min(snow_depth(c),max(0.05,htop(p)*0.8_r8)),0._r8)/(max(0.05,htop(p)*0.8_r8)))
         endif
           elai(p) = max(tlai(p)*(1.0_r8 - frac_sno(c)) + tlai(p)*fb*frac_sno(c), 0.0_r8)
           esai(p) = max(tsai(p)*(1.0_r8 - frac_sno(c)) + tsai(p)*fb*frac_sno(c), 0.0_r8)
           if (elai(p) < 0.05_r8) elai(p) = 0._r8
           if (esai(p) < 0.05_r8) esai(p) = 0._r8
    write(*,'(2es25.16)') elai(1), esai(1)
  end do
end program burial_oracle
