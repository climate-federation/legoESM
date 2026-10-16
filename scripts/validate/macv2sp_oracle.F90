program macv2sp_oracle
  ! Column oracle for scripts/data/adapt_cmip6_aerosol.py: calls the REFERENCE
  ! MACv2-SP routine sp_aop_profile (mo_simple_plumes_v1.f90, input4MIPs
  ! MPI-M-MACv2-SP-1-0 v20170201, unmodified) on one terrain-following column
  ! per case and prints the column optics.
  ! Build + run in a directory holding mo_simple_plumes_v1.f90 and MACv2.0-SP_v1.nc
  ! (sp_setup opens it from the CWD); NFF = a gfortran-built netcdf-fortran:
  !   gfortran -fdefault-real-8 -ffree-line-length-none -I$NFF/include \
  !     mo_simple_plumes_v1.f90 macv2sp_oracle.F90 -L$NFF/lib -Wl,-rpath,$NFF/lib -lnetcdff -o o
  !   ./o < macv2sp_oracle_cases.txt
  ! Input row: lat lon(0-360, the reference's convention) oro[m] year_fr lambda[nm]
  ! Output row: inputs, then column aod, aod-weighted ssa, (aod*ssa)-weighted asy, dNovrN.
  ! Feeds tests/unit/test_adapt_cmip6_aerosol.py::FORTRAN.
  use mo_simple_plumes, only: sp_aop_profile
  implicit none
  integer, parameter :: nlev = 60
  integer :: ncase, i, k
  real :: lat(1), lon(1), oro(1), year_fr, lambda, zh(0:nlev)
  real :: z(1,nlev), dz(1,nlev), dnovrn(1), aod(1,nlev), ssa(1,nlev), asy(1,nlev)
  real :: caod, csca
  read(*,*) ncase
  do i = 1, ncase
    read(*,*) lat(1), lon(1), oro(1), year_fr, lambda
    ! Stretched terrain-following half levels from the surface to 30 km.
    do k = 0, nlev
      zh(k) = oro(1) + (30000.0 - oro(1)) * (real(k) / nlev)**1.5
    end do
    do k = 1, nlev
      z(1,k) = 0.5 * (zh(k-1) + zh(k))
      dz(1,k) = zh(k) - zh(k-1)
    end do
    call sp_aop_profile(nlev, 1, lambda, oro, lon, lat, year_fr, z, dz, &
                        dnovrn, aod, ssa, asy)
    caod = sum(aod(1,:))
    csca = sum(aod(1,:) * ssa(1,:))
    write(*,'(5f12.5,4es24.16)') lat(1), lon(1), oro(1), year_fr, lambda, caod, &
         merge(csca / caod, 1.0, caod > tiny(1.0)), &
         merge(sum(aod(1,:) * ssa(1,:) * asy(1,:)) / csca, 0.0, csca > tiny(1.0)), dnovrn(1)
  end do
end program macv2sp_oracle
