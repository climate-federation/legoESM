MODULE par_oce
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=real64, jpi=12, jpj=10, nn_hls=2
   INTEGER, PARAMETER :: ntsi=3, ntei=10, ntsj=3, ntej=8
END MODULE par_oce

MODULE par_ice
   IMPLICIT NONE
   INTEGER, PARAMETER :: jpl=5
END MODULE par_ice
