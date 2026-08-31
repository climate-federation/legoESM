! Offline arithmetic discriminator for ZDF sweep row 30.
!
! The expression and statement boundaries are the active NEMO 5.0.2
! eosbn2.F90:296-305 simplified-EOS branch.  This helper is compiled with the
! oracle's recorded arch-conda production flags and called only by the CPU
! receipt probe; it is not linked into legoESM.
SUBROUTINE row30_seos_prd( n, pt, ps, pdepth, pout ) BIND(C)
   USE, INTRINSIC :: iso_c_binding, ONLY: c_int, c_double
   INTEGER(c_int), VALUE :: n
   REAL(c_double), INTENT(IN)  :: pt(n), ps(n), pdepth(n)
   REAL(c_double), INTENT(OUT) :: pout(n)
   INTEGER(c_int) :: ji
   REAL(c_double) :: zt, zs, zh, zn
   REAL(c_double), PARAMETER :: rn_T0=10.0_c_double, rn_S0=35.0_c_double
   REAL(c_double), PARAMETER :: rn_a0=0.165_c_double
   REAL(c_double), PARAMETER :: rn_b0=0.76554_c_double
   REAL(c_double), PARAMETER :: rn_lambda1=0.06_c_double
   REAL(c_double), PARAMETER :: rn_lambda2=0.0_c_double
   REAL(c_double), PARAMETER :: rn_mu1=1.4970e-4_c_double
   REAL(c_double), PARAMETER :: rn_mu2=0.0_c_double
   REAL(c_double), PARAMETER :: rn_nu=0.0_c_double
   REAL(c_double), PARAMETER :: r1_rho0=1.0_c_double/1026.0_c_double
   DO ji=1,n
      zt = pt(ji) - rn_T0
      zs = ps(ji) - rn_S0
      zh = pdepth(ji)
      zn = -rn_a0 * ( 1.0_c_double + 0.5_c_double*rn_lambda1*zt + rn_mu1*zh ) * zt &
         + rn_b0 * ( 1.0_c_double - 0.5_c_double*rn_lambda2*zs - rn_mu2*zh ) * zs &
         - rn_nu * zt * zs
      pout(ji) = zn * r1_rho0
   END DO
END SUBROUTINE row30_seos_prd
