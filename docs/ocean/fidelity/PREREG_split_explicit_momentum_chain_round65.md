# Preregistration: GM U streamfunction composition, round 65

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 64 certifies rows 8.3--8.7 and stops at row 8.8: 8,938 of 9,758 U
columns fail the `1e-12` accumulation bar, with maximum normalized error
`3.8631299623e-7`. NEMO `ldftra.F90:894-902` builds the U bolus transport
from the vertical difference of

`-r1_4 * e2u * (wslpi_i + wslpi_ip1) * (aeiu_k + aeiu_kp1) * wumask`.

The production helper instead forms two normalized half-sums first and then
multiplies them. Existing full-halo `eiv_dump_{u,psi_uw,wslpi,aeiu}.bin`
streams permit a no-new-NEMO-run peel. Capture the arguments actually consumed
by `nemo_eiv_bolus_transport`, and score, on the registered 9,758-column U
population:

1. face `aeiu` against the existing direct dump;
2. the `wslpi(k+1)` face sum against the shifted direct slope dump;
3. current normalized-half-sum psi against direct `zpsi_uw(2)`;
4. NEMO literal `-r1_4` association on the same captured operands;
5. the 2x2 oracle-slope/oracle-aeiu substitution table, under the literal
   association.

Operand and psi bars are `1e-12`. If both operands pass and the literal
same-operand psi passes while the normalized form fails, disposition is
`ROW8_8_LOCALIZED_TO_PSI_ASSOCIATION`. If an operand fails, name the first
failing operand and retain the 2x2 main/interaction table; do not assign
association ownership. A sign-flipped psi, a meridional roll, and the current
normalized form must be red-capable controls. Round 64 is admitted only at
SHA-256 `8858d60a51b07181e290fead087e4bab69c0d15e271bbdecb7d458ba12d4e4c7`.
