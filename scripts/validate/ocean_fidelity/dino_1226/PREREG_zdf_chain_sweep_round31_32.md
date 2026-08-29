# Preregistration: ZDF sweep rows 31--32 volume-form discriminators

Date: 2026-08-29. CPU-only matched day-180 lane. These substitutions were
registered in the master row table before measurement and require no new NEMO
dumps. Because the complete row-30 composite is red, every number here is
targeting-only until that ordered stop closes.

## Row 31 momentum

Reconstruct the NEMO pre-solve U/V arrays from Nbb velocity plus `rDt*Krhs`,
remove the stage-7 barotropic mode, apply the dumped bottom-drag correction and
surface-stress increment, then feed dumped `avm` and live face metrics to the
production implicit solver. The accumulating column bar is `1.0e-12`; report
U and V independently over their complete wet-column censuses and the four
southern focus columns.

If the production solve fails, evaluate a read-only discriminator preserving
`dynzdf.F90:199-214,340-380` literally: written matrix association, ordered
forward recurrence, and reverse substitution. `CONFIRM` for a solver-lowering
owner means the literal result has zero failing columns in both components.
`REFUTE` means any literal column still fails. A wrong-`rDt` arm and a one-cell
roll must fail.

## Row 32 tracers

Reconstruct the volume-form input
`(e3t(Kbb)*T(Kbb)+rDt*e3t(Kmm)*Krhs)/e3t(Kaa)`, compose dumped `avt` with the
native-slope K33 term, and feed the production paired implicit solve. Score T
and S independently at `1.0e-12`, including every focus column. A wrong-e3t
slot and a one-cell roll must fail. No disposition is promotable while an
earlier row is red.
