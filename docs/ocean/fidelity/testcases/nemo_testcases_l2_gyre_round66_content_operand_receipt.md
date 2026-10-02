# NEMO-testcases L2 GYRE round-66 content-operand receipt

Date: 2026-09-12

## Verdict

No production transcription is eligible to land. The preregistered
left-product tracer prediction is **REFUTED**: the named content operand is the
complete stage-3 `Krhs`, not `T/S(Kmm)`. The separately preregistered missing-LDF
mechanism is also **REFUTED by its frozen absolute threshold**, despite a 28x
improvement in its diagnostic substitution. No threshold was relaxed.

## Admission and active statements

The rerun reproduced `CONSUMED_FIELD_ADMISSION PASS`: 43/63 exact inherited
records, 20 changed records, 132 admitted values. Content T/S and Kbb/Kmm
thickness calibration were exactly zero unequal; all four admission plants
exited nonzero. Both round-66 one-ULP plants changed exactly one of 18,000 wet
cells.

At stage 3 the compiled program clears Krhs, accumulates advection/SBC/QSR/LDF
at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950`.
Its implicit update is `e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:548-562`.
legoESM reconstructs the WS thicknesses and split final content at
`ocean_model_latlon_cgrid.py:1888-1938`; production supplies its stage sources
at `ocean_model_latlon_cgrid.py:6327-6396` and consumes them at
`ocean_model_latlon_cgrid.py:7870-7908`.

## Reciprocal substitution (maximum absolute content difference)

| one operand | oracle into live T / S | live into oracle T / S |
|---|---:|---:|
| baseline | `1.679392692e-3 / 6.967976674e-5` | `0 / 0` |
| tracer Kbb | `1.679392692e-3 / 6.967976674e-5` | `0 / 0` |
| e3t Kbb | `1.679392692e-3 / 6.967976674e-5` | `0 / 0` |
| p2dt | `1.679392692e-3 / 6.967976674e-5` | `0 / 0` |
| e3t Kmm | `1.679392692e-3 / 6.967976663e-5` | `1.359695e-10 / 6.366463e-12` |
| **Krhs** | **`1.359695e-10 / 6.366463e-12`** | **`1.679392692e-3 / 6.967976663e-5`** |

Kbb tracer, Kbb thickness, and p2dt are bit-exact. The live five-operand
statement reconstructs captured split content within one rounding
(`2.274e-13` T, `1.819e-12` S); the oracle statement is bit-exact.

## Krhs boundary follow-up

Resolved NEMO selects iso-neutral Laplacian
(`round64/oracle_krhs_split/ocean.output:649-656`). The compiled operator reads
Kbb tracer gradients at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192`, constructs
Kmm-metric fluxes at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:230-246`, and adds
their divergence to Krhs at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:287-305`.
legoESM computes the shared GM/Redi tendency but sends it to `T_mid/S_mid`,
which the later WS restart replaces
(`ocean_model_latlon_cgrid.py:7132-7571`,
`ocean_model_latlon_cgrid.py:7870-7908`).

| row | T max | S max |
|---|---:|---:|
| live Krhs vs after-QSR | `6.198510e-11` | `5.135361e-12` |
| captured LDF vs oracle LDF increment | `3.901918e-12` | `1.396049e-13` |
| live+LDF Krhs vs after-LDF | `6.198290e-11` | `5.135350e-12` |
| live+LDF content vs oracle | `5.954040e-5` | `7.651458e-6` |

The final row improves T 28.2x but misses the preregistered `2e-6` content
ceiling, so the LDF routing arm is REFUTED and unlanded.

## FCT stop and Rule 12

The active FCT upstream/guess sequence is compiled at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-510`,
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:532-540`,
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:570-580`, and
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:602-611`.
Existing round-46/51 live rows show FCT is already handed unequal stage-3 Kmm
operands: u/v `7.546e-6/9.986e-6 m s-1`, T/S
`8.369e-7/6.795e-8`, e3t `2.473e-8 m`, and ww `4.833e-7`.
Their upstream kt=2 entry owner is u/v `2.747840475e-12/3.305560307e-12`.
Therefore the `6.20e-11/5.14e-12` complete-FCT tendency debt belongs to the
momentum/tracer ladder; no FCT statement or WRITE instrument is named.

No production row moved, so Rule 12 is N/A: GYRE kt3 remains
T/S `1.627511418e-4/6.327755180e-6`; day-30 remains T/S RMS
`1.239756827e-2/2.195296278e-3`. LOCK_EXCHANGE/OVERFLOW are value-inert, DINO's
leapfrog content is separate, and ORCA2 remains UNMEASURED with the executable
specification to run its production gate if a shared statement later moves.

## ASKED / UNASKED and open work

| state | item | disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration/carried-state/NEMO changes | not performed |

Open: preregister the correctly dimensioned FCT-content floor before deciding
whether the measured missing-LDF accumulation is eligible in a later round;
then resolve the kt=1-exit momentum owner before continuing inside FCT.
