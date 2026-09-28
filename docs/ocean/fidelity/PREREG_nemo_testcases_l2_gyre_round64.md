# Preregistration: NEMO GYRE L2 round 64 inert Krhs acquisition

**Frozen before the provisional read and before any R64 acquisition.**

## Admission question and exact decision

Round 63 changed the R46 run horizon from ten steps to two. Its admission
report has 63 inherited records: 27 byte-identical, 20 changed only in 194
admitted halo/undefined values, and 16 missing kt=3--10 records. Every one of
the 20 co-recorded comparisons says `consumed_equal=true`; therefore no
consumed-field perturbation is present to attribute to an added WRITE call.

R64 reuses the byte-identical R63 writer and four additive patches but makes
no namelist change. The R46 `nn_itend=10` deck is copied byte for byte.
CONFIRM inertness only if admission reports PASS, all 63 inherited records are
present, all common consumed fields are bit-identical, `mesh_mask.nc` is byte
identical, and both new records pass their exact calibration. REFUTE on any
missing record, consumed-field difference, restart/mesh difference, or
calibration inequality. Halo and registered undefined values retain the
existing round-21 dispositions; no new waiver is introduced.

The missing-record failure class is already controlled by
`tests/ocean/fidelity/test_nemo_testcase_round34_admission.py::test_a_missing_inherited_record_is_a_violation`.
The existing owned-bit admission plant must also exit nonzero.

## Provisional rejected-record read

The rejected R63 Krhs record may be read only to freeze the next admitted
measurement. It is not oracle evidence and cannot CONFIRM or REFUTE the
round-63 owner prediction.

The preview compares zero and then complete FCT advection, in compiled order
from `GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`.
For each tracer, the primary value is the exact written content boundary
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs`; a derived concentration-Krhs row is
reported separately. The first boundary is FCT advection only if zero is
bit-exact and after-advection content has at least one unequal wet cell.
Report unequal count, RMS, and maximum absolute difference for T and S. No
downstream owner is selected after the first unequal boundary.

## Acquisition identity

- source base/run: `GYRE_OMIP_L2_P3_SM_R46KT2`, round46/oracle_kt2_stage
- target: `GYRE_OMIP_L2_P3_SM_R64KRHS`
- target run: round64/oracle_krhs_split
- ten steps, one MPI rank, unchanged resolved configuration
- record point: kt=2, RK3 stage 3, inner domain
- CPU-only legoESM analysis; operator-only NEMO acquisition

No production physics, carried state, scheme, coefficient, threshold, or
record schema changes in this round.
