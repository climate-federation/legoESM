# GYRE L2 round 59 — pre-code self-review

Date: 2026-09-11. Parent `ee903981f2df22ba75ce7227d6cdef78051d85b5`.

The R58 terminal calibration is sound but incomplete: it rebuilds only
Prandtl and `avm/avt/dissl`. The round-57 contract also requires the recorded
matrix/RHS to reproduce `en`, and the recorded `en/rn2/e3t/taum` to reproduce
both mixing lengths before any model substitution.

That wider calibration fails. A literal Python transcription of the compiled
R58 recurrences at
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:466-483`
reconstructs the recorded `en_post_sweep` with 1,431 unequal cells over 17,400
wet solved interfaces. The mismatch is not physics: compiled NEMO allocates
`en` on `Nis0:Nie0,Njs0:Nje0` at
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdf_oce.f90:85-87`, while the
record's compiled writer declares its `p_rhs` dummy as full-domain
`jpi,jpj,jpk` and slices it at
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:173-189`.
The explicit-shape mismatch rebases/misindexes the reduced actual array, the
same defect class retracted in round 58 for `p_sh2/p_avt/p_dissl`.

Pre-code decision: stop the substitution walk; change only `p_rhs` to the
compiled `A2D(0)` bounds, add source-exact `en` and mixing-length calibration
plus a reachable RHS plant, and retarget operator acquisition to
`GYRE_OMIP_L2_P3_SM_R59TKE` under `phase3/round59/oracle_tke_operands`.
No model physics, configuration, state, trajectory harness, reconciliation
gate, freshwater pair, or #1484 guard is eligible to change on this record.

ASKED: calibration-first writer repair and a new target after any unequal
cell. UNASKED: none.

## Post-code self-review

The diff stays on the acquisition boundary: one writer dummy declaration,
one expanded calibration, one new plant, its synthetic fixture, the operator
target rename, and receipts. The repair matches compiled `en`'s explicit
bounds; preprocessing and syntax-only compilation pass. The synthetic record
forces the solve and length reconstructions to exact outputs, and each of the
eight record plants plus the stamp plant is exercised; 10 focused tests pass.
The unchanged R58 record is rejected with `en=1431`, `zmxlm=0`, `zmxld=0`,
proving the new check is both discriminating and the reason work stops.

No model-path result from the exploratory diagnosis is cited or landed. No
configuration or carried-state choice was made. UNVERIFIED: only a clean,
operator-built R59 record can validate the repaired writer at runtime.
