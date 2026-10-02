# Preregistration — ORCA2 round 108 EEN recorder-chain repair

Date: 2026-10-02. Base: `18b659b0333b50ddc5ec2c0ea68370121c49f78a`.
Scope is instrumentation only on hierarchy rung 0. Every later ORCA2 number
from this record will be **independent** because rung 0 starts from NEMO's own
from-rest state. No model physics, card field, configuration value, carried
state, sea-ice selector, or `unmeasured_features` entry may change.

## Observed failure and compiled source

The operator's round-107 run built successfully but stopped at initialization.
Both new rank-tagged files exist at zero bytes, and `ocean.output` records
`round105: missing EEN operand output directory`, followed by
`round105: cannot initialize EEN operand record`. This is an instrument
failure, not a physics measurement.

The round-107 binary inherits both recorders. Its compiled `dyn_spg_ts_init`
calls the round-105 initializer before the round-107 initializer
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1103-1104`). The
inherited initializer reads `ORCA2_R105_EEN_ACCUM_DIR`, requires an absolute
path, and opens a rank-tagged `STATUS='NEW'` stream
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/l4_r105_een_accum.f90:34-45`). The
round-107 launcher exported only `ORCA2_R107_EEN_STEP_DIR`. The repair is
therefore a launcher-only fresh-target run of the exact existing binary with
both recorder directories set to that one pre-created target. The failed
target is retained and never reused or deleted.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R108-P1 | The missing inherited environment variable is the sole runtime blocker. | A fresh-target run of the unchanged round-107 binary with both recorder directory variables set reaches `STOP 0`. | Any nonzero NEMO exit or other error: **REFUTED**; retain the run and stop at its first source-resolved failure. |
| R108-P2 | Both additions-only recorders initialize and dump exactly once on each rank. | Logs contain ranks 0 and 1 exactly once for each recorder's `INIT` and `DUMP`; the fresh target contains two nonempty round-105 streams and two nonempty round-107 streams. | Missing, duplicate, empty, or wrong-rank stream: refuse admission and repair only the instrument. |
| R108-P3 | The inherited round-105 stream remains observationally passive. | The fresh round-105 record is byte-identical rank by rank to the admitted round-105 record. | Any byte difference: refuse; the supposedly identical producer or inputs moved. |
| R108-P4 | The round-107 writer is observationally passive. | All twenty kt=1..10 terminal restart shards are byte-identical to the admitted round-105 source run. | Any restart byte differs: refuse the record and do not quote operands. |
| R108-P5 | The self-describing round-107 record is admissible for the signed-zero walk. | All header/payload/restart plants fire and the unplanted checker ends `PASS_R107_EEN_STEP_ADMISSION`. | Any plant stays green or the unplanted checker refuses: stop for a checker/writer repair; no physics claim. |

## Landing bar

This round lands documentation and acquisition tooling only. It stops for the
operator-run record unless an already complete fresh target exists. No
per-level operand number is quoted until R108-P1 through P5 pass. The next
measurement, after admission, compares `zpvo_nw`, the two live thicknesses,
the neighboring mask, the product, and the accumulator in compiled source
order to name the first remaining non-bit statement.

