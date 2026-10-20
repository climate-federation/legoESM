# PREREG — vertical-mixing lane V1 round 7 (vmix_r7)

Frozen before the kt 1..10 ladders are run. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1`
(admitted round 3). Decisions 129 = 1 and 130 = 1 in force.

Already measured (offline walk, evidence `gridladder_vmix_rounds/round7/`, done BEFORE this file —
stated so the order is visible): the card's kt-1 surface en(1) is bit-exact; a numpy
NEMO-literal recurrence (`zdftke.f90:457-475`) plus a literal nn_etau=1 increment
(`zdftke.f90:498-502`, scalar glibc `exp`; `htau` from `SIN(rpi/180*gphit)`, `zdftke.f90:864`)
applied to the card's own traced matrix and right-hand side reproduces NEMO's kt-1 en in
all 414,119 wet W cells. Owners: (a) the compiled solver contracts two multiply-subtract
pairs (3,072 cells off); (b) the card's `htau` takes `sin(deg2rad(degrees(lat_T)))` and the
increment takes XLA's `exp` (1,192 columns of htau and 59,798 exps differ from glibc).

Landing (V1-selected only; library defaults and every other card unchanged): source-round
the two solver products; `htau` from native gphit with the folded `rpi/180` and scalar libm
`sin`; scalar libm `exp` in the increment (the repo's `transcendentals` libm arm).

- **P1.** kt-1 `en`: 31,802 unequal -> 0. Falsifier: any unequal wet cell.
- **P2.** kt-1 `avm`/`avt` unequal counts fall (16,622 / 15,927 in round 6). Falsifier: either rises.
- **P3.** kt-2 `en` max miss falls below round 6's 5.57e-7. Falsifier: it does not.
- **P4.** V1 is not expected to certify this round (kt-1 u/v/T/S are non-bit in 169-560 cells,
  owned elsewhere). Falsifier: every row at the bar kt 1..10 under both labels.
