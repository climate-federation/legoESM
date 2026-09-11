# PREREG — the S-EOS's arithmetic: which statement owns the residual

Registered BEFORE the measurement and BEFORE any code change.
Round 43 left the kt=1 chain at `zv_frc` rel 1.1755e-11 and showed that, given
the SAME density, every remaining `hpg_sco` statement agrees to 2.2734e-13 —
52x below it.  So the next statement is the equation of state.  This
preregisters what will be measured, against what, and what would refute it.

## THE ORACLE'S STATEMENTS (read, not inferred)

DINO compiles `ln_seos = .true.` (`cfgs/DINO/EXP00/namelist_cfg:233`) and
`bld::tool::fppkeys key_qco key_vco_3d` (`cfgs/DINO/cpp_DINO.fcm`) — no
`key_RK3`, so the MLF branch, and `key_vco_3d` makes `gdept_3d` the full
THREE-DIMENSIONAL mesh array, not a 1-D ladder.

`cfgs/DINO/BLD/ppsrc/nemo/eosbn2.f90:357-369`, the in-situ branch the
pressure gradient consumes:

```
360  zt  = pts(ji,jj,jk,jp_tem,Knn) - rn_T0
361  zs  = pts(ji,jj,jk,jp_sal,Knn) - rn_S0
362  zh  = ((gdept_3d(ji,jj,jk) ) *(1._wp+r3t(ji,jj,Knn)))
363  ztm = tmask(ji,jj,jk)
365  zn =  - rn_a0 * ( 1._wp + 0.5_wp*rn_lambda1*zt + rn_mu1*zh ) * zt   &
366     &  + rn_b0 * ( 1._wp - 0.5_wp*rn_lambda2*zs - rn_mu2*zh ) * zs   &
367     &  - rn_nu * zt * zs
369  prd(ji,jj,jk) = zn * r1_rho0 * ztm
```

Coefficients as the record's OWN namelist resolves them
(`RUN_FROMREST_KT1/ocean.output:218-226`, and `:233` for `rho0`):
`rn_T0=10`, `rn_S0=35`, `rn_a0=0.165`, `rn_b0=0.76554`, `rn_lambda1=0.06`,
`rn_lambda2=0`, `rn_mu1=1.4970e-4`, `rn_mu2=0`, `rn_nu=0`, `rho0=1026`.
`r3t = ssh * r1_ht_0` (`domqco.f90:206`), `ht_0 = SUM(e3t_0*tmask)`.

Sibling branches, for the coverage table (each has a DIFFERENT evaluation
order, so they differ at roundoff even though they are algebraically equal):
`eos_insitu_pot_New_t` `:594-608` (`prhop` first, thermobaric subtracted
after), `rab_3d_t` `:1210-1221` (alpha/beta — FULL `rn_lambda1`, no 1/2),
`bn2_t` `:1487-1499`.

## THE CALIBRATION RECORD — NEMO's own density, per cell

`cfgs/DINO/MY_SRC/restart.F90:203-204` (compiled at
`BLD/ppsrc/nemo/restart.f90:190-191`) writes `rhd` from
`CALL eos( ts, Kmm, rhd )` — the 3-argument member of `INTERFACE eos`
(`eosbn2.f90:66-68`), i.e. `eos_insitu_New` and the branch above.
`stpmlf.f90:590` calls `rst_write(kstp, Nbb, Nnn)` AFTER the swap at
`:577-579`, so this `Kmm` is the AFTER state — the SAME level as the file's
own `tn`/`sn`/`sshn`, written in the adjacent statements.  So `rhd`, `tn`,
`sn` and `sshn` are one self-consistent tuple and no time level has to be
inferred (Rule 1d is satisfied by construction, not by memory).

This is a strictly better instrument than round 43's `zv_frc` chain: it is
per-cell, it has no pressure gradient, no depth mean and no `r1_hv_0` in it.

## PREREGISTERED OWNER, AND THE FALSIFIER

**Claim.** legoESM's S-EOS departs from NEMO's at the DEPTH OPERAND: the
pressure-gradient and `bn2` lanes read a ONE-DIMENSIONAL `gdept` ladder
(`eos.nemo_bn2_depth_ladders` returns `z_coord.t_depth_ref`, the NEMO
`gdept_1d`) where NEMO reads the THREE-DIMENSIONAL `gdept_3d(ji,jj,jk)`.
A second, far smaller statement is the ASSOCIATION: the production EOS
returns `rho0 + zn` and the caller recovers the anomaly as `rho/rho0 - 1`,
where NEMO multiplies `zn` by `r1_rho0` once and never forms `rho0 + zn`.

**Predicted sizes.**  The association's round trip must land near
`eps * rho0/|zn|` relative, i.e. ~2e-13 — and round 43 independently measured
the density residual at 1.3391e-13, so agreement there is a CONVERGENCE of two
instruments, not a new claim.  The depth operand's size is not predicted; it
is whatever `gdept_3d - gdept_1d` is on this mesh.

**Falsifier.**  If C1 below is not at the instrument's floor, the
transcription is wrong and NOTHING under it may be scored.  If R2 (1-D depth)
and R1 (3-D depth) measure the SAME, the depth operand is exonerated and the
owner is the association alone.  If R4 (3-D depth, round-trip association)
is at the floor, the association is exonerated instead.

## THE LADDER (each row differs from its neighbour in ONE statement)

| row | density statement | depth operand | association |
|---|---|---|---|
| C1 | literal `eosbn2.f90:365-369` | NEMO `gdept_0`(3-D)·(1+r3t) | `zn·r1_rho0` |
| D0 | — (geometry) | the card's `z_coord.nemo_gdept_0` vs NEMO's `gdept_0` | — |
| D1 | — (geometry) | `nemo_bn2_live_ladders` vs NEMO's `gdept_3d(Kmm)` | — |
| R1 | `eos.nemo_seos_prd_literal` | `nemo_gdept_0`(3-D)·(1+r3t) | `zn·(1/rho0)` |
| R2 | `eos.nemo_seos_prd_literal` | `nemo_bn2_live_ladders` (1-D) | `zn·(1/rho0)` |
| R3 | production `make_eos_fn("nemo_seos")` | `nemo_bn2_live_ladders` (1-D) | `rho/rho0 - 1` |
| R4 | production `make_eos_fn("nemo_seos")` | `nemo_gdept_0`(3-D)·(1+r3t) | `rho/rho0 - 1` |

R3 is what the pressure gradient runs today.  R2−R1 isolates the depth,
R4−R1 isolates the association, R3 is the total.

## THE BAR

The bar is EXACT (Rule 1b).  C1's own measured value is the floor and is
printed next to every row; a row is AT BAR only at or below it.  No tolerance
is granted to the model.  `--plant` moves one wet cell of the calibration by
one ulp: C1 must then read DEBT and the gate must refuse to score anything.
`--plant-lego` moves one wet cell of legoESM's array: every AT-BAR row must
move.

## WHAT THIS GATE CANNOT SEE (Rule 2)

* It scores the density at ONE time level of ONE step.  A statement that is
  right at kt=1 and wrong later (a time-level association, a stretch that is
  zero here) is invisible; `sshn` is NOT zero here (range −0.0449 .. +0.1169 m),
  so the `1+r3t` stretch IS exercised, which is why this record was chosen
  over the initial state.
* It cannot see a compensating pair inside the transcription.  Mitigation: C1
  is scored against a record legoESM had no part in producing, and every row
  below it differs from a neighbour by exactly one statement.
* It says nothing about `rab_3d`/`bn2`/`eos_insitu_pot` beyond the shared
  coefficients and the shared depth operand — those have their own gates.
