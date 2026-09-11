# PREREG — the kt=1 frozen forcing, and which operator owns its meridional gap

Written BEFORE the measurement it scores (round 43).  Registered: the owner,
the falsifier, and what each row can and cannot see.

## What is already CONFIRMED (round 42)

At kt=1 the barotropic loop's first non-bit substep is substep 1, both sides
close their own identity at 0 cells, so the break is the FROZEN FORCING:
`zv_frc` differs by max 8.414e-11 against NEMO's own rms 1.414e-06, and
substep-1 `v_exit` is exactly 117.391x that.  The gap is meridional.

## PREREGISTERED OWNER — `dyn_hpg` (`hpg_sco`)

Named before the measurement, on this reading of the oracle:

* `dynspg_ts.f90:275-276` (the record's own build, `cfgs/DINO_KT1_RANKDUMP`)
  builds the depth-mean base from
  `puu/pvv(:,:,:,Krhs)` at `dyn_spg` ENTRY, weighted by the REST thickness
  `e3v_3d` and divided by `hv_0` through `r1_hv_0` (`domain.f90:215`).
* `stpmlf.f90` calls `dyn_adv -> dyn_vor -> dyn_ldf -> dyn_hpg` before
  `dyn_spg`, and every one of those but `dyn_hpg` is proportional to the
  velocity, which is identically zero on the first step from rest.
* The three later writes to `zv_frc` are the 2-D Coriolis removal
  (`:300`, on `pvv_b(Kmm)` = 0 from rest, and `dyn_cor_2D` at `:1429-1452` is
  a bare weighted sum of `punb/pvnb` with no additive term, so zero in gives
  zero out), the implicit-drag increment (`:313`) and the wind (`:375`, the
  `ln_bt_fw=.FALSE.` averaged branch; DINO's wind is zonal).
  Resolved from the run's own `output.namelist.dyn`, not from the deck:
  `LN_RSTART=F, LN_APR_DYN=F, LN_DRGIMP=T, LN_HPG_SCO=T, LN_BT_FW=F,
  NN_BT_FLT=2, LN_DYNVOR_EEN=T`.

So the prediction is that `zv_frc` at kt=1 IS the depth mean of `hpg_sco`
and nothing else.

## FALSIFIER

If the depth mean of the transcribed `hpg_sco`, given NEMO's own mesh and
NEMO's own initial T/S, does NOT reproduce `spg_dump_zv_frc` to roundoff,
the operator set above is wrong and the owner is NOT `dyn_hpg`.  The number
that decides it: cells unequal and max|d| against NEMO's own rms 1.414e-06.

## WHAT EACH ROW CAN AND CANNOT SEE

| row | sees | blind to |
|---|---|---|
| the three `stp_dump_0{3,4,5}` 3-D dumps | **almost nothing — RETRACTED.** They are written by all 16 ranks to one untagged filename with `STATUS='REPLACE'`, so the surviving bytes are the last writer's: a rank that wrote nonzeros is overwritten by a later rank writing zeros. The tiles are not even one size (14 ranks at `jpi*jpj`=870, ranks 14/15 at 840) and the file is exactly `870*jpkm1` doubles, so it holds at most one tile — **8.4% of the domain, measured, not attributable to a cell**. Corroboration only. | everything else |
| `drg_/wnd_/cor2d_` increments | the per-cell increment, rank-tagged, stitched | the PRE-LOOP `dyn_cor_2D` removal is not dumped; its operand `pvv_b(Kmm)` being zero is read from the restart and from the oracle's linearity, not from a dump of the removal itself |
| the transcription vs `spg_dump_zv_frc` | the whole chain: EOS, the `hpg_sco` statements, the `:273` depth mean, `r1_hv_0` | a compensating pair of errors inside it (mitigated: the per-level walk below scores the SAME transcription level by level against legoESM) |
| the per-level walk | which statement of `hpg_sco` legoESM leaves | nothing about kt>=2, where the `(1+r3t)` stretch and the `zuap` slope term are no longer identically zero |

## RECORD

`/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump` (kt=1, 16 ranks).
