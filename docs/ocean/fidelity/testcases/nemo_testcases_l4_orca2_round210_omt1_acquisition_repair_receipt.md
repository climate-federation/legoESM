# ORCA2 round 210 — OMT-1 acquisition bootstrap repair

Date: 2026-10-09. Base: `c368e8b0d`. Preregistration commit:
`ac7be6812`. Repair commits: `c40a992a1`, `5f0e24251`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round210/`.
No trajectory number was measured. Future scores remain separately labelled
**independent OMT-1** and **given NEMO's entry OMT-1**.

## Result

The operator's round-209 run reached `STOP 0` on the mandatory two-step smoke,
then failed before producing an instrumented binary. The decisive build output
was:

```text
dirname: missing operand
You are installing a new configuration ORCA2_OMIP_L4_R209OMT1_P3 from  with sub-components:
./makenemo: line 365: /work_cfgs.txt: Permission denied
traadv.F90:84:2: fatal error: do_loop_substitute.h90: No such file or directory
```

This is a launcher defect, not physics evidence. `makenemo:263-280` resolves a
new configuration's `-r` operand only through `cfgs/ref_cfgs.txt`; round 209
passed instrumented work configuration `ORCA2_OMIP_L4_R90FRAMES`, registered
only in `cfgs/work_cfgs.txt`. The empty resolution propagated to
`makenemo:292-297`, so the component list and configuration directory were
empty. This retracts round 209's statement that the launcher copied the last
admitted build protocol.

The repair follows the successful round-166/169/172 protocol literally:

1. create fresh target `ORCA2_OMIP_L4_R210OMT1_P3` from registered reference
   `ORCA2_ICE_PISCES`;
2. copy every pinned `EXP00` and `MY_SRC` entry from
   `ORCA2_OMIP_L4_R90FRAMES`, and copy its CPP card under the new target name;
3. verify the closed 17-file source manifest in the target, touch the copied
   Fortran sources, and rebuild;
4. require the binary and the four compiled `r84_dump_frame` call sites before
   staging any ten-step run.

The OMT-1 physics remains exactly the round-209 edge: only
`ln_dynadv_OFF=.false.` and `ln_dynadv_vec=.true.` differ from OMT-0. The
compiled dispatcher is still
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:162-190`; the executing vector
external-mode association remains
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`. Drag,
momentum diffusion, tracer advection and tracer diffusion remain OFF. The
canonical deck hash is unchanged from round 209:
`d0cccebd76c9c4631c91e51a1af3a9b552d97f307c7c523020160c883fb5e73f`.

## Frozen-prediction disposition

| ID | disposition |
|---|---|
| R210-P1 | **REFUTED, retained:** the preregistered parent `ORCA2_OMIP_L4` is also a work configuration, not a registered reference. Reading `ref_cfgs.txt` and the admitted round-172 launcher named `ORCA2_ICE_PISCES`; after that one-line correction, static preflight passes. |
| R210-P2 | **CONFIRMED:** deck hash, 17 source hashes, frame patch hash, expected 80-frame inventory, admission gate and ladder gate are unchanged. Only target/evidence labels and build bootstrap moved. |
| R210-P3 | **UNMEASURED_WITH_SPEC:** only the operator-run build can prove the P3 binary and compiled call sites. |
| R210-P4 | **UNMEASURED_WITH_SPEC:** smoke, calibration, twins, month boundary and admission await operator execution. |
| R210-P5 | **CONFIRMED:** a planted work-config `-r` operand and a planted extra source-manifest member both refuse with `STATUS PLANT-FIRED`. |

## Validation and review

The committed `--preflight-only` run reports
`ORCA2_ROUND210_OMT1_PREFLIGHT_READY`. It re-renders the unchanged deck,
validates the closed source inventory, runs the writer preflight, and fires the
two deck plants plus the two new bootstrap plants. Evidence:
`round210/acquisition_preflight.log` and the JSON/plant files under
`round210/acquisition/`.

Focused round-209/210 tests pass 11/11. Shell syntax and Python compilation
pass. No `packages/` file changed, so no GYRE, DINO, tank, rung-0, rung-10 or
ORCA2 trajectory can move in this round.

Independent review and the one permitted full fidelity battery are reported
below after they complete. The citation gate is run both on this receipt and
on its cumulative default, with a shifted-span plant required to fire.

## OPEN

1. The operator runs the repaired launcher. The next round admits its record,
   runs both labelled OMT-1 ladders, records the month boundary, and names the
   first non-bit vector-branch statement.
2. The OMT-1 card remains prepared but unlanded until those gates complete.

No configuration or sea-ice decision is pending. The failed round-209 build
and the preregistered wrong parent name remain recorded as retractions.
