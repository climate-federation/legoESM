# Preregistration: live NEMO named `jpdyn_tau` slot wiring

Status: **ROUND-2 RETRACTION APPLIED.** The first
registration incorrectly called the applied `dynzdf` surface increment divided
by `rDt` NEMO's named `utrd_tau` diagnostic. Independent mechanism review
refuted that identity: active `trddyn.F90:164-168` defines the named diagnostic
without the MLF factor 1/2 and with `e3u/e3v(...,Kmm)`, whereas active
`dynzdf.F90:536-560` applies the half-sum using after-level thickness. The old
`CONFIRMED_SLOT_WIRED` label is therefore retracted here before another run.
That run demonstrated diagnostic transport for the applied-increment operand;
it did not wire the named diagnostic.

This corrected arm is a one-step CPU diagnostic write, not a free-running twin
or GPU ownership arm. The read-only oracle and its retained artifacts remain
unchanged; the source edit is applied in the isolated temporary clone.

The first corrected run classified **REFUTED_WRONG_SOURCE** and is retained as
a failed-instrument control. Runtime provenance showed `jpi=56`, `jpj=203`,
while the active untiled `T2D(0)` bounds are `Nis0:Nie0=3:54` and
`Njs0:Nje0=3:201` (52 by 199). The hook's explicit `(jpi,jpj)` dummy therefore
read beyond each 52-by-199 actual argument; its nonzero meridional slot and
0.579 zonal storage error were instrument corruption, not physics. Before the
next run, the registered edit changes the dummy to NEMO's native `T2D(0)` and
copies with `DO_2D(0,0,0,0)`. No number, gate, or scientific classification
changed. The failed run may not be cited as a wind-placement measurement.

## Claim and exact numbers

The dormant restart slot must receive the exact 2-D arrays that active
`trddyn.F90` supplies to `iom_put("utrd_tau")` and `iom_put("vtrd_tau")`.
The patch therefore calls `trddump_tau(z2dx,z2dy)` immediately after those
arrays are computed and writes one-step stream references from the same
in-memory arrays. These are two output paths, not independent evidence. It does
not multiply by `r1_Dt`, and it deliberately does not
enter the interval accumulator or `nacc_trd`.

The verifier prints these frozen numbers:

1. `u_slot_nonzero_count` and `u_reference_nonzero_count`;
2. `u_lower_max_abs` over levels 2..jpk and the meridional structural-zero
   maxima;
3. `u_storage_max_abs = max(abs(utrd_tau[0]-u_reference))`;
4. `u_storage_normalized_error`, the RMS storage error divided by the RMS
   reference over nonzero reference points;
5. separately, wet-only `u_named_to_applied_rms_ratio`, correlation,
   pointwise ratio mean error from 2, ratio standard deviation, and maximum
   pointwise deviation, where
   `applied = (zdf_u1_poststress-zdf_u1_prestress)/rDt`.

## Frozen classifications

- **CHECKED_CLEAN_WRITE_ONLY** iff `u_slot_nonzero_count > 0`, every
  lower-level value is exactly zero, the meridional slot/reference are exactly
  zero, and both storage maxima are no larger than
  `8*eps*max(1,max(abs(u_reference)))`.
- **REFUTED_UNWIRED** iff the zonal slot is zero while its reference is nonzero.
- **REFUTED_WRONG_SOURCE** iff a lower level is nonzero, a meridional
  structural-zero control fails, or `u_storage_normalized_error >= 0.05`.
- Anything else is **UNRESOLVED**.

The former source distinction
`CONFIRMED_NAMED_AND_APPLIED_DIFFER` and its 2.0966766111 ratio are
**RETRACTED**. The union mask admitted 324 dry coastal U faces, and the
normalized-error gate had no reachable REFUTE state. On every wet U face,
active `dynzdf` uses `zDt_2=rDt/2`, so the named no-half diagnostic is exactly
twice the applied increment. The corrected verifier prints that wet-only
identity and requires both absolute mean-ratio error from 2 and pointwise wet
ratio standard deviation `<=1e-8`; maximum pointwise deviation is descriptive
only. A planted single-face mutation must make the ratio standard deviation
fail, and a planted uniform +1 ratio shift must make the mean gate fail. It
issues no source-distinction classification.
This is an arithmetic/source-plumbing check, not independent physics evidence.

The verifier must prove that planted nonzero-lower-level, top-storage, and
wet-ratio violations fire their corresponding gates.

## Registered source edit

Apply `nemo_endwall_tau_slot.patch` at the NEMO 5.0.2 root. It adds
`trddump_tau`, calls it from the active `CASE(jpdyn_zdf)` in
`cfgs/DINO/MY_SRC/trddyn.F90`, and dumps the exact named arrays through a
second output path from the same in-memory values.
The existing `dynzdf` pre/post-stress streams remain untouched and provide the
separate applied-increment operand.

## Exact corrected temporary CPU execution

The existing isolated clone is reset only at the three touched override files,
using byte-for-byte copies from the read-only oracle, before applying the
corrected patch. A new run directory prevents confusion with the retracted
artifact.

```bash
SLOT_ROOT=/tmp/nemo-tau-slot-eaef2a1e1
ORACLE=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
cp "$ORACLE/cfgs/DINO/MY_SRC/trddump.F90" "$SLOT_ROOT/cfgs/DINO/MY_SRC/"
cp "$ORACLE/cfgs/DINO/MY_SRC/trddyn.F90" "$SLOT_ROOT/cfgs/DINO/MY_SRC/"
cp "$ORACLE/cfgs/DINO/MY_SRC/dynzdf.F90" "$SLOT_ROOT/cfgs/DINO/MY_SRC/"
git -C "$SLOT_ROOT" apply \
  /tmp/codex-basin-rect/scripts/validate/ocean_fidelity/dino_1226/nemo_endwall_tau_slot.patch
CONDA_NO_PLUGINS=true conda run -n nemo-build \
  "$SLOT_ROOT/makenemo" -n DINO -m conda -j 4
mkdir "$SLOT_ROOT/cfgs/DINO/RUN_TAU_SLOT_NATIVE_1R"
cp "$ORACLE/cfgs/DINO/RUN_D180_1STEP_1R/namelist_cfg" \
   "$ORACLE/cfgs/DINO/RUN_D180_1STEP_1R/namelist_ref" \
   "$SLOT_ROOT/cfgs/DINO/RUN_TAU_SLOT_NATIVE_1R/"
ln -s "$ORACLE/cfgs/DINO/RUN_TRAJ/DINO_00005760_restart.nc" \
  "$SLOT_ROOT/cfgs/DINO/RUN_TAU_SLOT_NATIVE_1R/DINO_00005760_restart.nc"
ln -s "$SLOT_ROOT/cfgs/DINO/BLD/bin/nemo.exe" \
  "$SLOT_ROOT/cfgs/DINO/RUN_TAU_SLOT_NATIVE_1R/nemo"
cd "$SLOT_ROOT/cfgs/DINO/RUN_TAU_SLOT_NATIVE_1R"
./nemo >run_1step.log 2>&1
```

Before verification, print SHA-256 for patched `trddump.F90`, `trddyn.F90`,
unchanged `dynzdf.F90`, executable, namelists, input/output restart, both named
reference streams, and all four `dynzdf` bracket streams. Then run:

```bash
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/verify_nemo_endwall_tau_slot.py \
  --run-dir /tmp/nemo-tau-slot-eaef2a1e1/cfgs/DINO/RUN_TAU_SLOT_NATIVE_1R \
  --restart DINO_00005761_restart.nc --rdt-seconds 5400
```

STOP after the classifications.

## Coordinator handoff

On a writable copy of the oracle, apply and build the same patch, copy
`RUN_D180_1STEP_1R` to a fresh `RUN_D180_TAU_SLOT_NAMED_1R`, point its `nemo`
symlink at the rebuilt executable, and run its frozen step-5761 namelist. Hash
the same sources, executable, inputs, output, references, and bracket streams;
then invoke the verifier above with that run directory. A later wind/eta run
may cite only `CHECKED_CLEAN_WRITE_ONLY` and must state that slot/reference
share one in-memory array.

## Refused-execution audit inherited from the first arm

The temporary clone required the oracle's untracked `cfgs/work_cfgs.txt`,
`arch/arch-conda.fcm`, and `cfgs/DINO/MY_SRC`; these were copied before build.
The sandbox denied PMIx listener creation, so the preregistered one-rank direct
executable fallback is used. Those operational changes produced no scientific
classification and changed no threshold.
