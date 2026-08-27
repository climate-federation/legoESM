# Preregistration: live NEMO `jpdyn_tau` slot wiring

Status: **DESIGN ONLY — STOP before running a rebuilt oracle.** This file and
the source patch/verifier must be committed before any patched NEMO output is
produced. The oracle tree is outside this writable sandbox, so the coordinator
owns the build and one-step execution.

## Claim and exact number

The instrumented DINO restart advertises `utrd_tau`/`vtrd_tau`, but the current
`trddump` store never receives `jpdyn_tau`. The proposed patch writes the exact
level-1 increment bracketed inside active `dyn_zdf`, divided by `rDt` so the
restart slot retains trend units. It deliberately does not enter the interval
accumulator or `nacc_trd`: this is a one-step source operand, not a new member
of the nine-term closure.

The verifier prints these registered numbers for the zonal slot:

1. `u_slot_nonzero_count` over the interior;
2. `u_lower_max_abs` over levels 2..jpk;
3. `u_reconstruction_max_abs = max(abs(rDt*utrd_tau[0] -
   (zdf_u1_poststress-zdf_u1_prestress)))`;
4. `u_reconstruction_normalized_error`, the RMS reconstruction error divided
   by the RMS bracket over nonzero bracket points.

The meridional slot and bracket are structural-zero controls and are printed
separately.

## Frozen classification

- **CONFIRMED_SLOT_WIRED** iff `u_slot_nonzero_count > 0`, every lower-level
  value is exactly zero, the meridional slot/bracket are exactly zero, and
  `u_reconstruction_max_abs <= 8*eps*max(1, max(abs(bracket)))`.
- **REFUTED_UNWIRED** iff `u_slot_nonzero_count == 0` while the zonal bracket
  has a nonzero value.
- **REFUTED_WRONG_SOURCE** iff a lower level is nonzero, the meridional
  structural-zero control fails, or `u_reconstruction_normalized_error >=
  0.05`.
- Anything else is **UNRESOLVED**. No placement or eta-ownership verdict is
  licensed by this slot check.

The verifier must also prove that planted nonzero-lower-level and
top-reconstruction violations make their corresponding gates fail.

## Registered source edit

Apply
`scripts/validate/ocean_fidelity/dino_1226/nemo_endwall_tau_slot.patch` at the
NEMO 5.0.2 root. The patch adds `trddump_tau` and calls it from the same
`kt==nit000`, one-rank bracket block in `cfgs/DINO/MY_SRC/dynzdf.F90`. The
routine zeroes the whole tau store and writes only level 1 from the already
captured post-minus-pre increment times `r1_Dt`.

## Exact coordinator handoff

From `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`:

```bash
git apply --check /path/to/legoesm/scripts/validate/ocean_fidelity/dino_1226/nemo_endwall_tau_slot.patch
git apply /path/to/legoesm/scripts/validate/ocean_fidelity/dino_1226/nemo_endwall_tau_slot.patch
conda run -n nemo-build ./makenemo -n DINO -m conda -j 4
cp -a cfgs/DINO/RUN_D180_1STEP_1R cfgs/DINO/RUN_D180_TAU_SLOT_1R
cd cfgs/DINO/RUN_D180_TAU_SLOT_1R
ln -sfn ../BLD/bin/nemo.exe nemo
rm DINO_00005761_restart.nc ocean.output run_1step.log
mpirun -np 1 ./nemo >run_1step.log 2>&1
```

The copied namelist is frozen at `nn_it000=nn_itend=5761` and reads the
step-5760 restart. Before accepting the output, record the patched source
SHA-256 values, NEMO executable SHA-256, namelist SHA-256, and `mpirun` exit
status. Then, from the legoESM checkout:

```bash
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/verify_nemo_endwall_tau_slot.py \
  --run-dir /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_D180_TAU_SLOT_1R \
  --restart DINO_00005761_restart.nc --rdt-seconds 5400
```

STOP after printing the classification. A later wind-placement/eta run may
cite `CONFIRMED_SLOT_WIRED`; it may not reinterpret any other classification.

