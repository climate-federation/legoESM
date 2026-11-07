# Receipt — VORTEX_SMT round 12 (lane round 224): SMT-3 acquisition

**Status: STOPPED_FOR_RECORD.**  This round preregistered the SMT-3 rung,
prepared and fail-closed-preflighted the operator-run NEMO acquisition, and
changed no production model code or certified trajectory.  The required NEMO
record does not yet exist, so no ladder row, first non-bit statement, or
landing verdict is claimed.

Base commit: `889afc57d8fd4efcca6a72eb7dd6a076197ca3fe` (round 223).
Writable-clone commits begin at `3884fe7cdd`.  Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round12_smt3_tracer_diffusion.md`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/`.

## 1. Outcome first

The requested one-module rung is mechanically prepared:

* SMT-2's deck is changed only in `&namtra_ldf` to laplacian, standard
  isoneutral mixing with MSC, coefficient mode 20, `rn_Ud=0.018 m/s`, and
  `rn_Ld=200 km`;
* new targets are `VORTEX_SMT3_VEC_R8_OMIP_L1` and
  `VORTEX_SMT3_VEC_R8_OMIP_L1_P3`; neither older build is modified;
* the 10-step arm writes the existing self-describing step/stage record plus
  a stage-3 post-`tra_ldf` T/S RHS boundary;
* the 100-day arm reuses the admitted SMT-3 binaries by their manifest and
  writes the shipped 3,000-step trajectory at daily cadence;
* restart byte identity, parse-to-EOF, required names, resolved namelist
  values, `STOP 0`, and three planted corruptions all fail closed.

The operator command is:

```text
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smt_round12_smt3/run.sh --run
```

The script is resumable only after an arm has an `ADMITTED` JSON and the
10-step binary manifest.  A partial directory is refused rather than reused.
It uses shell `SECONDS`; it never invokes `/usr/bin/time`.

## 2. Preregistered predictions

| prediction | round-224 evidence | verdict |
|---|---|---|
| R12-P1 controlled one-module deck | dry application prints only the inherited SMT-2 hunks plus the one `namtra_ldf` hunk | CONFIRMED for patch construction; runtime resolution pending |
| R12-P2 resolved output | runtime-only `ocean.output` check is installed for every active and companion value | UNMEASURED |
| R12-P3 passive self-describing record | parser, restart comparison and header/name/truncation plants are installed; synthetic tests pass | UNMEASURED on NEMO |
| R12-P4 10-step and 100-day run sanity | both arms are fail-closed on nonzero `mpirun`, missing restart and absent `STOP 0` | UNMEASURED |
| R12-P5 kt=1 remains at bar and no earlier first-over-bar | requires the record and legoESM card | UNMEASURED |
| R12-P6 first new boundary is post-LDF | stage-3 post-LDF boundary is now recorded; internal walk awaits it | UNMEASURED |
| R12-P7 no physics landing in acquisition round | no file under `packages/` or `src/` changed | CONFIRMED |

No failed prediction is hidden.  P1 is deliberately only a construction
result until NEMO echoes the resolved values; P2--P6 remain open.

## 3. Preflight and controls

The committed wrapper was run without `--run`.  The complete log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/preflight.log` and
ends with:

```text
GFORTRAN_SYNTAX_PASS .../vortex_r18_tracer_terms.F90
PREFLIGHT_OK  variant smt3vec: instrument and deck patches apply to the shipped sources
GFORTRAN_SYNTAX_PASS .../vortex_r18_tracer_terms.F90
PREFLIGHT_OK  variant smt3vec100d: instrument and deck patches apply to the shipped sources
ROUND224_SMT3_PREFLIGHT_PASS /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/oracle_vortex_smt3
```

The first preflight did not pass silently: it refused a malformed deck-patch
hunk count.  Commit `ba1cbaa455` corrects `-155,7` to the measured eight-line
old extent; the clean rerun above is the evidence used here.

The checker follows the self-describing-record rule: magic plus header, then
named `(rank,n1,n2,n3,payload)` groups to EOF.  It predicts no record size or
header tuple by hand.  Stage 1/2 require the original 15 groups; stage 3
requires 17 and names `ldf_t` and `ldf_s`.  The acquisition itself runs the
`header`, `field-name`, and `truncated` plants and refuses if any exits zero.

Focused test command (CPU, fp64 policy environment):

```text
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python -m pytest -q tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round224_smt3_record.py
```

Result: **7 passed in 0.10s**.  `bash -n` on both drivers, Python bytecode
compilation of the checker, and `git diff --check` also pass.

After the citation map was committed, the combined focused suite reported
**24 passed in 3.92s** (the seven SMT-3 controls plus all seventeen receipt-
citation-gate tests).  The round receipt citation gate found 7 citations, 0
unmapped, and 0 failures.  Shifting the compiled coefficient citation
`ldftra.f90:354-390` by two lines exited **1** with
`SYMBOL-NOT-AT-LINE`, so the control fires.

## 4. Rule-12 / Decision-43/45 table

This is an acquisition-only round, so the scientific rows cannot be scored.
The table is explicit to prevent a record-preparation result from being
mistaken for a trajectory result.

| required row | before | after | verdict |
|---|---:|---:|---|
| SMT-3 kt=1..10 registry | no SMT-3 card/record | not run | UNMEASURED |
| first-over-bar | SMT-2 first-over-bar kt=2 | not run | UNMEASURED |
| GYRE ladder | round-223 certified arm | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| GYRE day 30 / 240 / 360 | 2.3432437414839976e-06 / 6.5817049818294640e-05 / 5.4077372201617810e-05 K | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| SMT-1/SMT-2, six flat VORTEX cards, tanks | certified round-223 arms | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| DINO month | 2.053801169e-03 K | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| ORCA2 | pointer only | no ORCA2 code/card changed | UNMEASURED-with-spec |

Every future moved row remains subject to registration.  No AT-BAR row is
claimed to move, and no exception to a landing rule is requested.

## 5. Configuration choices

**UNASKED list: EMPTY.**  The six active values are copied from ORCA2 rung 0;
the eight companion values are the NEMO reference values that rung 0 leaves
unset.  Geometry, EOS, vertical physics, drag, momentum, barotropic program,
run length and timestep remain those already admitted for SMT-2.  The only
10-step/100-day difference is the established measurement cadence.

## 6. Compiled-source record

The source card is not interpreted from comments alone.  ORCA2 rung 0 pins
the active switch/coefficient tuple in
`orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:319-326`.
The values it leaves unset resolve from
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:918-933`.

In the presently compiled SMT-2 base, NEMO's namelist declaration contains
exactly those fields at
`VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:214-218`; NEMO reads
the reference first and the configuration second at `:231-233`.  The active
laplacian + z-partial-step + standard-isoneutral branch resolves
`nldf_tra=np_lap_i` at `:261-283`.  Coefficient mode 20 calls `ldf_c2d` with
the laplacian factor `0.5*rn_Ud` at `:354-390`.  The compiled tracer program
then dispatches `np_lap_i` to `traldf_iso_lap` at
`VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traldf.f90:105-122`.

These citations establish the exact branch the new deck requests; they do
not pretend the absent SMT-3 build has already run.  The next receipt must
replace/confirm them against the new target's own compiled `BLD/ppsrc/nemo`
and quote its resolved `ocean.output` before making a physics claim.

## 7. Independent review

The required separate command was attempted against the committed clean tree:

```text
codex exec --sandbox read-only -C <writable-clone> <adversarial review prompt>
```

It exited 1 before reading the diff.  The complete tool verdict was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

**Independent review unavailable in-sandbox.**  There is no `SHIP`, `HOLD`,
or `DO NOT SHIP` verdict to quote, and none is invented.  Per operator note C,
the round continues; the limitation is explicit.  Local adversarial checks
did find and correct two fail-closed issues before finalisation: the malformed
deck-patch extent stopped the first preflight, and the citation gate refused
three ambiguous repeated symbols until their exact occurrences were pinned.

## 8. OPEN — round 225

1. Operator runs the committed acquisition script.  Admit only if the plain
   and instrumented restarts are byte-identical, every named group parses,
   all three plants exit nonzero, both runs reach `STOP 0`, and the resolved
   `ocean.output` tuple is exact.
2. Read and cite the new target's own compiled source.  Build an explicit
   `VORTEX_SMT3_VEC-zps` legoESM card whose every switch is stated; prove its
   geometry and initial state identical to SMT-2.
3. Score kt=1..10 against the admitted record, register all 50 rows, and name
   the first-over-bar row.  R12-P5 is refuted if kt=1 leaves the bar or the
   first-over-bar moves earlier than kt=2.
4. From NEMO's recorded stage entry, compare pre-LDF and post-LDF tracer RHS.
   If post-LDF is the first non-bit boundary, walk `ldfslp` and
   `traldf_iso_lap` in compiled order over partial cells, including bottom
   slope limiting and MSC.  If pre-LDF is already non-bit, retain R12-P6 as
   REFUTED and walk that earlier boundary instead.
5. Run and score the 100-day arm, then evaluate any named statement under all
   standing GYRE/year, SMT, flat VORTEX, tanks, generic-card, DINO-month,
   citation, plant, and independent-review gates.  Write the exact ORCA2
   rung-0 operand pointer; do not infer it from the VORTEX result.

No production physics, configuration default, or carried state landed in
round 224.
