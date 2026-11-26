# ORCA2 round 172 — passive RHS replay names HPG at kt=8

**Status:** `STOPPED_FOR_RECORD`  
**Claim label:** every scientific number below is **independent**: the legoESM
rung-0 trajectory starts from the rung's own initial state, not NEMO's recorded
step entry.  
**Producer:** clean committed CPU/fp64 tree; evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round172/`.

## Question and compiled program

Round 171 admitted NEMO's rank-complete kt=8 accumulator record but refused
every legoESM component observer because each observer changed the executable.
This round therefore asks only which source-ordered accumulator is already
non-bit when evaluated offline from the existing passive trace.

The compiled rung-0 program calls HPG first, followed by LDF and VOR
(`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:145-157`), then KEG and ZAD
(`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:168-179`). The compiled HPG
is `hpg_sco`: its surface statements form `zhpi/zhpj`, `zuap/zvap`, then
overwrite the RK3 accumulator
(`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90:386-402`); its interior
recurrence and overwrite are
`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90:411-434`. These citations are from the record-producing
`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo` tree, not an uncompiled source arm.

## Instrument admission

The final gate is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round172_passive_rhs_replay.py`.
It re-admits both R170 rank shards, advances the ordinary and pre-existing live
trace runs through kt=1..7, and requires `T/S/u/v/ssh` to be
`np.array_equal`. HPG, LDF, VOR and KEG returned by the separate evaluator also
must equal the live trace bit-for-bit at every one of those seven steps. All 35
state comparisons and all 56 operator/face comparisons passed. The one-ULP
planted violation fired.

Two broader bridges were tried and retained as loud refusals:

- R172-P2 compared an existing trace RHS with a separately compiled graph. It
  was state-passive but not graph-identical at kt=2, so no score was emitted.
- R172-P2a/P3a tried to admit ZAD and a same-graph completed-RHS closure. ZAD
  changed at kt=2 and the production total contains terms outside the five-call
  accumulator, so those claims were refused. The final gate scores only the
  exact four-operator prefix.

Evidence hashes:

| Artifact | SHA-256 |
|---|---|
| `passive_rhs_replay.json` | `17596fef8ef67e79068c0f7b6f8671c948f05fb36700045dc5852be6b6735360` |
| `passive_rhs_replay.log` | `5c355d2e50208c046c5fcd58f0b607a22b0fa2d688b60125e2bbe8c1df3b1ffe` |

## Result — independent

The first non-bit accumulator and the first explosive U accumulator are both
`after_hpg`. The preregistered VOR prediction R172-P4 is **REFUTED**.

| independent kt=8 row | wet differing / wet cells | max absolute error | RMS error | legoESM max | NEMO max | argmax `[j,i,k]` |
|---|---:|---:|---:|---:|---:|---:|
| after HPG, U | 411,276 / 411,276 | `1.5835360371918837e51 m s-2` | `1.722678792952183e49 m s-2` | `1.5835360371918837e51` | `4.194087050767948e-05` | `[147,47,2]` |
| after HPG, V | 413,856 / 413,856 | `1.6377650405462966e51 m s-2` | `1.281941378896576e49 m s-2` | `1.6377650405462966e51` | `6.481611203365945e-05` | `[146,48,2]` |

At the U argmax, legoESM is `1.5835360371918837e51 m s-2` while NEMO is
`2.9570326936326197e-07 m s-2`. Later admitted prefix rows retain the same
explosive maxima; they do not create the error. The HPG U row differs on all
799,200 stored cells before the wet mask, so this is not a one-cell fold or
signed-zero effect.

Prediction dispositions are: R172-P1 confirmed; R172-P2 refuted; R172-P2a
refuted; R172-P2b confirmed; R172-P3 refuted; R172-P3a refuted; R172-P3b
confirmed; R172-P4 refuted; R172-P5 confirmed; R172-P6 confirmed. Failed
predictions remain in the preregistration.

## Acquisition prepared

The R170 accumulator record has only the five post-operator accumulators; it
does not carry HPG's kt=8 `rhd`, live `e3w`, live `gdept_z0`, reciprocal metric,
or `zhpi/zuap` statement boundaries. Therefore this round cannot select a
single HPG operand without post-hoc inference.

The operator launcher is:

`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-orca2-186302702/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round172_hpg8_acquisition/run.sh`

It creates the new `ORCA2_OMIP_L4_R172HPG8` target from the content-pinned R170
build. Its patch is additions-only and emits one self-describing kt=8 file per
rank. Each field header carries its own name, rank and dimensions; the checker
derives payload lengths from those headers, proves exactly-once rank coverage,
and requires all 20 ocean restarts to be byte-identical to R170. Six corrupt
record/restart plants and a writer-layout plant must fire. `--preflight-only`
passed the exact patch, preprocessing and Fortran syntax proof; the layout
plant exited 69 with `STATUS PLANT-FIRED layout`.

## Gates and scope

This round changes no `packages/` model file and makes no configuration choice.
Consequently there is no GYRE, DINO, tank, rung-0, or rung-7 trajectory to
re-certify. The successful result is a measurement and a fail-closed
acquisition request, not a physics landing.

Verification:

- focused round-172 replay tests: 9 passed;
- focused HPG acquisition tests: 8 passed;
- receipt citation gate: PASS, four citations, zero unmapped/failing entries;
- cumulative default citation gate: PASS, 274 citations, zero unmapped/failing
  entries;
- citation-shift plant: fired, exit 1;
- `tests/ocean/fidelity -n 12`: 2,750 passed, 7 skipped, 4 failed in 2,392.52
  seconds. The failures are the two standing worktree-stamp ratchets,
  `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants`, and the
  stale certified-harness pin in
  `test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`.
  None of their test or gate paths differs from the incoming `aa7c7a03b` tree.
  Full log SHA-256:
  `f5609a97bbbca842bba62e03f0d25394a54beaa28d42339a3a407acebd01bf58`.

The requested separate review verdict is: **independent review unavailable
in-sandbox**. `codex exec --sandbox read-only` failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
The retained review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

1. Operator: run the committed HPG8 launcher with `--run`.
2. Admit both rank shards and the additions-only restart comparison.
3. At kt=8, compare `rhd`, `e3w`, `gdept_z0`, `r1_e1u/r1_e2v`, then
   `zhpi/zhpj`, `zuap/zvap`, and their sums in the compiled statement order.
   The first non-bit operand or statement owns round 173.
4. Only a cited NEMO statement may then be promoted under the complete B57
   atomic-unit gates; until then the halo unit remains private and held.
