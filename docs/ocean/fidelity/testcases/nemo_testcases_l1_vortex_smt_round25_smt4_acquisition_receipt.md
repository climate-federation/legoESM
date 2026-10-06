# Receipt — VORTEX_SMT round 25 (lane round 237): SMT-4 acquisition

**Status: STOPPED_FOR_RECORD.** This round preregistered the SMT-4 rung,
prepared and fail-closed-preflighted the operator-run NEMO acquisition, and
changed no production model code or certified trajectory. The required NEMO
record does not yet exist, so no ladder row, first non-bit statement, or
landing verdict is claimed.

Base commit: `33c754c716ccf3fc6c50692ca9c30041ff7a7a64` (round 236).
Implementation commit: `93dbb587a723019d1bff616ca149fd1c131a4c18`.
Preregistration commit: `114067f336dff0258f5acf4f794906e8b59e76b1`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/`.

## 1. Outcome

The requested one-module rung is prepared mechanically:

* SMT-3 is changed only in `&namdyn_ldf` to div-rot, level Laplacian
  momentum diffusion with coefficient mode 20, `rn_Uv=0.1 m/s`,
  `rn_Lv=10 km`, and `rn_ahm_b=0`;
* new targets are `VORTEX_SMT4_VEC_R8_OMIP_L1` and
  `VORTEX_SMT4_VEC_R8_OMIP_L1_P3`; no existing target is modified;
* the 10-step arm reuses the existing self-describing stage-1/2/3 momentum
  record, including the accumulator immediately before and after `dyn_ldf`;
* the 100-day arm reuses the newly admitted SMT-4 executable by its binary
  manifest and writes the shipped 3,000-step trajectory at daily cadence;
* restart byte identity, parse-to-EOF, required names, resolved namelist
  values, `STOP 0`, and header/name/truncation plants all fail closed.

The operator command is:

```text
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smt_round25_smt4/run.sh --run
```

The acquisition script carries the clean-tree commit stamp into its manifest.
It uses shell `SECONDS`, never `/usr/bin/time`, and creates new build targets.

## 2. Preregistered predictions

| prediction | round-237 evidence | verdict |
|---|---|---|
| R25-P1 controlled one-module deck | dry application prints the inherited SMT-2 and SMT-3 hunks plus only the new `namdyn_ldf` hunk | CONFIRMED for patch construction; runtime resolution pending |
| R25-P2 resolved branch | exact runtime `ocean.output` checks are installed for the complete tuple and iso-level Laplacian dispatch | UNMEASURED |
| R25-P3 passive self-describing record | parser, restart comparison, required groups, and header/name/truncation plants are installed; synthetic controls pass | UNMEASURED on NEMO |
| R25-P4 run sanity | both arms refuse nonzero launch, absent `STOP 0`, missing restart/frame, or non-finite state | UNMEASURED |
| R25-P5 first new boundary is post-`dyn_ldf` | all three stages record pre/post-LDF momentum accumulators | UNMEASURED |
| R25-P6 acquisition-only disposition | no file under `packages/` or `src/` changed and no NEMO trajectory ran | CONFIRMED |

No failed prediction is hidden. P1 is only a construction result until the
new executable echoes the resolved card. P2--P5 remain open.

## 3. Preflight and controls

The committed wrapper was run without `--run`. The complete log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/preflight.log` and
ends with:

```text
GFORTRAN_SYNTAX_PASS .../vortex_r16_stage_terms.F90
PREFLIGHT_OK  variant smt4vec: instrument and deck patches apply to the shipped sources
GFORTRAN_SYNTAX_PASS .../vortex_r16_stage_terms.F90
PREFLIGHT_OK  variant smt4vec100d: instrument and deck patches apply to the shipped sources
ROUND237_SMT4_PREFLIGHT_PASS /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/oracle_vortex_smt4
```

The checker follows the self-describing-record rule: magic plus header, then
named `(rank,n1,n2,n3,payload)` groups to EOF. It predicts no record size or
header tuple by hand. The acquisition itself runs the `header`, `field-name`,
and `truncated` plants and refuses if any exits zero.

Focused controls before finalisation:

```text
tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round237_smt4_record.py
7 passed in 0.16s

tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round224_smt3_record.py
tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round237_smt4_record.py
16 passed in 0.25s
```

`bash -n` on the shared driver and wrapper and `git diff --check` also pass.
The final citation and focused-gate results are recorded below after their
committed inputs were exercised.

## 4. Scientific gate table

This is an acquisition-only round. The scientific rows cannot be scored and
are not inferred from the dry preflight.

| required row | before | after | verdict |
|---|---:|---:|---|
| SMT-4 kt=1..10 registry | no SMT-4 card/record | not run | UNMEASURED |
| first-over-bar and first non-bit producer | SMT-3 certified arm | not run | UNMEASURED |
| SMT-4 100-day trajectory | no SMT-4 card/record | not run | UNMEASURED |
| GYRE ladder | round-223/207 lineage certified arm | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| GYRE day 30 / 240 / 360 | 2.3432437414839976e-06 / 6.5817049818294640e-05 / 5.4077372201617810e-05 K | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| SMT-1/2/3, eight flat/SMT VORTEX cards, tanks | certified round-236 arms | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| DINO month | 2.053801169e-03 K | unchanged by docs/instrument work | NOT EXECUTED; no production change |
| ORCA2 | coefficient source and merge pointer only | no ORCA2 code/card changed | UNMEASURED-with-spec |

Every future moved row remains subject to registration. No AT-BAR row is
claimed to move, and no exception to a landing rule is requested.

## 5. Configuration choices

**UNASKED list: EMPTY.** ORCA2 rung 0 selects level Laplacian momentum
diffusion and a file-backed 3-D coefficient at
`orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:388-392`.
Decision 93 explicitly authorises the idealised-card stand-in coefficient
mode 20. The complete companion tuple comes from
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:1104-1135`. Geometry, EOS, tracer
diffusion, vertical mixing, drag, momentum advection, vorticity, pressure
gradient, barotropic program, timestep, and run length remain SMT-3's admitted
values.

## 6. Compiled-source record

The new SMT-4 target has not been built, so this receipt cites the compiled
SMT-3 base whose scientific program the new deck changes. NEMO declares and
reads `namdyn_ldf` reference-before-card at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-185`.
For z/partial-step geometry, level Laplacian resolves to `np_lap` at
`:221-276`; coefficient mode 20 computes `zUfac=0.5*rn_Uv` and calls
`ldf_c2d` at `:311-346`.

The compiled dispatcher maps `np_lap` to `dynldf_lev_lap` at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90:81-90`.
That operator forms curl and divergence with the live partial-cell
`e3f_3d`, `e3t_3d`, `e3u_3d`, and `e3v_3d` factors and adds them to the U/V
RHS at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.
The stage program calls `dyn_ldf` before the implicit vertical momentum solve
at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:387-405`.

These citations prove the branch requested by the deck; they do not pretend
the absent SMT-4 executable has run. The next receipt must cite the new
target's own `BLD/ppsrc/nemo`, confirm its resolved `ocean.output`, and admit
its passive record before making a physics claim.

## 7. Independent review and final gates

Pending against the committed receipt and citation map. The exact external
review verdict, citation-gate counts, shifted-citation plant, and final
focused-suite summary are appended before this receipt is final.

## 8. OPEN — round 238

1. Operator runs the committed acquisition. Admit only if the plain and
   instrumented step-10 restarts are byte-identical, every named group parses
   to EOF, all three plants exit nonzero, both runs reach `STOP 0`, and the
   resolved `ocean.output` tuple is exact.
2. Read and cite the new targets' compiled source. Add the explicit
   `VORTEX_SMT4_VEC-zps` legoESM card, prove geometry and initial state against
   NEMO, and score the 50-row kt=1..10 registry plus the 100-day checkpoints.
3. Compare each stage's pre/post-`dyn_ldf` accumulator from NEMO's recorded
   entry. If post-LDF is the first non-bit boundary, walk the level-Laplacian
   div/curl operator operand by operand over partial cells. If pre-LDF is
   already non-bit, keep R25-P5 as REFUTED and walk the earlier boundary.
4. Name the first non-bit statement with the new compiled-source citation and
   write the exact ORCA2 rung-0 operand pointer. Any production candidate is a
   later round under all GYRE/year, SMT/flat VORTEX, tanks, generic GYRE,
   private DINO-month, citation, plant, and independent-review gates.

No production physics, configuration default, or carried state landed in
round 237.
