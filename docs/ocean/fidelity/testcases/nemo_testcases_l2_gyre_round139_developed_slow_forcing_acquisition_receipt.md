# NEMO testcase L2 GYRE phase 3 — round 139 developed slow-forcing acquisition receipt

Date: 2026-09-21

Incoming lane tip: `b096d2f2e895f67b9375c140ec686365cf415f7c`

Preregistration commit: `73577cd29`

Record-tool commit: `3e3fa54c5`

Layout-control correction commit: `c7da3fb56`

Status: **STOPPED_FOR_RECORD — the minimum passive writer, parser, replay,
plants, and operator-ready acquisition script are committed, but the run
refused with exit 64 before `makenemo` because this sandbox cannot write the
NEMO configuration directory. No Round-139 NEMO target, run directory, or
operand record exists. Consequently every frozen scientific prediction is
UNMEASURED and no production statement or physics changed.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round139/`

## Outcome first

Round 138's first production-JIT non-bit boundary remains the only admitted
result: at step 1081 the completed frozen forcing differs on 580 / 580 wet U
faces by at most `4.2854247978022983e-13` and on 570 / 570 wet V faces by at
most `4.4333086294645174e-13`. The compiled program first copies `Ue_rhs` and
`Ve_rhs`, calls `dyn_cor_2D`, and subtracts its output through the native masks
at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325`.
The called four-point U/V Coriolis statements are
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:1336-1359`.

The admitted Round-137 record contains only the result of that subtraction.
It cannot discriminate incoming RHS error from initializing Coriolis error.
Round 139 therefore names no new first non-bit operand and makes no arithmetic
claim from the result alone.

## Frozen record contract

The additive source card records exactly six native owned 32-by-22 fp64
arrays immediately around the already executing call and subtraction:
incoming `Ue_rhs`/`Ve_rhs`, returned `zu_trd`/`zv_trd`, and final
`zu_frc`/`zv_frc`. The stream contract is:

`16 + 11*4 + 6*32*22*8 = 33,852 bytes`.

The parser rejects a wrong magic, version, step, time level, grid, scalar
width, owned extent, field count, truncation, trailing bytes, non-finite
values, or a non-bit replay of the two compiled subtractions. It records the
time level as `before`; the central time-level registry independently carries
the same classification. The run script:

- clones `GYRE_PISCES` under the new target
  `GYRE_OMIP_L2_P3_SM_R139SLOW`;
- copies the Round-137 `EXP00`, `MY_SRC`, and cpp card file by file;
- applies only the additive source patch and syntax-proves the result;
- preserves the exact Round-137 namelist and 1081-step run;
- passively admits the step-1080 restart, step-1081 process stream, external
  stream, and QCO stream byte for byte; and
- stamps the new record and runs header, truncation, replay-ULP, stamp, and
  passive-admission controls before printing `READY`.

It uses bash `SECONDS`, not unavailable `/usr/bin/time`, and every refusal
path emits a named `REFUSE:` line.

## Preflight, control, and blocked acquisition

The first dry preflight correctly refused with exit 66, but for an instrument
defect rather than a source-card defect. Its layout assertion counted the
new `zu_trd`/`zv_trd` continuation token globally; an inherited Round-137
writer contains the same continuation at another boundary. The code that
printed the false refusal was corrected to bind the continuation to the
unique Round-139 incoming-field WRITE. The failed result remains recorded in
`preflight.log`; it is retracted and is not evidence against the source card.

The corrected dry preflight exited 0 after printing
`SYNTAX_PROOF_PASS dynspg_ts.f90` and
`ROUND139_DEVELOPED_SLOW_FORCING_PREFLIGHT_READY`. The layout plant replaced
the final-output WRITE with a duplicate incoming WRITE and exited 69 after
printing both `STATUS PLANT-FIRED: layout` and a named refusal. These checks
exercise source-card structure only; they do not claim a NEMO record exists.

The real `--run` attempt then exited 64 with:

```text
REFUSE: NEMO configuration directory is not writable in this sandbox
```

The refusal precedes target creation and `makenemo`. Both the target
configuration and target run path are absent. No NEMO binary or model was run.

| preregistered item | observed result | disposition |
|---|---|---|
| additive six-field writer and exact 33,852-byte reader contract | syntax-proved; corrected layout check passes | CONFIRMED structurally |
| layout plant exits nonzero with named marker | exit 69, `STATUS PLANT-FIRED` | CONFIRMED |
| acquired record, replay, ancestry, passivity, stamps, and four record plants | acquisition stopped before target creation | UNMEASURED |
| first non-bit operand is incoming `Ue_rhs`, then `Ve_rhs` | no operand record | UNMEASURED |
| initializing Coriolis and masks are BIT | no operand record | UNMEASURED |
| NEMO-incoming directed arm makes final forcing BIT | no operand record | UNMEASURED |

None of the unmeasured predictions is promoted to a finding.

## Review, citations, and focused verification

The required separate read-only Codex command was invoked against the whole
Round-139 diff and evidence. It could not initialize its in-process app-server
inside the read-only sandbox, so it emitted no verdict. Its output was,
verbatim:

```text
Error while loading conda entry point: conda-anaconda-tos (cannot import name 'validate_prefix_exists' from 'conda.cli.install' (/home/dbalwada/miniconda3/lib/python3.13/site-packages/conda/cli/install.py))
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Independent review unavailable in-sandbox; the standing operator rule permits
the round to continue. No `DO NOT SHIP` verdict exists, and nothing scientific
or production-facing is being shipped.

The first receipt-citation attempt failed closed because the bare terminal
anchor `END SUBROUTINE dyn_cor_2D` also matches the preceding `_init`
subroutine. The map now pins the second occurrence rather than weakening or
removing the citation. From the corrected clean commit, the gate finds two
citations, zero unmapped citations, zero failures, and zero failing map
entries. Shifting the Coriolis citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`; all internal gate controls fire. The final post-receipt
artifacts are `citation_gate_final.json` and
`citation_gate_shifted_plant_final.json`.

The focused CPU/x64 invocation covered the inherited external-record parser,
the extended Round-83 slow-forcing walk and all new Round-139 parser/replay
controls, every receipt-citation control, and the full fidelity time-level
unit file. Its exact summary is `66 passed in 2.54s`.
The same focused set was rerun from a clean receipt tree and reported
`66 passed in 2.45s`.

Evidence digests before the final post-receipt citation rerun:

| artifact | SHA-256 |
|---|---|
| `preflight.log` (retracted first layout check) | `9207a526f467ca648978ca8a96fc5b3e55d9a7dff31bd06c5424d782b3a8fa0e` |
| `preflight_after_layout_fix.log` | `16a1761691ed525e02ae36a4ef3c1a0ddc51dabc98bc5d652276e7dcc0573de1` |
| `layout_plant.log` | `60babc006ceaa9d38fc3c47b387b7a2f18f0fd156ba53db5b56d49d60d493098` |
| `acquisition.log` | `a18526cea02650c9ef475a33a09ec4f9ef0a2da234f8c09f72b3b08f7cfee998` |
| `citation_gate.json` | `a375052efbd2dc68e21c40caee00e089b0965bee4c5a874309a943d037bd341a` |
| `citation_gate_shifted_plant.json` | `3addae94fae19b1cf5748ee05ef0d6546cc77a64809b5bb721b1e2a1b46edeb7` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests.log` | `12abf566cb464f24a41ef5d2435e86fdcb247b8477cb8c3a768aaa74c01c40fc` |
| `focused_tests_final.log` | `02645a8f4ba6aaec5390ea47de4aa958295b96f64586e6583a87e69478c9fd73` |
| `dynspg_ts_round139.patch` | `6f32f90a592053572a751088f7036e8d75dfb4a6da9877790f7bc4ae17d8eb70` |
| `run.sh` | `9d944098e4c8bfcfe5672c7f80d704e7895905079f64f84bc24179d7669b04a4` |

## Campaign surfaces and unchanged headline rows

This round changes diagnostic tooling, tests, citation pins, preregistration,
and this receipt only. It changes no production module, recipe, card,
configuration default, carried-state schema, restart contract, stabilizer, or
canonical NEMO source. Therefore the ladder/month/year arms are not rerun; the
incoming certified values remain:

| row | unchanged value |
|---|---:|
| kt2 U | `2.7377110452773967e-12 m/s` |
| kt2 V | `3.2849219221489645e-12 m/s` |
| kt3 T | `8.659373840202989e-7 K` |
| kt3 S | `7.027291104577671e-8 g/kg` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674193e-2 K` |
| day-360 T3D RMS | `1.122357391e-2 K` |

DINO, LOCK_EXCHANGE, OVERFLOW, and the tank statements execute no changed
production code and retain their prior dispositions. ORCA2 remains
`UNMEASURED-WITH-SPEC`: its ocean-only card would need the same native
incoming/Coriolis/mask/final registry before an identity claim.

## OPEN — round 140

1. Run the committed acquisition script in an environment where the NEMO
   configuration tree is writable. Require its
   `ROUND139_DEVELOPED_SLOW_FORCING_READY` marker; a failure remains a record
   stop, not a scientific result.
2. Admit the 33,852-byte record, all inherited byte identities, commit stamps,
   and every record/passivity plant. Re-anchor source citations to the new
   target's compiled `ppsrc` before using its numbers.
3. Extend the existing Round-83 production-JIT registry from the same admitted
   step-1080 restart. Require ordinary-state identity, the complete operand
   registry, a missing-row plant, and a consumed incoming-U ULP plant.
4. Adjudicate incoming RHS versus initializing Coriolis only from those direct
   rows and the directed arm. Preserve every frozen prediction and any
   refutation. Only then continue the winning operand's compiled-order walk.
5. Do not revisit QCO, FCT, EVD, TKE, or a held rest-state patch; no downstream
   walk may substitute for this missing upstream record.

`ACQUISITION_NEEDED`:
`/tmp/autopilot-work-1011423537/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round139_slow_forcing/run.sh`

`DECISION_NEEDED`: `NONE`

`ROUND_STATUS`: `STOPPED_FOR_RECORD`
