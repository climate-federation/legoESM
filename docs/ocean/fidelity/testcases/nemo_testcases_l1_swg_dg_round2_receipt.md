# SWG double-gyre round 2 receipt

Date: 2026-10-12

Status: **LANDED; NEMO LADDERS ACQUISITION_NEEDED**. The failed Round 1
writer-target bootstrap is repaired. Both acquisition tools now create every
writer target from the registered shipped `SWG` test case, then overlay the
selected deck source before applying the additive writer. No `makenemo`,
`mpirun`, NEMO integration, or GPU integration was run by the agent.

## Failure and disposition

The incoming tip was `e9e5d92c4066ba27a4aa3e1501c4a6952f5dd6ee`. The
operator's acquisition log is
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/swgdg_acq_run.log`.

The Maddison reference run completed, but both writer builds failed when
`makenemo -a SWG_DG` tried to use a created configuration as a reference.
NEMO only supplies reference component metadata for cases registered in
`tests/demo_cfgs.txt`; `SWG` is registered there and `SWG_DG` is not. The
failed build consequently had no reference directory or component list and
stopped while checking its keys. The operator preserved the partial targets
and evidence with `.failed1` suffixes, leaving the original target and evidence
names free.

The implementation is commit `b82999728a24b071f661fafea1376f3a3fe40e57`.
For both decks and both acquisition lengths it now:

1. requires the `-a` source to be exactly the registered shipped test case
   `SWG`;
2. creates the target with
   `./makenemo -a SWG -n <target> -m conda-scalarmath del_key key_xios`;
3. reads the created target's own cpp file and requires exactly
   `key_qco key_vco_3d key_RK3`;
4. overlays the selected deck's `MY_SRC` (`SWG` or the preflight-verified
   `SWG_DG`) before copying and patching the additive writer; and
5. checks the same target cpp keys again after the final compilation.

The `SWG_DG` preflight still proves that its `MY_SRC` is shipped `SWG/MY_SRC`
plus exactly the hash-pinned basin-extent patch. The reference binary remains
the selected deck binary. Only target creation is based on shipped `SWG`.

## Preflight and tests

All four no-run preflights pass at the implementation commit:

| Deck | Ladder | Selected source | Writer target | Printed target creation |
|---|---|---|---|---|
| shipped | ten step | `SWG` | `SWG_OMIP_L1_P3` | `makenemo -a SWG ... del_key key_xios` |
| shipped | thirty day | `SWG` | `SWG_OMIP_L1_P3D` | `makenemo -a SWG ... del_key key_xios` |
| Maddison | ten step | `SWG_DG` | `SWG_DG_OMIP_L1_P3` | `makenemo -a SWG ... del_key key_xios` |
| Maddison | thirty day | `SWG_DG` | `SWG_DG_OMIP_L1_P3D` | `makenemo -a SWG ... del_key key_xios` |

The direct regression tests execute the new source guard, accept `SWG`, reject
`SWG_DG` with exit 68, execute the target cpp guard against accepted and
XIOS-contaminated key sets, inspect both actual operator scripts for the exact
creation command and overlay order, and prove the command assertion fails when
the Round 1 invocation is planted. Round 3 correction (2026-10-12): the exact
`174/174` count is withdrawn because its command output was not archived under
the evidence root. The current-tip replacement battery and its exact count are
recorded in the Round 3 receipt and evidence log. Shell syntax and
`git diff --check` were rerun in Round 3.

The Round 1 review log at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/reviews/codex_review_swgdg_r1.log`
ends during tool execution and contains no reviewer verdict or actionable
Round 1 finding section. Its apparent `CONFIRMED` entries are quoted content
from an older shipped-SWG receipt, not findings about the double-gyre diff.
The operator-confirmed acquisition failure above is the finding repaired here.

## Unchanged fidelity claims

There is no change under `packages/`, no physics or numerical change, and no
change to the card, ladder, namelist patches, record writer, checker, or
long-run driver. The retained resolved namelist still supplies `nn_SWG`,
`rn_dx`, `rn_domsiz`, `rn_tau`, `rn_modified_grav`, `rn_Uv`, `rn_Lv`,
`rn_rfr`, `rn_beta`, `rn_f0`, and `rn_Dt` to the card. The established shipped
rows therefore remain unchanged and bit-identical: 338/338 given-entry,
338/338 chained, 50/50 independent, 29/29 carried-state, and 690/690 month.
The direct tiny-grid long-run driver test still passes.

## Acquisition needed

```text
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1/
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run_30d.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1_30d/
```

The predicted result remains AT-BAR and bit-identical at every row with each
one-ULP plant binding. That prediction is unmeasured on the Maddison deck until
the operator runs the two commands above.

## Choices and unverified items

ASKED: create every target from shipped `SWG`, remove XIOS, overlay the deck
source, require target keys, and retain the Round 1 targets and evidence paths.

UNASKED: none.

UNVERIFIED: the repaired operator-only build and execution path has not been
run; only its fail-closed preflight and direct regression tests have run.
