# NEMO testcase L2 GYRE phase-3 round-48 receipt

Date: 2026-09-11. CPU/fp64. Preregistered at `f826f1e6c8f9`; the clean
accumulator producer is `b7fe826268cf47d404aa4c0f9b7214c45a0d3ea5` and the
unchanged-trajectory producer is `fcacdf16d189ca1b790a47ffdb345933fcb58084` in
`/tmp/codex-round48-producer`. The canonical checkout could not create
`.git/index.lock`, so all round-48 paths remain for the operator to commit.

## Verdict

The frozen prediction is **CONFIRMED**. The first full model-path non-bit
accumulator at kt=2 is post-LDF in stage 1 and post-VOR in stages 2 and 3.
This closes the round-47 measurement gap; it does not identify the causal
statement inside LDF or VOR. Evidence is
`round48/round48_model_path_accumulators.json` (SHA-256
`683a889e657c1292cf9a99eb22b4b5299b46b2b2589a921471b19c1854d3d8a4`).

| kt=2 stage | post-HPG U/V unequal | first non-bit U/V | later source boundaries U/V unequal |
|---|---:|---:|---:|
| 1 | 0 / 0 | LDF 17,400 / 17,100 | VOR 17,400 / 17,100; ADV 17,400 / 17,100 |
| 2 | 0 / 0 | VOR 8,295 / 8,171 | ADV 17,399 / 17,099 |
| 3 | 0 / 0 | VOR 8,125 / 7,972 | ADV 17,399 / 17,099; LDF 17,400 / 17,100 |

The gate uses NEMO-given Kmm state for HPG/VOR/ADV and Kbb velocity for LDF.
That is required because compiled stage 1 calls `dyn_ldf(Kbb,Kbb)` before VOR
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`), compiled
stage 3 calls `dyn_ldf(Kbb,Kmm)`
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:695-711`), and the
live LDF stencil reads `pu/pv(Kbb)` and Kbb thickness
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:64-74`;
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:123-127`).
HPG overwrites Krhs because it is first
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynhpg.f90:400-427`); the recorded
pre-HPG scratch is not injected.

## Barotropic-memory alignment

| NEMO quantity | compiled-card boundary role | legoESM carry | disposition |
|---|---|---|---|
| `uu_b/vv_b` | Kmm seeds `un_e/vn_e`; Kaa receives final external mode (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:352-378`; `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:824-827`); `stp2d` passes the prognostic slots (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:301-308`) | `state.uu_b/vv_b` (`barotropic_latlon_cgrid.py:2858-2869`) | aligned representation |
| `ubb_e/ub_e/vbb_e/vb_e` | AB3 inputs (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-480`), rotated each substep (`:749-757`), restart I/O (`:953-980`) | four velocity members of deviation-form `state.bt_hist` (`state.py:648-666`; reconstruction `barotropic_latlon_cgrid.py:2053-2076`) | existing carry; record required |
| `sshbb_e/sshb_e` | AB3 and time-filter inputs (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:483-489`; `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:593-600`), rotated (`:759-761`), restart I/O (`:953-980`) | two SSH members of `bt_hist` | existing carry; record required |
| `un_adv/vn_adv` | reset each call, accumulated/finalized inside it (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:373-378`; `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:559-567`; `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:797-804`) | returned within-window transport, not cross-step state | not carried |
| `ub2_b/vb2_b` | absent from compiled RK3 `dynspg_ts`; RK3 ends before restart write (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:858-863`) | none | dead compiled arm; waived |

No missing carried quantity and no configuration choice were found. Therefore
no state or physics transcription is eligible or landed. The rejected
round-47 ZAD patch remains rejected and unapplied.

## Rule 12 and acquisition handoff

Only private diagnostic outputs were added. The ordinary two-step GYRE run is
unchanged: kt=2 U/V maximum residuals are exactly
`2.7478404751243857e-12` / `3.305560306813421e-12` before and after; first
over bar stays kt=2. Evidence `round48_GYRE_kt1_2.json` (SHA-256
`6f8565649d7171bfd41020b2482ec9f010f9bd330a5c12a842e9f579f5d332e8`).
LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 are unchanged-by-construction because
the normal return path has no new operation; no physics claim is made.

The prepared WRITE-only record contains kt=1 end and kt=2 start copies of the
six histories, current external mode, `un_adv/vn_adv`, and Kmm/Kaa U/V/SSH
(17 named 36x26 fp64 arrays per file). The gate requires boundary bit identity,
Kaa/current identity at kt=1 end, Kmm/current identity plus exact reset zeros at
kt=2 start, physical EOF, clean commit stamp, and inherited twin admission.
Run only as operator:

```bash
/usr/bin/bash /tmp/codex-gyre-git/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round48_bt_memory/run.sh
```

Focused acquisition tests plus citation/stamp ratchets: 37 passed. The
given-input accumulator and citation plants each exit 1; all new
header/truncation/boundary/seed/reset/stamp plants have hermetic red tests, and
the script runs them plus the inherited consumed-field twin plant.

| status | item | disposition |
|---|---|---|
| ASKED | model-path score and memory acquisition package | complete / awaiting operator record |
| UNASKED | new state, configuration, threshold, physics, or reapplying ZAD | none |
