# ORCA2 round 227 — OMT-4 implicit tracer solve exoneration

Date: 2026-10-10. Frozen base: `66af9c7e6`. Preregistration commit:
`b0c9d2cde`. Measurement commit: `4e963bdf0`. Status: **HELD**; no model
statement or card selector changed. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round227/`.

Every number below is **given NEMO's entry**. Sea ice, the shipped rung-10
card, and its `unmeasured_features` tuple are untouched.

## Source and active path

The compiled OMT record build selects `ln_zad_Aimp=.false.`,
`ln_zdfmfc=.false.`, `ln_zdfddm=.false.`, and `ln_traldf_OFF=.true.`. The
active `tra_zdf_imp` program therefore selects `avt` and builds one shared
temperature/salinity matrix at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:171-216`, forms the
lower/diagonal/upper bands at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:218-235`, eliminates the
diagonal at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:249-273`, and performs
the content recurrence and reverse substitution at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:283-299`.

The OMT-4 card resolves `zdf_implicit_solver_evaluation="nemo_literal"`.
Consequently upstream commit `5e368e87ba` ("build the tridiagonal bands
inside the level sweep") changes the generic/shared Thomas path and is not
executed by this card. No merge-or-transcribe decision is reached.

## Admitted records and controls

The gate reads rank 0 from the already-admitted OMT-4 P3 run under round 222:
the kt=1 stage-3 RKTR3 stream supplies Kbb/Kmm/Kaa tracers, Krhs and all three
`r3t` slots; the ZDF stream supplies `avt`. Its ownership control reproduces
Kmm on every wet cell (0 unequal). The passive production trace returns every
ordinary state leaf bit-for-bit equal, including T, S, u, v, and ssh.

The first checker draft was **REFUSED and retained**: it reconstructed
`e3w_1d` from coordinate midpoints and missed NEMO by up to 2.4737806155881543
m. NEMO's recorded W ladder, not the midpoint ladder, is the compiled
operand. With that correction, a full source replay reproduces recorded Kaa
on all 233,341 wet rank-0 cells: 0 unequal, 0.0 maximum, 0.0 rms.

## Result: the implicit solve is exonerated

The raw carried trajectory first differs at `e3w_now`: all 224,547 active
interfaces differ, by rms 2.7384085580483686e-05 m and maximum
1.3833879546041317e-03 m. That is carried stage-2 SSH/`r3t` debt, not an
implicit-solve statement.

The preregistered one-variable replay holds NEMO's recorded stage state fixed.
Under that substitution every row is bit-exact:

| source row | support | unequal | maximum | rms |
|---|---:|---:|---:|---:|
| `avt` / heat K | 224,547 | 0 | 0.0 | 0.0 |
| `e3w(Kmm)` | 224,547 | 0 | 0.0 | 0.0 |
| `e3t(Kaa)` | 233,341 | 0 | 0.0 | 0.0 |
| lower / diagonal / upper | 233,341 each | 0 each | 0.0 | 0.0 |
| eliminated diagonal | 233,341 | 0 | 0.0 | 0.0 |
| temperature content RHS | 233,341 | 0 | 0.0 | 0.0 |
| forward recurrence | 233,341 | 0 | 0.0 | 0.0 |
| solved temperature Kaa | 233,341 | 0 | 0.0 | 0.0 |

Thus there is no first non-bit statement inside the live stage-3 implicit
tracer solve. R227-P2 is **REFUTED**. The predicted overlap with main's generic
rewrite (R227-P3/P4) is also **REFUTED**. Statement sufficiency is NOT REACHED:
there is no candidate statement to combine with the atomic fold unit, so no
Decision-96 vote and no model landing are manufactured.

Artifacts:

- `implicit_walk.json`: `22b05758a3bb76f483379cb77c793e783996b4ae06e3a00e61e3662cf6c3f10c`
- `implicit_walk_v3.log`: `687f7e059f53391de0005f3907ac9d7d4101611a8db8cdc213f2131e7681735e`
- `implicit_walk_raw_v3.json`: `c4ab3b4f3a8d972b925f52f97b1f7aac38a49b7943e8906bbb19d2c0368a8159`

## Validation and review

The gate has direct classifier tests for source order, exact-replay refusal,
metric-source scope, and all four planted violations. The final validation
results are:

- the focused round-227 gate, receipt-citation, and citation-gate tests pass,
  25/25 (`focused_tests.log`, SHA256
  `b001a8bffd7016d99621f70607651d7ff8de89b389df48c6614d6d94d2a3ea7f`);
- the round receipt citation gate passes with four citations, zero failures,
  zero unmapped citations, and zero map-audit failures; the cumulative default
  receipt passes with 274 citations and the same zero counts;
- the rigid-shift plant on the matrix-band citation above exits nonzero and
  names the shifted first endpoint (`citation_plant.json`, SHA256
  `7cf88bec192ff38338fc83df4ec1dc8bf8e3d24374ec80b940d91012afa25f55`);
- the one prescribed `tests/ocean/fidelity -n 12` battery collected 3,108
  items. Before its quiet, still-running final LOCK control was mistakenly
  interrupted at 99%, its preserved log contained 3,088 passes, seven skips,
  and four registered pre-existing reds: the certified-year worktree-stamp
  ratchet, the round-35 stamp-scope ratchet, the report-emitter worktree-stamp
  ratchet, and the SI3 scalar-math provenance gate. The only scheduled test
  without a terminal result was
  `test_planted_stage_control_exits_nonzero_end_to_end`; its isolated rerun
  passed in 558.94 s (`battery_missing_id.log`, SHA256
  `6886bee2e29cb6f81a10373c2ae43141eebbdd8caaba2541308219792e423d10`).
  The full battery was not rerun.

The required `codex exec --sandbox read-only` review could not initialize its
app-server client because the sandbox denied its PATH-alias write. The verdict
is therefore **independent review unavailable in-sandbox**, not PASS
(`independent_review.log`, SHA256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

No `packages/` file changed, so the GYRE year, DINO month, tanks, rung-0,
rung-7/rung-10, and OMT-4 trajectories cannot execute a changed model
statement. Their previously certified values remain unchanged by
construction.

## OPEN

The stage-3 implicit tracer solve is closed. OMT-4's kt=8 live-W refusal still
requires a compensating partner, but it is upstream of `tra_zdf`: under the
atomic unit, walk the carried stage-2 SSH/`r3t` producer that first changes the
otherwise exact `e3w(Kmm)` operand, in completed-state/offline-replay order.
Do not climb to OMT-5 and do not merge main for this finding. The literal
nonosc/FCT associations and the implicit solve remain exonerated; none should
be transcribed again.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
