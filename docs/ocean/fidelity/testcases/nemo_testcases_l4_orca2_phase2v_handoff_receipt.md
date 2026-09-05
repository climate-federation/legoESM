# NEMO testcase Lane 4 — ORCA2 Phase-2v handoff receipt

Date: 2026-09-06

Parent: `74964cadc9de5644f1e7adaf91ad978722654d9b`

Status: **CONFIRMED — replacement acquisition prepared; MPI execution pending.**
No legoESM physics arithmetic changed.  No MPI/NEMO run was launched from the
sandbox.  This round stops at the two prepared user-shell launchers.

## 1. Rejected Phase-2u record: exact localization

`nemo_testcase_l4_orca2_phase2u_tke_acquisition_gate.py --audit-rejected`
decoded both 57,266,632-byte `NEMO_L4_TKEW_1` records to exact EOF.  The A/B
difference is 54,889 bytes at one-based file offsets 14,076,116 through
31,863,080.  Every changed element is at Fortran level `jk=31=jpk`, distributed
as follows:

| field | differing bytes | differing binary64 elements | levels |
|---|---:|---:|---|
| `zdiag_pre_solve` | 14,285 | 2,795 | 31 only |
| `zd_lw_pre_solve` | 12,520 | 1,569 | 31 only |
| `zd_up_pre_solve` | 1,279 | 382 | 31 only |
| `zdiag_after_forward` | 14,285 | 2,795 | 31 only |
| `zd_lw_after_forward` | 12,520 | 1,569 | 31 only |

**CONFIRMED:** the saved `l4_tke_*` arrays were already zeroed after
allocation.  The writer then overwrote those zeros by copying local
`zdiag/zd_lw/zd_up` through `mbkt+1`; in full-depth columns that includes
`jpk=31`, while NEMO defines these workspaces only at the surface and
`jk=2:jpkm1` (`zdftke.F90:284-288,442-460,512-547`).  The affected slots are
unowned/unconsumed workspace, but twin identity is an admission rule, so the
records remain **REJECTED_NONREPRODUCIBLE_TKE_RECORD**.  Their full decode is
in `nemo_testcases_l4_orca2_phase2v_tke_rejection.json`.

The dry-run schema exercise found a second temporal hygiene defect before the
replacement run: the earlier writer copied local `zWlc2` before the executed
no-Stokes branch assigned `zWlc2=zcsd*taum`.  The final writer captures it
immediately after that branch (`zdftke.F90:346-376` in the instrumented copy).
This is **CONFIRMED instrument hygiene**, not a physics choice.

Both Phase-2u directories and `/tmp/orca2_p2u_run_{a,b}.log` are retained and
flagged rejected.  Their model outputs remain the previously observed
100/100 inherited-record identity witness; their new TKE stream is never an
oracle target.

## 2. Final WRITE-only writer and build

The config-local `zdftke.F90` patch makes only these observation changes:

- zero-first all saved record arrays;
- copy matrix workspaces only through `MIN(mbkt,jpkm1)` and copy physical
  `en` through `MIN(mbkt+1,jpk)`;
- capture `zWlc2` only after assignment;
- include carried `dissl`, `htau`, and `hm_i` as source operands needed by
  the ordered scorer;
- derive all 24 per-field extents with `SHAPE` and the payload count with
  `PRODUCT`; retain rank-zero `lwp`, non-tiled, kt=2 arming.

No model array is assigned or read from an `l4_tke_*` array.  **CONFIRMED:**
the scalar-math rebuild completed successfully with binary SHA-256
`8ba6120d4c54c31caaefb595d87da43ef1bb38890208b36d6ca98da71a950a29`
(55,640,072 bytes) and `nm -D` found **0 `_ZGV*` symbols**.  The build log is
`/data/abyssal/dbalwada/nemo-testcases-l4/build_phase2v_tke_walk.log`, SHA-256
`518f2058d90148cba9a92664b2ccfcde560ed5f57941d0da8520261b80d6ff5e`.

The expected admission is 101/101 raw-identical records between independent
twins and 100/100 inherited records raw-identical to the Phase-2s V2 root,
plus ordinary-output identity to the uninstrumented variant control.  Magic,
extent, count, truncation, trailing-byte, canonical-slot, level-31 workspace,
and twin plants are preregistered to exit nonzero.  These are **PLAUSIBLE,
UNMEASURED_PENDING_USER_MPI**, not claimed results.

## 3. Prepared replacement twins

Both directories use the unchanged two-rank `(jpni,jpnj)=(2,1)` launcher
pattern, Bash `time`, `OMP_NUM_THREADS=1`, hash-guarded binary/deck/input
manifests, 41 symlinks (40 deck inputs plus `nemo`), and copied small deck and
launcher files.  Each directory currently has no model output, restart, or
oracle record.

| arm | directory | `run.sh` SHA-256 |
|---|---|---|
| A | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2` | `1137a8fbf19149783e0c7f19ebe496205d7a99ade3aa92b07189bf670dcd95b9` |
| B | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_b_10step_np2` | `a92c8b85aa6be737cefc881edc78706547a2d0b7032dc722c95026f4617a4d41` |

Deck and input manifest hashes are respectively
`e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059`
and `3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.

## 4. Ordered TKE walk prepared

`nemo_testcase_l4_orca2_phase2v_tke_walk_gate.py` first calls the twin,
inherited-record, schema, canonical-slot, and ordinary-output admission gate.
Only after admission does it JIT the CPU/fp64/scalar-libm walk.  Its ordered
inventory covers:

1. `nn_eice`, `zWlc2`, `zpelc`, `imlc`, `zhlc`, `zus3`, Langmuir `en`;
2. `pdlr`, `zdiag/zd_lw/zd_up`, the RHS, all three solve recurrences and floor;
3. `htau` and `nn_etau=1` penetration;
4. raw and bounded `nn_mxl=3` `mxlm/mxld`;
5. `avm`, `avt`, `nn_pdl=1`, and `dissl`.

Each row reports cellwise `unequal / n`, first cell class, and row-scale ULP;
each named row has a target-bit plant that exits nonzero.  Production helpers
are called for ice attenuation, Langmuir, the literal solve, etau, mixing
lengths, and coefficient assembly.  The statement-by-statement transcription
lives only in the harness and cannot become an alternate production numeric.
The gate was syntax checked and its complete row/plant path was exercised with
a temporary synthetic schema adapter.  That adapter used zero placeholders
for the three new operands and therefore supplies **no admissible numerical
result**; it was used only to prove shape/JIT/dispatch reachability.  The first
real non-bit statement and owner will be reported only after valid twins.

Ownership remains: card selectors or ORCA2 forcing operands are Lane 4;
shared TKE arithmetic is `GYRE_OWNER_SHARED_TKE` and will be handed over with
this reproducer.  EVD, IWM, and SI3 were not entered.

## 5. Phase-2t large-output relocation

**CONFIRMED:** the two per-run Phase-2t JSON files over 20 KiB were moved
byte-for-byte to `/data/abyssal/dbalwada/nemo-testcases-l4/phase2t/`:

- admission: `8fc276f7e94462ba28fa5527b4da2ac3cd08b3bec4e43d83c4ae5b17d12c7525`;
- schema walk: `f81732704ad249d62b4ad36c028cfb22357bf0125d3bece98252e5031cd1bcbf`.

The Phase-2t receipt and manifest now cite those paths.  No evidence was
deleted; small cited JSON summaries remain in git.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| decode the Phase-2u byte ranges | ASKED | CONFIRMED exact schema localization above |
| canonicalize and rebuild the writer | ASKED | CONFIRMED WRITE-only patch; scalar build green |
| add scorer operands to the same frame | ASKED-enabling | CONFIRMED `dissl_pre/htau/hm_i`, observation only |
| user-shell twin execution | ASKED | pending; prepared paths above |
| move Phase-2t per-run JSON >20 KiB | ASKED review carry | CONFIRMED byte-preserving relocation |
| prepare one-pass TKE gate and plants | ASKED | CONFIRMED ready; no physics verdict yet |
| score or pin the rejected Phase-2u stream | forbidden by twin rule | not done |
| change shared TKE arithmetic | Lane-4-forbidden | not done |
| enter EVD/IWM/SI3 | downstream/unasked | not done |
| edit shipped NEMO, delete artifacts, commit multi-MB data, push | forbidden | not done |

Session: `01a06d99-f562-7b11-bc63-e9b112877f54`.
