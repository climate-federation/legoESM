# ORCA2 round 95 — rung-0 split-explicit substep acquisition

Date: 2026-10-02. Base `0bfe1f676`; final tree recorded below. Scope is ocean
only. Every prospective rung-0 number is **independent**. This round produces
no ocean-model number because its required two-rank record does not yet exist.

## Verdict

**STOPPED_FOR_RECORD.** Round 94 localized the first remaining boundary to the
split-explicit output, but the inherited barotropic files are root-only legacy
streams. They cannot determine whether rank 1 crosses a boundary first and do
not meet the self-describing-record rule. A committed, preflight-clean
acquisition now requests one kt=1 substep stream per rank and proves observer
passivity against all twenty admitted kt=1..10 restart shards. No `packages/`
file, stabilizer, carried state, threshold, sea-ice field, or
`unmeasured_features` entry changes.

Decision 83 arrived after preregistration and is recorded as a loud amendment,
not folded into the original predictions. Its authorized namelist-only patch
keeps `nn_havtb=0`, changes the three cited-inert spellings `ln_spc_dyn`,
`ln_sssr_bnd`, and `nn_chldta` to rung-1 values, and removes six explicit
inactive `ln_zdf*` false assignments so the reference namelist supplies the
same false values. The target namelist is SHA-256 pinned. Its rerun must keep
all twenty terminal restarts byte-identical to the pre-harmonization round-93
record; otherwise the entire acquisition refuses.

## Why the existing record stops

The admitted round-93 directory contains one file for each inherited
`oracle_bt_*` family, without a rank tag. That matches the executing compiled
guards: the frozen coefficient writer is inside `lwp` at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:301-316`, and the legacy
substep, drag, transport-mean and first-two-substep streams are opened inside
the same root-writer guard at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:390-442`. Their fixed
headers also predate the binding per-array `(name, rank, dimensions, payload)`
format. R95-P1 is therefore **CONFIRMED** without decoding a payload.

Those streams remain useful background information, but they are not admitted
as the owner-naming instrument for a 148x180 tripolar domain split across two
ranks.

## Compiled order the new record covers

The rung-0 branch copies the sea-surface and momentum forcing, initializes the
external state, and enters its substep loop in that order at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:287-320`,
`:339-381`, and `:446`. Within every pass it evaluates:

1. mid-step velocity and sea-surface extrapolation plus face depths
   (`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:460-519`);
2. transports and after-SSH (`:530-557`);
3. the half-step-back sea surface and pressure gradient (`:601-615`);
4. the 2-D Coriolis trend and explicit drag (`:618-652`);
5. the vector-form velocity update (`:655-708`).

The new additions-only patch writes every operand and result after its compiled
statement. The stream header carries `icycle`; each field carries its own
name, rank and extents; the checker derives the required `j001..jNNN` frames
from that header and walks payloads to exact EOF. It does not predict a byte
count or header tuple. Rank origins and owned bounds must cover the global
domain exactly once.

## Controls and preflight

The launcher pins the round-93 source, binary, cpp keys, source namelist, deck
manifest and input manifest by SHA-256; refuses a dirty or uncommitted
producer; proves the source patch removes zero NEMO lines; applies both source
and Decision-83 deck patches at fuzz zero; pins the harmonized namelist digest;
preprocesses and compiles both the writer and patched `dynspg_ts`; refuses
existing target names; and never invokes NEMO without `--run`.

Preflight reports:

```text
SYNTAX_PROOF_PASS l4_r95_spgts_frames.f90 dynspg_ts.f90
ORCA2_ROUND95_RUNG0_SPGTS_PREFLIGHT_READY .../round95/acquisition/orca2_rung0_spgts_ranked_10step_np2
STATUS PLANT-FIRED layout
```

The synthetic self-describing-record controls pass 7/7. They derive two
substeps from the header and separately fire header, field-name, field-extent,
truncation and missing-frame violations; the swapped-rank control changes the
reported rank and is rejected by the full admission path. Once a real record
exists, admission additionally fires the restart-byte plant and requires every
terminal restart to be byte-identical to round 93.

## Frozen predictions

| Prediction | Verdict | Evidence |
|---|---|---|
| R95-P1 inherited streams inadmissible | **CONFIRMED** | One root-only file per family; compiled `lwp` guards; no rank-complete self-description. |
| R95-P2 observer passive | **UNMEASURED** | Requires operator-run target and twenty restart comparisons. |
| R95-P3 first non-bit no later than first after-SSH | **UNMEASURED** | Requires both rank payloads. |
| R95-P4 ORCA2 land-adjacent `e3f_0vor` owner | **UNMEASURED** | Requires the per-substep Coriolis cross test; VORTEX evidence is a prediction, not an ORCA2 result. |

## Gates, tests, and review

The acquisition preflight and its layout plant pass as quoted above. The
focused record plus citation-gate battery passes 24/24. The default citation
gate passes 274 citations and the round receipt passes all 10 citations, with
zero unmapped citations, failures, or map-audit failures; shifting the cited
Coriolis range by two lines makes the gate fail with exit 1.

The prescribed single `tests/ocean/fidelity -n 12` run reached 97%, displayed
the same six failures as round 93, and then reproduced the established silent
tail: every pytest process disappeared without a summary while the launcher
remained open. The six visible IDs were the round-51 private-hook ratchet,
round-129 record-backed gate, round-35 stamp scope, worktree stamp, recipe case
board, and the known SI3 scalar-math provenance gate. After three empty waits
with no pytest process, the stranded launcher was interrupted. This receipt
does not represent the wide battery as PASS and does not assign an unseen
failure.

The required separate `codex exec --sandbox read-only` review failed before
reading the diff: `failed to initialize in-process app-server client:
Read-only file system`. Verdict: **independent review unavailable in-sandbox**.

The package tree is unchanged from base, so GYRE's certified trajectory is
unchanged by construction; no claim is made about an unrun ORCA2 substep
record.

## OPEN

1. Operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round95_spgts_acquisition/run.sh --run`.
2. Admit both rank streams and all twenty byte-identical terminal restarts;
   failed plants or passivity stop the walk.
3. Replay the independent rung-0 substeps in the compiled order above and name
   the first active non-bit statement.
4. If the walk reaches `dyn_cor_2D`, perform the preregistered cross-operator
   check and test NEMO's four-cell masked `e3f_0vor` denominator as one variable.
5. Keep the round-94 slow-depth fix held until its GYRE cancelling pair is
   identified; it is not mixed into this acquisition.

## UNVERIFIED

- No round-95 payload or restart-passivity result exists yet.
- The first non-bit statement inside `dyn_spg_ts` is not named.
- The ORCA2 `e3f_0vor` cross-operator result is unmeasured.
- The rung-0 given-entry/independent ladders and month remain subsequent work.
