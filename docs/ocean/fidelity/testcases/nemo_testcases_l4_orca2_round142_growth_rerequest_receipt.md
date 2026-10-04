# ORCA2 round 142 — growth-record re-request

Date: 2026-10-04. Base: `25d7f732e`. Preregistration: `9ffb82bfb`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round142`.
Verdict: **STOPPED_FOR_RECORD**. The unchanged round-141 launcher is ready for
operator execution after the root-filesystem cleanup. No ocean model, card,
deck, acquisition artifact, selector, carried state, stabilizer, sea-ice field,
or `unmeasured_features` entry changes.

Every requested trajectory value remains **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity, and zero sea surface.
No Decision-52 recorded-entry value is mixed into this record.

## Frozen ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R142-P1 | CONFIRMED | The unchanged launcher's preflight exits 0 and names the original round-141 36-step target; `/tmp` has 94 GiB free. |
| R142-P2 | CONFIRMED | The target configuration, 10-step calibration run, and 36-step growth run are all absent. The failed operator attempt created none of them. |
| R142-P3 | CONFIRMED | Source-layout, hidden-deck-delta, and producer-content plants each exit 69 with `STATUS PLANT-FIRED`. |
| R142-P4 | UNMEASURED | No NEMO science record was produced in this sandbox round, so no NEMO-versus-legoESM growth number is reported. |
| R142-P5 | CONFIRMED | The committed delta is documentation-only; `packages/` and the round-141 acquisition directory are unchanged. |

## Re-request result

The operator's prior attempt stopped at the launcher's first `mktemp`, before
source staging, target creation, compilation, or NEMO. After cleanup, the same
committed launcher now prints:

```text
SYNTAX_PROOF_PASS l4_r141_growth.f90 stprk3.f90 traadv.f90
ORCA2_ROUND141_GROWTH_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round141/acquisition/orca2_rung0_growth_36step_np2
```

The source-order contract is unchanged: the compiled rung-0 step program calls
the split-explicit solve before stage 1 at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/stprk3.f90:204-215`. The launcher's
calibration restart identity and rank-complete self-describing record checks
remain admission predicates; this round does not infer their outcomes.

## Scope and controls

The re-request uses the exact committed round-141 `run.sh`; no hash, target,
deck, writer, checker, or expected stream changed. The preflight and all three
controls are preserved under the round-142 evidence root with SHA-256 hashes.
The requested operator command remains the launcher's default `--run` mode.

There is no `packages/` change, so the ORCA2, GYRE, DINO, tank, and generic-card
trajectories cannot move in this round. The record itself remains inadmissible
until the operator run ends with `PASS_R141_GROWTH_RECORD`.

## Validation and review

The default citation gate passes 274 citations and the round-142 gate passes
its one compiled citation, both with zero failures, unmapped citations, or
map-audit failures. Shifting that compiled citation by two lines makes the
round gate fail as required.

The unchanged round-141 focused battery passes **11/11**. The required
`tests/ocean/fidelity -n 12` battery reached 99% and then entered the known
silent xdist tail with zero live pytest processes; it is **incomplete, not
PASS**. Four pre-existing reds appeared and reproduced alone: SI3 scalar-math
source provenance, the certified GYRE spread-floor record's stale harness pin,
round-35 allow-dirty escape scope, and the worktree-stamp grow-only ratchet.
The round changes none of their code or records, and no round-142 focused test
failed. Logs are preserved under the evidence root.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. The operator runs the unchanged round-141 launcher. Any result other than
   `PASS_R141_GROWTH_RECORD` rejects every science value.
2. Once admitted, print NEMO versus legoESM at column `(j,i)=(87,159)` and its
   incident faces for steps 30..36 in recorded source order. The first step and
   row beyond `2e-10` own the next one-variable walk.
3. After a cited statement closes that boundary under the standing gates,
   rerun the independent rung-0 month and report its new first non-finite step
   or complete 240-step score. Decision 94 remains pending and untouched.
