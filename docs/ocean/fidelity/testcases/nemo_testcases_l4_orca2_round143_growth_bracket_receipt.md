# ORCA2 round 143 — finite-growth bracket and earlier-record request

Date: 2026-10-04. Base: `312db78c1`. Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round143.md` at
`3bdacf827`. Verdict: **STOPPED_FOR_RECORD**.

All numbers in the growth walk are **independent**: rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. They do not use
Decision 52's NEMO entry bridge. The separately checked rung-7 ladder remains
labelled **given NEMO's entry**. Sea ice and the ORCA2 card's
`unmeasured_features = ("linear_implicit_bottom_drag",)` were not changed.

## Result

The operator's round-141 record admits without repair. Its 14 self-describing
rank-step files cover both ranks at steps 30 through 36, and all 20 calibration
restart comparisons are byte-exact. Admission status is
`PASS_R141_GROWTH_RECORD`; the admission JSON SHA-256 is
`0e455a46fe3a9a506eb9ad301ca44a9b6d9181d7712d1f881c22a6462b0fd18c`.

At column `(j,i)=(87,159)`, the mechanically first recorded row over the frozen
`2e-10` floor is already the step-30 entry sea surface:

| boundary | legoESM | NEMO | absolute error |
|---|---:|---:|---:|
| step 30 `ssh_entry` | -0.42784726839941356 m | -0.49864564249907384 m | 0.07079837409966028 m |
| step 36 `ssh_entry` | -2.2017129174324044 m | 0.5306056622595993 m | 2.7323185796920035 m |
| step 36 `ssh_after` | -872.3737752568386 m | -0.4044080381491385 m | 871.9693672186895 m |
| step 36 `r3t_after` | -0.9994271794912696 | 1.8034294431949786e-05 | 0.9994452137857015 |

Thus the record is left-censored: the finite-magnitude growth began before its
first frame. It is not permissible to assign the error to the step-30
barotropic solve or to the later transport/FCT path. Every registered early
boundary value remains finite even at step 36; the later non-finite FCT value
is downstream of this record.

The growth-walk gate reports `PASS_ROUND143_GROWTH_WALK`; its JSON SHA-256 is
`9f9fdd73ad194ac0f51566aed78ea9b8d909c848171199d1e9186d3acce84661`.
For every step 30 through 36, the observed run reproduces the separately
compiled ordinary `T`, `S`, `u`, `v`, `ssh`, `uu_b`, and `vv_b` state
bit-for-bit.

## Frozen prediction ledger

| ID | Verdict | Evidence |
|---|---|---|
| R143-P1 | CONFIRMED | Existing record admitted; all admission plants fired. |
| R143-P2 | REFUTED | No registered early boundary is non-finite at step 36; only the downstream FCT path later loses finiteness. |
| R143-P3 | CONFIRMED, left-censored | The first row in the available window is step-30 `ssh_entry`, already 0.07079837409966028 m over the floor; no earlier frame exists. |
| R143-P4 | CONFIRMED | SSH/thickness is already in debt before the recorded barotropic and stage-1 transport groups. |
| R143-P5 | CONFIRMED | No model physics, card, deck, carried state, stabiliser, sea-ice selector, or unmeasured-feature change landed. |

## Instrument controls

The first broad diagnostic request correctly refused because constant mixing
has no prognostic TKE payload. A narrowed pair of independently compiled
barotropic/transport observers then correctly refused after moving ordinary
`S`, `ssh`, `uu_b`, and `vv_b` bits; none of those values was accepted as
science. The final test-private hook only makes live-trace completeness require
TKE fields when prognostic TKE is active. It is compared with a separately
compiled ordinary trajectory and is passive for all seven state arrays at all
seven steps. This is diagnostic plumbing only: no public configuration or
physics selector changed.

Because the package file moved by two lines, the default GYRE round-8 receipt
was re-anchored once with a `difflib.SequenceMatcher` old-to-new map. Its real
gate has 274 mapped citations, no failures, no unmapped citations, and no map
audit findings.

## Acquisition needed

No admitted stream covers steps 11 through 29. The committed launcher is:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round143_early_growth_acquisition/run.sh`

It creates the new `ORCA2_OMIP_L4_R143EARLY` target, a 10-step calibration
twin, and rank-complete self-describing frames for steps 11 through 30: 40
files (20 steps x 2 ranks), the same 14 fields in source order. Admission
requires all 20 calibration restarts to be byte-identical to the pinned rung-0
record. The checker derives field names, ranks, dimensions, and payload lengths
from each header and has nine planted violations. Launcher plants for source
layout, hidden deck delta, and producer content all fire. Preflight prints
`ORCA2_ROUND143_EARLY_GROWTH_PREFLIGHT_READY` with target
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round143/acquisition/orca2_rung0_early_growth_30step_np2`.
No in-sandbox MPI run was attempted.

## Standing gates

| gate | result |
|---|---|
| GYRE 70-row ladder, base versus tip | PASS; `ladder.residuals.npz` SHA-256 `7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3` on both; 0/70 rows moved; first over bar remains kt=3; plant fired. |
| GYRE independent 30-day daily trajectory | PASS; 30/30 snapshots byte-identical; day-30 SHA-256 `b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180` on base and tip. |
| ORCA2 rung-0 independent ladder | PASS; 0/200 rows moved from round 140; no exact row left the bar; first debt unchanged. |
| ORCA2 rung-7 given-entry ladder | PASS; 0/200 rows moved from round 140; no exact row left the bar; first debt unchanged; bit-loss plant fired. |

Focused round-143 and citation tests passed 39/39 before the receipt update.
The required `tests/ocean/fidelity -n 12` battery was launched once and reached
98%, then its xdist controller disappeared without a terminal summary. Its four
reported failures were re-run individually and reproduce the round-142 known
reds: SI3 scalar-math verbatim-source provenance, the round-129 certified-year
harness membership ratchet, nine round-35 stamp-scope offenders, and eleven
worktree-stamp offenders. No round-143 test failed. The battery is recorded as
INCOMPLETE, not PASS.

The separate read-only review could not initialize its app-server client on the
read-only filesystem. Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Operator runs the committed early-growth launcher. Admit no values unless
   both-rank step coverage, calibration restart identity, source ordering, and
   all plants pass.
2. On the admitted steps 11 through 30, select the first row over `2e-10` in
   compiled order. If step 11 is already over the floor, request an earlier
   rank-complete interval rather than infer an owner.
3. Only after an at-bar predecessor and over-bar successor exist, split that
   boundary one recorded operand at a time and name the first NEMO statement.

## Compiled-source citations

The rung-0 step program records entry SSH/thickness/external-mode state before
`stp_2D`, records the corresponding post-barotropic state, and only then calls
stage 1 in
`ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/stprk3.f90:202-221`.
The stage-1 transport path calls `wzv`, constructs `pFw`, and records
`r3t/zFu/zFv/zFw` in
`ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/traadv.f90:299-320`.
