# ORCA2 round 144 — early finite-growth bracket

Date: 2026-10-04. Base: `d4fbba7d2`. Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round144.md` at
`bb3e65c78`. Verdict: **STOPPED_FOR_RECORD**.

Every science value below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. Decision 52's NEMO
entry operand is not used. The rung-7 ladder remains separately labelled
**given NEMO's entry**. Sea ice and
`unmeasured_features = ("linear_implicit_bottom_drag",)` did not change.

## Result

The operator's round-143 early-growth record admits unchanged. Forty
self-describing files cover both ranks exactly once at every step from 11
through 30. All 20 calibration restart shards are byte-identical to the pinned
rung-0 record. All nine admission plants fire. Admission status is
`PASS_R143_EARLY_GROWTH_RECORD`; its JSON SHA-256 is
`5dc84ab4bd42052c5a43d185ea8a6d525105585b955b8334040a495bfc53ae26`.

At column `(j,i)=(87,159)`, the first available row is again already over the
frozen `2e-10` floor:

| boundary | legoESM | NEMO | absolute error |
|---|---:|---:|---:|
| step 11 `ssh_entry` | 0.8128679887999752 m | 0.8479530808471768 m | 0.03508509204720156 m |
| step 11 `r3t_entry` | 0.00093125490974999 | 0.000971449830300688 | 0.00004019492055069797 |
| step 11 `ssh_after` | 0.02176826223359598 m | 0.0717460414836972 m | 0.049977779250101224 m |
| step 11 `r3t_after` | 0.000024938614093583666 | 0.0014019425252578416 | 0.0013770039111642579 |

The source-ordered first row is step-11 `ssh_entry`, so this second interval is
also left-censored. No statement inside step 11 may own a difference already
present at its entry. The round therefore does not attribute the later
barotropic, transport, or FCT growth. Rank-complete steps 1 through 10 are
required to identify the at-bar predecessor and first over-floor successor.

The walk gate reports `PASS_ROUND144_EARLY_GROWTH_WALK`; its JSON SHA-256 is
`eab48dbea72888047fb6cd3aed091bb55c5292aea7c30cc99e5d155b3de42ef2`.
All registered values in both models remain finite through step 30. At every
observed step, the instrumented run reproduces the separately compiled
ordinary `T`, `S`, `u`, `v`, `ssh`, `uu_b`, and `vv_b` state bit-for-bit. The
CPU execution used fp64, libm transcendentals, x64-enabled JAX, and production
JIT; wall time was 617.0913712978363 s.

## Frozen prediction ledger

| ID | Verdict | Evidence |
|---|---|---|
| R144-P1 | CONFIRMED | Existing record admitted unchanged; all nine admission plants fired. |
| R144-P2 | CONFIRMED | Mechanically selected first row is step-11 `ssh_entry`, error 0.03508509204720156 m. |
| R144-P3 | CONFIRMED | Step 11 is already over the floor; the interval is left-censored. |
| R144-P4 | CONFIRMED | No registered row is non-finite through step 30. |
| R144-P5 | CONFIRMED | No model, card, deck, carried state, stabiliser, sea-ice selector, or unmeasured-feature change landed. |

## Acquisition needed

The committed launcher is:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round144_initial_growth_acquisition/run.sh`

It creates the new `ORCA2_OMIP_L4_R144INITIAL` target and the new run
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round144/acquisition/orca2_rung0_initial_growth_10step_np2`.
The writer emits 20 rank-step files (steps 1..10 x two ranks), with the same 14
self-described fields and fp64 payloads as the admitted later windows. The
unchanged 10-step namelist is byte-compared with the pinned rung-0 deck. A
separate calibration run must reproduce all 20 baseline restart shards
byte-for-byte. The checker derives names, ranks, dimensions, and payload lengths
from each header and has nine planted violations.

Preflight compiles the patched writer and both patched consumers, then prints
`ORCA2_ROUND144_INITIAL_GROWTH_PREFLIGHT_READY`. The source-layout,
hidden-deck-delta, and producer-content plants each fire. No in-sandbox MPI run
was attempted.

## Compiled source and source order

The measured record writes entry SSH/thickness/external-mode state before
`stp_2D`, writes the post-barotropic state after that call, and only then calls
stage 1 in
`ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/stprk3.f90:202-221`.
The stage-1 path calls `wzv`, constructs `pFw = e1e2t*ww`, then writes the
stage-1 transport frame in
`ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/traadv.f90:299-320`.

## Standing gates and review

There is no `packages/`, card, deck, or model diff from base `d4fbba7d2` to
this round. Therefore the round-143 GYRE 70-row archive and 30-day trajectory,
the ORCA2 rung-0 independent 200-row ladder, and the rung-7 given-entry 200-row
ladder are unchanged by construction; no scientific row can move. The new
work is a committed measurement adapter, NEMO acquisition instrument, tests,
citation routing, and this receipt.

Focused round-143/144 tests pass 31/31. The full `tests/ocean/fidelity -n 12`
battery result is recorded below after its single permitted launch.

The required separate `codex exec --sandbox read-only` review could not
initialize its in-process app-server client because the read-only filesystem
prevented setup. Verdict: **independent review unavailable in-sandbox**.

No scientific or configuration choice was made: ASKED choices none; UNASKED
choices empty.

## OPEN

1. Operator runs the committed initial-growth launcher. Admit no science value
   unless both-rank coverage, header-derived schema, calibration restart
   identity, source order, and every plant pass.
2. On admitted steps 1 through 10, mechanically select the first row over
   `2e-10`. Step-1 entry is the expected at-bar predecessor; retain a
   contradiction rather than infer it.
3. Split only the bracketed at-bar/over-bar boundary, one recorded operand at a
   time, in compiled stage order. Do not add a stabiliser or change the deck.
