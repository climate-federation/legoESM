# ORCA2 round 264 — stage-ratio placement reconciliation

Date: 2026-10-12. Base: `f73a2e6b1`. Preregistration commit:
`d6e26313d`. Corrected measurement commit: `9ba3b7399`. Disposition:
**HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry at
its tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, on CPU fp64/libm production JIT. No NEMO acquisition was needed.

## Compiled statement and controlled placement

NEMO builds the raw stage ratios in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217`, then associates
the scalar V ratio with sign `+1` before storing it at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170`. The
compiled T-pivot V copy is in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`. The associated
ratio feeds both the stage correction at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279` and the U/V
metric transports at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.

The frozen three-way control used the complete private fold unit and D114's
resolved mesh route everywhere else:

1. `off`: no scalar V-ratio association;
2. `round262`: the source-complete coupling, associating the ratio before both
   the live stage-geometry consumer and the carried-ratio consumer;
3. `nemo`: the preregistered name for a carry-store-only isolation.

The third name is retained because it is frozen into the preregistration and
artifacts, but its interpretation is corrected: it is **not** the complete
compiled statement because it omits the earlier live stage-geometry consumer.
The source-complete mode is `round262`.

## Frozen predictions and verdicts

R264-P1 is **CONFIRMED**. At matched NEMO program points, `off` reproduces
round 260 under both labels: stage-2 fold-band T is
`6.435500132795369e-9` K and S is `1.3842782919937235e-8` PSU.

R264-P2 is **REFUTED and retained**. The source-complete `round262` placement
does not create the predicted O(1) regression. It closes T to
`2.220446049250313e-16` K and S to `7.105427357601002e-15` PSU under both
labels. All three modes have the identical stage-1 `zFv` digest
`026051b74c144f8568eedc2a12d9ffed18bf3ebf8806f5012922f932c4a68d41`.
The carry-store-only isolation is exactly equal to `off`; therefore the
causal effect is the live stage-geometry consumer, not the carried store by
itself.

R264-P3 is **REFUTED and retained** for that same reason: the preregistered
carry-store-only arm neither recovers nor worsens the matched stage-2 result.
The source statement has to be coupled once, before both consumers, exactly
as the compiled program orders it.

R264-P4 is **CONFIRMED**. With that source-complete placement, both the
**independent** and **given NEMO's entry** prefixes complete kt=1 and kt=2;
all 18 carried arrays are finite after each step and there is no
live-thickness refusal. This retracts the round-262 pre-kt2 refusal as a
property of the older integration context, not of the scalar association.

R264-P5 is **CONFIRMED**. Both-label and three-mode coverage, record identity,
passivity, matched program points, source order, the next-boundary census,
and one-bit sensitivity are mechanically gated; all sixteen plants refuse.
The first combined-process measurement aborted with LLVM's compiler-memory
limit, so every citable arm was rerun in an isolated process. No number from
the aborted process is used.

The first corrected probe still compared stage 2 before legoESM's private
post-stage association against NEMO's post-`lbc_lnk` frame. Those six JSONs
are retained as `unmatched_stage2_*` and are explicitly non-citable. After
the program point was corrected, the source-complete arm's next valued
boundary is kt=1 stage-2 completed U: fold-band maximum
`0.003382014663707235` m/s. V is `3.199915622040597e-6` m/s and SSH is exact.
That U boundary, not T/S, owns the next source-ordered walk.

No Decision-96 landing vote is reached. The package arm and its temporary
helpers were removed; the final two model files are byte-identical to the
base. Consequently the final tree cannot move OMT, rung 0, rung 10, GYRE,
DINO, or tank trajectories.

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round264/`.
Principal SHA-256 values are:

- corrected six-arm classification:
  `97dbb445d2ddcb245af461a466f8edc272884a172c9f663b0a6966c3a6e50244`;
- independent / given source-complete measurements:
  `cd86aab52882d10d80c5018cb4d64238c0a55a6c55eefe80a5ddcf7a79e8a1a5` /
  `61721e448c4b1e17f80ae69b099e9f2680aae428818c4dd4d4baeaf3a294fd1e`;
- independent / given two-step prefixes:
  `19284a7fe2908269766e0af7712f8bbe40c5ec527e1bfef94d3f672b4dbe34ca` /
  `888019a693ce87759dba98761dbcba903c6d0f8311592e4f34dff709323c8a34`;
- prefix classification:
  `93a5c0fda4d071df064da101e18a9cc1a6eb4d2f58fcc26237511678102eaa60`;
- retained preregistered-classifier refusal:
  `9f6195f43c15c32fa7ee7990fb15caabeb6c5470496c7836381b2f0ab822b10d`.

The default cumulative citation gate and this receipt's citation gate pass
with zero unmapped citations. Their JSON SHA-256 values are
`3edcc1a37450fa2d821ccc222878d99446a5f40103a6fc4cb3ea7be786274075`
and `32c9eb9532f59b1f29dbe01bd327ff44cb6cdac7877ba1ed96b121209f454b45`.
The planted two-line shift of the compiled association citation refuses; its
JSON SHA-256 is
`7f89de512395921e23a3b3a08654598f0d63b2a0285a686e0c293272acc21d15`.

Focused tests, the once-only fidelity battery, and independent review are
reported after they run on the committed receipt.

## OPEN

Integrate the scalar `r3va` association once at the compiled call site so
both consumers read the associated value, inside the complete atomic fold
unit. Reproduce the matched T/S closure and finite kt=1..2 prefix, then walk
the kt=1 stage-2 completed U boundary in compiled U-update order. The unit
remains private until it completes OMT-4 kt=1..10 under both labels and passes
the full Decision-96 OMT-1..4, rung-0, rung-10, GYRE-year, DINO-month, tank,
moved-row, and test gates. Decision 116 is not re-emitted.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
