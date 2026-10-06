# ORCA2 round 161 — merged-tree month and debt ranking

Date: 2026-10-06. Base `f773f0799`; preregistration `796c53802`;
measurement tip `2fccf5976`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round161/`.
Verdict: **HELD**. The rung-0 month still first becomes non-finite at step 36,
and the exact external-mode halo pair still fails its salinity veto. The new
measurement gate and this receipt may land; no production model file changed.

Every rung-0 number below is **independent**: the hierarchy card starts from
its own climatological T/S, zero velocity, and zero sea surface. Every rung-7
number is **given NEMO's entry** under Decision 52. The labels are not mixed.
Sea ice, all six sea-ice selectors, and the shipped card's
`unmeasured_features` tuple are unchanged.

## Independent rung-0 month

The unchanged round-130 scorer re-admitted NEMO's finite two-rank kt=240
restart, instantiated the frozen rung-0 card, and advanced the production-JIT
CPU/fp64/x64/libm model. legoESM first becomes non-finite at exactly the prior
boundary:

```text
STATUS REFUSE: first non-finite step=36 field={'field': 'T', 'index': [86, 159, 0], 'value': nan}
```

The run reached step 30 in 186.0 seconds. It therefore **REFUTES** R161-P1:
the round-160 merge and constant-background EVD repair do not advance the
independent month beyond step 36. No terminal candidate exists, so all five
terminal RMS/maximum rows and both terminal rankings remain **UNMEASURED**.
The admitted NEMO program itself completes the three RK3 stages in source order
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227`; no stabiliser or deck
change is authorized by legoESM's refusal.

R161-P2 is **CONFIRMED**. Record admission completed before stepping, and the
committed scorer's `terminal-ulp` and `terminal-nonfinite` tests both refuse
their planted reports. The month log has SHA-256
`eaeb8481e5eb33d0bdf9f2bd28d998060a15e3ff31e8c9d7e9f038a41adc3ab7`.

## Given-NEMO-entry rung-7 V debt

The round-160 comparison artifact is pinned by its existing SHA-256
`e44b8a5d099c7d511f409192d261a04313a96a037595ecaa35da29a733c618c5`.
The new committed gate ranks only away-moving V-maximum rows, so every compared
quantity has units m/s and no cross-field units are mixed.

The registered kt=10 stage-3 row ranks **1 of 34**: maximum V error moves from
`0.4363833806287545` to `1.2990882244341995 m/s`, a
`+0.862704843805445 m/s` regression. The next row is kt=10 stage 2 at
`+0.856699655196707 m/s`. R161-P3 is **CONFIRMED**. The target-delta plant
refuses, and the ranking artifact has SHA-256
`4a5e0ea12e3d8921667acb06b837a5f9f8b52bc4a8e71823ae6e864861607c78`.
This is a magnitude ranking, not a new attribution: the round-160 batch joined
the GYRE merge and EVD repair, so this round does not assign the row to either
statement alone.

## Independent rung-0 halo replay

The control and private complete U-cyclic/V-fold association arm each complete
all 200 rows on the merged tree. NEMO associates the seven external-mode fields
together at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`; its two-rank
cyclic exchange is at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979` and
`:2060-2068`, followed by the north-fold dispatch at `:2105-2113`.
The executed T-pivot V permutation, overwrite, and sign are at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1712-1738` and
`:1747-1766`.

The private arm moves 195/200 rows, loses zero bit-exact rows, and leaves the
first debt at kt=1 stage-1 T. By RMS, 10 moved rows go toward NEMO and 185 go
away; by maximum, 65 go toward, 75 away, and 55 are equal. The decisive
salinity maximum moves from `0.4156673855238111` to
`0.41567240155913865`, an increase of `5.0160353275430225e-6`. The existing
salinity gate refuses, while its exact-row-loss plant also fires.

R161-P4 and R161-P5 are **CONFIRMED**. The merged statements did not remove
the compensating partner: the veto is the same magnitude as round 159's
`5.0160347484e-6` within `5.8e-13`. The halo pair remains separate and
**HELD**; rung 7 and the shared-card gates correctly were not run past this
terminal rung-0 veto. The control, candidate, and comparison artifact SHA-256
values are, respectively,
`09d9165b63c8cfe91f361a99eb2a8f8ec4bc2f114181175e1c40f63650b74d65`,
`2c8402a10ebc4f28b4ca6d2f42e5de563ba78c81d39700009040d69cb94b50d0`,
and `a4e3f249f956dcbdcea7010965c0e96c0c83fa4fa22052cf1d101601d1db9df2`.

## Prediction disposition

| prediction | disposition |
|---|---|
| R161-P1 month completes | **REFUTED**: first non-finite remains step 36 T `[86,159,0]`; terminal score unmeasured. |
| R161-P2 record and plants bind | **CONFIRMED**: record admission precedes stepping and both terminal plants refuse. |
| R161-P3 rung-7 V regression ranks first | **CONFIRMED**: rank 1/34 by same-unit maximum delta. |
| R161-P4 halo salinity compensation survives | **CONFIRMED**: maximum increases by `5.0160353275430225e-6`. |
| R161-P5 halo exact rows/first boundary survive | **CONFIRMED**: zero exact-row losses; first debt unchanged. |

## Validation and choices

The round-161 ranking tests pass 2/2 and its planted target delta fires. The
focused gate battery passes 22/22. The round receipt citation gate passes all
7 citations with zero unmapped spans; its compiled-line plant fires. The
cumulative receipt gate passes all 274 citations with zero unmapped spans.

The prescribed separate read-only review was attempted, but `codex exec`
could not initialize its app-server client on the read-only filesystem:
**independent review unavailable in-sandbox**.

The single `tests/ocean/fidelity -n 12` battery collected 2,669 tests. Before
being interrupted after ten minutes without progress at 99%, it recorded
2,644 passes, 7 skips, 5 failures, and 13 unfinished tests. The five observed
reds reproduce in isolation: the registered SI3 scalar-math provenance red;
the GYRE round-129 spread record rejecting a moved certified-year harness; the
round-179 acquisition plant being pre-empted by this receipt's dirty-tree
check; 12 VORTEX drivers failing the process-global allow-dirty scope ratchet;
and the registered worktree-stamp ratchet (11 emitters). No failure names a
round-161 gate or model path. The dirty-tree-pre-empted round-179 test is
re-run from the clean receipt commit below.

ASKED choices: none. UNASKED choices: empty. No configuration, forcing,
carried-state policy, stabiliser, sea-ice selector, or production model file
changed.

## OPEN

1. The rung-0 independent month is still blocked at step 36 T `[86,159,0]`.
   Resume the already-named finite-growth walk; do not add a stabiliser or
   change the hierarchy deck.
2. The rung-7 kt=10 stage-3 V maximum is the largest V-maximum regression in
   its batch but remains unattributed between the merged statements. Walk it
   one recorded operand at a time before any model landing.
3. Keep the U-cyclic/V-fold halo pair separate and HELD. Its local statement is
   exact, but its salinity compensation survives the merge and 185/195 moved
   RMS rows go away from NEMO.
