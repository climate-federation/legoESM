# ORCA2 round 148 — V north-neighbour arm refuted; entry inverse first

Date: 2026-10-05. Base `ca49e5e393`; measurement commit `96aaf3366`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round148.md`
at `0446798b9`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The proposed north-neighbour association does not own the V chain. NEMO first
extrapolates `zsshp2_e` and constructs the V-face live depth from the local and
north T cells at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`.
The executed T-pivot T halo sources its northern rows from rows below at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:584-615`.
The private/default-off arm transcribes that source in the existing face-depth
builder (`barotropic_latlon_cgrid.py:552-631`).

At substep 2 the frozen control reproduces 30 unequal full-domain
`mid_depth_v` cells (maximum 899 m) and 68 unequal full-domain `transport_v`
cells (maximum 155776.5627856178 transport units). The arm leaves both
censuses and maxima unchanged. It also leaves `continuity_dv` unchanged at 68
active cells, maximum 155776.5627856178 at `(j,i)=(147,135)`, and leaves
`after_ssh` unchanged at 68 active cells, maximum 0.003203816535399729 m at
the same cell. Thus R148-P3 and R148-P4 are **REFUTED**.

The corrected evidence artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round148/v_fold.json`,
SHA-256 `9fb466a46b0ba0698660257f4d93ec9bf017274f55a6befeed24e57a12d45610`.
The observer and all seven post-association arrays remain bit-exact, so
R148-P1 is **CONFIRMED**.

## Loud correction: R148-P2 is refuted

The preregistration called the midpoint V depth the first wrong compact
operand. That statement is withdrawn. The expanded source-order registry
shows the earlier substep-2 `entry_inverse_v` already differs on 68
full-domain northern V faces, maximum absolute difference
0.03332976059679253 at `(j,i)=(147,29)`. Active-only scoring is still exact
and had hidden this boundary operand.

NEMO updates `hvr_e` and carries it through the seven-field association at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.
legoESM instead reconstructs the entry reciprocal in the same helper that
forms the face depth (`barotropic_latlon_cgrid.py:552-631`). The frozen
30/68 control census is reproduced, but an earlier operand is non-bit;
therefore R148-P2 is mechanically **REFUTED**, not confirmed.

The non-vacuous V-depth plant starts at a cell that is exact before planting,
moves it by one ULP, requires the unequal census to rise by exactly one, and
exits 2 with `STATUS PLANT-FIRED`. The frozen-source control also exits 2.
Because the causal prerequisite failed, the preregistration forbids running
either ORCA2 ladder; R148-P5 is **UNMEASURED_PREREQUISITE_R148-P3**. The arm
remains private/default-off and the production card/deck are unchanged, so
R148-P6 is **CONFIRMED**.

## Shared path and verification

The GYRE base and tip reports compare PASS over all 70 certified rows: zero
status changes, zero violations, and unchanged first-over-bar kt=3. Their
residual NPZ files are byte-identical (SHA-256
`7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`).
All 30 independent daily snapshots are byte-identical to round 147; both
day-30 files have SHA-256
`b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`.

Focused tests pass 15/15. The one permitted `tests/ocean/fidelity -n 12`
battery was stopped as **INCOMPLETE**, not PASS, after reproducing round 147's
documented 99% xdist tail stall in
`test_prediction_plant_is_fail_closed`: 2,567 passed and 7 skipped. Its seven
completed reds comprised four registered pre-existing ratchets plus three
round-local citation failures. The latter exposed the missing mandatory
re-anchor, were repaired with the SequenceMatcher old-to-new map, and the
default citation gate now passes with `unmapped_citations == []`.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`.

ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Walk the earlier 68-cell substep-2 `entry_inverse_v` boundary first. As a
   one-variable arm, carry/substitute NEMO's previously updated and associated
   `hvr_e` against legoESM's reconstructed entry reciprocal; do not alter the
   midpoint depth, transport, difference, and SSH together.
2. Only after the entry reciprocal is exact, return to `mid_depth_v` and its
   row-below T-halo association, then re-test `transport_v`, `continuity_dv`,
   and `after_ssh` in source order.
3. Run the two ORCA2 ladders only after that chain closes. The registered
   approximately 31 PSU salinity exposure remains a hard landing veto. The
   rung-0 month and the parked EVD/bottom-drag merge items remain downstream.

