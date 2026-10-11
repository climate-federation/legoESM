# NEMO SWG level-1 round 1 — card, geometry, and acquisition receipt

Date: 2026-10-11.  Base: `17f6923ff76145ad36cd46de00ec66d5b22064d7`.
Preregistration commit: `31e9c2f9b678a4954ba778fd2ce922c7b8343bac`.
Implementation commit: `b466d706b6fb32dbd528c3b8df9a06c5463387b4`.
Review-fix implementation commit: `4b2feb297e910bb5b85008e5c11d5314a24cd0a7`.
Fix-round-2 implementation commit: `d70660e2780b50acf692649cac323d109f82eeb7`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round1`.

## Review-fix preregistration (2026-10-11)

This section was committed before changing code or running a post-review gate.
It freezes the five operator-review repairs and their falsifiers:

1. The resolved card will state every variable in the six namelist groups read
   by the compiled SWE initialization: `namsbc`, `namlbc`, `nam_vvl`,
   `namdyn_adv`, `namdyn_vor`, and `namdyn_ldf`.  The inventory gate will use a
   frozen literal source inventory rather than derive its expectation from the
   NamedTuple under test.  Removing any one field, planted with `ln_abl`, must
   report that field missing.  Compiled citations are `sbcmod.f90:157-163`,
   `dommsk.f90:149`, `domqco.f90:303-306`, `dynadv.f90:162`,
   `dynvor.f90:813-814`, and `ldfdyn.f90:177-180`.
2. Stage-entry admission will reject a NaN or sentinel anywhere in a stored
   array, including a point outside the owned crop.  A halo-NaN plant must
   raise `Refusal`; the clean synthetic record must remain admitted.
3. The ladder will score T, U, and V records with the card's wet T mask and
   the staggered masks returned by the shared `compute_face_masks`.  A dry-cell
   score plant must be ignored, a wet-cell plant must be DEBT, and any DEBT in
   an executable independent arm must produce a nonzero CLI result.
4. The resolved-inventory test will compare the card against the frozen
   compiled-reader inventory and an exact complete card-field inventory; it
   will no longer compare the NamedTuple to its own `_asdict()` or use a lower
   bound on field count.
5. Hand-derived U/V mask products in the SWG builder will be removed in favor
   of `compute_face_masks`.  The post-fix geometry gate must remain 34/34
   bit-identical and all three existing geometry plants must still exit
   nonzero.

Any changed numerical geometry row, newly admitted non-finite record, masked
wet-cell perturbation, unmasked dry-cell perturbation, or zero exit with DEBT
**REFUTES** its corresponding prediction and will be retained here.

**Preregistration correction, committed before the expanded implementation or
its gates:** item 1's six-group boundary is **REFUTED** by the operator review's
own compiled-source transcript.  It additionally names `ln_1st_euler`,
`nn_rstctl`, `nn_stock`, and `ln_mskland` from compiled
`domain.f90:282-286`, plus `ln_tile` from `domain.f90:288`.  The frozen reader
inventory is therefore expanded to those two groups and five resolved defaults.
Removing any of those fields must fail the same exact inventory gate.  This
correction changes no deck choice: the shipped values remain false, zero,
zero, false, and false respectively.

## Fix-round-2 preregistration (2026-10-11)

This section is committed before the second review fixes or their post-fix
gates.  The operator re-review's terminal verdict is frozen verbatim:

```text
DO NOT SHIP — The resolved engine inventory still omits 59 compiled namelist reads, and the ladder scores two staggered stress records with the wrong mask.
```

Two repairs, and only these two, are in scope:

1. A committed source-enumeration script will parse the 13 compiled SWE
   namelist declarations named by the review and require exactly 131 unique
   reads.  `SwgResolvedNamelist` will state every one with the resolved probe
   value; fields disabled by a controlling switch will be marked inert with
   the compiled branch citation.  The card inventory gate will compare
   against the script's source-derived mapping.  Removing **any** of its 131
   mapped card fields must report that field missing.  Any parsed total other
   than 131, duplicate mapping, unmapped compiled read, missing card field, or
   mismatch with the probe's resolved echo **REFUTES** this prediction.
2. `utau_b` will score with the native U mask and `vtau_b` with the native V
   mask, as required by compiled `sbc_oce.f90:116-117` and
   `sbcmod.f90:567-568`.  A plant assigning either field to the T mask must go
   red; clean reader self-check rows must remain at bar and bit-identical.
   A T-mask result for either field, a plant that stays green, or movement of
   any previously certified geometry/card row **REFUTES** this prediction.

## Verdict

**HELD.**  The operator's second independent review returned `DO NOT SHIP`.
Its two remaining blockers are fixed and committed, but no acquisition should
be run until the operator's next independent re-review returns `SHIP`.

The second review's terminal verdict is quoted verbatim:

```text
DO NOT SHIP — The resolved engine inventory still omits 59 compiled namelist reads, and the ladder scores two staggered stress records with the wrong mask.
```

The first operator review's terminal verdict is retained verbatim:

```text
DO NOT SHIP — The resolved card, stage-record admission, and future fidelity ladder each contain a reproduced blocking defect.
```

The original round-1 in-sandbox command had emitted the following terminal
diagnosis (verbatim); per the operator note it was not retried here:

```text
failed to connect to websocket: IO error: Operation not permitted (os error 1)
ERROR: Reconnecting... waiting for network
```

The complete original attempt remains `codex_review_attempt.log` in the
evidence directory.  The authoritative external review is
`codex_review_r1.log` there.

## Review findings and fixes

1. **Incomplete resolved switch inventory — first fix REFUTED; second fix
   completed.**  The first repair covered only 62 of 131 unique reads and its
   claim above is withdrawn.  A committed enumerator now parses all 13
   declarations on the compiled initialization path
   (`SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:332-352`): control and
   configuration at `SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:199-201`, all
   domain groups at `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:282-289`, lateral
   boundary groups at `SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:149-155`, and the
   six already enumerated readers below.  It reports 13 groups and exactly 131
   unique mapped card fields.  The card states all 131, including each
   structured `sn_cfctl` member, and its complete exact inventory is now 166
   fields.  The 131-way removal plant makes every single omission go red.

   Disabled reads are explicit rather than silently defaulted.  In particular,
   `ln_read_cfg = F` selects the user-defined domain branch at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:292-296`; `ln_rst_list = F` makes
   `nn_stocklist` inert at `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:354-358`;
   `ln_tile = F` bypasses both tile lengths at `:446-472`; the chunk counts are
   carried but disabled by `ln_nc4zip = F` at `:493-496`; and `ln_bdy = F`
   bypasses the boundary-mask consumers at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:192-203`.
2. **Stored-halo non-finite admission — fixed.**  Admission now checks the
   complete stored array while retaining the owned-crop bounds check.  Both a
   halo NaN and a halo sentinel are refused; the complete clean 30-record
   synthetic acquisition remains admitted.
3. **All-ones ladder masks and ungated debt — first fix PARTLY REFUTED; second
   fix completed.**  Restart and future independent rows use the card's wet T
   mask and native U/V slices from the shared staggered-face mask builder.
   `utau_b` is now classified U and `vtau_b` V, matching their compiled
   definitions at `SWG_OMIP_L1/BLD/ppsrc/nemo/sbc_oce.f90:116-117` and restart
   reads at `SWG_OMIP_L1/BLD/ppsrc/nemo/sbcmod.f90:567-568`.  Each old T mask
   differs from the correct stagger at 80 cells, and the wrong-stagger plant
   goes red.  Dry-cell plants remain excluded, wet T/U/V plants remain DEBT,
   and a DEBT result returns exit 1 with its first step.
4. **Vacuous resolved inventory test — fixed.**  The test no longer derives
   its expectation from the NamedTuple under test or accepts a field-count
   lower bound.  It compares two frozen source-derived inventories and proves
   non-vacuity with the two removal plants above.
5. **Duplicated mask numerics — fixed.**  The hand-derived U/V products were
   deleted.  The builder and ladder both call the shared `compute_face_masks`
   operator and take the native NEMO U/V record slices.  The geometry gate
   remains 34/34 bit-identical; signed-y, wet-cell, and thickness plants each
   exit 1.

Focused results at the fix-round-2 implementation commit: 163 SWG tests pass;
the source inventory reports 13 groups, 131 total and 131 unique reads; its
drop-`ln_abl` control exits 2; restart calibration remains at bar and
bit-identical while its numerical plant exits 1; and geometry remains 34/34
bit-identical while all three plants exit 1.  The 39 focused TSUNAMI tests pass.
The VORTEX card suite has 69 passes and the same
three pre-existing stale digest assertions for GYRE, LOCK_EXCHANGE, and
OVERFLOW already reproduced by the independent review at the untouched
round-1 base; their actual digests are unchanged.  TSUNAMI-zco, VORTEX-zco and
VORTEX_VEC-zco retain the independently reviewed hashes `8bb5ed6c1bc031bd`,
`f7489467bc156303`, and `5041b91c67bd1443`.  Acquisition preflight passes at
the fix-round-2 commit without invoking `makenemo` or `mpirun`.

## Frozen predictions and falsifiers

The preregistration was committed before any array in the operator probe was
opened.  It froze exact geometry, reader, acquisition, and planted-violation
criteria.  One source-derived coordinate prediction was **REFUTED and retained**:

- predicted written T origin: -62,500 m; measured: -12,500 m;
- predicted zero-based wet rectangle: rows/columns 3 through 82; measured: 1
  through 80;
- discriminator: the source-local computational array is 88 by 88, while the
  written NetCDF data domain is 84 by 84 after two MPP halo rows on each side
  are removed.  The source coordinate formula was unchanged; the preregistered
  source-to-record index association was wrong.

The survey now states that correction rather than silently changing the frozen
preregistration.

## Card and fail-closed boundary

`SwgResolvedNamelist` states the shipped deck's selected and inert switches.
`SWG_DEVIATIONS` contains only the toolchain removal of `key_xios`.
`build_swg_card()` is a pure configuration of the shared lat-lon C-grid lane,
with the shipped 84 by 84 data domain, 6,400 wet T cells, one executed 500 m
layer, 3,600 s step, 172,800 steps, and reduced gravity 0.02 m/s2.

The card is intentionally not executable.  `SWG_UNMEASURED` contains exactly:

1. ENS total-vorticity with the mask selection;
2. the SWE three-stage explicit-free-surface RK3 program;
3. stage-3-only viscosity, mean wind stress, and before-velocity layer drag.

The independent ladder arm exits 2 with a named `REFUSE` listing all three.
Thus the restart comparison below is explicitly reader calibration, not a
legoESM/NEMO fidelity claim.

## Geometry measurement

The corrected source transcription is bit-identical to all 34 admitted probe
rows: T/U/V/F coordinates, eight e1/e2 metrics, source and card Coriolis,
T/U/V/F masks, and the depth/thickness records.  `geometry.log` ends
`GEOMETRY IDENTICAL`; `geometry.json` contains the clean-tree commit stamp for
the final gate invocation.

All three preregistered controls fired and exited nonzero:

- signed-y Coriolis: 84 unequal T values;
- one flipped wet cell: one unequal mask value;
- one changed layer thickness: one unequal thickness value.

## Record readers and acquisition

The restart reader admitted exactly the ten numbered after-state restarts,
checked each internal step number and 3,600 s step, required finite float64
fields, and removed only the dummy bottom velocity record.  Its self-comparison
is bit-identical and all rows are at the 1e-15 bar; the one-cell plant leaves the
bar and exits nonzero.  Again, this proves the reader and scorer, not ocean
fidelity.

The ladder also accepts the second, stage-entry record kind through the same
admission module.  The additive acquisition consists of a stage-entry writer,
an additive patch on the copied SWE step source, a new
`SWG_OMIP_L1_P3` target, and an operator-only runner.  The default invocation is
preflight only.  It verifies the existing reference executable's SHA-256
`a5b312b8b8fccadd67152d446811c7d2bc9d0e1cd55dce0d3148c8b5c6817eb9`
before doing anything.  A real operator run must admit ten times three ordered
stage frames and compare every numeric field in its own grid-T/U/V/W output
bit-for-bit with the reference probe.  Every nonzero runner exit is preceded by
a named `REFUSE` line.

Preflight passed without invoking `makenemo` or `mpirun`.  The runner must stay
on hold until the external review is available.

## First non-bit statement

There is no model-fidelity row in round 1 because execution is refused.  The
first non-bit measurement in the round is the preregistered written-coordinate
association: the predicted T origin differed by exactly 50,000 m.  It was an
index-frame error, not a changed numerical statement, and correction of the
data-domain crop makes every geometry row bit-identical.  No executable ocean
operator was changed in response.

## Rule 12 disposition

| Landed statement | Shared implementation site | Who can execute it | Other-card disposition |
|---|---|---|---|
| SWG coordinate, mask, and one-layer card construction | SWG-named helpers and builder in the shared recipe module | SWG builder only | TSUNAMI, VORTEX, GYRE, LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 do not dispatch to the SWG case or helpers |
| SWG validation dispatch | Existing shared card validator, guarded by exact case name | SWG card only | Existing case branches are unchanged |
| Geometry, restart, and stage-entry tooling | SWG-named validation scripts | Explicit SWG CLI only | No model card imports or executes these scripts |
| Stage-entry writer | Operator runner copies it into the new NEMO target | New instrumented target only | Reference target and every legoESM card are untouched |

No shared executable numerical statement was changed, so there are no moved
ladder rows to register and no other card can execute this round's additions.
The SWG focused suite passed 23 tests and the TSUNAMI card suite passed 26.
A broader certified-card battery passed 145 tests and reported three stale
digest assertions for GYRE, LOCK_EXCHANGE, and OVERFLOW; the untouched base
checkout reproduces the same three failures with the same actual digests, so
they are pre-existing at the stated base rather than Rule-12 movement.

## Compiled-source citation map

Every source claim below is bound to the exact compiled branch that produced
the operator probe:

- The domain reader states the reviewed `namrun`, `namdom`, and `namtile`
  switches: `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:282-288`.
- The compiled SWE initialization path reaches the domain, SBC, lateral-
  diffusion, advection, and vorticity readers:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:332-352`; its own `namctl` and
  `namcfg` declarations are at `:199-201`.
- The complete domain declaration set, including `namnc4`, is
  `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:282-289`; the complete lateral-
  boundary declaration set is
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:149-155`.
- The complete surface-boundary reader inventory is
  `SWG_OMIP_L1/BLD/ppsrc/nemo/sbcmod.f90:157-163`; the lateral-boundary
  reader is `SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:149`.
- The before-level stresses are native U and V records at
  `SWG_OMIP_L1/BLD/ppsrc/nemo/sbc_oce.f90:116-117`, and restart admission
  requests those staggers explicitly at
  `SWG_OMIP_L1/BLD/ppsrc/nemo/sbcmod.f90:567-568`.
- The complete variable-volume reader inventory is
  `SWG_OMIP_L1/BLD/ppsrc/nemo/domqco.f90:303-306`.
- Momentum advection, vorticity, and lateral-diffusion reader inventories are
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:162`,
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:813-814`, and
  `SWG_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-180` respectively.
- Domain spacing, dimensions, gravity override, and closed lateral boundaries:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-166`.
- T/U/V/F metre coordinates, uniform metrics, and the absolute-y beta plane:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:123-168`.
- The one-layer W/T depth ladder:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:158-168`; flat-bottom and wet-mask
  construction: `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:206-235`.
- The shipped surface freshwater, salt, and heat fluxes are zero:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:84-87`.
- The executed ENS branch supplies total rather than Coriolis-only vorticity:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:250-251`.
- With one executed level, the initialized vertical-advection carry and empty
  interior loop make the bottom subtraction exactly zero:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dynzad.f90:111-146`.
- The three stage lengths are one third, one half, and one full step:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140,196,252`; stage 1 calls advection
  and vorticity in that order:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:148-149`.
- Stage 3 adds lateral diffusion, mean wind, and drag on the step-before
  velocity: `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`; it then advances
  free surface and recomputes thickness ratios:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:281-283`.

## OPEN

1. The operator independently re-reviews fix-round-2 commit `d70660e27`; do not
   run acquisition unless the terminal verdict is `SHIP`.
2. After `SHIP`, the operator runs the committed acquisition runner and returns
   the admitted stage-entry record plus passivity report.
3. Round 2 starts from that admitted record and implements the three declared
   unmeasured arms in the shared lane, including end-to-end propagation of the
   configured reduced gravity, then walks stage order without moving any
   existing at-bar row.

No configuration or carried-state choice is requested in this round.
