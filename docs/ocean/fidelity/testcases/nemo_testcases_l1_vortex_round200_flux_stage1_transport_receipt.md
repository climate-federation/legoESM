# VORTEX round 16 (lane round 200) — the flux card's stage-1 walk, and what it does NOT yet name

**ROUND_STATUS: HELD (measurement landed, no model change).**

**RETRACTION, up front.**  The first draft of this receipt named the stage
advective transport statement (`stprk3_stg.f90:276-277`) as the OWNER of the
flux card's kt=2 velocity residual.  **That claim is withdrawn.**  It is the
first boundary in NEMO's stage-1 order that is not bit-identical, and it stays
reported as that — but it cannot produce the magnitude it was said to own, and
two further controls in this round agree.  The independent reviewer caught it;
the arithmetic is in "Why the first non-bit boundary is not the owner" below.

What this round does land: the flux card's own per-stage term record, which did
not exist; a committed walk of NEMO's stage-1 statement chain against it; and
the two measurements that narrow the owner to **one of two statements**, with
the single run that separates them preregistered.  No model code changes, so
both VORTEX cards' certified registries and GYRE are unmoved by construction.

Pre-registration, frozen before any measurement:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round200.md`
(sha256 `1a44dc97480ef7ed13cdb7d4a7e6a82b03552dde77fd0be5d08adfae9c87fab0`,
`phase3/round200/prereg.sha256`).

## The target, and where it lives

The flux card `VORTEX-zco` carries, at the round-199 certified registry,
kt=2 `u = 1.1355340046037554e-07` and `v = 1.1354649764871994e-07` against a
bar of 1e-15; its kt=2 sea surface height has been at the bar since round 198
and its kt=2 tracers are at the bar.  Rounds 198 and 199 moved the vector card
to the bar and left these two rows untouched (0/50).

The existing kt=2 walk, re-run on this tip
(`phase3/round200/kt2_walk_flux.{json,log}`), reproduces the certified row from
its own arm 0 — `1.1355340046037554e-07`, the registry value to every digit —
and then shows that **no stage boundary owns the residual**: handing the model
NEMO's own stage-1 output, or its stage-2 output, does not reduce it.

| arm | u | v |
|---|---:|---:|
| the card as it ships | `1.135534e-07` | `1.135465e-07` |
| + NEMO's kt=1 entry | `1.135534e-07` | `1.135465e-07` |
| + NEMO's external-solve handoff | `1.135534e-07` | `1.135465e-07` |
| + NEMO's stage-1 output | `1.177202e-07` | `1.177235e-07` |
| + NEMO's stage-2 output | `1.203856e-07` | `1.201635e-07` |

The stage-local rows say why: **every stage re-makes the same error.**  Run
from NEMO's own entry and scored against NEMO's own output for that stage,
stage 1 alone is `4.358845e-08`, stage 2 alone `6.181787e-08`, stage 3 alone
`1.203856e-07`.  That is prediction P2's falsifier arm, recorded as such: the
owner is a per-stage statement, not a boundary, and the earliest place it can
be caught is stage 1.

## Why stage 1 is the right place to look, read off the compiled source

Under flux form NEMO's stage 1 runs exactly ONE momentum statement beyond what
`stp_2D` already completed.  In
`tests/VORTEX_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90`:

* `:262-272` the barotropic velocity correction `zub/zvb`;
* `:276-277` the advective transports
  `zFu = e2u*(e3t_1d*(1+r3u(Kmm)*umask)) * ( uu(Kmm) + zub*umask )`;
* `:298` `CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )`;
* `:301` `zFw = e1e2t*ww`;
* `:315` `IF( .NOT.ln_dynadv_vec ) CALL dyn_adv( ..., zFu, zFv, zFw )` — the
  comment on `:311` is NEMO's own: *"Flux Form : add the missing ADV to the 1st
  stage 3D RHS"*;
* `:371-378` the thickness-weighted update, `:437-450` the barotropic
  replacement and the stage output.

The advection is "missing" from the three-dimensional right-hand side because
`stp2d.f90:170` calls `dyn_adv_up3( ..., pUe=Ue_rhs, pVe=Ve_rhs )`, which the
compiled source labels *"2D RHS only"*, and cumulates its depth mean into the
barotropic forcing at `:183`.  So the flux card's pre-stage momentum array and
legoESM's own completed right-hand side carry the same NAME and not the same
QUANTITY — that row is reported below and is explicitly **not** used to name an
owner.

## The acquisition this round built, and its admission

No record carried those operands: round 192's stage-term record is the VECTOR
card's and opens only at stages 2 and 3.  This round extends the round-192
instrument to the flux build.

* One patched file, `stprk3_stg.F90`, additions only (the patch has exactly one
  `-` line, its own diff header).  The advection trend is recorded at its CALL
  SITE, so `dynadv.F90` stays shipped.
* A new read-only writer opens at **every** stage after that stage's WZV block
  and closes after the common barotropic correction, carrying the stage's `Kmm`
  velocities and sea surface height, `ww`, the three advective transports
  `zFu/zFv/zFw`, and the momentum right-hand side on both sides of every
  contributing statement, plus the state after the explicit update, after the
  implicit vertical-diffusion integration, and after the barotropic correction.
* Paired builds `VORTEX_R16_OMIP_L1` (shipped) and `VORTEX_R16_OMIP_L1_P3`
  (instrumented), same deck, one `mpirun` each.
* **Admission**: `restart_byte_identical: true`, `status: ADMITTED`
  (`phase3/round200/oracle_stage123_flux_terms/vortex_round200_stage123flx_admission.json`);
  the checker's plant turns it red (`PLANT_FIRED header`,
  `phase3/round200/acquisition.log`).  Note BD is respected: the checker parses
  every record's header and every group's own rank and extents; the only
  hard-coded expectations are the magic string and the list of required names.

## The walk, in NEMO's own stage-1 order

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round200_flux_stage1.py`,
every value taken from the **production-jitted** step through seams that
already exist (`phase3/round200/flux_stage1_walk.{json,log}`).  Exactness here
is bit equality, never AT-BAR.

| boundary | compiled citation | cells unequal | max abs | relative |
|---|---|---:|---:|---:|
| pre-stage RHS `u` | `stp2d.f90:126-171` | 18997 | `1.370473e-05` | `1.623e-01` |
| pre-stage RHS `v` | `stp2d.f90:126-171` | 18884 | `1.370473e-05` | `1.623e-01` |
| **`zFu`** | **`stprk3_stg.f90:276`** | **118** | **`1.862645e-09`** | `1.423e-16` |
| **`zFv`** | **`stprk3_stg.f90:277`** | **128** | **`1.862645e-09`** | `1.423e-16` |
| `zFw` | `stprk3_stg.f90:301` | 36443 | `2.447726e-08` | `2.800e-13` |
| `ww` | `stprk3_stg.f90:298` | 36444 | `2.719696e-17` | `2.800e-13` |
| stage-1 output `u` | `stprk3_stg.f90:437-450` | 6253 | `4.358845e-08` | `4.979e-08` |
| stage-1 output `v` | `stprk3_stg.f90:437-450` | 6338 | `4.354887e-08` | `4.974e-08` |

The first two rows are **convention-sensitive and carry no attribution**, for
the reason given above: in flux form NEMO's three-dimensional array does not
hold the advection and legoESM's does.

**FIRST NON-BIT BOUNDARY (not the owner): the stage advective transport
statement, `stprk3_stg.f90:276-277` of `VORTEX_OMIP_L1_P3`** — 118 of 36600
active `u` faces and 128 of 36600 active `v` faces, one unit in the last place
of the field's own peak.  Everything downstream of it in the stage is non-bit,
and nothing upstream of it that this walk can score as the same quantity on
both sides is.  Read the next section before using that sentence.

## What the 118 cells look like

`phase3/round200/zfu_cells.log`.  They are **not** a rim or mask artefact: zero
of them lie on the two-cell rim.  They sit in the vortex core
(`j` 23-35, `i` 28-36 on `u`; `j` 25-36, `i` 23-39 on `v`), spread over every
level (6 to 14 per level), and the largest discrepancy is exactly
`1.862645149230957e-09`, which is `2^-29` — one unit in the last place of a
double of magnitude `1e7`, the magnitude of the largest transports in the
field.  Measured per cell the discrepancy reaches 3662 units in the last place
on `u` and 7325 on `v`, because the affected faces include ones whose own
transport is as small as `4.7e+02`.

Those numbers are produced by the walk itself and land in its report
(`phase3/round200/flux_stage1_walk.json`, every row's `structure` block); they
are no longer quoted from a throwaway probe.

## Why the first non-bit boundary is NOT the owner

**Two independent arguments, both refuting it.**

**1. The amplification required does not exist.**  The transports are wrong by
`1.862645e-09` absolute, `1.4e-16` relative — one unit in the last place.  The
stage-1 OUTPUT is wrong by `4.358845e-08` m/s.  The whole of stage 1's explicit
velocity increment, taken from NEMO's own record as `update_u − kmm_u`, peaks
at `0.0811` m/s, so the output error is `5.4e-07` of the entire increment while
its inputs are wrong by `1.4e-16` of theirs.  That is more than eight orders of
amplification across one advection operator and one division by a thickness,
and nothing in those statements can supply it.  The same arithmetic disposes of
`ww` and `zFw`: a `zFw` discrepancy of `2.45e-08` on a cell of volume
`~4.4e+11` m³ moves the velocity by `~1e-16` over the stage, not by `4e-08`.
**The producer is downstream of `:301`.**

**2. The output error has no depth-uniform part, so the barotropic correction
cannot be making it either.**  The stage's barotropic replacement
(`stprk3_stg.f90:412-419`) adds exactly ONE number per column, the same at
every level.  Measured on the output rows this round: the largest column mean
of the error is `7.715e-17` on `u` and `8.772e-17` on `v` — at the bar — while
the largest departure from the column mean is `4.359e-08` and `4.355e-08`, the
whole of it.  **The error is entirely depth-varying.**

Those two together leave exactly two statements, both inside stage 1 and both
between `:301` and `:412`:

* the flux-form advection trend, `stprk3_stg.f90:315`
  (`IF( .NOT.ln_dynadv_vec ) CALL dyn_adv( ..., zFu, zFv, zFw )`), and
* the thickness-weighted explicit update, `stprk3_stg.f90:371-378`.

## Prediction P3's disposition: RECORDED BUT NOT RUN

The pre-registration predicted the flux-form advection trend and set its
falsifier as "`dyn_adv`'s trend bit-identical while another term is not".  **That
test was not run this round, and the receipt says so rather than implying the
walk settled it.**  The record this round built *does* carry both groups NEMO
needs for it — `adv_u/adv_v` (the accumulator after `:315`) and
`update_u/update_v` (the state after `:371-378`) — and the checker requires
both present.  What is missing is on legoESM's side: its momentum-accumulator
exposure is refused for stage 1 by construction
(`expose_momentum_operator_stage must be 2 or 3`), and there is no stage-1
raw-momentum seam.  Adding one is a model-file edit and therefore its own
round, under the full gate.  Because the oracle side is already acquired, that
round needs no NEMO run at all.

## Non-vacuity

* **Seam control, inside the walk.**  Every row is read out of a state slot the
  ordinary step also fills, so a hook that went inert would hand the walk the
  plain step output and each row would still score.  The walk runs the plain
  step once and refuses unless every exposed slot differs from it.  A plant
  that perturbs the candidate afterwards proves the scoring reacts, not that
  the seam is live, and is not relied on for this.
* **Plants, now a DIFFERENCE and not a status read.**  The first draft read
  visibility off the planted row's bit-exact status, and every row of this
  walk is already non-bit, so every plant would have reported `VISIBLE`
  whether or not the perturbation did anything.  The reviewer caught it; the
  plant is now a difference against the unplanted run of the same row.  All
  eight fire (`phase3/round200/walk_plants.txt`).
* **Record boundary control.**  The walk refuses before scoring anything unless
  the new record's post-correction group is identical, cell for cell, to the
  older stage-1 output record — a writer placed at the wrong boundary cannot be
  believed.  It passes.
* **The seam control itself is shown to fail.**  It is a module-level
  function, and a unit control feeds it exactly what it guards against — an
  exposed slot identical to the ordinary step output in every cell — and
  requires it to refuse, then requires it to stay silent on a live seam.
* **Unit controls** (`tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round200_flux_stage1.py`,
  7 passed): a synthetic record on a deliberately non-square plane; the checker
  admits a well-formed record, refuses a missing operand, refuses an unknown
  stage, refuses a disagreeing group count; the reader strips NEMO's halo and
  transposes rather than taking the first interior-sized block.

## Both cards and GYRE

**No model file changed this round.**
`git diff --name-only d7de69d51..HEAD -- packages src` is empty, so both
VORTEX cards' certified 50-row registries and every GYRE number are unmoved by
construction — 0 of 50 rows move on either card, and the flux card's kt=2
velocity is the same `1.1355340046037554e-07` the walk's own calibration arm
reproduces from the certified ladder.  The DINO month gate is not triggered by
a docs-and-harness landing (`land.sh` keys it on a `packages`/`src` diff) and
DINO's reference stays `2.053801168e-03` K.

## Choices made this round

| choice | ASKED / UNASKED |
|---|---|
| Extend the round-192 instrument to the flux build rather than write a second one | not a scientific choice; note BT ordered the acquisition |
| Record at the advection CALL SITE in `stprk3_stg` instead of patching `dynadv.F90` | mechanical: one fewer shipped file patched, same boundary |
| Deck for the new build is the committed flux deck, unchanged (`namelist_cfg_omip_l1.patch`) | no new option; the run.sh guard refuses any other advection-form pair |
| Report the pre-stage RHS row but exclude it from attribution | required by the quantity mismatch the compiled source states |
| Withdraw the first draft's owner claim rather than defend it | forced by the reviewer's scaling refutation and this round's own column-structure measurement |

UNASKED list: **empty**.  No default, scheme, bound, threshold or deck value
was changed; nothing the NEMO deck does not pin was selected.

## OPEN, in order

1. **Run prediction P3.  NO NEMO RUN IS NEEDED — the oracle side is already in
   hand.**  Add a stage-1 momentum-accumulator seam to legoESM (the existing
   one refuses stage 1 by construction) and score two rows from the record
   this round committed: the accumulator after the flux-form advection call
   (`adv_u/adv_v`, `stprk3_stg.f90:315`) and the state after the
   thickness-weighted update (`update_u/update_v`, `:371-378`).  Those two
   statements are the only candidates left, by the two refutations above.
   Prediction, to be frozen first: the advection trend carries it.  Falsifier:
   the advection accumulator is bit-identical and the update is not, which
   would make it the thickness weighting.  The seam is a model-file edit and
   needs the full Decision 43/45/55/59 gate, the vector card proven inert and
   GYRE byte-identical.
2. The transports' own one-unit-in-the-last-place difference
   (`stprk3_stg.f90:276-277`) stays open as a SEPARATE, smaller debt; split its
   two factors by recording NEMO's `zub/zvb` (`:270`) and its
   `e3t_1d*(1+r3u(Kmm)*umask)` face thickness, which this round's record does
   not carry, and score them against legoESM's existing
   `expose_stage1_transport_operand` arms.
3. The `fmask`-versus-`fe3mask` `r3f` factor (inert only at `rn_shlat=0`; cite
   and test at the cards' own `rn_shlat`).
4. Decision 74's 30/15/10-km ladder, once the flux card's kt=2 rows are at the
   bar or proven harness floor.
