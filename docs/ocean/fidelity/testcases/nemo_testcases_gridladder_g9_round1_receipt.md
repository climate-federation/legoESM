# Grid ladder rung G9 — round 1 receipt (gridladder_r46)

Status: **STOPPED_FOR_RECORD.** G9 = certified G5 + ORCA2's tracer
advection.  Deck, card and acquisition are built.  Nothing is scored against
NEMO, because no G9 NEMO record exists yet.

- Preregistration: `PREREG_nemo_testcases_gridladder_g9_round1.md`,
  committed `162f603ba` before any card change.
- Build commit `d6887bfc1`.  Line-neutral model form and citation re-anchor
  `cbd070c74`.
- Evidence: `gridladder_rounds/round46/` (`manifest.sha256` `ec1f62fe08ccfdf9`; state pickles excluded).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

## 1. Survey: G5 does not run ORCA2's tracer advection

| item | value |
|---|---|
| ORCA2 `&namtra_adv` | `ln_traadv_fct = T`, `nn_fct_h = 2`, `nn_fct_v = 2` (`ORCA2_OMIP_L4/EXP00/namelist_cfg:293-295`) |
| ORCA2 companions | from `namelist_ref`: every other scheme `F`; `nn_fct_imp = 1`, inert because `ln_zad_Aimp = F` |
| G5 deck and G5 record | `ln_traadv_OFF = .true.` (no tracer advection) |
| G5 card | `tracer_advection = "none"` |
| reusable transcription | the ORCA2 identity's `fct2`: centred 2nd order at RK3 stages 1-2, two-step upstream + `nonosc` at stage 3; the resolved raw-mesh V mask (D114) |

Searched for: `namtra_adv`, `traadv`, `tra_adv`, `fct2`, `nonosc`,
`tracer_advection`, `vmask`, `_resolved_stage_vmask_arm`,
`nemo_qco_resolved_mesh_operands`, `rk3_vector_raw_vmask`, across
`packages/`, `scripts/`, `tests/` and the receipts.  The tracer operator
exists; no physics code was written.  Next D128 item after G9: isoneutral
tracer diffusion.

**ORCA2 card, report only.**  Its tracer advection is the ORCA2 identity's
`fct2` (orders 2/2, `nn_fct_imp = 1`, CEN2 at stages 1-2, FCT at stage 3).
That matches `&namtra_adv`.  Its stage transport uses the compact V mask —
the gap below.

## 2. NEMO statements (each opened at its SUBROUTINE)

`traadv_fct.f90`, `traadv_cen.f90` and `dommsk.f90` are byte-identical in
the G5 record build and ORCA2_OMIP_L4; `traadv.f90` is identical in the
record build and ORCA2.  These lines hold in both.

| statement | line | SUBROUTINE |
|---|---|---|
| `NAMELIST/namtra_adv/` read | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:586-596` | `tra_adv_init` (:577) |
| `ln_traadv_fct` -> `nadv = np_FCT` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:626-633` | `tra_adv_init` |
| RK3: FCT at stage 3 only | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:491-494` | `tra_adv` (:463) |
| dispatch: stage 3 FCT, stages 1-2 centred | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:529-535` | `tra_adv` |
| centred 2nd-order horizontal flux | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_cen.f90:149-158` | `tra_adv_cen_t` |
| FCT upstream two-step | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:171` | `tra_adv_fct` (:77) |
| antidiffusive fluxes, order 2 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:195-199` | `tra_adv_fct` |
| the limiter call | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316` | `tra_adv_fct` -> `nonosc` (:743) |
| call, every stage | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:637` | `stp_RK3_stg` (:78) |
| the transport it receives | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283` | `stp_RK3_stg` |
| `vmask`: compact product, then the V fold exchange | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:211`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:232` | `dom_msk` (:116) |

## 3. The fold gap is part of G9

G9's tracer update is the first consumer of the T-pivot row's north-face
transport.  So G9 selects NEMO's fold-associated V mask in the stage
transport — the ORCA2 lane's round-303 statement.  The partner top-V fold
exchange is already selected on every G card.

Measured on NEMO's own G8 record (kt 1, stage 1, the record binary's
transport-operand dump; 30 wet levels):

- NEMO's `vmask` has 1,319 wet V faces on row 147 (the T-pivot row);
- `zFv` is nonzero on all 1,319; its maximum, 4.0158e5 m3/s, equals row 146's;
- the card's compact face mask differs from NEMO's on exactly 1,319 cells,
  all on row 147;
- the resolved D114 mask differs on 0 cells.

A committed test reads the dump and asserts all of this.

## 4. What changed

**Deck.**  G5's deck changes in four `&namtra_adv` lines only:
`ln_traadv_OFF` `.true.` -> `.false.`, `ln_traadv_fct` `.false.` -> `.true.`,
and `nn_fct_h = 2`, `nn_fct_v = 2` are added.  A test pins this diff and checks
every `&namtra_adv` key resolves to ORCA2's value.  Deck SHA-256
`570db996…0720`.  The ice namelist is G5's, byte for byte.

**Card.**  Case `ORCA2G9-tadv` uses G5's rung tables and changes two
selections:

- `tracer_advection = "fct2"` (the ORCA2 identity's own value) instead of
  `"none"`;
- `rk3_stage_transport_raw_vmask = True`: every stage transport reads the
  resolved raw-mesh V mask.  This is one new config field, default `False`,
  wired on the existing keyword line of the shared stage program.

The deck reader picks the `&namtra_adv` switch set by `ln_traadv_fct`, and
refuses stated `nn_fct_imp`, `nn_cen_h/v`, `ln_mus_ups`, `nn_ubs_v`.  The
validator now ties the deck's `ln_traadv_fct` to the case: before this, the
G5 builder accepted a G9 deck silently.  Fail-closed: `ORCA2G9_UNMEASURED`
lists `nemo_post_si3_surface_records`, `g9_record_not_admitted` and
`tracer_advection_fold_step`.

**Other cards are unchanged.**

- G2–G8 cards built at the base commit and at this tree: configs identical
  except the new field, `False` on all seven; coordinates, grids, initial
  states and unmeasured lists identical (`other_cards_unchanged.txt`).
- Traced step jaxpr, addresses masked, base vs tree: G5 `339a8016d262c4f6`
  both sides, G8 `03ccdb94113eab61` both sides (`jaxpr/`).
- Only the recipe assigns the new field (test).

## 5. Card-side check (card against card, not a NEMO score)

On G5's recorded surface input (`card_g9_vs_g5_kt2.txt`,
`card_g9_mask_vs_control.txt`):

- G9 vs G5 after kt 2: T and S differ on 430,552 of 799,200 cells; all fields
  finite.
- G9 vs the one-variable control (mask off), after kt 1: sea level and
  barotropic velocities identical; T differs on rows 145-147 (2,682 cells),
  S on rows 146-147 (2,660), u on row 147, v on rows 147-148.  The selection
  acts only on the fold band at kt 1, as intended.
- **Flag (PLAUSIBLE card defect, not a NEMO comparison).**  With the mask,
  pivot-row surface salinity after kt 1 spans 22.28–41.09 PSU (control
  26.52–35.50; G5 after kt 2 26.53–33.55), and 44.52 PSU after kt 2 at
  (j, i) = (147, 131).  One 3-hour step should not move salinity by
  13 PSU.  Leading candidate: the card's tracer update does not make the
  pivot row self-mirrored the way NEMO's end-of-stage tracer fold exchange
  does (round 303: "NEMO's row-147 T is self-mirrored; the card's is not").
  The record's stage-1 operand dump (NEMO's `vv`, `zvb`, `vmask`, `zFv`) and
  stage tracer trends on row 147 discriminate this.  Not fixed this round:
  no record, and it is not yet a cited statement.

## 6. Tests and plants (CPU, fp64)

| battery | result |
|---|---|
| new G9 file | 13 passed (295 s) |
| G2 ladder, G7, G8 files | 44 passed, 9 deselected |
| G3, G5, G6 files | 39 passed, 2 deselected |
| default gate: citation gate + private-import ratchet | 23 passed |
| default gate: TKE + freshwater | 84 passed |
| default gate: recipe + barotropic state | 33 passed |
| default gate: face-mask rank, split in two | 1 passed (494 s) + 1 passed |

Production plants against the G9 file, each reverted and checked with
`git status` (6/6 red, `plants.log`):

1. model wiring off (both the first and the final one-line form): 1 red each;
2. G9 tracer advection back to `"none"`: 1 red;
3. G9 drops the transport mask: 1 red;
4. the reader drops the `&namtra_adv` set: 4 red;
5. the deck-to-case tie removed: 1 red.

Re-anchor: my model insertion first shifted 51 gate citations; it now sits
on its existing line (0 shifted).  Six recipe citations and one ladder
citation shifted mechanically (reanchor tool); none fell inside a changed
hunk.

Gates before the final commit (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH):

| gate | result |
|---|---|
| this receipt's citation gate | `PASS`: 13 citations, 0 failures, 0 unmapped (`receipt_citation_gate.json`) |
| its shift plant on `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283` | exits 1, `FAIL` |
| default citation-gate test + private-import ratchet + G9 file (fast part) | 35 passed (`final_gates.log`) |

## 7. Acquisition (ACQUISITION_NEEDED)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run.sh`,
G8's pattern.  Preflight passes at `d6887bfc1` (`preflight.log`).

- **No build.**  The operator is a namelist switch.  G5's round-31 binaries
  are reused, pinned by SHA-256: writer-free `ORCA2G5_OMIP_L4_R31REF_FIX1`
  (`e8ebf4ae…`), record `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1` (`41cd9a63…`);
  keys `key_RK3 key_qco key_si3 key_vco_1d3d`.
- **Inputs:** G4's (`round21/oracle_g4_r21/p6_a`); no extra file.
- **Namelist diff against G5's deck:** the four `&namtra_adv` lines above.
- **Log checks:** NEMO must print `ln_traadv_OFF = F`, `ln_traadv_fct = T`,
  the other schemes `F`, and `FCT scheme is used. Horizontal order: 2,
  Vertical order: 2`; and still no lateral momentum diffusion, drag or
  momentum advection.
- **Records:** G5's set (per-step surface input, per-step restarts from three
  twins, passivity against the writer-free reference), plus what the record
  binary already writes at kt 1: stage-1 transport operands (NEMO's `vmask`
  and `zFv` on row 147), per-stage transports, and stage 1/2/3 tracer RHS
  before and after `tra_adv` — advection acting on the fold row.
- **Steps:** the writer-free reference runs up to 100 steps to find NEMO's
  boundary; records cover `min(completed, 10)`.  A longer record is not
  expected: G5 stopped beginning kt 11 and G7 at kt 13; only G8, with
  viscosity, ran 100.  G9 adds no momentum damping (prediction P5).

Evidence goes to `gridladder_rounds/round46/oracle_g9_r46/`.
