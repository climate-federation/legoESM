# Grid ladder rung G7 — round 1 receipt (gridladder_r41)

Status: **STOPPED_FOR_RECORD.** G7 = certified G5 + ORCA2's linear bottom
drag.  Deck, card and acquisition are built.  Nothing is scored, because no
G7 NEMO record exists yet.

- Preregistration: `PREREG_nemo_testcases_gridladder_g7_round1.md`,
  committed `26e330e30` before any card change.
- Build commit `0a81f804f`.  Citation re-anchor `15366f003`.
- Evidence: `gridladder_rounds/round41/` (`manifest.sha256` `3acbebb3ed7929a1`).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

## 1. Survey: G5 does not run ORCA2's bottom drag

| item | value |
|---|---|
| ORCA2 `&namdrg` | `ln_lin = .true.` only (ORCA2_OMIP_L4 `namelist_cfg:249-252`) |
| ORCA2 companions | from `namelist_ref`: `ln_drgimp=T` :817, `ln_drgice_imp=T` :818, `rn_Cd0=1.e-3` :834, `rn_Uc0=0.4` :835, `ln_boost=F` :839 |
| G5 deck and G5 record | `ln_drg_OFF = .true.` (free slip) |
| G5 card | no bottom drag; ice-ocean top drag only |
| reusable transcription | SMT-2 linear-drag bundle (certified on VORTEX); ORCA2 lane OMT-2 uses the same bundle |
| top + bottom together | model refused it before this round |

Searched for: `namdrg`, `ln_lin`, `nemo_linear`, `rCdU_bot`/`rCdU_top`,
`implicit_ice_ocean_top_drag`, `zdf_drag_in_matrix`,
`barotropic_drag_substep`, across `packages/`, `scripts/`, `tests/` and the
receipts.  Found: the SMT-2 bundle, the G5 top-drag face helper, and the
guard.  Nothing combined top and bottom drag.

Premise correction: G5 runs momentum advection OFF, not in flux form.  G7
keeps that setting, so G6's pending question (Decision 132) does not apply.

## 2. NEMO statements (each opened at its SUBROUTINE)

`zdfdrg.f90`, `dynspg_ts.f90` and `dynzdf.f90` are byte-identical between G5's build and ORCA2_OMIP_L4.  The
line numbers below hold in both.

| statement | line | SUBROUTINE |
|---|---|---|
| `rCdU_bot = - pCd0 * rn_Uc0` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:276` | `zdf_drg_lin` (:258) |
| `np_lin`, `pCd0 = rn_Cd0 * zmsk_boost` (ssmask :525) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:540-547` | `drg_init` (:427) |
| combined `pCdU = r1_2*((bot)+(top))` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1431-1432` | `dyn_drg_init` (:1404) |
| bottom residual, then top residual | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1462-1463`, `:1487-1488` | `dyn_drg_init` |
| `dyn_zdf` bottom RHS, then top RHS | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:166-169`, `:175-178` | `dyn_zdf` (:67) |
| `dyn_zdf` bottom diagonal, then top diagonal | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:306`, `:312`, `:473-474`, `:479-480` | `dyn_zdf` (:67) |

## 3. What changed

**Deck.** The G5 deck changes in two lines only:

- `ln_drg_OFF` goes `.true.` -> `.false.`;
- `ln_lin` goes `.false.` -> `.true.`.

A test pins this diff.  Deck SHA-256 is `48a8ea9d…cc28`.  The ice namelist is
G5's, byte for byte.

**Card.** Case `ORCA2G7-drg` is G5's rung tables plus SMT-2's linear-drag
bundle.  I moved that bundle into one helper, used by both SMT-2 and G7;
SMT-2's values are unchanged.  The deck reader now:

- picks the drag switches by `ln_lin`;
- refuses any stated `ln_drgimp`, `ln_drgice_imp`, `&namdrg_bot` or
  `&namdrg_top` line, because the card transcribes `namelist_ref`'s values;
- requires `ln_non_lin` and `ln_loglayer` to be `.false.` on every rung.

The card fails closed: `ORCA2G7_UNMEASURED` lists
`nemo_post_si3_surface_records`, `g7_record_not_admitted` and
`linear_bottom_drag_multilevel_step`.

**Model: the one missing statement.** The external-mode coefficient sums the
bottom and top pairs in one statement.  Before this round, the substep loop
kept only the top coefficient when both drags were set.  Now:

- the top-drag face helper accepts NEMO's bottom T-field,
  `-(rn_Cd0*ssmask)*rn_Uc0`, with `-0` on dry columns, in place of its zero
  bottom pair;
- the substeps read that one combined coefficient;
- the residual and `dyn_zdf` were already in NEMO's bottom-then-top order and
  are unchanged.

The guard now accepts only the full linear bundle.  Partial combinations and
the quadratic law still raise.

**Other cards are unchanged.** G5 and G6 pass no bottom field, so they run
the same expression as before.  Two callers set the new argument: the
barotropic substep site, and the spy in the new test.

**Ladder instrument.** G7 is added to the ice-case lists.  The dyn_zdf replay
arm refuses G7, because it transcribes zero drag.

## 4. Tests and plants (CPU, fp64)

| battery | result |
|---|---|
| new G7 file | 13 passed, 28.5 s |
| G5 top-drag consumers + G5 + G6 + VORTEX SMT card + ORCA2 OMT-2 ladder | 53 passed, 247 s |
| G2-family ladder + G4 (two stale case-list expectations updated: the tables gained G7) | 38 passed, 11 deselected |
| G3 | 9 passed, 2 deselected |
| G1 tools | 31 passed |
| citation gate + private-import ratchet | 23 passed |

The G1 ladder set and slow-marked tests were not run.  G1's card path is
unchanged apart from the reader's two new required-false lines, which every
G deck states.

Production plants, each run against its own test, tree restored and
`cmp`-verified (5/5 red, `plants.log`):

1. substeps pass no bottom field;
2. the helper ignores the bottom pair;
3. the guard accepts any scheme;
4. the card drops the bundle;
5. the reader drops the drag switch set.

Re-anchor: 147 code citations in the gate map and in the GYRE round-8
receipt were shifted mechanically (`reanchor.py`).  None fell inside a
changed hunk.

## 5. Acquisition (ACQUISITION_NEEDED)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g7_orca2drg/run.sh`
follows the G6 pattern.

- **Build to copy from: none.**  The drag law is a namelist switch.  G5's
  round-31 binaries are reused, pinned by SHA-256:
  - writer-free `ORCA2G5_OMIP_L4_R31REF_FIX1` (`e8ebf4ae…`);
  - record `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1` (`41cd9a63…`);
  - keys `key_RK3 key_qco key_si3 key_vco_1d3d`.
- **Inputs:** G4's (`round21/oracle_g4_r21/p6_a`).  Their `namelist_ref` is
  ORCA2's, byte-identical (`b04f2ce4…`).
- **Namelist diff against G5:** the two `&namdrg` lines in section 3.
- **Records:** G5's set.  Per-step `oracle_ocean_surface_input_kt*.bin`
  (including `rCdU_ice`) for kt 1..N, plus restarts at steps 2..N from three
  twins.  `rCdU_bot` is a constant that NEMO prints in its log.
- **Steps:**
  1. 2-step smoke;
  2. writer-free 100-step boundary run;
  3. N = min(completed, 10) rounded down to the forcing cadence;
  4. twins A, B, C;
  5. admission with `g5_records.py admit` (passivity first).
- **Log checks:** `ln_drg_OFF = F`, `ln_lin = T`, `ln_drgimp = T`,
  `ln_drgice_imp = T`, `linear BOTTOM friction (constant coef = Cd0*Uc0 = 4.0000000000000002E-004 )`,
  and the vector form must be absent.  These strings match the ORCA2 lane's
  real log (`oracle_phase2v_tke_boundary_np2`), and G5's log fails them.
- **Evidence root:** `gridladder_rounds/round41/oracle_g7_r41`.
- Preflight ran clean at `0a81f804f`:
  `G7_PREFLIGHT_OK … expected_boundary=kt11_after_10_completed`.

## 6. Open, frozen in the preregistration

When the record exists, P3 predicts that kt 1 is NOT bit-identical.  The most
likely first non-bit statement is the north-fold V-row bottom rate: the
shared bottom face helper has no fold image, while NEMO computes `rCdU_bot`
in the folded halo.  The next candidates are the bottom residual's
arithmetic form, then the diagonal's order on one-level columns.

Decision 122: the full ORCA2 card does not select linear drag.  It still
lists `linear_implicit_bottom_drag` as unmeasured.

## Choices

ASKED: G7's definition (the brief); the record set, which is G5's.

UNASKED, offered for revert:
- the reader now requires `ln_non_lin` and `ln_loglayer` to be `.false.` on
  every G rung (G1-G6 decks already state this);
- the reader refuses stated `ln_drgimp`, `ln_drgice_imp`, `&namdrg_bot` or
  `&namdrg_top` lines;
- the guard now admits the linear top+bottom bundle, and only that one.

UNVERIFIED: any G7 number against NEMO (there is no record); the full ORCA2
card's response.
