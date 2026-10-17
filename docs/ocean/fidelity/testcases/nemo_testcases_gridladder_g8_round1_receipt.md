# Grid ladder rung G8 — round 1 receipt (gridladder_r45)

Status: **STOPPED_FOR_RECORD.** G8 = certified G5 + ORCA2's lateral momentum
diffusion.  Deck, card and acquisition are built.  Nothing is scored against
NEMO, because no G8 NEMO record exists yet.

- Preregistration: `PREREG_nemo_testcases_gridladder_g8_round1.md`,
  committed `2fcbad6c0` before any card change.
- Build commit `f3d177511`.  Citation re-anchor and rung-list expectations
  `ac07c3de4`.  Receipt citation-map entries `a3a45055a`.
- Evidence: `gridladder_rounds/round45/` (`manifest.sha256` `ce99b173c661c72b`).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

## 1. Survey: G5 does not run ORCA2's lateral momentum diffusion

| item | value |
|---|---|
| ORCA2 `&namdyn_ldf` | `ln_dynldf_lap = T`, `ln_dynldf_lev = T`, `nn_ahm_ijk_t = -30` (`ORCA2_OMIP_L4/EXP00/namelist_cfg:371-373`) |
| ORCA2 companions | from `namelist_ref`: `ln_dynldf_OFF=F` :1107, `nn_dynldf_typ=0` (div-rot) :1108; `rn_Uv`, `rn_Lv`, `rn_ahm_b` read but unused under -30 |
| G5 deck and G5 record | `ln_dynldf_OFF = .true.` (no lateral momentum diffusion) |
| G5 card | the ORCA2 identity's viscosity selectors, switched off by `A_h = 0`; no coefficient carried |
| reusable transcription | the ORCA2 card's file reader for the coefficient; the shared stage program's `dyn_ldf` at stage 1 and stage 3; ORCA2 lane OMT-3's edge (`A_h` 0 -> 1e5, same four selectors) |

Searched for: `namdyn_ldf`, `dynldf`, `ldf_dyn`, `ahmt`/`ahmf`,
`eddy_viscosity`, `nemo_ahm_3d_file`, `lateral_viscosity_coefficient_source`,
`spatial_lateral_viscosity`, across `packages/`, `scripts/`, `tests/` and the
receipts.  Found all of the above; nothing new had to be written for physics.

Premise correction: G5 runs momentum advection OFF, not in flux form.  G8
keeps G5's setting (OFF), and drag stays OFF.

**ORCA2 card, report only.**  Its lateral-momentum selection already matches
the namelist: harmonic div-curl level operator, coefficient read from the
file, free slip with the boundary condition inside the masked coefficient.
`A_h = 1e5` is an inert selector.  The card still lists
`spatial_lateral_viscosity` as unmeasured.  No lateral-momentum setting on it
is GYRE-inherited.

## 2. NEMO statements (each opened at its SUBROUTINE)

`ldfdyn.f90`, `dynldf.f90` and `dynldf_lev.f90` are byte-identical across
the G5 record build, its writer-free twin and ORCA2_OMIP_L4.  The record
build's call lines equal ORCA2's, so these lines hold in both.

| statement | line | SUBROUTINE |
|---|---|---|
| `NAMELIST/namdyn_ldf/` read | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:177-179` | `ldf_dyn_init` (:149) |
| lap + lev -> `nldf_dyn = np_lap` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:248-250` | `ldf_dyn_init` |
| `-30`: `ahmt_3d`/`ahmf_3d` read, `'T'`/`'F'` exchange | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:348-353` | `ldf_dyn_init` |
| laplacian mask, levels 1..jpkm1 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:387-390` | `ldf_dyn_init` |
| `np_lap` -> `dynldf_lev_lap` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf.f90:83-85` | `dyn_ldf` (:60) |
| operator: `zwf`, `zwt`, U/V update | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:121-140` | `dynldf_lev_lap` (:64) |
| call 1, `dyn_ldf(kt, Kbb, Kbb, ...)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:144` | `stp_2D` (:63) |
| call 2, stage 3, `dyn_ldf(kstp, Kbb, Kmm, ...)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:493` | `stp_RK3_stg` (:78) |

## 3. What changed

**Deck.**  G5's deck changes in four `&namdyn_ldf` lines only:

- `ln_dynldf_OFF` goes `.true.` -> `.false.`;
- `ln_dynldf_lap` goes `.false.` -> `.true.`;
- `ln_dynldf_lev = .true.` and `nn_ahm_ijk_t = -30` are added.

A test pins this diff and checks that every `&namdyn_ldf` key resolves to
ORCA2's value.  Deck SHA-256 is `cc531ae1…02cb`.  The ice namelist is G5's,
byte for byte.

**Card.**  Case `ORCA2G8-ldf` uses G5's rung tables.  It changes two things:

- it keeps the ORCA2 identity's viscosity selection (`A_h = 1e5`) instead of
  zeroing it.  This is OMT-3's edge exactly;
- it reads the coefficient with the ORCA2 card's own reader, on the G
  builder's masks, and carries it on the vertical coordinate.

No model code changed.  The deck reader now picks the `&namdyn_ldf` switch set
by `ln_dynldf_lap`.  It refuses any stated line that ORCA2 leaves to
`namelist_ref`.  The card fails closed: `ORCA2G8_UNMEASURED` lists
`nemo_post_si3_surface_records`, `g8_record_not_admitted` and
`lateral_momentum_diffusion_multilevel_step`.

**Other cards are unchanged.**  G2–G7 cards were built at the base commit and
at this commit.  Their pickled configs, coordinates, grids and initial states
are byte-identical: SHA-256 `7c75962d…36be` on both sides
(`other_cards_unchanged.txt`).  The only package file changed is the recipe,
so identical card inputs mean identical traced programs.  G1 and G1b were not
rebuilt; their builder path only adds a table lookup that is false for them.

**Ladder instrument.**  G8 is added to the ice-case lists.

## 4. Card-side check (card against card, not a NEMO score)

I stepped G8 and G5 independently on G5's recorded surface input
(`card_g8_vs_g5_kt1_kt2.txt`):

- after kt 1: 0 differing cells in every field.  From rest, both diffusion
  calls act on a zero velocity.  This is the card's side of prediction P3;
- after kt 2: the diffusion engages.  Over 0.41 M of 0.80 M U cells differ,
  with a median difference of 5.8e-5 m/s and a maximum of 0.33 m/s;
- the maximum is at surface cell (j, i) = (115, 144).  There G5's U grows
  from 0.25 to 0.58 m/s and G8 holds it at 0.255 m/s;
- every field is finite.  Each step takes 78–85 s on CPU.

## 5. Tests and plants (CPU, fp64)

| battery | result |
|---|---|
| new G8 file | 12 passed |
| G2 ladder, G4, citation gate, private-import ratchet, G8 (after re-anchor) | 75 passed, 11 deselected |
| default gate, part 1 (citation gate, ratchet, TKE, recipe, freshwater) | 132 passed |
| default gate, part 2 (face-mask rank, barotropic state) | 10 passed, 529 s |
| G3, G5, G6, G7 card files | 58 passed |

The first default-gate run hit its 590 s cap inside the slow face-mask file
(`default_gate.log`).  It was re-run in the two parts above.

Production plants, each run against the G8 file, tree restored and checked
with `git status` (5/5 red, `plants.log`):

1. G8 zeroes `A_h` (1 red);
2. G8 drops the coefficient (2 red);
3. the reader drops the `&namdyn_ldf` set (5 red);
4. the rung table selects nothing (3 red);
5. the reader accepts lines ORCA2 leaves unstated (2 red).

Re-anchor: 7 code citations in the gate map shifted mechanically.  None fell
inside a changed hunk.

Gates before the final commit (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH):

| gate | result |
|---|---|
| this receipt's citation gate | `PASS`: 9 citations, 0 failures, 0 unmapped (`receipt_citation_gate.json`) |
| its shift plant on the stage-1 call line | exits 1, `FAIL` |
| default citation-gate test + private-import ratchet + G8 file | 35 passed (`final_gates.log`) |

## 6. Acquisition (ACQUISITION_NEEDED)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g8_orca2ldf/run.sh`
follows the G7 pattern.

- **Build to copy from: none.**  The operator is a namelist switch.  G5's
  round-31 binaries are reused, pinned by SHA-256:
  - writer-free `ORCA2G5_OMIP_L4_R31REF_FIX1` (`e8ebf4ae…`);
  - record `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1` (`41cd9a63…`);
  - keys `key_RK3 key_qco key_si3 key_vco_1d3d`.
- **Inputs:** G4's (`round21/oracle_g4_r21/p6_a`), plus ORCA2's shipped
  `eddy_viscosity_3D.nc` from `nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0`.
  Its SHA-256 is `fc142ce0…99cb`, the same digest the ORCA2 lane's
  coefficient gate pins.  NEMO opens it by that fixed name.
- **Namelist diff against G5:** the four `&namdyn_ldf` lines in section 3.
- **Records:** G5's set: per-step `oracle_ocean_surface_input_kt*.bin` for
  kt 1..N, plus restarts at steps 2..N from three twins.
- **Steps:**
  1. 2-step smoke;
  2. writer-free 100-step boundary run;
  3. N = min(completed, 10) rounded down to the forcing cadence;
  4. twins A, B, C;
  5. admission with `g5_records.py admit` (passivity first).
- **Log checks:** `ln_dynldf_OFF = F`, `nn_dynldf_typ = 0`,
  `ln_dynldf_lap = T`, `ln_dynldf_blp = F`, `ln_dynldf_lev = T`,
  `nn_ahm_ijk_t = -30`, `iso-level laplacian operator`, `read in
  eddy_viscosity_3D.nc file`; plus drag still off and the vector form absent.
  The diffusion strings match the ORCA2 lane's OMT-3 smoke log, and G5's log
  fails them.  The drag-off check passes on G5's log.
- **Evidence root:** `gridladder_rounds/round45/oracle_g8_r45`.
- Preflight ran clean at `f3d177511`:
  `G8_PREFLIGHT_OK … expected_boundary=kt11_after_10_completed`.

## 7. Open, frozen in the preregistration

P3 predicts NEMO's kt 1 equals G5's behaviour, because diffusion acts on the
rest state.  P4 predicts kt 2 is NOT bit-identical.  The most likely first
non-bit statement is the north-fold V row: NEMO's V update on the last owned
row reads the divergence term on the fold halo row, while the shared
operator pads it with zero.  The next candidates are stage 1's divisor slot
(`Kbb` passed as `Kmm`), then the thickness association on partial cells.
The viscous number reaches 0.390 at most (below 0.5).

Decision 122: the full ORCA2 card's selection is unchanged.  It still lists
`spatial_lateral_viscosity` as unmeasured.

## Choices

ASKED: G8's definition (the brief); ORCA2's three lines; G5's binaries and
record set.

UNASKED, offered for revert:
- G8's deck keeps G5's `ln_dynldf_blp = .false.` line (equal to the reference);
- the reader refuses stated `&namdyn_ldf` lines that ORCA2 leaves to
  `namelist_ref`, and refuses `ln_dynldf_lev`/`nn_ahm_ijk_t` on non-G8 decks;
- `A_h = 1e5` is the inert selector value (the ORCA2 identity's and OMT-3's).

UNVERIFIED: any G8 number against NEMO (there is no record); the full ORCA2
card's response.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r45/repo
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r45/repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_g8_orca2ldf/run.sh
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — G8 deck, card and acquisition are built (other cards byte-identical, 5/5 plants red); codex review: paused (quota).
