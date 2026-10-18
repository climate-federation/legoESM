# Lane B — round 16 (vmix_r16): Decision 138 survey; L1 = G5 + ORCA2's bottom boundary layer built

Status: **L1 CERTIFIED** (resume, 2026-10-18; section 5): bit-for-bit with NEMO's record kt 1..10 under both labels,
every field, after three L1-selected NEMO statements. Round start: survey of the four Decision-138 pieces; L1 (G5 +
ORCA2's diffusive bottom boundary layer) built, L1-selected, acquisition defined. Decisions in force: D128-D130,
D136, D138. Codex review: paused (quota) at round start; resume: codex review: operator. Evidence: `gridladder_vmix_rounds/round16/`. Preregistration:
`PREREG_nemo_testcases_gridladder_vmix_l1_round16.md`, committed `63afa503c` before any L1 code.

Commits: `63afa503c` prereg; `ff628988a` L1 card, deck, acquisition, checker, tests, ladder case lists; `5cd5c2bd8`
eight citation pins re-anchored; this receipt and its citation-map entries.

## 1. Survey (every line read in `ORCA2_OMIP_L4`; nothing measured unless said)

**Compiled in.** `trabbc`, `trabbl`, `traldf`, `traldf_iso`, `traldf_lev`, `ldfslp`, `ldftra`, `trazdf`: preprocessed
sources byte-identical in ORCA2_OMIP_L4, G5's writer-free build and V1's record build; `.text`/`.rodata` of every one
of those objects identical in all three, and in G5's record build for `trabbl.o`, `trabbc.o`. G5's record build's
`stprk3_stg.o` and `eosbn2.o` `.text` equal ORCA2_OMIP_L4's (its writer-free build differs only by the absent
writers). All four pieces are run-time switches: nothing needs building.

**Inputs.** Only geothermal heating reads a file: `geothermal_heating.nc` (`sn_qgh`, variable `heatflow`, mW/m2). It
is NOT in the ladder's sixteen-file input set, but it IS in the on-disk `ORCA2_ICE_v5.0.0` tarball (extracted copy and
tarball member both sha `e369762f…`, 116,499 bytes) — the same route V4 took for `zdfiwm_forcing_orca2.nc`. Nothing
needs fetching or vendoring. Isoneutral diffusion (`nn_aht_ijk_t = 20`), the eddy-induced velocity
(`nn_aei_ijk_t = 21`) and the BBL compute their coefficients from the grid and the state: no file.

| piece | ORCA2 deck | NEMO code (ORCA2_OMIP_L4) | base | our model today |
|---|---|---|---|---|
| bottom boundary layer | `ORCA2_OMIP_L4/EXP00/namelist_cfg:268-272` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:527` init; `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:357` coefficients, called at stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:560`); `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:217` trend, after `tra_bbc` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:739`) | G5 (`nn_bbl_adv = 0`: no advection) | NEMO-literal diffusive arm exists (init, gate with NEMO's sign-of-zero, trend association), certified statement-by-statement on ORCA2's own `ahu_bbl` record (Sept); already selected by the full ORCA2 card; reachable on G5's tracer path |
| geothermal heating | `ORCA2_OMIP_L4/EXP00/namelist_cfg:256-263` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:177` init (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:243`: flux times `r1_rho0_rcp` times `1.e-3`); `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:158-159` adds `qgh_trd0/e3t(Kmm)` to the bottom cell's trend at stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:738`) | G5 | a geothermal module exists, but as a post-step update with `Q/(rho0*c_sw*h)` on live thickness, not NEMO's stage-3 trend statement; the NEMO-identity step has no geothermal path. Extend that module with NEMO's statement; read the tarball file |
| isoneutral diffusion | `ORCA2_OMIP_L4/EXP00/namelist_cfg:300-307` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:181` coefficient (mode 20 via `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfc1d_c2d.f90:125`); slopes `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfslp.f90:99` each step (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:189`); operator `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:73` and `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:746` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734`); implicit part in the vertical solve (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:185`) | G5 (acts alone) | NEMO-literal standard operator with MSC, implicit K33, mode-20 metric coefficient exists — the VORTEX SMT-3 card selects exactly ORCA2's block; measured there, not bit-for-bit (SMT rounds 14-24 held), so the largest piece |
| eddy-induced velocity | `ORCA2_OMIP_L4/EXP00/namelist_cfg:317-323` | coefficient `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:707` each step (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:193`); NEMO refuses it without isoneutral diffusion (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:611-612`); enters ONLY through the stage-3 advective transport (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-257`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:892`) | G9 + isoneutral | Treguier coefficient and a NEMO-form bolus-transport option exist; declared unmeasured on the full ORCA2 card |

**Proposed order** (user may redirect):
1. **L1 = G5 + BBL** — transcription exists and is statement-certified; one switch; no file.
2. **L2 = G5 + geothermal** — one new NEMO statement in the stage-3 trend, plus the tarball file.
3. **L3 = G5 + isoneutral diffusion** — biggest (slopes, rotated operator, vertical-solve part).
4. **L4 = eddy-induced velocity** — needs G9 (lane A) and isoneutral diffusion underneath; see the question below.

**Question for the user (not needed before G9 certifies; L1-L3 unaffected).** NEMO will not run the eddy-induced
velocity without isoneutral diffusion, so it cannot be one switch from G9:
1. first a G9 + isoneutral rung (isoneutral re-certified with advection on), then + eddy-induced velocity [my pick:
   keeps one switch per rung];
2. one rung = G9 + isoneutral + eddy-induced velocity.

**Findings, one line each (no action taken).**
- The full ORCA2 card runs no geothermal heating, while ORCA2's deck sets it (`ORCA2_OMIP_L4/EXP00/namelist_cfg:256`);
  it is not in that card's declared-unmeasured list either.
- ORCA2's deck also sets the mixed-layer eddy scheme (`ORCA2_OMIP_L4/EXP00/namelist_cfg:312`); it is on neither lane's
  list.

## 2. L1 built (L1-selected; nothing else moves)

- **Card** `ORCA2L1-bbl` = G5's card with the existing diffusive-BBL arm selected: `bbl_diffusive_option` 0 -> 1,
  `bbl_aht_m2_s` 0 -> 1000, `bbl_gamma_s` 0 -> 10 (ORCA2's `rn_gambbl`, inert under `nn_bbl_adv = 0`),
  `bbl_adv_option` 0. Measured: exactly these three config fields differ from G5's card; initial state identical in
  every field. No model code changed; no new model state.
- **Deck** = G5's plus ORCA2's five `&nambbl` lines, verbatim (only diff; ice deck identical). The reader requires them
  on L1, refuses them on every other ladder rung, and refuses a stray `&nambbl` key; the selector checks config and
  card metadata against ORCA2's values (not against the builder's own helper).
- **Card-side sanity, not a score** (L1's card stepped against G5's admitted NEMO record, kt 1): u, v, sea level and
  both barotropic velocities 0 unequal; surface T differs in 36 cells (max 1.38e-9), surface S in 27 (max 1.56e-8). So
  the card's BBL acts only in the stage-3 tracer update, as NEMO's placement predicts.
- **Purity** (`ff628988a` vs base `f8b546f1a6`): one-step jaxprs of the full ORCA2 (138,746 lines), DINO (123,234),
  G5 (168,423), V1 (174,013), V2 (174,046), V3 (174,349), V4 and V5 (176,180) **IDENTICAL**; L1's (169,583) differs
  from G5's.
- **Tests** (`test_nemo_testcase_l1_orca2bbl.py`): 8 passed — deck diff, ORCA2's own lines and types, reader refusals
  both ways plus a planted line, card = G5 + BBL only, selector refusals (BBL off on L1, metadata disagreeing, BBL on
  G5), footprint geometry, record checker plants, run script. **Plant** (builder's helper returns no BBL for L1): the
  build refuses `['ln_trabbl']`, 2 tests error; restored.
- Ladder: L1 added to the case lists (forced, SI3). No other ladder change.

## 3. ACQUISITION_NEEDED (exact)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_orca2bbl/run.sh --run` from a clean checkout (default is
read-only preflight; measured `L1_PREFLIGHT_OK` at `ff628988a`).
- **Builds nothing.** Record = G5's `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1` (sha `41cd9a63…`); writer-free reference =
  G5's `ORCA2G5_OMIP_L4_R31REF_FIX1` (sha `e8ebf4ae…`). Preflight pins both, checks `trabbl.o`/`trabbc.o` sections and
  `trabbl.f90` in both against ORCA2_OMIP_L4, and the record binary's `stprk3_stg.o`/`eosbn2.o` `.text`.
- **Namelist diff vs G5's deck**: the five `&nambbl` lines (deck sha `d922a0f2…`; preflight also re-derives the diff).
- **Inputs**: G5's sixteen, sha-pinned by the G5 tool. Nothing new.
- **Records**: G5's set — restarts 2..N over twins A/B/C, surface-input and step-entry records each kt; twins must agree
  on both record kinds every kt.
- **What shows BBL acting**: the kt-2 step-entry record (state after kt 1) against G5's admitted one (sha
  `930211d3…`): u, v, sea level bit-identical; T or S different in >= 1 wet cell; every changed column touches a wet
  face whose two bottom levels differ (7,088 U + 6,705 V faces; 11,936 of 16,433 wet columns; the two fold rows
  admitted wholesale). Plant: G5's own record is refused (exit 66, 0 cells moved).
- **Logs**: `ln_trabbl = T`, `nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl = 1000`, `rn_gambbl = 10`, no advective-BBL
  line, `ln_trabbc = F`, G5's `ln_zdfcst = T` and `nn_ice = 2` (regexes checked against NEMO's real BBL print).
- **Steps**: preflight -> 2-step smoke (record binary; acting check) -> 100-step boundary (writer-free) -> three twins
  to min(boundary, 10) -> admission (`L1_ACQUISITION_PASS`). Evidence: `gridladder_vmix_rounds/round16/oracle_l1_r16/`.
- Next round: score L1 both labels on the ladder (P2-P3); add a kt 1-2 record test.

## 4. Gates (CPU, x64, `land.sh` PYTHONPATH)

- **Default gate** (`land.sh` orca2 list, incl. the citation-gate tests and the private-import ratchet) at `d6f3bda3c`:
  **145 passed** (run in three chunks, 92 + 2 + 51, because one test alone takes 484 s).
- **Per-receipt citation gate** on this receipt (`--from-heading` this title): **PASS, failures 0**, 32 citations,
  unmapped 0, map entries failing audit 0; a planted shifted citation exits 1. First runs: 32 unmapped (added) and five
  ORCA2 files unknown to the gate (registered); before that, eight pins shifted by the card edit (re-anchored).
- **L1 + V1-V5 + G2-G5 card files and both G1 card unit files**: **190 passed, 1 failed** (417 s, 24 workers). The
  failure (G1 stand-in expects `no_ice`, the card says `no_internal_ice_model`) fails identically at base
  `f8b546f1a6`: pre-existing, not touched here.
- Acquisition dry checks: `bash -n` clean; `L1_PREFLIGHT_OK`; the acting check refuses G5's own record (exit 66).

**Decision 122.** `trabbl.f90` is byte-identical in G5's builds and ORCA2_OMIP_L4; the L1 statements are the ones the
full ORCA2 card already selects.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| gate `(0.5 - SIGN(0.5, -zgdrho*mgrhu))*ahu_bbl_0` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:430` | `bbl` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:357`) |
| bottom depth `gdept_1d(mbkt)*(1+r3t(Kmm))` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:412` | `bbl` |
| `ahu_bbl_0 = rn_ahtbbl*e2_e1u*e3u_bbl_0*ssumask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:600` | `tra_bbl_init` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:527`) |
| diffusive trend on the bottom cell | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:258` | `tra_bbl_dif` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:217`) |

## Choices made

ASKED: survey; order proposal; L1 = the first G5 piece in that order (BBL), exactly ORCA2's five lines; acquisition
defined, nothing scored; other cards unchanged (D130).
UNASKED, offered for revert:
- `bbl_gamma_s = 10` on L1 (ORCA2's `rn_gambbl`; inert because `nn_bbl_adv = 0`; the full ORCA2 card states 0);
- the acting check's footprint excludes flat faces (relies on the September ORCA2 gate's measurement of NEMO's sign of
  zero; if wrong, the acquisition refuses rather than admits);
- eight citation pins re-anchored by measured offsets (anchors unchanged).

UNVERIFIED:
- L1 not scored. P3 (at the bar or bit-for-bit) is PLAUSIBLE. Read off the code: the card adds the BBL trend in
  content form inside the stage-3 tracer update, and builds the bottom depth from a thickness ratio; NEMO adds it to the
  trend between `tra_bbc` and `tra_dmp` and uses `gdept_1d(mbkt)*(1+r3t(Kmm))`. Either may cost last bits.
- The survey's "our model today" column is read off the code; only the BBL arm has a statement-level certification.

## 5. Resume: L1 scored on its record, three statements fixed, L1 certified

Record: `gridladder_vmix_rounds/round16/oracle_l1_r16` (operator: `L1_PREFLIGHT_OK` at `34ff2492a`, nothing built;
`L1_ACQUISITION_PASS steps=1..10 restarts=2..10`). CPU, x64, `land.sh` PYTHONPATH. Operator decision on section 1's
question: option (1) (isoneutral diffusion its own rung; the eddy-induced velocity later, on G9 + isoneutral).
Evidence: `round16/score/`.

**Where the BBL acts, on the record** (NEMO L1 vs G5's admitted record, kt 1): stages 1 and 2 bit-identical in T, S,
u, v and sea level; stage 3 differs in T (2,655 cells) and S (2,282) only, u/v/sea level 0 unequal; the 913 moved
columns all lie in the footprint (columns touching a wet face whose two bottom levels differ), 0 outside. The checker
refuses G5's own record (exit 1). As the survey said.

**First score (as built): not certified.** kt 1 exact at the surface; kt 2 onward DEBT under both labels. The walk:
kt 1 stage 3 was the first differing state (stages 1-2 exact), T 52 / S 6 cells. Three NEMO statements, each fixed
L1-selected (two new switches, default off; nothing else selects them):

| # | NEMO statement | ORCA2_OMIP_L4 line | SUBROUTINE | card before | cells it cost |
|---|---|---|---|---|---|
| 1 | BBL trend added to Krhs, after the surface terms, before the vertical solve | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:258`, called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:739` | `tra_bbl_dif` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:217`) | added as content to the stage flux divergence | kt 1: 45 bottom T cells, 1-2 units in the last place (S exact) |
| 2 | after the stage-3 update and vertical solve, T/S exchanged on the north fold: the image half of the pivot row takes its mirror | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796` -> `ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:620-627` | `stp_RK3_stg` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:78`); `lbc_nfd_dp` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:561`) | no tracer exchange; the BBL broke the row's symmetry | kt 1: 7 T + 6 S cells in two pivot-row columns |
| 3 | BBL bottom depth `gdept_1d(mbkt)*(1+r3t(Kmm))`, the stored stage-3 ratio | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:412` | `bbl` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:357`) | reference depth times a ratio of thickness sums: equal only while the entry sea level is zero | kt 4 (given NEMO's entry): one face's gate flips, 6 T + 4 S cells |

- (1) and (3) sit behind `bbl_nemo_stage3_krhs`, (2) behind `rk3_fold_exchange_tracer_pivot_row`; the L1 card selects
  both and its selector checks them. Measured: the card's bottom reference depth equals `gdept_1d(mbkt)` in all 16,433
  wet columns; (1) uses the card's existing stage-3 `e3t(Kmm)`, (3) its existing stage-3 `1+r3t(Kmm)`.
- Not a statement change elsewhere: the gate, coefficients and geometry are unchanged.

**Score after the fixes (packages tree `eaabad51`, commit `40b5e4413`):**
- **INDEPENDENT kt 1-10:** 68 rows + 36 all-level rows (u, v, T, S at kt 2-10): every row max 0, unequal 0, dry
  unequal 0. 4,162 comparison entries, none nonzero.
- **GIVEN-NEMO-ENTRY (entries kt 1/3/5/7/9, one- and two-step):** 68 rows + 20 all-level rows: every row max 0.
  2,608 entries, none nonzero.
- D127 census (flushed subnormals) empty: no unequal cell. Classes: every row exact. **L1 CERTIFIED.**
- Stage 3 at kt 1 and kt 4 (from NEMO's entry): T and S 0 unequal at every level.

**Purity** (base `f8b546f1a6` vs the fixed tree): one-step jaxprs of the full ORCA2 (138,746 lines), DINO (123,234),
G5 (168,423), V1 (174,013), V2 (174,046), V3 (174,349), V4 and V5 (176,180) **IDENTICAL**; L1's changes
(169,583 -> 169,643).

**L1 record test** (`test_nemo_testcase_l1_orca2bbl.py`, kt 1-2, skipped when the record is absent; the V-rung
record helper takes `frame=False`, nothing duplicated): control (the BBL moves NEMO's kt-2 T vs G5's record); the
stage check above; every independent-arm field kt 1-2, wet and dry. Model step jitted, state a traced input, x64.
Plus a record-free check of the pivot-row exchange.
- **Plant** (scratch clone; the stage-3 Krhs addition removed): **1 failed**, kt 1 T and S unequal.
- **Restored** (this tree): L1 file **10 passed**, 0 skipped.

Preregistration: P1 confirmed (acquisition); P2 confirmed (stage 3 only, footprint only); **P3 refuted as built**
(three statements, the two suspects it named among them), confirmed after the fixes; P4 confirmed (purity above).

Gates (resume; CPU, x64, `land.sh` PYTHONPATH; commits `40b5e4413` fix + tests, `ebe2eb3e9` this section, then four
map entries and ORCA2's fold file registered, 63 pins re-anchored by the fix's own offsets and two in-span extents
grown, anchors unchanged, `b2a9315ca`):
- **Default gate** (`land.sh` orca2 list, incl. the citation-gate tests and the private-import ratchet) at
  `b2a9315ca`, clean tree: **145 passed** (661 s).
- **Per-receipt citation gate** on this receipt (`--from-heading` this title): **PASS, failures 0**, 36 citations,
  unmapped 0, map entries failing audit 0; a planted shifted citation exits 1.
- **V3 + V4 + V5 card files** (they share the record helper): **23 passed**. L1 file: 10 passed (above).

ASKED (resume): score both labels; confirm where the BBL acts; fix cited, L1-selected statements; L1 record test with
plant; gates.
UNASKED (resume), offered for revert:
- two new config switches (default off) rather than changing the existing BBL arm for every card;
- the pivot-row exchange placed after the vertical solve, before the conservation fixer (NEMO has no fixer);
- `--check-bbl-stages` added to the L1 checker;
- a record-free unit check of the pivot-row exchange.

Findings, one line each (no action taken):
- The card's BBL north neighbour on the pivot row reads the row's own mirror; with a stored pivot row NEMO's ghost is
  the mirror of the row below. Inert on this record (the image half is overwritten; no source-half cell differed).
- The full ORCA2 card selects the diffusive BBL without these three statements.

UNVERIFIED (resume): beyond kt 10 nothing scored; the full ORCA2 card's BBL not re-measured.

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r16/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: L1_CERTIFIED — G5 + ORCA2's diffusive BBL bit-for-bit with NEMO kt 1..10 under both labels (4,162 + 2,608 entries, every one 0, dry included) after three L1-selected NEMO statements (BBL trend into stage-3 Krhs, trabbl.f90:258; BBL depth gdept_1d*(1+r3t(Kmm)), trabbl.f90:412; T/S north-fold exchange of the pivot row, stprk3_stg.f90:795-796); the BBL acts at stage 3 only, T/S only, inside the footprint; every other card's jaxpr identical; L1 record test fails with the BBL wiring removed; codex review: operator.
