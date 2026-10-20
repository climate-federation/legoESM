# Vertical-mixing lane — round 15 (vmix_r15): D136 (V2 re-scored, still bit-for-bit); V5 = V4 + ORCA2's nn_havtb built

Status: **V2 still CERTIFIED** on the step-entry EVD trigger (Decision 136). **V5 CERTIFIED** (resume,
2026-10-18): bit-for-bit with NEMO's record kt 1..10 under both labels, every field, no fix needed (section 6).
Decisions in force: D128-D130, D136. Codex review: paused (quota). Evidence:
`gridladder_vmix_rounds/round15/`. Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v5_round15.md`,
committed `8d0a4fc7a` before any V5 code.

Commits: `b0f3a74af` D136 (V2 on the step-entry trigger); `8d0a4fc7a` V5 prereg; `5951ae22e` V5 card, deck,
acquisition, tests; `3e9e61e24` this receipt and its citation-map entries; `7aa15be50` 11 shifted pins
re-anchored; this update (gate numbers).

## 1. Decision 136: V2 on EVD's step-entry N² trigger

- NEMO: `zdf_evd` fires where `MIN(rn2, rn2b) <= -1e-12`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109`), on the arrays built once on Nbb
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171`, `:174`). V3 and V4 already read them (round 13).
- Change (`b0f3a74af`): every EVD rung (V2-V5) selects the step-entry trigger; the selector pins it. V2's card diff
  vs V1 gains the switch; V3's diff vs V2 loses it. No model code changed.
- **V2 re-scored on its existing round-12 record** (`round12/oracle_v2_r12`, twins A/B/C; packages tree
  `d9226b5f`, clean, CPU x64):
  - INDEPENDENT kt 1-10: 149 rows, every row max 0, unequal 0, dry unequal 0.
  - GIVEN-NEMO-ENTRY (entries kt 1/3/5/7/9): 113 rows, every row max 0, unequal 0, dry unequal 0.
  - 7,501 + 4,463 comparison entries in the two JSONs; none nonzero. Bit-for-bit, as round 13's review measured.
- **Jaxprs**, base `1fc44b9b98a2` vs `b0f3a74af`: full ORCA2 (138,746 lines), DINO (123,234), V1 (174,013),
  V3 (174,349), V4 (176,180) **IDENTICAL**; V2 **differs** (178,423 → 174,046 lines: the stage-3 N² rebuild is gone).
- V2 + V3 card/selector tests: 12 passed. Planted `trigger_step_entry_bn2 = False` now fails V2's and V3's selector.

## 2. V5 survey

- ORCA2 sets `nn_havtb = 1` (`ORCA2_OMIP_L4/EXP00/namelist_cfg:401`); V4's deck has 0 (namelist_ref's value).
- NEMO, `zdf_phy_init` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:126`): `avtb_2d = 1`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:215`), then under `nn_havtb == 1`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:217`) three latitude-band WHERE statements
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:221-223`): 1 → 0.1 → 1 over -15..15 degrees. It seeds
  `avt_k = avtb_2d*avtb*wmask` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:227`), then calls `zdf_iwm_init`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:283`), which resets `avtb_2d = 1`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:441-443`).
- **So on ORCA2's deck the shape reaches one thing**: the kt-1 `avt_k` seed, read by TKE's buoyancy term
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:424`; cold start). The per-step floor
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:710`) reads the reset value 1. Round 14 predicted this.
- **Input files**: none new. The shape reads `gphit` from the domain file already in the ladder's input set
  (`ln_read_cfg`, `ORCA2_OMIP_L4/EXP00/namelist_cfg:33`). No band T point sits exactly on a band edge;
  4,302 wet columns lie in [-15, 15).
- **Compiled in**: `zdfphy.f90` and `zdf_oce.f90` byte-identical in ORCA2_OMIP_L4, V1's record build and G5's
  writer-free build (sha prefixes `903e0d1220a2f9b3`, `875d466f3ae19d35`); `zdfphy.o` `.text`/`.rodata` identical in
  all three; no FMA instruction in `zdfphy.o`; all built with `-fdefault-real-8` (NEMO's bare literals are doubles).
- **No new model state**: `avt_k` is the card's existing carried `tke_avt` field, which the card's TKE already reads
  as its pre-closure diffusivity.
- **Our repo** (searched `havtb|avtb_2d|equatorial` in packages, src, scripts, tests): the full ORCA2 card already
  transcribed the three statements inline. Moved into one helper; the full card and V5 both call it.
- No stop condition: nothing fetched or vendored, no new state.

## 3. V5 built (V5-selected; nothing else moves)

- **Card** `ORCA2V5-havtb` = V4's model config unchanged; initial state = V4's except `tke_avt`
  = `(avtb_2d*rn_avt0)*wmask`, NEMO's association. 113,081 of 414,119 wet W cells move (4,302 columns, all in the
  band; values 1.2e-6 to 1.0993e-5).
- **Deck** = V4's with `nn_havtb = 1` (only diff); ice deck identical. The reader requires 1 on V5 and 0 elsewhere;
  the card validator checks the seed on every V rung (the builder runs it).
- **Purity** (`5951ae22e` vs the D136 tip): full ORCA2, DINO, V1-V4 one-step jaxprs **IDENTICAL**; V5's one-step
  jaxpr **IDENTICAL to V4's** (176,180 lines). Initial states of the full ORCA2 card and V1-V4 vs base
  `1fc44b9b98a2`: all 17 fields equal (the helper refactor moved no bit). V5 vs V4: only `tke_avt` differs.
- **Tests**: V5 file 7 passed (deck diff, ORCA2's own line, reader refusals both ways, the shape against a scalar
  transcription at band edges, card = V4 + seed in band cells only, validator refuses swapped seeds, run script).
  Plant (builder drops the V5 seed): the build refuses `['nn_havtb']`, 2 tests error; restored, tree clean.

## 4. ACQUISITION_NEEDED (exact)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v5_orca2havtb/run.sh --run` from a clean checkout
(default is read-only preflight, `V5_PREFLIGHT_OK`).
- **Builds nothing.** Record binary = V1's (`ORCA2V1_OMIP_L4_R1FULLENTRY`, sha `b2ea4a47…`); writer-free reference =
  G5's (`ORCA2G5_OMIP_L4_R31REF_FIX1`, sha `e8ebf4ae…`). Preflight checks `zdfphy.o` and `zdfiwm.o` sections and
  seven zdf/tracer sources against ORCA2_OMIP_L4.
- **Namelist diff vs V4's deck**: `nn_havtb = 1` only. Deck sha `3358b329bfee1330…`.
- **Inputs**: V4's seventeen (V3's sixteen sha-pinned by the G5 tool + `zdfiwm_forcing_orca2.nc`). Nothing new.
- **Records**: V4's set (restarts at every writable step 2..N over twins A/B/C; surface-input and hm_i frames each
  kt; kt-1 post-`zdf_phy` frame; twins agree). **Shows the shape acting**: the kt-1 frame's `en` must differ from
  V4's admitted frame (sha `18a8a836…`) in ≥ 1 cell, and no field (avm, avt, avs, en) may differ outside
  -15 ≤ gphit < 15. Plants: V4's own frame refuses (0 cells differ); V3's frame refuses (301,038 cells outside the
  band). V4's IWM-vs-V3 and V3's DDM frame checks still run.
- **Logs** must show `nn_havtb = 1` plus every V4 line (IWM banner and forced floors, DDM, EVD, TKE).
- **Steps**: preflight → 2-step smoke (record binary; frame check) → 100-step boundary (writer-free) → three twins
  to min(boundary, 10) → admission. Evidence: `gridladder_vmix_rounds/round15/oracle_v5_r15/`.
- Next round: add V5 to the ladder's case lists, score per the prereg (P1-P3).

## 5. Gates (CPU, x64, `land.sh` PYTHONPATH)

- **Default gate** (`land.sh` list, incl. the citation-gate tests and the private-import ratchet) at `7aa15be50`:
  **143 passed** (1,110 s).
- **Per-receipt citation gate** on this receipt (`--from-heading` this title): **PASS, failures 0**, 16 citations,
  unmapped 0, map entries failing audit 0. First runs: 7 unmapped (added), then 11 recipe pins shifted by this
  round's edits (re-anchored by mechanical offsets from this round's own hunks, `7aa15be50`; no anchor text changed;
  the default receipt's citations follow).
- **V1-V5 card files (incl. the V2 kt 1-2, V3 kt 1-3 and V4 kt 1-2 record tests) + G1 card unit file**:
  **93 passed, 1 failed** (788 s). The failure (G1 stand-in card expects `no_ice`, the card says
  `no_internal_ice_model`) fails identically at base `1fc44b9b98a2`: pre-existing, not touched here.
- Acquisition dry checks: `bash -n` clean; `--check-havtb-frame` refuses V4's and V3's frames (exit 66).

**Decision 122.** `zdfphy.f90` is byte-identical in V1's build and ORCA2_OMIP_L4.

| statement | ORCA2_OMIP_L4 line | V1 build line | SUBROUTINE |
|---|---|---|---|
| `IF( nn_havtb == 1 ) THEN` + three WHEREs | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:217` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfphy.f90:217` | `zdf_phy_init` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:126`) |
| `avt_k = avtb_2d * avtb * wmask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:227` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfphy.f90:227` | `zdf_phy_init` |

## Choices made

ASKED: D136 (V2 → step-entry trigger, re-score on the existing record); V5 = ORCA2's `nn_havtb` exactly as its deck
sets it; survey; card; acquisition; off on other cards (D130).
UNASKED, offered for revert:
- the full ORCA2 card's three statements moved into the shared helper (bit-identical state, identical jaxpr);
- the card validator now checks the `avt_k` seed on every V rung (passes on V1-V4 as built);
- the acquisition adds "kt-1 frame differs from V4's in en, only inside the band" as an admission condition;
- 11 recipe citation pins (and the default receipt's matching citations) re-anchored to keep the gate green.

UNVERIFIED:
- V5 not scored (no record). P3 (bit-for-bit as built) is PLAUSIBLE only.
- Beyond kt 1, NEMO's `avt_k` seed is never read again on this deck (cold start, overwritten at kt 1); read off
  the code, not measured.

## 6. Resume: V5 scored on its record

Record: `gridladder_vmix_rounds/round15/oracle_v5_r15` (operator: `V5_PREFLIGHT_OK` at `9af53c16f`, nothing built;
`V5_ACQUISITION_PASS steps=1..10 restarts=2..10`). CPU, x64, `land.sh` PYTHONPATH; packages tree `3ef41cfc`
(= this round's tip; the ladder edit is scripts-only). Commit `4363c96d8`: ladder takes V5; V5 record test.

**Score (no model change).**
- **INDEPENDENT kt 1-10:** 149 rows (surface sea level, barotropic transports, u, v, T, S each kt; all-level u, v,
  T, S, en, avm_k, avt_k, dissl and surface avm at kt 2-10): every row max 0, unequal 0, dry unequal 0.
- **GIVEN-NEMO-ENTRY (entries kt 1/3/5/7/9, one- and two-step):** 113 rows, every row max 0, unequal 0.
- **kt-1 post-`zdf_phy` frame:** consumed avt, avs, avm and the card's en after `zdf_tke`, each 0 unequal of
  414,119 wet W cells, 0 dry.
- 12,723 + 7,633 + 604 comparison entries in the three JSONs; none nonzero. D127 census empty (no unequal cell).
  Classes: every row exact.

**Where nn_havtb acts, on the record** (kt-1 frame, V5 vs V4's certified frame):

| field | NEMO cells moved | card cells moved | outside -15 <= gphit < 15 | same cells |
|---|---|---|---|---|
| en | 7,595 | 7,595 | 0 / 0 | yes |
| avt | 7,567 | 7,567 | 0 / 0 | yes |
| avs | 7,567 | 7,567 | 0 / 0 | yes |
| avm | 7,529 | 7,529 | 0 / 0 | yes |

- Only through the kt-1 `avt_k` seed: its first reader is TKE's buoyancy term
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:424`); the per-step floor reads `avtb_2d` after
  `zdf_iwm_init`'s reset (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:443`). Every moved cell is in the band, as the
  survey predicted; the card's TKE takes the same seed and the score above is exact, so nothing else moved.

Preregistration:
- **P1 CONFIRMED** (acquisition passed; log checks in the acquisition).
- **P2 CONFIRMED**: en moved in 7,595 wet W cells, none outside the band (table).
- **P3 CONFIRMED**: bit-for-bit kt 1..10, both labels, no further change.
- P4 confirmed in section 3 (no model code changed since).

**V5 record test** (`tests/ocean/fidelity/test_nemo_testcase_v5_orca2havtb.py`, kt 1-2, skipped when the record is
absent; extends V4's through the shared helper): control (seed moves NEMO's kt-2 T vs V4's record); kt-1 frame avt,
avs, avm, en wet and dry; every independent-arm field kt 1-2; shape acts in the same cells both sides, none outside
the band. Model step jitted, state (incl. the seed) a traced input, x64.
- **Plant** (scratch clone; nn_havtb seed removed from builder and validator, deck reader kept): **1 failed**,
  avt unequal in 7,567 wet W cells. Evidence: `round15/resume/record_plant.log`.
- **Restored** (this tree): V3 + V4 + V5 card files **23 passed**, 0 skipped (1,081 s; tests as committed in `4363c96d8`). Evidence: `round15/resume/record_restored.log`.

Gates (resume; commits `4363c96d8` ladder + test, `e08d7a28c` receipt, then one map entry and one ladder pin re-anchored, +32 lines, anchors unchanged, `39e14c256`; this update fills numbers):
- Default gate (`land.sh` orca2 list, incl. citation-gate tests and private-import ratchet) at `39e14c256`: **145 passed** (1,158 s).
- Per-receipt citation gate on this receipt (`--from-heading` this title): **PASS, failures 0**, 17 citations, unmapped 0, map entries failing audit 0 (first runs: one IWM-reset citation unmapped; the ladder pin above shifted).

ASKED (resume): score V5 both labels; confirm where nn_havtb acts; V5 record test with plant.
UNASKED (resume), offered for revert:
- the shared record helper asserts every kt-1 frame row given (V3/V4: the same three rows as before);
- the ladder's kt-1 frame arm adds V5's en row and the band census (V5 only);
- one ladder citation pin re-anchored (+32 lines) to keep the gate green.

UNVERIFIED (resume): beyond kt 10 nothing scored; avs beyond kt 1 certified through T and S only.

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r15/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: V5_CERTIFIED — V4 + ORCA2's nn_havtb = 1 bit-for-bit with NEMO kt 1..10 under both labels (149 + 113 rows, every row 0; kt-1 frame avt/avs/avm/en 0 unequal, wet and dry), no fix needed; nn_havtb acts only through the kt-1 avt_k seed, 7,529-7,595 cells, all inside -15 <= gphit < 15, same cells on the card; V5 record test added (fails with the seed removed: avt 7,567 cells); codex review: paused (quota).
