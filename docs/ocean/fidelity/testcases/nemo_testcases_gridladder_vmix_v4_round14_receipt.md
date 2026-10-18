# Vertical-mixing lane — round 14 (vmix_r14): V4 = V3 + ORCA2's internal-wave mixing CERTIFIED

Status: **V4 CERTIFIED** (resume, 2026-10-17). V4 = certified V3 + ORCA2's internal-wave mixing exactly as ORCA2 sets
it (`ORCA2_OMIP_L4/EXP00/namelist_cfg:396` `ln_zdfiwm = .true.`; `&namzdf_iwm` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:421-435`). After one association fix, every field, every level, kt 1..10,
bit-for-bit with NEMO's admitted record under both labels; worst row under each label: 0 (exact). Sections 1-4 are
the pre-record half; section 5 scores V4.

Decisions in force: D128-D130. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round14/`.
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v4_round14.md`, committed `4eb17a2a0` before any V4 code.

Commits: `4eb17a2a0` prereg; `d62474cd0` statement, card, deck, unit test; `e5ef6981e` acquisition and card tests;
`cf5a75855` citation map (69 pins re-anchored by mechanical offsets from this round's own hunks; no anchor text
changed; the default receipt's citations follow); `1e38cc6ba` the tsdiff factor 5 named at module level (inline-
coefficient ratchet; same value, same association); `00cfb3e1c`, `e8b3d72cb` this receipt and its 10 map entries;
this update (final numbers).

Own note, stated first: this round was interrupted by the account's weekly limit after `1e38cc6ba`; the resume
re-ran every gate at the final tip rather than reuse logs taken before that commit.

## 1. Survey

NEMO (all routines in `ORCA2_OMIP_L4/BLD/ppsrc/nemo`):
- `zdf_iwm` is `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:140`, called at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:372` after `zdf_ddm` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363`).
  `zdf_iwm_init` is `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:373`, called at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:283`.
- **Input file**: `zdfiwm_forcing_orca2.nc` (six maps: `power_bot`, `power_cri`, `power_nsq`, `power_sho`,
  `scale_bot`, `scale_cri`), read once (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:476`). Not in the ladder's input
  set (`round21/oracle_g4_r21/p6_a`), but in the official `ORCA2_ICE_v5.0.0` tarball already on disk, sha256
  `f4279899be77119f…`; every p6_a input checked is byte-identical to that tarball, and every ORCA2_OMIP_L4 run of the
  campaign reads this exact file. **Nothing fetched, nothing vendored**: the acquisition copies it, pinned by sha256.
- **Compiled in**: `zdfiwm.f90` byte-identical (sha256 `13ba29ab…`) in ORCA2_OMIP_L4, V1's record build and G5's
  writer-free build; `zdfiwm.o` `.text`/`.rodata` identical in all three (only the source-path string section
  differs). Nothing to build.
- **No new model state**: the six maps are static 2-D inputs; everything else `zdf_iwm` reads (step-entry `rn2`,
  `r3t`, `ht_0`, 1-D depth ladders, `wmask`) is already on the card.
- **NEMO's own coupling** (D129): `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-838` forces `rn_emin = 1e-10`,
  `rmxl_min = 1e-3`; `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:441-443` resets `avmb = 1.4e-6`, `avtb = 1e-10`,
  `avtb_2d = 1`.
- **Finding (no action)**: that reset means ORCA2's `nn_havtb = 1` shapes only the `avt_k` seed
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:227`), not the per-step floor
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:710`). A later `nn_havtb` rung would be nearly inert on ORCA2's deck.
- Our repo (searched `zdfiwm|zdf_iwm|ln_zdfiwm|iwm|internal.wave|de.lavergne|power_cri|zdfiwm_forcing` in packages,
  src, scripts, tests): one IWM module (a generic port), one forcing loader, the full ORCA2 card's reader. Extended
  all three; no second module.

## 2. V4 built (V4-selected; nothing else moves)

- **Statement**: `nemo_zdf_iwm` in the existing IWM module, selected by `IWMConfig.form = "nemo_literal"` (default
  `"generic"`, unchanged), transcribing `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:233-316` in source order: depths
  stretched by `(1+r3t)`, sequential level sums, raw N² in the buoyancy Reynolds number, constant efficiency,
  T/S-differential ratio, clamp, then adds onto avt, avs and avm. Every operation rounded; EXP/TANH/LOG10 through the
  scalar-libm policy (refuses native).
- **Place**: shared K-profile builder, after the literal DDM (NEMO's order). Stretch factor rides the step-entry bn2
  bundle.
- **Card** `ORCA2V4-iwm` = V3 plus the IWM selection, the six maps from the deck root, and the TKE coupling above
  (`en` seed `1e-10` on wet W points; `avt_k`/`avm_k` seeds unchanged).
- **Deck** = V3's plus ORCA2's `ln_zdfiwm` line and `&namzdf_iwm` group verbatim (only diff); ice deck identical.

## 3. ACQUISITION_NEEDED (exact)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v4_orca2iwm/run.sh --run` from a clean checkout (default
is read-only preflight, `V4_PREFLIGHT_OK`).
- **Builds nothing.** Record binary = V1's (`ORCA2V1_OMIP_L4_R1FULLENTRY`, sha `b2ea4a47…`); writer-free
  reference = G5's (`ORCA2G5_OMIP_L4_R31REF_FIX1`, sha `e8ebf4ae…`). Preflight checks both against ORCA2_OMIP_L4
  (`zdfiwm.o` sections, seven zdf/tracer sources, `zdfddm.o`).
- **Namelist diff vs V3's deck**: `ln_zdfiwm = .true.` plus the `&namzdf_iwm` group (`ln_mevar = F`,
  `ln_tsdiff = T`, `cn_dir = './'`, six `sn_*` lines). Deck sha `8c3a8013d4b5ed40…`.
- **Inputs**: V3's sixteen (sha-pinned by the G5 tool), plus `zdfiwm_forcing_orca2.nc` (sha `f4279899be77119f…`).
- **Records**: V3's set (restarts every writable step 2..N over twins A/B/C; surface-input and hm_i frames each kt;
  twins agree) plus the kt-1 post-`zdf_phy` frame
  (`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:186`), which **shows IWM acting**: its avt must differ
  from V3's admitted frame in ≥ 1 cell, and still pass V3's avs ≠ avt check.
- **Logs** must show `ln_zdfiwm = T`, `ln_mevar = F`, `ln_tsdiff = T`, the `zdf_iwm_init` banner, four
  `Dissipation … TW` lines, the forced `rn_emin`/`rmxl_min` line, and V3's DDM/EVD/TKE lines.
- **Steps**: preflight → 2-step smoke (record binary; frame must show IWM) → 100-step boundary run (writer-free)
  → three twins to min(boundary, 10) → admission. Evidence: `gridladder_vmix_rounds/round14/oracle_v4_r14/`.
- Next round: add V4 to the ladder's case lists, then score per the prereg (P1-P3).

## 4. Tests and gates (CPU, fp64, `land.sh` PYTHONPATH, final tip `1e38cc6ba` unless stated)

- **Statement vs an independent scalar transcription** (Python floats, glibc libm), eager and jit: **bit-for-bit**
  (4 passed). Plants, each fails ≥ 1 arm: tree-reduced sum; native TANH; divide by ρ₀ reassociated; ratio
  reassociated (`unit_plants.log`).
- Card/deck/selector/run-script tests (V4) + V3, V2 card files + IWM unit files: **35 passed** (333 s).
- Frame plant: V4's frame check on V3's record **refuses** (exit 66, `unequal_avt_vs_v3=0 of 825840`).
- **Flag-off jaxprs**, base `0b9443cef279` vs `e5ef6981e` (clean): full ORCA2 138,746 lines and DINO 123,234 lines
  **IDENTICAL**; V2 one-step 178,423 lines and V3 one-step 174,349 lines **IDENTICAL**. V4 one step: 176,175 lines,
  libm callbacks 137 → 142. `1e38cc6ba` changes one literal inside the V4-only statement; V4's jaxpr re-traced at
  the final tip is **IDENTICAL** to the `e5ef6981e` one. Structure only; forcing frames borrowed from V3's record.
- Ratchets (dispatch, inline coeffs, constants, param specs, MPAS IWM): tip and base **5 failed** each, the same 5
  (land slab-land guard, land restart coefficient, JRA55 test constant, shortwave spec, MPAS IWM splice); none in
  this diff. The new form guard sits in a function already in the dispatch baseline.
- **Default gate** (`land.sh` list, at `1e38cc6ba`): **143 passed** (1,129 s). After this receipt's map entries:
  citation-gate tests + private-import ratchet **23 passed**.
- **Per-receipt citation gate** on this receipt (`--from-heading` this title): **PASS, failures 0**, 14 citations,
  unmapped 0, map entries failing audit 0 (first run: 4 range/anchor pins failed; fixed by pinning unique endpoints).
- Acquisition preflight at `e5ef6981e`: `V4_PREFLIGHT_OK` (nothing built).

## 5. V4 scored (resume after the record)

Record: `gridladder_vmix_rounds/round14/oracle_v4_r14` (operator: `V4_PREFLIGHT_OK` at `f7945d134`, nothing built;
`V4_ACQUISITION_PASS steps=1..10 restarts=2..10`; `V4_IWM_FRAME unequal_avt_vs_v3=414119 of 825840`). CPU, x64,
`land.sh` PYTHONPATH. Commits: `4315dc485` ladder; `6ff844d6d` hbot division fix; `b2500fd0b` this receipt; `9ad669e78` map entries for its citations; `9e7c93520`
one ladder pin re-anchored (+15 lines, anchor unchanged); this update (gate numbers).

**First run: the ladder refused.** It built the model without the card's forcing maps, and the literal statement
refuses to run without them. Fixed in the ladder only: every model it builds now takes the card's maps (none on
V1-V3, so their runs are unchanged).

**First score, V4 as built (`f7945d134` model code): not at the bar.**
- kt 1: the avt/avs/avm the solves consume vs NEMO's post-`zdf_phy` frame: 41,546 / 41,825 / 31,459 wet W cells
  unequal (max 1.6e-16 / 1.6e-16 / 1.7e-16 absolute; normalised 1.6e-18 / 1.6e-18 / 1.1e-16); T and S unequal in
  41 / 36 cells.
- First over-bar row: kt 2, both labels (sea level 4.8e-15, u 2.0e-14, T 3.7e-12, S 1.0e-11 absolute; DEBT).

**Walk, kt 1, `zdf_iwm` first** (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:140`;
`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfiwm.f90:140`; byte-identical, sha prefix `13ba29ab568a4592`).
- Pre-IWM K = the V4 card's own consumed K at kt 1 with IWM switched off (IWM is the last statement of `zdf_phy`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:372`).
- NEMO's statements in scalar doubles with glibc `exp`/`tanh`/`log10`, on the card's operands: equal to NEMO's
  frame in **every** wet W cell, avt, avs and avm (0 unequal of 414,119). So the operands are NEMO's:
  - column depth: the card's equals NEMO's `ht_0`, the level-ordered sum of `e3t_0*tmask`
    (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:198`, in `dom_init` at
    `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:128`), in every wet column (P3's first suspect: not it);
  - the 1-D depth ladders equal NEMO's mesh record's; the stored `hcri = 1/scale_cri`
    (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:485`) agrees.
- The card's statement alone equals the scalar transcription eagerly and under jit with the maps as arguments
  (0 unequal). Under jit with the maps as closure constants, as the model step builds it, it reproduces the card's
  41,546 cells exactly.
- Owner: the two divisions by `hbot_iwm`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:267-268`, the same lines in
  `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfiwm.f90:267-268`). With the map a constant, XLA rewrote both
  divisions as a multiply by a folded reciprocal (compiled HLO: 8 divides; 10 after the fix).
- Fix (statement used by V4 alone): `hbot` passes through the existing source-rounding guard, keeping a true
  division (round 13's DDM fix, same class). After it: kt-1 avt/avs/avm 0 unequal; the constant-maps jit 0.

**Final score (`6ff844d6d` model code; packages tree `8c365061`).**
- **INDEPENDENT kt 1-10:** 149 rows (surface sea level, barotropic transports, u, v, T, S each kt; all-level u, v, T,
  S, en, avm_k, avt_k, dissl and surface avm at kt 2-10): every row max 0, unequal 0.
- **GIVEN-NEMO-ENTRY (entries kt 1/3/5/7/9, one- and two-step):** 113 rows, every row max 0, unequal 0.
- **kt-1 frame, consumed coefficients:** avt, avs, avm each 0 unequal of 414,119 wet W cells, 0 dry.
- No flushed-subnormal cells (D127 census empty: no unequal cell). Bar 1e-15 normalised; nothing needed it.

**Internal-wave mixing is active.**
- kt-1 frame vs V3's certified frame: NEMO's V4 avt, avs and avm differ in all 414,119 wet W cells; the card's in
  the same 414,119 cells (same cell set, each field).
- avs > avt at 41,612 cells and avs < avt at 372,507, NEMO and card alike (V3: 145,986 / 266,311).
- NEMO's V4 vs V3 restarts, unequal cells: T and S 430,552 at kt 2, 5, 10; u 413,030; en 414,119 (kt 2) →
  430,483 (kt 10); avt_k 414,122 → 430,544.

Preregistration:
- **P1 CONFIRMED** (acquisition passed; switches and banners per the acquisition's log checks).
- **P2 CONFIRMED**: (a) 414,119 kt-1 avt cells differ from V3's frame; (b) 430,552 T cells differ at kt 2.
- **P3 REFUTED as listed**: none of the four named owners. The owner was the `hbot_iwm` divisions
  (section 5's walk), an XLA rewrite, not a transcription error.
- **P4 CONFIRMED** at `6ff844d6d`: full ORCA2 (138,746 lines) and DINO (123,234) one-step jaxprs identical to base
  `0b9443cef279`; V2 (178,423) and V3 (174,349) one-step jaxprs identical too. V4 one step: 176,180 lines (was
  176,175; the guard).

Tests (CPU, x64):
- Statement test gains a jit case with the maps as constants: 5 passed. Fix removed: **1 failed** (that case),
  4 passed. Tree verified clean after.
- V4 card/deck/selector tests + IWM unit file: **11 passed** (at `6ff844d6d`).

**Decision 122.** `zdfiwm.f90` is byte-identical in V1's build and ORCA2_OMIP_L4.

| statement | ORCA2_OMIP_L4 line | V1 build line | SUBROUTINE |
|---|---|---|---|
| `... / hbot_iwm(ji,jj)` (twice) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:267-268` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfiwm.f90:267-268` | `zdf_iwm` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:140`) |
| `hcri_iwm = 1._wp / sf_iwm(6)%fnow` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:485` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfiwm.f90:485` | `zdf_iwm_init` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:373`) |

Gates (resume):
- **Default gate** (`land.sh` list, incl. the citation-gate tests and the private-import ratchet) at `9e7c93520`:
  **145 passed** (1,006 s).
- **Per-receipt citation gate** on this receipt (`--from-heading` this title): **PASS, failures 0**, 21 citations,
  unmapped 0, map entries failing audit 0 (first run: 8 unmapped and one ladder pin shifted by this round's ladder
  edit; fixed by the two commits above).
- Ratchets (dispatch, inline coeffs, constants, param specs, MPAS IWM) at `6ff844d6d`: **5 failed**, the same 5 as
  base; none in this diff.

## Choices made

ASKED: survey; V4 = ORCA2's IWM exactly as its deck sets it; card; acquisition; off on other cards (D130); IWM's
TKE coupling arrives with this rung (D129).
UNASKED, offered for revert:
- the literal form requires the scalar-libm policy (raises under native);
- new refusals (none reachable today): literal IWM with `ln_mevar = T`, without the salt-diffusivity return, the
  step-entry bn2 bundle, the wet W mask or forcing maps, or a column-uniform 1-D depth ladder; an `&namzdf_iwm`
  group on any rung but V4;
- the acquisition adds "kt-1 avt differs from V3's frame" as an admission condition;
- the default (gyre round 8) receipt's citations were re-anchored to keep its gate green.

ASKED (resume): score V4 both labels; show IWM active; walk the first over-bar row; land V4-selected fixes.
UNASKED (resume), offered for revert:
- the ladder passes the card's maps to every model it builds (None on V1-V3);
- the ladder's kt-1 frame arm takes V4 and adds the IWM-active census against V3's admitted frame (fixed path);
- the `hbot` guard (the only way found to keep XLA's true division, as in round 13).

UNVERIFIED:
- beyond kt 1, avs is not in NEMO's records; it is certified through T and S (bit-for-bit kt 1-10);
- beyond kt 10: nothing scored;
- other divisions by a constant elsewhere in V4's step were not audited one by one; every score row is exact, so
  none moved a bit in these ten steps.

## 6. Second resume: V4 record test (review HOLD)

Independent Claude review reproduced the certification and held on one gap: removing the internal-wave call from the
K-profile builder left every V4 test green (no record test, unlike V2 and V3).
- **Added** (`f08d48983`): V4 record test, kt 1-2, against `round14/oracle_v4_r14` (skipped when absent): control
  (IWM moves NEMO's kt-2 T vs V3's record), kt-1 consumed avt/avs/avm wet and dry, every independent-arm field kt 1-2,
  IWM active in the same cells on both sides. V3's record-test body moved into two shared helpers; both tests call
  them (no copy). No arithmetic changes.
- **Plant** (scratch clone, internal-wave call replaced by a pass-through): **1 failed**, avt unequal in 414,119 wet
  W cells. **Restored**: V3 + V4 test files **15 passed**, 0 skipped (612 s). Evidence: `round14/resume2/`.
- Gates at this update: default gate __DEFAULT__; per-receipt citation gate __CITE__.

UNASKED (second resume), offered for revert: V3's record test now calls the shared helpers (same assertions).

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r14/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE (round 13's V2 EVD-trigger question was taken as D136; untouched here)
ROUND_STATUS: V4_CERTIFIED — V4 record test added (kt 1-2 every field + kt-1 avt/avs/avm; fails with IWM removed); V3 + ORCA2's internal-wave mixing (ln_zdfiwm T, ln_mevar F, ln_tsdiff T, six zdfiwm_forcing_orca2 maps, NEMO's forced TKE floors) bit-for-bit with NEMO kt 1..10 under both labels, every field incl. kt-1 avt/avs/avm and en/avm_k/avt_k/dissl (worst row 0 under each), after one V4-selected association fix (zdf_iwm's hbot_iwm divisions kept true under jit); IWM active in all 414,119 wet W cells, same cells on the card; codex review: paused (quota).
