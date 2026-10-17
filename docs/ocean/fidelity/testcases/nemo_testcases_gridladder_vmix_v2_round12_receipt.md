# Vertical-mixing lane — round 12 (vmix_r12): V1 pairing guard; V2 = V1 + EVD CERTIFIED

Status: **V2 CERTIFIED** (resume, 2026-10-17). V2 = certified V1 + ORCA2's enhanced vertical diffusion
(`ln_zdfevd` T, `nn_evdm` 0, `rn_evd` 100). Every field, every level, kt 1..10, bit-for-bit with NEMO's admitted
record under both labels. Worst row under each label: 0 (exact). Sections 1-6 are the pre-record half of the round;
section 7 scores V2.

Decisions in force: D128-D130. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round12/`.
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v2_round12.md`, committed `c833a83bd` before the card.

Commits: `9264d1b1c` pairing guard; `c833a83bd` V2 prereg; `d8523c5bc` V2 card, deck, acquisition, tests;
`96847a5a9` receipt, citation re-anchors and map entries; two follow-ups pinning/deduplicating map keys; this
update (test numbers).

## 1. V1 hardening: the two r3 statements must be selected together

- The step-entry shear builder now raises when exactly one of the round-10 rounding guard and the round-11
  stored-reciprocal form is selected. NEMO's sequence (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-269`
  stored, then `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:98-108` adds 1) needs both.
- Applies to every card (the check is in the shared builder). Only V1 selects either; it selects both.
- Unit test: both one-sided selections raise. Plant (guard removed): **2 failed, 4 passed**; restored: 6 passed.
- The old "divide form is not NEMO" case moved from (guard on, reciprocal off) to (both off), now illegal
  to mix.
- Flag-off one-step jaxprs, base `e01d443d3008` vs this tip `d8523c5bc` (clean tree): full ORCA2 138,747 lines
  and DINO 123,235 lines, **both IDENTICAL** (`jaxpr_all_final.log`).

## 2. Survey: ORCA2's vertical-physics switches vs V1

Sources: ORCA2's resolved namelist from a run of ORCA2_OMIP_L4's own deck (`nemo-testcases-l4/runs/
uninstrumented_10step/output.namelist.dyn`; its `namelist_cfg` differs from `ORCA2_OMIP_L4/EXP00/namelist_cfg`
only in run length/output lines 21-23; `namelist_ref` identical) vs V1's record resolved namelist
(`oracle_v1_r1/twin_a/output.namelist.dyn`). All routines are in `ORCA2_OMIP_L4/BLD/ppsrc/nemo`.

| switch (ORCA2 value; namelist line) | V1 | runs in | same? |
|---|---|---|---|
| ln_zdftke = T (cfg :389) | T | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:336` `zdf_tke`; init `:266` | yes |
| &namzdf_tke: nn_mxl 3, nn_etau 1, ln_mxhsw F, nn_mxlice 2 (cfg :406-415); rn_ediff 0.1, rn_ediss 0.7, rn_ebb 67.83, rn_emin 1e-6, rn_emin0 1e-4, rn_bshear 1e-20, nn_pdl 1, ln_mxl0 T, rn_mxl0 0.04, rn_mxlice 10, ln_lc T, rn_lc 0.15, nn_htau 1, rn_efr 0.05, nn_eice 1, nn_bc_surf/bot 1 (ref :1225-1261) | all 21 resolved values identical | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:732` `zdf_tke_init` | yes |
| ln_zdfevd = T, nn_evdm = 0, rn_evd = 100 (cfg :390-392) | **F** | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359` `zdf_evd` | **no → V2** |
| ln_zdfddm = T, rn_avts 1e-4, rn_hsbfr 1.6 (cfg :393-395) | **F** | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363` `zdf_ddm` | **no** |
| ln_zdfiwm = T (cfg :396; &namzdf_iwm :421-436) | **F** | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:372` `zdf_iwm` | **no** (D129: also sets TKE's floors) |
| nn_havtb = 1 (cfg :401) | **0** | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:217` in `zdf_phy_init` | **no** |
| nn_avb = 0, rn_avm0 1.2e-4, rn_avt0 1.2e-5 (cfg :398-400) | same | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:206` | yes |
| ln_zdfnpc = F (ref :1190) | F | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:762` `tra_npc` | yes (off) |
| ln_zdfmfc = F (ref :1193) | F | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:280` | yes (off) |
| ln_zdfswm = F (ref :1201) | F | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:371` | yes (off) |
| ln_zdfcst/ric/gls/osm = F, ln_zad_Aimp = F | same | `zdf_phy_init` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:126`) | yes |

So lane B's scope is NOT complete. Remaining order (round 1): V2 EVD, V3 DDM, V4 IWM, V5 nn_havtb.
Outside lane B, one other resolved difference: bottom drag (`ln_drg_OFF` T in V1, ORCA2 linear) — momentum
physics, not a vertical-mixing switch.

## 3. V2 built (V2-selected; nothing else moves)

- **Reused, not rebuilt** (searched `zdfevd`/`evd`/`enhanced_diffusion` in `packages/ocean`): the shared
  NEMO-literal EVD (GYRE, SMT and full ORCA2 cards already use it). V2 keeps the ORCA2 identity's own EVD arm,
  which the G-ladder had switched off, and states `nu_conv = 0` for nn_evdm = 0.
- NEMO: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:56` `zdf_evd`; avt replaced by rn_evd where
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109` fires; momentum only under
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121`.
- Deck: V1's deck plus ORCA2's three lines (only diff). The reader requires them on V2 and refuses them on V1.
- Card vs V1, every config leaf: moved = {convection scheme, convective viscosity}. Initial state: identical.
- Selector check pins all 11 EVD fields. Plants (6 fields + scheme off) each give exactly `["ln_zdfevd"]`.
  Plant (check always true): 1 failed; plant (reader drops the three lines): 1 failed + 2 errors.
- One V2 model step traces (178,423 jaxpr lines). Structure only: forcing frames borrowed from V1's record,
  no number taken (`v2_trace.log`).

## 4. ACQUISITION_NEEDED (exact)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v2_orca2evd/run.sh` (preflight: `V2_PREFLIGHT_OK`
at `d8523c5bc`). Run `run.sh --run` from a clean checkout.
- **Builds nothing.** Record binary = V1's (`ORCA2V1_OMIP_L4_R1FULLENTRY`, sha `b2ea4a47…`). Writer-free
  reference = G5's (`ORCA2G5_OMIP_L4_R31REF_FIX1`, sha `e8ebf4ae…`). Both compile `zdfevd`, `zdfphy` and
  `zdftke` byte-identically to ORCA2_OMIP_L4 (sha prefixes `eebb8a65…`, `903e0d12…`, `0f290190…`; checked).
- **Namelist diff vs V1's deck:** `ln_zdfevd = .true.`, `nn_evdm = 0`, `rn_evd = 100.` (ORCA2 cfg :390-392).
  Ice deck byte-identical to V1's.
- **Records:** V1's set — restarts at every writable step 2..N over twins A/B/C; surface-input and hm_i frames
  each kt; twins must agree on every frame.
- **Steps:** preflight (hashes, inputs, compiled-file identity) → 2-step smoke with the record binary → 100-step
  boundary run with the writer-free binary → three twins to min(boundary, 10) steps → admission. Each run's log
  must show ln_zdfevd T, nn_evdm 0, the EVD banner, and every V1 switch unchanged.
- Evidence: `gridladder_vmix_rounds/round12/oracle_v2_r12/`.
- Next round: add V2 to the ladder's case lists (it names V1 alone today), then score per the prereg.

## 5. Finding (not fixed, not this lane's card)

The full ORCA2 card's EVD mixes momentum too (convective viscosity 100 m²/s, i.e. nn_evdm = 1, inherited from
GYRE's identity). ORCA2's deck sets nn_evdm = 0 (cfg :391), so NEMO leaves viscosity alone
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121`). Measured by building the card: `nu_conv=100.0`. Under
D122 this is the ORCA2 lane's to fix; V2 will certify the nn_evdm = 0 form.

## 6. Tests (synchronous, CPU, fp64, `land.sh` PYTHONPATH)

- `land.sh` list without the face-mask file, plus the r3 unit test, V1 and V2 card tests and V1's kt 1-2
  record test (`gate1.log`): **165 passed, 3 failed** — all three in the citation gate, from map pins this
  round's edits shifted. Re-anchored (23 pins, mechanical offsets +1..+7, no anchor text changed).
- Citation gate tests + private-import ratchet after the fix (`gate2.log`): **23 passed**.
- Face-mask file, two processes: **1 passed** (472 s), **1 passed** (54 s).
- Per-receipt citation gate on this receipt (`--from-heading` this title): **PASS, failures 0**, 17 citations,
  unmapped 0, map entries failing audit 0 (`receipt_gate.log`).
- V1's kt 1-2 every-field bit-for-bit record test is in `gate1.log`'s 165 passed (V1 unchanged by the guard).

| evidence (`round12/`) | sha256 prefix |
|---|---|
| `jaxpr_all_final.log` / `jaxpr_orca2_fix.txt` / `jaxpr_dino_fix.txt` | 2af5627e14ac2473 / 8e7e8817080fbadc / 760bf60d3c5c7226 |
| `v2_trace.py` / `v2_trace.log` | 6e6b5e22150215bb / 8ca4cb988ec55e34 |
| `gate1.log` / `gate2.log` / `gate_fm1.log` / `gate_fm2.log` / `receipt_gate.log` | fabe909a80a01b89 / 7b9dfb51c5bc942b / 23057066ed7d085a / 1edcd6042ff68eb4 / 0d1f982ade5bdc10 |

## 7. V2 scored (resume after the record)

Record: `gridladder_vmix_rounds/round12/oracle_v2_r12` (operator: `V2_PREFLIGHT_OK` at `d19bf5efb`, nothing built;
`V2_ACQUISITION_PASS steps=1..10 restarts=2..10`). Scored at `e1dd4c29f` (model code tree `025199c33f5b`, unchanged
since `d19bf5efb`); CPU, x64.

- **P1 CONFIRMED.** Acquisition passed; each run's log shows `ln_zdfevd = T`, `nn_evdm = 0`, the `zdf_evd` banner.
- **P2 CONFIRMED (EVD is not inert).** NEMO's V2 vs V1 restarts, unequal cells: kt 2 T 349,592 / S 348,996;
  kt 4-10 T and S 430,552 each; en, avt_k, avm_k and u, v, ssh differ from kt 2 on.
- **INDEPENDENT kt 1-10:** 149 rows (surface sea level, barotropic transports, u, v, T, S each kt; all-level u, v,
  T, S, en, avm_k, avt_k, dissl and surface avm at kt 2-10): every row max 0, unequal 0.
- **GIVEN-NEMO-ENTRY (entries kt 1/3/5/7/9, one-step and two-step to kt 2/4/6/8/10):** 113 rows, every row max
  0, unequal 0.
- No flushed-subnormal cells (D127 census empty). Bar 1e-15 normalised; nothing needed it.
- **P3 FALSIFIED** (as preregistered, the "certifies as built" branch). The EVD trigger's own N² (not V1's
  per-operation-rounded one) produced no over-bar row in this window. No walk: there is no first non-bit row.
- **P4 CONFIRMED.** This round's model code is `d8523c5bc`'s; its flag-off ORCA2 and DINO jaxprs are identical to
  base (section 1). The resume touched only the scoring ladder and a test.
- Instrument: matching V2's record is not matching V1's. The card agrees with V2's record in every cell where
  V2's and V1's records differ in 430,552 T cells, so the scorer separates the two.

Ladder change (`e1dd4c29f`, script + test only): V2 joins the ladder's forced, sea-ice and TKE case lists; the TKE
list gates the restart TKE rows, NEMO-entry TKE leaves and the hm_i forcing. The kt-1 post-`zdf_phy` frame row stays
V1-only: on V2 that frame holds post-EVD avt while the card's carried leaf is the closure's (NEMO's restart
`avt_k` is the closure's too, written by `zdftke.f90:924`).

New test: V2 kt 1-2 every field bit-for-bit, plus the EVD-moves-kt-2-T control. Real: **7 passed** (157 s).
Plants: V2 off the TKE list → **1 failed, 6 passed** (model refuses the missing hm_i); TKE rows V1-only →
**1 failed, 6 passed** (assert on the TKE rows).

**Decision 122.** `zdfevd.f90`, `zdfphy.f90`, `zdftke.f90`, `eosbn2.f90` are byte-identical in V1's build and
ORCA2_OMIP_L4 (sha prefixes `eebb8a658810c104`, `903e0d1220a2f9b3`, `0f290190a33e4713`, `f08e8e47dab55811`).

| statement | ORCA2_OMIP_L4 line | V1 build line | SUBROUTINE |
|---|---|---|---|
| avt = rn_evd where MIN(rn2,rn2b) <= -1e-12 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfevd.f90:108-109` | `zdf_evd` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:56`) |
| avm untouched unless nn_evdm = 1 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfevd.f90:121` | `zdf_evd` |
| `CALL zdf_evd` after the avt_k copy | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfphy.f90:359` | `zdf_phy` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:296`) |
| trigger operand: before N² on Nbb, then rn2 = rn2b | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171`, `:174` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:177`, `:180` | `stp_RK3` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:62`) |
| `CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:185` | `stp_RK3` |

So under RK3 both trigger arms are the same before-N²; the card's "now/before" pair reads that.

| evidence (`round12/`) | sha256 prefix |
|---|---|
| `indep1_10.json` / `indep1_10.log` | a595bdbfb5adff60 / 82b0d645e692459b |
| `given_all.json` / `given_all.log` | 0f52e229579e3c02 / 5e69dd6bb4ff3b43 |
| `score_v2.sh` / `v2test.sh` | babdff714eba01b6 / 927f751331166901 |
| `v2test_real.log` / `v2test_plant.log` / `v2test_plant2.log` | bea3dbeef365e0ef / 49de19cc4f19d582 / 0290afc46227d6c8 |
| `acquisition_v2.log` (operator) | 79d3aa5035835c04 |

`given_all.log`'s hash equals round 11's V1 log: its row names carry no case and every row is 0; the JSONs differ.

## Choices made

ASKED: pairing guard; survey; V2 = first missing switch (EVD) exactly as ORCA2 sets it; card; acquisition.
UNASKED, offered for revert:
- the guard raises in the shared builder (any card), not only in V1's card check;
- V2 reuses V1's record binary and records (no new avt writer; EVD's post-closure avt is not in NEMO's restart,
  so attribution, if needed, is through T/S and an offline replay);
- V2 carries its own copy of V1's ice deck (test pins byte identity).

- resume: the kt-1 post-`zdf_phy` frame row is not registered for V2 (its avt is post-EVD, the card's leaf is not);
- resume: the new kt 1-2 V2 record test (lives in the V2 test file, record-gated, about 2.5 min).

UNVERIFIED:
- the card's post-EVD avt itself: NEMO's restart holds the closure's `avt_k`, so EVD's avt is certified only
  through T and S (bit-for-bit kt 1..10, where EVD moves 430,552 T cells);
- beyond kt 10: nothing scored.

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r12/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: V2_CERTIFIED — V1 + ORCA2's EVD (ln_zdfevd T, nn_evdm 0, rn_evd 100) bit-for-bit with NEMO kt 1..10 under both labels, every field incl. en/avm_k/avt_k/dissl (worst row 0 under each); codex review: paused (quota); next rung V3 = V2 + double diffusion.
