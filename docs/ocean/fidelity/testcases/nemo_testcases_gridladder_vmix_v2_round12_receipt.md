# Vertical-mixing lane — round 12 (vmix_r12): V1 pairing guard; V2 = V1 + EVD preregistered

Status: **STOPPED_FOR_RECORD.** V1 hardening landed. Survey: V1 runs ORCA2's TKE exactly as ORCA2 sets it, but
four of ORCA2's vertical-mixing switches are still off. V2 = V1 + enhanced vertical diffusion (EVD), the first
of them. V2's card and acquisition are built. Nothing about V2 is scored until its record exists.

Decisions in force: D128-D130. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round12/`.
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v2_round12.md`, committed `c833a83bd` before the card.

Commits: `9264d1b1c` pairing guard; `c833a83bd` V2 prereg; `d8523c5bc` V2 card, deck, acquisition, tests;
this receipt with its citation map entries.

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
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:107-110` fires; momentum only under
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
  record test: see `gate1.log` (result quoted in the final message).
- Per-receipt citation gate on this receipt: see final message.

## Choices made

ASKED: pairing guard; survey; V2 = first missing switch (EVD) exactly as ORCA2 sets it; card; acquisition.
UNASKED, offered for revert:
- the guard raises in the shared builder (any card), not only in V1's card check;
- V2 reuses V1's record binary and records (no new avt writer; EVD's post-closure avt is not in NEMO's restart,
  so attribution, if needed, is through T/S and an offline replay);
- V2 carries its own copy of V1's ice deck (test pins byte identity).

UNVERIFIED:
- whether EVD fires at all in 10 steps from ORCA2's initial state (prereg P2 decides it);
- the V2 card's numbers: nothing scored.

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r12/repo
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r12/repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_v2_orca2evd/run.sh
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — V1 pairing guard landed; survey found EVD, DDM, IWM and nn_havtb still off; V2 (V1 + EVD) card and acquisition built, awaiting the record.
