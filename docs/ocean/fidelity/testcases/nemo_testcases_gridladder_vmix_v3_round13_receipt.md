# Vertical-mixing lane — round 13 (vmix_r13): V3 = V2 + ORCA2's double diffusion CERTIFIED

Status: **V3 CERTIFIED** (resume, 2026-10-17). V3 = certified V2 + ORCA2's double-diffusive mixing (`ln_zdfddm` T,
`rn_avts` 1e-4, `rn_hsbfr` 1.6; `ORCA2_OMIP_L4/EXP00/namelist_cfg:393-395`). After two association fixes, every field,
every level, kt 1..10, bit-for-bit with NEMO's admitted record under both labels; worst row under each label: 0 (exact).
Sections 1-4 are the pre-record half; section 5 scores V3.

Decisions in force: D128-D130. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round13/`.
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v3_round13.md`, committed `5ce2a0cb9` before any V3 code.

Commits: `5ce2a0cb9` prereg; `71b40a158` V3 statement, card, deck, acquisition, tests; `790eb4501` dispatch-baseline
entry; `1e7b7a6eb` this receipt; `d25a1a24d`, `85bbcb75c`, `397489eea`, `ffe02e136` citation map (7 new entries;
71 pins re-anchored by mechanical offsets, one range also +1 at its end; no anchor text changed); `35ee4f626` the default receipt's 9 model
citations re-anchored (+14); this update (final numbers).

Own error, stated first: a plant loop restored the double-diffusion module with `git checkout --` while it was
uncommitted, which reverted it; four plants ran on the reverted file and are void. The edit was re-applied, staged,
and every plant re-run (section 4).

## 1. Survey

NEMO (all routines in `ORCA2_OMIP_L4/BLD/ppsrc/nemo`):
- `zdf_ddm` is `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:59`. It is called at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363`, inside `zdf_phy` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:296`),
  after `zdf_evd` and before `zdf_iwm`. With the switch off NEMO sets avs = avt.
- It writes a separate salt diffusivity (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:170`) and adds to avt and avm
  (`:171`, `:172`). Tracer diffusion builds a second matrix for salt from avs
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:175`).
- avs is recomputed every step from that step's T, S, alpha, beta and N²: **no new model state** (the card carries
  none; nothing for the user to decide on state).
- Its operands are bn2's: same 1-D `zrw`, same `zaw`/`zbw` times the two T masks
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1632-1636`), on the step-entry `rab_b`/`rn2b`.

Compiled in? Yes, identically:

| file | ORCA2_OMIP_L4 | V1 record build | G5 writer-free build |
|---|---|---|---|
| `zdfddm.f90` / `zdfphy.f90` / `trazdf.f90` / `zdf_oce.f90` (sha256 prefix) | `6b0be5e4…` / `903e0d12…` / `02905616…` / `875d466f…` | same | same |
| `zdfddm.o` | `71f25284f7e61caa` | same | same |

Build flags `-fdefault-real-8 -O3 -fno-tree-vectorize`, no `-march`: unsuffixed literals are double and there is no
fused multiply-add.

Our repo (searched `zdfddm|zdf_ddm|double.?diffus|salt.?finger|rn_avts|hsbfr|ln_zdfddm` in packages, src, scripts,
tests): one double-diffusion module, a documented hybrid (Large/CVMix fingering cubic, sign-preserving ratio, no
momentum term); the C-grid model already carries a salt diffusivity into the generic tracer solve; the
NEMO-literal tracer solve refused distinct heat/salt matrices. Extended all three; no second module.

Which NEMO output shows it acting: V1's record binary dumps avm, avt, avs, en right after `zdf_phy` at kt 1
(`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:186`). Restarts hold only the closure's avt_k, so later
steps show DDM through T, S, u, v. Control on V2's record: **0 of 825,840** interior W cells have avs ≠ avt.

## 2. V3 built (V3-selected; nothing else moves)

- **Statement**: `nemo_zdf_ddm` in the existing double-diffusion module, selected by a new config value
  `form="nemo_literal"` (default `"hybrid"`, unchanged). Each operation rounded in source order; both EXPs through
  the scalar-libm precision policy (as V1's EXP); refuses the native policy.
- **Operands**: the card's existing step-entry bn2 (V1's per-operation-rounded `rab_3d_t`/`bn2_t`) now also returns
  its temperature and salinity contributions on request; V3 requests them. No other card does.
- **Place**: shared K-profile builder, after EVD, before IWM (NEMO's order). avs leaves through an opt-in return;
  the NEMO-literal pair solve takes an optional salt diffusivity and builds salt's own matrix with the same assembly.
- **Refusals** (new combinations, none reachable today): literal DDM with IWM, with an isoneutral fold, with the
  vertical-K test overrides, or on the physics-provided-K path.
- **Card**: `build_orca2v3_ddm_card` = V2's card plus the DDM selection. Card vs V2, every config leaf:
  moved = {ddm enabled, ddm form}. Initial state identical. Deck = V2's plus ORCA2's three lines (only diff); the
  reader requires them on V3 and refuses them on V2.

## 3. ACQUISITION_NEEDED (exact)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v3_orca2ddm/run.sh` — run `run.sh --run` from a clean
checkout (default is read-only preflight, `V3_PREFLIGHT_OK`).
- **Builds nothing.** Record binary = V1's (`ORCA2V1_OMIP_L4_R1FULLENTRY`, sha `b2ea4a47…`); writer-free reference =
  G5's (`ORCA2G5_OMIP_L4_R31REF_FIX1`, sha `e8ebf4ae…`). Preflight checks `zdfddm`, `zdfevd`, `zdfphy`, `zdftke`,
  `trazdf`, `zdf_oce` sources and `zdfddm.o` against ORCA2_OMIP_L4.
- **Namelist diff vs V2's deck**: `ln_zdfddm = .true.`, `rn_avts = 1.e-4`, `rn_hsbfr = 1.6` (ORCA2 cfg :393-395).
  Ice deck byte-identical to V2's.
- **Records**: V2's set (restarts every writable step 2..N over twins A/B/C; surface-input and hm_i frames each kt;
  twins agree) plus the kt-1 post-`zdf_phy` frame, which must show avs ≠ avt in ≥ 1 cell (P2a) and agree across twins.
- **Steps**: preflight → 2-step smoke (record binary; its frame must show DDM) → 100-step boundary run (writer-free)
  → three twins to min(boundary, 10) → admission. Every log must show `ln_zdfddm = T`, both parameters, `use double
  diffusive mixing: avs /= avt`, and V2's EVD/TKE lines.
- Evidence: `gridladder_vmix_rounds/round13/oracle_v3_r13/`.
- Next round: add V3 to the ladder's case lists, then score per the prereg.

## 4. Tests and gates (synchronous, CPU, fp64, `land.sh` PYTHONPATH)

- **Statement vs an independent scalar transcription** (Python floats, `math.exp` = glibc), 4,000 interfaces across
  fingering, both diffusive branches, the single-signed floor, the `zds` floor, the 0.5/1/1.6/2 edges and the N² edge:
  **bit-for-bit** (avs ≠ avt in > 1,000 cases). Plants, each **1 failed**: mask edge `<=`→`<`; native EXP;
  `/rn_hsbfr`→`*(1/rn_hsbfr)`; avs built from the heat term; N² offset dropped (after adding an N² ∈ (−1e-12, 0)
  case — first run passed, my test lacked that case). One planned plant (power order) was void: a·(a·a) = (a·a)·a.
- Salt matrix: salt solved with K_2 equals salt solved alone on K_2; temperature unchanged; K_2 = None identical.
- Card/deck/selector/run-script tests (V3) + V2's file incl. its kt 1-2 record test: **17 passed** (183 s).
  Frame plant: V3's frame check on V2's record **refuses** (exit 66, `unequal_avs_avt=0 of 825840`).
- **Flag-off jaxprs**, base `e1b56d6004c6` vs final tip `35ee4f626` (clean): full ORCA2 138,747 lines and DINO
  123,235 lines **IDENTICAL**; V2 one-step 178,423 lines **IDENTICAL**. V3 one step traces: 178,721 lines, libm
  callbacks 135 → 137 (the two EXPs). Structure only, forcing frames borrowed from V2's record, no number taken.
  (Same result at `71b40a158`.)
- Ratchets (dispatch, strict-validate, param specs, inline coeffs, constants, contracts, existing DDM tests,
  reachability, bounds, private imports): **6,372 passed, 4 failed** — the same 4 fail on base (land slab-land guard,
  shortwave spec, land restart coefficient, a JRA55 test constant), none in this diff. The new form guard is locked in
  the dispatch baseline.
- **Default gate at the final tip** (`land.sh` list): citation-gate tests + private-import ratchet + TKE terms +
  recipe + freshwater + prognostic barotropic + V3 + unit files **150 passed, 1 failed** (the default receipt's 9
  model citations, shifted by this diff) → re-anchored → citation-gate file + ratchet **23 passed**. Face-mask file,
  two processes: **1 passed** (596 s), **1 passed** (60 s). V2 + V3 + unit files **17 passed** (169 s); V1 file
  **14 passed** (default `not slow`).
- **Per-receipt citation gate** on this receipt (`--from-heading` this title): **PASS, failures 0**, 10 citations,
  unmapped 0, map entries failing audit 0.
- Acquisition preflight at `35ee4f626`: `V3_PREFLIGHT_OK` (nothing built).

| evidence (`round13/`) | sha256 prefix |
|---|---|
| `jaxpr_all_final.log` / `jaxpr_orca2_fix.txt` / `jaxpr_dino_fix.txt` | 2af5627e14ac2473 / 7484b3c5d84cadd9 / a241c2926df7a006 |
| `vtrace_final.log` / `jaxpr_v2_fix.txt` / `jaxpr_v3_fix.txt` / `vtrace.py` | 7c7b9604c2cb52e3 / 8547d7e248cf2b76 / a5bec0ede95b82c9 / fa11edf5c2117111 |
| `unit.log` / `unit_plants.log` / `t1.log` / `ratchets.log` / `ratchets_base.log` | 84a62be660accfa1 / c126dce32835e8c6 / 2ad538921206c4dc / 65319b8c537d6601 / aeda013a23d0edf5 |
| `gate1.log` / `gate2.log` / `gate3.log` / `gate4.log` / `gate_fm1.log` / `gate_fm2.log` / `receipt_gate.log` | 45285acd934232b8 / dc0afcd736099b23 / 02acf18a4db47c29 / 0fea8fd1fbcaa6db / 01740623ef8dd364 / a2494d24cec5c6f4 / bf37493338f8f5d8 |

`jaxpr_all_final.log`'s hash equals round 12's: both logs read "orca2 IDENTICAL / dino IDENTICAL"; the jaxprs differ.

## 5. V3 scored (resume after the record)

Record: `gridladder_vmix_rounds/round13/oracle_v3_r13` (operator: `V3_PREFLIGHT_OK` at `8fedda974`, nothing built;
`V3_ACQUISITION_PASS steps=1..10 restarts=2..10`; `V3_DDM_FRAME unequal_avs_avt=412297 of 825840`). CPU, x64, `land.sh`
PYTHONPATH. Commits: `568c70c42` ladder; `5e5c3c645` DDM division fix; `cae45eeee` EVD trigger fix; `dd00283a6` map
entries and two comment line numbers; `68ba80bdf` receipt draft; `1517758c5` citation re-anchor; this update.

Own error, stated first: my round-13 statement test ran only eagerly; the jitted statement was never compared, and it
differed.

**First score, V3 as built (`8fedda974`): not at the bar.**
- kt 1: the avt/avs/avm the solves consume vs NEMO's post-`zdf_phy` frame: 9,735 / 15,105 / 5,113 wet W cells
  unequal (max 1.7e-18 / 8.7e-19 / 6.9e-18 absolute); T and S unequal in 2 cells.
- First over-bar row: kt 2, both labels (T max 2.7e-13, S 1.7e-12; all-level T 1.41e-13 normalised).

**Walk 1, kt 1, `zdf_ddm` first.**
- NEMO's statements (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:59`) in scalar doubles with glibc `exp`, on the card's
  own operands: equal to NEMO's frame in **every** wet W cell, avt, avs and avm (0 unequal). The operands and
  source order are NEMO's.
- The card's statement run eagerly equals that too (0). Run under jit alone, it differs in exactly the card's cells,
  all of them salt-fingering.
- Optimized HLO: `zrau / rn_hsbfr` is lowered as `zrau * 0.625`. NEMO divides by a runtime namelist value
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:161`, the same line in
  `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfddm.f90:161`, inside `zdf_ddm` at
  `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfddm.f90:59`).
- Fix: the divisor passes through the existing source-rounding guard, keeping a true division (isolated: 0 vs
  70,442 unequal of 200,000). After it: kt-1 avt/avs/avm 0 unequal; kt 1 and 2 every row exact.

**Walk 2, the next over-bar row: kt 3, all-level T and S, one column.**
- Grid point (j, i) = (118, 34) (0-based), levels 7-16, max 0.434 °C at level 12. The card's kt-2 state equals NEMO's
  kt-2 restart bit for bit.
- NEMO's step-entry N² at W level 13 there is −1.873e-9, so `zdf_evd` fires
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109`, in `zdf_evd` at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:56`).
  Those rn2/rn2b are built once on Nbb (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171`, `:174`) before `zdf_phy`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`, in `stp_RK3` at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:62`).
- The card's EVD trigger rebuilt N² on the ssh of the RK3 stage-3 solve (−0.431 m there vs −0.133 m at step
  entry): +8.4e-10, so it did not fire. NEMO's T is mixed over levels 12-16, the card's over 13-16.
- That is the only trigger disagreement in the domain (NEMO's test fires on 4,810 interfaces, the card on 4,809). On
  step-entry geometry the same trigger gives −1.873e-9 and 0 disagreements.
- Fix (V3-selected): a new EVD switch, default off, makes the trigger read the step-entry bn2 arrays the card
  already builds for TKE and DDM. Only V3's card selects it; its selector check pins it on V3 and off on V2.

**Final score (`cae45eeee` model code; packages tree `c8bd07a6`, later changed by two comments only).**
- **INDEPENDENT kt 1-10:** 149 rows (surface sea level, barotropic transports, u, v, T, S each kt; all-level u, v, T,
  S, en, avm_k, avt_k, dissl and surface avm at kt 2-10): every row max 0, unequal 0.
- **GIVEN-NEMO-ENTRY (entries kt 1/3/5/7/9, one- and two-step):** 113 rows, every row max 0, unequal 0.
- **kt-1 frame, consumed coefficients:** avt, avs, avm each 0 unequal of 414,119 wet W cells, 0 dry.
- No flushed-subnormal cells (D127 census empty: no unequal cell). Bar 1e-15 normalised; nothing needed it.

**Double diffusion is active.**
- NEMO's kt-1 frame: avs > avt (salt fingering) at 145,986 wet W cells, avs < avt (diffusive convection) at 266,311.
  The card: the same two counts.
- Card branches at kt 1, of 412,734 stable wet interfaces: fingering (R > 1) 146,423; diffusive (R < 1) 266,311
  (0.5 < R < 1: 8,602; R < 0.5: 257,709, of which 209,643 single-signed at the 1e-20 floor).
- NEMO's V3 vs V2 restarts, unequal cells kt 2-10: T 430,552; S 430,551 (kt 2), then 430,552; u 413,030; en 115,092 →
  285,302; avt_k 172,498 → 196,022.

Preregistration:
- **P1 CONFIRMED** (acquisition passed; DDM lines and banner in every log, per the acquisition).
- **P2 CONFIRMED**: (a) 412,297 kt-1 cells with avs ≠ avt; (b) 430,552 T cells differ from V2's record at kt 2.
- **P3 CONFIRMED for the first owner** (the division at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:161`, inside the predicted :159-172; not the EXP). Not
  predicted: the second owner, the EVD trigger's geometry, outside `zdf_ddm`.
- **P4 CONFIRMED**: full ORCA2 (138,748 lines) and DINO (123,236) one-step jaxprs identical to base `e1b56d6004c6`;
  V2's one step (178,423 lines) identical too.

Tests (CPU, x64):
- Statement test now also runs under jit: 4 passed. Division fix removed: **1 failed** (jit), 3 passed.
- New record test (V3 kt 1-3 every field, kt-1 avt/avs/avm, branch census, DDM-moves-kt-2-T control): **1 passed**
  (493 s). Plant A (division fix removed): **1 failed**, kt-1 avt. Plant B (EVD switch ignored): **1 failed**, kt-3
  T and S. Tree verified clean after each.
- V3 card/selector tests: 11 passed (the card diff now includes the EVD switch; planted switch flips give
  `["ln_zdfevd"]` on V3 and V2).

**Decision 122.** `zdfddm.f90` and `zdfevd.f90` are byte-identical in V1's build and ORCA2_OMIP_L4 (sha prefixes
`6b0be5e4ab056501`, `eebb8a658810c104`).

| statement | ORCA2_OMIP_L4 line | V1 build line | SUBROUTINE |
|---|---|---|---|
| zrr = zrau / rn_hsbfr | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:161` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfddm.f90:161` | `zdf_ddm` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:59`) |
| avt = rn_evd where MIN(rn2,rn2b) <= -1e-12 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfevd.f90:108-109` | `zdf_evd` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:56`) |
| bn2 on Nbb, then rn2 = rn2b | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171`, `:174` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:177`, `:180` | `stp_RK3` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:62`) |
| `CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:185` | `stp_RK3` |
| `CALL zdf_evd`, then `CALL zdf_ddm` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359`, `:363` | `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfphy.f90:359`, `ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdfphy.f90:363` | `zdf_phy` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:296`) |

**Finding (not fixed; a default question, RULE 3).** V2's card and the full ORCA2 card build the EVD trigger the same
way as V3 did before this fix. In V2's ten certified steps no interface flipped, so V2's numbers stand; the trigger is
still not NEMO's statement there.

Resume gates (CPU, x64, `land.sh` PYTHONPATH):
- `land.sh` list (without the face-mask file) plus the V1, V2, V3, statement, r3-guard and ladder test files, at
  `dd00283a6`: **186 passed, 3 failed, 9 deselected** (the repo's default `not slow`). The 3 are citation-gate map
  pins shifted by this resume's edits → re-anchored (10 pins, mechanical offsets from the files' own diffs; no anchor
  text changed) → citation-gate file + private-import ratchet: **23 passed**.
- Face-mask file, two processes: **1 passed** (471 s), **1 passed** (53 s).
- Ratchets (dispatch, strict-validate, param specs, inline coeffs, constants, contracts, reachability, bounds,
  private imports, existing DDM/EVD tests): **6,362 passed, 4 failed** — the same 4 that fail on base (section 4).
- Per-receipt citation gate on this receipt (`--from-heading` this title): **PASS, failures 0**, 26 citations,
  unmapped 0, map entries failing audit 0.
- Flag-off jaxprs at the final model code (`dd00283a6`): full ORCA2 138,748 lines, DINO 123,236, V2 one step 178,423:
  all **IDENTICAL** to base.

| evidence (`round13/`) | sha256 prefix |
|---|---|
| `indep1_10.json` (as built) / `indep1_10_fix.json` (DDM fix) / `indep1_10_fix2.json` / `indep1_10_fix2.log` | cd2e8b0cb423077f / 98c6bcb726556d12 / 277c9f79628b6368 / cb1b289d93206dbb |
| `given_all_fix2.json` / `given_all_fix2.log` / `score_v3_fix2.sh` | 8ef0f97c9408b7b4 / 5e69dd6bb4ff3b43 / 07e07497961c7317 |
| `ddm_kt1.json` / `ddm_kt1_fix.json` / `ddm_walk.py` / `ddm_walk.json` / `ddm_walk_fix.json` | 1e97c9c11df1aa22 / ed8bac2e53c5587b / 30b131029de42c32 / f2e14c0197814364 / 2e8bc17e5d12b8f3 |
| `kt3_evd.py` / `kt3_evd.json` / `kt3_n2.py` / `kt3_n2.log` | c6ea6a8f35d0f8e3 / de4230828073bd71 / 8f60384b335c9b3b / dce5d92410ca1fe6 |
| `v3test_real.log` / `v3test_plantA.log` / `v3test_plantB.log` / `resume_unit.log` / `resume_unit_plant.log` | d33f68177af991e1 / 0c54245bc677594f / 4dfd289c7084c3fb / 68990868b1682922 / 30c5b06ec9b61d64 |
| `gate_resume.log` / `gate_resume_cite.log` / `ratchets_resume.log` / `gate_resume_fm1.log` / `gate_resume_fm2.log` | 8b2ed1b3a81f868a / 536991d1c6ea4996 / 2f5028d05644bbd5 / ea02289e1f19f102 / 478c3790a1e9dbc9 |
| `receipt_gate_resume.log` / `jaxpr_resume_final.log` / `jaxpr_orca2_resume.txt` / `jaxpr_dino_resume.txt` / `jaxpr_v2_resume.txt` | 0ec9110ae4da8522 / 6dd8bd1ddc80f73b / 663832c195f70e0d / 41c263bd8f5f727d / 8547d7e248cf2b76 |
| `acquisition_v3.log` (operator) | 826bc35c9d0bd505 |

`given_all_fix2.log`'s hash equals round 12's: its row names carry no case and every row is 0; the JSONs differ.

## Choices made

ASKED: survey; V3 = next missing switch (DDM) exactly as ORCA2 sets it; card; acquisition; off on other cards (D130).
ASKED (resume): score V3 both labels; walk the first over-bar row; land V3-selected association fixes.
UNASKED, offered for revert:
- the literal form requires the scalar-libm policy (raises under native);
- the salt diffusivity leaves through a new opt-in return value rather than a carried field;
- four new refusals (literal DDM with IWM, isoneutral fold, K test overrides, physics-provided-K path);
- the acquisition adds the kt-1 avs ≠ avt check as an admission condition;
- the default (gyre round 8) receipt's 9 model citations were re-anchored to keep its gate green.

- resume: the divisor guard in the literal DDM (the only way found to keep XLA's true division);
- resume: the new EVD switch is a separate boolean, default off, selected by V3 alone (D130), rather than a new value
  of the existing time-level switch;
- resume: the ladder's new kt-1 frame arm and the private trace's salt-diffusivity field;
- resume: the new V3 record test (about 8 minutes, record-gated).

UNVERIFIED:
- beyond kt 1, avs is not in NEMO's records; it is certified through T and S (bit-for-bit kt 1-10, where DDM moves
  430,552 T cells);
- beyond kt 10: nothing scored.

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r13/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: V2 and the full ORCA2 card still rebuild EVD's trigger N2 on the stage-3 ssh (latent; no flip in V2's 10 steps). 1 = leave both; 2 = move V2 to the step-entry trigger and re-score it on its existing record (ORCA2 card stays lane A's, D122); 3 = move both. Pick: 2.
ROUND_STATUS: V3_CERTIFIED — V2 + ORCA2's double diffusion (ln_zdfddm T, rn_avts 1e-4, rn_hsbfr 1.6) bit-for-bit with NEMO kt 1..10 under both labels, every field incl. kt-1 avt/avs/avm and en/avm_k/avt_k/dissl (worst row 0 under each), after two V3-selected association fixes (DDM divisor under jit; EVD trigger on step-entry rn2/rn2b); codex review: paused (quota).
