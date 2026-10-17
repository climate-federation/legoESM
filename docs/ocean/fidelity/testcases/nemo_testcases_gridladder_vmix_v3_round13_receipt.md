# Vertical-mixing lane — round 13 (vmix_r13): V3 = V2 + ORCA2's double diffusion, built, awaiting record

Status: **STOPPED_FOR_RECORD.** V3 = certified V2 + ORCA2's double-diffusive mixing (`ln_zdfddm` T, `rn_avts` 1e-4,
`rn_hsbfr` 1.6; `ORCA2_OMIP_L4/EXP00/namelist_cfg:393-395`). Card and acquisition built; nothing scored (no record).

Decisions in force: D128-D130. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round13/`.
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v3_round13.md`, committed `5ce2a0cb9` before any V3 code.

Commits: `5ce2a0cb9` prereg; `71b40a158` V3 statement, card, deck, acquisition, tests; `790eb4501` dispatch-baseline
entry; `1e7b7a6eb` this receipt; `d25a1a24d`, `85bbcb75c`, `397489eea`, `ffe02e136` citation map (7 new entries;
72 pins re-anchored by mechanical offsets, no anchor text changed); `35ee4f626` the default receipt's 9 model
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

## Choices made

ASKED: survey; V3 = next missing switch (DDM) exactly as ORCA2 sets it; card; acquisition; off on other cards (D130).
UNASKED, offered for revert:
- the literal form requires the scalar-libm policy (raises under native);
- the salt diffusivity leaves through a new opt-in return value rather than a carried field;
- four new refusals (literal DDM with IWM, isoneutral fold, K test overrides, physics-provided-K path);
- the acquisition adds the kt-1 avs ≠ avt check as an admission condition;
- the default (gyre round 8) receipt's 9 model citations were re-anchored to keep its gate green.

UNVERIFIED:
- the card's double diffusion on real ORCA2 fields: no V3 step was run (no record); wiring shown only by the trace;
- `rn2` the card feeds DDM lacks NEMO's trailing wmask; inert by construction (DDM multiplies by wmask), not measured;
- beyond kt 1, avs is not in NEMO's records; it is checked only through T and S.

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r13/repo
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r13/repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_v3_orca2ddm/run.sh
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — V3 (V2 + ORCA2's double diffusion) card and acquisition built, V3-selected, flag-off jaxprs identical; codex review: paused (quota).
