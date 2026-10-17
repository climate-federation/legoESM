# Vertical-mixing lane — round 14 (vmix_r14): V4 = V3 + ORCA2's internal-wave mixing BUILT, record needed

Status: **V4 BUILT, NOT SCORED.** V4 = certified V3 + ORCA2's internal-wave mixing exactly as ORCA2 sets it
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:396` `ln_zdfiwm = .true.`; `&namzdf_iwm` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:421-435`). No new data source and no new model state. The NEMO record does not
exist yet; nothing is scored this round.

Decisions in force: D128-D130. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round14/`.
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v4_round14.md`, committed `4eb17a2a0` before any V4 code.

Commits: `4eb17a2a0` prereg; `d62474cd0` statement, card, deck, unit test; `e5ef6981e` acquisition and card tests;
`cf5a75855` citation map (69 pins re-anchored by mechanical offsets from this round's own hunks; no anchor text
changed; the default receipt's citations follow); `1e38cc6ba` the tsdiff factor 5 named at module level (inline-
coefficient ratchet; same value, same association); this receipt and its map entries.

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
- Card/deck/selector/run-script tests (V4) + V3, V2 card files + IWM unit files: see `cards_final.log`.
- Frame plant: V4's frame check on V3's record **refuses** (exit 66, `unequal_avt_vs_v3=0 of 825840`).
- **Flag-off jaxprs**, base `0b9443cef279` vs `e5ef6981e` (clean): full ORCA2 138,746 lines and DINO 123,234 lines
  **IDENTICAL**; V2 one-step 178,423 lines and V3 one-step 174,349 lines **IDENTICAL**. V4 one step: 176,175 lines,
  libm callbacks 137 → 142. `1e38cc6ba` changes one literal inside the V4-only statement; V4's jaxpr re-traced at
  the final tip is **IDENTICAL** to the `e5ef6981e` one. Structure only; forcing frames borrowed from V3's record.
- Ratchets (dispatch, inline coeffs, constants, param specs, MPAS IWM): tip and base **5 failed** each, the same 5
  (land slab-land guard, land restart coefficient, JRA55 test constant, shortwave spec, MPAS IWM splice); none in
  this diff. The new form guard sits in a function already in the dispatch baseline.
- **Default gate** (`land.sh` list): see `gate_default_final.log`.
- **Per-receipt citation gate** on this receipt: see `receipt_gate.log`.
- Acquisition preflight at `e5ef6981e`: `V4_PREFLIGHT_OK` (nothing built).

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

UNVERIFIED:
- nothing about V4 against NEMO is measured: no record exists;
- the full ORCA2/DINO jaxpr identity was measured at `e5ef6981e`; `1e38cc6ba` is read off the diff (one literal in a
  function no other card calls).

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r14/repo
ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/testcases/nemo_testcase_v4_orca2iwm/run.sh --run (nothing built: V1 record binary + G5 writer-free binary; deck = V3's + ln_zdfiwm and ORCA2's &namzdf_iwm; inputs = V3's + zdfiwm_forcing_orca2.nc from the on-disk ORCA2_ICE_v5.0.0 tarball, sha-pinned; records = V3's set + kt-1 zdf frame whose avt must differ from V3's; smoke → boundary → twins A/B/C to ≤10 steps) into gridladder_vmix_rounds/round14/oracle_v4_r14
DECISION_NEEDED: NONE (V2's step-entry EVD trigger decision from round 13 still pending; untouched here)
ROUND_STATUS: V4_BUILT_AWAITING_RECORD — V3 + ORCA2's internal-wave mixing (ln_zdfiwm T, ln_mevar F, ln_tsdiff T, six zdfiwm_forcing_orca2 maps, NEMO's forced TKE floors); no new data source, no new state; flag-off jaxprs of every other card identical to base; not scored; codex review: paused (quota).
