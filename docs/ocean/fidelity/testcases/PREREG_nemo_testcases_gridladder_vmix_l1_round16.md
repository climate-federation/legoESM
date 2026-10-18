# PREREG — lane B rung L1 (preregistered round 16, vmix_r16)

Frozen before any L1 code or measurement. No L1 record exists; nothing is scored this round.
Decisions 128-130, 136, 138 in force. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round16/`.

**L1 = certified G5 + ORCA2's bottom boundary layer, exactly as ORCA2 sets it** — nothing else
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:268-272`):
`ln_trabbl = .true.`, `nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl = 1000.`, `rn_gambbl = 10.`
(G5's deck: `ln_trabbl = .false.` only). Why BBL first: see the round-16 receipt's survey (order proposal).

NEMO (all in `ORCA2_OMIP_L4/BLD/ppsrc/nemo`):
- `tra_bbl_init` (`trabbl.f90:527`): `mbku_d`, `mgrhu/v` = sign of `gdept_1d(mbkt)` differences (`:583-591`),
  `e3u_bbl_0 = MIN(e3u_3d(mbkt(i+1)), e3u_3d(mbkt(i)))` (`:593-596`), `ahu_bbl_0 = rn_ahtbbl*e2_e1u*e3u_bbl_0*ssumask`
  (`:600-601`).
- `bbl` (`trabbl.f90:357`), called at stage 3 only (`stprk3_stg.f90:560`): bottom `ts(Kbb)`, depth
  `gdept_1d(mbkt)*(1+r3t(Kmm))` (`:412`), `eos_rab` 2-D (`:417`), gate `ahu_bbl = (0.5 - SIGN(0.5, -zgdrho*mgrhu))*ahu_bbl_0`
  (`:422-442`).
- `tra_bbl` (`trabbl.f90:154`) -> `tra_bbl_dif` (`:217`, `:249-266`), called at stage 3 after `tra_ldf` and `tra_bbc`,
  before `tra_dmp` and `tra_zdf` (`stprk3_stg.f90:739`). `nn_bbl_adv = 0`: `tra_bbl_adv` never runs; `rn_gambbl` is
  read and printed only.

## The card change (L1-selected; every other card unchanged)

- Card `ORCA2L1-bbl` = G5's card with the existing diffusive-BBL arm selected: `bbl_diffusive_option = 1`,
  `bbl_aht_m2_s = rn_ahtbbl`, `bbl_adv_option = 0`, `bbl_gamma_s = rn_gambbl` (inert under `nn_bbl_adv = 0`), all read
  from the deck. Reuses `legoesm.ocean.physics.bbl_adv` (`nemo_bbl_diffusive_geometry`,
  `nemo_bbl_diffusive_coefficients`, `apply_bbl_diffusive_tendency`), already wired into the stage-3 tracer RHS and
  selected by the full ORCA2 card. No model code changes; no new model state.
- Deck = G5's deck with ORCA2's five `&nambbl` lines (only diff); ice deck identical. The reader requires them on L1
  and refuses any of them elsewhere.

## Acquisition (defined exactly; operator runs it)

- **Binaries reused, nothing built**: record = G5's `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1` (sha `41cd9a63…`), writer-free
  reference = G5's `ORCA2G5_OMIP_L4_R31REF_FIX1` (sha `e8ebf4ae…`). Preflight: `trabbl.o`, `stprk3_stg.o`, `eosbn2.o`
  `.text` of the record binary = ORCA2_OMIP_L4's; `trabbl.o`/`trabbc.o` `.text`/`.rodata` of both = ORCA2_OMIP_L4's.
- **Namelist diff vs G5's deck**: the five `&nambbl` lines above.
- **Inputs**: G5's sixteen sha-pinned files. Nothing new.
- **Records**: G5's set (restarts 2..N over twins A/B/C, surface-input records each kt, step-entry records each kt).
  **Shows BBL acting**: the kt-2 step-entry record (state after kt 1) vs G5's admitted one
  (`gridladder_rounds/round31/oracle_g5_r31/twin_a`): u, v and sea level bit-identical (BBL acts only on the stage-3
  tracer update); T or S differ in >= 1 wet cell; every differing cell lies in a column with a wet U or V face whose
  two bottom levels differ (7,088 U + 6,705 V such faces; 11,936 of 16,433 wet columns), or on the two fold rows.
- **Logs**: `ln_trabbl = T`, `nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl = 1000`, `rn_gambbl = 10`.
- **Steps**: preflight -> 2-step smoke (record binary; acting check) -> 100-step boundary (writer-free) -> three twins
  to min(boundary, 10) -> admission.

## Predictions

- **P1 (record).** The run resolves the five lines, completes >= 10 steps, twins agree. Falsifier: acquisition refuses.
- **P2 (BBL acts, only where NEMO's code allows).** As defined above. Falsifier: no T/S change (BBL inert on this
  state: report, do not certify), any u/v/ssh change, or a changed column outside the footprint.
- **P3 (score, PLAUSIBLE).** L1 at the bar or bit-for-bit kt 1..10 both labels. First suspects if not: the content-form
  insertion of the BBL trend vs NEMO's Krhs order (after `tra_bbc`, before `tra_zdf`), and the card's stage-depth form
  of `gdept_1d(mbkt)*(1+r3t(Kmm))`. Falsifier: any over-bar row.
- **P4 (purity).** One-step jaxprs of G5, V1-V5, the full ORCA2 and DINO cards identical to base `f8b546f1a6`;
  L1's jaxpr differs from G5's; L1's initial state equals G5's in every field.
