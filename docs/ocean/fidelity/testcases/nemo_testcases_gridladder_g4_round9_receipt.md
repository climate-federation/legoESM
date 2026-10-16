# Grid ladder — G4 round 9 (gridladder_r30)

Status: **LANDED. G4 CERTIFIED.** Every field is bit-identical to NEMO under
both labels at kt 1..10: surface rows 68/68 and all-level rows exact,
INDEPENDENT and GIVEN-NEMO-ENTRY.

Preregistration `PREREG_nemo_testcases_gridladder_g4_round9.md` (`4ca40515a`,
before any measurement). Record `gridladder_rounds/round21/oracle_g4_r21`.
Landing commit `21bc1881d`. Evidence `gridladder_rounds/round30/`
(`manifest.sha256`): `stage3_after.json` dc4fa34429b342af, `stage3_before.json`
5c440ce6a2044dcb, `g4_independent_kt1_10.json` d2b3b2e5b2c3cb49,
`g4_given_13579.json` 55ae3709b1dba4ca, `ratio_ladder.log` ac0968f45599d61f
(probe `probe/ratio_ladder.py` e9c717704602466d), `codex_review.log`
a11c5684ade3c444.

## Where the stage-3 tracer solve reads its thicknesses

At stage 3 (Kbb = N, Kmm = N+1/2, Kaa = N+1) every NEMO thickness is the
stored ratio `e3t_3d*(1+r3t(K)*tmask)`, with:

- `r3t(Kmm)` = stage 2's `r1_2*(r3t(Kbb)+r3ta)`
  (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stprk3_stg.f90:215`, np_HYB, the
  compiled default); `r3ta = ssha*r1_ht_0`
  (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/domqco.f90:257`).
- non-solar heat divisor, top cell
  (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:307`);
- runoff divisor `h_rnf`, reset to the top `e3t(Kmm)` by the stage-3
  divergence (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcrnf.f90:281`);
- RGB solar divisor `ze3t = e3t(Kmm)`
  (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/traqsr.f90:387`, read at :393);
- solve RHS `e3t(Kbb)*T + p2dt*e3t(Kmm)*Krhs`
  (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trazdf.f90:288-289`);
- off-diagonals on `e3w_1d*(1+r3t(Kmm))`
  (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trazdf.f90:231-232`) and the diagonal on
  `e3t(Kaa)` (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trazdf.f90:233`).

## Ratio ladder against NEMO's dumped ratios (kt 1, prediction 1)

| quantity (wet columns, 16433) | unequal | max abs |
|---|---:|---:|
| card `1+r3t(Kbb)` vs `1+e_r3t_bb` | 0 | 0 |
| card `1+r3t(Kaa)` vs `1+3_r3t_aa` | 0 | 0 |
| card stored stage-2 `1+r3t(Kmm)` vs `1+2_r3t_aa` | 0 | 0 |
| card top `h(Kbb)`, `h(Kaa)` vs `e3t_0*(1+r3t)` | 0 | 0 |
| card top Kmm thickness MEAN vs `e3t_0*(1+r3t(Kmm))` | 5625 | 1.776e-15 |

Prediction 1 CONFIRMED: the card's Kbb and Kaa thicknesses were already
NEMO's; the stage-3 Kmm thickness was the one non-NEMO rung.

Rule 11, my own instrument: the probe's first run (without the campaign's
fp64 policy) built the card's water depth in single precision and showed
1497 columns off by up to 1.2e-4 m; it also compared `(1+r3t)-1` with `r3t`.
Both were probe defects, not card defects; the table above is the fixed probe
(`probe/ratio_ladder.py`).

## The landing

One option, `rk3_stage3_tracer_nemo_stored_thickness`, makes stage 3's Kmm
thickness `e3t_0*(1+r3t(Kmm))` on NEMO's stored stage-2 ratio at every
stage-3 tracer read: heat and runoff divisors, RGB solar, solve RHS (with
round 8's written-order rounding), and the off-diagonal `e3w(Kmm)` — the whole
solve together, as round 8's finding 4 required. Kbb and Kaa are unchanged
(already NEMO's).

| kt 1 stage 3 (GIVEN-NEMO-ENTRY) | T unequal | T max abs | other fields |
|---|---:|---:|---|
| option deselected (control) | 200 | 7.105427357601002e-15 | S, ssh, uu_b, vv_b, v exact; u 56 signed zeros |
| option selected | 0 | 0 | same |

Prediction 2 CONFIRMED (target 0 reached). Round 8's finding 4 (RHS alone
worse by up to 9.8e-9 on 90 runoff cells) is consistent: the RHS was moved
off the thickness the divisors and off-diagonals still used.

Selection: default False; only G4 selects it (`_ORCA2G_TRA_STAGE_ORDER`, the
same per-rung table as rounds 7-8). G1, G1b, G2, G3 and every other card do not
select it; with it off every touched expression is the previous one. Caller
grep: the field is set only in `nemo_testcase_recipe.py`. Selecting it
without `rk3_stage3_heat_nemo_written` raises.

## Both labels, kt 1..10 (scored at `21bc1881d`)

| label | surface rows | not bit-identical | all-level rows | unequal cells |
|---|---:|---:|---:|---:|
| INDEPENDENT (kt 1..10) | 68 | 0 | 36 (kt 2..10 × u, v, T, S) | 0 |
| GIVEN-NEMO-ENTRY (entries 1/3/5/7/9 + two-step) | 68 | 0 | 20 | 0 |

Every region (fold, pivot rows, seam, south, open interior, coast) is exact.
Round 8's six DEBT rows (kt 9-10 ssh, u, v; worst 5.596e-14) are gone: the
INDEPENDENT drift was the kt 1 stage-3 temperature leave growing. Prediction 3
CONFIRMED. Instrument non-vacuity: `--plant score` exits 1 (kt 1 ssh DEBT,
1 cell); `--plant level` over kt 1..2 exits 1 (kt 2 u partial DEBT, 1 cell).

**G4 is CERTIFIED**: every field at the bar — in fact bit-identical — under
both labels at kt 1..10.

## Decision 122 (ORCA2_OMIP_L4 build, each opened at its SUBROUTINE)

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| stage-2 stored `r3t(Kmm)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221` | `stp_RK3_stg` (np_HYB default, :49) |
| `r3ta = ssha*r1_ht_0` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:257` | `dom_qco_r3c_RK3` |
| non-solar heat divisor | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307` | `tra_sbc_RK3` |
| runoff `h_rnf = e3t(Kmm)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:281` | `sbc_rnf_div` |
| RGB solar `ze3t` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traqsr.f90:388` | `qsr_RGBc` |
| solve RHS | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288-289` | `tra_zdf_imp` |
| off-diagonals / diagonal | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:231-232`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:233` | `tra_zdf_imp` |

The same compiled lines run on the full ORCA2 build (stage lines shift +6).
The full ORCA2 card does not select the option; UNMEASURED there.

## Tests and review

- Fast G4 + G2-ladder + private-import + dispatch: 67 passed, 1 failed (the
  known pre-existing slab-land dispatch guard).
- Slow G4 stage route: 1 passed (381.08 s, after the review fix); asserts
  every stage-3 value exact with the option, 200 T cells with it deselected,
  and round 8's before arm (heat statement off) at 489.
- Citation gate (from this receipt's first heading): PASS, 16 citations, map
  audit clean; shifting `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stprk3_stg.f90:215` exits 1 (SYMBOL-NOT-AT-LINE).
  The model-file shift was remapped with the reanchor tool (53 lines); one
  span's extent grew 172 -> 174 with the same endpoint statements.
- Citation tests: 193 passed. Config field-order test updated (field appended
  last).
- Single review (codex): "DO NOT SHIP" on two test items, no transcription
  defect ("NEMO signs, levels, RGB ladder, RHS, and e3w(Kmm) agree; momentum
  uses separate dz_half_u/v"; stage pair is (ssh Kbb, ssha); default False,
  G4 only). Acted on:
  1. CONFIRMED: the heat-deselected control now hit the new dependency guard.
     Fixed: deselecting heat also deselects the Kmm option (round 8's before
     arm); its 489 assertion is restored and passes.
  2. CONFIRMED coverage gap: salinity is exact with and without the option.
     Registered, not fixable on this record: NEMO's stage-3 salt Krhs is zero
     everywhere at kt 1 (`sfx` 0, runoff salinity 0 on all 3400 runoff
     cells), so the salt divisor change multiplies zero; S is also exact under
     both labels kt 1..10.

ASKED: transcribe the stored-ratio thicknesses for the whole stage-3 solve,
G4-selected; re-score kt 1 stage 3 and both labels kt 1..10; Decision-122
pointers; tests; one review. UNASKED: none.

UNVERIFIED: the full ORCA2 card's response (pointer only); the guard that
raises when the option is selected without the heat option has no direct test;
the salinity half of the option has no failing control on this record.

## OPEN

G4 is certified. Next rung per the ladder plan (sea ice) or the Decision-122
landing of this statement on the full ORCA2 card (measure there first).
