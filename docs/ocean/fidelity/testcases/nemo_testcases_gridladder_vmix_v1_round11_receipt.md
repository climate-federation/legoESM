# Vertical-mixing lane — V1 round 11 (vmix_r11)

Status: **V1 CERTIFIED.** Every field and level is bit-for-bit with NEMO at kt 1..10 under both labels.
- Owner of the kt-6 T debt: the step-entry sea-level ratio `r3u`/`r3v`. NEMO multiplies by its two stored
  factors in sequence. The compiled card folded the two constants into one.
- Landed V1-selected (default False on every other card, D130). Flag-off ORCA2 and DINO step jaxprs are
  identical to base `d31991655`.

Decisions in force: D129 = 1, D130 = 1. Codex review: paused (quota).
Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round11.md`, frozen at `959769e7e` before any
measurement. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3). Evidence:
`gridladder_vmix_rounds/round11/` (`manifest.sha256`, 40 files; state pickles excluded).

Commits: `959769e7e` prereg; `df7931442`, `6b164e180`, `3b19a7221` probe (closure from NEMO's entry, trace kind,
shear builder evaluated alone) with direct tests; `9a7487ac3` fix; `152db9f5a` citation re-anchors; this receipt.

| file | sha256 prefix |
|---|---|
| `closure_kt5_nemo.json` / `closure_kt5_nemo_avx.json` / `closure_kt5_nemo_fix.json` | 38380a7f0e5d93cd / 191d5febd1d293be / 55c9211c50c89434 |
| `trace_kt5_nemo.npz` / `trace_kt5_nemo_fix.npz` | 2de909b91be2d673 / 973acfb882889bcf |
| `replay_prandtl_kt5.log` / `replay_sh2_kt5.log` / `replay_sh2.py` | b7dd81764de3ebd5 / ce610a0c78e3ebd8 / 95605f60e7022e11 |
| `fma_toy_r11.py` / `jaxpr_all.log` | bffe785f41e358a3 / 2af5627e14ac2473 |
| `indep1_5.log` / `indep6_10.log` / `given_all.log` | 55df3e8cc3716cdc / 5501d20034a71b45 / 5e69dd6bb4ff3b43 |

## 1. One step from NEMO's kt-4 state (P1 CONFIRMED, P3 CONFIRMED)

Committed probe, closure kind, entry = NEMO's full kt-4 restart, step 5, scored against NEMO's kt-5 restart
(unequal cells):

| arm | u | v | T | S | ssh | en | avm | avt | dissl |
|---|---|---|---|---|---|---|---|---|---|
| plain | 13 | 10 | 5 | 4 | 0 | 1 | 2 | 5 | 2 |
| self (identity control) | bit-identical to plain | | | | | | | | |
| NEMO's `avm_k` | 0 | 0 | 5 | 4 | 0 | 1 | 2 | 5 | 2 |
| NEMO's `avt_k` | 13 | 10 | **0** | **0** | 0 | 1 | 2 | 5 | 2 |
| both | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 5 | 2 |

- T/S cells: 4 T and 3 S in column (34,120) (levels 1-4), plus one each at (119,51) level 10. Worst T
  1.33e-15 absolute (4.4e-17 normalised); worst S 1.42e-14 (3.5e-16).
- In column (34,120) only `avt` differs, at NEMO w-level 2. Energy and viscosity are exact there.

## 2. Not an FMA (P2 FALSIFIED)

- The same one step with FMA disabled for the whole program (`XLA_FLAGS=--xla_cpu_max_isa=AVX`): every count
  above unchanged.
- That switch works. On a toy jit it takes `a*b+c` from 949 to 0 unequal cells. It leaves division by a
  constant rewritten (1,391 cells either way).

## 3. Walk at (34,120), compiled order

1. **Diffusivity tail.** Offline replay of NEMO's `zri`, `pdlr` and `avt` from the traced operands. The trace
   is passive (traced step bit-identical to plain). The replay equals the card's published `avt` in all 5
   residue cells. So the tail is NEMO's, and an operand is not. One ulp up on the shear gives NEMO's `avt`
   at (34,120). At (119,52) only a shear shift does.
2. **Shear.** Offline NEMO-order replay of `zdf_sh2` from NEMO's kt-4 restart. Kbb = Kmm at step entry
   (`zdf_phy` call below). It gives 1.6418652047573123e-07 at (34,120) w-level 2. The card's traced shear is
   1.641865204757312e-07, 1 ulp low. It is also 1 ulp off at (34,120) w-5, (119,51) w-3 and (119,52) w-3.
3. **Builder alone (committed probe).** The card's step-entry shear builder on NEMO's entry gives:
   - op-by-op: 1.6418652047573123e-07, NEMO's value; 60 cells differ from the in-step shear;
   - under `jit` with the state traced: 1.641865204757312e-07, equal to the in-step shear in every cell.

   So a compile rewrite inside the builder is the cause, and it is not an FMA.
4. **The rewrite.** NEMO forms `pr3u = 0.5*(e1e2t*ssh + e1e2t*ssh) * r1_hu_0 * r1_e1e2u`, left to right, with
   `r1_e1e2u = 1/e1e2u` stored. The card wrote `num*r1_hu0/a_u`. Both factors are constants. XLA turns the
   division into a reciprocal multiply and then folds the two constants together:
   `num*(r1_hu_0*(1/e1e2u))`. In the offline shear replay, that folded form gives the card's compiled value in
   all 6 cells checked (including the two at (34,120)). NEMO's order gives NEMO's value.

**Classification: association defect** (written association that lets constants fold). This is a transcription
fix, not a compiler effect beyond reach.

## 4. Landed (`9a7487ac3`)

`TKEConfig.nemo_sh2_r3_stored_reciprocal` (V1-selected, default False) builds the ratio in NEMO's form:
- each `e1e2t*ssh` product rounded before the face sum;
- `num*r1_hu_0` rounded;
- then multiplied by `1/e1e2u`.

It refuses the in-solve shear and every non-qco metric source (the round-9/10 pattern). Round 10's guard
(round the ratio before the "1 +") stays.

The product rounding was added because, in the unit toy, an FMA also forms in the numerator. The global FMA
switch takes the toy from 31 unequal cells to 0. In the full V1 step this has not been measured separately.

- Unit test: the builder under `jit` with traced sea level equals a numpy NEMO-order replay of
  `domqco.f90:266-269` + `zdfsh2.f90:104-108`, bit for bit. Without the flag, 77 of 2,460 face values differ (test
  pinned). Plants: removing the `num*r1_hu_0` rounding, or the product rounding, each gives `1 failed, 3 passed`.
- After the fix, one step from NEMO's kt-4 state: **every field and level 0 unequal at kt 5**, in all three
  arms (plain, self, NEMO's `avt`).
- Flag-off one-step jaxprs, full ORCA2 (138,748 lines) and DINO (123,236 lines), base `d31991655` vs `9a7487ac3`:
  **identical** (P4 CONFIRMED).

## 5. Both labels, kt 1..10 (re-scored at `9a7487ac3`)

- **INDEPENDENT kt 1-10:** every surface row (sea level, barotropic transports, u, v, T, S) and every level row
  (u, v, T, S, en, avm, avt, dissl, surface avm): max 0, unequal 0. Round 10 had 18 rows over the bar at kt 6-10.
- **GIVEN-NEMO-ENTRY (kt 1/3/5/7/9, one-step and two-step):** every row max 0, unequal 0. Round 10 had two-step
  kt-6 T 1.21e-15 normalised in 2 cells.
- No flushed-subnormal cells reported (D127).

**P5 CONFIRMED: V1 is CERTIFIED**, bit-for-bit, under both labels.

## 6. Decision 122 (ORCA2_OMIP_L4)

`domqco.f90` (`9a3b302cd3f8868f`), `domhgr.f90` (`84e2c258c0f7c8a2`), `zdfsh2.f90` (`e8bb2398700d0d3d`),
`zdfphy.f90` (`903e0d1220a2f9b3`) and `zdftke.f90` (`0f290190a33e4713`) are byte-identical in the V1 build
(`ORCA2V1_OMIP_L4_R1FULLENTRY`) and `ORCA2_OMIP_L4`. `stprk3.f90` differs (the `zdf_phy` call is at :185 in V1).

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| `pr3u`/`pr3v` = 0.5*(e1e2t*ssh + ...) * r1_hu_0 * r1_e1e2u | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-269` | `dom_qco_r3c_RK3` (:237), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:167` in `stp_RK3_stg` (:78) |
| `r1_e1e2u = 1/e1e2u` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:170` | `dom_hgr` (:58) |
| `e3w_1d*(1+r3u)` in the shear | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:98-108` | `zdf_sh2` (:56) |
| `CALL zdf_sh2( Kbb, Kmm, avm_k, sh2 )` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:318` | `zdf_phy` (:296) |
| `CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179` | `stp_RK3` (:62) |
| `zri`, `p_pdlr`; `p_avt = MAX(p_pdlr*p_avt, ...)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:397-401`, `:405`, `:717` | `tke_tke` (:199), `tke_avn` (:531) |

The full ORCA2 card runs this statement too. It stays off there (D130) until ORCA2 measures and selects it
(D96/D122). Pointer only: the full ORCA2 card was not measured this round.

## 7. Tests (synchronous, CPU, fp64, `land.sh` PYTHONPATH)

- Default receipt citation gate after re-anchoring: 17 passed.
- Receipt citation gate on this receipt: section 8.
- Focused battery: section 8.

## 8. Gate lines

(filled before the final commit)

## Choices made

ASKED (this round's brief): commit the one-step-from-NEMO probe; walk (34,120) in compiled order; land the
association fix V1-selected; re-score both labels.

UNASKED, offered for revert:
- a NEW flag instead of widening round 10's `nemo_sh2_r3_source_rounded` (so each statement stays separately
  selectable for the ORCA2 import);
- the `e1e2t*ssh` product rounding rides the same flag. It was measured in the unit toy, not separately in the
  full step;
- the offline shear/Prandtl replays are evidence-directory scripts (`replay_sh2.py`, `replay_prandtl.py`), not
  committed. The committed probe's op-by-op vs jit split and the landed unit test carry the claim.

UNVERIFIED:
- whether the numerator FMA guard is needed in the full V1 step (the global FMA switch moved nothing there
  before the fix);
- `jnp.nextafter(x, x)` as an opaque rounding boundary rests on rounds 6-10's use, and on this round's plants.
