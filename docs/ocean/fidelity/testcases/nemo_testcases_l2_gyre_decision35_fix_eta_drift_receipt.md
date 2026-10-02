# NEMO testcase L2 GYRE — decision 35 receipt: the card drops `fix_eta_drift`

Date: 2026-09-11. Before-arm base `3a8d94ea1985` (round 54, detached worktree
`/tmp/gyre-acq-r54`); after-arm `8590f9af002e` (this change, committed before
it was measured — the stamp is fail-closed and refused the first self-check on
the dirty tree). CPU, production JIT, fp64/scalar-libm. Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_decision35_fix_eta_drift.md`
(written before the after-arm ran). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decision35_fix_eta_drift/`.

Provenance note: the after-arm was measured at `8590f9af002e` (this change on
the round-54 base, the one-variable comparison); that commit was then rebased
onto round 55 and is `5be840acf2be` on the branch. The evidence files carry
the pre-rebase stamp. Round 55's TKE-floor change was NOT in the measured
tree, so its effect is attributed separately by that round.

## 1. The decision (ASKED)

The certified GYRE NEMO-identity card applied `fix_eta_drift=True`: a global
uniform eta shift sized by an area-weighted volume residual
(`ocean_model_latlon_cgrid.py:6686-6751`). The user was asked (decision 35)
whether to keep a global correction NEMO does not have; answer, 2026-09-11:
**turn it off — "no hidden extras; a fixer could hide model errors."**

## 2. What NEMO does (compiled `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo`)

- `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:137`: `pssh(ji,jj,Kaa) = ( pssh(ji,jj,Kbb) - rDt * ( r1_rho0 *
  emp(ji,jj) + zhdiv(ji,jj) ) ) * ssmask(ji,jj)` — emp enters LOCALLY.
- `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:553`: `ssha_e(ji,jj) = ( sshn_e(ji,jj) - rDt_e * (
  ssh_frc(ji,jj) + zhdiv ) ) * ssmask(ji,jj)` — the barotropic substep
  carries the source locally; the baroclinic step receives the filter average
  of that trajectory. No global sum exists in NEMO's free-surface path.
- `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/usrdef_sbc.f90:151-157`: NEMO de-means the card's own emp
  (`emp(ji,jj) = emp(ji,jj) - zsumemp * tmask(ji,jj,1)`), so the area-mean
  source is zero by construction (`5.5e-22` kg/m2/s, card reconciliation
  receipt §9.1): on this card the projection had nothing to correct.

## 3. What changed (commit `8590f9af002e`)

| file | change |
|---|---|
| `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` (GYRE branch only) | `fix_eta_drift=True -> False`; `freshwater_closure="real_freshwater"` stays — the `none` arm KILLS the E-P channel (`3.95e-03` K after two steps, reconciliation receipt §7), so it is not an alternative |
| `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` | the #1484 guard that forces `fix_eta_drift=True` under `real_freshwater` is exempted ONLY when `barotropic_continuity_evaluation="nemo_literal"` — the statement that makes the source enter eta the way NEMO's `GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:553` does. The generic lane keeps the guard (its 55 %-delivery claim was measured on that lane). A relaxation admits one new combination and changes no arithmetic of any configuration that constructs today |
| `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_phase3_gate.py` | the ladder's `_replace(freshwater_closure=..., fix_eta_drift=True)` override is gone; the gate certifies the card's own program (one card, one program) |
| `tests/ocean/unit/test_real_freshwater_closure.py` | admission test for the exemption; **non-vacuous: with the exemption reverted, `1 failed`; with it, `16 passed`** (`unit_tests.log`) |
| `tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py` | the card's two rows pinned (`real_freshwater`, `False`); the program-drift plant now expects the closure row alone (the legacy program shares the card's `fix_eta_drift`); `8 passed` |

Gate self-check on the committed tree: all seven plants REFUSED
(`reconciliation_selfcheck2.log`).

## 4. Ladder kt=1..10, scalar-math v2 roots, before vs after

The before-arm reproduces the round-53 certified record exactly
(`before_kt1_10.json` vs `round53/default_roots_gyre_kt1_10_final.json`:
0 rows differ) — the instrument is the certified one.

| prediction | measured | verdict |
|---|---|---|
| `kt2.before.ssh` `4.336808689942018e-19 -> 0.0`, 0 cells unequal | `4.3368086899420177e-19 -> 0.0`, `600 -> 0` unequal | **CONFIRMED** — the fixer's shift was the entire kt=2 ssh residual; legoESM's end-of-step-1 eta is now bit-identical to NEMO's |
| `kt2.before.u/v` unchanged bit-for-bit | `2.7478404751243857e-12` / `3.3055603068134209e-12`, `17399`/`17100` unequal, identical | **CONFIRMED** |
| `kt2.before.T/S` unchanged | `6.0543576871612622e-16` / `5.7863742516519688e-16`, identical | **CONFIRMED** |
| kt>=3 rows move within their DEBT; no AT-BAR row leaves the bar; first-over-bar stays `kt2 u/v` | 32 rows at kt>=3 moved, largest relative move `9.4e-09` (`kt6.before.ssh` `3.2648728680723105e-06 -> 3.2648728986450770e-06`), every one already DEBT; AT-BAR rows leaving the bar: none; first-over-bar `kt2 u/v` before and after; barotropic first-over-bar `kt2 uu_b/vv_b` before and after | **CONFIRMED** |

Registered rows (Rule 12): `kt2.before.ssh` improved to exact; 32 DEBT rows at
kt=3..10 moved by <= `9.4e-09` relative in either direction (full list:
`after_kt1_10.json` against `before_kt1_10.json`). The step-2 tracer rows are
untouched (`kt3.before.T/S` `3.7223442430e-04` / `3.6832457463e-05` normalised
before and after): the fixer was never part of the round-54 owner.

## 5. Days 1–30 against `year_owners/nemo_seed0`

30-day members from rest (`--snap-steps 6`, daily), scored by
`nemo_testcase_l2_gyre_year_owners.py --day-gap` (`before_day_gap.json`,
`after_day_gap.json`).

| day | 3-D T rms gap, fixer ON | fixer OFF | relative change |
|---:|---:|---:|---:|
| 1 | `2.6994482e-03` | `2.6994482e-03` | `6.9e-12` |
| 5 | `3.8895058e-03` | `3.8895058e-03` | `1.0e-10` |
| 10 | `5.3766814e-03` | `5.3766814e-03` | `6.1e-10` |
| 30 | `1.4241019e-02` | `1.4241019e-02` | `-8.4e-11` |

Worst relative change over all 30 days and 8 fields: `3.5e-08` (day 24,
S rms `6.704246989929467e-04 -> 6.704246752900293e-04`). Prediction (equal to
>= 4 significant digits) **CONFIRMED**. The month is unchanged: the fixer was
inert on this card, as §2 says it had to be.

## 6. Rule 12 per card

| card | resolved pair (printed from `build_nemo_testcase_card`) | result |
|---|---|---|
| GYRE-zco | `real_freshwater`, `fix_eta_drift=False`, `nemo_literal` continuity | measured, §4–5 |
| LOCK_EXCHANGE-zco | `virtual_salt_flux`, `fix_eta_drift=False`, `nemo_literal` | never constructs the admitted pair; the card edit is inside the GYRE branch; the guard change admits a combination and executes no arithmetic — not re-run, 0 rows by construction |
| OVERFLOW-zps | `virtual_salt_flux`, `fix_eta_drift=False`, `nemo_literal` | same |
| DINO (separate branch) | shares the guard file only | shared-statement risk = a guard relaxation; 0 rows by construction |
| ORCA2 | — | UNMEASURED-with-spec |

## 7. ASKED / UNASKED

| # | choice | status |
|---|---|---|
| 1 | `fix_eta_drift=False` on the GYRE NEMO-identity card | **ASKED** (decision 35) |
| 2 | the guard exemption is keyed on `barotropic_continuity_evaluation="nemo_literal"` rather than on a new dedicated field | **UNASKED — mechanism choice**, offered for revert: it reuses the field that already means "the continuity statement is NEMO's", so no new knob; a dedicated field would be a knob whose default keeps the guard |
| 3 | the ladder gate's override removed | consequence of "one card, one program" (the override was left in place at unification only because that file belonged to another lane); no configuration selected |
| 4 | reconciliation drift-plant expectation `['freshwater_closure']` | mechanical: the legacy program now shares the card's `fix_eta_drift` |
| 5 | 30-day members run with `--snap-steps 6` | measurement protocol matching the `year_owners` daily record, not a model choice |

## 8. Open

- The generic lane's guard still asserts a 55 % delivery measured at n=20 with
  the cosine filter; whether NEMO's own AB3-AM4 filter at n=50 delivers the
  full source over successive steps is a question for a card with a non-zero
  net freshwater source (DINO), not this one.
- The step-2 tracer owner (round 54: the TKE closure's avt at kt=2) is
  unchanged by this decision and remains first.
