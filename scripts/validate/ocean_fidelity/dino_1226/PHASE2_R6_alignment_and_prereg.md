# PHASE 2 — alignment row R6, built as an option, and the pre-registered A/B

Written and committed BEFORE either arm ran.  Nothing above the RESULT heading
is edited afterwards; git history proves the order.

NEMO paths are `cfgs/DINO/MY_SRC` (the build that produced every oracle number
in this campaign).  `src/OCE` differs and its line numbers do not apply.

---

## The ordered alignment rows

### R6-a — NEMO SITE 1, the NOW-level advective-velocity swap

    dynspg_ts.F90:1171-1173, inside dyn_spg (stpmlf.F90:332)

    puu(:,:,jk,Kmm) = ( puu(:,:,jk,Kmm) + un_adv(:,:)*r1_hu(:,:,Kmm)
       &              - puu_b(:,:,Kmm) ) * umask(:,:,jk)

| | |
|---|---|
| thickness | `hu(Kmm)` — the NOW level |
| installed mean | `un_adv/hu(Kmm)`, the SECONDARY transport-weighted substep average (accumulated `:736`, normalised `:999`) |
| removed mean | `puu_b(:,:,Kmm)`, the stored NOW barotropic velocity |
| lifetime | WITHIN-STEP.  Consumed by `tra_adv` (`stpmlf.F90:528`) and the rest of the now-level block, then UNDONE at `stpmlf.F90:787-790` (the `.NOT.ln_bt_fw` branch; `ln_bt_fw=.false.`, `RUN_90D_TWIN/namelist_cfg:353`) before `dyn_atf_qco` (`:613`).  Never reaches the committed state. |

**legoESM under the option: NOT BUILT, deliberately.**  This is an advecting-
velocity convention for the within-step advection, not a state update, and
building it honestly means building its removal too — two changes, and a
different question from the one this option asks.  Which substep average is
installed is the separate `barotropic_reconcile_target` lane, already tested
and REFUTED as a lever on the 90-day gate (`PREREG_gate90_reconcile_ab.md`).

### R6-b — NEMO SITE 2, the AFTER-level committed reconciliation

    stpmlf.F90:754-765, in mlf_baro_corr, called at :578
    — AFTER dyn_zdf (:396) and BEFORE dyn_atf_qco (:613)

    zue(ji,jj) = SUM_k e3u(ji,jj,jk,Kaa) * puu(ji,jj,jk,Kaa) * umask(ji,jj,jk)
    puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - zue(ji,jj)*r1_hu(ji,jj,Kaa)
       &                + uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)

| | |
|---|---|
| thickness | `e3u(Kaa)` in the sum AND `hu(Kaa)` in the divisor — the AFTER level, both |
| installed mean | `uu_b(:,:,Kaa)`.  DINO sets `ln_dynadv_vec=.TRUE.` (`namelist_cfg:321`), so `dynspg_ts.F90:978-980` accumulates VELOCITIES (`za1*ua_e`) and `:1001` divides by `r1_wgt1s`.  `puu_b(Kaa)` is therefore ALREADY a velocity, and the transport-to-velocity division at `:1164` does NOT run — it sits in the `ELSE` branch (`:1139-1166`) that `:1124`'s `ln_dynadv_vec` test skips.  So `uu_b(Kaa)` is the PRIMARY velocity-weighted boxcar average, exactly. |
| removed mean | the column's OWN after-thickness mean, RECOMPUTED here — not a stored field |
| what it accomplishes | it annihilates whatever column mean `dyn_zdf` deposited.  Measured on the 90-day DINO twin: `+17.916 m3/s2` per southern u-row, 12x the realized spin-up rate, discarded every step |

**legoESM under the option:** `barotropic_common.after_level_column_mean_reconcile`,
called from BOTH outer-step paths (`_leapfrog_step`, `_nemo_mlf_step`) at NEMO's
position — after `_apply_implicit_vertical_mixing`, before the conservation
fixer and the Asselin filter.

| | |
|---|---|
| thickness | `min_cell_to_uface(compute_layer_thickness(naa.eta, ...))` — `naa.eta` is the barotropic solve's after ssh, the same level `e3t_aft` is built from |
| installed mean | `btu_exp`/`btv_exp`, the depth mean the barotropic solve produced, i.e. whichever average `barotropic_reconcile_target` selected.  Under `"velocity_avg"` that IS `uu_b(Kaa)` and this row is NEMO's line verbatim; under the card's `"transport_avg"` it is the secondary average in NEMO's slot. |
| removed mean | the column's own after-thickness mean, recomputed — same as NEMO |

The two config fields are ORTHOGONAL on purpose: one picks WHICH average, the
other picks WHERE and at which time level's thickness it is enforced.

### R6-c — legoESM WITHOUT the option, for contrast

ONE reconciliation, inside the barotropic solve (the shared `_reconcile_targets`
helper, both entry points), at the NOW thickness; the leap-frog combine then
pins the after-level velocity's NOW-thickness column mean to `btu_exp`.  Nothing
runs after the implicit vertical solve.

### R6-d — the two halves of R6 do NOT carry the same weight

**(i) The THICKNESS half (now -> after) is STRUCTURALLY INERT under pure
z-star.**  Every layer of a column rescales by the same `(H+eta)/H`, and the
min-rule face depth inherits that factor, so a thickness-weighted column mean is
IDENTICAL at the two levels.  Measured `1.1e-16 m/s` on a flat-bottom z-star
channel and gated by `tests/ocean/unit/test_barotropic_after_reconcile.py::
test_inert_on_pure_zstar_without_partial_cells`.  It is live only through
partial cells — which the DINO card does have.

**(ii) The SECOND-SITE half is already MOSTLY done, from the other side.**
`zdf_baroclinic_only=True` (on for this card) strips the depth mean BEFORE the
implicit solve and re-adds it unchanged after, which achieves NEMO's discard
without a post-solve corrector.  What survives is the difference between that
strip/re-add weighting (the solve's `dz_u`, floored at `min_water_column_m`) and
the combine's own (`h_u`, floor `1e-10`).  On the valid-stack 90-day twin that
residue is the retained `ZDF bt = +0.296 m3/s2` per southern u-row.

So R6 is a REAL fidelity row and a SMALL one, and it was named as the last
untested structural DIFF before either of those two facts was measured.

---

## The pre-registered A/B

### Arms — one variable

| | arm A (baseline) | arm B (R6 option) |
|---|---|---|
| `barotropic_after_reconcile` | `"off"` (card default) | `"nemo_mlf_baro_corr"` |
| selected by | no env var | `DINO_AFTER_RECONCILE=nemo_mlf_baro_corr` |
| GPU | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` |

Held identical and stamped in every log: HEAD (this commit), clean tracked
tree, recipe `nemo_dino_kamm_mlf`, `--bridge-before`, IC = NEMO
`DINO_00005760_restart.nc` (day 180), `LEGOESM_NEMO_E3T=both`, corrected
seasonal clock (`t0 = 15552000 s` from the restart's own `adatrj`), fp64
control dtype via `run_fp64.py`, dt 2700 s, 32 steps/day, 90 days.
`barotropic_reconcile_target` stays at the card's `transport_avg` on BOTH arms —
flipping it too would be two variables.

### The three numbers, and where each comes from

1. **southern-band realized spin-up rate**, band mean over u-rows 1..13,
   90-day, `[m3/s2 per u-row]` — `southern_term_torque_accum.py`, compared
   against NEMO's `+1.430` from `nemo_accum_torque.py`.
2. **day-90 south-group transport deficit** `[Sv]` and the fraction of NEMO's
   `+3.097 Sv` spin-up captured — `southern_circulation_budget.py`.
3. **the five-metric acceptance gate** at 5x the noise floor, including the
   full-section ACC gap — `acceptance_gate_90d.py`, unmodified, with its own
   two fatal self-checks (NEMO y10 ACC 121.07 Sv; band volume 2.694775e16 m3).

Noise floors (#1492 2.1 micro-ensemble, n=3): ACC **0.091 Sv**, upper contrast
1.1e-4, deep contrast 4.5e-5, southern surface sigma max/mean 9.5e-5 kg/m3.

### Valid-stack arm-A values already measured (this branch, HEAD's parent)

    band realized dR/dt      +1.069   (NEMO +1.430, gap +0.361)
    stage row ZDF bt         +0.296   <- what the option removes
    day-90 south-group gap   -0.772 Sv, 75% of NEMO's spin-up captured
    full-section ACC gap     -0.5965 Sv   (channel band +0.2874)

### PREDICTION, quantitative and falsifiable

Turning the option ON removes legoESM's retained `ZDF bt` row from the realized
rate and re-pins the after-level column mean at the after thickness.  The
thickness half is a partial-cell-only effect and is expected to be small.  So:

* **Expected**: the band realized rate FALLS by roughly **0.30** (from `+1.069`
  toward `+0.77`), and the NEMO-minus-lego band gap therefore GROWS from
  `+0.361` to roughly **`+0.66`** — the WRONG direction for the deficit.
* **CONFIRMS R6 owns the deficit**: the band gap closes by more than 50%
  (`+0.361 -> < +0.18`) AND the full-section ACC gap shrinks toward the
  `0.091 Sv` floor.
* **REFUTES, null**: every metric moves by less than its own noise floor.
* **REFUTES, adverse** (what is expected): the band gap GROWS beyond the
  window scatter (`sd 0.225` over the nine matched windows).

A refutation is a real outcome, not a failed run.  R6 is being tested because
it was the last named untested structural DIFF, and R6-d already bounds how
much of the deficit it could plausibly carry.

### Also recorded whatever the verdict

The full five-metric gate table on BOTH arms, each arm's day-0 gate line,
`tau_x` range, resolved ladder and clock, and the stability tail.  **A density
metric degrading past its floor on arm B is a Rule-8 finding to be reported,
not a reason to revert the option** — no gate constant and no card default is
edited by this test.

### Implementation gates already passed

`tests/ocean/unit/test_barotropic_after_reconcile.py`: 23 passed.  Proven
NON-VACUOUS by reverting the model-side insertion — the four tests that gate it
(behaviour on both step paths, dispatch raise on both step paths) then FAIL
while the 19 kernel/config tests still pass.
`tests/test_dispatch_hardening.py`: 24 passed, with the new guard pinned in the
grow-only baseline.

---
