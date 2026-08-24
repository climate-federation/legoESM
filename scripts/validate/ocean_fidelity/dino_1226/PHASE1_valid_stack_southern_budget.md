# PHASE 1 — the southern-basin spin-up budget, RE-MEASURED on the valid stack

Every number below was produced at HEAD `ebddfaa06`, clean tracked tree
(`PROVENANCE: dirty_tracked_files=0` in every log), `CUDA_VISIBLE_DEVICES=0/1`,
`JAX_ENABLE_X64=1`, fp64 control dtype printed by `run_fp64.py` in every log,
`LEGOESM_NEMO_E3T=both` (NEMO's own 3-D thickness AND T-depth ladders), and the
corrected seasonal clock (`t0 = 15552000 s`, read from the restart's own
`adatrj`, stamped in every log).

## What changed between the quarantined runs and these

THREE variables moved at once, as a BUNDLE.  Nothing below attributes anything
to one of them individually, and no old-vs-new difference here is a
one-variable result:

1. seasonal forcing clock: relative (antiphase with NEMO) -> absolute (`116c23c02`)
2. precision policy: fp32 control dtype -> fp64, forced and stamped (`245effa20`, `ebddfaa06`)
3. vertical geometry: NEMO's analytic 1-D `e3t_1d` ladder -> NEMO's own 3-D
   `e3t_0` + `gdept` ladders (`c3584c0b7`)

## Band and row definitions — IDENTICAL to the quarantined runs

Unchanged code, quoted so the comparison is on the same reducer:

* band = T-rows `1..13`, latitudes `-69.5 .. -64.9`
  (`southern_circulation_budget.py:143`, `ROWS = range(1, J0)`; row 0 is dry and
  that is asserted, not assumed, by control C5 — measured 0 wet u-cells).
* row circulation `R(j) = sum_i e1u(i,j) * sum_k e3u_0 * u * umask`
  (`southern_circulation_budget.row_circulation`, :169).  The reducer's weights
  are NEMO `mesh_mask` arrays and did not change.
* the accumulated-torque probe uses the SAME reducer
  (`nemo_accum_torque.row_int_3d` delegates to `B.row_int_trend`), verified at
  run time: device-vs-recorded reducer band max |diff| `3.6e-12 m3/s2`.

## Instrument controls that passed before any number below was read

`southern_term_torque_accum.py`: W gate (surface residual at k>0 exactly 0;
row integral vs analytic wind torque, band max rel `3.7e-04`), reducer gate
`3.6e-12`, S4 instrumented-vs-production step `1.11e-16 m/s` against
`max|u| = 0.744` (0.5x fp64 eps), S5 geometry gate `0.000e+00`, stage capture
exactly 1 implicit-vmix entry per step, plant `0`.

`southern_circulation_budget.py`: C1 dtype (all geometry float64), C2 no
non-finite wet cells on either side on any day, C3 day-0 torque identity at
1.0x the fp32 snapshot quantum on every term, C4 NEMO's committed +3.10 Sv
south-group spin-up reproduced (+3.097), C5 geometry, P1a/P1b/P2/P3 planted
controls, control V v-row offset decided by the day-0 identity (`2.97e-08` vs
`4.61e-01`).

`nemo_accum_torque.py`: T0 identities I1/I2/I3 at `~3.3e-10` absolute,
`~3.0e-14` relative to operand, against a tolerance of `5.12e-12` relative, on
a cancellation scale of `1.114e+04 m3/s2/row`.

NEMO's own accumulated side (`RUN_ACC90`) STILL LOADS and was not re-run: the
final dump carries the full 2880 steps, `rn_acc_plant = 0`, and all three
identities pass.  Its band realized rate is `+1.430 m3/s2/row`.

## THE BAND TABLE — quarantined alongside valid, both labelled

Per-row band mean, 90-day, `[m3/s2 per u-row]`.  The three left columns are
WITHDRAWN and are printed only so the shape of the revision is visible.

| stage row | QUARANTINED `off/transport_avg` | QUARANTINED `off/velocity_avg` | QUARANTINED `trueT/transport_avg` | **VALID `both/transport_avg`** |
|---|---|---|---|---|
| BARO solve | +0.578 | +0.707 | +0.368 | **+0.773** |
| BCLIN expl+diss | -2.129 | -2.148 | -2.130 | **+0.000** |
| ZDF bt | +0.339 | +0.319 | +0.324 | **+0.296** |
| ZDF bc | +2.098 | +2.118 | +2.101 | **-0.000** |
| POST fixer | 0.000 | 0.000 | 0.000 | **0.000** |
| = realized dR/dt | +0.886 | +0.996 | +0.663 | **+1.069** |
| NEMO realized | +1.430 | +1.430 | +1.430 | **+1.430** |
| **NEMO - lego** | +0.544 | +0.434 | +0.767 | **+0.361** |

The `+-2.1` BCLIN/ZDF-bc pair is GONE.  It was a ladder mismatch, not physics:
the reducer weights with NEMO's `e3u_0` while the model integrated on the
analytic 1-D ladder, so the combine's "zero-depth-mean" baroclinic pieces did
not reduce to zero under the reducer.  The two entries always cancelled to
`-0.031`, so no earlier conclusion rested on them; they are now measured at
`0.000` and `-0.000`.

## THE SAME QUANTITY FROM THE OTHER INSTRUMENT

`southern_circulation_budget.py` reaches the deficit from the STATES ALONE (no
torque model, no discretization choice):

| quantity | QUARANTINED | **VALID** |
|---|---|---|
| day-90 south-group transport deficit [Sv] | -1.320 / -1.297 / -1.483 (three arms) | **-0.772** |
| fraction of NEMO's spin-up captured | 52-58% | **75%** |
| band-summed 90-d change in R [1e6 m3/s] | (not recorded per arm) | NEMO +144.51 vs lego **+108.00** |
| per-row deficit, band mean [m3/s2] | -0.61 (the campaign's headline) | **-0.361** |

The two instruments agree to `0.001 m3/s2` on the band-mean deficit
(`-0.361` from states, `+0.361` as `NEMO - lego` from the accumulated stage
rows), and the accumulated stage sum matches legoESM's own trajectory drift to
`0.02%` (per-row max `0.0010`), so the cross-model pairing is like-for-like.

CONTEXT the deficit has to live inside: on this same stack the whole-model
day-90 ACC gap is `-0.5965 Sv` while the re-entrant channel band is `+0.2874 Sv`
(`PREREG_gate90_ladder_promotion.md`), implying roughly `-0.9 Sv` outside the
channel.  The southern band measured here carries `-0.772 Sv` of that.

## PER-ROW SHAPE — single-signed on all 13 rows

| row | lat | NEMO dR/dt | lego dR/dt | NEMO - lego |
|---|---|---|---|---|
| 1 | -69.5 | +0.124 | -0.555 | +0.680 |
| 2 | -69.2 | +0.047 | -0.715 | +0.763 |
| 3 | -68.8 | -0.069 | -0.934 | +0.865 |
| 4 | -68.4 | -0.109 | -0.670 | +0.561 |
| 5 | -68.1 | +0.124 | -0.386 | +0.510 |
| 6 | -67.7 | +0.630 | +0.227 | +0.404 |
| 7 | -67.3 | +1.262 | +0.985 | +0.277 |
| 8 | -66.9 | +2.034 | +1.834 | +0.200 |
| 9 | -66.5 | +2.919 | +2.785 | +0.134 |
| 10 | -66.1 | +3.396 | +3.317 | +0.078 |
| 11 | -65.7 | +3.137 | +3.054 | +0.083 |
| 12 | -65.3 | +2.739 | +2.671 | +0.068 |
| 13 | -64.9 | +2.348 | +2.280 | +0.068 |
| band | | +1.430 | +1.069 | **+0.361** (sd 0.277, 13/13 same sign) |

The deficit is now CONCENTRATED, not flat: rows 1-5 (against the southern wall)
carry `-3.376` of it and rows 6-13 carry `-1.319`.  Proportionality to NEMO's
own rate is refuted (`deficit / NEMO-rate` spans `-1588% .. +1234%`), and so is
proportionality to the wind.

## PER-WINDOW GAP CURVE — matched 10-day windows, both sides

| window [d] | NEMO | legoESM | NEMO - lego |
|---|---|---|---|
| 0-10 | +1.322 | +1.099 | +0.223 |
| 10-20 | +1.501 | +0.814 | +0.687 |
| 20-30 | +2.024 | +1.224 | +0.800 |
| 30-40 | +2.110 | +1.643 | +0.467 |
| 40-50 | +1.493 | +1.233 | +0.260 |
| 50-60 | +1.255 | +1.006 | +0.249 |
| 60-70 | +1.162 | +1.054 | +0.108 |
| 70-80 | +1.110 | +0.853 | +0.257 |
| 80-90 | +0.888 | +0.690 | +0.198 |
| mean | | | **+0.361**, sd 0.225, range 0.692, 9/9 positive |

The gap is single-signed in every window but is NOT flat in time: it peaks at
`+0.800` in days 20-30 and falls to `+0.108` by days 60-70.  A constant
per-step offset is therefore not the shape (the probe's own flatness criterion,
sd < 0.25 x |mean|, is not met).

## WHICH STAGE ROW WEARS THE DEFICIT NOW

The BAROTROPIC-SOLVE row, as before — but the arithmetic is different and the
old size is withdrawn.

* NEMO's realized row IS its barotropic solve, exactly: identity I3 holds to
  `3.3e-10` on a `1.1e+04` cancellation scale, because `mlf_baro_corr`
  (`stpmlf.F90:754-765`) discards the implicit vertical solve's column mean —
  measured at `+17.9 m3/s2/row`, i.e. 12x the realized rate, thrown away every
  step.
* legoESM's realized row is `BARO +0.773` PLUS a retained `ZDF bt +0.296`.
  legoESM has no post-solve corrector; it emulates the discard from the other
  side, by stripping the depth mean BEFORE the implicit solve and re-adding it
  after (`zdf_baroclinic_only=True`, on for this card).  The `+0.296` residue
  is what survives that route.
* So the barotropic solve's own shortfall is `0.773 - 1.430 = -0.657`, of which
  the retained `+0.296` masks 45%.

INDEPENDENT NEMO-side estimate of that shortfall, for scale only:
`-0.951 +- 0.211 m3/s2/row` (was `-1.146 +- 0.358` on the withdrawn stack).
It stays **PLAUSIBLE** and may not be promoted: NEMO's barotropic net is
recovered as `utrd_spg + (pre-spg trends)`, a 595:1 cancellation whose
SYSTEMATIC common-mode bound is of order `1 m3/s2` and does not average down
over windows.  The `+-` figure bounds random scatter only.

## RETRACTIONS

**RETRACTED — "legoESM captures only 52-58% of NEMO's southern spin-up."**  On
the valid stack it captures **75%**, and the day-90 south-group deficit is
`-0.772 Sv`, not `-1.30 .. -1.48 Sv`.

**RETRACTED — the `-0.61 m3/s2/row` band deficit, and every threshold sized
against it** (including `PRIOR_DEFICIT` in `nemo_accum_torque.py` and the
"18% of the deficit" reading of the reconcile-target flip).  The valid-stack
value is `-0.361`.

**RETRACTED — the `-2.129 / +2.098` BCLIN and ZDF-bc stage rows.**  They were a
reducer-vs-model ladder mismatch and measure `0.000` once both sides integrate
on NEMO's own ladder.  Nothing was concluded from them; they are withdrawn so
nothing is.

**NOT retracted:** which row wears it.  The barotropic solve still does.

## Artifacts

    /tmp/dino_valid/accum_both.npz          southern_term_torque_accum, 90 d, 10-d intervals
    /tmp/dino_valid/budget_valid.npz        southern_circulation_budget, arm validstack_both
    /tmp/dino_valid/nemo_vs_lego_valid.npz  nemo_accum_torque --table, per-window arrays
    /tmp/dino_valid/twin90_valid.npz        the 90-day twin the budget probe reads

These live outside the repo and are not tracked; every one is regenerated by
the command recorded in its own log next to it.
