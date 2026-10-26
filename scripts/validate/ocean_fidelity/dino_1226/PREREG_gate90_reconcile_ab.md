# PRE-REGISTRATION — 90-day twin acceptance gate, barotropic-reconcile A/B

Written and committed BEFORE either arm ran. Nothing below is edited after the
runs; the result goes in a separate commit.

## The question

The #1455 M2 shape test (commit `e31c9e99b`) found that flipping
`barotropic_reconcile_target` from the card's `transport_avg` to `velocity_avg`
removes 77% of the window-to-window variance of the 90-day southern-band torque
gap against NEMO (verdict LIKELY, n=1 per arm), and the band-rate A/B
(`bbfdb8e5d`) put 18% of the southern spin-up rate on the same flip. Neither is
a whole-model climate metric. This is the payoff test: does the flip move the
**headline day-90 ACC deficit** on the acceptance gate.

M2's alignment row R4 is the reason `velocity_avg` is the candidate-faithful
arm, quoted verbatim from `e31c9e99b`:

> R4  BAROTROPIC VELOCITY, AFTER LEVEL             MATCH only under velocity_avg
>     NEMO   puu_b(:,:,Kaa) += wgtbtp1*ua_e :979 (ln_dynadv_vec => velocities,
>            not transports), /r1_wgt1s :1001. [...]
>     The card RESOLVES to transport_avg (verified at runtime, not from a
>     label), so on the card this row is a DIFF.

R6 of that same table records what this A/B canNOT fix: legoESM has ONE
reconciliation site where NEMO has two, and the thickness/divisor mismatch
**survives both arms**. So a null here does not exonerate the reconcile term as
a whole — it bounds this one choice.

## Arms (one variable)

| | arm A (baseline) | arm B (candidate-faithful) |
|---|---|---|
| `barotropic_reconcile_target` | `transport_avg` (the card's resolved value, printed at runtime) | `velocity_avg` |
| selected by | card default, no env var | `DINO_RECONCILE_TARGET=velocity_avg` |
| GPU | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` |

Held identical: git SHA `b06186dea`; recipe `nemo_dino_kamm_mlf`;
`--bridge-before`; IC = NEMO `DINO_00005760_restart.nc` (day 180) from
`RUN_STEPDUMP`; `LEGOESM_NEMO_E3T=off`; fp64 via `run_fp64.py`
(`JAX_ENABLE_X64=1` alone is not enough — the grid would still build fp32);
dt 2700 s, 32 steps/day, 90 days; surface forcing on (both arms print
`tau_x[Pa] min/max` and the day-0 gate next to it).

`LEGOESM_NEMO_E3T=off` is the 1-D thickness ladder — deliberately NOT NEMO's
true `e3t_0`. The true ladder (`both`) is a known unfixed instability from a
NEMO restart (`nemo_state_bridge.py`: max|u| 0.66 -> 2.2 m/s over 20 d), and it
is the ladder every recorded arm of this campaign, M2 included, ran on. Using
`both` here would change two variables at once.

The old gate2 arm npz files are from older SHAs and are NOT used. Arm A is
re-run fresh at this SHA.

## Instrument

`acceptance_gate_90d.py` (unmodified — no gate-constant or metric edits), which
runs its own two fatal self-checks (NEMO y10 ACC = 121.07 Sv through this
harness; band volume = 2.694775e16 m3) before comparing. Baseline: NEMO's own
day-90 continuation, `RUN_90D_TWIN/DINO_00008640_restart_*` (16 tiles, NOW
level). Gate level 5x, and the 1x table recorded too.

Noise floors (#1492 2.1 micro-ensemble, n=3): ACC **0.091 Sv**, upper contrast
1.1e-4, deep contrast 4.5e-5, surface sigma max/mean 9.5e-5 kg/m3.

## Prediction

Headline quantity: `deficit = |ACC_lego(day 90) - ACC_NEMO(day 90)|` on
`acc_thermal_wind.acc_full`, currently ~1.74 Sv, of which 76-85% is the
southern gyre.

- **Expected**: the deficit SHRINKS on arm B by roughly **0.15-0.3 Sv**
  (18% of the southern band rate carried onto the whole-model metric).
- **CONFIRMS the funnel**: deficit shrinks by **> 0.1 Sv**, i.e. beyond the
  0.091 Sv run-to-run floor.
- **REFUTES**: deficit unchanged within +-0.091 Sv, or larger on arm B.

An earlier whole-model ACC test of this same option, in a different measurement
context (`bee6cc5ac` era), moved ACC the WRONG way by 0.048 Sv — below this
floor, and therefore not itself a result either way. A refutation here is a
real and expected outcome, not a failed run.

Also recorded for both arms, whatever the ACC verdict: the full five-metric
gate table (ACC, upper contrast, deep contrast, southern surface sigma max,
southern surface sigma mean) against NEMO, plus each arm's day-0 gate line and
`tau_x` range.

If either arm blows up, the step, day and T-range at blow-up are the finding,
reported next to the active thickness ladder (`off`).

---

# RESULT — REFUTED

Committed separately from everything above; the pre-registration text is
unedited (git history proves the order).

Both arms ran at SHA `b06186dea`, stable to day 90 (`STABLE=True`, 2880 steps,
213 s / 217 s wall). Day-0 gates identical and exact on both
(`max|dT|=max|d_eta|=max|du|=max|dv|=0.000e+00`, `max|u0|=0.7516`,
`max|v0|=0.8595`), before-level bridge exact on both, forcing on and identical
(`tau_x[Pa] min/max = -0.200/0.100`), `LEGOESM_NEMO_E3T=off` verified from each
process's own environment, fp64 control dtype printed by both. Arm B's log
carries the single line `ABLATION: barotropic_reconcile_target=velocity_avg`;
arm A's carries none. One variable.

Gate instrument self-checks passed before every comparison: NEMO y10 ACC
121.07 Sv (recorded 121.07), band volume 2.694775e16 m3 (rel 5.0e-08). The gate
was also run `--self-test`: an identical pair passes at 1x, the synthetic
violation fails all five metrics by >10x their thresholds. Non-vacuous.

## Acceptance gate, both arms vs NEMO day 90 (level 5x)

| metric | arm A `transport_avg` | arm B `velocity_avg` | NEMO d90 |
|---|---|---|---|
| ACC [Sv] | 66.098876 | 66.185965 | 65.369204 |
| upper contrast <1400 m [kg/m3] | -0.286279 | -0.286329 | -0.288182 |
| deep contrast >1400 m [kg/m3] | -0.010989 | -0.010999 | -0.011258 |
| S-band surface sigma MAX [kg/m3] | 0.906415 | 0.906479 | 0.909343 |
| S-band surface sigma MEAN [kg/m3] | 0.779886 | 0.779885 | 0.802915 |
| **gate tally** | PASS 0 / FAIL 5 | PASS 0 / FAIL 5 | — |

## The flip's effect, against each metric's own noise floor

| metric | gap A | gap B | gap B - gap A | floor | readable? |
|---|---|---|---|---|---|
| ACC [Sv] | 7.297e-01 | 8.168e-01 | **+8.709e-02** | 9.1e-02 | no — below floor |
| upper contrast | 1.903e-03 | 1.853e-03 | -5.000e-05 | 1.1e-04 | no |
| deep contrast | 2.688e-04 | 2.593e-04 | -9.413e-06 | 4.5e-05 | no |
| sigma MAX | 2.928e-03 | 2.864e-03 | -6.402e-05 | 9.5e-05 | no |
| sigma MEAN | 2.303e-02 | 2.303e-02 | +1.226e-06 | 9.5e-05 | no |

**VERDICT: REFUTED**, on the pre-registered criterion. The ACC gap does not
shrink by >0.1 Sv; it GROWS by 0.087 Sv, which is itself below the 0.091 Sv
floor and therefore unreadable in either direction. Every one of the five gate
metrics moves by less than its own noise floor. At 90 days, on this gate, the
choice is climate-inert.

The null is NOT vacuous — the knob demonstrably bites the solution:

    day   0   max|du|=0.000e+00   max|dT|=0.000e+00   max|d_eta|=0.000e+00
    day  30   max|du|=3.013e-02   max|dT|=2.167e-01   max|d_eta|=2.717e-03
    day  60   max|du|=4.721e-02   max|dT|=3.464e-01   max|d_eta|=1.935e-03
    day  90   max|du|=4.066e-02   max|dT|=2.025e-01   max|d_eta|=2.274e-03

A 0.35 K local temperature difference and a 4.7 cm/s velocity difference that
integrate to nothing on all five gate metrics is the finding.

## TWO CORRECTIONS TO THIS TEST'S OWN PREMISE, both loud

**(1) The headline number was wrong by 2.4x, and so was its sign.** The day-90
ACC gap at this SHA is **0.730 Sv**, not the ~1.74 Sv the pre-registration
carried, and legoESM is **ABOVE** NEMO (66.10 vs 65.37), so it is an ACC
**EXCESS**, not a deficit. The pre-registered "expected shrink of 0.15-0.3 Sv"
was scaled off a quantity 2.4x too large. The CONFIRMS/REFUTES thresholds were
stated in absolute Sv, so the verdict is unaffected — but any downstream claim
sized against "the 1.74 Sv ACC deficit" on this metric at this branch state
should be re-checked before it is used again.

**(2) The funnel's implied SIGN does not survive at the whole-model level.**
Both models start from the same day-180 restart at ACC 61.946 Sv:

    day        0       30       60       90     90-day change
    NEMO   61.946        -        -   65.369        +3.423 Sv
    arm A  61.946   64.239   65.095   66.099        +4.153 Sv  (+21% vs NEMO)
    arm B  61.946   64.488   65.448   66.186        +4.240 Sv  (+24% vs NEMO)

legoESM spins the ACC up 21% FASTER than NEMO over this window, and the flip
adds a further ~2.5%. The band-rate A/B (`bbfdb8e5d`) reported the flip
"recovers 18% of the southern spin-up rate", i.e. it SPEEDS the spin-up — and
that direction reproduces here (arm B is above arm A at day 30 and day 60 by
+0.25 and +0.35 Sv). But the premise that legoESM is too SLOW is false on this
metric: it is too FAST, so a faithful-looking change that speeds it up moves
the whole-model gap the WRONG way. The two lanes agree on the mechanism's sign
and disagree on which side of NEMO the model sits.

Note also the transient: the arms separate to +0.35 Sv by day 60 and then
re-converge to +0.087 Sv by day 90. A test reading this A/B at day 30-60 would
have called it a 3-4x larger effect than the day-90 gate sees.

## What this does and does not settle

- **Settles**: `barotropic_reconcile_target` is not a lever on the 90-day
  acceptance gate. It should not be carried further as a candidate fix for the
  ACC gap on the strength of the M2 or band-rate results.
- **Does not settle**: M2's own claim. M2 measured the SHAPE of a southern-band
  torque gap, not ACC, and its 77% variance result stands on its own terms.
  This gate says that shape agreement does not propagate to any gate metric at
  90 days.
- **Untouched by both arms**: M2's row R6 — legoESM has ONE reconciliation site
  where NEMO has TWO, and the after-level thickness/divisor mismatch survives
  both arms. That row is still open and is not what was tested here.
- **Not a 90-day question**: all five metrics still FAIL the 5x gate on BOTH
  arms, by 1.6x to 48x their thresholds. The gap this campaign is chasing is
  larger than anything this option controls.
