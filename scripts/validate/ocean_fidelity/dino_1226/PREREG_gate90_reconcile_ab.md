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
