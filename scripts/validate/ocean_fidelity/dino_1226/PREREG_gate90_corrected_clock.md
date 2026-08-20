# PRE-REGISTRATION — the 90-day acceptance gate RE-BASELINED on the corrected seasonal clock

Written and committed BEFORE the run. Git history proves the order.

## Why this run exists

`076217667` CONFIRMED that the twin harness handed the DINO analytic surface
forcing a RELATIVE time (`t = (k+1)*dt`, seasonal year restarted at zero) while
NEMO evaluates the same formulas on the ABSOLUTE day of year
(`usrdef_sbc.F90:536-547`, `ztime = REAL(kt)*rn_Dt`). The twin's IC is NEMO's
step 5760 — day 180 of a 360-day year — so the two models ran EXACTLY
antiphase: legoESM's southern channel in summer against NEMO's in winter, and
the northern basin the other way round.

Measured in the one-variable A/B (`twin_seasonal_clock_ab.py`, re-run on the
hardened instruments in `afd8e06b6`): the offset owns **99.1 % of the day-30**
southern-band surface density gap (-0.013224 -> -0.000125 kg/m3, i.e. from
139x the noise floor down to 1.3x).

Every 90-day twin number this campaign produced was therefore scored across
that antiphase. This run re-establishes the baseline on the corrected clock,
which is now the harness DEFAULT (read from the restart's own `adatrj`, not
hardcoded).

## The run (one variable vs the recorded baseline: the clock)

Recipe `nemo_dino_kamm_mlf`, `--bridge-before`, `--days 90 --save-3d`,
IC = NEMO `DINO_00005760_restart.nc` (day 180) from `RUN_90D_TWIN`,
`LEGOESM_NEMO_E3T=off` (the 1-D thickness ladder — the same ladder EVERY
recorded arm of this campaign ran on, deliberately not NEMO's true `e3t_0`),
fp64 via `run_fp64.py`, dt 2700 s, 32 steps/day, `CUDA_VISIBLE_DEVICES=0`,
`tau_x` range and the day-0 gate printed in the log.

Instrument: `acceptance_gate_90d.py`, UNMODIFIED — no gate-constant, threshold
or metric edits. Its two fatal self-checks (NEMO y10 ACC = 121.07 Sv through
this harness; band volume = 2.694775e16 m3) run before any comparison, and the
`--self-test` non-vacuity check is run separately.

Baseline on the NEMO side is unchanged: NEMO's own continuation from the same
restart, NOW level, `RUN_90D_TWIN/DINO_{kt}_restart_*` (16 tiles, stitched).

## What the recorded (ANTIPHASE) baseline says — the numbers being replaced

Arm A of `c38e8a3ea` (SHA `b06186dea`), day 90, level 5x: **PASS 0 / FAIL 5.**

| metric | legoESM (antiphase) | NEMO d90 | gap | floor | gap/floor |
|---|---|---|---|---|---|
| ACC [Sv] | 66.098876 | 65.369204 | +7.297e-01 | 9.1e-02 | 8.0x |
| upper contrast <1400 m [kg/m3] | -0.286279 | -0.288182 | +1.903e-03 | 1.1e-04 | 17x |
| deep contrast >1400 m [kg/m3] | -0.010989 | -0.011258 | +2.688e-04 | 4.5e-05 | 6.0x |
| S-band surface sigma MAX [kg/m3] | 0.906415 | 0.909343 | -2.928e-03 | 9.5e-05 | 31x |
| S-band surface sigma MEAN [kg/m3] | 0.779886 | 0.802915 | -2.303e-02 | 9.5e-05 | 242x |

## Predictions, recorded before the run

1. **Density metrics collapse toward their floors.** The A/B removed 99.1 % of
   the day-30 sigma-mean gap. If that carries to day 90, the sigma MEAN gap
   falls from 2.3e-02 to order 1e-04 - 1e-03 kg/m3, i.e. from 242x the floor
   to single-digit multiples. sigma MAX, upper contrast and deep contrast are
   expected to move the same way and the same direction.
   - CONFIRMS the carry-over: sigma MEAN gap <= 3.0e-03 kg/m3 (a >=87 % cut).
   - REFUTES it: sigma MEAN gap >= 1.5e-02 kg/m3 (<35 % cut) — i.e. the day-30
     collapse does not survive to day 90 and something else dominates by then.
   - Anything between is PARTIAL and gets reported as such.
   The A/B measured DAY 30 only, and armB's residual was still 1.3x the floor
   with the published anomaly still rising at day 90. A PASS at 5x on the
   density metrics is possible but is NOT predicted here.

2. **ACC is UNPREDICTED.** The recorded +0.730 Sv excess was measured across
   the antiphase, so it carries no information about the corrected clock — it
   is not a baseline this run is trying to beat, and neither a larger nor a
   smaller gap would be a surprise. Whatever comes out is recorded SIGNED
   (legoESM minus NEMO), next to the 0.091 Sv floor. No CONFIRM/REFUTE band is
   pre-registered for ACC because none can be honestly justified; declaring one
   after the fact is forbidden.

3. **Day 0 is unchanged.** The bridge is untouched, so the day-0 gate must
   still be exact (`max|dT| = max|du| = max|dv| = max|d_eta| = 0.000e+00`) and
   the day-0 sigma gap must reproduce the recorded +3.4e-08 kg/m3. If it does
   not, the run differs in more than the clock and NOTHING below is readable.

4. **Stability.** The recorded arms ran stable to day 90 in ~215 s wall. A
   blow-up under the corrected clock would itself be the finding (step, day and
   T-range reported next to the active `off` ladder), not a failed run.

## Also recorded, free (no extra runs)

`kamm_twin_90d.py --save-3d` already snapshots days 0/30/60/90, and NEMO's own
`RUN_90D_TWIN` dumps exist at kt 6720 (day 30) and 7680 (day 60). The gate's
own `load_candidate(day=)` / `load_nemo_day90(kt=)` entry points are therefore
called at days 30 and 60 as well, with the SAME unmodified metrics and floors.
This is the first clean look at how the corrected-clock gaps EVOLVE; it is
descriptive and carries no pre-registered band.

## Standing caveats that this run does not remove

- `LEGOESM_NEMO_E3T=off` is not NEMO's true thickness ladder. Using `both`
  would change two variables at once and is a known unfixed instability from a
  NEMO restart.
- Within the clock, T* and Qsr cannot be separated — the clock moves both
  cosines together (`280bba6a8` item 1). "The clock is the owner" is CONFIRMED;
  "T* rather than Qsr" remains PLAUSIBLE and untested.
- The noise floors are an n=3 micro-ensemble estimate (#1492 2.1).
