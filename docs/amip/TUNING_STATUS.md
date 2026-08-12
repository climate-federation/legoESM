# Calibration campaign — status at handover (2026-08-11)

Protocol: Gjini, Morzfeld & Watson-Parris, *Calibrating an auto-differentiable
intermediate-complexity atmospheric model with reanalysis data*. Three stages:
variance diagnostic → parameter screening → ensemble Kalman inversion.

## Stage 1 — COMPLETE, both windows

15 members: 3 parameter sets (low / default / high) × 5 initial conditions,
365 days each. All 15 verified to have bound both their parameter set and
their IC before any number was computed.

**The question Stage 1 exists to answer:** can a short window measure a
parameter's effect, or is it just measuring which initial state you started
from? Written as (a) = spread across parameter sets, (b) = spread across ICs.

**Answer: parameters dominate, decisively.** (a)/(b) at 3 months:

| statistic | (a)/(b) | | statistic | (a)/(b) |
|---|---|---|---|---|
| rlut bias | **107×** | | pr bias | 11× |
| rsut bias | **57×** | | tas pattern | 4.8× |
| net TOA | **27×** | | prw pattern | 3.8× |
| tas bias | **24×** | | **pr pattern** | **1.4× — marginal** |
| prw bias | **22×** | | | |

At 12 months the picture holds (4× to 124×) with one change: **`tas` pattern
correlation drops to 1.6×** and should not be scored.

**So 3-month scoring is legitimate**, which makes Stages 2 and 3 far cheaper.
Only precipitation pattern is unresolvable at that window.

## Stage 1 — the biases tuning will never close

Long-run model mean vs observations, area-weighted, months 1-12:

| | model | obs | bias | vs 5% floor |
|---|---|---|---|---|
| rsut | 135.4 | 98.78 | **+36.5** | **7.4×** |
| pr | 2.07 | 3.09 | **−1.02** | **6.6×** |
| rlut | 216.6 | 240.5 | **−23.9** | **2.0×** |
| net TOA | −14.7 | +0.9 | **−15.6** | far outside |
| prw | 24.5 | 24.3 | +0.20 | inside |
| tas | 285.4 | 287.1 | −1.77 | inside |

Structural, not sampling noise: two independent years reproduce every bias to
within 0.06 W/m². **Without a correctly sized error floor an optimiser will
push parameters to their bounds trying to close physics it cannot reach.**

## Three findings that change how Stage 3 must be set up

1. **The paper's 5% error floor breaks for two statistics.** 5% of 287 K is
   14 K — meaningless, because Kelvin has an arbitrary zero, and it would
   accept essentially any temperature bias. 5% of a +0.9 W/m² net TOA is
   0.045 W/m², absurdly tight against a 15.6 bias. Both need **absolute**
   floors, or net TOA is dropped in favour of `rsut`/`rlut`, which carry the
   same information without a near-zero denominator.
2. **Internal variability is NOT parameter-independent** — it varies by up to
   3.3-3.6× across the three settings. A `P_tau` estimated at default
   parameters understates the noise elsewhere by over 3× for `net_toa` and
   `rsut_r`. Carry an inflation factor or estimate per set.
3. **The three parameter sets compensate internally.** `rsut` bias is 36.6 at
   default and 37.5 at high — nearly identical — while low gives 59.5.
   Raising `rh_crit` is offset by raising `Nc_default`/`conv_cloud_coeff`. So
   the measured (a) UNDERSTATES what individual parameters can do. Separating
   them is exactly Stage 2's job.

## Stage 2 — designed and pre-registered, BLOCKED

Pre-registration: `/scratch/b/b309178/stage2/PREREGISTRATION.md`. Noise floors
frozen from the Stage 1 3-month (b) column. Verdict rule fixed in advance:
`S = max over statistics of |hi − lo| / floor`; CONSTRAINABLE at S ≥ 3,
MARGINAL 1-3, DROP below 1.

**The blocker: only 8 of 25 core parameters can be varied on the production
lane.** `--params` splits names between a flat-config route and a
pipeline route; the MPAS lane **refuses** the pipeline route outright
(`run_amip.py` ~3048), correctly, because MPAS rebuilds its scheme configs
from the flat `ExperimentConfig` and a post-setup override would be silently
ignored. That guard must stay.

Settable today: `CloudConfig.{alpha_xr,q_c_diagnostic,rh_crit}`,
`HinesConfig.{Fmax,total_rms_wind}`, `MorrisonConfig.k_au`,
`LouisConfig.{Ri_crit,l_mix_max}`.

Not settable: all of Bechtold convection, most of Morrison, both surface
exchange coefficients, every land parameter, every sea-ice parameter, all
albedos, McFarlane.

**A second finding:** the repo's own reachability audit
(`tests/unit/test_params_reachability_audit.py`) reports only **3** of these as
unreachable, because it measures reachability in principle rather than on the
lane we run. It is green while 17 core parameters cannot be varied in
production.

**The fix** — extend the flat-config route so those parameters reach the MPAS
scheme rebuild, and make the audit lane-aware — was scoped and started but not
completed. It unblocks both Stage 2 options (screen 8, or screen all 25) and
Stage 3.

## Two parameters deliberately NOT touched

`CloudConfig.Nc_default` and `cloud_rh_crit` are reserved calibration
decisions. Nothing in this week's work changed either. Note that
`Nc_default` may be the wrong lever entirely: switching on the
**aerosol-CCN coupling** (`--aerosol-ccn`, already implemented, prerequisites
all met, simply never set in the production config) makes droplet number
respond to the aerosol field and moves global effective radius from 8.0 to
10.5 µm — most of the way to the 11-14 µm target — while producing a
land/ocean contrast no constant can. That measurement is one controlled arm
from being settled; see the handover issue.
