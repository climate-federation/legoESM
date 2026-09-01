# NEMO testcase lane 1: full-duration statistical preregistration

Status: **PREREGISTERED — no full-duration legoESM metric has been computed and
no comparison arm has been launched at this commit.** This post-PR extension
freezes the transition from deterministic trajectory fidelity to statistical
equivalence for the certified `LOCK_EXCHANGE-zco` and `OVERFLOW-zps` cards.

### Pre-score amendment: endpoint membership is not an FCT verdict

After the first preregistration commit and the N4 executions, but still before
computing any L64/L32 metric, a dry run of the scorer against N2 alone found
that the proposed endpoint snap predicate would reject the certified oracle:
OVERFLOW N2 ends at `20.00000000000307 C`, an excess of
`3.0695e-12 K`, while its `sqrt(6120)*eps*20` floor is only
`3.4741e-13 K`. This is not new physics evidence. It reproduces the phase-1
finding that the temperature excess is **UNMEASURED** at 796 epsilon-relative
(`nemo_testcases_l1_phase1_receipt.md:21-26,68`). The phase-1 gate separately
hard-fails only a gross relative excursion above `1e-6`
(`nemo_testcase_oracle_gate.py:399-415`).

The endpoint convention below is therefore amended before any legoESM metric:
the scorer prints the raw excess, the `sqrt(N)*eps` floor, and `AT-BAR` or
`UNMEASURED`; it hard-fails an excess/field-scale above `1e-6`; and it snaps to
the nearest endpoint only for histogram-bin and census-boundary membership.
This bookkeeping operation cannot promote the FCT verdict and never modifies
the deterministic field comparison. The planted `10 K` wet-cell excursion
still hard-fails. This amendment retracts only the proposed hard failure at
the roundoff-classification floor; all metric distances and the three-way
statistical verdict table remain unchanged.

## Question, arms, and immutable samples

The question is whether legoESM's full-duration solution is distinguishable
from NEMO 5.0.2 on the physical diagnostics each case was built to expose,
after calibrating every diagnostic against both numerical precision and a
shipped alternative advection scheme. There is no ensemble and no stochastic
spread to estimate. No ensemble-style standard error will be invented.

| arm | case configuration | arithmetic / execution | purpose |
|---|---|---|---|
| `N2` | certified NEMO FCT2 baseline | existing CPU `REAL(wp)` artifacts | oracle |
| `N4` | same NEMO configuration, only coupled FCT order `(nn_fct_h,nn_fct_v)=(4,4)` | one new CPU run per case | physically meaningful scheme spread |
| `L64` | certified legoESM card | explicit `PrecisionPolicy.fp64()` | candidate |
| `L32` | identical legoESM card | explicit `PrecisionPolicy.fp32()` with JAX x64 disabled | measured roundoff floor |

The `N4` arm is not an invented model. NEMO ships paired FCT2/FCT4 variants:
`tests/OVERFLOW/EXPREF/namelist_zps_FCT2_flux_ubs_cfg:64-69` versus
`namelist_zps_FCT4_flux_ubs_cfg:65-70`, and
`tests/LOCK_EXCHANGE/EXPREF/namelist_FCT2_flux_ubs_cfg:63-68` versus
`namelist_FCT4_flux_ubs_cfg:63-68`. The executed source selects horizontal
second/fourth order at `src/OCE/TRA/traadv_fct.F90:183-248` and vertical
second/compact-fourth order at `:254-278`. Under `key_RK3`, both orders share
the same two-step low-order predictor (`traadv_fct.F90:153-161`). The copied
alternative namelist must differ from the certified one only in
`cn_exp`, `nn_fct_h`, and `nn_fct_v`; a semantic diff finding any other
assignment hard-fails before NEMO starts. In particular, the shipped FCT4
files' other historical differences are not copied into this one-variable
arm.

The registered states are deliberately limited by the already-certified
oracle artifacts. No curve is drawn at unobserved oracle times:

| case | initial entry | midpoint entry and physical time | full-duration state |
|---|---:|---:|---:|
| OVERFLOW-zps | `kt=1`, `t=0 s` | `kt=3060`, `t=(3060-1)10=30590 s` | restart after 6120 steps, `t=61200 s` |
| LOCK_EXCHANGE-zco | `kt=1`, `t=0 s` | `kt=30600`, `t=(30600-1)1=30599 s` | restart after 61200 steps, `t=61200 s` |

The entry records are NEMO `Nbb` / before fields. The final restarts are
`tn/sn/un/vn/sshn` after the registered last step. legoESM states are sampled
at those exact physical times: initial, after 3059/30599 completed steps, and
after 6120/61200 completed steps. Entry and restart frames are never mixed in
one row without those labels.

The baseline oracle inputs are hash-pinned:

| case / artifact | SHA256 |
|---|---|
| OVERFLOW entry `kt=1` | `cf0183e563aba8b8848bc5dea470c9e50aab2d987ff5da86241e8f82494d5c66` |
| OVERFLOW entry `kt=3060` | `ae3e27c43bb649a40db50de24357421d55a6378dc6658c3a5b9e5f64a8382d4c` |
| OVERFLOW final restart | `dab392f2f058b44e8c10c600a41c9be73ba37656e2af478193a3f6f27bd67160` |
| OVERFLOW namelist / binary | `ec1eac4a45fb8c07a0facce5e4eefb6510d8e3f1e364f5c5597e60ae83ccc53e` / `eb4acf9651b887a3da8834281112d472692caa0bbadcb0d69779e91dee92e6cb` |
| LOCK entry `kt=1` | `c9f23d441865c566e3edf0301aca4f6e440259ade1722b725a0aa0fd4c2839e1` |
| LOCK entry `kt=30600` | `76bb241ddcc1f1a7148eceb2a0807e7bd6afb376b4f1139d100ee8234709a342` |
| LOCK final restart | `15a7883e5e27df2fec36716c29375ec947892a842ee901e5526e10525db5c7a6` |
| LOCK namelist / binary | `ae34648ecdf44893e8543f0516511d239ce59161fa5b2de9924f0169d8185dd4` / `d297e236afd0097fc64533f4182fada9d58cc0458c359af06e08106fb796a259` |

The unusable phase-1 XIOS history files are explicitly waived: they contain
only time-averaged W/TKE diffusivity variables, no T/S/u/ssh, and have corrupt
coordinate values. They supply no number in this extension.

## Common frame, weights, and deterministic bridge

All state comparisons reuse the phase-3 gate's loader and staggering:

- NEMO entry halos are stripped by `[2:-2,2:-2]`; global restarts need no halo
  strip. T is at T centres. NEMO `un` and entry `uu(...,Nbb)` are compared to
  legoESM's instantaneous prognostic U faces mapped as `u[:,1:,:]`.
- Every reduction uses the intersection of the certified wet T/U masks. There
  is no interpolation, dry fill, or XIOS field.
- Volume diagnostics use `area_T * compute_layer_thickness(ssh,H,z_coord)` in
  float64 accumulation on every arm. The same certified partial-cell geometry
  is used to interpret NEMO and legoESM states.
- Potential density for water-mass ordering is the canonical NEMO TEOS-10
  polynomial evaluated at geometric depth zero. This fixed reference removes
  compressive in-situ density changes from the irreversible-mixing diagnostic.

The deterministic bridge is the normalized wet `L_inf` error used by the kt60
gate:

`E_q(t) = max_wet |L64_q(t)-N2_q(t)| / max(max_wet |N2_q(t)|,1)`

for T-centre and instantaneous U-face fields. The summary plot retains the
certified `kt=1..60` series and appends the registered midpoint and final
states. It must show the frame change from before-entry to final-restart
explicitly. No statistical verdict is inferred from the shape of this curve;
its registered scalar score is the maximum of its midpoint and final values.

## Metric registry

All thresholds and crossings use linear sub-cell interpolation. A missing
crossing, empty qualifying water mass, non-finite value, or unexpected number
of crossings is `OUTSIDE`, not a zero.

### OVERFLOW-zps

NEMO initializes a 10 C reservoir west of 20 km against 20 C ambient water
(`tests/OVERFLOW/MY_SRC/usrdef_istate.F90:58,65-75`) above the partial-cell
tanh slope (`usrdef_zgr.F90:90-103`). NEMO documents the experiment as a test
of scheme- and coordinate-dependent spurious mixing and shows the cold mass
forming a bottom-trapped dense plume (NEMO 5 manual, OVERFLOW time-series;
`tests/README.rst:127-143`). The cold/ambient midpoint is frozen at 15 C.

1. **Plume descent curve, metres.** At each registered state, select wet
   T-cells with `T <= 15 C`; report the deepest selected T-cell centre measured
   positive downward using the live column thickness. Score the maximum
   absolute curve separation over the three samples. This is the documented
   deepest-cold-water-level diagnostic; it is intentionally grid-level
   quantized and is never smoothed into an apparent sub-level result.
2. **Bottom plume-front curve, kilometres.** In every wet column take its
   deepest active T cell. Find all adjacent-x 15 C crossings and select the
   rightmost crossing connected to the initial cold reservoir. Report its
   sub-cell x coordinate and score the maximum curve separation.
3. **Final slope temperature distribution.** The slope is frozen as columns
   with `500 m < H_bathy < 2000 m`. Form a volume-weighted histogram on fixed
   0.25 C bins spanning `[10,20] C`; report the probability vector and score
   total-variation distance `0.5*sum|p_L-p_N|`. The already-certified phase-1
   endpoint convention applies before binning: raw excess is classified
   against `sqrt(N_steps)*eps(dtype)*max(|T_endpoints|,1)` and remains loudly
   `UNMEASURED` when above it. An excess larger than `1e-6` relative hard-fails;
   otherwise only endpoint membership is snapped while raw extrema are
   emitted. This clarification is made before any new legoESM metric is
   computed and responds to the pre-existing phase-1 roundoff receipt, not to
   an extension-arm result.
4. **Final slope water-mass census.** On the same volume (with only the
   endpoint-membership convention above),
   report fractions `cold=[10,12)`, `mixed=[12,18)`, and
   `ambient=[18,20]` C. The score is the maximum absolute fraction difference.
5. **Deterministic T/U bridge.** Registered as above.

### LOCK_EXCHANGE-zco

The case initializes 5 C west and 30 C east of 32 km at uniform S=35
(`tests/LOCK_EXCHANGE/MY_SRC/usrdef_istate.F90:58,65-75`) in a flat 20 m tank
(`usrdef_zgr.F90:62-67`). The standard diagnostics are bottom-front location,
temperature bounds, and normalized sorted Reference Potential Energy; this is
the Petersen et al. (2015) spurious-mixing benchmark. The midpoint is frozen at
17.5 C.

1. **Dense bottom-front curve, kilometres.** In each column take the deepest
   active T cell, find ascending 17.5 C crossings, and select the rightmost
   crossing continuously connected to the initial 32 km front. Score maximum
   absolute curve separation.
2. **Front speed, m/s, plus external anchor.** Fit displacement from the
   sampled initial position through the origin by least squares over the two
   positive registered times. The external Benjamin (1968) anchor is
   `c_B = 0.5*sqrt(g' H)`, `H=20 m`,
   `g'=g*(rho(5,35,0)-rho(30,35,0))/rho_ref`, with the same canonical NEMO
   TEOS-10 surface potential density used on both models. Report each arm's
   `c/c_B`. The scored value is the absolute difference in `c/c_B`; the anchor
   is not itself a tolerance and cannot make two disagreeing models agree.
3. **Reference/background potential energy curve.** Compute parcel volumes
   from live eta, sort surface-potential-density parcels densest first, pack
   them from the bottom in a notional column of the total wet area, and compute
   `RPE=g*sum(rho_i*z_i*V_i)`. Report
   `RPE_rel(t)=(RPE(t)-RPE(0))/abs(RPE(0))`; score maximum absolute curve
   separation. The fixed-reference-volume RPE variant is forbidden because it
   is already retracted in `docs/ocean/experiments/lock_exchange_benchmark.md`.
4. **Temperature-variance decay.** Report the volume-weighted wet variance
   divided by its initial value,
   `sum(V*(T-mean_V(T))^2)/sum(V) / variance(0)`. Score maximum absolute curve
   separation.
5. **Deterministic T/U bridge.** Registered as above.

## Exact floor and verdict logic

For each scalar or curve score `m`, compute three non-negative distances with
the identical reducer:

- candidate distance `D_m = distance(L64,N2)`;
- precision floor `F_m = distance(L32,L64)`;
- scheme spread `S_m = distance(N4,N2)`.

The metric verdict is exhaustive and may contain only these exact strings:

| predicate | verdict |
|---|---|
| `D_m <= F_m` | `INDISTINGUISHABLE-AT-FLOOR` |
| `D_m > F_m` and `D_m <= S_m` | `WITHIN-SCHEME-SPREAD` |
| `D_m > F_m` and `D_m > S_m` | `OUTSIDE` |

Equality is deliberately inclusive. If both floors are zero, only exact
equality clears the first two predicates. No percentage, correlation, visual
similarity, or analytic-anchor proximity can override this table. A metric
with missing/non-finite inputs or a failed coverage control is `OUTSIDE` with a
machine-readable reason.

These are the preregistered confirm/refute values: `D<=F` confirms
indistinguishability at the measured precision floor; `F<D<=S` confirms only
membership in NEMO's shipped FCT-order spread; `D>max(F,S)` refutes both.

## Execution decision and controls

NEMO remains one CPU process with no `mpirun`. legoESM starts on CPU. After the
preregistration commit, a compile-warmed 200-step timing arm may be run without
emitting science metrics. If its linear projection exceeds 60 minutes for a
full arm, CPU is declared impractical and the production run may pin exactly
one GPU via `CUDA_VISIBLE_DEVICES=0`; backend, device, wall time, policy, and
state/geometry dtypes must be recorded. No result may combine CPU and GPU
segments.

Artifacts live below
`/data/abyssal/dbalwada/nemo-testcases-l1/full_statistical/`; figures are copied
to `/tmp/l1_figures_full/`. The scorer must:

1. hash-check all baseline inputs and semantic-diff each N4 namelist;
2. inventory every registered state and metric exactly once;
3. reject wrong shapes, time labels, masks, dtypes, or non-finite wet values;
4. print fp64/fp32 state and geometry dtypes;
5. carry a planted state perturbation that makes a deterministic row
   `OUTSIDE`, a planted census perturbation that makes its verdict red, and a
   planted unregistered-metric control that hard-fails;
6. emit the three distances and exact verdict predicate for every row.

Figures use the lane-1 conventions: NEMO, legoESM, and independently scaled
symmetric differences for sections at the three registered states; metric
curves show N2, N4, L64, and L32 with units and floor class; every footer names
case, time/frame, mask/reduction, backend, precision, git SHA, and whether the
legoESM state was recomputed or loaded from the committed-run artifact.
