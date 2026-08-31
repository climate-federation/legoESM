# Preregistration: V-face metric x QCO association climate factorial

Date: 2026-08-29. Frozen before any climate arm is run. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Climate question

Does the NEMO-faithful V-face Mercator metric, together with NEMO's literal
QCO continuity association, materially reduce the frozen southern-basin
transport deficit (`Gbasin90`, baseline -0.9519122331848315 Sv at day 360)
and the independently frozen five-day wall flicker?

This is an intervention test on a production fix already in the twin path. The
pre-measurement `AMENDMENT_round15_climate_release_round16.md` explicitly
supersedes round 15's former coupled release condition. The registry remains
stopped at row 1.2 and this experiment cannot promote that row or rows 1.3--6.

## Design

Run the full 2x2 at one clean producer commit, from one bit-identical bridged
BEFORE state, with fp64, NEMO E3T mode `both`, corrected T-point prior stress,
and every configuration field fixed except these two selectors:

| Arm | `vface_zonal_metric_evaluation` | `barotropic_continuity_evaluation` |
|---|---|---|
| M0A0 control | `legacy_tracer_midpoint` | `generic` |
| M0A1 | `legacy_tracer_midpoint` | `nemo_literal` |
| M1A0 | `nemo_vpoint` | `generic` |
| M1A1 shipped | `nemo_vpoint` | `nemo_literal` |

The metric selector changes only `dx_v`/NEMO `e1v`; a committed non-confound
test requires true faces, T/U metrics, `dy_v`, areas, and Coriolis to remain
bit-identical. NEMO constructs `gphiv` at the half-index and then evaluates
`e1v` there (`usrdef_hgr.F90:98,108,113`). The association selector changes
only the evaluated form of `e2u*ua_e*zhup2_e`, `e1v*va_e*zhvp2_e`, and the
multiply-by-reciprocal divergence (`dynspg_ts.F90:698-704,722-724`).

The day-360 basin arms save 3-D days 0 and 360. The separate five-day arms
save every-step fp64 SSH (160 samples at dt=2700 s) for the frozen wall scorer.

## Frozen basin bars

The M0A0 gap must reproduce -0.9519122331848315 Sv within twice the frozen
floor, where `F=0.06173656216045926 Sv`; otherwise the experiment is
`INVALID_CONTROL`.

The headline contrast is `gap(M1A1)-gap(M0A0)`; positive means that the
negative basin deficit shrank. The imported basin-lane decision tree is
unchanged:

1. `|delta| <= 2F`: `UNRESOLVED/FLOOR`.
2. Any frozen acceptance-metric regression beyond that metric's frozen floor:
   `UNRESOLVED/COMPENSATION`.
3. `delta/|M0A0 gap| >= 0.10`: `CONFIRMED`.
4. `delta/|M0A0 gap| <= 0.02`: `REFUTED`.
5. Otherwise: `UNRESOLVED`.

The factorial table reports the metric main effect at generic association,
the association main effect at each metric, the metric effect at literal
association, the interaction, and the combined effect. For each component:
`|effect|<=2F` is `BOUNDED_AT_FLOOR`; beyond the floor, response >=10% is
`MATERIAL_IMPROVEMENT`, <=-10% is `MATERIAL_REGRESSION`, absolute response
<=2% is `BOUNDED_SMALL`, and the interval between is `UNRESOLVED`.

## Frozen wall bars

The M0A0 control is valid only if its first-eight-step all-domain ratio lies
in `[2.60,3.18]` and its wall share in `[0.38,0.59]`. Relative to the certified
NEMO comparator, M1A1 is `CONFIRMED` only if ratio <=1.25 and wall share
<=0.17; it is `REFUTED` if ratio >=2.30 and wall share >=0.38; otherwise it is
`UNRESOLVED`. The four-corner values and additive interaction are reported,
but no new post-hoc wall interaction threshold is permitted.

## Admission and controls

Every NPZ must stamp the same clean producer commit, session, float64 storage,
NEMO ladder `both`, bridged BEFORE state, T-point stress carry, selector pair, run
length, and day-0 state. All non-selector run-config fields and masks must be
bit-identical. The basin scorer retains the imported reducer plants and
compensation gate. The wall scorer requires the certified NEMO file SHA-256
`52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a`,
land-poison identity, and planted-amplitude recovery in `[0.95,1.05]`.

No result outside these bars may be promoted. No arm has been run as of this
commit.

## 2026-08-29 receipt-only amendment after the Block-3 STOP

Block 3 stopped before loading any day-360 field or computing any registered
statistic. Its compressed `runtime/stagger` message was caused by a scorer
literal: the harness stamps `twin_start_mode="bridged"`, while both factorial
scorers required the descriptive label `"BRIDGED_BEFORE"`. The canonical
harness admission code and artifacts use the former string.

The STOP exposed a second omission before scoring: the factorial scorers did
not bind the reconstructed T-point stress content to the resolved wind-profile
selector. An exact CPU/fp64 replay through `_analytic_dino_tpoint_stress`, on
the same bridged 199x52 geometry and reconstruction time 15552000 s, changed
only `dino_wind_profile_evaluation` and reproduced:

- `factored_smoothstep` ->
  `b6a08b8395017c8e3f8df0b8b13eefa75fdfe7be3770d788beaaf1ca514127ae`;
- `nemo_literal` ->
  `cad9b34958cba58812f1dce2c6c441c8fd6b5f2733c20b91065f4d60507592b7`.

The delta is confined to `tau_x`: 3,276 float64 elements differ, maximum
absolute delta `1.3877787807814457e-16`, RMS delta
`2.0276965279154703e-17`; `tau_y` is bit-identical. All four completed basin
artifacts stamp `nemo_literal`, `cad9...`, time 15552000 s, the same initial
state hash, and producer `e013e95ca54957a4454878ed7118e623da0a19ba`.

Admission is therefore amended without changing a science bar: require the
literal harness stamp `twin_start_mode="bridged"`, and bind the stress hash to
the resolved selector. Pre-selector retained artifacts remain mapped to the
historical factored hash; current complete-card `nemo_literal` artifacts map
to `cad9...`. Planted start-label and content-hash swaps must both be rejected.
No arm is rerun, and Blocks 1--2 remain valid.
