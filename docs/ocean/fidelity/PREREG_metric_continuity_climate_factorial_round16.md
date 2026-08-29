# Preregistration: V-face metric x QCO association climate factorial

Date: 2026-08-29. Frozen before any climate arm is run. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Climate question

Does the NEMO-faithful V-face Mercator metric, together with NEMO's literal
QCO continuity association, materially reduce the frozen southern-basin
transport deficit (`Gbasin90`, baseline -0.9519122331848315 Sv at day 360)
and the independently frozen five-day wall flicker?

This is an intervention test on a production fix already in the twin path. It
is released independently of the ordered source-equivalence registry: the
registry remains stopped at row 1.2 and this experiment cannot promote that
row or rows 1.3--6.

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
NEMO ladder `both`, `BRIDGED_BEFORE`, T-point stress carry, selector pair, run
length, and day-0 state. All non-selector run-config fields and masks must be
bit-identical. The basin scorer retains the imported reducer plants and
compensation gate. The wall scorer requires the certified NEMO file SHA-256
`52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a`,
land-poison identity, and planted-amplitude recovery in `[0.95,1.05]`.

No result outside these bars may be promoted. No arm has been run as of this
commit.
