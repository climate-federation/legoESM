# ORCA2 round 299 preregistration — kt=7 transport-component split

Date: 2026-10-16. Frozen base: `4a46c6da3`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. Round 298 names the
active-tracer transport checkpoint as the first causal kt=7 stage-1 growth
boundary. This round substitutes the admitted U, V, and W transports one at a
time in compiled order, then requests a leaf-operand record only if the first
causal component is not covered at kt=7. It changes no production package,
card, deck, selector, carried state, stabiliser, sea-ice field, or
`unmeasured_features` entry.

## Resolved compiled program

The admitted OMT-4 deck prints `ln_dynadv_vec=.true.`, `ln_shuman=.false.`,
`ln_wave=.false.`, `ln_ldfeiv=.false.`, `ln_mle=.false.`, and
`ln_zad_Aimp=.false.`. NEMO therefore builds the carried horizontal transports
at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:276-292`, calls the
active-tracer transport routine at `:565-570`, and records its U, V, W result
at `:599-601`. On this resolved branch `tra_adv_trp_t` leaves the active
horizontal fields unchanged apart from the bottom-zero assignment at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/traadv.f90:248-252`, diagnoses `ww`
from those transports at `:267-302`, and writes `zFw=e1e2t*ww` at `:314-316`.

The component arms reuse round 298's passive candidate trace and exact offline
tracer replay. Each arm replaces exactly one of the native metric transports
`zFu`, `zFv`, or `zFw` with its admitted NEMO value; the other two remain the
candidate values. The full three-component arm remains the round-298 control.
The frozen causal threshold is ten times the kt=7 entry fold-band salinity
maximum: `3256.037473733219` PSU.

## Frozen predictions and falsifiers

1. **R299-P1 — provenance and controls.** The round-297 record admits with two
   ranks and 23 fields; both labels run from one clean CPU/fp64/libm tree;
   traced and ordinary completed states are array-identical; the candidate
   offline tracer replay remains exact. Any miss refuses every scientific
   number.
2. **R299-P2 — component agreement.** Both labels independently name the same
   first causal component among `trp_u`, `trp_v`, and `trp_w`. Label
   disagreement is a refusal, not an averaged result.
3. **R299-P3 — first component.** `trp_u` alone remains above the frozen 10x
   threshold and `trp_v` alone takes the completed fold-band salinity below
   it; `trp_v` is therefore predicted to be the first causal component. A U
   owner, a W owner, or no single-component owner is retained as **REFUTED**
   and becomes the finding.
4. **R299-P4 — single-component discrimination.** The candidate triplet
   reproduces round 298's `12046.543040268563` PSU result and the all-NEMO
   triplet reproduces `325.50387971291514` PSU under both labels. If either
   control moves, no component claim is made.
5. **R299-P5 — leaf coverage.** If U or V is causal, the existing admitted
   operand stream is insufficient because its source-exact
   `e2u/e3u/uu/zub/umask` and `e1v/e3v/vv/zvb/vmask` record exists only at
   kt=1, not kt=7. If W is causal, the existing stream likewise lacks the
   kt=7 `wzv` thickness/ratio operands. A missing leaf record produces one
   fail-closed, rank-complete acquisition request; no operand is inferred from
   a later state.
6. **R299-P6 — measurement only.** No constituent of the private fold unit
   lands. Admission, label coverage, source order, passivity, replay,
   candidate/full-NEMO controls, first-component selection, Decision-127
   subnormal handling, and a normal one-bit perturbation each have a planted
   violation that must refuse.

## Stop conditions

A causal component without source-complete kt=7 operands ends
`STOPPED_FOR_RECORD` with an absolute `run.sh`; a component and covered leaf
ends `HELD` with the cited statement. The complete fold unit remains private
and atomic until OMT-4 completes under both labels and all standing gates pass.
Decision 126 remains pending and is not re-raised; shipped runoff and sea ice
remain unchanged.
