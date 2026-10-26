# ORCA2 round 82 preregistration — hierarchy rung 0

Frozen before constructing or running the rung-0 deck.  Base:
`178904859a1757c4e5d62f115d8d2c4552979cfc`.  This round implements Decision
79 and the binding rung-0 definition in operator note B21.  It does not change
the shipped `ORCA2-zps` card, its sea-ice debt tuple, any shared model default,
or any NEMO source.

## Claim and protocol

Rung 0 is **independent** ORCA2 geometry and shipped climatological T/S with
only the deck's dynamics core: EOS-80, FCT tracer advection, EEN vector
momentum, SCO pressure gradient, split-explicit free surface, shipped lateral
operators and enhanced vertical diffusion.  Constant vertical mixing replaces
TKE at the GYRE-card values, `rn_avm0=1.2e-4 m2/s` and
`rn_avt0=1.2e-5 m2/s`.  Surface forcing/restoring/freshwater budgeting,
runoff, shortwave, T/S damping, bottom boundary layer, geothermal heating,
GM/MLE, TKE, IWM, double diffusion, river-mouth diffusion, ice, special-zero
dynamics and differential T/S mixing are off.  No higher-rung placement is
chosen here.

The NEMO acquisition has two two-rank from-rest runs made from one rung-0
deck: 10 steps with a restart at step 10, and 240 steps with a restart at step
240.  The executable and CPP keys are shared between them; only `nn_itend` and
`nn_stock` may differ.  The card is a new explicit rung-0 identity; the shipped
ORCA2 card remains byte-for-byte unchanged.  The operator runs the committed
launcher; the agent runs preflight only.

## Frozen predictions and falsifiers

1. The rung-0 namelist differs from the admitted shipped ocean deck only in
   the assignments enumerated by the rung-0 deck gate.  Any extra assignment,
   changed CPP key, changed input manifest, live ice, or live excluded module
   **REFUTES** the deck.
2. NEMO resolves `ln_zdfcst=.true.`, `ln_zdftke=.false.`, the two frozen
   constant coefficients above, and `nn_ice=0`.  A different resolved value
   **REFUTES** the rung.
3. The 10-step and 240-step runs complete with finite fp64 ocean restart
   fields on both ranks.  A missing shard, non-finite payload, wrong step,
   nonzero ice selection, or incomplete run **REFUTES** record admission.
4. The rung-0 card states every switched mechanism explicitly and refuses any
   unresolved feature.  Falling through a shipped-card default, retaining a
   TKE/IWM carried field as an active input, or altering the shipped card
   **REFUTES** the implementation.
5. The given-entry and independent ten-step ladders and the independent month
   are **UNMEASURED pending acquisition**.  No first non-bit statement or
   magnitude is predicted post hoc.  If acquisition is absent, disposition is
   `STOPPED_FOR_RECORD` with the launcher path.

## Mechanical controls

The deck gate must refuse an undeclared namelist delta, a changed constant
mixing value, a live excluded module, a changed CPP file, and a hidden month
deck delta.  The record gate must additionally refuse a one-ULP restart
payload, missing rank, wrong terminal step, and non-finite value.  Tests show
each plant firing.  Citation validation runs on the round receipt and the
campaign default with a real line-shift plant.  A separate read-only Codex
review is requested after the implementation diff is complete.

ASKED: construct rung 0 exactly as Decisions 79/B21 specify and hand its
operator-run acquisition forward if no admissible record exists.

UNASKED: higher-rung ordering, changes to the shipped ORCA2 card or sea-ice
debt tuple, new stabilizers, altered thresholds, carried-state changes, NEMO
source edits, and any interpretation of unmeasured ladder/month magnitudes.
