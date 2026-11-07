# Preregistration — VORTEX_SMT round 14 (lane round 226): admit and score SMT-3

Frozen before reading any SMT-3 trajectory value or running legoESM against
the admitted record.  Base: lane tip `be6e4fb91` (round 225).  Evidence lives
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round226/`; the immutable
NEMO record is round 224's operator-run acquisition.

## Compiled branch read first

The new build's own `ocean.output` resolves `ln_traldf_lap`,
`ln_traldf_iso`, and `ln_traldf_msc` true, `nn_aht_ijk_t=20`,
`rn_Ud=0.018`, and `rn_Ld=200000`; all companion switches retain the
reference values preregistered in round 224.  The compiled
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo` branch dispatches that tuple to
`traldf_iso_lap`; exact line spans will be registered only after the receipt's
citation map is built.

## Predictions and falsifiers

* **R14-P1 — acquired record admission.**  Both admission JSON files say
  `ADMITTED`; both NEMO arms reached `STOP 0`; the plain/instrumented step-10
  restarts are byte-identical; the self-describing records parse to EOF; and
  every header/name/truncation plant exits nonzero.  REFUTED by any missing
  proof or disagreement with the operator log.
* **R14-P2 — one-module explicit card.**  `VORTEX_SMT3_VEC-zps` differs from
  SMT-2 only in the resolved `namtra_ldf` module.  Geometry, initial state,
  vertical mixing, EVD, drag, momentum, barotropic program, timestep and run
  length are bit-identical to SMT-2.  Its isoneutral/MSC configuration states
  every NEMO option it consumes; no library default selects a physical
  choice.  REFUTED by any other changed field or an unstated selector.
* **R14-P3 — ladder order.**  All five kt=1 rows stay AT-BAR and the first
  over-bar row is not earlier than SMT-2's kt=2 row.  Every one of the 50
  rows is registered, including worsening rows.  REFUTED by any kt=1 loss or
  an earlier first-over-bar row.
* **R14-P4 — post-LDF discriminator.**  At stage 3, given NEMO's recorded
  entry and transports, the pre-LDF tracer RHS remains at the round-219 bar
  while the post-LDF T/S RHS is non-bit.  The first owned statement is in
  NEMO's compiled slope/coefficient/`traldf_iso_lap` sequence, not FCT.
  REFUTED if pre-LDF is first, post-LDF is bit-exact, or the residue is
  inherited from an earlier operand.
* **R14-P5 — shipped-length sanity.**  The 100-day legoESM arm completes with
  finite T/S/u/v/ssh, and the scoring script reproduces its own kt=1..10
  registry before accepting any daily number.  REFUTED by non-finite state,
  missing daily restart, or a ten-step mismatch.
* **R14-P6 — no premature landing.**  A statement lands only if it is locally
  bit-exact given NEMO inputs and passes the standing SMT, flat-VORTEX, tank,
  GYRE ladder/year, generic-card, DINO-month, citation, plant, and independent
  review gates.  Otherwise the round is HELD with the named first statement.

## No hidden choices

Decision 93 already authorises this rung.  The six active values and eight
companions remain exactly those preregistered in round 224.  No configuration,
carried state, threshold, stabiliser, target name, or record format is chosen
in this round beyond expressing that resolved NEMO deck as an explicit card.
