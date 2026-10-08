# Preregistration — VORTEX_SMT round 12 (lane round 224): rung SMT-3, lateral tracer diffusion

Frozen before any SMT-3 measurement.  Base: lane tip `889afc57d` (round 223 /
VORTEX_SMT round 11).  Decision 93's seamount mini-ladder changes exactly one
module against SMT-2: lateral tracer diffusion.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/`.

## The switch set, cited

The ORCA2 rung-0 deck is
`phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg`.
Its `&namtra_ldf` block states `ln_traldf_lap=.true.` at line 319,
`ln_traldf_iso=.true.` at line 320, `ln_traldf_msc=.true.` at line 321,
`nn_aht_ijk_t=20` at line 323, `rn_Ud=0.018 m/s` at line 325 and
`rn_Ld=200.e3 m` at line 326.  SMT-3 copies those six values exactly.

Rung 0 leaves the companion values unset, so SMT-3 keeps the compiled NEMO
reference values from `cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref`:
`ln_traldf_blp=.false.` (line 920), `ln_traldf_lev=.false.` (923),
`ln_traldf_hor=.false.` (924), `ln_traldf_triad=.false.` (926),
`rn_slpmax=0.01` (930), `ln_triad_iso=.false.` (931), `rn_sw_triad=1`
(932), and `ln_botmix_triad=.false.` (933).  No value is inferred from the
physics.

## Predictions and falsifiers

* **R12-P1 — controlled deck.**  SMT-3 differs from SMT-2 in one namelist
  hunk, `&namtra_ldf`, and only in the six rung-0 values above plus the
  necessary withdrawal of SMT-2's `ln_traldf_OFF=.true.`.  REFUTED if the
  deck diff changes another module or another resolved value.
* **R12-P2 — resolved card.**  NEMO's `ocean.output` echoes the three active
  switches, coefficient mode 20, `rn_Ud=0.018`, `rn_Ld=200.e3`, and the eight
  cited companion values.  REFUTED by any different resolved value.
* **R12-P3 — passive record.**  The plain and instrumented kt=10 restarts are
  byte-identical.  Every self-describing record parses to EOF, contains each
  required named array, and a required-array mutation makes admission exit
  nonzero.  REFUTED by a restart byte or parser/control failure.
* **R12-P4 — run sanity.**  Both the ten-step and shipped-length 100-day NEMO
  runs reach `STOP 0`, emit no non-finite T/S/u/v/ssh, and write the requested
  restart cadence.  REFUTED by a non-finite field, missing frame or nonzero
  exit.
* **R12-P5 — ladder order.**  The SMT-3 card starts from the same geometry and
  initial state as SMT-2, so all kt=1 rows stay at the bar and the first row
  over the bar is not earlier than SMT-2's kt=2 row.  Every moved row will be
  registered.  REFUTED by a kt=1 loss or an earlier first-over-bar.
* **R12-P6 — first boundary.**  Given NEMO's recorded stage entry, advection
  remains at the existing round-219 bar and the first new non-bit tracer
  boundary is the stage-3 lateral-diffusion increment.  The prior is a
  partial-cell slope or MSC statement in the compiled `traldf_iso.f90` /
  `ldfslp.f90`, not FCT.  REFUTED if a pre-LDF boundary is first, or if the
  post-LDF boundary is bit-exact; no internal statement will be named until
  the boundary measurement exists.
* **R12-P7 — landing gates.**  A later physics candidate may land only under
  the standing GYRE ladder/year, both SMT registries, six flat VORTEX cards,
  tanks, DINO month, generic-card, citation and review gates.  This acquisition
  round itself changes no production physics and lands no scientific claim.

## Acquisition boundary

The required SMT-3 NEMO record does not exist.  Sandbox PMIx launches are
forbidden by operator note AM, so this round writes and preflights an
operator-run script with new target names.  The next round admits the record,
builds the explicit card, runs the ladder, and performs the compiled-order
walk.  A failed acquisition prediction remains in its receipt.

## No hidden choices

The six changed values are copied from ORCA2 rung 0.  The eight companions are
the exact reference values rung 0 leaves unset.  Run length, output cadence,
geometry, EOS, advection, vertical mixing, drag, momentum and barotropic
settings are inherited byte-for-byte from the admitted SMT-2 deck; only the
measurement cadence differs between the ten-step and 100-day arms, as in the
earlier seamount rungs.
