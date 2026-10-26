# Preregistration — round 196 / VORTEX round 12

Frozen BEFORE any measurement of this round.  Lane tip `d3fa8b22a`
(round 195's HELD landing).  Case `VORTEX_VEC-zco`.  Evidence root
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round196/`.
Decision 82 (user, 2026-10-01 20:00): round 194's two-solve candidate
stays HELD, the barotropic solve is walked first, the two-ULP ratchet
is unchanged.

## What round 195 left

The first non-bit producer of the kt=2 U/V residual that survives the
held candidate (`1.2462615e-08` / `1.0562746e-08`) is the external-mode
(barotropic) solve.  Substituting NEMO's recorded barotropic quintuple
takes the stage-2/3 output velocity to `1.11e-16` / `2.22e-16`; with
legoESM's own solve it is `1.2459e-08` / `1.0560e-08`.  No record
carries the barotropic solve's SUBSTEP operands, so the statement
inside `dyn_spg_ts` could not be named.

## What this round builds

A self-describing per-substep record of `dyn_spg_ts` on the
`VORTEX_VEC_R8`-equivalent vector card, acquired on a NEW paired build
(`VORTEX_VEC_R12_OMIP_L1` reference, `VORTEX_VEC_R12_OMIP_L1_P3`
instrumented), same deck, additive write-only instrument.  One file per
`kt`; per substep `jn` the operands the loop reads and writes, at six
compiled boundaries: the mid-step extrapolation, the mid-step
transport, the after-SSH, the half-step-back interpolation and the
surface-pressure gradient, the Coriolis and bottom-stress trends, the
velocity update plus the running filter sums; plus a loop-entry frame
(the forcing terms, the three velocity/ssh time levels, the face
depths, the drag coefficients and the two time-filter weight vectors)
and a loop-exit frame (the filtered quintuple `ssh`, `uu_b`, `vv_b`,
`un_adv`, `vn_adv` after the division by the weight sums).

## Predictions (falsifiers named)

* **P1 — the record admits.** Every `kt=1..10` record parses by its own
  header (magic + sixteen header integers + `(name, rank, n1, n2, n3,
  payload)` groups to EOF), carries `icycle` substep frames per `kt`,
  and the checker's four plants (field-name, truncated payload, missing
  frame, header mismatch) each turn it red while the unplanted run is
  green.  FALSIFIER: any plant that does not fire, or any required
  group absent.
* **P2 — the instrument is additive.** The instrumented build's step-10
  restart is byte-identical to the uninstrumented reference build's AND
  to round 192's `VORTEX_VEC_R8_OMIP_L1_P3` step-10 restart on the same
  card.  FALSIFIER: any byte difference; that would mean the writer
  changed the answer and the record is inadmissible.
* **P3 — the loop-entry operands are NOT the owner.** Given NEMO's
  recorded loop-entry frame, legoESM's first-substep predictor
  velocities reproduce NEMO's to the compiled-rounding floor
  (normalized max abs `< 1e-15`).  FALSIFIER: a first-substep
  extrapolation residual above `1e-12`, which would move the owner
  upstream of `dyn_spg_ts` into the slow-forcing assembly.
* **P4 — the owner is one statement, re-made every substep.** The first
  non-bit boundary at substep 1 is the same boundary at substep 2 and
  at substep `icycle`.  FALSIFIER: a different first non-bit boundary
  at substep 2 than at substep 1 (that would make it an accumulation,
  not a fixed transcription difference).
* **P5 — nothing lands unless a single cited statement closes every
  gate.** The default verdict is HELD.  FALSIFIER: none; this is the
  landing rule, recorded so a landing has to be argued against it.

## Gates that bind if anything lands

Decisions 43/45/55/59; the certified 50-row trajectory gate with the
two-ULP ratchet UNCHANGED (Decision 71/82); GYRE byte-identical at day
30 `2.3432510206121264e-06`, day 240 `6.581707093530567e-05`, day 360
`5.407735418221895e-05` K, with the certified kt=1..10 ladder re-run at
minimum; the generic NEMO-GYRE recipe gate; the DINO month gate with a
private work dir; the citation gate with its plant firing.  The held
two-solve candidate stays HELD (Decision 82); any re-test of it is a
measurement arm reported in the receipt, never a landing.

## Option choices

Every option this round selects is the shipped VORTEX deck's, unchanged
from rounds 3-11 (S-EOS per Decision 69, `ln_zad_Aimp` unset per
Decision 70, `ln_dynadv_vec`/`ln_dynvor_een` true per Decision 73,
`nn_e = 48`, `nn_bt_flt = 3`, `ln_bt_fw = .true.`).  The record's own
parameters — which boundaries are dumped and under what names — are
instrument choices, not model choices, and are listed above before the
measurement.  Anything the deck does not pin that this round would have
to choose is reported as DECISION_NEEDED.
