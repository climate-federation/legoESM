# ORCA2 round 84 preregistration — hierarchy rung-0 card and first walk

Frozen before reading any rung-0 state payload or running legoESM against the
operator's round-83 record.  Base: `d061f3197c0013b333838af52e9cf9edc1c02b08`.
Every trajectory number in this round is labelled **independent**: rung 0
starts from the climatological T/S and zero sea surface built by its own deck,
not from the shipped ocean-ice card's recorded entry.

## Admitted inputs

Round 83 already preregistered the acquisition contract.  The operator log
reports `PASS_RUNG0_RECORD`: two deterministic two-rank ten-step twins, steps
1..10, and one finite two-rank 240-step from-rest run.  This round re-runs that
admission and all five plants before consuming a payload.  Failure of any
plant or provenance row makes the record inadmissible and stops the round.

## Compiled statements read before prediction

The rung-0 resolved log selects linear bottom drag with implicit friction and
prints `Cd0*Uc0 = 4.0e-4 m/s`.  The compiled initialization forms that constant
as `pCd0=rn_Cd0*mask` followed by `pCdU=-pCd0*rn_Uc0`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:258-284,525-539`).  The implicit
momentum solve inserts the stored bottom coefficient in the deepest-cell
diagonal and the barotropic correction
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:148-171,293-306`).  The split
external mode averages the same T-point coefficient to U/V faces
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1425-1463`).

## Frozen predictions and falsifiers

1. Record admission remains `PASS_RUNG0_RECORD`: 40 restart shards, 100
   array-equal T/S/u/v/ssh twin comparisons, terminal month step 240, and all
   five controls (one ULP, missing rank, wrong step, non-finite payload,
   hidden month-deck delta) refuse.  Any changed count or green plant
   **REFUTES** the record.
2. The explicit rung-0 legoESM card differs from the shipped ORCA2 card only
   in switches assigned to hierarchy rungs 1..10: it selects constant
   vertical mixing (`A_v=1.2e-4`, `K_v=1.2e-5`), disables TKE/IWM/DDM,
   shortwave, runoff, BBL, GM eddy velocity and MLE, retains rung-0 enhanced
   diffusion and pure Redi isoneutral diffusion, uses exact-zero surface
   forcing, and carries no ice entry adjustment.  Any implicit inherited
   higher-rung module **REFUTES** the card.
3. The card's independent entry T/S/u/v/ssh is bit-identical to NEMO's own
   rung-0 `output.init` on both ranks.  In particular ssh is exact zero on wet
   and dry cells; retaining the shipped card's SI3 mass adjustment must make
   the plant fail.  One unequal cell in any field **REFUTES** this prediction.
4. Before a new implementation, the executable-card gate refuses exactly the
   named `linear_implicit_bottom_drag` gap: the shared model accepts implicit
   drag only for nonlinear/log-layer laws, while rung 0 selects the compiled
   linear law above.  A different first refusal **REFUTES** the attribution.
   A source-cited extension may land only if a direct operand gate proves the
   linear T-point and face coefficients bit-exact and a planted coefficient
   violation fires.
5. If the drag statement is discharged, both independent and given-entry
   ten-step walks run against the same rung-0 entry and compare all five fields
   at every step.  The first non-bit row is expected no later than the kt=1
   end state.  A restart-only record that cannot localise that row to one
   compiled statement is insufficient: the round writes a fail-closed staged
   operand acquisition and reports `STOPPED_FOR_RECORD`, never a guessed
   owner.
6. The independent month is scored only after the ten-step walk has a named
   first statement and no earlier execution blocker.  Otherwise every month
   field remains explicitly `UNMEASURED`, even though the NEMO terminal record
   exists.

## Landing predicate

Rung 0 lands only with an admitted record, explicit card, bit-exact entry,
source-owned first non-bit statement, complete given-entry and independent
ten-step scores, independent month scores, firing controls, and the usual
GYRE/DINO/tank/citation/test/review gates for any package change.  No threshold,
carried state, shipped ORCA2 card, sea-ice tuple, higher-rung switch, or
stabilizer changes.

ASKED: admit and score hierarchy rung 0 end to end.

UNASKED: changing the rung definition, relaxing the 2-ULP bar, adding an
uncited drag law, inferring a statement from a terminal restart, or advancing
to rung 1 before rung 0 lands.
