# Preregistration: ZDF sweep row 30 (`ldf_slp`)

Date: 2026-08-29. CPU-only matched day-180 lane (`DINO_1226_LANE=d180`).

The ordered frontier reached row 30 only after rows 28 and 29 passed.  The
existing direct `eiv_dump_prd_arg.bin` slot is the first consumed operand of
`ldf_slp`; no new oracle run or dump is required.

## Frozen bars and controls

- Score every wet depth column, not a layer average.  The row passes only when
  zero tracer/W columns of 9,920, zero U columns of 9,758, and zero V columns
  of 9,868 exceed `1.0e-15` after normalization by the oracle field RMS.
  Report the four registered southern focus columns separately.
- First score `prd`, then `zgrv/zgru`, then the four final slopes in NEMO
  execution order.  Stop at the first failing operand.
- The planted `prd` one-ULP value perturbation must increase the exact-unequal
  column count by exactly one. The one-cell horizontal roll, bar-scale value
  perturbation, and nonfinite plant must each produce science-bar failures.

The one-ULP arm is an exact-identity red control: a single ULP is smaller than
the normalized `1.0e-15` science bar for this field, so requiring it to cross
that bar would be a false control.  The independent bar-scale perturbation,
horizontal roll, and nonfinite arms must cross the science bar on the identical
consumed-level mask used by the baseline.  This clarification changes no score.

## Registered substitutions

The baseline uses BEFORE (`Nbb`) T/S but CURRENT (`Nnn`) z-star geometry and
forms `prd` by the density round trip `rho/rho0-1`.  Apply substitutions in
this fixed order:

1. replace only the slope geometry by the carried BEFORE `eta_before` z-star
   Jacobian, matching `CALL eos(ts,Nbb,rhd)` and `gdept(...,Nbb)` at
   `stpmlf.F90:219` and `eosbn2.F90:296-305`;
2. holding that geometry fixed, form the simplified-EOS `zn*r1_rho0`
   directly in NEMO's written association, avoiding the non-oracle
   `rho0+zn -> /rho0 -> -1` round trip.

`CONFIRM` for a substitution means the new first-operand census is strictly
smaller.  The production fix is accepted only if the composite reaches
`0/9,920`, every southern focus column passes, both controls fire, JIT and AD
remain live, and all non-oracle DINO recipes retain their legacy config path.

The registered production selections are
`gm_redi_slope_prd_geometry_stage=before_step` and
`gm_redi_slope_prd_evaluation=nemo_literal`.  They are defaults only on
`nemo_dino_kamm` and its inherited `nemo_dino_kamm_mlf` card.  The legacy
`current_step`/`density_roundtrip` path remains the global default and is the
explicit opt-in control on those cards.

## Loud amendment after the first substitution (2026-08-29)

The phrase “slope geometry” above was too broad and is retracted.  Direct
source reread shows two geometry operands: `eos(ts,Nbb,rhd)` uses `gdept(Nbb)`
at `stpmlf.F90:219`, but `rn2b` is intentionally BEFORE T/S on CURRENT
geometry because `eos_rab(...Nbb...,Nnn)` and `bn2(...rn2b,Nnn)` pass `Nnn`
at `stpmlf.F90:205,207`.  The option and substitution therefore move only
the `prd`/EOS depth.  N2, MLD anchoring, and every later ldf_slp geometry
consumer remain on the existing CURRENT Jacobian.  The prematurely combined
implementation was never scored or promoted.

## Loud qualification after the complete row composite (2026-08-29)

The first operand peel closed `prd`, both dumped `zgrv` slots, the complete
dumped j/W recurrence, and `wslpj`, but that is not the whole registered row.
The complete source-order composite also scores the U/V block at
`ldfslp.F90:217-293`. Its final `uslp` and `vslp` receipts remain red, so the
earlier working description “row 30 fixed” is retracted. The U interval is
bounded only by exact `prd` and failing final `uslp` because `zgru` is not yet
dumped. The V sibling is bounded by exact dumped `zgrv` and failing `vslp`;
the missing intermediate operands are not guessed.

Before the next NEMO execution, add deterministic write-only slots in this
order for each direction: `zgru/zgrv(iik)`, `zau/zav`, pre-bound `zbu/zbv`,
post-bound `zbu/zbv`, raw `zwz/zww`, and the post-Shapiro U/V result. The first
slot over the `1.0e-15` column bar owns the divergence. The existing
deterministic-writer rule applies: initialize the entire haloed buffer before
filling the interior, prove all pre-existing streams byte-identical, require
the certified restart SHA, and plant one-ULP plus one-cell-roll controls. No
production change beyond the already-localized `prd`/carried-rn2b/W-chain
fixes is promoted from this incomplete interval.

Rows 31 and 32 may run their already-registered no-new-dump volume-form
discriminators as targeting only. They cannot be promoted across row 30.

The frozen held instrumentation, SHA gates, exact operand order, and human
execution blocks for this interval are in
`docs/ocean/fidelity/PREREG_zdf_chain_sweep_round30_uv_operands.md`.
