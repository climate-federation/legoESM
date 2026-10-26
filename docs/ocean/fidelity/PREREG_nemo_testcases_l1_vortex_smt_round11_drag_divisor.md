# PREREGISTRATION — VORTEX_SMT round 11 (lane round 223): land the implicit bottom-drag divisor

Frozen before any measurement of this round.  Base: lane tip `a2bf3a0bc`
(round 222 / VORTEX_SMT round 10, which HELD this statement).  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round11/`.
Before arm: round 10's own after-arm registries at that tip
(`phase3/vortex_smt/round10/after/` and `.../round10/inert/`).

## The statement

NEMO's implicit bottom friction divides by its own bottom-level U/V scale
factor, not by a two-cell average:

```
zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) )
            / (e3u_3d(ji,jj,iku) *(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))
```

`dynzdf.f90:306` (U) and `:473` (V twin, `e3v_3d`/`r3v`/`vmask`/`mbkv`), plus
the `ln_dynspg_ts` barotropic re-add at `:166`/`:168`, in the compiled ppsrc of
the SMT-2 build `VORTEX_SMT2_VEC_R8_OMIP_L1_P3`.

## Predictions, each with its falsifier

* **R11-P1 (form and time level).**  `ln_drgimp = .true.` (`namelist_ref:817`)
  selects the IMPLICIT form: `dynzdf.f90:121` calls `zdf_drg_exp` only under
  `.NOT.ln_drgimp`, and `:306`/`:473` sit inside `IF( ln_drgimp )`.  The
  stretch factor is read at **Kaa**, the AFTER level.  `e3u_3d` is the
  REFERENCE face thickness, loaded from the mesh variable `e3u_0`
  (`domzgr.f90:186`, `:201`), i.e. the min-rule face of the two reference T
  thicknesses over z partial steps.  FALSIFIER: any of those three readings
  is wrong in the compiled source.
* **R11-P2 (the falsifier of record, from round 10 OPEN item 1).**  The SMT-2
  registry's kt=2 `u` row falls from **2.193091e-04** to the order of the
  6.44e-06 barotropic term, i.e. **at least two decades**.  FALSIFIER: it does
  not fall by two decades.
* **R11-P3.**  SMT-2's first-over-bar stays **kt=2** and never moves earlier;
  every moved row registered.  FALSIFIER: an earlier first-over-bar, or an
  AT-BAR row leaving the bar unregistered.
* **R11-P4 (flat cards, inert BY CONSTRUCTION).**  On a flat bottom the
  min rule equals the average (both neighbours carry the same reference
  thickness), so the reference half of the divisor cannot move.  The six flat
  VORTEX cards (`VORTEX-zco`, `VORTEX_VEC-zco`, and the 15 km and 10 km pairs)
  score **0/50 rows moved** each.  FALSIFIER: any row moves.
* **R11-P5 (seamount cards without drag).**  `VORTEX_SMT-zps`,
  `VORTEX_SMT_VEC-zps` and `VORTEX_SMT1_VEC-zps` do not select a bottom-drag
  law, so the edited block does not execute: **0/50** each.  FALSIFIER: any
  row moves.
* **R11-P6 (tanks).**  `LOCK_EXCHANGE-zco` 0/50 (flat, and its deck selects no
  bottom drag).  `OVERFLOW-zps` is a zps tank and MAY move; whichever way it
  goes the direction is registered rather than excused.  FALSIFIER for
  LOCK_EXCHANGE: any row moves.
* **R11-P7 (GYRE, note BZ).**  GYRE is a flat-bottom zco box, so the reference
  half is identical by construction, but the STRETCH half is not: NEMO's `r3u`
  is the `e1e2t`-weighted ssh mean over the two columns divided by `hu_0`
  (`domqco.F90:219-222`) while the previous divisor averaged the two
  already-stretched T thicknesses with a plain 0.5 weight.  Those agree
  algebraically on the U faces of a flat box but NOT on the V faces, where
  `e1e2t` changes with latitude.  PREDICTION: the GYRE ten-step ladder moves
  rows.  **If it moves at all, the statement is HELD, not landed** — that is
  round 10's own preregistered falsifier and note BZ's certified year, and it
  is recorded here before the run.  If it does NOT move, the certified year is
  run (8 days + digests, note BZ) and must stay bit-identical.
* **R11-P8 (DINO).**  DINO is partial-cell, so the statement is LIVE on it and
  the month gate must be MEASURED, not predicted.  Direction registered either
  way.  FALSIFIER of the landing: the DINO month gate goes red.

## Landing rule for this round

LAND only if: R11-P2 holds, R11-P3 holds, every inert card is 0/50, GYRE does
not move (ladder AND year), and the DINO month gate passes.  Any one of those
failing and the statement goes back to HELD with the measurement reported.
