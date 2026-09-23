# Preregistration: row-5 post-`dyn_zdf` WZV call 2, round 41

Date: 2026-08-30. Frozen before the first round-41 numerical execution.
Session `01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Question and ordered release

The official round-40 receipt, SHA-256
`723fe0e74724febab71537ef31cde16e32388c9da01c9058ef2c37629067258b`,
certifies row 4 (`dyn_zdf`) at the accumulating bar in both components.  This
round asks only whether legoESM's production vertical velocity at the next
ordered boundary matches NEMO's second `wzv` result.

NEMO's executed order is `dyn_zdf` at `cfgs/DINO/MY_SRC/stpmlf.F90:396`, the
second `wzv` at `:411-412`, and `mlf_baro_corr` later at `:578`.  The active
QCO recurrence is the bottom-up left accumulation in
`cfgs/DINO/MY_SRC/sshwzv.F90:198-228`.  The retained full-halo oracle stream is
`wzv_dump_ww_call2.bin`; `wzv_dump_ww_call1.bin` is only the wrong-order
control.

## Capture boundary and shape contract

The production array is captured while directly executing the committed
single-pass `_nemo_mlf_step` transcription (the same verification rung used
by this ordered source walk), at its only
`diagnose_w_from_flux_div(...)` call in `_step_impl`, after the barotropic
transport correction has assembled `mass_flux_u/v` and immediately before
the tracer update.  The single-pass outer step then performs the implicit
`dyn_zdf` analogue and reaches
`_apply_after_level_reconcile`; a second hook at that boundary requires the
state's stored W to be the exact adjacent-interface average of the captured
array.  The measured array is therefore proven to be the production W field
carried across the post-`dyn_zdf` call-2 boundary, before row-6
reconciliation.  Both hooks are diagnostic only: they return the original
objects unchanged.  The raw source hook can observe setup compositions; the
selected call-2 array is the unique one whose adjacent-interface average is
byte-identical to the W stored at the single post-`dyn_zdf` boundary.  The
boundary and unique-match gates must each fire once and both hooks must be
restored.  The
two DINO cards still select the historical `leapfrog` dispatcher; this
instrument deliberately avoids its two-pass decomposition because that
would produce multiple W arrays and would not be NEMO's one-pass `stpmlf`
call order.

The legoESM array is genuinely interior-only, declared by the production
operators as `(n_lat,n_lon,nlev+1) = (199,52,37)`.  It is not padded.  The
NEMO writer at `sshwzv.F90:276-300` writes the genuine full halo
`(jpk,jpj,jpi) = (36,203,56)` and the loader alone strips `nn_hls=2` to the
same `(199,52,36)` active interface population.  Both shapes are hard gates.

No new Fortran writer or unit is introduced.  The retained canonical writer
uses unit 9103 for its mutually exclusive call-1/call-2 files.  Admission
enumerates the whole NEMO source tree and requires every numeric `OPEN` on
9103 to be those two `sshwzv.F90` branches; this reuses the reserved
deterministic-writer allocation instead of claiming a colliding unit.

## Bars and disposition

The signed call-2 field is scored by the registered `wzv (vertical velocity)`
ACCUMULATING class, maximum/RMS bar `1e-12`, over exactly 9,920 active T
columns at each valid interface selected by the three-dimensional T mask.

* `ROW5_WZV_CALL2_AT_BAR`: the selected call-2 production capture is AT BAR,
  duplicate captures are byte-identical, identity is AT BAR, and every
  control fires.
* `ROW5_WZV_CALL2_DIVERGED`: all admissions and controls pass but call 2 is
  DEBT.  The next peel is the first differing operand of the call-2
  continuity composition; row 6 stays ordered-blocked.
* `INVALID`: any provenance, shape, hook-count, restoration, bracket, or
  control receipt fails.

The call-order control must have power: the production capture's normalized
RMS error against call 1 must exceed its error against call 2.  A sign reversal,
one-cell zonal roll, and a planted point perturbation of
`2e-12 * RMS(call2)` must each classify off bar.  The unchanged-input identity
must classify AT BAR.

Only `ROW5_WZV_CALL2_AT_BAR` releases row 6 (`mlf_baro_corr`).  All later
free-surface-filter, momentum-RHS, and tracer-tail rows remain ordered-blocked
until then.
