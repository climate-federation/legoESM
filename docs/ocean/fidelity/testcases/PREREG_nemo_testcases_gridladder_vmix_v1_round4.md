# V1 round 4 preregistration — bottom-up mixing-length thickness

Date: 2026-10-16.  Frozen before inspecting the V1 restart arrays or running
the round-4 replay.  Decision 129 remains pending; this uses the admitted
as-built V1 deck (NEMO's own TKE floors, internal-wave mixing off).

## Fixed source reading

The compiled V1 `tke_avn` subroutine opens at
`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/zdftke.f90:531`.  It initializes
all `jpk` rows of both length work arrays to `rmxl_min` at :594-595, writes the
raw length through `jpkm1` at :643-645, scans the upward length at :686-688,
scans the downward length from `jk=jpkm1` using row `jk+1` at :690-692, and
forms `zmxlm`/`zmxld` at :694-698.  The observable coefficient and
dissipation statements are :706-711.  The compiled `ORCA2_OMIP_L4` file is
byte-identical, so these are also the Decision-122 production citations.

The card supplies only `jpkm1=30` raw `e3t` rows.  Therefore its literal
downward scan leaves the deepest carried W row at the `rmxl_min` seed instead
of executing NEMO's `jk=30` update with `e3t_3d(:,:,31)`.  Rows below a
partial column's bottom but above 31 are already present in the card's raw
mesh and remain part of this measurement; they are not assumed wrong.

## Frozen predictions and falsifiers

P1 — owner.  At kt 1, cellwise reconstructed NEMO `zmxlm` and `zmxld` will
differ from the card first at W level 30 in full-depth columns.  Supplying the
true raw `e3t(:,:,31)` row and executing the cited `jk=30` update will remove
every over-bar (`>1e-15*max(|NEMO|,1)`) `zmxlm`/`zmxld` difference attributable
to the length sweep.  REFUTE if the first difference is at another statement,
or any over-bar length cell remains after this one substitution.

P2 — downstream closure.  With no other input changed, the same substitution
will reduce kt-1 over-bar `avm` and `avt` from round 3's 148/147 cells to zero.
REFUTE if either count remains nonzero.  Last-place unequal cells may remain
AT-BAR and will be reported, not called bit-identical.

P3 — scope.  The true `e3t(:,:,31)` row will change only the V1 card-selected
TKE path.  G1-G5 and the full ORCA2 card remain unchanged unless the latter is
separately selected under Decision 122.  REFUTE if any existing card's resolved
configuration or default behavior moves.

## Measurement and non-vacuity

The round-4 instrument will report, cell by cell at kt 1, the card and NEMO
`zmxlm`, `zmxld`, the raw thickness read at every wet bottom and immediately
below it, and before/after `en`, `avm`, `avt`.  NEMO lengths are reconstructed
only where the compiled observables make them defined: `zmxld=sqrt(en)/dissl`
(:711), and `zmxlm=avm/(rn_ediff*sqrt(en))` only where the background MAX at
:709 is inactive.  The source-ordered replay is the primary comparison where
that inverse is censored.

Controls: refuse wrong shapes/non-finite fields; prove the NEMO restart and
kt-1 frame agree on `en`/`avm`; require at least one full-depth column where
the planted old 30-row association changes the replay; plant one ULP into the
extra row and require the cellwise comparison to fail.  Both V1 labels are
then re-scored at kt 1..10 with the unchanged round-3 protocol.
