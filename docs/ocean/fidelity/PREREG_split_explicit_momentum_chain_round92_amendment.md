# Round-92 amendment: raw W-slope source association

Date: 2026-08-30. Frozen after the first production carry run and before this
arithmetic change. Provisional production receipt SHA-256:
`7e02f72f7c7c686a1fb7da16f52a7dd3b047ee846d3734ae37c1774065441d5f`.

The carried pair reduces skew to `3.319416367691509e-15`; A33 is
`4.8788494395565624e-15`. The committed row-30 existing-dump walk shows every
j-direction operand through `zfk` bit-exact; the first nonzero error is the raw
W expression, while final smoothing reduces it. NEMO `ldfslp.F90:320-328`
forms both integer-selector arms, adds them, and applies `wmask`. Production
currently uses `where`, which elides the zero arm and changes last-bit
association.

For the already selected literal slope path only, transcribe the written
multiply/add/mask expression for both W directions. Generic cards retain the
existing `where` expression byte-for-byte. Confirmation requires the raw and
final W fields not to regress, the production skew to pass `1e-15`, and a
literal-to-legacy reversion to remain red. If A33 survives, its own written
square/coefficient association is the next ordered operand.
