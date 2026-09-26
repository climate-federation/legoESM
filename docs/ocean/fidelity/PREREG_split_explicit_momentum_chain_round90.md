# Preregistration: vertical Redi skew coefficient operands, round 90

Date: 2026-08-30. Frozen before measurement. The admitted round-89 receipt is
SHA-256 `555520203984430e988c075646e7feb674cc522de33f5cadfc4052ccdc560a76`.

Run an offline 2^4 factorial against the held NEMO temperature A31+A32 stream,
with production's source-literal mask, metric, coefficient, and gradient
association held fixed. Factors independently substitute NEMO's existing
`ahtu`, `ahtv`, `wslpi`, and `wslpj` streams. Also score every raw operand
against production at `1e-15`. All four held streams and round 89 are
SHA-bound; no NEMO run is permitted.

A main effect owns only if it removes at least 90% of the baseline maximum
column error and its single-substitution arm passes. Otherwise report all six
pair interactions and the full four-way removal. A coupled subset owns only
if it passes and removes at least 90%; choose the smallest passing subset in
NEMO source order. The all-oracle arm must be bit-exact, or the measurement is
invalid. Include identity, finite, sixteen-arm, baseline-recomposition,
wet-point, roll, sign, and non-inert substitution controls.

An owned operand is carried at the matched Kmm stage only on the two DINO NEMO
cards, with generic byte pins and a planted legacy-operand violation. If no
subset passes, hold A31 and A32 separately. A33 and salinity remain
ordered-later.
