# Preregistration: vertical Redi skew operand factorial, round 88

Date: 2026-08-30. Frozen before measurement. The admitted round-87 receipt is
SHA-256 `df56a496d6e7ebc14d9e8214cdc6742010d3b8b4286c0d9b2704526593ed6caf`.

Run an offline 2^3 factorial against the held NEMO temperature A31+A32 stream.
The factors are: `C`, current versus NEMO-source-literal zA31/zA32 assembly
from the existing ahtu/ahtv, wslpi/wslpj, metrics, and masks; `G`, current
left-associated versus NEMO pair-pair four-gradient association; and `T`,
Kbb/before versus Kmm/now tracer gradients. NEMO's source fixes the faithful
arm at C1/G1/T0 (`traldf_iso_scheme.h90:21-48,109-125`). No new NEMO run is
permitted for this measurement.

The faithful arm must pass the unchanged `1e-15` pointwise bar. A factor owns
the skew residual only if its main effect removes at least 90% of the baseline
maximum column error and the faithful arm passes. If no single factor clears
90%, report all pair and three-way interactions normalized by the baseline;
an interaction owns only if it removes at least 90% and its corresponding arm
passes. The Kmm/now arm is a red-capable time-level control: if it alone passes,
stop as an invalid Kbb registration rather than changing production. Include
identity, finite, wet-point, roll, sign, eight-arm, exact input-hash, and
faithful-arm controls.

If C1/G1/T0 closes skew, build only the owned source association on the two
DINO cards and byte-pin all other configurations. If it remains red, the next
held split is A31 versus A32 with their literal coefficients and gradient
sums. A33 and all salinity rows remain ordered-later until skew is at bar.
