# ZDF wall-epoch selector-lattice result

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Bound receipts

The completed scorer artifact is
`/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34/zdf_wall_epoch_lattice_score.json`,
SHA-256
`05dc4c7edefc8b6e705026812b3a4d6ae90098715b60a940094a3de421c45a16`.
It was produced by committed scorer `96b3858f29334f983f33a3a22961387c416ba667`
from clean model producer
`9ac2550d4f559b1c73c2b65174fc2e436025176a`; scorer-log SHA-256 is
`43183e20c7e793ff48e496145970d94359f8f1136b276357ac7f6e282e7c8b38`.
The compact committed receipt is
`dino_zdf_wall_epoch_group_bisect_result.json`.

All six admitted artifacts are clean, stable, float64, session-matched, and
carry 160 full haloed SSH samples of shape `(160,199,52)`. All share T-point
restart stress hash
`cad9b34958cba58812f1dce2c6c441c8fd6b5f2733c20b91065f4d60507592b7`.
The scorer enumerated 14,336 legal selector subsets, recorded all three
structurally unreachable contrasts, and fired every direct and conditional
classifier plant, including `EPOCH_RESTORED`.

## Registered verdict

**`LOCALIZED_TO_TKE_CORE_AT_FAITHFUL_SLOPE_N2`.** The current corrected-T
control is ratio `1.6511846170859847`, wall share `0.17020235431211447`;
the historical epoch is `1.1607251697830108`, `0.11941867158491266`.
Reverting only the legal TKE-core group—literal matrix association, literal
Thomas recurrence, and literal Langmuir association—to the legacy factored /
shared / vectorized implementations gives `1.163579045567568` and
`0.11693494406953533`, inside both historical bands. That closes
`0.9941812196701466` of the ratio shift and `1.0489079834701893` of the
wall-share shift.

The attribution is not a slope interaction in disguise. `slope_n2_only` is
`BOUNDED_SMALL`; the core x slope-N2 interaction is only `-0.0010501686000104872`
and `-0.015519449028966605` of the two epoch shifts. MXL/final-ZDF, the four
remaining slope selectors, and entry given its forced legacy dependencies are
also `BOUNDED_SMALL`. `core_slope_n2` and `entry_closed` restore the epoch only
because both contain the owning core; their reachable incremental contrasts do
not create a second owner.

The active NEMO operations are Langmuir source assembly and update at
`zdftke.F90:401-468`, matrix/RHS assembly at `:499-517`, and the three source-
ordered recurrences at `:547-565`. The corresponding production selectors are
in `tke.py:1183-1198,1429-1475,2956-2986`. The ordered ZDF sweep already proved
the literal implementations match those oracle operations; this climate result
does not retract their transcription fidelity.

## Faithful-but-worse interpretation and next discriminator

This is a compensating-error result: the legacy TKE core masked a remaining
wall-flicker error, and making the core NEMO-faithful exposed that error as the
rise from the `1.16/0.119` epoch to `1.65/0.170`. It does **not** yet show that
the factored matrix, rather than the shared recurrence or vectorized Langmuir
association, supplied the compensation, and it does not identify the other
error being cancelled. Calling the literal matrix defective, or naming its
hidden counterpart now, would overrun the registered group contrast.

The discriminating next measurement is a legal three-arm peel at faithful
slope N2: legacy solver only, legacy Langmuir only, and legacy solver plus
Langmuir with the literal matrix retained. Together with the existing faithful
control and full-core-legacy arm, these form the solver x Langmuir four-corner
at faithful matrix and the only reachable conditional matrix contrast (literal
versus factored with both consumers held legacy). Reuse the same historical
bands and `>=0.50` owner / `<=0.10` bounded-small bars. The winning contrast
then gets a matched-state wall/interior ladder through Langmuir-updated RHS,
matrix diagonals, forward/backward recurrence, solved TKE, `avm/avt`, vertical
momentum increment, and first-eight-step SSH. That ladder—not the present group
arm—can name what the legacy core was cancelling at the walls.

## #1696 boundary

This result changes the open-evidence list but not #1696's COMPLETE code
boundary. #1696 remains the V-face metric plus literal barotropic-continuity
unit with its exact day-180 acceptance and its real day-360 basin
`UNRESOLVED/FLOOR` result. It must still not claim that its metric/continuity
pair owns the wall climate response. Its former `INVALID_CONTROL` wall
qualification can now cross-reference this separately registered follow-up:
the epoch drift is localized to the already-landed ZDF TKE-core selector group,
not to #1696's intervention. The TKE-core sub-peel and the ordered momentum rows
1.2--6 remain follow-up work and must not be appended as a tail to #1696.

## Campaign-ledger paragraph (draft)

Three registered climate tests landed on 2026-08-29: the complete faithful ZDF
bundle was **REFUTED** as an MLD improvement because southern-basin day-90 MLD
RMS worsened from `18.6864 m` legacy to `22.4795 m` faithful while both arms
passed the 5x acceptance gate; the NEMO V-face metric plus literal continuity
moved the day-360 basin gap by only `+0.01310 Sv` (`1.376%`, `-0.95198` to
`-0.93888 Sv`) and therefore remained formally **UNRESOLVED/FLOOR**, ruling out
material ownership at the registered resolution; and the legal wall-epoch
lattice localized the `1.16 -> 1.65` flicker rise to the TKE matrix/recurrence/
Langmuir core at faithful slope N2 (`99.42%` ratio closure, `104.89%` wall-share
closure), a **faithful-but-worse** result showing that the legacy core had been
cancelling a still-unidentified wall error rather than that the literal NEMO
transcription is wrong.
