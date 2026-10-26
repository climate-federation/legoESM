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

The active NEMO operations are Langmuir source assembly and update, matrix/RHS
assembly at `zdftke.F90:403-420`, and the three source-ordered recurrences at
`:451-469`. The corresponding production selectors are in
`physics/vertical_mixing/tke.py:1183-1219,1272-1475,2682-2737`. The ordered ZDF sweep already proved
the literal implementations match those oracle operations; this climate result
does not retract their transcription fidelity.

## TKE-core sub-peel: matrix owner

The registered five-arm sub-peel completed cleanly. Its scorer artifact is
`/tmp/dino-zdf-wall-epoch-tke-core-peel-01a04e34/zdf_wall_epoch_tke_core_score.json`,
SHA-256
`6486740988f209f8dd8cb57dfbdf2d59d03039b2f61bf39ee71abf8954bdf8fa`;
scorer-log SHA-256 is
`f1eadc9bf9e9c568f3f3cf15761cdab175d5877e25fbbf1f5f9606bd075f8b87`.
The producer is `9ac2550d4f559b1c73c2b65174fc2e436025176a`, the scorer is
`74c442dd8166be4d88de8ad60f7d2eab3d576f7b`, and the admitted control/full-core
artifacts retain their model-diff-zero producer receipt. Every arm has the full
haloed SSH shape `(160,199,52)`, the common corrected-T stress hash, and a
passing runtime/stability gate. The compact committed receipt is
`dino_zdf_wall_epoch_tke_core_subpeel_result.json`.

The registered disposition is
**`LOCALIZED_TO_MATRIX_GIVEN_LEGACY_SOLVER_AND_LANGMUIR`**. The solver main
effect closes `-0.2079%/-1.2148%` of the ratio/wall-share shifts, the Langmuir
main effect `0.00059%/0.00317%`, and their interaction is effectively zero
(`9.97e-10/-8.85e-8` as a fraction). All are `BOUNDED_SMALL`. In contrast,
changing the TKE matrix from literal live `e3t/e3w` thicknesses to the legacy
factored thickness while solver and Langmuir are held legacy closes
`99.6254%` of the ratio shift and `106.1025%` of the wall-share shift, satisfying
the registered majority-owner bars. The earlier group-only qualification is
therefore retired: the e3t/e3w thickness-swap fix is the majority owner of the
`1.16 -> 1.65` wall epoch rise.

This remains faithful-but-worse. NEMO's live-thickness matrix coefficients
`zzd_up/zzd_lw`, diagonal, and RHS are assembled at `zdftke.F90:403-420`;
they control how TKE is spread vertically and dissipated in each column before
the solved TKE sets mixing length and `avm/avt`. Restoring those thicknesses
therefore unmasks a wall-column error that the factored matrix had damped or
redistributed. The climate contrast localizes the coupling to this vertical
TKE pathway; it does **not** identify the unmatched feeder. Plausible feeders
are the matrix/RHS operands already entering at that point—carried `p_avm`,
shear production `p_sh2`, stratification destruction `p_avt*rn2`, and
`dissl*en`—or the downstream TKE-to-`avm/avt` map, not the now-bounded solver
or Langmuir association.

Register item **WALL-TKE-MATRIX-FEEDER** is the discriminating next
measurement. Inventory existing day-180 dumps first, then compare wall and
interior columns in source order: live `e3t/e3w`; `p_avm`; `p_sh2`;
`p_avt*rn2`; `dissl*en`; assembled `zzd_up/zzd_lw/zdiag` and RHS; solved
`en`; mixing lengths and `avm/avt`; vertical-momentum increment; first-eight-
step SSH. Before measurement, freeze pointwise bars and one-at-a-time oracle
substitutions plus the matrix-operand interaction needed for any cancelling
pair. The first operand whose held substitution removes at least 50% of both
epoch shifts is the owner; at most 10% on both bounds it small. Until that
registered ladder runs, the feeder remains `OPEN_UNRESOLVED`.

## Faithful-but-worse interpretation and next discriminator

This is a compensating-error result: the legacy TKE matrix masked a remaining
wall-flicker error, and making the core NEMO-faithful exposed that error as the
rise from the `1.16/0.119` epoch to `1.65/0.170`. It does **not** yet show that
the literal matrix is defective, and it does not identify the other error being
cancelled. Naming its hidden counterpart now would overrun the registered
sub-peel contrast.

That three-arm peel is now complete; the registered next discriminator is
WALL-TKE-MATRIX-FEEDER above. That operand ladder—not the present climate arm—
can name what the legacy matrix was cancelling at the walls.

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
