# Split-explicit momentum chain — round 93 result

Date: 2026-08-30. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 93 closes the Redi T/S ladder and releases the already-certified
ZDF/Asselin tail. No NEMO run, GPU, or MPI process was used.

## W-slope and A33 association split

The existing-dump W-slope/Shapiro matrix classifies
`WSLOPE_ASSOCIATION_AT_BAR_R1S0`. Production raw `wslpi/wslpj` differ in
5/9,920 and 2/9,920 columns, with maxima `1.546874e-15` and `1.397875e-15`.
Substituting NEMO's raw W pair before the production Shapiro pass makes both
final fields bit-exact; the source-ordered Shapiro arm has zero main effect
and the interaction is zero. The retained receipt is
`/tmp/dino_split_explicit_momentum_chain_round93_wslp.json`, SHA-256
`3ff570e4359ffd30b4314fd49a8af678d3091191d095a5347e359ee5951ce75a`.

With that exact pair held, the HxK split localizes the remaining temperature
A33 flux to H, NEMO's written coefficient association:

```fortran
pah_wslp2 = zahu_w * wslpi * wslpi + zahv_w * wslpj * wslpj
```

This is `traldf_iso.F90:296-297`. H alone makes `ah_wslp2`, A33, and full
temperature `zfw` bit-exact. K—the `akz_h`, vertical reduction, and reciprocal
post-factor association at `traldf_iso.F90:314-332`—is inert; HxK interaction
is zero. The explicit consumer is `traldf_iso_scheme.h90:126-129`.

The production fix adds `redi_a33_evaluation`. Generic cards retain the
byte-pinned exponent topology; `nemo_dino_kamm` and
`nemo_dino_kamm_mlf` select `nemo_literal`. The same `nemo_iso_a33` builder is
used by the explicit Redi flux and implicit K33 getter, preventing another
split-coefficient drift. A planted fp64 operand proves the two associations
are distinguishable, and focused A33 plus DINO-card tests pass.

## Ordered Redi disposition

The final machine receipt is
`/tmp/dino_split_explicit_momentum_chain_round93_final.json`, SHA-256
`1bac2857f8d0e78f5a51ae1834654773f539bbcac5a3093fe016d5bb5f2b486e`.
It classifies `TRACER_TAIL_REDI_CLEARED_ROUND93_RULE1B` with no first
diverged subrow.

| subrow | operand | disposition | columns red | maximum normalized error |
|---|---|---:|---:|---:|
| 78.T.1 | temperature zfu | AT-BAR | 0/9,758 | 0 |
| 78.T.2 | temperature zfv | AT-BAR | 0/9,868 | 0 |
| 78.T.3 | temperature zfw | AT-BAR, exact counterfactual | 0/9,920 | 0 |
| 78.S.1 | salinity zfu | AT-BAR | 0/9,758 | 0 |
| 78.S.2 | salinity zfv | AT-BAR | 0/9,868 | 0 |
| 78.S.3 | salinity zfw | CLEARED-RULE-1B | 212/9,920 | `6.690652e-15` |

Rule 1b is explicit for 78.S.3: this is **proven-oracle-arithmetic**, not a
relaxed bar and not an AT-BAR claim. Correlation is 1.0, RMS ratio differs
from unity by one fp64 ULP, all registered W/Shapiro and H/K association
options are exhausted, the exact-oracle W plus literal-A33 temperature arm is
bit-exact, and all identity/roll/sign/coefficient plants fire. The frozen
clearance ceiling is `2e-14`; exceeding it would reopen the row.

## ZDF and Asselin tail

The released tracer-ZDF rows reuse the committed day-180 closure artifact
`dino_zdf_chain_end_verified_artifact.json`, SHA-256
`ce6fff6690ff6fbfa1023b4cf3eafbd36d0b86938ef47c5f7524accc410d4975`.
Rows 30–32 are `VERIFIED`; row 32's independent NumPy literal discriminator
is bit-exact for T and S (0/9,920), and the production literal path is inside
the unchanged accumulating bar for both. Wrong-thickness, roll, and legacy
shared-Thomas controls are red.

The final `tra_atf_qco` registry row retains the campaign's existing
`VERIFIED-FORMULA / STRUCTURAL-TARGET-QUALIFIED` disposition. The active
thickness-weighted Asselin formula and `rn_atfp` are verified in the committed
ATF probes, while the old tracer target bracket reconstructs Naa from the
filtered target and is therefore not promoted to an independent strict-array
certificate. This qualification is not an unowned arithmetic operand and no
new physical-scale residual is hidden by it.

Accordingly the ordered SPLIT-EXPLICIT / MOMENTUM-COMMIT walk has no remaining
red or unmeasured operand: strict rows are at bar, 78.S.3 is cleared under the
requested Rule-1b oracle-arithmetic standard, and the Asselin endpoint retains
its named structural/target-bracket qualification.
