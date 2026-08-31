# Preregistration: row-17 literal TKE solve

Date: 2026-08-28. Status: **design frozen; not implemented**.

Rows 1--15 are verified and inactive row 16 is waived. Row 17 stops the
ordered sweep: production's shared normalized Thomas application diverges in
1,374/9,920 columns (maximum normalized error `0.3044579035485571` at the
registered `1e-12` row-17 bar), while every matrix/RHS/boundary input, both
forward recurrences, and the post-hoc source-ordered NEMO recurrence score
0/9,920. All four southern focus columns pass the bar. Rows 18--32 remain
unmeasured.

The registered option is `tke_solver_evaluation`. `nemo_literal` will be the
faithful default only on `nemo_dino_kamm` and `nemo_dino_kamm_mlf` and must
require their already-literal matrix path. `shared_thomas` remains the
byte-identical default everywhere else and the explicit historical opt-in on
those two cards.

The literal path will transcribe active `zdftke.F90:547-565` as ordered JAX
scans: seed the prescribed surface row with `zdiag(1)=1/en(1)` and
`zd_lw(1)=1`; scan the forward diagonal recurrence through `jpkm1`; scan the
forward RHS recurrence separately with division before the trailing multiply;
seed `en(jpkm1)=zd_lw/zdiag`; reverse-scan the back substitution only through
level 2; then apply `MAX(en,rn_emin)*wmask`. It must not normalize or solve the
held `jpk` row as an ordinary Thomas row.

CONFIRM is 0/9,920 failed columns and 4/4 southern focus passes against the
direct postsolve dump at the unchanged `1e-12` bar. Any failure stops inside
row 17, localized in the recurrence order above. Required red tests are a
hand-computed surface/interior/bottom system, a division/multiply association
discriminator, a `jpkm1` terminal-row discriminator, per-column wet-bottom
masks, eager/JIT equivalence, finite reverse-mode gradients, and exact
`shared_thomas` output identity on every unchanged card.
