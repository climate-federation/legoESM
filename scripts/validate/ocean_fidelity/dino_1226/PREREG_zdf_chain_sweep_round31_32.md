# Preregistration: ZDF sweep rows 31--32 volume-form discriminators

Date: 2026-08-29. CPU-only matched day-180 lane. These substitutions were
registered in the master row table before measurement and require no new NEMO
dumps. Because the complete row-30 composite is red, every number here is
targeting-only until that ordered stop closes.

## Row 31 momentum

Reconstruct the NEMO pre-solve U/V arrays from Nbb velocity plus `rDt*Krhs`,
remove the stage-7 barotropic mode, apply the dumped bottom-drag correction and
surface-stress increment, then feed dumped `avm` and live face metrics to the
production implicit solver. The accumulating column bar is `1.0e-12`; report
U and V independently over their complete wet-column censuses and the four
southern focus columns.

If the production solve fails, evaluate a read-only discriminator preserving
`dynzdf.F90:199-214,340-380` literally: written matrix association, ordered
forward recurrence, and reverse substitution. `CONFIRM` for a solver-lowering
owner means the literal result has zero failing columns in both components.
`REFUTE` means any literal column still fails. A wrong-`rDt` arm and a one-cell
roll must fail.

## Row 32 tracers

Reconstruct the volume-form input
`(e3t(Kbb)*T(Kbb)+rDt*e3t(Kmm)*Krhs)/e3t(Kaa)`, compose dumped `avt` with the
native-slope K33 term, and feed the production paired implicit solve. Score T
and S independently at `1.0e-12`, including every focus column. A wrong-e3t
slot and a one-cell roll must fail. No disposition is promotable while an
earlier row is red.

## Promotion clarification (2026-08-29, before row-30 execution)

Row-30 closure only removes the ordering block. Row 31 next requires the
literal matrix and ordered Thomas path in production and U `0/9758`, V
`0/9868` at `1.0e-12`. Row 32 follows only after that pass and requires the
volume-form production T and S results both at `0/9920`. The unchanged-card
scope and complete gates are frozen in
`docs/ocean/fidelity/PREREG_zdf_chain_sweep_round30_uv_operands.md`.

## Production implementation amendment (2026-08-29, before implementation)

Row 30 is now closed by its held scorer and its independent raw/post-Shapiro
composite.  The production option is frozen as
`zdf_implicit_solver_evaluation`, with values `nemo_literal` and
`shared_thomas`.  `shared_thomas` is the byte-identical historical default on
every card.  Exactly `nemo_dino_kamm` and `nemo_dino_kamm_mlf` select
`nemo_literal` by default; selecting `shared_thomas` on either is the legacy
control.

The faithful entry point preserves the oracle's two distinct written systems:

- momentum builds face coefficients from the two T-point `avm` operands in
  the association written at `dynzdf.F90:182-195,293-296`, then performs the
  increasing-`k` diagonal and RHS recurrences and decreasing-`k` substitution
  at `:322-345` (and the V sibling at `:356-371`);
- tracers build the unnormalised content matrix with `e3t(Kaa)` on the
  diagonal at `trazdf.F90:218-221`, consume the undivided content RHS written
  at `:271-278`, share the matrix/factors between T and S when DDM is off, and
  apply the ordered reverse recurrence at `:281-286`.

The row-31 production bar remains U `0/9758`, V `0/9868` at `1.0e-12`; the
row-32 production bar remains T and S `0/9920` at `1.0e-12`.  Both include all
four focus columns.  Red controls are: `shared_thomas` must reproduce the
registered row-31 miss, wrong `rDt` and a one-cell roll must fail, and for row
32 a pre-normalised/incorrect-`e3t` RHS and a one-cell roll must fail.  Unit
tests must include hand-computed unequal-thickness columns, JIT, finite AD,
selector typo rejection, exact default-versus-explicit legacy identity, and
resolved-card scope assertions.  No new NEMO dump is authorised or needed.

### Row-32 residual peel amendment (2026-08-29, before measurement)

If the first production literal run leaves a tracer residual, peel without new
oracle dumps in this order: undivided content RHS (T then S), `avt+akz`, lower
and upper coefficients, `e3t(Kaa)-(zwi+zws)`, diagonal recurrence, RHS
recurrence, reverse substitution.  The cheap discriminator is the same full
literal system evaluated by an unfused NumPy host loop versus the pure-JAX
ordered scans.  If the host loop is `0/9920` and JAX is red, the remaining
owner is compiled arithmetic lowering inside the first stage where an
optimization barrier closes the census.  Every candidate keeps the `1e-12`
column bar, all focus scores, and wrong-e3t/roll controls; a barrier is retained
only if its production JIT result is `0/9920` for both tracers.
